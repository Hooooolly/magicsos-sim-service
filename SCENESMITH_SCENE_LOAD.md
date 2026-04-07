# SceneSmith Scene Load Pipeline

How SceneSmith-generated scenes export to USD and load into Isaac Sim with WebRTC streaming.

## Quick Start

```bash
# 1. Export scene (on a running sim pod)
curl -X POST http://<pod-ip>:6800/scene/export_scene \
  -H 'Content-Type: application/json' \
  -d '{"scene_dir": "/data/embodied/scene/runs/2026-03-15/05-58-51/scene_000", "name": "my_scene"}'

# 2. Load into Isaac Sim (same pod, scene already loaded in streaming)
curl -X POST http://<pod-ip>:6800/scene/load \
  -H 'Content-Type: application/json' \
  -d '{"usd_path": "/data/embodied/scene/library/my_scene_123456.usda"}'
```

Or from the frontend:
1. **Scene** page → Library → select scene → **Export USD**
2. **Sim** page → select scene from dropdown → **Load**

## Pipeline Overview

```
SceneSmith output (GLTF meshes + house_state.json)
  │
  ▼  POST /scene/export_scene
  ├─ Parse house_state.json (rooms, walls, floors, objects, lights)
  ├─ Compute room offsets (placed_rooms position + width/2, depth/2 = center)
  ├─ Convert each GLTF → USD via omni.kit.asset_converter
  │   ├─ Copy gltf/pbr.mdl into each asset subdir (required for RTX)
  │   └─ Asset stays Y-up (rotation applied in scene assembly)
  ├─ Assemble scene USDA:
  │   ├─ /World/Lights/ — DomeLight + DistantLight + SphereLights per room
  │   ├─ /World/Structure/ — wall Cubes with door/window openings split
  │   │   ├─ Solid walls → single Cube
  │   │   ├─ Walls with doors → segments around door gap
  │   │   ├─ Walls with windows → segments + sill + lintel + glass pane
  │   │   ├─ Fully open walls → skipped entirely
  │   │   └─ Floor Cubes per room (slightly above Z=0)
  │   └─ /World/Objects/{name} — each object with:
  │       ├─ translate (SceneSmith position + room offset)
  │       ├─ orient (SceneSmith quaternion)
  │       ├─ scale (0.01 — asset_converter outputs centimeters)
  │       └─ /asset with +90° X rotation (GLTF Y-up → Z-up) + USD reference
  └─ Save to /data/embodied/scene/library/{name}_{timestamp}.usda

  │
  ▼  POST /scene/load
  ├─ Clear previous scene from /World (keep defaultGroundPlane)
  ├─ Open scene USD as read-only stage
  ├─ AddReference /World/Lights → /World/SceneLights
  ├─ AddReference /World/Structure → /World/SceneWalls
  ├─ AddReference each /World/Objects/{name} → /World/SceneObjects/{name}
  │   ├─ Add 0.01 scale if missing (backward compat for older exports)
  │   └─ simulation_app.update() every 3 objects (keep streaming alive)
  └─ Scene renders in WebRTC ✓
```

## Export Details

### Room Layout

SceneSmith `house_state.json` contains:
- `layout.placed_rooms[]` — room positions (bottom-left corner), width, depth
- `layout.room_geometries{}` — walls/floor per room (coords relative to room center)
- `rooms{}` — objects per room with transforms

Room center = `placed_rooms.position + (width/2, depth/2)`. All wall and object coordinates get this offset added.

### Wall Openings

Each wall in `placed_rooms` has an `openings[]` array:

| Type | Behavior |
|------|----------|
| `open` | Full-height gap (e.g., kitchen→studio open floor plan). Wall skipped if opening covers full wall length. |
| `door` | Full-height gap at specified position and width |
| `window` | Gap with sill (lower wall kept) and lintel (upper wall kept) |

A wall with openings gets split into multiple Cube segments:
- Segments left/right of each opening
- Lintel above doors/open (if opening doesn't reach ceiling)
- Sill below windows
- Glass pane in window openings (thin transparent Cube with UsdPreviewSurface glass material)

### Window Glass

Each window opening gets a thin transparent Cube (1cm thick) with a UsdPreviewSurface material:
- `diffuseColor`: (0.85, 0.92, 0.97) — light blue tint
- `opacity`: 0.25 — semi-transparent
- `ior`: 1.5 — glass refraction
- `roughness`: 0.05 — near-mirror smooth
- `metallic`: 0.0

### Asset Conversion

`omni.kit.asset_converter` converts GLTF → USD with these settings:
```python
ctx.ignore_materials = False
ctx.ignore_cameras = True
ctx.ignore_animations = True
ctx.ignore_light = True
ctx.export_preview_surface = True
ctx.use_meter_as_world_unit = True
ctx.convert_stage_up_z = True  # doesn't work for GLTF, rotation applied instead
```

**Critical**: Each asset USD references `@gltf/pbr.mdl@` (MDL shader). The export copies `/isaac-sim/kit/mdl/core/mdl/gltf/pbr.mdl` into each asset's `gltf/` subdirectory. Without this file, RTX renderer silently stops producing frames.

### Coordinate System

- SceneSmith: Z-up, meters
- GLTF: Y-up
- asset_converter output: Y-up (doesn't convert axis for GLTF)
- Fix: +90° X rotation on the `asset` prim in the scene USD
- Scale: 0.01 (asset_converter outputs centimeters despite `use_meter_as_world_unit`)

### Output Structure

```
/data/embodied/scene/library/
  my_scene_123456.usda              # Scene assembly
  my_scene_123456_assets/           # Converted assets
    studio_bed_0/
      studio_bed_0.usd              # Mesh + materials
      Image_0.png                   # Texture
      gltf/pbr.mdl                  # MDL shader (copied from Isaac Sim)
    coffee_mug_0/
      coffee_mug_0.usd
      Image_0.png
      gltf/pbr.mdl
    ...
```

## Load Details

### Why Not open_stage

`open_stage()` replaces the entire USD stage, destroying the NVCF WebRTC streaming render targets. The stream goes permanently black and cannot reconnect. This is an Isaac Sim architectural limitation.

### Incremental AddReference

Instead, `scene_load` keeps the existing stage and adds scene content as USD references:

1. Clear `/World` children (except `defaultGroundPlane`)
2. Add `/World/Lights` as reference under `/World/SceneLights`
3. Add `/World/Structure` as reference under `/World/SceneWalls`
4. Add each `/World/Objects/{name}` individually under `/World/SceneObjects/{name}`
5. Call `simulation_app.update()` every 3 objects

Step 5 is critical: adding all objects without yielding blocks the main thread, causing NVCF streaming heartbeat timeout → "Stream stopped" → unrecoverable.

### Scale Backward Compatibility

Older exported USDs (before 2026-04-04) lack `xformOp:scale` on objects. `scene_load` detects this and adds `(0.01, 0.01, 0.01)` automatically. Without this, objects are 100x too large and the camera ends up inside them (appears black).

## Problems Solved (Historical)

### 1. open_stage kills WebRTC (2026-04-03)
**Root cause**: NVCF streaming render targets invalidated when stage replaced.
**Fix**: Use AddReference instead of open_stage.

### 2. Bulk loading crashes streaming (2026-04-03)
**Root cause**: Main thread blocked too long → NVCF heartbeat timeout.
**Fix**: Incremental loading with simulation_app.update() every 3 objects.

### 3. Objects 100x too large / black screen (2026-04-04)
**Root cause**: asset_converter outputs centimeters, missing 0.01 scale.
**Fix**: Export adds scale; scene_load adds it for older exports.

### 4. Missing gltf/pbr.mdl freezes RTX (2026-04-04)
**Root cause**: Asset USDs reference MDL shader file that doesn't exist next to them.
**Fix**: Export copies pbr.mdl into each asset's gltf/ subdirectory.

### 5. Furniture rotated/flipped (2026-04-04)
**Root cause**: GLTF Y-up meshes loaded into Z-up scene without axis conversion.
**Fix**: +90° X rotation on asset prim in scene USD.

### 6. Rooms disconnected (2026-04-04)
**Root cause**: placed_rooms.position is bottom-left corner, room_geometries are relative to center. Offset was using position directly instead of position + size/2.
**Fix**: Center = position + (width/2, depth/2).

### 7. Walls solid / no doors or windows (2026-04-05)
**Root cause**: Walls exported as single Cubes without considering openings.
**Fix**: Split walls into segments around door/window/open openings. Open-type openings on partial walls (e.g., studio north wall with both open and door) create full-height gaps.

### 8. No window glass (2026-04-06)
**Root cause**: Window openings were empty holes with no glass.
**Fix**: Add thin transparent Cube with UsdPreviewSurface glass material (opacity=0.25, ior=1.5) in each window opening.

## Isaac Sim Version Compatibility

### 4.5.0 (production)
- Works with incremental AddReference loading
- `omni.isaac.core` imports
- NVCF streaming via `LIVESTREAM=2` env var
- Known RTX bug with many MDL materials (OMPE-73245) — mitigated by pbr.mdl placement

### 5.1.0 (tested)
- SimulationApp needs `enable_livestream=True`, `livestream_library="omni.services.livestream.nvcf"`
- Explicitly enable `omni.kit.livestream.core`, `omni.kit.livestream.webrtc`, `omni.services.livestream.nvcf`
- Curobo JIT fails (cuda-11.8 + gcc > 11) — only affects robot motion planning
- Backend env var `INTERACTIVE_SIM_IMAGE` controls Docker image

## Code Locations

| What | Where |
|------|-------|
| Export | `run_interactive.py` → `_handle_export_scene()` |
| Load | `run_interactive.py` → `scene_load` command in main loop |
| Asset converter | `_convert_gltf()` in `_handle_export_scene()` |
| Wall splitting | Inside `_handle_export_scene()`, wall openings loop |
| Glass panes | Inside `_handle_export_scene()`, glass_panes list + assembly |
| pbr.mdl copy | Inside `_handle_export_scene()`, after `_convert_gltf()` |
| Scale fix | `scene_load` → checks missing `xformOp:scale` |
| Backend export API | `infra-backend/apis/sim.py` → `export_saved_scene_usd()` |
| Frontend Export button | `SceneLibraryTab.jsx` → `handleExportUsd()` |
| Frontend Load | `simulation/page.jsx` → `handleLoadScene()` |

## Luban Agent Integration

Luban can perceive and edit SceneSmith scenes loaded in Isaac Sim.

### Scene Perception

Luban's `scene_observe` tool traverses the USD stage and reports all objects with positions and spatial relations. The traverse code in `perception.py` handles both:
- **Simple scenes**: `/World/ObjectName` (depth 2)
- **SceneSmith scenes**: `/World/SceneObjects/coffee_mug_0` (depth 3)

Objects under `/World/SceneObjects`, `/World/SceneLights`, and `/World/SceneWalls` are included at depth 3. Deeper prims (mesh internals, materials) are skipped.

### Object Type Inference

`_infer_object_type()` in `perception.py` maps prim names to semantic types:
- Standard: table, mug, ball, franka, openarm, cube, sphere
- SceneSmith furniture: bed, sofa, chair, bookshelf, desk, nightstand, refrigerator, toilet, bathtub, vanity, cabinet, counter
- Structure: floor, wall, light

### Editing Operations

Luban can modify SceneSmith scenes via tool calls (no SSH, uses bridge `code/execute` API):

| Operation | Luban Tool | Example |
|-----------|-----------|---------|
| List objects | `scene_observe` | "场景里有什么物体" |
| Delete object | `scene_remove_object` | "删掉书架" → removes `/World/SceneObjects/studio_bookshelf_0` |
| Add object | `scene_add_object` | "在桌子上加一个苹果" |
| Move object | `scene_move_object` | "把椅子移到桌子旁边" |

### Router Configuration

`prompts.yaml` routes delete/add/move/observe requests to "tool" tier (Qwen tool-call), not "claude" tier (SSH fallback):
```yaml
- "删掉X", "remove X", "delete X" → tool (scene_remove_object)
- "场景里有什么", "what objects" → tool (scene_observe)
```

### Files Changed for Luban Integration

| File | Change |
|------|--------|
| `luban/scene/perception.py` | Traverse SceneObjects/SceneWalls/SceneLights at depth 3; added furniture type inference |
| `luban/prompts.yaml` | Router explicitly routes delete/add/observe to "tool" tier |

## Key Directories

| Path | Contents |
|------|----------|
| `/data/embodied/scene/library/` | Exported USD scenes + asset dirs |
| `/data/embodied/scene/runs/` | SceneSmith generation outputs |
| `/isaac-sim/kit/mdl/core/mdl/gltf/pbr.mdl` | MDL shader (source for copy) |
| `/data/embodied/agent/luban/` | Luban agent code (hostPath on 108) |

## Camera Controls

The Sim page has camera control buttons: 📷 ← → ↑ ↓ ⇧ ⇩ + −

### How It Works

```
Frontend button click
  → POST /v1/sim/proxy/bridge/camera/move  (generic bridge proxy)
  → Bridge /camera/move endpoint            (Flask, run_interactive.py)
  → _enqueue_cmd("camera_move")             (thread-safe queue to main loop)
  → Main loop: ViewportCameraState API      (the only API that overrides viewport manipulator)
```

### Button Mapping

All movement is **view-relative** (screen-space), not world-axis:

| Button | Direction | Behavior |
|--------|-----------|----------|
| 📷 | overview | `frame_viewport_prims()` — frames all SceneObjects + SceneWalls |
| ← | left | Move along camera's local -right vector |
| → | right | Move along camera's local +right vector |
| ↑ | forward | Move along camera's forward direction (XY plane projection) |
| ↓ | back | Move along camera's backward direction (XY plane projection) |
| ⇧ | up | Move along camera's local up vector |
| ⇩ | down | Move along camera's local down vector |
| + | zoomIn | Move along camera's forward vector (toward target) |
| − | zoomOut | Move along camera's backward vector (away from target) |

Step size is 2.0 meters per click (configurable via `step` parameter).

### Key Technical Details

**Why ViewportCameraState**: Isaac Sim's viewport manipulator locks the camera prim (`/OmniverseKit_Persp`). Direct `xformOp:translate` writes via `op.Set()` get silently overridden every frame. `ViewportCameraState.set_position_world()` updates the manipulator's internal state, so the change persists.

**View-relative math**: Camera axes are computed from position and target:
- `forward = normalize(target - position)`
- `right = normalize(cross(forward, world_up))`  (world_up = Z)
- `up = cross(right, forward)`
- Position and target move together to keep viewing angle stable

**Proxy routing**: Frontend uses `/v1/sim/proxy/bridge/camera/move` (generic bridge proxy), not a dedicated backend endpoint. The `host` and `port` query params route to the correct sim pod.

### Code Locations

| What | Where |
|------|-------|
| Frontend buttons | `simulation/page.jsx` → `moveCamera()` callback |
| Bridge endpoint | `run_interactive.py` → `@bridge.route("/camera/move")` |
| Main loop handler | `run_interactive.py` → `camera_move` command block |
| ViewportCameraState | `omni.kit.viewport.utility.camera_state` (Isaac Sim built-in) |

### Bugs Fixed (2026-04-07)

1. **404**: Frontend called `/v1/sim/bridge/camera/move` (nonexistent). Fix: use generic proxy `/v1/sim/proxy/bridge/`
2. **500 NameError**: Bridge endpoint used `request` instead of `flask_request`. Fix: correct import name
3. **Camera not moving**: `op.Set()` on camera prim overridden by viewport manipulator. Fix: use `ViewportCameraState.set_position_world()`
4. **Wrong directions**: World-axis offsets don't match screen directions. Fix: compute view-relative axes from camera position/target
