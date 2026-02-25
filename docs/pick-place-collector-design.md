# Pick-and-Place Data Collector — Design Document

## 1. Overview

Build a standalone pick-and-place data collector that runs inside Isaac Sim 4.5.0 on node1 (108). The collector picks an object from a random position on a table, moves it to another random position, and repeats for N episodes. Output format: LeRobot v3.0 dataset (parquet + MP4 videos).

No MagicSim dependency. Uses `omni.isaac.core` ArticulationAction for joint control and `UsdPhysics.FixedJoint` for grasp attachment — the same pattern as the working `isaac_kitchen_collector.py`.

**New file**: `sim-service/isaac_pick_place_collector.py`
**Reuse**: `sim-service/lerobot_writer.py` (LeRobot v3.0 writer, no changes needed)

---

## 2. Task Definition

Each episode:
1. Reset scene — object spawns at random position on table surface
2. Robot starts at HOME joint configuration (gripper open)
3. Move to object: APPROACH_PICK → DOWN_PICK → CLOSE_GRIPPER
4. Lift and transport: LIFT → MOVE_TO_PLACE → DOWN_PLACE → OPEN_GRIPPER
5. Retract: RETRACT → HOME
6. Record state/action/cameras at every step → save episode

**Success criteria**: Object ends at place position (within 5cm tolerance), robot returns to HOME.

---

## 3. Waypoint Sequence

Extend the existing 5-waypoint grasp sequence to a 9-waypoint pick-and-place cycle:

```
Step   Waypoint         Arm Config               Gripper    Purpose
─────────────────────────────────────────────────────────────────────
 1     HOME             [0, -0.785, 0, -2.356,   OPEN       Start position
                         0, 1.571, 0.785]
 2     APPROACH_PICK    Computed from pick_pos    OPEN       Above pick target
 3     DOWN_PICK        Computed from pick_pos    OPEN       At pick target height
 4     CLOSE_GRIPPER    Same as DOWN_PICK         CLOSED     Grasp object
 5     LIFT             Same as APPROACH_PICK     CLOSED     Lift object
 6     MOVE_TO_PLACE    Computed from place_pos   CLOSED     Above place target
 7     DOWN_PLACE       Computed from place_pos   CLOSED     At place target height
 8     OPEN_GRIPPER     Same as DOWN_PLACE        OPEN       Release object
 9     RETRACT          Same as MOVE_TO_PLACE     OPEN       Retract upward
```

### Waypoint Computation

Each episode generates `pick_pos` (x, y) and `place_pos` (x, y) randomly on the table surface. The waypoint arm configurations derive from these positions.

**Approach**: Use predefined joint-space waypoints with Y-axis offset to approximate different XY positions. The existing collector uses fixed joint waypoints that map to a specific workspace region. For varied pick/place positions, we apply delta offsets to joints 0 (base rotation) and 3 (elbow) to shift the end-effector XY.

```python
# Base waypoints (from existing collector, proven to work)
HOME       = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
APPROACH   = [0.0, -0.3,   0.0, -1.5,   0.0, 1.8,   0.785]
GRASP_DOWN = [0.0,  0.1,   0.0, -1.2,   0.0, 2.0,   0.785]
LIFT       = [0.0, -0.5,   0.0, -1.8,   0.0, 1.6,   0.785]

# For pick-and-place, use same base waypoints for both pick and place phases.
# Shift XY by adjusting joint 0 (base rotation, ±0.3 rad for ±10cm lateral)
# and optionally joint 1 (shoulder) for forward/back reach.
#
# Alternative (recommended): Use omni.isaac.motion_generation.ArticulationKinematicsSolver
# for IK if available. Falls back to joint-offset method if not.
```

### Linear Interpolation Between Waypoints

Each transition uses `steps_per_segment` frames (default 30 = 1 second at 30 FPS):

```python
for step in range(steps_per_segment):
    alpha = (step + 1) / steps_per_segment
    arm_target = (1 - alpha) * start_arm + alpha * end_arm
    gripper_target = (1 - alpha) * start_gripper + alpha * end_gripper
    _set_joint_targets(franka, arm_target, gripper_target)
    world.step(render=True)
    # record state + action + cameras
```

Total frames per episode: 9 transitions × 30 steps = 270 frames (9 seconds at 30 FPS).

---

## 4. Scene Setup

### 4.1 Table

Create a `FixedCuboid` (static collider) as the table surface:

```python
from omni.isaac.core.objects import FixedCuboid

table = FixedCuboid(
    prim_path="/World/Table",
    position=[0.5, 0.0, 0.4],    # In front of robot
    size=1.0,
    scale=[0.6, 0.8, 0.02],      # 60cm x 80cm x 2cm
    color=[0.6, 0.4, 0.2],       # Wood color
)
```

Table surface Z = 0.41 (center 0.4 + half thickness 0.01).

### 4.2 Franka Robot

Load from S3 assets (NOT Nucleus — Nucleus URLs are broken in Isaac Sim 4.5.0):

```python
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage

assets_root = get_assets_root_path()
franka_usd = assets_root + "/Isaac/Robots/Franka/franka.usd"
add_reference_to_stage(usd_path=franka_usd, prim_path="/World/Franka")
# Position at origin — table is placed in front of it
```

Robot base position: `(0.0, 0.0, 0.0)` — table surface is at Z=0.41, within Franka reach (workspace ~0.3-0.8m from base).

### 4.3 Graspable Objects

Use either simple `DynamicCuboid` (for fast prototyping) or YCB objects from S3:

```python
# Option A: Simple cube (fast, reliable physics)
from omni.isaac.core.objects import DynamicCuboid
obj = DynamicCuboid(
    prim_path="/World/PickObject",
    position=[pick_x, pick_y, table_z + OBJECT_SIZE/2],
    size=0.04,          # 4cm — fits in Franka gripper (max ~8cm opening)
    mass=0.1,           # 100g
    color=[0.8, 0.1, 0.1],
)

# Option B: YCB objects (for visual diversity)
YCB_OBJECTS = [
    "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
    "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
]
# add_reference_to_stage(assets_root + ycb_path, "/World/PickObject")
```

### 4.4 Physics Material (High Friction)

Apply to object for reliable grasping:

```python
from pxr import UsdPhysics, UsdShade

grip_mat = UsdShade.Material.Define(stage, "/World/GripMaterial")
physics_api = UsdPhysics.MaterialAPI.Apply(grip_mat.GetPrim())
physics_api.CreateStaticFrictionAttr(1.5)
physics_api.CreateDynamicFrictionAttr(1.0)
physics_api.CreateRestitutionAttr(0.0)

# Bind to object
UsdShade.MaterialBindingAPI.Apply(obj_prim).Bind(
    grip_mat, UsdShade.Tokens.weakerThanDescendants, "physics"
)
```

### 4.5 Cameras

Two cameras, same as grasp collector:

| Camera | Prim Path | Position | Orientation | Purpose |
|--------|-----------|----------|-------------|---------|
| cam_high | `/World/OverheadCam` | `(0.3, -0.3, 1.3)` relative to robot | 25° tilt, focal 14mm | Overview of workspace |
| cam_wrist | `{robot}/panda_hand/wrist_cam` | `(0.05, 0, 0.04)` on hand | 90° pitch | Close-up of grasp |

Resolution: 512×512 pixels, 30 FPS.

### 4.6 Lighting

```python
from pxr import UsdLux

# Dome light (ambient)
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.GetIntensityAttr().Set(3000.0)

# Key light (directional)
key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
key.GetIntensityAttr().Set(5000.0)

# Workspace light (sphere, above table)
work = UsdLux.SphereLight.Define(stage, "/World/WorkLight")
work.GetIntensityAttr().Set(15000.0)
work.GetRadiusAttr().Set(0.15)
# Position above table center
```

---

## 5. Episode Randomization

### 5.1 Pick Position

Random XY on table surface, constrained to Franka reachable area:

```python
# Table surface bounds (relative to robot base)
TABLE_X_RANGE = (0.3, 0.7)   # Forward from robot
TABLE_Y_RANGE = (-0.3, 0.3)  # Left-right
TABLE_Z = 0.41                # Table surface height

pick_x = rng.uniform(*TABLE_X_RANGE)
pick_y = rng.uniform(*TABLE_Y_RANGE)
```

### 5.2 Place Position

Random XY on table, must be at least `MIN_PICK_PLACE_DIST` from pick position:

```python
MIN_PICK_PLACE_DIST = 0.1  # 10cm minimum separation

for _ in range(100):  # rejection sampling
    place_x = rng.uniform(*TABLE_X_RANGE)
    place_y = rng.uniform(*TABLE_Y_RANGE)
    dist = ((place_x - pick_x)**2 + (place_y - pick_y)**2) ** 0.5
    if dist >= MIN_PICK_PLACE_DIST:
        break
```

### 5.3 Object Variety

Each episode randomly selects from object pool:

```python
# Option A: Same cube, random color
color = rng.uniform(0.1, 0.9, size=3)

# Option B: Random YCB object (if using YCB)
ycb_idx = rng.integers(len(YCB_OBJECTS))
```

### 5.4 Waypoint Noise

Same as grasp collector — add ±0.05 rad noise to each waypoint joint angle:

```python
WAYPOINT_NOISE_RAD = 0.05
noise = rng.uniform(-WAYPOINT_NOISE_RAD, WAYPOINT_NOISE_RAD, size=7)
noisy_waypoint = base_waypoint + noise
```

---

## 6. Grasp Mechanism (FixedJoint)

Isaac Sim 4.5.0 Franka gripper physics simulation is unreliable for small objects. The existing grasp collector uses `UsdPhysics.FixedJoint` to rigidly attach the object to the end-effector — this works 100% of the time.

### 6.1 Attach (after CLOSE_GRIPPER)

```python
from pxr import UsdPhysics

# Check proximity first
eef_pos = get_eef_world_position(stage, eef_prim_path)
obj_pos = target_object.get_world_pose()[0]
xy_dist = np.linalg.norm(eef_pos[:2] - obj_pos[:2])
z_dist = abs(eef_pos[2] - obj_pos[2])

if xy_dist < 0.08 and z_dist < 0.15:
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/GraspJoint")
    joint.GetBody0Rel().SetTargets([eef_prim_path])     # panda_hand
    joint.GetBody1Rel().SetTargets([object_prim_path])   # /World/PickObject
    # Object now moves with hand
```

### 6.2 Detach (after OPEN_GRIPPER at place position)

```python
joint_prim = stage.GetPrimAtPath("/World/GraspJoint")
if joint_prim and joint_prim.IsValid():
    stage.RemovePrim("/World/GraspJoint")
# Object drops to table (gravity)
```

### 6.3 Episode Reset

At the start of each episode, remove any leftover GraspJoint, then teleport object to new pick position:

```python
# Remove old joint
joint_prim = stage.GetPrimAtPath("/World/GraspJoint")
if joint_prim and joint_prim.IsValid():
    stage.RemovePrim("/World/GraspJoint")

# Reset object position
target_object.set_world_pose(
    position=np.array([pick_x, pick_y, TABLE_Z + OBJECT_SIZE/2]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
)

# Settle physics
for _ in range(10):
    world.step(render=True)
```

---

## 7. Data Format (LeRobot v3.0)

### 7.1 State Vector (23 dimensions)

Same structure as grasp collector:

```
Index   Name              Description
──────────────────────────────────────────────
0-6     joint_pos_0..6    7 arm joint positions (rad)
7-13    joint_vel_0..6    7 arm joint velocities (rad/s)
14-15   gripper_pos_0..1  2 gripper finger positions (m)
16-18   eef_pos_x/y/z     End-effector position (m)
19-22   eef_quat_w/x/y/z  End-effector orientation (quaternion)
```

### 7.2 Action Vector (8 dimensions)

```
Index   Name              Description
──────────────────────────────────────────────
0-6     action_joint_0..6  7 arm joint targets (rad)
7       action_gripper     Gripper target (0.04=open, 0.0=closed)
```

### 7.3 Cameras

| Camera | Key | Resolution | FPS |
|--------|-----|-----------|-----|
| Overhead | `observation.images.cam_high` | 512×512 | 30 |
| Wrist | `observation.images.cam_wrist` | 512×512 | 30 |

### 7.4 Output Directory Structure

```
{output_dir}/
├── meta/
│   ├── info.json                    # Dataset metadata, feature shapes
│   ├── stats.json                   # Per-feature mean/std/min/max
│   ├── tasks.jsonl                  # Task description
│   └── episodes/chunk-000/
│       └── file-000.parquet         # Episode index + lengths
├── data/chunk-000/
│   └── file-000.parquet             # All frames: state, action, timestamps
└── videos/
    ├── observation.images.cam_high/chunk-000/
    │   ├── episode_00000.mp4
    │   ├── episode_00001.mp4
    │   └── ...
    └── observation.images.cam_wrist/chunk-000/
        ├── episode_00000.mp4
        └── ...
```

### 7.5 Writer Usage

```python
from lerobot_writer import SimLeRobotWriter

writer = SimLeRobotWriter(
    output_dir=args.output,
    repo_id="local/franka_pick_place",
    fps=30,
    robot_type="franka",
    state_dim=23,
    action_dim=8,
    camera_names=["cam_high", "cam_wrist"],
    camera_resolution=(512, 512),
    state_names=STATE_NAMES,   # 23 names
    action_names=ACTION_NAMES,  # 8 names
)

# Per frame:
writer.add_frame(episode_index, frame_index, state, action, timestamp, next_done)
writer.add_video_frame("cam_high", rgb_high)
writer.add_video_frame("cam_wrist", rgb_wrist)

# Per episode:
writer.finish_episode(episode_index, length=frame_count, task="pick and place")

# End:
writer.finalize()
```

---

## 8. Core Functions to Implement

### 8.1 `_make_pick_place_waypoints(pick_pos, place_pos, rng)`

Generate the 9-waypoint sequence for one episode.

**Input**: `pick_pos` (x, y), `place_pos` (x, y), `rng` (numpy Generator)
**Output**: `list[tuple[str, np.ndarray, float]]` — `[(name, arm_7d, gripper_val), ...]`

Logic:
1. Compute `approach_pick_joints` — joint angles that bring EEF above `pick_pos`
2. Compute `down_pick_joints` — lower EEF to table height at `pick_pos`
3. Compute `approach_place_joints` — joint angles that bring EEF above `place_pos`
4. Compute `down_place_joints` — lower EEF to table height at `place_pos`
5. Apply `WAYPOINT_NOISE_RAD` noise to each

**Joint computation approach** (two options):

**Option A — Joint offset method (simpler, works without IK solver)**:
```python
# Base waypoints target center of table (0.5, 0.0)
# Joint 0 (base) rotates ±0.3 rad → shifts Y by ±10cm
# Joint 1 (shoulder) adjusts ±0.2 rad → shifts X (reach)
def _pos_to_joint_offset(target_x, target_y, center_x=0.5, center_y=0.0):
    dx = target_x - center_x
    dy = target_y - center_y
    j0_offset = np.arctan2(dy, 0.5) * 0.8   # base rotation
    j1_offset = -dx * 0.5                     # shoulder for reach
    return np.array([j0_offset, j1_offset, 0, 0, 0, 0, 0])
```

**Option B — IK solver (more accurate, if available)**:
```python
# Use omni.isaac.motion_generation if available
from omni.isaac.motion_generation import ArticulationKinematicsSolver
ik_solver = ArticulationKinematicsSolver(robot_articulation, ...)
joint_targets = ik_solver.compute_inverse_kinematics(target_position, target_orientation)
```

**Recommendation**: Start with Option A for reliability. Add Option B as enhancement if joint-offset positions are inaccurate.

### 8.2 `_run_pick_place_episode(...)`

Main episode function. Signature follows existing `_run_episode()`:

```python
def _run_pick_place_episode(
    episode_index: int,
    world,
    franka,
    cameras: dict[str, Camera],
    stage,
    eef_prim_path: str,
    get_prim_at_path,
    usd, usd_geom,
    writer: SimLeRobotWriter,
    rng: np.random.Generator,
    fps: int,
    steps_per_segment: int,
    target_object,
    table_z: float,
) -> int:  # returns frame count
```

Steps:
1. `world.reset()` + 10 settle frames
2. Remove old GraspJoint if exists
3. Generate random `pick_pos`, `place_pos`
4. Teleport object to `pick_pos`
5. Generate waypoints via `_make_pick_place_waypoints()`
6. Teleport robot to HOME (kinematic, no physics)
7. For each waypoint transition:
   - Interpolate arm + gripper over `steps_per_segment` frames
   - Each frame: `_set_joint_targets()`, `world.step(render=True)`, record state/action/cameras
8. After CLOSE_GRIPPER: check proximity → attach FixedJoint
9. After OPEN_GRIPPER: detach FixedJoint
10. Return total frame count

### 8.3 Reuse from `isaac_kitchen_collector.py` (copy directly)

These functions work as-is — copy them into the new file:

| Function | Lines | Purpose |
|----------|-------|---------|
| `_to_numpy()` | 189-198 | Convert any tensor/array to numpy |
| `_pad_or_trim()` | 201-206 | Pad/trim array to fixed size |
| `_extract_state_vector()` | 346-366 | Build 23-dim state from Franka |
| `_set_joint_targets()` | 369-389 | Apply ArticulationAction to Franka |
| `_capture_rgb()` | 392-428 | Get RGB from Camera object |
| `_get_eef_pose()` | 316-343 | Get EEF world position + quaternion |
| `_resolve_eef_prim()` | 293-313 | Find panda_hand prim path |

### 8.4 Constants

```python
STATE_DIM = 23
ACTION_DIM = 8
FPS = 30
CAMERA_NAMES = ["cam_high", "cam_wrist"]
CAMERA_RESOLUTION = (512, 512)
ROBOT_TYPE = "franka"
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
WAYPOINT_NOISE_RAD = 0.05
OBJECT_SIZE = 0.04        # 4cm cube
OBJECT_MASS = 0.1         # 100g
MIN_PICK_PLACE_DIST = 0.1 # 10cm min pick-to-place distance
STEPS_PER_SEGMENT = 30    # frames per waypoint transition

# Table
TABLE_X_RANGE = (0.3, 0.7)
TABLE_Y_RANGE = (-0.3, 0.3)
TABLE_Z = 0.41

# HOME waypoint (same as grasp collector)
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# STATE_NAMES and ACTION_NAMES: same 23 + 8 names from isaac_kitchen_collector.py
```

---

## 9. CLI Interface

```bash
/isaac-sim/python.sh /sim-service/isaac_pick_place_collector.py \
    --num-episodes 50 \
    --output /sim-service/datasets/pick_place \
    --steps-per-segment 30 \
    --seed 42 \
    --streaming --streaming-port 49102
```

### Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--num-episodes` | 50 | Number of episodes |
| `--output` | `/sim-service/datasets/pick_place` | Output directory |
| `--headless` | True | Headless mode |
| `--fps` | 30 | Dataset FPS |
| `--steps-per-segment` | 30 | Interpolation steps per waypoint transition |
| `--seed` | 12345 | Random seed |
| `--streaming` | False | Enable WebRTC monitoring |
| `--streaming-port` | 49102 | WebRTC port |
| `--num-objects` | 1 | Number of objects on table (future: multi-object) |
| `--use-ycb` | False | Use YCB objects instead of cubes |

---

## 10. Bridge API Integration

The interactive `run_interactive.py` already has stubbed `/collect/start|status|stop` endpoints. Replace the stub `_run_collection()` with actual pick-and-place logic.

### 10.1 Current Stubs (run_interactive.py:362-388)

```python
# Current stub — just sleeps:
for step in range(100):
    time.sleep(1.0 / 30.0)
```

### 10.2 Replace With

```python
def _run_collection():
    # Import the collector module
    from isaac_pick_place_collector import run_collection_in_process

    run_collection_in_process(
        world=world,                    # Shared World instance
        simulation_app=simulation_app,
        num_episodes=num_episodes,
        output_dir=output_dir,
        stop_event=_collect_stop,
        progress_callback=lambda ep: _state["collect_progress"].update({"completed": ep}),
    )
```

### 10.3 Thread Safety

The collector MUST run on the main thread (Isaac Sim API is not thread-safe). Two approaches:

**Approach A — Command queue (recommended)**:
Collection request goes to `_cmd_queue`. Main loop detects "collecting" state and calls `_run_one_collection_step()` each frame instead of `simulation_app.update()`.

```python
# Main loop modification
while simulation_app.is_running():
    _process_commands()
    if _state["collecting"]:
        _run_one_collection_step()  # drives one frame of collection
    elif PHYSICS_RUNNING:
        world.step(render=True)
    else:
        simulation_app.update()
```

**Approach B — Separate process**:
Launch collection as a separate K8s pod (same pattern as `isaac_kitchen_collector.py`). Simpler but uses an extra GPU.

**Recommendation**: Use Approach A for interactive instances. Use standalone script (Approach B) for bulk collection jobs.

### 10.4 Frontend Trigger

Frontend already has a "Collect Data" button. Wire it to:

```
POST /v1/sim/bridge/collect/start
{
    "num_episodes": 50,
    "output_dir": "/data/datasets/pick_place_001",
    "skill": "pick_place"
}
```

Backend `apis/sim.py` proxies to Bridge API on the instance.

---

## 11. `main()` Function Structure

```python
def main():
    args = parse_args()

    # 1. Create SimulationApp
    simulation_app = SimulationApp({"headless": True, ...})

    # 2. Optional: enable WebRTC streaming
    if args.streaming:
        enable_nvcf_streaming(args.streaming_port)

    # 3. Create World + ground plane
    world = World(physics_dt=1/120, rendering_dt=1/30)

    # 4. Create table (FixedCuboid)
    table = create_table(stage)

    # 5. Add Franka from S3
    franka_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
    add_reference_to_stage(franka_usd, "/World/Franka")

    # 6. Create cameras (overhead + wrist)
    cam_high = Camera(prim_path="/World/OverheadCam", ...)
    cam_wrist = Camera(prim_path="/World/Franka/panda_hand/wrist_cam", ...)

    # 7. Setup lighting
    setup_lighting(stage)

    # 8. Create graspable object (will be repositioned each episode)
    obj = DynamicCuboid(prim_path="/World/PickObject", ...)
    apply_grip_material(stage, "/World/PickObject")

    # 9. world.reset() + warmup
    world.reset()
    warmup_frames(world, simulation_app, 50)
    franka.initialize()

    # 10. Initialize writer
    writer = SimLeRobotWriter(output_dir=args.output, ...)

    # 11. Episode loop
    rng = np.random.default_rng(args.seed)
    for ep in range(args.num_episodes):
        frames = _run_pick_place_episode(
            episode_index=ep, world=world, franka=franka,
            cameras=cameras, stage=stage, ...,
            rng=rng, target_object=obj, table_z=TABLE_Z,
        )
        writer.finish_episode(ep, length=frames, task="pick and place")

    # 12. Finalize dataset
    writer.finalize()
    simulation_app.close()
```

---

## 12. Known Issues & Gotchas

| Issue | Workaround |
|-------|-----------|
| Franka S3 USD uses `xformOp:translate` (Double precision), but setting it with Float raises `Tf.ErrorException` | Catch `pxrInternal` exceptions — operation still succeeds |
| `world.reset()` loses camera initialization | Call `camera.initialize()` after every `world.reset()` |
| Gripper physics unreliable for small objects | Use `FixedJoint` attach/detach instead of physics gripper |
| `kubectl cp /dev/stdin` creates symlink, not file | Always SCP to host first, then `kubectl cp` |
| warmup takes 40-90s on 2080 Ti | Run 50-100 `simulation_app.update()` frames before starting collection |
| Joint offset method gives imprecise XY | Verify EEF position after each waypoint, log deviations for tuning |
| Single GPU per instance | Schedule collection jobs sequentially, or use multiple GPUs |
| `imageio` not in Isaac Sim 4.5.0 by default | `pip install imageio[pyav]` in pod startup or Dockerfile |

---

## 13. Infrastructure & Connectivity

### 13.1 Cluster Topology

```
Holly's Mac ──VPN──→ magics-hgx (129.105.2.160, control plane, 8x A100)
                           │
                           └──→ node1 / 108 (192.168.100.108, 8x RTX 2080 Ti)
                                 └── Isaac Sim pods run HERE (NVENC required for WebRTC)
```

### 13.2 SSH Access

```bash
# HGX (control plane, kubectl lives here)
ssh hgx                  # alias in ~/.ssh/config → magics@magics-hgx.cs.northwestern.edu

# node1 / 108 (Isaac Sim GPU worker)
ssh -p 12345 magics@192.168.100.108
```

### 13.3 K8s Pod Pattern

Isaac Sim runs as K8s pod on node1 with `hostNetwork: true` (all pods share 108's IP). Each instance gets unique ports via env vars. The backend `apis/sim.py` creates pods via `POST /v1/sim/instances/launch`.

Key pod spec requirements:
- Image: `nvcr.io/nvidia/isaac-sim:4.5.0`
- `hostNetwork: true` (required for WebRTC UDP)
- `nvidia.com/gpu: 1` resource limit
- `nodeSelector: kubernetes.io/hostname: node1`
- Volume mount: `/home/magics/sim-service → /sim-service` (hostPath)
- Start command: `/isaac-sim/python.sh -m pip install flask && /isaac-sim/python.sh -u /sim-service/run_interactive.py`

### 13.4 Detailed Reference Docs

For complete architecture, port allocation, Bridge API endpoints, and known pitfalls, see:
- **`sim-interactive-service.md`** (in the memory directory) — full system architecture
- **`isaac-sim-pitfalls.md`** — WebRTC streaming gotchas, NVENC requirements
- **`run_interactive.py`** source — Bridge API implementation, command queue pattern

---

## 14. Deployment

### 14.1 Standalone Script (for batch collection)

```bash
# On 108, start collector pod
kubectl apply -f sim-collector-job.yaml

# Or run directly in existing Isaac Sim pod:
kubectl exec -it sim-interactive-xxx -- \
    /isaac-sim/python.sh -m pip install flask imageio pyav
kubectl exec -it sim-interactive-xxx -- \
    /isaac-sim/python.sh /sim-service/isaac_pick_place_collector.py \
        --num-episodes 100 --output /data/datasets/pp_001
```

### 14.2 Git Sync

```bash
# Local → GitHub → 108
cd /Users/holly/Documents/code/sim/sim-service
git add isaac_pick_place_collector.py
git push origin main

# On 108:
cd /home/magics/sim-service
GIT_TERMINAL_PROMPT=0 git pull origin main
```

### 14.3 Dataset Copy to HGX

```bash
# After collection completes on 108:
kubectl cp sim-interactive-xxx:/data/datasets/pick_place /tmp/pick_place
scp -r -P 12345 magics@192.168.100.108:/tmp/pick_place /data/datasets/

# Or from HGX directly:
ssh -p 12345 magics@192.168.100.108 "tar czf - /data/datasets/pick_place" | tar xzf - -C /data/datasets/
```

---

## 15. Testing Checklist

- [ ] Scene renders: table, Franka, object visible in WebRTC stream
- [ ] Object spawns at random position on table each episode
- [ ] Franka picks up object (FixedJoint attached, object moves with hand)
- [ ] Franka places object at different position (FixedJoint detached, object stays)
- [ ] State vector is 23-dim, action vector is 8-dim
- [ ] Both cameras (cam_high, cam_wrist) capture non-black frames
- [ ] LeRobot v3.0 output: `meta/info.json`, `data/`, `videos/` all populated
- [ ] `meta/stats.json` has reasonable min/max values (not all zeros)
- [ ] Video MP4 files play correctly and show pick-place sequence
- [ ] 10-episode test run completes without crashes
- [ ] 50-episode run produces ~13,500 frames (270 frames × 50 episodes)
