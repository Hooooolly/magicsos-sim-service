# Collector Hotfix Log - 2026-02-25 - Strict Reuse Only

## Background
Collect was unexpectedly rebuilding/patching scene content, which overrode user-arranged simulation scenes.

## Scope
Files changed:
- `run_interactive.py`
- `isaac_pick_place_collector.py`

## Changes
1. Collect API defaults to strict scene mode.
- `/collect/start` now defaults `scene_mode` to `strict`.
- Main-thread collect request parser also defaults `scene_mode` to `strict`.

2. `scene_mode=auto` no longer patches scene.
- In `run_collection_in_process(...)`, `auto/reuse/existing/patch/preserve` are mapped to strict reuse-only behavior.
- `template/generated` scene mode is disabled and raises an explicit error.

3. Reuse setup is forced into strict validation.
- `_setup_pick_place_scene_reuse_or_patch(...)` now forces `allow_patch=False`.
- Missing required scene components now fail fast with explicit errors instead of auto-creating:
  - table
  - Franka articulation
  - target grasp object
  - overhead camera (`/World/OverheadCam`)
  - wrist camera (`<robot>/panda_hand/wrist_cam`)
- Target object must already have rigid body and mesh collision APIs.

4. No automatic scene re-layout in strict flow.
- No robot base auto-alignment.
- No target object recentering.
- No template fallback.

## Expected behavior
Collect uses the user’s current scene only and never auto-builds a fallback scene.
If required scene elements are missing, collect fails with a direct reason instead of silently modifying the scene.

## Update (camera exception)
- In strict scene mode, only cameras are allowed to be auto-added when missing:
  - /World/OverheadCam
  - <robot>/panda_hand/wrist_cam
- Table/Franka/target object remain strict-required and are never auto-created.

## Update (articulation init)
- In strict mode, collector now also runs world.reset() before initialize() to ensure valid articulation views and avoid  errors.
- This reset is for sim-view initialization on current stage; no template scene creation path is enabled.

## Update (articulation init)
- In strict mode, collector now also runs `world.reset()` before `initialize()` to ensure valid articulation views and avoid "Franka articulation not ready" errors.
- This reset is for sim-view initialization on current stage; template scene creation is still disabled.
