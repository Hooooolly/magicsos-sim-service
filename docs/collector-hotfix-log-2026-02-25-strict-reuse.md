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

## Update (articulation-ready hotfix)
- `run_interactive.py`: collect start no longer recreates `World` by default.
  - New env gate: `COLLECT_RECREATE_WORLD_BEFORE_START` (default `0`).
- `isaac_pick_place_collector.py`: after `world.reset()`, collector now calls `world.play()` and steps a few frames before `franka.initialize()` to ensure articulation view is available.

## Rollback note
- Rolled back `run_interactive.py` and `isaac_pick_place_collector.py` to commit `0af7cf6` behavior to restore previously working scene collect flow.
- Reason: strict-mode regression caused articulation initialization failures and unstable collect behavior.

## Update (annotation + scene-stability hotfix, 2026-02-25 20:4x)
- `grasp_poses/mug_grasp_pose.json`
  - Added metadata scale hint for Isaac C1 mug:
    - `position_scale_hint: 0.01`
    - `position_unit: cm`
- `isaac_pick_place_collector.py`
  - Removed collect-time automatic scene mutation calls:
    - no auto robot base alignment during collect setup
    - no auto object recenter during collect setup
    - no per-attempt object recenter loop
  - Added annotation sanitize fallback:
    - if sanitize filters all candidates, fallback to unsanitized annotation set (enabled by default via `COLLECT_ALLOW_UNSANITIZED_ANNOTATION_FALLBACK=1`).
  - Motivation:
    - keep user-arranged scene stable
    - avoid false-negative “annotation missing” when sanitize is over-restrictive for valid mug annotations.
