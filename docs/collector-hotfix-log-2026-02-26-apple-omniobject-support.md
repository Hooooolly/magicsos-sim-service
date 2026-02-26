# Collector Hotfix Log (2026-02-26): OmniObject Apple/Ball Pose Support

## What changed

- Added OmniObject token aliases for collector annotation lookup:
  - `apple`, `apple001`, `001`
  - `ball`, `sphere`, `ball012`, `012`, `015`, `016`
  - `cube`, `block`
- Added runtime scene helper `create_apple(...)` in `run_interactive.py`:
  - Loads `apple_001` USD from known local paths.
  - Applies rigid body + mesh collision.
  - Binds `grasp_pose_path` / `annotation_path` custom data to apple annotation file when present.
- Updated runtime helper import sanitizer and stage-mutation detector to include `create_apple`.
- Added ready-to-use annotation files into `sim-service/grasp_poses/`:
  - `apple_001_grasp_pose.json`
  - `ball_012_grasp_pose.json`

## Why

- Collector could only reliably enforce mug annotation workflows.
- We need a pre-annotated OmniObject path for quick validation of non-mug grasp targets.

## Validation

- `python3 -m py_compile run_interactive.py isaac_pick_place_collector.py`
- JSON parse sanity:
  - `apple_001_grasp_pose.json` has `grasp.body` poses.
  - `ball_012_grasp_pose.json` has `grasp.body` poses.

## Quick usage

- In scene chat code path:
  - `create_apple(stage, prim_path="/World/Apple", position=(0.30, 0.00, 0.79))`
- Then collect with hint:
  - `POST /collect/start` with `target_objects=["apple"]`

