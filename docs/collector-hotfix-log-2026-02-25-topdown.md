# Collector Hotfix Log (2026-02-25)

## Scope
- Service: `sim-service` (`isaac_pick_place_collector.py`)
- Goal: diagnose repeated collect failures, force a top-down debug path, and add runtime guards for unstable object states.

## Code Changes
- Added env flag parser:
  - `_read_bool_env(...)`
- Added top-down override:
  - `COLLECT_FORCE_TOPDOWN_GRASP=1` forces top-down orientation and bypasses annotation orientation selection.
- Added robot base alignment in reuse mode:
  - `_align_robot_base_for_collect(...)`
  - `COLLECT_ALIGN_ROBOT_BASE=1` (default true)
- Added workspace sanity helpers:
  - `_ensure_object_within_workspace(...)`
  - `COLLECT_RECENTER_OBJECT_IF_OUTSIDE_WORKSPACE=1`
  - `COLLECT_RECENTER_OBJECT_EACH_ATTEMPT=1`
- Fixed missing import:
  - added `import math` for base alignment yaw computation.
- Template mode fixes:
  - imported `DynamicCuboid` in `_setup_pick_place_scene_template(...)`
  - forced table workspace inference to use `TABLE_PRIM_PATH`
    via `_infer_table_workspace(..., table_prim_path=TABLE_PRIM_PATH)`

## Runtime Validation Performed
- Hot-updated pod file:
  - Pod: `embodied/sim-interactive-tablenew`
  - Deployed via `kubectl cp` + `/code/execute` module reload.
- Enabled runtime flags inside Isaac process:
  - `COLLECT_FORCE_TOPDOWN_GRASP=1`
  - `COLLECT_ALIGN_ROBOT_BASE=1`
  - `COLLECT_RECENTER_OBJECT_IF_OUTSIDE_WORKSPACE=1`
  - `COLLECT_RECENTER_OBJECT_EACH_ATTEMPT=1`

## Observed Results
- Top-down override is active (seen in logs).
- Core failure persists:
  - repeated IK jump/clamp warnings
  - `reach_before_close` checks fail on all 3 retries
  - episodes end as `done_with_failures`
- In reuse mode, object pose repeatedly explodes (bbox center drifts to huge values).
- In template mode, failures are now deterministic and cleaner, but still not grasp-success.
- Latest template run also showed intermittent stale-prim error:
  - `Accessed invalid expired 'Cube' prim </World/PickPlaceCollector/Table>`

## Conclusion
- Current failures are not only intent/prompt issues; they include runtime scene/physics instability and planning robustness gaps.
- Top-down path is now available as a controlled debug mode, but not yet sufficient for reliable success on mug grasp.
