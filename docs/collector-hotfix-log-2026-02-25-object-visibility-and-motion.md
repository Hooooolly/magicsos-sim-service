# Collector Hotfix Log - Object Visibility and Motion Stability (2026-02-25)

## Issue
- In early collected videos, mug was often not visible.
- Robot arm appeared to move aggressively ("flailing") while failing grasp attempts.

## Root Cause
- Workspace recenter logic only checked XY; if object dropped below table (or sank/interpenetrated) but XY still on-table, object was not recovered.
- Target object resolver in reuse scene mode could pick a poor candidate (hidden/off-table object) because selection was mostly token/order based, without tabletop proximity scoring.
- Joint per-step command limit was relatively high by default, increasing apparent motion aggressiveness when repeatedly replanning.

## Code Changes
- File: `isaac_pick_place_collector.py`

1. Improved target object selection for reuse/patch mode
- Added tabletop-aware scoring in `_resolve_target_object_prim(...)`:
  - supports workspace/top-z aware ranking,
  - prefers hinted object classes,
  - favors candidates on/near tabletop,
  - penalizes below-table candidates,
  - lightly prefers objects with grasp annotations.

2. Added Z-aware workspace recenter guard
- Updated `_ensure_object_within_workspace(...)`:
  - now validates both XY and Z,
  - recenters object to tabletop center if below/above safe Z range even when XY is valid.

3. Better diagnostics in logs
- Recenter logs now print `raw_bottom_z/raw_top_z` and `reason_xy/reason_z`.
- Reuse-mode setup now logs selected target object path and bbox.

4. Reduced default arm command aggressiveness
- Added env-controlled command limit:
  - `COLLECT_MAX_ARM_STEP_RAD` (default `0.03`, clamped to `[0.005, 0.10]`).

## Validation Notes
- Static check passed:
  - `python3 -m py_compile isaac_pick_place_collector.py`
- Isaac runtime behavior should be validated in-instance by checking:
  - selected object log line,
  - no repeated "invisible object" retries,
  - fewer violent oscillations during failed attempts.

## Additional Patch - Annotation Retry / IK Screening (MagicSim-style) (2026-02-25)

### Why
- Previous annotation usage still retried weak candidates too often.
- Failed grasp attempts did not strongly feed back into next annotation candidate choice.
- Candidate list could include unreachable poses; this caused extra flailing before fallback.

### What Changed
- File: `isaac_pick_place_collector.py`

1. Added failed annotation pose cache + retry feedback
- New cache key is per `(object_prim_path, annotation_path)`.
- On failed phases, current annotation target index is marked failed:
  - `reach_verify_failed`
  - `close_verify_failed`
  - `retrieval_verify_failed`
- On successful retrieval, failed cache for that key is cleared.

2. Upgraded annotation selector with group ordering + IK screening
- `_select_annotation_grasp_target(...)` now:
  - prioritizes `functional_handle` group over `body`,
  - rotates candidate order by attempt index,
  - avoids previously failed pose IDs when alternatives exist,
  - rejects candidates outside tabletop workspace envelope,
  - runs IK feasibility screening:
    - pass 1: strict orientation,
    - pass 2: relaxed orientation.
- Selected target now carries debug metadata:
  - `candidate_group`
  - `ik_feasible`
  - `ik_orientation_relaxed`
  - `annotation_tip_mid_to_hand_converted`

3. Added attempt-level logs for observability
- Logs now show:
  - carried failed pose IDs at episode start (if any),
  - selected annotation index/group and IK mode,
  - failed pose ID marks and reasons.

### Config Knobs Added
- `COLLECT_ANNOTATION_IK_SCREEN_MAX_CANDIDATES` (default `64`, clamped `1..512`)
- `COLLECT_ANNOTATION_POSE_IS_TIP_MID_FRAME` (default enabled)

### Validation
- Static check passed after patch:
  - `python3 -m py_compile isaac_pick_place_collector.py`

## Additional Patch - Persistent Logs (2026-02-25)

### Why
- Runtime debugging depended on terminal logs only; pod restarts made root-cause tracing difficult.

### What Changed
- `run_interactive.py` now tees stdout/stderr into persistent log files under `/data/embodied/logs/sim-service` by default.
- Bridge `/health` includes `log_file` and `log_dir`.
- Standalone collector now writes a file log (default `/data/embodied/logs/sim-service/collector`, fallback `<output_dir>/_logs`).
- Added usage/reference doc: `docs/log-persistence.md`.
