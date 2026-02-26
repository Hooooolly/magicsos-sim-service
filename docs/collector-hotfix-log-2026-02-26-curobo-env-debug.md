# Collector Hotfix Log (2026-02-26): Curobo Env + Debug Observability

## Scope
- File: `isaac_pick_place_collector.py`
- Goal:
  - Ensure interactive collect defaults to Curobo planner path unless explicitly overridden.
  - Add richer runtime diagnostics for "object vs target vs eef" drift analysis.

## Changes
1. Planner backend default
- `COLLECT_PLANNER_BACKEND` default changed from `ik` to `curobo`.
- Startup log now prints selected planner backend and raw env value.

2. Verbose debug switch
- Added `COLLECT_VERBOSE_DEBUG` env switch (default enabled).
- Startup log prints verbose mode status.

3. Episode-level thresholds print
- At episode start, logs reach/close gate thresholds and planner mode.

4. Attempt-level world-state print
- Added `collect-debug` logs at:
  - `pre_plan`
  - `pre_close_gate`
- Each log includes:
  - object world position
  - eef world position
  - target world position (if available)
  - deltas (`obj_to_target`, `eef_to_target`, `eef_to_obj`)

5. Failure logs include thresholds
- `reach_before_close failed` now logs corresponding threshold values.
- `close_verify failed` now logs threshold values.
- `retrieval_verify failed` now logs threshold values.

## Expected Outcome
- Easier diagnosis of:
  - wrong target frame / wrong annotation transform
  - planner not converging to target
  - gating failures due to strict thresholds vs execution drift
- New interactive instances should run Curobo path by default when env is propagated.
