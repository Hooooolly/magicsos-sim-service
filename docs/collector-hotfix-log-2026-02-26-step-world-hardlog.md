# Collector Hotfix Log - 2026-02-26 - Per-step World Pose Hard Log

## Why
Need deterministic per-step diagnostics to quickly separate:
- wrong target pose (`target_world` drift from `object_world`), vs
- execution/tracking failure (`eef_world` never reaches `target_world`).

## Changes
File: `isaac_pick_place_collector.py`

1. Added env switch:
- `COLLECT_STEP_WORLD_LOG` (default `1`)
- startup log prints effective value.

2. Added per-step hard log in `_record_frame()`:
- new helper `_log_step_world_debug(frame_id)`
- logs every recorded frame with:
  - `object_world=(x,y,z)`
  - `eef_world=(x,y,z)`
  - `target_world=(x,y,z)` or `(none)`
- includes `episode`, `attempt`, `frame` for alignment with collect timeline.

## Log Example
`collect-step: episode=1 attempt=2 frame=187 object_world=(...) eef_world=(...) target_world=(...)`

## Notes
- Logging is gated by both `COLLECT_VERBOSE_DEBUG=1` and `COLLECT_STEP_WORLD_LOG=1`.
- This intentionally increases log volume for diagnosis runs.
