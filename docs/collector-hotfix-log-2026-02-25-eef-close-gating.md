# Collector Hotfix Log - 2026-02-25 - EEF Frame + Close Gating

## Context
- Symptom in collect: robot looked like "hand connector" chasing pose and poking object/table.
- Symptom in grasp phase: gripper sometimes did not reliably close at the stabilized contact pose.
- Reference behavior from MagicSim (`AtomicSkill/Grasp.py`): `MoveL` reach/halt first, then `ParallelGripper` close.

## Root Cause
- In `_execute_transitions`, early reach halt was applied to both `DOWN_PICK` and `CLOSE`.
- When `CLOSE` transition was early-halted, close hold loop could be skipped.
- After early halt in `DOWN_PICK`, close transition could still drift arm toward waypoint instead of closing at current stabilized arm pose.
- IK frame preference could pick `right_gripper` before `panda_hand` in some robots, increasing frame mismatch risk against collector metrics/annotation conversion that are hand-frame based.

## Changes
- File: `isaac_pick_place_collector.py`

1. Early halt scope tightened:
- only apply early halt on `DOWN_PICK` (descending phase),
- no early halt on `CLOSE`.

2. Added close anchor behavior:
- when `DOWN_PICK` halts early, capture current arm joints as `close_anchor_arm`,
- run close at this anchor instead of re-chasing `DOWN_PICK` target.

3. Decoupled arm motion and gripper close/open during transition interpolation:
- for transition end `CLOSE`/`OPEN`, keep gripper command at start value during interpolation,
- perform actual close/open in the explicit hold loops.
- and reduce `CLOSE`/`OPEN` transition interpolation to a single step (phase handoff only).

4. IK frame preference updated:
- prefer `panda_hand` / `panda_hand_tcp` before `right_gripper`,
- added warning log when selected IK frame differs from `eef_prim` leaf.

## Validation
- `python3 -m py_compile isaac_pick_place_collector.py` passed.

## Expected Runtime Effect
- close phase should no longer start while arm is still chasing post-halt target,
- fewer "poking/scooping" attempts before close,
- better consistency between annotation pose conversion, reach metric, and IK frame.
