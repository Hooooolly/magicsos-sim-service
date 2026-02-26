# Collector Hotfix Log — 2026-02-26 — Curobo Early-Close Tail Skip

## Scope
- File: `isaac_pick_place_collector.py`
- Area: Curobo pick phase (`PRE_GRASP -> DOWN_PICK` execution branch)

## Context
- `DOWN_PICK` planning already used `include_object=False` during Curobo world update.
- Remaining issue: when Curobo reach gate was satisfied mid-trajectory, code still executed the synthetic `DOWN_PICK -> CLOSE` tail transition, which could continue driving toward the nominal down-pick target before close hold.

## Change
- Capture `_execute_curobo_trajectory(...)` return value as `curobo_early_reach_halt`.
- If `curobo_early_reach_halt == True`, skip the `DOWN_PICK -> CLOSE` tail transition.
- Keep close/verify flow unchanged: close hold still uses current arm pose after pick phase handling.

## Expected Effect
- Preserve early-halt pose for close instead of over-driving toward nominal down-pick waypoint.
- Better alignment with MagicSim-style early-close behavior.

## Validation
- `python3 -m py_compile sim-service/isaac_pick_place_collector.py` passed.
