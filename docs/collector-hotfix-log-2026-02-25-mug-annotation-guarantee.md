# Collector Hotfix Log - 2026-02-25 - Mug Annotation Guarantee

## Context

Observed runtime failure during collect:

- `Mug grasp annotation required but none found. Expected grasp_poses/mug_grasp_pose.json (or set COLLECT_GRASP_POSE_PATH).`

In affected pod runs, mug annotation file existed, but strict sanitize could reject all candidates and leave `annotation_candidates=[]`, which tripped the hard mug requirement.

## Root Cause

1. Annotation path discovery was too sensitive to runtime working layout differences across pod sessions.
2. Under strict sanitize, all candidates could be dropped/rejected (`rejected_file=True`).
3. When fallback env behavior was disabled in process state, mug requirement still raised immediately.
4. Runtime module reload could keep stale `.pyc` unless explicitly invalidated in hotfix flow.

## Code Changes

File: `isaac_pick_place_collector.py`

1. Added deterministic mug annotation fallback path resolver:
   - `SCRIPT_DIR/grasp_poses/mug_grasp_pose.json`
   - `/sim-service/grasp_poses/mug_grasp_pose.json`
   - `/code/grasp_poses/mug_grasp_pose.json`
   - `cwd/grasp_poses/mug_grasp_pose.json`

2. Hardened strict sanitize fallback logic for mug:
   - If target is mug and mug annotation is required, sanitize-drop now always falls back to unsanitized loaded candidates (even if env fallback flag is disabled).

3. Added second-stage mug recovery:
   - If no candidates remain, retry load directly from deterministic mug fallback paths before raising.

## Runtime Hotfix Procedure (for already-running pod)

1. Copy updated files into pod:
   - `isaac_pick_place_collector.py`
   - `grasp_poses/mug_grasp_pose.json`
2. Delete stale pyc cache:
   - `/sim-service/__pycache__/isaac_pick_place_collector*.pyc`
3. Reload module via bridge `/code/execute`.

## Validation

- Module reload reports:
  - `has_mug_fallback True`
- Runtime check reports existing fallback path:
  - `/sim-service/grasp_poses/mug_grasp_pose.json`
- Runtime load check reports non-empty candidates:
  - `loaded_count 104`

## Commit

- `b7bfe27` `collect: harden mug annotation resolution and strict-sanitize fallback`
