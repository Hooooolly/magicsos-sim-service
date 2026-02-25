# Regression Audit (2026-02-25)

## Scope
Reviewed the last two-day logs and cross-checked with current code status to explain why previously fixed issues appear to return.

Checked docs:
- `sim-service/docs/collector-implementation-log-2026-02-24.md`
- `sim-service/docs/pick-place-collector-design.md`
- `sim-service/docs/mug-grasp-pose-workflow.md`
- `infra-backend/docs/sim-openclaw-integration-log-2026-02-24.md`
- `infra-backend/docs/sim-scene-chat-helper-prompt-hotfix-2026-02-25.md`
- `infra-backend/docs/sim-mug-scale-hotfix-2026-02-25.md`

## Findings

### A) `sim-service` collector hotfixes are still present (not rolled back)
- Pure-physics grasp path (no `FixedJoint` attach) is present.
- Retry + phase gates are present (`GRASP_MAX_ATTEMPTS`, reach/close/retrieval checks).
- Tip-mid reach gate and replan-from-current-state are present.
- Episode timeout path is present (`episode_timeout_sec`, env fallback).
- Emergency stop route is present in bridge (`/emergency_stop`).
- Auto dataset output normalization to `/data/embodied/datasets` is present.
- Grasp GT export fields are present (`observation.object_pose_world`, `observation.grasp_target_*`).

Conclusion: current "old problems back" are unlikely caused by `sim-service` collector code rollback.

### B) Main regression source is likely in `infra-backend` scene-chat pipeline
- Scene-chat still runs deterministic verification/fix passes after primary code execution.
- User-facing code uses `display_code = code` after helper injection.  
  Effect: helper function bodies can still appear in returned code blocks, making output long/confusing.
- Unsupported-object deterministic gate is present, but delayed/queued turn behavior in chat can still make users observe scene change after an "unavailable" reply when another pending action executes.

## Why this matches observed symptoms
- "Reply says unavailable but scene still changed": consistent with asynchronous queued action overlap, not collector rollback.
- "Code block suddenly very long again": consistent with helper-injected `code` being returned as `display_code`.
- "Collector logic looked reverted": logs show latest collector features are still in source.

## Recommended next fix order
1. Backend `scene-chat`: return pre-injection user code for display (keep execution code internal).
2. Backend `scene-chat`: tighten verification-fix trigger to run only when strict object issues are detected, not broadly.
3. Frontend chat handoff: ensure one user turn maps to one scene action result (no delayed cross-turn replay).
4. Keep forcing instance pod refresh after deploy so runtime is not mixing old in-pod hotcopied code with new repo state.

## Notes
- This audit is code-level only (repository state). Runtime pods can still behave differently if not fully rolled out/restarted.
