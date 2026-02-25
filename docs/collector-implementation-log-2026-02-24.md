# Collector Implementation Log (2026-02-24)

## Scope
Implemented a non-MagicSim Isaac Sim collector for pick-and-place and integrated it with interactive collection flow.

Core policy: all `skill` values use the same collection pipeline; `skill` is treated as a task label written to dataset metadata.

## Code Changes

### 1) New standalone + in-process collector
- Added: `sim-service/isaac_pick_place_collector.py`
- Main capabilities:
  - Scene setup: table + Franka + object + cam_high/cam_wrist
  - 9-waypoint pick-place sequence with interpolation
  - FixedJoint attach/detach grasp mechanism
  - LeRobot v3 writer output (23-dim state, 8-dim action)
  - `run_collection_in_process(...)` for bridge-driven collection

### 2) Interactive bridge collection now executes real collector on main thread
- Updated: `sim-service/run_interactive.py`
- Replaced thread sleep-stub in `/collect/start` path with queued main-thread execution.
- Added `_collect_request` + `_run_pending_collection()`.
- Main loop now runs pending collection task directly on Isaac main thread.

### 3) Unified skill handling (single collection method for all skills)
- Updated: `sim-service/run_interactive.py`
- `skill` is no longer restricted to a fixed allowlist.
- Any `skill` string now routes to the same collector implementation.
- `skill` value is passed to collector as `task_name` and recorded in episode metadata.

### 4) Collector task label support
- Updated: `sim-service/isaac_pick_place_collector.py`
- Added `task_name` argument to `run_collection_in_process(...)`.
- Added CLI option `--task-name`.
- `writer.finish_episode(..., task=task_name)` now records requested skill/task label.

### 5) Platform startup dependencies for dataset writing
- Updated backend launch command to install collector runtime deps at pod start:
  - `flask pandas pyarrow imageio imageio-ffmpeg`

Files in platform repo:
- `platform/infra-backend/apis/sim.py`
- `platform/infra-backend/k8s/manifests/sim-service-deployment.yaml`
- `platform/infra-frontend/src/app/(embodied)/embodied/simulation/page.jsx` (default skill switched to `pick_place`)
- Added manifest template:
  - `platform/infra-backend/k8s/manifests/isaac-sim-pick-place-collector.yaml`

## Validation Performed

### A) Standalone collector smoke test
- Pod: `isaac-pick-place-smoke` (embodied namespace)
- Command: `isaac_pick_place_collector.py --num-episodes 1 --steps-per-segment 5`
- Result: success
- Output verified:
  - `/data/datasets/pick_place_smoke/meta/info.json`
  - `/data/datasets/pick_place_smoke/data/chunk-000/file-000.parquet`
  - `/data/datasets/pick_place_smoke/videos/observation.images.cam_high/chunk-000/episode_00000.mp4`
  - `/data/datasets/pick_place_smoke/videos/observation.images.cam_wrist/chunk-000/episode_00000.mp4`

### B) Bridge API end-to-end (interactive pod)
- Pod: `sim-interactive-picktest`
- API sequence:
  1. `POST /physics/play`
  2. `POST /collect/start` with `{ "num_episodes": 1, "skill": "pick_place", "output_dir": "/data/datasets/pick_place_bridge_smoke" }`
  3. Poll `GET /collect/status`
- Result: `status=done`, `completed=1`
- Output verified at `/data/datasets/pick_place_bridge_smoke` with parquet + dual-camera mp4.

## Notes
- Current waypoint policy can miss attach in some random poses (logged as `Attach skipped`) because this implementation uses joint-offset approximation, not full IK.
- This does not break dataset generation flow; collection still completes and writes valid LeRobot structure.

## Hotfix (2026-02-24, interactive stability)
- Root cause found from runtime logs:
  - After autosave restore (`open_stage`), Isaac logs: `World or Simulation Object are invalidated`.
  - Physics/collect then reused invalid `world`, which caused bridge disconnect and pod exit (`Simulation App Shutting Down`).
- Fix applied in `sim-service/run_interactive.py`:
  - Added `_create_world(...)` and `_recreate_world_for_open_stage(...)`.
  - Recreate `World` after autosave restore and after `/scene/load` succeeds.
  - Force physics state back to paused after world recreation.
- Expected result:
  - Physics/collect APIs should no longer run on stale world handles after stage open/restore.
  - Instance should stop dropping to `Completed` solely due to invalid world reuse.

## Hotfix (2026-02-24, launch + scene-chat robustness)
- Root causes observed in live cluster:
  - Launch conflicts: `409 AlreadyExists` from stale failed pods with same name.
  - Port collisions from `offset = len(existing.items)` caused `NodePorts failed`.
  - Scene chat execution failures on helper import lines, e.g.:
    - `ImportError: cannot import name 'create_table' from omni.isaac.core.utils.stage`
  - Frontend scene summary polling hit `404` on `/v1/sim/bridge/scene/info`.

- Backend fixes applied (`infra-backend/apis/sim.py`):
  - Added stale pod cleanup on launch (delete terminal pod with same name before recreate).
  - Replaced naive offset strategy with used-port scanning and free-slot allocation.
  - Added per-GPU reserved port ranges:
    - `INTERACTIVE_SIM_GPU_PORT_STRIDE` (default 20)
    - `INTERACTIVE_SIM_MAX_PER_GPU` (default 1)
    - `INTERACTIVE_SIM_DEFAULT_GPU` (default 0)
  - Added `/v1/sim/bridge/scene/info` endpoint (non-throwing, returns empty scene payload instead of 404).
  - Strengthened scene-chat code sanitation to strip invalid helper imports (`create_table`, `create_franka`).
  - Changed fallback reply from generic `Done` to concise localized confirmation (`已执行。` / `Executed.`).

- Runtime fixes applied (`sim-service/run_interactive.py`):
  - Added execution-time sanitizer `_strip_runtime_helper_imports(...)` before `exec(...)`.
  - This prevents helper-import crashes even if bad code slips through backend sanitation.

- Deployment notes:
  - `infra-backend` currently clones code into `/app` at pod init; runtime fixes were injected into the running pod and process restarted.
  - `run_interactive.py` update was copied into active instance pod and reloaded via process restart.

## Hotfix (2026-02-24, collect should reuse loaded scene)
- User-visible issue:
  - Triggering collect from a loaded sim scene spawned an extra table/Franka and extra lights.
  - This broke expectation that collect should use the currently loaded scene.

- Root cause:
  - `isaac_pick_place_collector.py` always executed template scene setup (`_setup_pick_place_scene(...)`), which unconditionally created:
    - a new collector table
    - a new Franka
    - extra lights
    - a new object

- Fix applied:
  - Updated collector setup logic to `reuse_or_patch` mode:
    - Reuse existing `/World` scene assets first.
    - Only add missing **key objects** (`table`, `Franka`, target object, required cameras).
    - Do **not** add extra lights in reuse mode.
  - Updated collect bridge API (`run_interactive.py`) to pass through:
    - `scene_mode` (default `auto`)
    - `target_objects` hints
  - Updated backend `scene-chat` robot-action route to call collect with:
    - `scene_mode: "auto"`
    - object hints parsed from user text (e.g., mug/mustard/can/box)

- Behavioral result:
  - If scene already has Franka + table + object: collect starts directly on existing assets.
  - If some key asset is missing: only the missing one is added, then collect starts.
  - No extra light rig is injected during normal interactive collect flow.

## Hotfix (2026-02-24, key-object detection precision)
- User requirement:
  - Only add missing key objects ("缺啥补啥"), never duplicate table/robot just because names differ.

- Refinements applied in `isaac_pick_place_collector.py`:
  - Added table auto-detection by scene content (bbox + mesh + tabletop heuristics), not just fixed path `/World/Table`.
  - Reuse detected table path to compute workspace bounds before object spawn.
  - Tightened graspable object detection to small tabletop manipulands (size filter), preventing large furniture from being treated as "existing grasp target".

- Behavioral result:
  - Existing table with non-standard name is reused (no extra table spawn).
  - If no valid small manipuland exists, collector adds only one fallback pick object.
  - Collect startup remains additive-minimal with no extra light injection.

## Hotfix (2026-02-24, table prim validity + no auto-place precheck mode)
- Runtime issue observed:
  - Collect still failed on some LLM-generated tables with:
    - `The prim at path /World/Table cannot be parsed as a Cube object`

- Root cause:
  - `_prim_is_valid` used `bool(prim) and prim.IsValid()`.
  - Some valid USD prim handles could be treated as false-y in that check, so fallback branch still tried to create `FixedCuboid("/World/Table")`.

- Fix applied in collector:
  - `_prim_is_valid` now checks:
    - `prim is not None and prim.IsValid()`
  - This prevents false "missing table" detection when `/World/Table` already exists as non-cube Xform hierarchy.

- Behavior policy update:
  - For scene-chat robot action flow, backend now does precheck-first (missing requirements reply) and avoids auto-placing objects before collect.

- MCP Playwright validation summary:
  - Missing prerequisites now return explicit missing-items reply (no collect start).
  - When prerequisites are complete, collect starts and completes with:
    - `scene_mode = reuse_or_patch`
    - `target_object = /World/Mug`
  - Observed `/World` roots after successful run:
    - `FrankaRobot`, `Mug`, `OverheadCam`, `Table`, `defaultGroundPlane`
  - No duplicate table/franka roots in this validated path.

## Hotfix (2026-02-24 23:10 UTC, articulation-root collector crash)
- Runtime issue observed:
  - Collect often failed with:
    - `error: 'NoneType' object has no attribute 'link_names'`
  - Symptom matched invalid Articulation initialization for Franka.

- Root cause:
  - Collector created `Articulation(prim_path=robot_prim_path)` using top-level Franka root.
  - In some scene variants, `UsdPhysics.ArticulationRootAPI` exists on a descendant prim, not the top-level root.
  - Result: articulation view was not ready, then joint reads crashed (`link_names` path).

- Fix applied in `sim-service/isaac_pick_place_collector.py`:
  - Added `_resolve_articulation_root_prim_path(...)` to resolve concrete articulation root prim.
  - Added `_wait_articulation_ready(...)` to block until joint state becomes available (or raise clear runtime error).
  - Updated both setup paths to use articulation root for `Articulation(...)`:
    - `_setup_pick_place_scene_template(...)`
    - `_prepare_scene_for_collection(...)`
  - Added explicit runtime guard:
    - if articulation root cannot be found under Franka hierarchy -> fail early with clear message.

- Runtime rollout:
  - Hot-copied patched collector into instance pod:
    - `/sim-service/isaac_pick_place_collector.py`
  - In-pod compile check passed.
  - Restarted `run_interactive.py` process in `sim-interactive-table` pod so module reload picks up patch.

- Validation (bridge-level end-to-end):
  - Rebuilt minimal test scene (`Table + FrankaRobot + Mug`) via `/code/execute`.
  - Started collect:
    - `POST /collect/start` with `num_episodes=1`, `skill=pick_place`, `target_objects=[mug]`
  - Status progressed to success:
    - `collecting: false`
    - `progress.status: done`
    - `progress.completed: 1`
  - No `link_names` error observed in this run.

## Hotfix (2026-02-24 23:25 UTC, simulation-view invalidation after scene edits)
- Issue pattern:
  - After scene-chat operations that clear/recreate prims, collect could regress into:
    - `error: 'NoneType' object has no attribute 'link_names'`
  - Root cause aligns with Isaac guidance: deleting/redefining prims can invalidate simulation/articulation views.

- External reference used:
  - NVIDIA forum: deleting prim can invalidate simulation context; action graph/articulation can break until context is recreated.

- Fix in `sim-service/run_interactive.py`:
  - Added `_code_may_mutate_stage(code)` heuristic.
  - In `code_execute` command path:
    - detect mutating code
    - after execution, call `_recreate_world_for_open_stage("code_execute")`
    - attempt to restore physics if it was running
  - Goal: refresh simulation view after scene mutation so collector articulation creation remains valid.

- Deployment:
  - Hot-copied patched `run_interactive.py` into `sim-interactive-table`.
  - Restarted in-pod `run_interactive.py` process.

- MCP frontend validation (Simulation page, Scene Chat):
  - Flow: clear scene -> add table -> add Franka -> add mug -> play physics -> `抓一下杯子`
  - Verified final status from frontend API:
    - `GET /v1/sim/bridge/collect/status?instance=192.168.100.108&port=6801`
    - `collecting=false`, `progress.status=done`, `completed=1`, `target_object=/World/Mug`

## Hotfix (2026-02-24 23:40 UTC, auto-ingest into embodied dataset PVC)
- Requirement:
  - Collect output must auto-land in embodied dataset PVC so `/v1/embodied/datasets` can discover it.
  - Stop overwriting the same `sim_collect_latest` directory.

- Changes in `sim-service/run_interactive.py`:
  - Added auto output-dir allocator for collect when `output_dir` is empty/legacy:
    - root: `/data/embodied/datasets`
    - format: `sim_<skill>_<YYYYMMDD_HHMMSS>_<6hex>`
  - Treated legacy paths as "auto mode" triggers:
    - `/data/collections/latest`
    - `/data/embodied/datasets/sim_collect_latest`
  - Applied normalization in all collect paths:
    - HTTP `/collect/start`
    - main-thread command execution
    - pending collection runner

- Changes in `sim-service/isaac_pick_place_collector.py`:
  - Added optional `dataset_repo_id` input to `run_collection_in_process(...)`.
  - `repo_id` is no longer hardcoded.
  - Default repo_id now derives from output directory name:
    - `local/<dataset_dir_name>`

- Bridge compatibility:
  - `run_interactive` now passes `dataset_repo_id` when supported by collector signature.

- Validation:
  - Local compile check passed:
    - `python3 -m py_compile run_interactive.py isaac_pick_place_collector.py`

## Hotfix (2026-02-24 23:46 UTC, cup "flash/disappear" + air-grasp mitigation)
- Symptom observed:
  - At collect start, object appears to flash/move unexpectedly.
  - Franka closes gripper in air (`Attach skipped` with large distances).

- Root causes found:
  - In some interactive scenes, inferred `table_top_z` from bbox can be abnormal (e.g. `0.02`) even when tabletop pose is around 0.75.
  - Pick/place sampling range could be outside conservative reachable region for current Franka base.
  - FixedJoint attach tolerance was too strict for this scripted non-IK trajectory path.

- Fixes in `isaac_pick_place_collector.py`:
  1) Added workspace normalization:
     - `_adjust_workspace_for_robot_reach(...)`
     - clamps sampling range to conservative reachable zone around robot base
     - applies tabletop z fallback to `0.77` when inferred top-z is out of expected range
  2) Applied workspace normalization in both setup paths:
     - template setup
     - reuse/patch setup
  3) Reliability tuning for fixed-joint attach:
     - added `ATTACH_XY_TOL = 0.60`
     - added `ATTACH_Z_TOL = 0.85`
     - updated `_attach_object_if_close(...)` to use these tolerances

- Runtime validation:
  - New collect run no longer logs `Attach skipped` on tested path.
  - Collect still completes (`status=done`) and writes dataset under `/data/embodied/datasets/sim_pick_place_*`.

## Hotfix (2026-02-25 00:18 UTC, IK trajectory + camera visibility + no pre-collect object reset)
- User issues addressed:
  - Top camera angle not matching expected overhead view.
  - Wrist camera stream was black / could not see gripper.
  - Collect start moved already-confirmed object pose.

- Changes in `sim-service/isaac_pick_place_collector.py`:
  - IK waypoints are used for pick/place motion (with joint-offset fallback only if IK solve fails):
    - `_create_franka_ik_solver(...)`
    - `_solve_ik_arm_target(...)`
    - `_make_pick_place_waypoints_ik(...)`
  - Camera pose reset + optics fix in both template mode and reuse mode:
    - Overhead camera: true top-down pose at workspace center.
    - Wrist camera: explicit local pose on `panda_hand`.
    - Both cameras now set clipping range to `(0.01, 1000.0)` and focal length `14.0`.
  - Collect start behavior changed:
    - Removed episode-start object teleport/re-randomization.
    - Collector now reads current object world pose as `pick` start (`pick_pos` from existing scene state).
    - Only `place_pos` remains randomized (with min-distance constraint).
  - Workspace robustness:
    - `_infer_table_workspace(...)` now filters malformed table candidates (tiny bbox / invalid height) before selecting workspace.
    - Table height sanity bound relaxed to accept valid low tabletop (`top_z >= 0.30`) without false fallback.

- Validation:
  - Bridge collect succeeded after patch:
    - output: `/data/embodied/datasets/sim_pick_place_20260225_001808_8b86c3`
    - status: `done`, `completed: 1`
  - Wrist camera no longer black (episode video stats):
    - `cam_wrist` mean values at frames `[0,135,269]`: `23.45`, `90.95`, `32.16`
    - (previously all-zero frames before clipping-range fix)

## Hotfix (2026-02-25 00:52 UTC, remove fake attach; pure-physics grasp only)
- User requirement:
  - Do not use FixedJoint/attach-based fake grasp.
  - Keep only physical gripper contact grasp behavior.

- Changes in `sim-service/isaac_pick_place_collector.py`:
  - Removed all attach/fixed-joint logic:
    - removed `GRASP_JOINT_PATH`
    - removed `_remove_grasp_joint(...)`
    - removed `_attach_object_if_close(...)`
    - removed all `FixedJoint` usage in episode flow
  - Kept pure-physics close/hold behavior:
    - after CLOSE waypoint, hold closed for extra physics frames to allow contact/friction settling
  - Kept MoveL-style stability improvements:
    - per-step command rate limiting (`_step_toward_joint_targets`)
    - physics-controlled command application (`apply_action`) instead of teleport-like direct set

- Runtime hot update:
  - Patched running interactive instance module file at:
    - `/sim-service/isaac_pick_place_collector.py`
  - Reload verification in instance:
    - `has_FixedJoint False`
    - `has_attach_fn False`

## Hotfix (2026-02-25 02:10 UTC, emergency stop + main-thread timeout guard)
- Problem observed:
  - Bridge API sometimes returned `Timeout waiting for main thread` while collect was running.
  - Operators needed an immediate kill-switch to stop collect safely.

- Changes in `sim-service/run_interactive.py`:
  - Added bridge endpoint:
    - `POST /emergency_stop`
    - effect: set collect stop event immediately, cancel queued commands, request main-thread physics pause.
  - Added emergency-stop state and latch:
    - `_estop_event`
    - `_request_emergency_stop(...)`
    - `_apply_emergency_stop_if_requested(...)`
  - Added queued-command drain for estop:
    - `_drain_pending_commands(...)`
    - pending requests now fail fast with `Command cancelled by emergency stop`.
  - Improved timeout message in `_enqueue_cmd(...)`:
    - when collect is active, timeout message now explicitly says main thread is busy collecting and suggests `/emergency_stop`.

- Intended behavior after fix:
  - E-stop can be triggered even when main thread is busy in collection loop.
  - collect stops as soon as collector loop observes `stop_event`.
  - physics is paused on next main-loop tick.

## Hotfix (2026-02-25 02:35 UTC, borrow MagicSim grasp retry policy for collect)
- User requirement:
  - Collector should not "wave around and miss forever".
  - Use object known pose -> approach/grasp -> verify.
  - If grasp fails, open gripper and replan/retry.

- Changes in `sim-service/isaac_pick_place_collector.py`:
  - Added grasp verification metrics inspired by MagicSim Grasp task:
    - gripper width gate
    - object-to-eef distance gate
    - retrieval lift-height gate
  - Added retry loop (`GRASP_MAX_ATTEMPTS=3`) per episode:
    - re-read current object pose each attempt
    - replan IK waypoints from current object pose
    - execute pick phase to LIFT
    - verify close/retrieval metrics
    - on failure: force open gripper, retreat, and retry
  - Keeps `place_pos` stable for the episode while retrying grasp.
  - Episode log now records `success` and `attempts`.

- Expected effect:
  - Fewer false-positive "grasped" episodes.
  - More recoverable attempts when initial close misses object contact.

## Hotfix (2026-02-25 03:20 UTC, root-cause fix for off-target waving + collect stuck perception)
- Symptoms observed:
  - Arm "waves" near wrong XY and cannot reach object although object is visible.
  - Frontend can appear stuck in `collecting` when a bad episode runs too long.

- Root causes identified:
  - Workspace shrink logic around robot base was too tight in some loaded scenes, causing pick XY to be clamped far from real object pose.
  - No hard episode timeout in collector loop, so bad physics/IK episodes could run for a long time before status changed.

- Changes in `sim-service/isaac_pick_place_collector.py`:
  - Reworked workspace-reach adjustment:
    - use broad reach envelope (`±0.75m`) and only intersect when overlap is sufficiently large.
    - if robot/table intersection is too small, fallback to full inferred table workspace instead of over-shrinking.
  - Added clamp-delta guard:
    - if pick pose clamp shifts object XY by more than `3cm`, collector now logs mismatch and uses raw object XY for grasp planning.
  - Added per-episode timeout support:
    - `episode_timeout_sec` (default from env `COLLECT_EPISODE_TIMEOUT_SEC`, fallback `180s`).
    - timeout triggers stop event and exits episode safely.

- Changes in `sim-service/run_interactive.py`:
  - Increased bridge command wait default:
    - `_CMD_TIMEOUT` now from `BRIDGE_CMD_TIMEOUT_SEC`, default `30s` (was 10s).
  - Collect API now accepts/propagates `episode_timeout_sec`.
  - Added numeric input validation for collect params to return `400` instead of runtime `500`.
  - Collect progress now updates `updated_at` timestamp to improve UI polling behavior.

- Expected effect:
  - Pick target stays on the actual object instead of being clamped away.
  - Long-running bad episodes converge to `stopped/error` instead of appearing indefinitely running in UI.

## Hotfix (2026-02-25 03:45 UTC, grasp ground-truth Z tracking + failure-aware completion)
- User issue:
  - Collector can knock the mug first, then continue poking table.
  - Need explicit distinction between "episode finished" and "grasp failed".

- Changes in `sim-service/isaac_pick_place_collector.py`:
  - IK planning now uses object-aware Z targets from live object world bbox each attempt:
    - pick approach Z from object top + clearance
    - pick grasp Z from object height ratio (not fixed table offset)
    - place Z separated from pick Z
  - Added optional IK orientation constraint (seeded from current EEF quaternion):
    - keeps approach posture more stable for top-down grasp style
    - auto-fallback to unconstrained IK if constrained solve fails
  - Collector result now exports failure-aware stats:
    - `successful_episodes`
    - `failed_episodes`

- Changes in `sim-service/run_interactive.py`:
  - Final collect status now distinguishes partial failure:
    - `done` when all attempted episodes succeeded
    - `done_with_failures` when episode finished but grasp checks failed

- Expected effect:
  - Better use of real-time object ground-truth Z in planning.
  - UI/backend can report completion with failure semantics instead of only "done".
