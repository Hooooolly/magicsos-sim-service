# sim-service: MagicSim Wrapper Service

## Overview

Standalone FastAPI service that wraps MagicSim as a library. Runs on node 108 alongside Isaac Sim (headless). OpenClaw orchestrates the full pipeline by calling this service's HTTP endpoints.

## Architecture

```
OpenClaw (HGX, orchestrator)
    │ HTTP calls via skill
    ▼
sim-service (108, FastAPI :5900)
    │ Python imports
    ▼
MagicSim (library, same process)
    │ Kit API (pxr, omni)
    ▼
Isaac Sim (108, headless)
```

**Design principle**: MagicSim is someone else's repo. We treat it as a read-only library. All our code lives in `sim/sim-service/`. We import MagicSim classes but never modify MagicSim source.

## Pipeline

```
SceneSmith (HGX A100)
    │ generates scene_state.json + USD assets
    ▼
OpenClaw rsync → 108
    │ files now on 108
    ▼
sim-service /scene/load
    │ scene_loader.py converts scene_state.json → MagicSim YAML
    │ SceneManager loads all objects at exact positions
    ▼
sim-service /robot/spawn
    │ Franka (default) spawned at computed position
    ▼
sim-service /collect/start
    │ AtomicSkillManager (Grasp/Place/Reach) + RecordManager
    │ IK solver (differential IK or pink IK)
    ▼
Output (two parallel writes from same sim loop):
├── LeRobot3 parquet     → /data/datasets/{id}/  → RLInf training
└── MagicSim RecordManager → /data/trajectories/{id}/ → user verification
```

## Data Output: Two Writers, One Sim Loop

```
每一步 Isaac Sim step:
├── robot.joint_positions  ──┐
├── robot.action            ─┤── LeRobot3 Writer → parquet + mp4
├── camera[*].rgb           ─┤     /data/datasets/{dataset_id}/
├── camera[*].depth         ─┘     ├── meta/info.json
│                                   ├── data/chunk-000/file-000.parquet
│                                   └── videos/observation.images.*/chunk-000/file-000.mp4
│
├── MagicSim RecordManager ──────── 用户验证 (视频回放 + JSON)
│     /data/trajectories/{job_id}/
│     ├── action/action_merged.json
│     ├── env/camera/cam_*/rgb/rgb.mp4
│     └── collect/collect_merged.json
```

**LeRobot3 parquet columns**: `episode_index`, `action` (7D float32), `observation.state` (14D), `observation.images.cam_high`, `observation.images.cam_wrist`

**RLInf 直接用**: `dataset_path="/data/datasets/{dataset_id}"`, Hydra config 指向这个路径

## Camera Configuration

相机跟场景无关，只跟任务和机器人相关。纯 YAML 配置，MagicSim CameraManager 自动创建。

### 默认相机配置（Franka 桌面操作）

```yaml
# cam_high: 俯视全局视角（固定位置）
cam_high:
  camera:
    mesh: pinhole
    resolution: [224, 224]
    frequency: 30
    pos: [0.0, 0.0, 1.5]       # 桌面上方 1.5m
    ori: [0, -90, 0]            # 正下方看
    clipping_range: [0.1, 5]
  annotator:
    rgb_capture: { type: rgb, device: cpu }

# cam_wrist: 腕部相机（挂 Franka 末端，跟着动）
cam_wrist:
  camera:
    mesh: pinhole
    resolution: [224, 224]
    frequency: 30
    pos: [0.05, 0.0, 0.0]      # gripper 前方 5cm
    ori: [0, -45, 0]            # 45° 向下看操作区域
    mount_link: "robot_0/panda_hand"  # ← 挂到机器人 link
    clipping_range: [0.01, 2.0]
  annotator:
    rgb_capture: { type: rgb, device: cpu }
```

**关键**: `mount_link` 让相机跟随机器人关节运动。不填则固定在世界坐标。
支持 pinhole/kinect/realsense，支持 domain randomization（训练时 pos/ori 加噪声）。

## K8s Job Deployment（80× RTX 2080 Ti）

每个 collection job = 1 GPU = 1 Isaac Sim 实例。K8s 调度到 108/109/110 节点。

```
infra-backend API
    │ POST /embodied/collection/start
    ▼
CollectionJob DB record (status: pending)
    │ background_tasks.add_task()
    ▼
create_data_collection_job()           # k8s/sim_job.py
    │ ConfigMap (job.json) + K8s Job
    ▼
K8s Job on RTX 2080Ti node
    │ collection_runner.py
    ├── MagicLauncher (headless Isaac Sim)
    ├── scene_loader → SceneManager
    ├── CameraManager (YAML config)
    ├── Franka spawn
    ├── AtomicSkill collection loop
    │   ├── LeRobot3 Writer (parquet + mp4)
    │   └── RecordManager (user verification)
    └── Output → /data/datasets/{id}/ + /data/trajectories/{id}/
```

**K8s Job 参数**:
- Image: `rlinf/rlinf:agentic-rlinf0.1-isaaclab`
- GPU: 1× RTX 2080 Ti (`nvidia.com/gpu.product: NVIDIA-GeForce-RTX-2080-Ti`)
- SHM: 100Gi (Isaac Sim 必需)
- Volumes: /data (hostPath), /scenesmith (SceneSmith outputs), /config (ConfigMap)
- host_network: true (Gloo P2P)

**函数**: `create_data_collection_job()` in `infra-backend/k8s/sim_job.py`
**Helper**: `get_collection_job_status()`, `get_collection_job_logs()`, `cancel_collection_job()`

## File Structure

```
sim/sim-service/
├── DESIGN.md            # This file
├── main.py              # FastAPI app + endpoints
├── client.py            # Python SDK (OpenClaw/agent_server use)
├── scene_loader.py      # SceneSmith → MagicSim config converter
├── collection_runner.py # K8s Job 入口：Isaac Sim + 采集 + LeRobot3 写入 (TODO)
├── lerobot_writer.py    # LeRobot3 parquet 直写 helper (TODO)
├── camera_config.yaml   # 默认相机配置 (TODO)
└── requirements.txt     # fastapi, uvicorn, requests
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Isaac Sim status, GPU info, loaded scene |
| `POST` | `/scene/load` | Load SceneSmith scene (accepts scene_dir path) |
| `GET` | `/scene/info` | Current scene: object list, prim count, stage path |
| `POST` | `/robot/spawn` | Spawn robot (type, position, orientation) |
| `GET` | `/robot/state` | Joint positions, end-effector pose, gripper state |
| `POST` | `/collect/start` | Start data collection (skill type, targets, output dir) |
| `GET` | `/collect/status` | Collection progress (trajectory count, state) |
| `POST` | `/sim/step` | Advance simulation N steps |
| `POST` | `/sim/reset` | Reset environment |

## Key Design Decisions

### 1. Queue-based command dispatch

Isaac Sim requires its main loop on the main thread. FastAPI runs in a background thread. We use a request/response queue (same pattern as MagicSim's GymServer) to dispatch commands from HTTP handlers to the sim loop.

### 2. scene_loader.py preserves SceneSmith layout

The existing `scenesmith_bridge` randomizes object placement. `scene_loader.py` reads `scene_state.json` and generates MagicSim YAML with **exact positions/orientations** from SceneSmith. Each object becomes a named instance with fixed `pos`, `ori`, `scale`.

### 3. TidyBot service pattern

Follows the established pattern from TidyBot-Services:
- `main.py`: FastAPI with lifespan, Pydantic models, `/health` endpoint
- `client.py`: Pure Python SDK using `requests` (no heavy deps)
- Registers in `services_wishlist/catalog.json` for auto-discovery
- OpenClaw skill (`SKILL.md`) documents how to use the service

### 4. MagicSim as library

We import and use:
- `SceneManager` — scene/object loading
- `AtomicSkillManager` — Grasp, Place, Reach, Push skills
- `RecordManager` — trajectory recording
- `DifferentialIKController` / `PinkIKController` — IK solving
- `MagicLauncher` — Isaac Sim initialization
- `Franka.py` config — robot definition

We do NOT modify any MagicSim files.

## Reused Code

| Source | What | How |
|--------|------|-----|
| `scenesmith_bridge/` | `discover_room_usd()`, `discover_asset_roots()` | Import functions |
| `isaac_bridge_api.py` | Prim traversal, robot state reading | Reference pattern |
| MagicSim `GymServer.py` | Queue-based HTTP ↔ sim loop | Reference architecture |
| MagicSim `AtomicSkillManager` | Manipulation skills | Python import |
| MagicSim `RecordManager` | Trajectory recording | Python import |
| MagicSim `SceneManager` | Scene/object management | Python import |
| TidyBot `yolo-service/` | FastAPI + client SDK structure | Template |

## Deployment

- **Host**: 192.168.100.108 (RTX 2080 Ti)
- **Port**: 5900
- **Runtime**: Isaac Sim Python (`/isaac-sim/python.sh`)
- **MagicSim path**: Must be on PYTHONPATH
- **Start**: `/isaac-sim/python.sh main.py` or via Docker

## Integration with OpenClaw

OpenClaw skill at `skills/sim-service/SKILL.md` documents:
- Service endpoint: `http://192.168.100.108:5900`
- When to use each endpoint
- Typical workflow sequence
- Error handling

OpenClaw calls sim-service via HTTP, reads responses, decides next step. The service is stateful (loaded scene persists until reset or new load).

## Catalog Registration

```json
{
  "sim-service": {
    "type": "service",
    "description": "MagicSim wrapper. Loads SceneSmith scenes into Isaac Sim, spawns robots, runs IK-based data collection with trajectory recording.",
    "service_repo": "local (sim/sim-service/)",
    "client_sdk": "sim/sim-service/client.py",
    "host": "http://192.168.100.108:5900",
    "endpoints": ["GET /health", "POST /scene/load", "GET /scene/info", "POST /robot/spawn", "GET /robot/state", "POST /collect/start", "GET /collect/status", "POST /sim/step", "POST /sim/reset"],
    "version": "0.1.0"
  }
}
```
