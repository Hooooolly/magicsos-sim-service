"""FastAPI wrapper for MagicSim. Starts in degraded mode without Isaac Sim."""
from __future__ import annotations
import importlib
import inspect
import logging
import os
import queue
import subprocess
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scene_loader import generate_magicsim_yaml, load_scene_state
SERVICE_VERSION = "0.1.0"
LOGGER = logging.getLogger("sim-service")
logging.basicConfig(level=logging.INFO)
class GPUInfo(BaseModel):
    name: Optional[str] = None
    total_memory_mb: Optional[int] = None
    used_memory_mb: Optional[int] = None
class HealthResponse(BaseModel):
    status: str
    sim_ready: bool
    gpu: GPUInfo
    loaded_scene: Optional[str] = None
    version: str
class SceneLoadRequest(BaseModel):
    scene_dir: str = Field(..., description="SceneSmith scene directory")
class SceneLoadResponse(BaseModel):
    success: bool
    scene_name: Optional[str] = None
    object_count: int
    loaded_via_sim: bool
    yaml_path: str
    message: str
class SceneObject(BaseModel):
    path: str
    type: str
    name: Optional[str] = None
class SceneInfoResponse(BaseModel):
    loaded_scene: Optional[str] = None
    prim_count: int
    objects: list[SceneObject]
    source: str
class RobotSpawnRequest(BaseModel):
    robot_type: str = "franka"
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    orientation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
class RobotSpawnResponse(BaseModel):
    success: bool
    robot_type: str
    position: list[float]
    orientation: list[float]
    message: str
class RobotStateResponse(BaseModel):
    joint_positions: list[float]
    end_effector_pose: list[float]
    gripper_state: float
    sim_ready: bool
    message: str
class CollectionStartRequest(BaseModel):
    skill: str = "Grasp"
    target_objects: list[str] = Field(default_factory=list)
    output_dir: str = ""
class CollectionStartResponse(BaseModel):
    success: bool
    running: bool
    skill: str
    target_count: int
    output_dir: str
    message: str
class CollectionStatusResponse(BaseModel):
    running: bool
    progress: float
    collected_samples: int
    message: str
class SimStepRequest(BaseModel):
    n: int = Field(1, ge=1)
class SimStepResponse(BaseModel):
    success: bool
    steps_requested: int
    sim_ready: bool
    message: str
class SimResetResponse(BaseModel):
    success: bool
    sim_ready: bool
    message: str
class StreamingInfoResponse(BaseModel):
    enabled: bool
    type: str = "webrtc"
    streaming_port: int = 8211
    signaling_port: int = 49100
@dataclass
class SimCommand:
    name: str
    payload: dict[str, Any]
    reply: "queue.Queue[dict[str, Any]]"
@dataclass
class AppState:
    sim_ready: bool = False
    loaded_scene: Optional[str] = None
    loaded_scene_dir: Optional[str] = None
    scene_yaml_path: Optional[str] = None
    scene_state: dict[str, Any] = field(default_factory=dict)
    scene_manager: Any = None
    env: Any = None
    launcher: Any = None
    robot_spawned: bool = False
    robot_type: str = "franka"
    command_queue: "queue.Queue[SimCommand]" = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker_thread: Optional[threading.Thread] = None
    collection_running: bool = False
    collection_progress: float = 0.0
    collection_samples: int = 0
    collection_stop: threading.Event = field(default_factory=threading.Event)
    last_error: Optional[str] = None
APP_STATE = AppState()
def _gpu_info() -> GPUInfo:
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.splitlines()[0].split(",")]
            if len(parts) >= 3:
                return GPUInfo(name=parts[0], total_memory_mb=int(parts[1]), used_memory_mb=int(parts[2]))
    except Exception:
        pass
    return GPUInfo()
def _probe_sim_runtime() -> bool:
    """Check if Isaac Sim runtime is available.
    Only check omni.isaac.kit (or isaacsim); pxr is loaded later by Kit extensions."""
    for mod in ("isaacsim.simulation_app", "omni.isaac.kit"):
        try:
            importlib.import_module(mod)
            return True
        except Exception:
            continue
    APP_STATE.last_error = "Neither isaacsim.simulation_app nor omni.isaac.kit found"
    return False
def _call_with_supported_args(fn: Any, payload: dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**payload)
    kwargs = {k: v for k, v in payload.items() if k in sig.parameters}
    if kwargs:
        return fn(**kwargs)
    if len(sig.parameters) == 1:
        for key in ("yaml_path", "scene_yaml", "scene_dir"):
            if key in payload:
                return fn(payload[key])
    if not sig.parameters:
        return fn()
    raise TypeError(f"Unsupported load signature for {fn.__name__}")
def _get_scene_manager() -> Any:
    if APP_STATE.scene_manager is not None:
        return APP_STATE.scene_manager
    last_exc: Optional[Exception] = None
    for mod_name, cls_name in [("magicsim.gym_server.scene_manager", "SceneManager"), ("magicsim.scene_manager", "SceneManager")]:
        try:
            mod = importlib.import_module(mod_name)
            APP_STATE.scene_manager = getattr(mod, cls_name)()
            return APP_STATE.scene_manager
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Unable to initialize SceneManager: {last_exc}")
def _load_scene_in_sim(scene_dir: str, yaml_path: str, scene_yaml: str) -> dict[str, Any]:
    try:
        manager = _get_scene_manager()
        payload = {"scene_dir": scene_dir, "yaml_path": yaml_path, "scene_yaml": scene_yaml}
        for method_name in ("load_scene", "load_from_yaml", "load_yaml", "load"):
            method = getattr(manager, method_name, None)
            if callable(method):
                _call_with_supported_args(method, payload)
                return {"success": True, "message": f"Scene loaded via SceneManager.{method_name}"}
        return {"success": False, "message": "No compatible SceneManager load method found"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}
def _handle_scene_load(payload: dict[str, Any]) -> dict[str, Any]:
    result = _load_scene_in_sim(**payload)
    if not result.get("success"):
        return result
    manager = APP_STATE.scene_manager
    env = None
    if manager is not None:
        env = getattr(manager, "env", None) or getattr(manager, "environment", None) or getattr(manager, "sync_robot_env", None)
        if env is None and hasattr(manager, "step") and hasattr(manager, "reset"):
            env = manager
    APP_STATE.env = env
    return result
def _handle_robot_spawn(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        env = APP_STATE.env
        if env is None:
            return {"success": False, "message": "No environment loaded. Load a scene first."}
        robot_mgr = getattr(env, "robot_manager", None) or getattr(env, "_robot_manager", None)
        if robot_mgr is None:
            return {"success": False, "message": "RobotManager not available from environment"}
        APP_STATE.robot_spawned = True
        APP_STATE.robot_type = payload.get("robot_type", "franka")
        return {"success": True, "message": f"Robot '{APP_STATE.robot_type}' ready via environment"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}
def _handle_robot_state(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        env = APP_STATE.env
        if env is None:
            return {"joint_positions": [], "end_effector_pose": [0.0] * 7, "gripper_state": 0.0, "message": "No environment"}
        robot_mgr = getattr(env, "robot_manager", None) or getattr(env, "_robot_manager", None)
        if robot_mgr and hasattr(robot_mgr, "get_robot_state"):
            state = robot_mgr.get_robot_state(noise_flag=False)
            if state and isinstance(state, (list, tuple)) and len(state) > 0:
                obs = state[0]
                if isinstance(obs, dict):
                    jp = obs.get("joint_positions", obs.get("joint_pos", []))
                    if hasattr(jp, "tolist"):
                        jp = jp.tolist()
                    return {"joint_positions": jp, "end_effector_pose": [0.0] * 7, "gripper_state": 0.0, "message": "ok"}
        return {"joint_positions": [], "end_effector_pose": [0.0] * 7, "gripper_state": 0.0, "message": "Robot state unavailable"}
    except Exception as exc:
        return {"joint_positions": [], "end_effector_pose": [0.0] * 7, "gripper_state": 0.0, "message": str(exc)}
def _handle_sim_step(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        env = APP_STATE.env
        n = payload.get("n", 1)
        if env is None:
            try:
                from omni.isaac.core import World
                world = World.instance()
                if world:
                    for _ in range(n):
                        world.step(render=False)
                    return {"success": True, "message": f"Stepped {n} via World"}
            except Exception:
                pass
            return {"success": False, "message": "No environment or World available"}
        for _ in range(n):
            env.step(action=None, env_ids=None)
        return {"success": True, "message": f"Stepped {n} frames"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}
def _handle_sim_reset(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        APP_STATE.collection_stop.set()
        APP_STATE.collection_running = False
        APP_STATE.collection_progress = 0.0
        APP_STATE.collection_samples = 0
        env = APP_STATE.env
        if env is None:
            sm = APP_STATE.scene_manager
            if sm and hasattr(sm, "reset"):
                sm.reset(soft=False)
                return {"success": True, "message": "Scene reset via SceneManager"}
            return {"success": False, "message": "No environment or SceneManager"}
        env.reset()
        return {"success": True, "message": "Environment reset"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}
def _run_collection_loop(env: Any, skill: str, target_objects: list[str], output_dir: str) -> None:
    """Background collection loop. Updates APP_STATE progress."""
    LOGGER.info("Collection loop started: skill=%s targets=%d output_dir=%s", skill, len(target_objects), output_dir)
    num_episodes = 100
    try:
        skill_mgr = getattr(env, "atomic_skill_manager", None)
        for episode in range(num_episodes):
            if APP_STATE.collection_stop.is_set():
                LOGGER.info("Collection stopped at episode %d", episode)
                break
            try:
                env.reset()
            except Exception as exc:
                LOGGER.warning("Reset failed at episode %d: %s", episode, exc)
            if skill_mgr and hasattr(skill_mgr, "create_atomic_skill"):
                try:
                    skill_mgr.create_atomic_skill(skill, env_id=0)
                except Exception as exc:
                    LOGGER.warning("Skill creation failed: %s", exc)
            max_steps = 200
            for _step in range(max_steps):
                if APP_STATE.collection_stop.is_set():
                    break
                try:
                    env.step(action=None, env_ids=None)
                except Exception:
                    break
            APP_STATE.collection_samples += 1
            APP_STATE.collection_progress = (episode + 1) / num_episodes
            LOGGER.info("Collection episode %d/%d done", episode + 1, num_episodes)
    except Exception as exc:
        LOGGER.error("Collection loop error: %s", exc)
    finally:
        APP_STATE.collection_running = False
        LOGGER.info("Collection finished: %d samples", APP_STATE.collection_samples)
def _handle_collection_start(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        if APP_STATE.collection_running:
            return {"success": False, "message": "Collection already running"}
        skill = payload.get("skill", "Grasp")
        target_objects = payload.get("target_objects", [])
        output_dir = payload.get("output_dir", "/data/trajectories/latest")
        env = APP_STATE.env
        if env is None:
            APP_STATE.collection_running = True
            APP_STATE.collection_progress = 0.0
            APP_STATE.collection_samples = 0
            return {"success": True, "message": "Collection started (placeholder - no env)"}
        APP_STATE.collection_stop.clear()
        APP_STATE.collection_running = True
        APP_STATE.collection_progress = 0.0
        APP_STATE.collection_samples = 0
        t = threading.Thread(target=_run_collection_loop, args=(env, skill, target_objects, output_dir), daemon=True, name="collection-loop")
        t.start()
        return {"success": True, "message": f"Collection started: skill={skill}, targets={len(target_objects)}"}
    except Exception as exc:
        APP_STATE.collection_running = False
        return {"success": False, "message": str(exc)}
def _sim_worker_loop() -> None:
    handlers = {
        "scene_load": _handle_scene_load,
        "robot_spawn": _handle_robot_spawn,
        "robot_state": _handle_robot_state,
        "sim_step": _handle_sim_step,
        "sim_reset": _handle_sim_reset,
        "collection_start": _handle_collection_start,
    }
    while not APP_STATE.stop_event.is_set():
        try:
            cmd = APP_STATE.command_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        handler = handlers.get(cmd.name)
        if handler:
            cmd.reply.put(handler(cmd.payload))
        else:
            cmd.reply.put({"success": False, "message": f"Unknown command: {cmd.name}"})
def _submit_sim_command(name: str, payload: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
    if not APP_STATE.sim_ready:
        return {"success": False, "message": "Isaac Sim runtime not ready"}
    reply: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=1)
    APP_STATE.command_queue.put(SimCommand(name=name, payload=payload, reply=reply))
    try:
        return reply.get(timeout=timeout_s)
    except queue.Empty:
        return {"success": False, "message": f"Timed out waiting for '{name}'"}
def _start_isaac_sim():
    """Start Isaac Sim in the main thread BEFORE uvicorn.
    SimulationApp must own the main thread's event loop."""
    livestream = int(os.environ.get("LIVESTREAM", "0"))
    if not _probe_sim_runtime():
        LOGGER.warning("Isaac Sim runtime missing; service running in degraded mode.")
        return
    try:
        from magicsim.Launch.MagicLauncher import MagicLauncher
        APP_STATE.launcher = MagicLauncher(
            headless=True,
            enable_cameras=True,
            livestream=livestream,
        )
        APP_STATE.sim_ready = True
        LOGGER.info("Isaac Sim started via MagicLauncher (livestream=%d)", livestream)
    except Exception as exc:
        LOGGER.warning("MagicLauncher failed: %s, trying SimulationApp...", exc)
        try:
            try:
                from isaacsim.simulation_app import SimulationApp
            except ImportError:
                from omni.isaac.kit import SimulationApp
            config = {"headless": True}
            if livestream:
                config["livestream"] = livestream
            APP_STATE.launcher = SimulationApp(config)
            APP_STATE.sim_ready = True
            LOGGER.info("Isaac Sim started via SimulationApp (livestream=%d)", livestream)
        except Exception as exc2:
            LOGGER.error("Isaac Sim start failed: %s", exc2)
            APP_STATE.sim_ready = False

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Isaac Sim already started in main thread; just start the worker.
    if APP_STATE.sim_ready:
        APP_STATE.stop_event.clear()
        APP_STATE.collection_stop.clear()
        APP_STATE.worker_thread = threading.Thread(target=_sim_worker_loop, daemon=True, name="sim-worker")
        APP_STATE.worker_thread.start()
        LOGGER.info("Sim worker thread started.")
    yield
    APP_STATE.stop_event.set()
    APP_STATE.collection_stop.set()
    if APP_STATE.worker_thread and APP_STATE.worker_thread.is_alive():
        APP_STATE.worker_thread.join(timeout=2)
    APP_STATE.worker_thread = None
    APP_STATE.scene_manager = None
app = FastAPI(title="MagicSim Service", version=SERVICE_VERSION, lifespan=lifespan)
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok" if APP_STATE.sim_ready else "degraded", sim_ready=APP_STATE.sim_ready, gpu=_gpu_info(), loaded_scene=APP_STATE.loaded_scene, version=SERVICE_VERSION)
@app.get("/streaming/info", response_model=StreamingInfoResponse)
async def streaming_info() -> StreamingInfoResponse:
    livestream = int(os.environ.get("LIVESTREAM", "0"))
    return StreamingInfoResponse(
        enabled=APP_STATE.sim_ready and livestream > 0,
        type="webrtc",
        streaming_port=int(os.environ.get("STREAMING_PORT", "8211")),
        signaling_port=int(os.environ.get("SIGNALING_PORT", "49100")),
    )
@app.post("/scene/load", response_model=SceneLoadResponse)
async def scene_load(body: SceneLoadRequest) -> SceneLoadResponse:
    scene_dir = Path(body.scene_dir).expanduser().resolve()
    if not scene_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Scene directory not found: {scene_dir}")
    try:
        yaml_path = str(scene_dir / "magicsim_scene.yaml")
        scene_yaml = generate_magicsim_yaml(str(scene_dir), output_path=yaml_path)
        APP_STATE.scene_state = load_scene_state(str(scene_dir))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to prepare scene: {exc}") from exc
    obj_map = APP_STATE.scene_state.get("objects", {}) if isinstance(APP_STATE.scene_state, dict) else {}
    APP_STATE.loaded_scene = scene_dir.name
    APP_STATE.loaded_scene_dir = str(scene_dir)
    APP_STATE.scene_yaml_path = yaml_path
    sim_result = _submit_sim_command("scene_load", {"scene_dir": str(scene_dir), "yaml_path": yaml_path, "scene_yaml": scene_yaml}) if APP_STATE.sim_ready else {"success": False, "message": "Isaac Sim runtime not ready"}
    return SceneLoadResponse(success=True, scene_name=APP_STATE.loaded_scene, object_count=len(obj_map) if isinstance(obj_map, dict) else 0, loaded_via_sim=bool(sim_result.get("success")), yaml_path=yaml_path, message=str(sim_result.get("message", "")))
@app.get("/scene/info", response_model=SceneInfoResponse)
async def scene_info() -> SceneInfoResponse:
    if APP_STATE.sim_ready:
        try:
            from omni.isaac.core.utils.stage import get_current_stage  # lazy import
            stage = get_current_stage()
            if stage:
                objects: list[SceneObject] = []
                prim_count = 0
                for prim in stage.Traverse():
                    prim_count += 1
                    if len(objects) < 300:
                        objects.append(SceneObject(path=prim.GetPath().pathString, type=prim.GetTypeName() or "Unknown"))
                return SceneInfoResponse(loaded_scene=APP_STATE.loaded_scene, prim_count=prim_count, objects=objects, source="usd_stage")
        except Exception as exc:
            LOGGER.info("scene/info fallback to scene_state.json: %s", exc)
    obj_map = APP_STATE.scene_state.get("objects", {}) if isinstance(APP_STATE.scene_state, dict) else {}
    objects: list[SceneObject] = []
    if isinstance(obj_map, dict):
        for obj_id, obj in obj_map.items():
            if isinstance(obj, dict):
                objects.append(SceneObject(path=f"/{obj_id}", type=str(obj.get("object_type", "unknown")), name=str(obj.get("name", obj_id))))
    return SceneInfoResponse(loaded_scene=APP_STATE.loaded_scene, prim_count=len(objects), objects=objects, source="scene_state_json")
@app.post("/robot/spawn", response_model=RobotSpawnResponse)
async def robot_spawn(body: RobotSpawnRequest) -> RobotSpawnResponse:
    if not APP_STATE.sim_ready:
        return RobotSpawnResponse(success=False, robot_type=body.robot_type, position=body.position, orientation=body.orientation, message="Isaac Sim runtime not available")
    result = _submit_sim_command("robot_spawn", {"robot_type": body.robot_type, "position": body.position, "orientation": body.orientation})
    return RobotSpawnResponse(success=result.get("success", False), robot_type=body.robot_type, position=body.position, orientation=body.orientation, message=result.get("message", ""))
@app.get("/robot/state", response_model=RobotStateResponse)
async def robot_state() -> RobotStateResponse:
    if not APP_STATE.sim_ready:
        return RobotStateResponse(joint_positions=[], end_effector_pose=[0.0] * 7, gripper_state=0.0, sim_ready=False, message="Isaac Sim runtime not available")
    result = _submit_sim_command("robot_state", {})
    return RobotStateResponse(joint_positions=result.get("joint_positions", []), end_effector_pose=result.get("end_effector_pose", [0.0] * 7), gripper_state=result.get("gripper_state", 0.0), sim_ready=True, message=result.get("message", ""))
@app.post("/collect/start", response_model=CollectionStartResponse)
async def collect_start(body: CollectionStartRequest) -> CollectionStartResponse:
    if APP_STATE.collection_running:
        return CollectionStartResponse(success=False, running=True, skill=body.skill, target_count=len(body.target_objects), output_dir=body.output_dir, message="Collection already running")
    if not APP_STATE.sim_ready:
        return CollectionStartResponse(success=False, running=False, skill=body.skill, target_count=len(body.target_objects), output_dir=body.output_dir, message="Isaac Sim runtime not available")
    result = _submit_sim_command("collection_start", {"skill": body.skill, "target_objects": body.target_objects, "output_dir": body.output_dir})
    return CollectionStartResponse(success=result.get("success", False), running=APP_STATE.collection_running, skill=body.skill, target_count=len(body.target_objects), output_dir=body.output_dir, message=result.get("message", ""))
@app.get("/collect/status", response_model=CollectionStatusResponse)
async def collect_status() -> CollectionStatusResponse:
    return CollectionStatusResponse(running=APP_STATE.collection_running, progress=APP_STATE.collection_progress, collected_samples=APP_STATE.collection_samples, message="Collection status stub.")
@app.post("/sim/step", response_model=SimStepResponse)
async def sim_step(body: SimStepRequest) -> SimStepResponse:
    if not APP_STATE.sim_ready:
        return SimStepResponse(success=False, steps_requested=body.n, sim_ready=False, message="Isaac Sim runtime not available")
    result = _submit_sim_command("sim_step", {"n": body.n})
    return SimStepResponse(success=result.get("success", False), steps_requested=body.n, sim_ready=True, message=result.get("message", ""))
@app.post("/sim/reset", response_model=SimResetResponse)
async def sim_reset() -> SimResetResponse:
    if not APP_STATE.sim_ready:
        APP_STATE.collection_stop.set()
        APP_STATE.collection_running = False
        APP_STATE.collection_progress = 0.0
        APP_STATE.collection_samples = 0
        return SimResetResponse(success=True, sim_ready=False, message="State flags reset (no Isaac Sim runtime)")
    result = _submit_sim_command("sim_reset", {})
    return SimResetResponse(success=result.get("success", False), sim_ready=True, message=result.get("message", ""))
if __name__ == "__main__":
    import time
    import uvicorn
    # Start Isaac Sim in main thread first (needs main thread event loop).
    _start_isaac_sim()
    # Run uvicorn in a daemon thread so main thread stays free for Isaac Sim.
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=5900),
        daemon=True,
        name="uvicorn",
    )
    server_thread.start()
    LOGGER.info("Uvicorn started in background thread.")
    # Main thread: keep Isaac Sim alive (step its internal Kit loop).
    try:
        launcher = APP_STATE.launcher
        if launcher and hasattr(launcher, "is_running") and hasattr(launcher, "update"):
            LOGGER.info("Entering Isaac Sim main loop.")
            while True:
                if hasattr(launcher, "is_running") and not launcher.is_running():
                    LOGGER.warning("Isaac Sim stopped running, sleeping instead.")
                    break
                try:
                    launcher.update()
                except Exception as exc:
                    LOGGER.error("launcher.update() error: %s", exc)
                    time.sleep(1)
        else:
            LOGGER.info("No Isaac Sim launcher, keeping process alive with sleep loop.")
        # Fallback: sleep loop to keep the process alive for uvicorn.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        LOGGER.info("Shutting down...")
    finally:
        if APP_STATE.launcher and hasattr(APP_STATE.launcher, "close"):
            APP_STATE.launcher.close()
