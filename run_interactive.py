#!/usr/bin/env python3
"""Interactive Isaac Sim service — scene loading + robot placement + physics + data collection.

Two modes:
  - Viewer mode (default): renders scene, no physics stepping
  - Physics mode: full physics simulation (gravity, collision, joint control)

Toggle via Bridge API: POST /physics/play, POST /physics/pause

Ports (env vars):
  HEALTH_PORT  = 5900  (health check)
  BRIDGE_PORT  = 5800  (Bridge API — scene/robot/physics/collect control)
  WEBRTC_PORT  = 49100 (WebRTC signaling, injected via kit_args)
  KIT_API_PORT = 8011  (Kit API for NVCF streaming)
  MEDIA_PORT   = 47998 (WebRTC media UDP, fixed via fixedHostPort)
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time
import threading
import re
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlencode

# ── Suppress Isaac Sim deprecation warnings (5.1.0 transition) ──
import warnings
warnings.filterwarnings("ignore", message=".*has been deprecated in favor of.*")
warnings.filterwarnings("ignore", message=".*is deprecated.*")

# ── Env defaults ──────────────────────────────────────────────
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ.setdefault("PRIVACY_CONSENT", "Y")
os.environ.setdefault("LIVESTREAM", "2")  # NVCF streaming

HEALTH_PORT = int(os.environ.get("HEALTH_PORT", "5900"))
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "5800"))
WEBRTC_PORT = int(os.environ.get("WEBRTC_PORT", "49100"))
KIT_API_PORT = int(os.environ.get("KIT_API_PORT", "8011"))
MEDIA_PORT = int(os.environ.get("MEDIA_PORT", "47998"))
SIM_SESSION_NAME = (
    os.environ.get("SIM_SESSION_NAME")
    or os.environ.get("POD_NAME")
    or os.environ.get("HOSTNAME")
    or "interactive"
)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    token = token.strip("._-")
    return token or "session"


class _TeeTextStream:
    """Mirror text writes to multiple streams."""

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]
        self.encoding = getattr(self._streams[0], "encoding", "utf-8") if self._streams else "utf-8"

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return bool(self._streams and self._streams[0].isatty())
        except Exception:
            return False

    def fileno(self):
        for s in self._streams:
            try:
                return int(s.fileno())
            except Exception:
                continue
        try:
            return int(sys.__stderr__.fileno())
        except Exception:
            return 2


def _resolve_persistent_log_dir() -> Path | None:
    candidates: list[Path] = []
    env_path = os.environ.get("SIM_LOG_DIR", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("/data/embodied/logs/sim-service"),
            Path("/data/datasets/embodied/logs/sim-service"),
            Path("/data/logs/sim-service"),
            Path("/tmp/sim-service-logs"),
        ]
    )
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            continue
    return None


_RUNTIME_LOG_PATH: str | None = None
_RUNTIME_LOG_DIR: str | None = None
_runtime_log_file_handle = None


def _setup_persistent_stdio_log() -> str | None:
    global _RUNTIME_LOG_PATH, _RUNTIME_LOG_DIR, _runtime_log_file_handle
    enabled = os.environ.get("SIM_STDIO_LOG_TEE", "1").strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return None
    log_dir = _resolve_persistent_log_dir()
    if log_dir is None:
        return None
    _RUNTIME_LOG_DIR = str(log_dir)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    session = _safe_token(SIM_SESSION_NAME)
    fname = f"run_interactive_{session}_{ts}_{os.getpid()}.log"
    log_path = log_dir / fname
    try:
        fh = log_path.open("a", encoding="utf-8", buffering=1)
    except Exception:
        return None

    _runtime_log_file_handle = fh
    _RUNTIME_LOG_PATH = str(log_path)
    sys.stdout = _TeeTextStream(sys.__stdout__, fh)
    sys.stderr = _TeeTextStream(sys.__stderr__, fh)

    try:
        latest = log_dir / f"run_interactive_{session}_latest.log"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(log_path.name)
    except Exception:
        pass
    return _RUNTIME_LOG_PATH


_setup_persistent_stdio_log()
logging.basicConfig(
    level=getattr(logging, os.environ.get("SIM_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
LOG = logging.getLogger("interactive-bridge")
if _RUNTIME_LOG_PATH:
    LOG.info("Persistent runtime log enabled: %s", _RUNTIME_LOG_PATH)
elif _RUNTIME_LOG_DIR:
    LOG.warning("Persistent runtime log directory resolved but file unavailable: %s", _RUNTIME_LOG_DIR)
else:
    LOG.warning("Persistent runtime log disabled: no writable log directory")
# Default off: stale autosave restores were causing invalid world/camera states.
AUTOSAVE_ENABLED = os.environ.get("SIM_AUTOSAVE_ENABLED", "0").strip() != "0"
AUTOSAVE_DIR = os.path.join("/data", "sim_sessions", SIM_SESSION_NAME)
AUTOSAVE_STAGE_PATH = os.path.join(AUTOSAVE_DIR, "last_stage.usda")
AUTOSAVE_META_PATH = os.path.join(AUTOSAVE_DIR, "last_stage_meta.json")
EMBODIED_DATASETS_ROOT = os.environ.get("EMBODIED_DATASETS_ROOT", "/data/embodied/dataset/sim")
ROBOT_ASSET_ROOT = Path(os.environ.get("SIM_ROBOT_ASSET_ROOT", "/data/embodied/asset/robots"))
DEFAULT_ROBOT_TYPE = os.environ.get("SIM_DEFAULT_ROBOT_TYPE", "openarm_bimanual").strip() or "openarm_bimanual"
LEGACY_COLLECT_OUTPUT_DIRS = {
    "/data/collections/latest",
    "/data/embodied/datasets/sim_collect_latest",
}
_SCENE_AUTO_LIFT_RAW = os.environ.get("SIM_SCENE_AUTO_LIFT", "0").strip().lower()
SCENE_AUTO_LIFT_ENABLED = _SCENE_AUTO_LIFT_RAW in {"1", "true", "yes", "on"}
LOG.info(
    "Scene postprocess auto-lift after code_execute: %s (SIM_SCENE_AUTO_LIFT=%r)",
    "enabled" if SCENE_AUTO_LIFT_ENABLED else "disabled",
    _SCENE_AUTO_LIFT_RAW,
)

# ── Inject Kit args for WebRTC + Kit API ports BEFORE any Isaac imports ──
sys.argv.append(f"--kit_args=--/app/livestream/port={WEBRTC_PORT}")
sys.argv.append(f"--kit_args=--/exts/omni.services.transport.server.http/port={KIT_API_PORT}")
# Fix WebRTC media UDP port so NodePort can forward it
sys.argv.append(f"--kit_args=--/app/livestream/fixedHostPort={MEDIA_PORT}")
print(f"[interactive] WebRTC media UDP fixed to port {MEDIA_PORT}")

# Note: Isaac Sim 5.1.0 emits ~90 deprecation warnings from internal
# extension wrappers (omni.isaac.* → isaacsim.*). These are harmless and
# cannot be suppressed without modifying the container. Will disappear in 6.x.

# ── Start Isaac Sim via IsaacLab AppLauncher ──────────────────
print("[interactive] Starting Isaac Sim (headless + NVCF streaming)...")
parser = argparse.ArgumentParser(description="Interactive Isaac Sim")
parser.add_argument("--headless", action="store_true", default=True)

try:
    from omni.isaac.lab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print("[interactive] Isaac Sim started via IsaacLab AppLauncher")
except ImportError:
    # Fallback: try direct SimulationApp (Isaac Sim 5.x+)
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({
        "headless": True,
        "width": 1920,
        "height": 1080,
        "enable_livestream": True,
        "livestream_library": "omni.services.livestream.nvcf",
    })
    print("[interactive] Isaac Sim started via SimulationApp (fallback, livestream enabled)")

# ── Imports available only after Kit starts ───────────────────
import omni.usd
import omni.kit.app
from pxr import Gf, UsdGeom, Usd

# Isaac Sim version-compat imports (5.1.0 isaacsim.* or 4.5.0 omni.isaac.*)
# These must run after Kit extensions are loaded. On 5.1.0, isaacsim.core.*
# is only available after extension startup, so we defer with a lazy loader.
_ISAAC_API = None

def _init_isaac_compat():
    global _World, _add_ref, _create_prim, _Articulation, _ArticulationAction, _ISAAC_API
    if _ISAAC_API is not None:
        return
    try:
        from isaacsim.core.api import World as _World
        from isaacsim.core.utils.stage import add_reference_to_stage as _add_ref
        from isaacsim.core.utils.prims import create_prim as _create_prim
        from isaacsim.core.api.articulations import Articulation as _Articulation
        from isaacsim.core.utils.types import ArticulationAction as _ArticulationAction
        _ISAAC_API = "5.x"
    except ImportError:
        from omni.isaac.core import World as _World
        from omni.isaac.core.utils.stage import add_reference_to_stage as _add_ref
        from omni.isaac.core.utils.prims import create_prim as _create_prim
        from omni.isaac.core.articulations import Articulation as _Articulation
        from omni.isaac.core.utils.types import ArticulationAction as _ArticulationAction
        _ISAAC_API = "4.x"
    print(f"[interactive] Isaac API: {_ISAAC_API}")

# ── Create viewport for NVCF frame capture ────────────────────
try:
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    if vp is None:
        from omni.kit.widget.viewport import ViewportWidget
        print("[interactive] No active viewport — creating one...")
        _vp_widget = ViewportWidget(width=1920, height=1080)
        vp = get_active_viewport()
    if vp:
        vp.resolution = (1920, 1080)
        print(f"[interactive] Viewport ready: {vp}, resolution 1920x1080")
    else:
        print("[interactive] WARNING: no viewport — streaming may not work")
except Exception as exc:
    print(f"[interactive] WARNING: viewport setup failed: {exc}")

# ── Set streaming ports via carb settings BEFORE enabling extensions ──
# kit_args via sys.argv don't reliably reach Kit — set directly via carb API.
try:
    import carb.settings
    _settings = carb.settings.get_settings()
    _settings.set("/app/livestream/port", WEBRTC_PORT)
    _settings.set("/app/livestream/fixedHostPort", MEDIA_PORT)
    _settings.set("/exts/omni.services.transport.server.http/port", KIT_API_PORT)
    # Keep Isaac app alive when control-ui websocket disconnects.
    _settings.set("/app/livestream/nvcf/allowSessionResume", True)
    print(f"[interactive] Carb settings: livestream/port={WEBRTC_PORT}, "
          f"fixedHostPort={MEDIA_PORT}, http/port={KIT_API_PORT}")
except Exception as exc:
    print(f"[interactive] WARNING: carb settings: {exc}")

# ── Enable RTX viewport renderer (required for scene rendering in headless mode) ──
try:
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("omni.kit.viewport.rtx", True)
    print("[interactive] omni.kit.viewport.rtx enabled")
except Exception as exc:
    print(f"[interactive] WARNING: viewport.rtx: {exc}")

# ── Enable NVCF streaming extension ──────────────────────────
try:
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    # WebRTC core must be enabled before NVCF
    for ext_name in (
        "omni.kit.livestream.core",
        "omni.kit.livestream.webrtc",
        "omni.services.livestream.nvcf",
    ):
        try:
            ext_mgr.set_extension_enabled_immediate(ext_name, True)
        except Exception:
            pass
    print("[interactive] NVCF streaming extensions enabled")
except Exception as exc:
    print(f"[interactive] WARNING: NVCF extension: {exc}")

# Initialize Isaac API compat layer (must be after extension startup)
# Force a few app updates to ensure all extensions finish loading
for _ in range(5):
    simulation_app.update()
_init_isaac_compat()

# ── Detect actual Kit API port (may differ from KIT_API_PORT if taken) ──
_actual_kit_port = KIT_API_PORT
try:
    import carb.settings
    _p = carb.settings.get_settings().get("/exts/omni.services.transport.server.http/port")
    if _p and int(_p) > 0:
        _actual_kit_port = int(_p)
except Exception:
    pass
print(f"[interactive] Kit API port: requested={KIT_API_PORT}, actual={_actual_kit_port}")


def _sanitize_dataset_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip().lower())
    token = token.strip("._-")
    return token or "pick_place"


def _make_auto_collect_output_dir(skill: str) -> str:
    skill_token = _sanitize_dataset_token(skill)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = uuid.uuid4().hex[:6]
    return os.path.join(EMBODIED_DATASETS_ROOT, f"sim_{skill_token}_{timestamp}_{suffix}")


def _normalize_collect_output_dir(output_dir: str | None, skill: str) -> str:
    raw = "" if output_dir is None else str(output_dir).strip()
    if not raw or raw in LEGACY_COLLECT_OUTPUT_DIRS:
        try:
            os.makedirs(EMBODIED_DATASETS_ROOT, exist_ok=True)
        except Exception:
            pass
        return _make_auto_collect_output_dir(skill)
    return raw


def _create_world(*, add_ground_plane: bool):
    _init_isaac_compat()
    World = _World

    w = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0)
    if add_ground_plane:
        try:
            w.scene.add_default_ground_plane()
        except Exception as exc:
            print(f"[world] WARNING: add_default_ground_plane failed: {exc}")
    return w


def _recreate_world_for_open_stage(reason: str) -> bool:
    """Rebuild World after open_stage(), which invalidates prior World handles.

    World is a singleton (inherits SimulationContext). Calling World()
    returns the *same* instance, so we must manually clear the scene
    registry to drop expired prim wrappers left over from clear-scene.
    """
    global world, PHYSICS_RUNNING
    try:
        if world is not None:
            try:
                world.stop()
            except Exception:
                pass
            # World() is singleton — clear stale scene registry before "recreate"
            try:
                registry = getattr(world.scene, "_scene_registry", None)
                if registry is not None:
                    for attr_name in list(vars(registry)):
                        bucket = getattr(registry, attr_name, None)
                        if isinstance(bucket, dict):
                            bucket.clear()
                    print(f"[world] cleared stale scene registry for {reason}")
            except Exception as _reg_exc:
                print(f"[world] WARNING: failed to clear scene registry: {_reg_exc}")
        world = _create_world(add_ground_plane=False)
        PHYSICS_RUNNING = False
        if isinstance(globals().get("_state"), dict):
            _state["physics"] = False
        print(f"[world] recreated after {reason} (physics PAUSED)")
        return True
    except Exception as exc:
        world = None
        PHYSICS_RUNNING = False
        if isinstance(globals().get("_state"), dict):
            _state["physics"] = False
        print(f"[world] ERROR: recreate failed after {reason}: {exc}")
        return False


# ── World + physics (paused by default) ──────────────────────
try:
    world = _create_world(add_ground_plane=True)
    # Don't call world.play() — start in viewer mode
    PHYSICS_RUNNING = False
    print("[interactive] World created (physics PAUSED)")
except Exception as exc:
    world = None
    PHYSICS_RUNNING = False
    print(f"[interactive] WARNING: World creation failed: {exc}")

# ── Shared state ─────────────────────────────────────────────
_state = {
    "sim_ready": True,
    "physics": False,
    "scene": None,
    "robots": {},
    "active_robot_type": os.environ.get("SIM_ROBOT_TYPE", "").strip() or DEFAULT_ROBOT_TYPE,
    "step": 0,
    "collecting": False,
    "collect_progress": None,
    "estop_requested": False,
    "last_estop_reason": None,
    "last_estop_at": None,
    "autosave_enabled": AUTOSAVE_ENABLED,
    "autosave_stage_path": AUTOSAVE_STAGE_PATH if AUTOSAVE_ENABLED else None,
    "restored_from_autosave": False,
    "last_autosave_reason": None,
    "log_file": _RUNTIME_LOG_PATH,
    "log_dir": _RUNTIME_LOG_DIR,
}

# ── Thread-safe command queue for main-thread execution ──────
# Isaac Sim API is NOT thread-safe — must execute on main thread.
# Flask handlers push commands here; main loop executes and signals done.
import queue
import inspect
_cmd_queue = queue.Queue()
try:
    _CMD_TIMEOUT = float(os.environ.get("BRIDGE_CMD_TIMEOUT_SEC", "30"))
except Exception:
    _CMD_TIMEOUT = 30.0
try:
    _SCENE_LOAD_TIMEOUT = float(os.environ.get("BRIDGE_SCENE_LOAD_TIMEOUT_SEC", "300"))
except Exception:
    _SCENE_LOAD_TIMEOUT = 300.0
_estop_event = threading.Event()

# ── Health server (port 5900) ────────────────────────────────
from http.server import HTTPServer, BaseHTTPRequestHandler

class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({
                "status": "ok",
                "sim_ready": _state["sim_ready"],
                "physics": _state["physics"],
                "step": _state["step"],
                "scene": _state["scene"],
                "robots": _state["robots"],
                "collecting": _state["collecting"],
                "kit_api_port": _actual_kit_port,
                "version": "2.0.0-interactive",
            })
        elif self.path == "/streaming/info":
            body = json.dumps({
                "enabled": True,
                "type": "webrtc-nvcf",
                "streaming_port": WEBRTC_PORT,
                "kit_api_port": _actual_kit_port,
            })
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *a):
        pass

threading.Thread(
    target=lambda: HTTPServer(("0.0.0.0", HEALTH_PORT), _Health).serve_forever(),
    daemon=True, name="health",
).start()
print(f"[interactive] Health server on :{HEALTH_PORT}")

# ── Bridge API (port 5800) ───────────────────────────────────
from flask import Flask, request as flask_request, jsonify, send_file
import traceback

bridge = Flask(__name__)
bridge.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
USD_SUFFIXES = {".usd", ".usda", ".usdc"}
DOWNLOAD_CACHE_DIR = Path(
    os.environ.get("SIM_DOWNLOAD_CACHE_DIR", "/tmp/sim-service-downloads")
)


def _get_stage():
    ctx = omni.usd.get_context()
    return ctx.get_stage() if ctx else None


def _get_active_robot_type(preferred: str = "") -> str:
    if preferred:
        return str(preferred).strip()

    active = str(_state.get("active_robot_type", "") or "").strip()
    if active:
        return active

    robots = _state.get("robots") or {}
    for robot in robots.values():
        if isinstance(robot, dict):
            robot_type = str(robot.get("type", "") or "").strip()
            if robot_type:
                return robot_type

    return DEFAULT_ROBOT_TYPE


def _robot_asset_candidates(filename: str, robot_type: str = "", explicit_path: str = "", sibling_of: str = "") -> list[str]:
    filename = str(filename or "").strip()
    if not filename:
        return []

    resolved_robot_type = _get_active_robot_type(robot_type)
    candidates: list[str] = []

    explicit = str(explicit_path or "").strip()
    if explicit:
        candidates.append(str(Path(explicit).expanduser()))

    sibling = str(sibling_of or "").strip()
    if sibling:
        sibling_path = Path(sibling).expanduser()
        if sibling_path.name == filename:
            candidates.append(str(sibling_path))
        else:
            candidates.append(str(sibling_path.parent / filename))

    if resolved_robot_type:
        candidates.append(str(ROBOT_ASSET_ROOT / resolved_robot_type / filename))

    if resolved_robot_type == DEFAULT_ROBOT_TYPE:
        candidates.append(str(Path(__file__).resolve().parent / filename))

    return _unique_preserve_order(candidates)


def _load_robot_asset_yaml(filename: str, robot_type: str = "", explicit_path: str = "", sibling_of: str = ""):
    import yaml as _yaml

    candidates = _robot_asset_candidates(
        filename=filename,
        robot_type=robot_type,
        explicit_path=explicit_path,
        sibling_of=sibling_of,
    )
    load_errors = []
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            with open(candidate, encoding="utf-8") as f:
                return candidate, (_yaml.safe_load(f) or {}), candidates
        except Exception as exc:
            load_errors.append(f"{candidate}: {exc}")

    if load_errors:
        raise RuntimeError("; ".join(load_errors))

    return None, None, candidates


def _contains_unsafe_franka_physics_code(code: str) -> bool:
    """Block code patterns that corrupt Franka articulation physics hierarchy."""
    if not code:
        return False
    txt = str(code)
    # Direct API apply on Franka-related paths.
    direct_apply = re.search(
        r"UsdPhysics\.(?:RigidBodyAPI|CollisionAPI|MeshCollisionAPI)\.Apply\([^)]*(?:/World/Franka|FrankaRobot|/Franka\W)",
        txt,
        re.IGNORECASE | re.DOTALL,
    )
    if direct_apply:
        return True

    # Traverse + startswith('/World/Franka...') + physics apply pattern.
    traverse_apply = (
        re.search(r"stage\.Traverse\(\)", txt, re.IGNORECASE)
        and re.search(r"startswith\(\s*['\"](?:/World/Franka|/World/FrankaRobot)", txt, re.IGNORECASE)
        and re.search(r"UsdPhysics\.(?:RigidBodyAPI|CollisionAPI|MeshCollisionAPI)\.Apply", txt, re.IGNORECASE)
    )
    return bool(traverse_apply)


def _strip_runtime_helper_imports(code: str) -> str:
    """Remove invalid helper imports that LLM may generate for runtime-injected helpers."""
    if not code:
        return code
    helper_names = {"create_table", "create_franka", "create_mug", "create_apple", "create_ball"}
    stage_import_re = re.compile(r"^(\s*from\s+omni\.isaac\.core\.utils\.stage\s+import\s+)(.+)$")
    cleaned = []
    for line in code.splitlines():
        raw = line.rstrip()
        m = stage_import_re.match(raw)
        if m:
            prefix, imports = m.group(1), m.group(2)
            parts = [p.strip() for p in imports.strip().strip("()").split(",") if p.strip()]
            kept = []
            for part in parts:
                name = part.split(" as ")[0].strip()
                if name in helper_names:
                    continue
                kept.append(part)
            if kept:
                cleaned.append(prefix + ", ".join(kept))
            continue

        if "import" in raw and any(h in raw for h in helper_names) and "def " not in raw:
            tmp = re.sub(
                r"\b(create_table|create_franka|create_mug|create_apple|create_ball)\b(?:\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?\s*,?\s*",
                "",
                raw,
            )
            tmp = re.sub(r",\s*,", ", ", tmp).strip()
            if tmp.endswith("import"):
                continue
            if tmp:
                cleaned.append(tmp)
            continue

        cleaned.append(line)
    return "\n".join(cleaned)


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _prim_has_articulation_root(prim) -> bool:
    from pxr import UsdPhysics, Usd

    if prim is None or not prim.IsValid():
        return False
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return True
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            return True
    return False


def _runtime_create_table(
    stage=None,
    prim_path: str = "/World/Table",
    width: float = 1.2,
    depth: float = 0.8,
    height: float = 0.75,
    top_thickness: float = 0.04,
):
    """Runtime scene helper exposed to scene-chat code as create_table(...)."""
    from pxr import UsdGeom, UsdPhysics, Gf

    stage = stage or _get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available")

    existing = stage.GetPrimAtPath(prim_path)
    if existing and existing.IsValid():
        return prim_path

    UsdGeom.Xform.Define(stage, prim_path)

    top = UsdGeom.Cube.Define(stage, f"{prim_path}/Top")
    top.GetSizeAttr().Set(1.0)
    top_xf = UsdGeom.Xformable(top.GetPrim())
    top_xf.ClearXformOpOrder()
    top_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(height)))
    top_xf.AddScaleOp().Set(Gf.Vec3d(float(width), float(depth), float(top_thickness)))
    UsdPhysics.CollisionAPI.Apply(top.GetPrim())

    leg_inset = 0.05
    leg_w = 0.05
    lx = float(width) / 2.0 - leg_inset
    ly = float(depth) / 2.0 - leg_inset
    leg_h = float(height) - float(top_thickness) / 2.0
    for i, (px, py) in enumerate(((-lx, -ly), (lx, -ly), (-lx, ly), (lx, ly))):
        leg = UsdGeom.Cube.Define(stage, f"{prim_path}/Leg{i}")
        leg.GetSizeAttr().Set(1.0)
        lxf = UsdGeom.Xformable(leg.GetPrim())
        lxf.ClearXformOpOrder()
        lxf.AddTranslateOp().Set(Gf.Vec3d(float(px), float(py), leg_h / 2.0))
        lxf.AddScaleOp().Set(Gf.Vec3d(leg_w, leg_w, leg_h))
        UsdPhysics.CollisionAPI.Apply(leg.GetPrim())

    return prim_path


def _runtime_create_franka(
    stage=None,
    prim_path: str = "/World/FrankaRobot",
    position=(0.40, 0.00, 0.77),
    usd_path: str | None = None,
):
    """Runtime scene helper exposed to scene-chat code as create_franka(...)."""
    from pxr import UsdGeom, Gf
    add_reference_to_stage = _add_ref  # compat

    stage = stage or _get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available")

    candidates = []
    if usd_path:
        candidates.append(str(usd_path))

    try:
        try:
            from isaacsim.core.utils.nucleus import get_assets_root_path
        except ImportError:
            from omni.isaac.core.utils.nucleus import get_assets_root_path

        assets_root = get_assets_root_path()
        if assets_root:
            candidates.extend(
                [
                    assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                    assets_root + "/Isaac/Robots/Franka/franka.usd",
                ]
            )
    except Exception:
        pass

    candidates.extend(
        [
            "/home/user/magicphysics/MagicPhysics/packages/MagicSim/Assets/Robots/franka_umi.usd",
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Robots/Franka/franka.usd",
        ]
    )
    candidates = _unique_preserve_order(candidates)

    prim = stage.GetPrimAtPath(prim_path)
    if (not prim.IsValid()) or (not _prim_has_articulation_root(prim)):
        if prim and prim.IsValid():
            try:
                stage.RemovePrim(prim_path)
                for _ in range(3):
                    simulation_app.update()
            except Exception:
                pass

        loaded_ok = False
        last_error = ""
        for robot_usd in candidates:
            try:
                add_reference_to_stage(usd_path=robot_usd, prim_path=prim_path)
                for _ in range(10):
                    simulation_app.update()
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid() and _prim_has_articulation_root(prim):
                    loaded_ok = True
                    break
                stage.RemovePrim(prim_path)
                for _ in range(3):
                    simulation_app.update()
            except Exception as exc:
                last_error = str(exc)
                continue

        if not loaded_ok:
            raise RuntimeError(
                f"Franka prim invalid or non-articulation at {prim_path}. "
                f"last_error={last_error or 'none'}"
            )

    xform = UsdGeom.Xformable(prim)
    px, py, pz = position
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(px), float(py), float(pz)))
    return prim_path


def _runtime_create_mug(
    stage=None,
    prim_path: str = "/World/Mug",
    position=(0.30, 0.00, 0.823),
    variant: str = "C1",
    usd_path: str | None = None,
):
    """Runtime scene helper exposed to scene-chat code as create_mug(...).

    Current policy: lock to SM_Mug_C1 only, so grasp annotations stay consistent.
    """
    from pxr import UsdGeom, UsdPhysics, Gf
    add_reference_to_stage = _add_ref  # compat

    stage = stage or _get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available")

    requested_variant = str(variant or "C1").strip().upper()
    if requested_variant and requested_variant != "C1":
        print(f"[scene] create_mug variant '{requested_variant}' ignored; forcing C1")
    if usd_path:
        print(f"[scene] create_mug usd_path override ignored; forcing C1 asset: {usd_path}")

    c1_rel = "Isaac/Props/Mugs/SM_Mug_C1.usd"
    c1_http = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.1/Isaac/Props/Mugs/SM_Mug_C1.usd"
    )

    candidates = []

    try:
        try:
            from isaacsim.core.utils.nucleus import get_assets_root_path
        except ImportError:
            from omni.isaac.core.utils.nucleus import get_assets_root_path

        assets_root = get_assets_root_path()
        if assets_root:
            candidates.append(f"{assets_root}/{c1_rel}")
    except Exception:
        pass

    candidates.append(c1_http)
    candidates = _unique_preserve_order(candidates)
    px, py, pz = position

    existing = stage.GetPrimAtPath(prim_path)
    if existing and existing.IsValid():
        try:
            stage.RemovePrim(prim_path)
            for _ in range(2):
                simulation_app.update()
        except Exception:
            pass

    loaded_ok = False
    last_error = ""
    for mug_usd in candidates:
        try:
            add_reference_to_stage(usd_path=mug_usd, prim_path=prim_path)
            # Keep update budget small before we move the prim to target pose,
            # otherwise users see a brief flash at the authored/default location.
            for _ in range(2):
                simulation_app.update()
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                try:
                    xf = UsdGeom.Xformable(prim)
                    xf.ClearXformOpOrder()
                    xf.AddTranslateOp().Set(Gf.Vec3d(float(px), float(py), float(pz)))
                    xf.AddScaleOp().Set(Gf.Vec3d(0.01, 0.01, 0.01))
                except Exception:
                    pass
                for _ in range(2):
                    simulation_app.update()
                loaded_ok = True
                break
            stage.RemovePrim(prim_path)
        except Exception as exc:
            last_error = str(exc)
            continue

    if not loaded_ok:
        raise RuntimeError(
            f"Mug prim invalid at {prim_path}. last_error={last_error or 'none'}"
        )

    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(px), float(py), float(pz)))
    xf.AddScaleOp().Set(Gf.Vec3d(0.01, 0.01, 0.01))

    # Bind mug grasp annotation explicitly so collector resolves the intended file first.
    try:
        ann_path = Path(__file__).resolve().parent / "grasp_poses" / "mug_grasp_pose.json"
        if ann_path.exists():
            ann_abs = str(ann_path.resolve())
            prim.SetCustomDataByKey("grasp_pose_path", ann_abs)
            prim.SetCustomDataByKey("annotation_path", ann_abs)
    except Exception:
        pass

    try:
        UsdPhysics.RigidBodyAPI.Apply(prim)
    except Exception:
        pass
    for p in stage.Traverse():
        path = str(p.GetPath())
        if not path.startswith(prim_path):
            continue
        try:
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p)
                UsdPhysics.MeshCollisionAPI.Apply(p)
        except Exception:
            continue

    return prim_path


def _runtime_create_apple(
    stage=None,
    prim_path: str = "/World/Apple",
    position=(0.30, 0.00, 0.79),
    usd_path: str | None = None,
):
    """Runtime scene helper exposed to scene-chat code as create_apple(...)."""
    from pxr import UsdGeom, UsdPhysics, Gf
    add_reference_to_stage = _add_ref  # compat

    stage = stage or _get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available")

    candidates = []
    if usd_path:
        candidates.append(str(usd_path))
    candidates.extend(
        [
            "/sim-service/assets/OmniObject3D/apple_001/Object_rigid.usd",
            "/home/magics/rlinf-workspace/assets/OmniObject3D/Selected_Objs/apple/apple_001/Object_rigid.usd",
            "/data/assets/OmniObject3D/Selected_Objs/apple/apple_001/Object_rigid.usd",
        ]
    )
    candidates = _unique_preserve_order(candidates)
    px, py, pz = position

    existing = stage.GetPrimAtPath(prim_path)
    if existing and existing.IsValid():
        try:
            stage.RemovePrim(prim_path)
            for _ in range(2):
                simulation_app.update()
        except Exception:
            pass

    loaded_ok = False
    last_error = ""
    for apple_usd in candidates:
        apple_path = Path(str(apple_usd)).expanduser()
        if (not str(apple_usd).startswith(("omniverse://", "http://", "https://"))) and (not apple_path.exists()):
            continue
        try:
            add_reference_to_stage(usd_path=str(apple_usd), prim_path=prim_path)
            # Move object early to avoid visible teleport from default pose.
            for _ in range(2):
                simulation_app.update()
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                try:
                    xf = UsdGeom.Xformable(prim)
                    xf.ClearXformOpOrder()
                    xf.AddTranslateOp().Set(Gf.Vec3d(float(px), float(py), float(pz)))
                except Exception:
                    pass
                for _ in range(2):
                    simulation_app.update()
                loaded_ok = True
                break
            stage.RemovePrim(prim_path)
        except Exception as exc:
            last_error = str(exc)
            continue

    if not loaded_ok:
        raise RuntimeError(
            f"Apple prim invalid at {prim_path}. last_error={last_error or 'none'}"
        )

    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(px), float(py), float(pz)))

    try:
        ann_candidates = [
            Path(__file__).resolve().parent / "grasp_poses" / "apple_001_grasp_pose.json",
            Path("/sim-service/grasp_poses/apple_001_grasp_pose.json"),
            Path("/code/grasp_poses/apple_001_grasp_pose.json"),
        ]
        for ann_path in ann_candidates:
            if ann_path.exists():
                ann_abs = str(ann_path.resolve())
                prim.SetCustomDataByKey("grasp_pose_path", ann_abs)
                prim.SetCustomDataByKey("annotation_path", ann_abs)
                break
    except Exception:
        pass

    try:
        UsdPhysics.RigidBodyAPI.Apply(prim)
    except Exception:
        pass
    for p in stage.Traverse():
        path = str(p.GetPath())
        if not path.startswith(prim_path):
            continue
        try:
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p)
                UsdPhysics.MeshCollisionAPI.Apply(p)
        except Exception:
            continue

    return prim_path


def _runtime_create_ball(
    stage=None,
    prim_path: str = "/World/Ball",
    position=(0.30, 0.00, 0.79),
    usd_path: str | None = None,
):
    """Runtime scene helper exposed to scene-chat code as create_ball(...)."""
    from pxr import UsdGeom, UsdPhysics, Gf
    add_reference_to_stage = _add_ref  # compat

    stage = stage or _get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available")

    candidates = []
    if usd_path:
        candidates.append(str(usd_path))
    candidates.extend(
        [
            "/sim-service/assets/OmniObject3D/ball_012/Object_rigid.usd",
            "/home/magics/rlinf-workspace/assets/OmniObject3D/Selected_Objs/ball/ball_012/Object_rigid.usd",
            "/data/assets/OmniObject3D/Selected_Objs/ball/ball_012/Object_rigid.usd",
        ]
    )
    candidates = _unique_preserve_order(candidates)
    px, py, pz = position

    existing = stage.GetPrimAtPath(prim_path)
    if existing and existing.IsValid():
        try:
            stage.RemovePrim(prim_path)
            for _ in range(2):
                simulation_app.update()
        except Exception:
            pass

    loaded_ok = False
    last_error = ""
    for ball_usd in candidates:
        ball_path = Path(str(ball_usd)).expanduser()
        if (not str(ball_usd).startswith(("omniverse://", "http://", "https://"))) and (not ball_path.exists()):
            continue
        try:
            add_reference_to_stage(usd_path=str(ball_usd), prim_path=prim_path)
            # Move object early to avoid visible teleport from default pose.
            for _ in range(2):
                simulation_app.update()
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                try:
                    xf = UsdGeom.Xformable(prim)
                    xf.ClearXformOpOrder()
                    xf.AddTranslateOp().Set(Gf.Vec3d(float(px), float(py), float(pz)))
                except Exception:
                    pass
                for _ in range(2):
                    simulation_app.update()
                loaded_ok = True
                break
            stage.RemovePrim(prim_path)
        except Exception as exc:
            last_error = str(exc)
            continue

    if not loaded_ok:
        raise RuntimeError(
            f"Ball prim invalid at {prim_path}. last_error={last_error or 'none'}"
        )

    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(px), float(py), float(pz)))

    try:
        ann_candidates = [
            Path(__file__).resolve().parent / "grasp_poses" / "ball_012_grasp_pose.json",
            Path("/sim-service/grasp_poses/ball_012_grasp_pose.json"),
            Path("/code/grasp_poses/ball_012_grasp_pose.json"),
        ]
        for ann_path in ann_candidates:
            if ann_path.exists():
                ann_abs = str(ann_path.resolve())
                prim.SetCustomDataByKey("grasp_pose_path", ann_abs)
                prim.SetCustomDataByKey("annotation_path", ann_abs)
                break
    except Exception:
        pass

    try:
        UsdPhysics.RigidBodyAPI.Apply(prim)
    except Exception:
        pass
    for p in stage.Traverse():
        path = str(p.GetPath())
        if not path.startswith(prim_path):
            continue
        try:
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p)
                UsdPhysics.MeshCollisionAPI.Apply(p)
        except Exception:
            continue

    return prim_path


def _code_may_mutate_stage(code: str) -> bool:
    """Best-effort detection for code that mutates USD stage topology/transforms."""
    txt = str(code or "")
    if not txt:
        return False
    mutation_patterns = [
        r"\bRemovePrim\(",
        r"\bDefine\(",
        r"\badd_reference_to_stage\(",
        r"\bcreate_prim\(",
        r"\bAddTranslateOp\(",
        r"\bAddRotate[A-Za-z]*Op\(",
        r"\bAddScaleOp\(",
        r"\.Set\(",
        r"RigidBodyAPI\.Apply\(",
        r"CollisionAPI\.Apply\(",
        r"MeshCollisionAPI\.Apply\(",
        r"\bcreate_table\(",
        r"\bcreate_franka\(",
        r"\bcreate_mug\(",
        r"\bcreate_apple\(",
        r"\bcreate_ball\(",
    ]
    return any(re.search(p, txt) for p in mutation_patterns)


def _sanitize_franka_root_rigidbody(stage) -> list[str]:
    """Remove accidental RigidBodyAPI on Franka root prims before physics play."""
    from pxr import UsdPhysics

    cleaned = []
    for prim in stage.Traverse():
        if not prim or not prim.IsValid():
            continue
        path = str(prim.GetPath())
        low = path.lower()
        if not (low.endswith("/franka") or low.endswith("/frankarobot")):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            try:
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                cleaned.append(path)
            except Exception as exc:
                print(f"[physics] WARN remove RigidBodyAPI failed on {path}: {exc}")
    return cleaned


def _lift_objects_above_table_if_needed(
    stage,
    clearance: float = 0.004,
    max_delta_z: float = 0.12,
) -> list[dict]:
    """Lift root objects that interpenetrate table top so they stay visible and graspable."""
    if stage is None:
        return []

    table_top = stage.GetPrimAtPath("/World/Table/Top")
    world = stage.GetPrimAtPath("/World")
    if not table_top.IsValid() or not world.IsValid():
        return []

    try:
        bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        table_range = bbox_cache.ComputeWorldBound(table_top).GetRange()
        table_min = table_range.GetMin()
        table_max = table_range.GetMax()
        table_top_z = float(table_max[2])
    except Exception:
        return []

    moved = []
    for prim in world.GetChildren():
        if not prim or not prim.IsValid():
            continue
        name = prim.GetName() or ""
        low = name.lower()
        if low in {"table", "frankarobot", "defaultgroundplane"}:
            continue
        if prim.IsA(UsdGeom.Camera):
            continue

        try:
            bbox_cache.Clear()
            pr = bbox_cache.ComputeWorldBound(prim).GetRange()
            pmin = pr.GetMin()
            pmax = pr.GetMax()
        except Exception:
            continue

        # Only auto-lift objects that are over table footprint.
        cx = float((pmin[0] + pmax[0]) * 0.5)
        cy = float((pmin[1] + pmax[1]) * 0.5)
        if (
            cx < float(table_min[0]) - 0.02
            or cx > float(table_max[0]) + 0.02
            or cy < float(table_min[1]) - 0.02
            or cy > float(table_max[1]) + 0.02
        ):
            continue

        min_z = float(pmin[2])
        target_min_z = table_top_z + float(clearance)
        if min_z >= target_min_z:
            continue

        delta_z = target_min_z - min_z
        # Guard against bad bounds (common on some referenced assets) that can
        # otherwise launch objects far above the table.
        if delta_z > float(max_delta_z):
            print(
                f"[scene] skip auto-lift for {prim.GetPath()}: "
                f"suspicious delta_z={delta_z:.4f} (> {max_delta_z:.4f})"
            )
            continue
        xf = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            current = Gf.Vec3d(0.0, 0.0, 0.0)
        else:
            try:
                t = translate_op.Get()
                current = Gf.Vec3d(float(t[0]), float(t[1]), float(t[2]))
            except Exception:
                current = Gf.Vec3d(0.0, 0.0, 0.0)

        translate_op.Set(Gf.Vec3d(current[0], current[1], current[2] + delta_z))
        moved.append({"path": str(prim.GetPath()), "delta_z": round(delta_z, 4)})

    return moved


def _save_autosave_stage(reason: str) -> bool:
    """Persist the current USD stage so a restarted pod can restore it."""
    if not AUTOSAVE_ENABLED:
        return False
    stage = _get_stage()
    if not stage:
        return False

    try:
        os.makedirs(AUTOSAVE_DIR, exist_ok=True)
        root_layer = stage.GetRootLayer()
        if root_layer is None:
            return False
        ok = bool(root_layer.Export(AUTOSAVE_STAGE_PATH))
        if not ok:
            print(f"[autosave] export failed: {AUTOSAVE_STAGE_PATH}")
            return False
        meta = {
            "saved_at_unix": time.time(),
            "reason": reason,
            "scene": _state.get("scene"),
            "session": SIM_SESSION_NAME,
        }
        with open(AUTOSAVE_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
        _state["last_autosave_reason"] = reason
        print(f"[autosave] saved ({reason}) -> {AUTOSAVE_STAGE_PATH}")
        return True
    except Exception as exc:
        print(f"[autosave] save failed ({reason}): {exc}")
        return False


def _restore_autosave_stage() -> bool:
    """Restore previously autosaved stage during startup."""
    if not AUTOSAVE_ENABLED:
        return False
    if not os.path.exists(AUTOSAVE_STAGE_PATH):
        return False

    try:
        ctx = omni.usd.get_context()
        if not ctx:
            return False
        print(f"[autosave] restoring stage from: {AUTOSAVE_STAGE_PATH}")
        ok = bool(ctx.open_stage(AUTOSAVE_STAGE_PATH))
        if not ok:
            print(f"[autosave] restore failed: open_stage returned false")
            return False
        for _ in range(300):
            simulation_app.update()
            if ctx.get_stage_state() == omni.usd.StageState.OPENED:
                break
            time.sleep(0.01)
        _recreate_world_for_open_stage("autosave_restore")
        _state["scene"] = AUTOSAVE_STAGE_PATH
        _state["robots"] = {}
        _state["physics"] = False
        _state["restored_from_autosave"] = True
        print(f"[autosave] restore done: {AUTOSAVE_STAGE_PATH}")
        return True
    except Exception as exc:
        print(f"[autosave] restore failed: {exc}")
        return False


@bridge.route("/health", methods=["GET"])
def bridge_health():
    return jsonify({
        "status": "ok",
        "sim_ready": _state["sim_ready"],
        "physics": _state["physics"],
        "scene": _state["scene"],
        "robots": _state["robots"],
        "step": _state["step"],
        "collecting": _state["collecting"],
        "autosave_enabled": _state["autosave_enabled"],
        "autosave_stage_path": _state["autosave_stage_path"],
        "restored_from_autosave": _state["restored_from_autosave"],
        "log_file": _state.get("log_file"),
        "log_dir": _state.get("log_dir"),
    })


# ── Scene endpoints ──────────────────────────────────────────

@bridge.route("/scene/load", methods=["POST"])
def scene_load():
    data = flask_request.get_json(silent=True) or {}
    usd_path = data.get("usd_path")
    if not usd_path:
        return jsonify({"error": "usd_path required"}), 400
    return _enqueue_cmd(
        "scene_load",
        usd_path=usd_path,
        timeout_override=max(_CMD_TIMEOUT, _SCENE_LOAD_TIMEOUT),
    )


@bridge.route("/scene/save", methods=["POST"])
def scene_save():
    data = flask_request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)[:60].strip()
    ts = int(time.time())
    output_dir = os.environ.get("SIM_SCENE_LIBRARY", "/data/embodied/scene/library")
    output_path = os.path.join(output_dir, f"{safe_name}_{ts}.usda")
    return _enqueue_cmd("scene_save", output_path=output_path, output_dir=output_dir)


# ── SceneSmith → USD export (main-thread, uses asset_converter) ──


def _strip_mdl_shaders(usd_path):
    """Replace MDL shaders with UsdPreviewSurface in a converted USD.

    asset_converter outputs MDL materials (gltf/pbr.mdl) which cause RTX
    shader compilation overload on 2080 Ti, freezing NVCF streaming.
    This rewrites each MDL Shader to UsdPreviewSurface, preserving the
    diffuse color and texture bindings.
    """
    try:
        from pxr import Usd, UsdShade, Sdf
        stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            return

        changed = False
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Shader":
                continue
            shader = UsdShade.Shader(prim)
            impl_src = shader.GetImplementationSource()
            if impl_src != "sourceAsset":
                continue
            mdl_asset = shader.GetSourceAsset("mdl")
            if not mdl_asset or "pbr.mdl" not in str(mdl_asset.resolvedPath or mdl_asset.path):
                continue

            # Read existing texture/color inputs before overwriting
            diff_tex = None
            diff_color = None
            for inp in shader.GetInputs():
                name = inp.GetBaseName()
                if name == "diffuse_texture":
                    diff_tex = inp.Get()
                elif name == "diffuse_color_constant":
                    diff_color = inp.Get()

            # Overwrite to UsdPreviewSurface
            shader.SetShaderId("UsdPreviewSurface")
            # Clear MDL source asset attrs
            for attr_name in ("info:mdl:sourceAsset", "info:mdl:sourceAsset:subIdentifier"):
                attr = prim.GetAttribute(attr_name)
                if attr and attr.IsValid():
                    prim.RemoveProperty(attr_name)

            # Set basic PBR params
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            if diff_color:
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diff_color)
            else:
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.7, 0.7, 0.7))

            changed = True

        if changed:
            stage.GetRootLayer().Save()
            print(f"[export] Stripped MDL shaders in {Path(usd_path).name}")
    except Exception as exc:
        print(f"[export] WARNING: MDL strip failed for {usd_path}: {exc}")


def _handle_export_scene(scene_dir, output_name, room_filter=None):
    """Export SceneSmith scene to USD. Uses omni.kit.asset_converter.

    Args:
        scene_dir: SceneSmith output directory
        output_name: name for output USD
        room_filter: if set, only export this room (e.g. "living_room")

    Handles multi-room house_state: objects + room walls + doors/windows + ceiling lights.
    Runs on Flask thread — converter + pxr USD write are offline operations.
    Returns dict with usd_path on success.
    """
    import asyncio
    import omni.kit.asset_converter as ac
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Vt

    scene_dir = str(scene_dir)
    state_path, is_house = _find_scenesmith_scene_state(scene_dir)
    if not state_path:
        raise FileNotFoundError(f"No scene state found in {scene_dir}")

    with open(state_path) as f:
        scene_state = json.load(f)

    # ── Parse objects + room info ────────────────────────────
    all_objects = {}     # oid → obj dict (with _room_dir, _room_offset)
    ceiling_lights = []
    room_walls = []      # list of wall dicts with global coords
    glass_panes = []     # list of window glass pane dicts
    placed_rooms = {}    # room_name → (ox, oy)

    if is_house and "rooms" in scene_state:
        scene_root = str(Path(state_path).parent.parent)
        layout = scene_state.get("layout", {})

        # Room offsets: placed_rooms[i].position is the room's bottom-left corner
        # in global coords. room_geometries walls are relative to room center.
        # So offset = position + (width/2, depth/2) to get the center.
        layout_rooms_list = layout.get("rooms", [])
        placed_rooms_list = layout.get("placed_rooms", [])
        for i, pr in enumerate(placed_rooms_list):
            rid = pr.get("room_id", "")
            if not rid and i < len(layout_rooms_list):
                rid = layout_rooms_list[i].get("id", f"room_{i}")
            pos = pr.get("position", [0, 0])
            pw = float(pr.get("width", 0))
            pd = float(pr.get("depth", 0))
            # Center = bottom-left + half size
            cx = float(pos[0]) + pw / 2.0
            cy = float(pos[1]) + pd / 2.0
            placed_rooms[rid] = (cx, cy)
            print(f"[export_scene] room '{rid}' pos=({pos[0]},{pos[1]}) size={pw}x{pd} center=({cx},{cy})")

        # Build wall opening map from placed_rooms
        # Key: (room_id, direction) → list of openings
        _wall_openings = {}
        for pr in placed_rooms_list:
            prid = pr.get("room_id", "")
            for pw in pr.get("walls", []):
                direction = pw.get("direction", "")
                openings = pw.get("openings", [])
                if openings:
                    _wall_openings[(prid, direction)] = openings

        # Room geometries → walls (optionally filtered)
        for rname, rg in layout.get("room_geometries", {}).items():
            if room_filter and rname != room_filter:
                continue
            ox, oy = placed_rooms.get(rname, (0, 0))
            for wall in rg.get("walls", []):
                wid = wall.get("object_id", "wall")
                t = wall.get("transform", {}).get("translation", [0, 0, 0])
                bmin = wall.get("bbox_min", [-0.5, -0.02, -0.5])
                bmax = wall.get("bbox_max", [0.5, 0.02, 0.5])

                # Check openings for this wall
                direction = wid.replace("_wall", "")  # "north_wall" → "north"
                openings = _wall_openings.get((rname, direction), [])

                # Skip wall if an "open" opening covers the full wall length
                # (wall extent along its primary axis)
                wall_extent = max(
                    abs(bmax[0] - bmin[0]),
                    abs(bmax[1] - bmin[1]),
                )
                has_full_open = any(
                    o.get("opening_type") == "open"
                    and float(o.get("width", 0)) >= wall_extent * 0.9
                    for o in openings
                )
                if has_full_open:
                    print(f"[export_scene] skip open wall {rname}/{wid}")
                    continue

                # Determine wall orientation: north/south walls extend along X,
                # east/west walls extend along Y
                is_ns = "north" in wid or "south" in wid  # extends along X
                wall_pos = (float(t[0]) + ox, float(t[1]) + oy, float(t[2]))

                # All openings that create gaps in the wall
                real_openings = []
                full_wall_h = float(bmax[2] - bmin[2]) * 2  # full height
                for o in openings:
                    otype = o.get("opening_type", "")
                    if otype in ("door", "window"):
                        real_openings.append(o)
                    elif otype == "open":
                        # Open = full-height gap
                        real_openings.append({
                            **o,
                            "height": full_wall_h,
                            "sill_height": 0.0,
                        })

                if not real_openings:
                    # Solid wall, no openings
                    room_walls.append({
                        "id": f"{rname}_{wid}",
                        "pos": wall_pos,
                        "bbox_min": bmin, "bbox_max": bmax,
                    })
                else:
                    # Split wall around openings
                    # Wall local extent along primary axis
                    if is_ns:
                        wall_min, wall_max = float(bmin[0]), float(bmax[0])
                        wall_thick_min, wall_thick_max = float(bmin[1]), float(bmax[1])
                    else:
                        wall_min, wall_max = float(bmin[1]), float(bmax[1])
                        wall_thick_min, wall_thick_max = float(bmin[0]), float(bmax[0])
                    wall_len = wall_max - wall_min
                    wall_z_min, wall_z_max = float(bmin[2]), float(bmax[2])
                    wall_height = wall_z_max - wall_z_min

                    # Sort openings by position along wall
                    sorted_openings = sorted(
                        real_openings,
                        key=lambda o: float(o.get("position_along_wall", 0))
                    )

                    # Build wall segments
                    cursor = wall_min
                    seg_idx = 0
                    for opening in sorted_openings:
                        o_pos = float(opening.get("position_along_wall", 0))
                        o_width = float(opening.get("width", 0.9))
                        o_height = float(opening.get("height", 2.1))
                        o_sill = float(opening.get("sill_height", 0))
                        # Opening center in local wall coords
                        o_start = wall_min + o_pos
                        o_end = o_start + o_width

                        # Segment before opening
                        if o_start > cursor + 0.01:
                            seg_min = cursor
                            seg_max = o_start
                            if is_ns:
                                sb = [seg_min, wall_thick_min, wall_z_min]
                                se = [seg_max, wall_thick_max, wall_z_max]
                            else:
                                sb = [wall_thick_min, seg_min, wall_z_min]
                                se = [wall_thick_max, seg_max, wall_z_max]
                            room_walls.append({
                                "id": f"{rname}_{wid}_s{seg_idx}",
                                "pos": wall_pos, "bbox_min": sb, "bbox_max": se,
                            })
                            seg_idx += 1

                        # Segment above opening (lintel)
                        if o_sill + o_height < wall_height - 0.01:
                            lintel_z_min = wall_z_min + o_sill + o_height
                            if is_ns:
                                sb = [o_start, wall_thick_min, lintel_z_min]
                                se = [o_end, wall_thick_max, wall_z_max]
                            else:
                                sb = [wall_thick_min, o_start, lintel_z_min]
                                se = [wall_thick_max, o_end, wall_z_max]
                            room_walls.append({
                                "id": f"{rname}_{wid}_above{seg_idx}",
                                "pos": wall_pos, "bbox_min": sb, "bbox_max": se,
                            })
                            seg_idx += 1

                        # Segment below opening (sill, for windows)
                        if o_sill > 0.01:
                            sill_z_max = wall_z_min + o_sill
                            if is_ns:
                                sb = [o_start, wall_thick_min, wall_z_min]
                                se = [o_end, wall_thick_max, sill_z_max]
                            else:
                                sb = [wall_thick_min, o_start, wall_z_min]
                                se = [wall_thick_max, o_end, sill_z_max]
                            room_walls.append({
                                "id": f"{rname}_{wid}_sill{seg_idx}",
                                "pos": wall_pos, "bbox_min": sb, "bbox_max": se,
                            })
                            seg_idx += 1

                        # Glass pane for windows
                        if opening.get("opening_type") == "window" and o_sill > 0.01:
                            glass_z_min = wall_z_min + o_sill
                            glass_z_max = wall_z_min + o_sill + o_height
                            thick_center = (wall_thick_min + wall_thick_max) / 2
                            if is_ns:
                                gb = [o_start, thick_center - 0.005, glass_z_min]
                                ge = [o_end, thick_center + 0.005, glass_z_max]
                            else:
                                gb = [thick_center - 0.005, o_start, glass_z_min]
                                ge = [thick_center + 0.005, o_end, glass_z_max]
                            glass_panes.append({
                                "id": f"{rname}_{wid}_glass{seg_idx}",
                                "pos": wall_pos, "bbox_min": gb, "bbox_max": ge,
                            })

                        cursor = o_end

                    # Final segment after last opening
                    if cursor < wall_max - 0.01:
                        if is_ns:
                            sb = [cursor, wall_thick_min, wall_z_min]
                            se = [wall_max, wall_thick_max, wall_z_max]
                        else:
                            sb = [wall_thick_min, cursor, wall_z_min]
                            se = [wall_thick_max, wall_max, wall_z_max]
                        room_walls.append({
                            "id": f"{rname}_{wid}_s{seg_idx}",
                            "pos": wall_pos, "bbox_min": sb, "bbox_max": se,
                        })
            # Floor — geometry.width is Y extent, geometry.length is X extent
            # Use placed_rooms width (X) and depth (Y) from the matching room
            floor_data = rg.get("floor", {})
            if floor_data:
                pr_match = next(
                    (pr for pr in placed_rooms_list if pr.get("room_id") == rname),
                    None,
                )
                if pr_match:
                    fx = float(pr_match.get("width", rg.get("length", 5.0)))
                    fy = float(pr_match.get("depth", rg.get("width", 5.0)))
                else:
                    fx = float(rg.get("length", 5.0))
                    fy = float(rg.get("width", 5.0))
                room_walls.append({
                    "id": f"{rname}_floor",
                    "pos": (ox, oy, 0.0),
                    "bbox_min": [-fx/2, -fy/2, 0.0],
                    "bbox_max": [fx/2, fy/2, 0.01],
                    "is_floor": True,
                })

        # Objects from rooms (optionally filtered)
        for room_name, room_data in scene_state.get("rooms", {}).items():
            if not isinstance(room_data, dict):
                continue
            if room_filter and room_name != room_filter:
                continue
            room_objs = room_data.get("objects", {})
            if not isinstance(room_objs, dict):
                continue
            ox, oy = placed_rooms.get(room_name, (0, 0))
            room_dir = os.path.join(scene_root, f"room_{room_name}")
            if not os.path.isdir(room_dir):
                for d in Path(scene_root).iterdir():
                    if d.is_dir() and room_name in d.name:
                        room_dir = str(d)
                        break
            for oid, obj in room_objs.items():
                obj["_room_dir"] = room_dir
                obj["_room_offset"] = (ox, oy)
                all_objects[oid] = obj
                desc = obj.get("description", "").lower()
                if any(w in desc for w in ["light", "lamp", "pendant", "chandelier", "flush"]):
                    t = obj.get("transform", {}).get("translation", [0, 0, 2.5])
                    ceiling_lights.append({
                        "id": oid,
                        "pos": (float(t[0]) + ox, float(t[1]) + oy, float(t[2])),
                    })
        print(f"[export_scene] house: {len(all_objects)} objects, {len(placed_rooms)} rooms, "
              f"{len(room_walls)} walls, {len(ceiling_lights)} lights")
    else:
        # Single room
        room_dir = str(Path(state_path).parent.parent.parent)
        objects_raw = scene_state.get("objects", {})
        if isinstance(objects_raw, list):
            objects_raw = {o.get("object_id", f"obj_{i}"): o for i, o in enumerate(objects_raw)}
        for oid, obj in objects_raw.items():
            obj["_room_dir"] = room_dir
            obj["_room_offset"] = (0, 0)
            all_objects[oid] = obj
        rg = scene_state.get("room_geometry", {})
        if rg:
            for wall in rg.get("walls", []):
                t = wall.get("transform", {}).get("translation", [0, 0, 0])
                room_walls.append({
                    "id": wall.get("object_id", "wall"),
                    "pos": tuple(float(v) for v in t),
                    "bbox_min": wall.get("bbox_min", [-0.5, -0.02, -0.5]),
                    "bbox_max": wall.get("bbox_max", [0.5, 0.02, 0.5]),
                })
            # Floor for single room — geometry.length=X, geometry.width=Y
            floor_data = rg.get("floor", {})
            if floor_data:
                fx = float(rg.get("length", 5.0))
                fy = float(rg.get("width", 5.0))
                room_walls.append({
                    "id": "floor",
                    "pos": (0.0, 0.0, 0.0),
                    "bbox_min": [-fx/2, -fy/2, 0.0],
                    "bbox_max": [fx/2, fy/2, 0.01],
                    "is_floor": True,
                })
        print(f"[export_scene] single room: {len(all_objects)} objects, {len(room_walls)} walls")

    # ── Convert GLTF→USD per object using asset_converter ────
    library = os.environ.get("SIM_SCENE_LIBRARY", "/data/embodied/scene/library")
    os.makedirs(library, exist_ok=True)
    ts = int(time.time())
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", output_name)[:60]
    cache_dir = os.path.join(library, f"{safe_name}_{ts}_assets")
    os.makedirs(cache_dir, exist_ok=True)

    # Copy gltf/pbr.mdl so asset_converter's MDL references resolve at runtime
    _mdl_candidates = [
        Path("/isaac-sim/kit/mdl/core/mdl/gltf/pbr.mdl"),
        Path("/isaacsim/kit/mdl/core/mdl/gltf/pbr.mdl"),
    ]
    for _mdl_src in _mdl_candidates:
        if _mdl_src.is_file():
            _mdl_dst = Path(cache_dir) / "gltf"
            _mdl_dst.mkdir(exist_ok=True)
            import shutil
            shutil.copy2(str(_mdl_src), str(_mdl_dst / "pbr.mdl"))
            print(f"[export_scene] Copied pbr.mdl to {_mdl_dst}")
            break

    converted = []  # list of (oid, usd_path, obj)
    skipped = 0

    def _convert_gltf(src, dst):
        ctx = ac.AssetConverterContext()
        ctx.ignore_materials = False
        ctx.ignore_cameras = True
        ctx.ignore_animations = True
        ctx.ignore_light = True
        ctx.export_preview_surface = True
        ctx.use_meter_as_world_unit = True
        ctx.convert_stage_up_z = True  # GLTF Y-up → USD Z-up
        task = ac.get_instance().create_converter_task(str(src), str(dst), ctx)
        # Wait for converter to finish (use asyncio loop, not sim app update)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(task.wait_until_finished())
        if not Path(dst).exists():
            return None
        # No post-conversion rotation here — the +90° X rotation is applied
        # on the asset prim in the scene USD assembly (see Objects section below).
        return dst

    for oid, obj in all_objects.items():
        geom_rel = obj.get("geometry_path", "")
        if not geom_rel:
            skipped += 1
            continue
        room_dir = obj.get("_room_dir", "")
        gltf_path = os.path.join(room_dir, geom_rel) if room_dir else geom_rel
        if not os.path.exists(gltf_path):
            skipped += 1
            continue
        safe_oid = _safe_prim_name(oid)
        obj_cache_dir = os.path.join(cache_dir, safe_oid)
        os.makedirs(obj_cache_dir, exist_ok=True)
        usd_file = os.path.join(obj_cache_dir, f"{safe_oid}.usd")
        if not os.path.exists(usd_file):
            result = _convert_gltf(gltf_path, usd_file)
            if not result:
                print(f"[export_scene] SKIP {oid}: convert failed")
                skipped += 1
                continue
            # Copy gltf/pbr.mdl next to each USD so MDL references resolve
            _obj_gltf_dir = Path(obj_cache_dir) / "gltf"
            if not (_obj_gltf_dir / "pbr.mdl").exists():
                for _mc in _mdl_candidates:
                    if _mc.is_file():
                        _obj_gltf_dir.mkdir(exist_ok=True)
                        shutil.copy2(str(_mc), str(_obj_gltf_dir / "pbr.mdl"))
                        break
        converted.append((oid, usd_file, obj))
        if len(converted) % 10 == 0:
            print(f"[export_scene] converted {len(converted)} objects...")

    print(f"[export_scene] converted {len(converted)}, skipped {skipped}")

    # ── Assemble scene USD ───────────────────────────────────
    scene_path = os.path.join(library, f"{safe_name}_{ts}.usda")
    stage = Usd.Stage.CreateNew(scene_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    # Lights
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(800)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
    key.CreateIntensityAttr(2500)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
    kxf = UsdGeom.Xformable(key.GetPrim())
    kxf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    for i, cl in enumerate(ceiling_lights):
        cname = _safe_prim_name(cl["id"])
        light = UsdLux.SphereLight.Define(stage, f"/World/Lights/{cname}")
        light.CreateIntensityAttr(20000)
        light.CreateRadiusAttr(0.1)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
        lxf = UsdGeom.Xformable(light.GetPrim())
        lxf.AddTranslateOp().Set(Gf.Vec3d(*cl["pos"]))

    # Room walls + floors
    for wdata in room_walls:
        wid = _safe_prim_name(wdata["id"])
        is_floor = wdata.get("is_floor", False)
        bmin = wdata["bbox_min"]
        bmax = wdata["bbox_max"]
        sx = max(0.001, float(bmax[0] - bmin[0]))
        sy = max(0.001, float(bmax[1] - bmin[1]))
        sz = max(0.001, float(bmax[2] - bmin[2]))
        cx = float((bmin[0] + bmax[0]) * 0.5 + wdata["pos"][0])
        cy = float((bmin[1] + bmax[1]) * 0.5 + wdata["pos"][1])
        cz = float((bmin[2] + bmax[2]) * 0.5 + wdata["pos"][2])
        prim_path = f"/World/Structure/{wid}"
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.GetSizeAttr().Set(1.0)
        if is_floor:
            cube.GetDisplayColorAttr().Set([(0.55, 0.45, 0.3)])  # wood-like
        else:
            cube.GetDisplayColorAttr().Set([(0.92, 0.92, 0.9)])  # off-white walls
        wxf = UsdGeom.Xformable(cube.GetPrim())
        wxf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
        wxf.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    # Glass panes for windows
    for gdata in glass_panes:
        gid = _safe_prim_name(gdata["id"])
        gbmin = gdata["bbox_min"]
        gbmax = gdata["bbox_max"]
        gsx = max(0.001, float(gbmax[0] - gbmin[0]))
        gsy = max(0.001, float(gbmax[1] - gbmin[1]))
        gsz = max(0.001, float(gbmax[2] - gbmin[2]))
        gcx = float((gbmin[0] + gbmax[0]) * 0.5 + gdata["pos"][0])
        gcy = float((gbmin[1] + gbmax[1]) * 0.5 + gdata["pos"][1])
        gcz = float((gbmin[2] + gbmax[2]) * 0.5 + gdata["pos"][2])
        glass_path = f"/World/Structure/{gid}"
        glass_cube = UsdGeom.Cube.Define(stage, glass_path)
        glass_cube.GetSizeAttr().Set(1.0)
        glass_cube.GetDisplayColorAttr().Set([(0.85, 0.92, 0.97)])  # light blue tint
        glass_cube.GetDisplayOpacityAttr().Set([0.2])
        gxf = UsdGeom.Xformable(glass_cube.GetPrim())
        gxf.AddTranslateOp().Set(Gf.Vec3d(gcx, gcy, gcz))
        gxf.AddScaleOp().Set(Gf.Vec3d(gsx, gsy, gsz))
        # Add glass material
        glass_mat = UsdShade.Material.Define(stage, f"{glass_path}/GlassMaterial")
        glass_shader = UsdShade.Shader.Define(stage, f"{glass_path}/GlassMaterial/Shader")
        glass_shader.SetShaderId("UsdPreviewSurface")
        glass_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.92, 0.97))
        glass_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.25)
        glass_shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
        glass_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.05)
        glass_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        glass_mat.CreateSurfaceOutput().ConnectToSource(glass_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(glass_cube.GetPrim()).Bind(glass_mat)
    if glass_panes:
        print(f"[export_scene] Added {len(glass_panes)} glass panes")

    # Objects
    for oid, usd_file, obj in converted:
        safe = _safe_prim_name(oid)
        prim_path = f"/World/Objects/{safe}"
        wrapper = UsdGeom.Xform.Define(stage, prim_path)

        # Transform with room offset
        t = obj.get("transform", {}).get("translation", [0, 0, 0])
        ox, oy = obj.get("_room_offset", (0, 0))
        gx, gy, gz = float(t[0]) + ox, float(t[1]) + oy, float(t[2])

        xf = UsdGeom.Xformable(wrapper.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(gx, gy, gz))
        q = obj.get("transform", {}).get("rotation_wxyz")
        if q and len(q) == 4:
            xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        # omni.kit.asset_converter writes glTF transforms at centimeter scale even
        # when the scene assembly here is authored in meters. Apply the unit
        # correction at the wrapper so object placement stays in SceneSmith meters.
        sf = obj.get("scale_factor", 1.0)
        asset_scale = 0.01
        if isinstance(sf, (int, float)):
            asset_scale *= float(sf)
        xf.AddScaleOp().Set(Gf.Vec3d(asset_scale, asset_scale, asset_scale))

        asset_prim = stage.DefinePrim(f"{prim_path}/asset", "Xform")
        # GLTF Y-up → USD Z-up: +90° X rotation on asset prim
        asset_axf = UsdGeom.Xformable(asset_prim)
        asset_axf.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 0))
        asset_prim.GetReferences().AddReference(_relative_usd_reference(scene_path, usd_file))

    # Apply collision on all meshes
    col_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_col.CreateApproximationAttr("convexHull")
            col_count += 1

    stage.GetRootLayer().Save()
    _flatten_scene_usd_in_place(scene_path, cache_dir)
    _make_tree_world_readable(cache_dir)
    _make_tree_world_readable(scene_path)
    print(f"[export_scene] DONE: {scene_path} ({len(converted)} objects, "
          f"{len(room_walls)} walls, {len(ceiling_lights)} lights, {col_count} collision meshes)")

    return {
        "success": True,
        "usd_path": scene_path,
        "objects": len(converted),
        "walls": len(room_walls),
        "lights": len(ceiling_lights),
        "skipped": skipped,
    }


# ── SceneSmith → USD import (legacy trimesh path) ────────────

def _find_scenesmith_scene_state(output_dir):
    """Search for scene state in SceneSmith output directory.

    Priority: house_state.json (multi-room) > final scene_state.json (single room).
    Returns (path, is_house_state) tuple.
    """
    root = Path(output_dir)

    # Priority 1: combined house_state (multi-room, has all rooms)
    # Try both with and without scene_000/ prefix (caller may pass either level)
    for pattern in [
        "combined_house_after_ceiling/house_state.json",
        "scene_000/combined_house_after_ceiling/house_state.json",
        "combined_house_after_*/house_state.json",
        "scene_000/combined_house_after_*/house_state.json",
        "combined_house/house_state.json",
        "scene_000/combined_house/house_state.json",
    ]:
        for hs in sorted(root.glob(pattern)):
            return str(hs), True

    # Priority 2: direct scene_state.json
    p = root / "scene_state.json"
    if p.exists():
        return str(p), False

    # Priority 3: per-room final scene_state
    for ss in sorted(root.glob("scene_000/room_*/scene_states/final_scene/scene_state.json")):
        return str(ss), False

    # Priority 4: any scene_state.json
    for ss in sorted(root.rglob("scene_state.json")):
        if "final_scene" in str(ss):
            return str(ss), False
    for ss in sorted(root.rglob("scene_state.json")):
        return str(ss), False

    return None, False


def _build_bridge_url(path: str, **params) -> str:
    query = urlencode(
        {k: v for k, v in params.items() if v is not None and str(v).strip() != ""}
    )
    return f"{path}?{query}" if query else path


def _cleanup_download_cache(max_age_sec: int = 6 * 60 * 60) -> None:
    try:
        DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    cutoff = time.time() - max_age_sec
    for child in DOWNLOAD_CACHE_DIR.glob("*"):
        try:
            if child.is_file() and child.stat().st_mtime < cutoff:
                child.unlink()
        except Exception:
            continue


def _make_scene_download_archive(scene_dir: str) -> Path:
    root = Path(scene_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    state_path, _ = _find_scenesmith_scene_state(str(root))
    if not state_path:
        raise ValueError(f"Not a valid SceneSmith output directory: {root}")

    _cleanup_download_cache()
    DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    parts = [_safe_token(part) for part in root.parts[-3:] if str(part).strip()]
    base_name = "_".join(parts) or "scenesmith_output"
    archive_base = DOWNLOAD_CACHE_DIR / f"{base_name}_{int(time.time())}_{os.getpid()}"
    archive_path = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=str(root.parent),
        base_dir=root.name,
    )
    return Path(archive_path)


def _write_portable_usd_root_copy(source_usd: Path, target_usd: Path, assets_dir: Path | None) -> None:
    if source_usd.suffix.lower() != ".usda" or assets_dir is None:
        shutil.copy2(source_usd, target_usd)
        return

    text = source_usd.read_text(encoding="utf-8")
    text = _rewrite_portable_usd_text(text, assets_dir)
    target_usd.write_text(text, encoding="utf-8")


def _make_tree_world_readable(root: str | Path) -> None:
    base = Path(root)
    if not base.exists():
        return

    targets = [base]
    if base.is_dir():
        targets.extend(sorted(base.rglob("*")))

    for path in targets:
        try:
            if path.is_symlink():
                continue
            path.chmod(0o755 if path.is_dir() else 0o644)
        except Exception:
            continue


def _relative_usd_reference(scene_path: str | Path, asset_path: str | Path) -> str:
    rel = os.path.relpath(str(asset_path), start=str(Path(scene_path).parent))
    return Path(rel).as_posix()


def _rewrite_portable_usd_text(text: str, assets_dir: Path) -> str:
    """Rewrite asset references so a USD file stays portable next to its assets dir."""
    for child in sorted(assets_dir.rglob("*")):
        if not child.is_file():
            continue

        rel_from_assets = child.relative_to(assets_dir).as_posix()
        bundle_rel = f"./{assets_dir.name}/{rel_from_assets}"
        local_rel = f"./{rel_from_assets}"
        abs_path = os.path.abspath(str(child))

        text = text.replace(abs_path, bundle_rel)
        text = text.replace(f"@{local_rel}@", f"@{bundle_rel}@")

    return text


def _flatten_scene_usd_in_place(scene_path: str | Path, assets_dir: str | Path | None) -> None:
    """Flatten a composed USD scene while preserving portable texture paths."""
    scene_path = Path(scene_path)
    if scene_path.suffix.lower() != ".usda":
        return

    assets_path = Path(assets_dir) if assets_dir else None
    if assets_path is None or not assets_path.exists():
        return

    from pxr import Usd

    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        return

    tmp_path = scene_path.with_name(f"{scene_path.stem}_flatten_tmp{scene_path.suffix}")
    stage.Export(str(tmp_path))
    text = tmp_path.read_text(encoding="utf-8")
    text = _rewrite_portable_usd_text(text, assets_path)
    scene_path.write_text(text, encoding="utf-8")
    tmp_path.unlink(missing_ok=True)


def _resolve_downloadable_usd_assets_dir(usd_path: Path) -> Path | None:
    assets_dir = usd_path.with_name(f"{usd_path.stem}_assets")
    if assets_dir.exists() and assets_dir.is_dir():
        return assets_dir
    return None


def _resolve_scenesmith_room_dir(scene_root: Path, room_name: str) -> Path | None:
    direct = scene_root / f"room_{room_name}"
    if direct.is_dir():
        return direct
    for child in sorted(scene_root.iterdir()):
        if child.is_dir() and room_name in child.name:
            return child
    return None


def _normalize_scene_state_objects(raw_objects) -> dict[str, dict]:
    if isinstance(raw_objects, list):
        return {
            str(item.get("object_id") or item.get("id") or f"obj_{idx}"): item
            for idx, item in enumerate(raw_objects)
            if isinstance(item, dict)
        }
    if isinstance(raw_objects, dict):
        return {
            str(key): value
            for key, value in raw_objects.items()
            if isinstance(value, dict)
        }
    return {}


def _resolve_texture_for_gltf_path(gltf_path: Path) -> Path | None:
    if not gltf_path.exists():
        return None

    preferred = gltf_path.with_name("Image_0.png")
    if preferred.is_file():
        return preferred

    for child in sorted(gltf_path.parent.iterdir()):
        if child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return child
    return None


def _collect_scenesmith_texture_sources(scene_dir: str) -> dict[str, Path]:
    state_path, is_house = _find_scenesmith_scene_state(scene_dir)
    if not state_path:
        return {}

    try:
        scene_state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except Exception:
        return {}

    texture_sources: dict[str, Path] = {}
    state_file = Path(state_path)

    if is_house and isinstance(scene_state.get("rooms"), dict):
        scene_root = state_file.parent.parent
        for room_name, room_data in scene_state.get("rooms", {}).items():
            if not isinstance(room_data, dict):
                continue
            room_dir = _resolve_scenesmith_room_dir(scene_root, str(room_name))
            if room_dir is None:
                continue
            for object_id, obj in _normalize_scene_state_objects(room_data.get("objects", {})).items():
                geom_rel = str(obj.get("geometry_path") or "").strip()
                if not geom_rel:
                    continue
                texture_path = _resolve_texture_for_gltf_path(room_dir / geom_rel)
                if texture_path is not None:
                    texture_sources[object_id] = texture_path
        return texture_sources

    objects = _normalize_scene_state_objects(scene_state.get("objects", {}))
    room_dir = state_file.parent.parent.parent
    for object_id, obj in objects.items():
        geom_rel = str(obj.get("geometry_path") or "").strip()
        if not geom_rel:
            continue
        texture_path = _resolve_texture_for_gltf_path(room_dir / geom_rel)
        if texture_path is not None:
            texture_sources[object_id] = texture_path
    return texture_sources


def _read_portable_usd_source_text(source_usd: Path) -> str:
    text = source_usd.read_text(encoding="utf-8")
    if 'references = @' not in text:
        return text

    from pxr import Usd

    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        return text

    tmp_path = DOWNLOAD_CACHE_DIR / f"{source_usd.stem}_{int(time.time())}_{os.getpid()}_flatten_tmp.usda"
    stage.Export(str(tmp_path))
    try:
        return tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)


def _rewrite_flattened_portable_usd_text(text: str, texture_targets: dict[str, str]) -> str:
    lines = text.splitlines(keepends=True)
    out_lines: list[str] = []
    brace_depth = 0
    objects_decl_pending = False
    objects_depth = None
    in_objects = False
    current_object = None
    current_object_depth = None
    object_decl_pending = None

    for line in lines:
        stripped = line.strip()
        updated = line.replace("@gltf/pbr.mdl@", "@./gltf/pbr.mdl@")

        if not in_objects and stripped.startswith('def "Objects"'):
            objects_decl_pending = True
        elif in_objects and current_object is None and objects_depth is not None and brace_depth == objects_depth:
            match = re.match(r'^def Xform "([^"]+)"(?:\s*\(|\s*)$', stripped)
            if match:
                object_decl_pending = match.group(1)

        if current_object and "asset inputs:texture" in updated:
            target = texture_targets.get(current_object)
            if target:
                updated = re.sub(r"@[^@]+@", f"@{target}@", updated, count=1)

        out_lines.append(updated)

        brace_depth += updated.count("{") - updated.count("}")

        if objects_decl_pending and "{" in updated:
            in_objects = True
            objects_depth = brace_depth
            objects_decl_pending = False

        if object_decl_pending and "{" in updated:
            current_object = object_decl_pending
            current_object_depth = brace_depth
            object_decl_pending = None

        if current_object_depth is not None and brace_depth < current_object_depth:
            current_object = None
            current_object_depth = None

        if objects_depth is not None and brace_depth < objects_depth:
            in_objects = False
            objects_depth = None
            current_object = None
            current_object_depth = None
            object_decl_pending = None

    return "".join(out_lines)


def _resolve_portable_pbr_mdl_path() -> Path | None:
    candidates = [
        Path("/isaac-sim/kit/mdl/core/mdl/gltf/pbr.mdl"),
        Path("/isaacsim/kit/mdl/core/mdl/gltf/pbr.mdl"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _make_portable_flattened_usd_archive(usd_path: str, scene_dir: str) -> Path | None:
    root = _resolve_downloadable_usd_path(usd_path)
    if root.suffix.lower() != ".usda":
        return None

    texture_sources = _collect_scenesmith_texture_sources(scene_dir)
    mdl_path = _resolve_portable_pbr_mdl_path()
    if not texture_sources or mdl_path is None:
        return None

    _cleanup_download_cache()
    DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    base_name = _safe_token(root.stem) or "scene"
    stamp = f"{int(time.time())}_{os.getpid()}"
    archive_path = DOWNLOAD_CACHE_DIR / f"{base_name}_{stamp}_usd_bundle.zip"
    portable_root = DOWNLOAD_CACHE_DIR / f"{base_name}_{stamp}_portable{root.suffix}"
    bundle_dir = DOWNLOAD_CACHE_DIR / f"{base_name}_{stamp}_bundle"
    textures_dir = bundle_dir / "textures"
    gltf_dir = bundle_dir / "gltf"
    textures_dir.mkdir(parents=True, exist_ok=True)
    gltf_dir.mkdir(parents=True, exist_ok=True)

    texture_targets: dict[str, str] = {}
    for object_id, source_path in texture_sources.items():
        target_dir = textures_dir / _safe_token(object_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        texture_targets[object_id] = f"./textures/{target_dir.name}/{target_path.name}"

    portable_text = _read_portable_usd_source_text(root)
    portable_text = _rewrite_flattened_portable_usd_text(portable_text, texture_targets)
    portable_root.write_text(portable_text, encoding="utf-8")
    shutil.copy2(mdl_path, gltf_dir / "pbr.mdl")

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(portable_root, arcname=root.name)
        archive.write(gltf_dir / "pbr.mdl", arcname="gltf/pbr.mdl")
        for child in sorted(textures_dir.rglob("*")):
            if child.is_file():
                rel_path = child.relative_to(bundle_dir).as_posix()
                archive.write(child, arcname=rel_path)
    return archive_path


def _make_usd_download_archive(usd_path: str, scene_dir: str | None = None) -> Path:
    if scene_dir:
        portable_archive = _make_portable_flattened_usd_archive(usd_path, scene_dir)
        if portable_archive is not None:
            return portable_archive

    root = _resolve_downloadable_usd_path(usd_path)
    assets_dir = _resolve_downloadable_usd_assets_dir(root)
    if assets_dir is None:
        return root

    _cleanup_download_cache()
    DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    base_name = _safe_token(root.stem) or "scene"
    archive_path = DOWNLOAD_CACHE_DIR / f"{base_name}_{int(time.time())}_{os.getpid()}_usd_bundle.zip"
    portable_root = DOWNLOAD_CACHE_DIR / f"{base_name}_{int(time.time())}_{os.getpid()}_portable{root.suffix}"
    _write_portable_usd_root_copy(root, portable_root, assets_dir)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(portable_root, arcname=root.name)
        for child in sorted(assets_dir.rglob("*")):
            if child.is_file():
                rel_path = child.relative_to(assets_dir).as_posix()
                archive.write(child, arcname=f"{assets_dir.name}/{rel_path}")
    return archive_path


def _resolve_downloadable_usd_path(usd_path: str) -> Path:
    path = Path(usd_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"USD file not found: {path}")
    if path.suffix.lower() not in USD_SUFFIXES:
        raise ValueError(f"Not a USD file: {path}")
    return path


def _parse_sdf_mass(sdf_path):
    """Extract mass from SDF XML file."""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(sdf_path)
        for mass_el in tree.getroot().iter("mass"):
            return float(mass_el.text)
    except Exception:
        pass
    return 0.1


def _safe_prim_name(name):
    """Sanitize string for use as USD prim name."""
    import re as _re
    s = _re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
    if s and s[0].isdigit():
        s = "_" + s
    return s or "_unnamed"


def _gltf_to_usd_via_pxr(gltf_path, usd_path, obj_id="object"):
    """Convert GLTF/GLB to USD using trimesh (mesh data) + pxr (USD write)."""
    import trimesh
    from pxr import Gf, Usd, UsdGeom, Vt

    scene = trimesh.load(gltf_path)
    # Normalize: Scene → single mesh
    if isinstance(scene, trimesh.Scene):
        mesh = scene.dump(concatenate=True)
    elif isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        raise ValueError(f"Cannot load mesh from {gltf_path}")

    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"Empty mesh: {gltf_path}")

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    safe_id = _safe_prim_name(obj_id)
    root = UsdGeom.Xform.Define(stage, f"/{safe_id}")
    stage.SetDefaultPrim(root.GetPrim())

    usd_mesh = UsdGeom.Mesh.Define(stage, f"/{safe_id}/Mesh")
    points = [Gf.Vec3f(*v) for v in mesh.vertices.tolist()]
    usd_mesh.CreatePointsAttr(points)
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())

    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        normals = [Gf.Vec3f(*n) for n in mesh.vertex_normals.tolist()]
        usd_mesh.CreateNormalsAttr(normals)
        usd_mesh.SetNormalsInterpolation("vertex")

    stage.GetRootLayer().Save()
    return str(usd_path)


def _stamp_physics_on_usd(usd_path, object_type, sdf_path=None):
    """Apply physics APIs to a per-object USD file.

    manipuland: RigidBody + Mass + Collision (dynamic, can be picked up)
    furniture/other: Collision only (static collider)
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print(f"[scenesmith-import] WARNING: cannot open {usd_path}")
        return False

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        children = list(stage.GetPseudoRoot().GetChildren())
        if not children:
            print(f"[scenesmith-import] WARNING: no prims in {usd_path}")
            return False
        default_prim = children[0]
        stage.SetDefaultPrim(default_prim)

    # Collision on all meshes
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mc.CreateApproximationAttr("convexHull")

    if object_type == "manipuland":
        # Dynamic rigid body
        UsdPhysics.RigidBodyAPI.Apply(default_prim)
        rb = UsdPhysics.RigidBodyAPI(default_prim)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(False)

        mass = _parse_sdf_mass(sdf_path) if sdf_path and os.path.exists(str(sdf_path)) else 0.1
        UsdPhysics.MassAPI.Apply(default_prim)
        UsdPhysics.MassAPI(default_prim).CreateMassAttr(float(mass))

    stage.GetRootLayer().Export(str(usd_path))
    return True


def _assemble_scene_usd(scene_path, converted_objects, ceiling_lights=None):
    """Create combined scene USD from per-object USDs."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf

    stage = Usd.Stage.CreateNew(str(scene_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    # Default ground plane + environment (required for Isaac Sim render pipeline + lighting)
    DEFAULT_ENV_USD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd"
    gp_prim = stage.DefinePrim("/World/defaultGroundPlane", "Xform")
    gp_prim.GetReferences().AddReference(DEFAULT_ENV_USD)

    # Ambient dome light (always add for visibility)
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(800)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))

    # Key distant light from above
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
    key.CreateIntensityAttr(2500)
    key.CreateAngleAttr(1.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
    key_xf = UsdGeom.Xformable(key.GetPrim())
    key_xf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    # Ceiling lights from SceneSmith (if any)
    for i, cl in enumerate(ceiling_lights or []):
        pos = cl.get("pos", [0, 0, 2.5])
        cl_name = _safe_prim_name(cl.get("id", f"ceiling_light_{i}"))
        light = UsdLux.SphereLight.Define(stage, f"/World/Lights/{cl_name}")
        light.CreateIntensityAttr(20000)
        light.CreateRadiusAttr(0.1)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
        cl_xf = UsdGeom.Xformable(light.GetPrim())
        cl_xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    # Objects
    for item in converted_objects:
        obj = item["obj"]
        usd_file = item["usd"]
        obj_id = obj.get("object_id") or obj.get("id", "unknown")
        safe_id = _safe_prim_name(obj_id)

        prim_path = f"/World/{safe_id}"
        prim = stage.OverridePrim(prim_path)
        prim.GetReferences().AddReference(_relative_usd_reference(scene_path, usd_file))

        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()

        # Translation
        t = obj.get("transform", {}).get("translation", [0, 0, 0])
        xf.AddTranslateOp().Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))

        # Rotation (wxyz quaternion)
        q = obj.get("transform", {}).get("rotation_wxyz")
        if q and len(q) == 4:
            xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))

    n_lights = len(ceiling_lights or [])
    stage.GetRootLayer().Save()
    assets_dir = Path(str(scene_path)).with_name(f"{Path(str(scene_path)).stem}_assets")
    _flatten_scene_usd_in_place(scene_path, assets_dir)
    _make_tree_world_readable(assets_dir)
    _make_tree_world_readable(scene_path)
    print(f"[scenesmith-import] Assembled scene: {scene_path} ({len(converted_objects)} objects, {n_lights} ceiling lights)")
    return True


def _convert_scenesmith_to_usd(output_dir, scene_name, objects_meta=None):
    """Convert SceneSmith output to physics-ready scene.usda.

    Args:
        output_dir: SceneSmith output directory (absolute path)
        scene_name: Human-readable name for the scene
        objects_meta: Optional list of dicts with {id, group, physics} from backend

    Returns: absolute path to generated .usda file
    """
    state_path, is_house = _find_scenesmith_scene_state(output_dir)
    if not state_path:
        raise FileNotFoundError(f"No scene_state.json found in {output_dir}")

    with open(state_path) as f:
        scene_state = json.load(f)

    # Collect all objects from house_state (multi-room) or scene_state (single room)
    objects = {}
    ceiling_lights = []  # Track ceiling lights for auto Light prim
    room_dirs = {}  # room_name → directory path

    if is_house and "rooms" in scene_state:
        scene_root = str(Path(state_path).parent.parent)  # scene_000/
        for room_name, room_data in scene_state.get("rooms", {}).items():
            if not isinstance(room_data, dict):
                continue
            room_objs = room_data.get("objects", {})
            if isinstance(room_objs, list):
                room_objs = {o.get("object_id", f"obj_{i}"): o for i, o in enumerate(room_objs)}
            elif not isinstance(room_objs, dict):
                continue
            # Find room directory
            room_dir_path = os.path.join(scene_root, f"room_{room_name}")
            if not os.path.isdir(room_dir_path):
                # Try without prefix
                for d in Path(scene_root).iterdir():
                    if d.is_dir() and room_name in d.name:
                        room_dir_path = str(d)
                        break
            room_dirs[room_name] = room_dir_path

            for oid, obj in room_objs.items():
                obj["_room"] = room_name
                obj["_room_dir"] = room_dir_path
                objects[oid] = obj
                # Track ceiling lights
                desc = obj.get("description", "").lower()
                if any(w in desc for w in ["light", "lamp", "pendant", "chandelier", "flush"]):
                    pos = obj.get("transform", {}).get("translation", [0, 0, 2.5])
                    ceiling_lights.append({"id": oid, "pos": pos, "desc": obj.get("description", "")})
        print(f"[scenesmith-import] house_state: {len(objects)} objects across {len(room_dirs)} rooms, {len(ceiling_lights)} ceiling lights")
    else:
        objects = scene_state.get("objects", {})
        if isinstance(objects, list):
            objects = {o.get("object_id", f"obj_{i}"): o for i, o in enumerate(objects)}

    # Room dir fallback for single-room scenes
    if not room_dirs:
        room_dir = str(Path(state_path).parent.parent.parent)
    else:
        room_dir = None  # Use per-object _room_dir

    # Merge backend metadata
    meta_map = {}
    if objects_meta:
        for o in objects_meta:
            if isinstance(o, dict) and "id" in o:
                meta_map[o["id"]] = o

    # Output paths
    library = os.environ.get("SIM_SCENE_LIBRARY", "/data/embodied/scene/library")
    os.makedirs(library, exist_ok=True)
    ts = int(time.time())
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", scene_name)[:60]
    assets_dir = os.path.join(library, f"{safe_name}_{ts}_assets")
    os.makedirs(assets_dir, exist_ok=True)

    converted = []
    for obj_id, obj in objects.items():
        geom_rel = obj.get("geometry_path", "")
        if not geom_rel:
            continue

        # Resolve GLTF path: per-room dir or single room dir
        obj_room_dir = obj.get("_room_dir", room_dir)
        gltf_path = os.path.join(obj_room_dir, geom_rel) if obj_room_dir else geom_rel
        if not os.path.exists(gltf_path):
            print(f"[scenesmith-import] WARNING: GLTF missing: {gltf_path}")
            continue

        usd_file = os.path.join(assets_dir, f"{_safe_prim_name(obj_id)}.usda")
        try:
            usd_file = _gltf_to_usd_via_pxr(gltf_path, usd_file, obj_id)
        except Exception as ex:
            print(f"[scenesmith-import] WARNING: convert failed for {obj_id}: {ex}")
            continue

        # Determine physics type
        meta = meta_map.get(obj_id, {})
        obj_type = meta.get("group") or obj.get("object_type", "furniture")
        obj_type = obj_type.strip().lower()

        # SDF path for mass
        sdf_rel = obj.get("sdf_path", "")
        sdf_path = os.path.join(room_dir, sdf_rel) if sdf_rel else None

        _stamp_physics_on_usd(usd_file, obj_type, sdf_path)

        converted.append({"obj": obj, "usd": usd_file, "group": obj_type})
        print(f"[scenesmith-import] {obj_id} ({obj_type}) -> {os.path.basename(usd_file)}")

    if not converted:
        raise ValueError(f"No objects converted from {output_dir}")

    scene_path = os.path.join(library, f"{safe_name}_{ts}.usda")
    _assemble_scene_usd(scene_path, converted, ceiling_lights=ceiling_lights)
    return scene_path


@bridge.route("/scene/import_scenesmith", methods=["POST"])
def scene_import_scenesmith():
    """Convert SceneSmith output to physics-ready USD and save to scene library."""
    data = flask_request.get_json(silent=True) or {}
    output_dir = data.get("scenesmith_output_dir")
    name = data.get("name", "scenesmith_scene")
    objects_meta = data.get("objects")

    if not output_dir:
        return jsonify({"error": "scenesmith_output_dir required"}), 400
    if not os.path.isdir(output_dir):
        return jsonify({"error": f"Directory not found: {output_dir}"}), 404

    try:
        usd_path = _convert_scenesmith_to_usd(output_dir, name, objects_meta)
        print(f"[scenesmith-import] DONE: {usd_path}")
        return jsonify(
            {
                "success": True,
                "usd_path": usd_path,
                "usd_download_url": _build_bridge_url(
                    "/scene/download_usd", usd_path=usd_path, scene_dir=output_dir
                ),
                "scenesmith_output_download_url": _build_bridge_url(
                    "/scene/download_output", scene_dir=output_dir
                ),
            }
        )
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(ex)}), 500


@bridge.route("/scene/info", methods=["GET"])
def scene_info():
    stage = _get_stage()
    if not stage:
        return jsonify({"error": "No stage"}), 500

    prims = []
    for prim in stage.Traverse():
        prims.append({
            "path": prim.GetPath().pathString,
            "type": prim.GetTypeName(),
        })

    return jsonify({
        "scene": _state["scene"],
        "prim_count": len(prims),
        "prims": prims[:200],
    })


@bridge.route("/camera/move", methods=["POST"])
def camera_move():
    """Move camera in a direction. Enqueued to main loop for safe execution."""
    data = request.get_json(silent=True) or {}
    direction = data.get("direction", "overview")
    step = float(data.get("step", 2.0))
    return _enqueue_cmd("camera_move", direction=direction, step=step)


@bridge.route("/viewport/reset", methods=["POST"])
def viewport_reset():
    """Reset viewport camera to default overview angle."""
    stage = _get_stage()
    if not stage:
        return jsonify({"error": "No stage"}), 500
    cam = stage.GetPrimAtPath("/OmniverseKit_Persp")
    if not cam.IsValid():
        return jsonify({"error": "Persp camera not found"}), 404
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    eye = Gf.Vec3d(0.7, 0.6, 0.7)
    target = Gf.Vec3d(0.1, 0.0, 0.3)
    up = Gf.Vec3d(0, 0, 1)
    fwd = (target - eye).GetNormalized()
    right = Gf.Cross(fwd, up).GetNormalized()
    new_up = Gf.Cross(right, fwd).GetNormalized()
    rot_mtx = Gf.Matrix4d()
    rot_mtx.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
    rot_mtx.SetRow(1, Gf.Vec4d(new_up[0], new_up[1], new_up[2], 0))
    rot_mtx.SetRow(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0))
    rot_mtx.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1))
    xform_op = xf.AddTransformOp()
    xform_op.Set(rot_mtx)
    cam_api = UsdGeom.Camera(cam)
    cam_api.GetFocalLengthAttr().Set(20.0)
    return jsonify({"success": True, "camera": "reset to default view"})


@bridge.route("/scene/snapshot", methods=["GET"])
def scene_snapshot():
    """Rich scene snapshot for save: objects with position, physics, bbox, lighting, cameras."""
    stage = _get_stage()
    if not stage:
        return jsonify({"error": "No stage"}), 500

    from pxr import UsdPhysics, UsdLux

    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    world = stage.GetPrimAtPath("/World")

    objects = []
    lights = []
    cameras = []

    for prim in (world.GetChildren() if world.IsValid() else []):
        if not prim.IsValid():
            continue
        path = prim.GetPath().pathString
        name = prim.GetName()
        type_name = prim.GetTypeName()

        # Skip internal prims
        if type_name in ("Scope", "Shader", "Material"):
            continue

        # Camera
        if prim.IsA(UsdGeom.Camera):
            cam_data = {"path": path, "name": name}
            try:
                xf = UsdGeom.Xformable(prim)
                mat = xf.ComputeLocalToWorldTransform(0)
                cam_data["position"] = [float(mat[3][0]), float(mat[3][1]), float(mat[3][2])]
            except Exception:
                pass
            cameras.append(cam_data)
            continue

        # Lights
        if prim.IsA(UsdLux.BoundableLightBase) or prim.IsA(UsdLux.NonboundableLightBase):
            light_data = {"path": path, "name": name, "type": type_name}
            try:
                xf = UsdGeom.Xformable(prim)
                mat = xf.ComputeLocalToWorldTransform(0)
                light_data["position"] = [float(mat[3][0]), float(mat[3][1]), float(mat[3][2])]
            except Exception:
                pass
            lights.append(light_data)
            continue

        # Regular object
        obj = {"path": path, "name": name, "type": type_name}

        # Position
        try:
            xf = UsdGeom.Xformable(prim)
            translate_op = None
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op:
                t = translate_op.Get()
                obj["position"] = [float(t[0]), float(t[1]), float(t[2])]
        except Exception:
            pass

        # Bounding box
        try:
            bbox_cache.Clear()
            bbox = bbox_cache.ComputeWorldBound(prim).GetRange()
            bmin, bmax = bbox.GetMin(), bbox.GetMax()
            obj["bbox_min"] = [float(bmin[0]), float(bmin[1]), float(bmin[2])]
            obj["bbox_max"] = [float(bmax[0]), float(bmax[1]), float(bmax[2])]
        except Exception:
            pass

        # Physics properties (check prim and children)
        has_rigid_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_articulation = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        if not has_collision:
            for child in prim.GetAllChildren():
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    has_collision = True
                    break
        obj["physics"] = {
            "rigid_body": has_rigid_body,
            "collision": has_collision,
            "articulation": has_articulation,
        }

        objects.append(obj)

    return jsonify({
        "scene": _state["scene"],
        "objects": objects,
        "lights": lights,
        "cameras": cameras,
        "physics_running": _state.get("physics", False),
        "robots": _state.get("robots", {}),
    })


# ── Robot endpoints ──────────────────────────────────────────

ROBOT_USD_MAP = {
    "franka": "/home/user/magicphysics/MagicPhysics/packages/MagicSim/Assets/Robots/franka_umi.usd",
    "franka_umi": "/home/user/magicphysics/MagicPhysics/packages/MagicSim/Assets/Robots/franka_umi.usd",
    "openarm": "/data/embodied/asset/robots/openarm_bimanual/configuration/openarm_bimanual_base.usd",
    "openarm_bimanual": "/data/embodied/asset/robots/openarm_bimanual/configuration/openarm_bimanual_base.usd",
}

# ── Inference routes ──────────────────────────────────────────

_inf_cameras = {}        # {cam_name: (render_product, annotator)}
_inf_cam_resolution = (640, 480)
_inf_robot_prim = None   # cached ArticulationRootAPI prim
_inf_dc = None           # cached dc interface
_inf_init_pending = False  # deferred init flag
_inf_init_phase = 0        # 0=idle, 1=play+step, 2=init_art, 3=cameras, 4=done
_inf_init_result = None    # result dict when done
_inf_init_error = None     # error string if failed
_inf_init_step_count = 0   # frame counter for phased init


def _find_robot_articulation():
    """Find first ArticulationRootAPI prim in the scene."""
    from pxr import UsdPhysics
    stage = _get_stage()
    if not stage:
        print("[_find_robot_articulation] no stage")
        return None
    prim_count = 0
    for prim in stage.Traverse():
        prim_count += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim
    print(f"[_find_robot_articulation] traversed {prim_count} prims, no ArticulationRootAPI found")
    return None


@bridge.route("/robot/init_inference", methods=["POST"])
def robot_init_inference():
    """Trigger deferred inference init on main thread.

    Sets a flag — main loop picks it up and runs the actual init.
    Returns immediately. Poll GET /robot/init_inference for status.
    CRITICAL: No Isaac Sim imports or API calls here (Flask thread).
    """
    global _inf_init_pending, _inf_init_phase, _inf_init_result, _inf_init_error, _inf_init_step_count, _inf_dc
    data = flask_request.get_json(silent=True) or {}
    force = data.get("force", False)
    if _inf_dc is not None and not _inf_init_pending and not force:
        return jsonify({"status": "ready", **(_inf_init_result or {})})
    if _inf_init_pending and _inf_init_phase < 4:
        # Already initializing — don't restart even if force=True
        return jsonify({"status": "initializing", "phase": _inf_init_phase})
    if force and _inf_dc is not None:
        # Force re-init: clear cached state so main loop rebuilds everything
        _inf_dc = None
        _inf_init_result = None
        _inf_cameras.clear()
        print("[init_inference] force re-init requested")
    if _inf_init_error and not force:
        return jsonify({"status": "error", "error": _inf_init_error}), 500
    _inf_init_pending = True
    _inf_init_phase = 0
    _inf_init_result = None
    _inf_init_error = None
    _inf_init_step_count = 0
    return jsonify({"status": "started"})


@bridge.route("/robot/init_inference", methods=["GET"])
def robot_init_inference_status():
    """Poll init status. No Isaac Sim calls here (Flask thread)."""
    if _inf_init_error:
        return jsonify({"status": "error", "error": _inf_init_error}), 500
    if _inf_dc is not None and _inf_init_result:
        return jsonify({"status": "ready", **_inf_init_result})
    if _inf_init_pending:
        return jsonify({"status": "initializing", "phase": _inf_init_phase})
    return jsonify({"status": "not_initialized"})


@bridge.route("/robot/obs", methods=["GET"])
def robot_obs():
    """Get robot joint positions + camera images for inference.

    During replay, reads from _replay_obs (updated each frame by replay loop).
    Otherwise, delegates to main thread via _enqueue_cmd.
    """
    # During replay, return cached obs from replay loop (main thread is blocked)
    replay_obs = _state.get("replay_obs")
    if _state.get("replaying") and replay_obs:
        return jsonify(replay_obs)
    if _inf_dc is None:
        return jsonify({"error": "Robot not initialized. Call /robot/init_inference first."}), 400
    return _enqueue_cmd("inf_obs")


@bridge.route("/robot/action", methods=["POST"])
def robot_action():
    """Send joint position targets to robot.

    Delegates to main thread via _enqueue_cmd to avoid GIL/PhysX deadlock.
    """
    if _inf_dc is None:
        return jsonify({"error": "Robot not initialized"}), 400

    data = flask_request.get_json(silent=True) or {}
    positions = data.get("positions")
    if positions is None:
        return jsonify({"error": "positions required"}), 400

    return _enqueue_cmd("inf_action", positions=positions)


@bridge.route("/robot/config", methods=["GET"])
def robot_config():
    robot_type = _get_active_robot_type(flask_request.args.get("robot_type", ""))
    explicit_path = flask_request.args.get("robot_config", "") or flask_request.args.get("path", "")
    camera_meta_path = flask_request.args.get("camera_meta", "")
    try:
        config_path, config, candidates = _load_robot_asset_yaml(
            "robot_config.yaml",
            robot_type=robot_type,
            explicit_path=explicit_path,
            sibling_of=camera_meta_path,
        )
    except Exception as exc:
        return jsonify({
            "error": f"Failed to load robot_config.yaml: {exc}",
            "robot_type": robot_type,
        }), 500

    if config_path is None:
        return jsonify({
            "error": "robot_config.yaml not found",
            "robot_type": robot_type,
            "candidates": candidates,
        }), 404

    if not isinstance(config, dict):
        return jsonify({
            "error": "robot_config.yaml must parse to an object",
            "robot_type": robot_type,
            "path": config_path,
        }), 500

    return jsonify(config)


@bridge.route("/robot/spawn", methods=["POST"])
def robot_spawn():
    data = flask_request.get_json(silent=True) or {}
    robot_type = data.get("robot_type", "franka")
    prim_path = data.get("prim_path", "/World/Robot")
    position = data.get("position", [0, 0, 0])
    usd_path = data.get("usd_path") or ROBOT_USD_MAP.get(robot_type)

    if not usd_path:
        return jsonify({"error": f"Unknown robot_type: {robot_type}"}), 400

    return _enqueue_cmd("robot_spawn", robot_type=robot_type, prim_path=prim_path,
                        position=position, usd_path=usd_path)


@bridge.route("/robot/state", methods=["GET"])
def robot_state():
    return jsonify({
        "robots": _state["robots"],
        "physics": _state["physics"],
    })


# ── Physics control (thread-safe via command queue) ──────────

def _enqueue_cmd(cmd_type, **kwargs):
    """Enqueue a command for main-thread execution, wait for result."""
    timeout = kwargs.pop("timeout_override", None) or _CMD_TIMEOUT
    result_event = threading.Event()
    cmd = {"type": cmd_type, "result": None, "error": None, "event": result_event, **kwargs}
    _cmd_queue.put(cmd)
    if not result_event.wait(timeout=timeout):
        if _state.get("collecting"):
            return jsonify({"error": "Timeout waiting for main thread (busy collecting). Use /emergency_stop to interrupt immediately."}), 504
        return jsonify({"error": "Timeout waiting for main thread"}), 504
    if cmd["error"]:
        return jsonify({"error": cmd["error"]}), 500
    return jsonify(cmd["result"])


def _drain_pending_commands(error_message: str) -> int:
    """Fail all queued commands immediately (used by emergency stop)."""
    drained = 0
    while True:
        try:
            cmd = _cmd_queue.get_nowait()
        except queue.Empty:
            break
        cmd["error"] = error_message
        cmd.get("event").set()
        drained += 1
    return drained


def _request_emergency_stop(reason: str = "manual_estop") -> dict:
    """Emergency-stop entrypoint safe to call from bridge thread."""
    global _collect_request, _replay_request, _replay_record_request
    _collect_stop.set()
    _collect_request = None
    _replay_stop.set()
    _replay_request = None
    _replay_record_request = None
    _estop_event.set()
    _state["estop_requested"] = True
    _state["last_estop_reason"] = str(reason)
    _state["last_estop_at"] = time.time()
    if _state.get("collect_progress"):
        _state["collect_progress"]["status"] = "emergency_stop_requested"
    drained = _drain_pending_commands("Command cancelled by emergency stop")
    return {
        "status": "estop_requested",
        "collecting": bool(_state.get("collecting")),
        "drained_commands": drained,
        "reason": str(reason),
    }


@bridge.route("/physics/play", methods=["POST"])
def physics_play():
    if not world:
        return jsonify({"error": "World not available"}), 500
    return _enqueue_cmd("physics_play")


@bridge.route("/physics/pause", methods=["POST"])
def physics_pause():
    if not world:
        return jsonify({"error": "World not available"}), 500
    return _enqueue_cmd("physics_pause")


@bridge.route("/physics/state", methods=["GET"])
def physics_state():
    return jsonify({
        "physics": _state["physics"],
        "step": _state["step"],
    })


# ── Code execution (for OpenClaw) ────────────────────────────

@bridge.route("/code/execute", methods=["POST"])
def code_execute():
    data = flask_request.get_json(silent=True) or {}
    code = data.get("code", "")
    if not code:
        return jsonify({"error": "code required"}), 400
    return _enqueue_cmd("code_execute", code=code)


# ── Scene export (SceneSmith → USD with asset_converter) ─────

@bridge.route("/scene/export_scene", methods=["POST"])
def scene_export_scene():
    """Export a SceneSmith scene to physics-ready USD using omni.kit.asset_converter.

    Params:
        scene_dir: SceneSmith output directory
        name: output scene name
        room: (optional) export only this room (e.g. "living_room"). Default: all rooms.
    """
    data = flask_request.get_json(silent=True) or {}
    scene_dir = data.get("scene_dir", "")
    output_name = data.get("name", "exported_scene")
    room_filter = data.get("room")  # None = all rooms
    if not scene_dir:
        return jsonify({"error": "scene_dir required"}), 400
    if not os.path.isdir(scene_dir):
        return jsonify({"error": f"Directory not found: {scene_dir}"}), 404
    try:
        result = _handle_export_scene(scene_dir, output_name, room_filter=room_filter)
        usd_path = result.get("usd_path")
        if usd_path:
            result["usd_download_url"] = _build_bridge_url(
                "/scene/download_usd", usd_path=usd_path, scene_dir=scene_dir
            )
        result["scenesmith_output_download_url"] = _build_bridge_url(
            "/scene/download_output", scene_dir=scene_dir
        )
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@bridge.route("/scene/download_output", methods=["GET"])
def scene_download_output():
    """Download the original SceneSmith output directory as a zip archive."""
    scene_dir = (flask_request.args.get("scene_dir") or "").strip()
    if not scene_dir:
        return jsonify({"error": "scene_dir required"}), 400
    try:
        archive_path = _make_scene_download_archive(scene_dir)
        return send_file(
            str(archive_path),
            as_attachment=True,
            mimetype="application/zip",
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@bridge.route("/scene/download_usd", methods=["GET"])
def scene_download_usd():
    """Download an exported USD file, bundling sibling assets when present."""
    usd_path = (flask_request.args.get("usd_path") or "").strip()
    scene_dir = (flask_request.args.get("scene_dir") or "").strip()
    if not usd_path:
        return jsonify({"error": "usd_path required"}), 400
    bundle_requested = str(flask_request.args.get("bundle") or "1").strip().lower() not in {"0", "false", "no"}
    try:
        path = (
            _make_usd_download_archive(usd_path, scene_dir=scene_dir or None)
            if bundle_requested
            else _resolve_downloadable_usd_path(usd_path)
        )
        return send_file(
            str(path),
            as_attachment=True,
            mimetype="application/zip" if path.suffix.lower() == ".zip" else "application/octet-stream",
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ── Data collection ──────────────────────────────────────────

_collect_stop = threading.Event()
_collect_request = None
_replay_request = None
_replay_stop = threading.Event()
_replay_record_request = None

@bridge.route("/collect/start", methods=["POST"])
def collect_start():
    if _state["collecting"] or _collect_request is not None:
        return jsonify(
            {
                "status": "already_running",
                "collecting": True,
                "progress": _state.get("collect_progress"),
            }
        ), 200
    data = flask_request.get_json(silent=True) or {}
    try:
        num_episodes = int(data.get("num_episodes", 10))
        steps_per_segment = int(data.get("steps_per_segment", os.environ.get("COLLECT_STEPS_PER_SEGMENT_DEFAULT", "70")))
        episode_timeout_sec = float(data.get("episode_timeout_sec", os.environ.get("COLLECT_EPISODE_TIMEOUT_SEC", "300")))
    except Exception:
        return jsonify({"error": "num_episodes/steps_per_segment/episode_timeout_sec must be numeric"}), 400
    skill = str(data.get("skill", "pick_place") or "pick_place")
    output_dir = _normalize_collect_output_dir(data.get("output_dir"), skill)
    scene_mode = str(data.get("scene_mode", "auto") or "auto")
    target_objects = data.get("target_objects")
    reset_mode = str(data.get("reset_mode", "full") or "full").strip().lower()
    try:
        rounds_per_episode = int(data.get("rounds_per_episode", 1))
    except Exception:
        rounds_per_episode = 1
    try:
        object_position_noise = float(data.get("object_position_noise", 0.0))
    except Exception:
        object_position_noise = 0.0
    if target_objects is not None and not isinstance(target_objects, list):
        return jsonify({"error": "target_objects must be a list when provided"}), 400

    if num_episodes <= 0:
        return jsonify({"error": "num_episodes must be > 0"}), 400
    if steps_per_segment <= 0:
        return jsonify({"error": "steps_per_segment must be > 0"}), 400
    if episode_timeout_sec < 0:
        return jsonify({"error": "episode_timeout_sec must be >= 0"}), 400

    # Main-thread execution only (Isaac Sim API is not thread-safe).
    return _enqueue_cmd(
        "collect_start",
        num_episodes=num_episodes,
        steps_per_segment=steps_per_segment,
        episode_timeout_sec=episode_timeout_sec,
        output_dir=output_dir,
        skill=skill,
        scene_mode=scene_mode,
        target_objects=target_objects,
        reset_mode=reset_mode,
        rounds_per_episode=rounds_per_episode,
        object_position_noise=object_position_noise,
    )


@bridge.route("/collect/status", methods=["GET"])
def collect_status():
    return jsonify({
        "collecting": _state["collecting"],
        "progress": _state["collect_progress"],
    })


@bridge.route("/collect/stop", methods=["POST"])
def collect_stop():
    if not _state["collecting"]:
        return jsonify({"error": "No collection running"}), 400
    _collect_stop.set()
    if _state["collect_progress"]:
        _state["collect_progress"]["status"] = "stopping"
    return jsonify({"status": "stopping"})


@bridge.route("/replay/start", methods=["POST"])
def replay_start():
    """Start replaying a LeRobot dataset episode on the robot."""
    if _state["collecting"] or _collect_request is not None:
        return jsonify({"error": "Cannot replay while collecting"}), 400
    if _replay_request is not None or _state.get("replaying"):
        return jsonify({"error": "Replay already running"}), 400

    data = flask_request.get_json(silent=True) or {}
    dataset_path = data.get("dataset_path", "")
    episode_index = data.get("episode_index", 0)
    speed = data.get("speed", 1.0)
    scene_usd_path = data.get("scene_usd_path", "")
    camera_meta_path = data.get("camera_meta", "")

    if not dataset_path:
        return jsonify({"error": "dataset_path is required"}), 400

    try:
        episode_index = int(episode_index)
        speed = float(speed)
    except Exception:
        return jsonify({"error": "episode_index must be int, speed must be float"}), 400

    return _enqueue_cmd(
        "replay_start",
        dataset_path=dataset_path,
        episode_index=episode_index,
        speed=speed,
        scene_usd_path=scene_usd_path,
        camera_meta=camera_meta_path,
    )


@bridge.route("/replay/status", methods=["GET"])
def replay_status():
    return jsonify({
        "replaying": _state.get("replaying", False),
        "progress": _state.get("replay_progress"),
    })


@bridge.route("/replay/stop", methods=["POST"])
def replay_stop():
    if not _state.get("replaying"):
        return jsonify({"error": "No replay running"}), 400
    _replay_stop.set()
    if _state.get("replay_progress"):
        _state["replay_progress"]["status"] = "stopping"
    return jsonify({"status": "stopping"})


@bridge.route("/replay/record", methods=["POST"])
def replay_record():
    """Replay ALL episodes from a dataset with wrist camera recording.

    Produces a new LeRobot v3 dataset with observation.images.left_wrist_cam
    and observation.images.right_wrist_cam video streams.

    Body JSON:
        source_dataset: str — path to source LeRobot dataset
        output_dataset: str — path for new dataset (default: source + "_sim_cam")
        episodes: list[int] | null — episode indices to record (null = all)
        speed: float — replay speed multiplier (default 1.0)
        camera_meta: str — path to camera_meta.yaml (default: auto-detect)
    """
    if _state["collecting"] or _collect_request is not None:
        return jsonify({"error": "Cannot record while collecting"}), 400
    if _replay_request is not None or _state.get("replaying"):
        return jsonify({"error": "Replay already running"}), 400
    if _replay_record_request is not None:
        return jsonify({"error": "Replay record already queued"}), 400

    data = flask_request.get_json(silent=True) or {}
    source_dataset = data.get("source_dataset", "")
    if not source_dataset:
        return jsonify({"error": "source_dataset is required"}), 400

    output_dataset = data.get("output_dataset", "") or (source_dataset.rstrip("/") + "_sim_cam")
    episodes = data.get("episodes")  # None = all
    speed = float(data.get("speed", 1.0))
    camera_meta_path = data.get("camera_meta", "/data/embodied/asset/robots/openarm_bimanual/camera_meta.yaml")

    return _enqueue_cmd(
        "replay_record",
        source_dataset=source_dataset,
        output_dataset=output_dataset,
        episodes=episodes,
        speed=speed,
        camera_meta=camera_meta_path,
    )


@bridge.route("/emergency_stop", methods=["POST"])
def emergency_stop():
    """Immediate kill-switch: stop collect + cancel queued commands + pause physics on next frame."""
    data = flask_request.get_json(silent=True) or {}
    reason = data.get("reason", "manual_estop")
    result = _request_emergency_stop(reason=reason)
    return jsonify(result)


# ── Start Bridge API in daemon thread ────────────────────────
def _run_bridge():
    bridge.run(host="0.0.0.0", port=BRIDGE_PORT, threaded=True)

threading.Thread(target=_run_bridge, daemon=True, name="bridge-api").start()
print(f"[interactive] Bridge API on :{BRIDGE_PORT}")

# ── Warmup frames ────────────────────────────────────────────
print("[interactive] Running warmup frames...")
for _ in range(60):
    simulation_app.update()
_restore_autosave_stage()

# ── Pre-warm Curobo CUDA JIT inside this process ─────────────
if os.environ.get("COLLECT_PLANNER_BACKEND", "").strip().lower() == "curobo":
    def _curobo_jit_warmup():
        """Import curobo modules to trigger CUDA JIT compilation so the
        first collect doesn't waste 6+ minutes recompiling."""
        try:
            print("[interactive] Pre-warming curobo CUDA extensions (JIT) ...")
            from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401
            print("[interactive] Curobo JIT warmup OK")
        except Exception as exc:
            print(f"[interactive] Curobo JIT warmup skipped: {exc}")
    _curobo_jit_warmup()

print(f"[interactive] Ready. WebRTC port={WEBRTC_PORT}, Kit API=8011, Bridge={BRIDGE_PORT}")

# ── Main-thread command processor ────────────────────────────
def _process_commands():
    """Drain command queue and execute on main thread (called each frame)."""
    global PHYSICS_RUNNING, _collect_request, _replay_request, _replay_record_request, _inf_dc, _inf_init_result
    while not _cmd_queue.empty():
        try:
            cmd = _cmd_queue.get_nowait()
        except queue.Empty:
            break

        cmd_type = cmd["type"]
        try:
            if cmd_type == "physics_play":
                stage = _get_stage()
                if stage is not None:
                    cleaned = _sanitize_franka_root_rigidbody(stage)
                    if cleaned:
                        print(f"[physics] removed accidental robot root RigidBodyAPI: {cleaned}")
                world.play()
                PHYSICS_RUNNING = True
                _state["physics"] = True
                print("[interactive] Physics PLAY")
                cmd["result"] = {"physics": True}

            elif cmd_type == "physics_pause":
                world.pause()
                PHYSICS_RUNNING = False
                _state["physics"] = False
                print("[interactive] Physics PAUSE")
                cmd["result"] = {"physics": False}

            elif cmd_type == "scene_load":
                usd_path = cmd["usd_path"]
                print(f"[interactive] Loading scene: {usd_path}")

                if not os.path.exists(usd_path):
                    cmd["error"] = f"Scene file not found: {usd_path}"
                else:
                    # Enqueue individual object loads so main loop keeps
                    # running between them (preserves NVCF streaming heartbeat).
                    # Read the scene USD to find all object references, then
                    # add them one by one with simulation_app.update() in between.
                    try:
                        stage = _get_stage()
                        from pxr import Sdf, UsdLux

                        # 1. Clear previous scene
                        for child in list(stage.GetPrimAtPath("/World").GetChildren()):
                            name = child.GetName()
                            if name in ("defaultGroundPlane", "GroundPlane"):
                                continue
                            stage.RemovePrim(child.GetPath().pathString)
                        simulation_app.update()

                        # 2. Read scene USD to extract object refs + transforms
                        scene_stage = Usd.Stage.Open(usd_path)
                        scene_world = scene_stage.GetPrimAtPath("/World")
                        if not scene_world or not scene_world.IsValid():
                            cmd["error"] = f"No /World prim in {usd_path}"
                        else:
                            # Add lights first (from /World/Lights)
                            lights_prim = scene_stage.GetPrimAtPath("/World/Lights")
                            if lights_prim and lights_prim.IsValid():
                                light_xf = stage.DefinePrim("/World/SceneLights", "Xform")
                                light_xf.GetReferences().AddReference(usd_path, Sdf.Path("/World/Lights"))
                                simulation_app.update()
                                print("[scene_load] Lights added")

                            # Add structure (walls) as single reference
                            struct_prim = scene_stage.GetPrimAtPath("/World/Structure")
                            if struct_prim and struct_prim.IsValid():
                                wall_xf = stage.DefinePrim("/World/SceneWalls", "Xform")
                                wall_xf.GetReferences().AddReference(usd_path, Sdf.Path("/World/Structure"))
                                simulation_app.update()
                                print("[scene_load] Walls added")

                            # Add objects one by one from /World/Objects
                            objects_prim = scene_stage.GetPrimAtPath("/World/Objects")
                            obj_count = 0
                            if objects_prim and objects_prim.IsValid():
                                for child in objects_prim.GetChildren():
                                    child_name = child.GetName()
                                    obj_xf = stage.DefinePrim(f"/World/SceneObjects/{child_name}", "Xform")
                                    obj_xf.GetReferences().AddReference(
                                        usd_path, Sdf.Path(f"/World/Objects/{child_name}")
                                    )
                                    # Check if object has scale (older exports lack 0.01 cm→m)
                                    xformable = UsdGeom.Xformable(obj_xf)
                                    has_scale = any(
                                        op.GetOpName() == "xformOp:scale"
                                        for op in xformable.GetOrderedXformOps()
                                    )
                                    if not has_scale:
                                        xformable.AddScaleOp().Set(Gf.Vec3d(0.01, 0.01, 0.01))
                                    obj_count += 1
                                    # Let main loop breathe every 3 objects
                                    if obj_count % 3 == 0:
                                        simulation_app.update()

                            # Final settle
                            for _ in range(10):
                                simulation_app.update()

                            prim_count = sum(1 for _ in stage.Traverse())
                            _state["scene"] = usd_path
                            _state["robots"] = {}
                            print(f"[interactive] Scene loaded: {usd_path} ({obj_count} objects, {prim_count} prims)")
                            cmd["result"] = {"success": True, "scene": usd_path, "objects": obj_count}

                    except Exception as load_exc:
                        cmd["error"] = f"Scene load failed: {load_exc}"
                        print(f"[scene_load] ERROR: {load_exc}")
                        import traceback
                        traceback.print_exc()

            elif cmd_type == "camera_move":
                try:
                    from omni.kit.viewport.utility import get_active_viewport, frame_viewport_prims
                    vp = get_active_viewport()
                    direction = cmd.get("direction", "overview")
                    step = float(cmd.get("step", 2.0))
                    stage = _get_stage()

                    if direction == "overview":
                        # Frame all scene objects
                        paths = []
                        for group in ("SceneObjects", "SceneWalls"):
                            grp = stage.GetPrimAtPath(f"/World/{group}")
                            if grp and grp.IsValid():
                                paths += [str(c.GetPath()) for c in grp.GetChildren()]
                        if paths:
                            frame_viewport_prims(vp, paths[:30])
                        cmd["result"] = {"camera": "overview"}
                    else:
                        # Move camera via viewport.transform in main loop (safe here)
                        t = vp.transform
                        pos = t.GetRow(3)
                        offsets = {
                            "left": (-step, 0, 0, 0),
                            "right": (step, 0, 0, 0),
                            "forward": (0, step, 0, 0),
                            "back": (0, -step, 0, 0),
                            "up": (0, 0, step, 0),
                            "down": (0, 0, -step, 0),
                        }
                        nt = Gf.Matrix4d(t)
                        if direction in ("zoomIn", "zoomOut"):
                            f = 0.85 if direction == "zoomIn" else 1.18
                            nt.SetRow(3, Gf.Vec4d(pos[0]*f, pos[1]*f, pos[2]*f, pos[3]))
                        else:
                            d = offsets.get(direction, (0, 0, 0, 0))
                            nt.SetRow(3, Gf.Vec4d(pos[0]+d[0], pos[1]+d[1], pos[2]+d[2], pos[3]))
                        # Set transform multiple times across frames to override
                        # viewport controller's per-frame camera reset
                        for _ in range(10):
                            vp.transform = nt
                            simulation_app.update()
                        new_pos = vp.transform.GetRow(3)
                        cmd["result"] = {"camera": direction, "pos": [new_pos[0], new_pos[1], new_pos[2]]}
                    print(f"[camera] {direction}")
                except Exception as cam_exc:
                    cmd["error"] = f"Camera move failed: {cam_exc}"
                    print(f"[camera] ERROR: {cam_exc}")

            elif cmd_type == "scene_save":
                stage = _get_stage()
                if not stage:
                    cmd["error"] = "No stage loaded"
                else:
                    out_dir = cmd["output_dir"]
                    out_path = cmd["output_path"]
                    os.makedirs(out_dir, exist_ok=True)
                    root_layer = stage.GetRootLayer()
                    if root_layer is None:
                        cmd["error"] = "No root layer"
                    elif bool(root_layer.Export(out_path)):
                        print(f"[interactive] Scene saved: {out_path}")
                        cmd["result"] = {"success": True, "usd_path": out_path}
                    else:
                        cmd["error"] = f"USD export failed: {out_path}"

            elif cmd_type == "robot_spawn":
                stage = _get_stage()
                if not stage:
                    cmd["error"] = "No stage loaded"
                else:
                    add_reference_to_stage = _add_ref  # compat
                    add_reference_to_stage(usd_path=cmd["usd_path"], prim_path=cmd["prim_path"])
                    xform = UsdGeom.Xformable(stage.GetPrimAtPath(cmd["prim_path"]))
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(Gf.Vec3d(*cmd["position"]))
                    for _ in range(10):
                        simulation_app.update()
                    _state["robots"][cmd["prim_path"]] = {
                        "type": cmd["robot_type"],
                        "position": cmd["position"],
                        "usd": cmd["usd_path"],
                    }
                    _state["active_robot_type"] = cmd["robot_type"]
                    _save_autosave_stage("robot_spawn")
                    print(f"[interactive] Spawned {cmd['robot_type']} at {cmd['prim_path']}")
                    cmd["result"] = {"success": True, "prim_path": cmd["prim_path"],
                                     "robot_type": cmd["robot_type"]}

            elif cmd_type == "inf_obs":
                # Read joint positions + camera images on main thread
                import base64 as _b64
                import io as _obs_io
                from PIL import Image as _PILImage
                import numpy as _np
                pos_list = []
                names = []
                try:
                    positions = _inf_dc.get_joint_positions()
                    pos_list = positions.tolist() if positions is not None else []
                    names = [str(n) for n in _inf_dc.dof_names] if _inf_dc.dof_names is not None else []
                except Exception as _obs_err:
                    # Articulation invalid (e.g. after scene reload) — clear cache
                    _inf_dc = None
                    _inf_init_result = None
                    print(f"[inf_obs] articulation read failed, cleared _inf_dc: {_obs_err}")
                images = {}
                for cam_name, (rp, annot) in _inf_cameras.items():
                    try:
                        rgba = annot.get_data()
                        if rgba is not None and hasattr(rgba, "shape") and len(rgba.shape) >= 2 and rgba.size > 0:
                            avg_px = float(rgba[:,:,:3].mean())
                            if avg_px < 1.0:
                                print(f"[inf_obs] WARNING: {cam_name} is BLACK (avg_pixel={avg_px:.2f}, shape={rgba.shape})")
                            img = _PILImage.fromarray(rgba[:, :, :3])
                            buf = _obs_io.BytesIO()
                            img.save(buf, format="JPEG", quality=80)
                            images[cam_name] = _b64.b64encode(buf.getvalue()).decode()
                        else:
                            print(f"[inf_obs] WARNING: {cam_name} annotator returned None or empty")
                    except Exception as e:
                        print(f"[inf_obs] ERROR {cam_name}: {e}")
                cmd["result"] = {
                    "positions": pos_list, "names": names,
                    "timestamp": time.time(), "images": images,
                }

            elif cmd_type == "inf_action":
                # Set joint position targets on main thread
                import numpy as _np
                target = _np.array(cmd["positions"], dtype=_np.float32)
                n_dof = _inf_dc.num_dof
                if len(target) < n_dof:
                    target = _np.pad(target, (0, n_dof - len(target)))
                elif len(target) > n_dof:
                    target = target[:n_dof]
                _inf_dc.set_joint_positions(target)
                cmd["result"] = {"success": True}

            elif cmd_type == "code_execute":
                import io as _io
                import builtins as _builtins
                from pxr import UsdPhysics, UsdShade, Sdf, Vt
                import sys as _sys
                stdout_capture = _io.StringIO()
                exec_code = _strip_runtime_helper_imports(cmd.get("code", ""))
                stage_mutated = _code_may_mutate_stage(exec_code)
                if _contains_unsafe_franka_physics_code(exec_code):
                    stdout_capture.write(
                        "Blocked unsafe Franka physics edit. "
                        "Do not apply RigidBody/Collision APIs on /World/Franka* roots.\n"
                    )
                    cmd["result"] = {"success": True, "output": stdout_capture.getvalue()}
                    continue
                # Tolerate LLM code that does `import UsdPhysics` (without pxr prefix).
                # pxr bindings are real Python modules, so aliasing them in sys.modules
                # allows plain imports to resolve instead of failing with ModuleNotFoundError.
                _sys.modules.setdefault("UsdPhysics", UsdPhysics)
                _sys.modules.setdefault("UsdGeom", UsdGeom)
                _sys.modules.setdefault("UsdShade", UsdShade)
                _sys.modules.setdefault("Usd", Usd)
                _sys.modules.setdefault("Sdf", Sdf)
                _sys.modules.setdefault("Vt", Vt)
                _sys.modules.setdefault("Gf", Gf)
                # Try importing common Isaac Sim utilities
                _extra_globals = {}
                try:
                    add_reference_to_stage = _add_ref  # compat
                    _extra_globals["add_reference_to_stage"] = add_reference_to_stage
                except ImportError:
                    pass
                try:
                    create_prim = _create_prim  # compat
                    _extra_globals["create_prim"] = create_prim
                except ImportError:
                    pass
                exec_globals = {
                    "__builtins__": _builtins,
                    "omni": omni, "Gf": Gf, "Sdf": Sdf, "Vt": Vt,
                    "UsdGeom": UsdGeom, "Usd": Usd, "UsdPhysics": UsdPhysics, "UsdShade": UsdShade,
                    "stage": _get_stage(), "world": world, "simulation_app": simulation_app,
                    "create_table": _runtime_create_table,
                    "create_franka": _runtime_create_franka,
                    "create_mug": _runtime_create_mug,
                    "create_apple": _runtime_create_apple,
                    "create_ball": _runtime_create_ball,
                    "print": lambda *a, **kw: stdout_capture.write(" ".join(str(x) for x in a) + "\n"),
                    **_extra_globals,
                }
                try:
                    exec(exec_code, exec_globals)
                except Exception as _exec_exc:
                    _exc_str = str(_exec_exc)
                    # USD pxr precision mismatch is a warning, not a real error
                    if "pxrInternal" in _exc_str and "Proceeding" in _exc_str:
                        pass  # operation succeeded despite warning
                    else:
                        raise
                # Optional postprocess: auto-lift objects above table if requested.
                # Default OFF to avoid visible "spawn then jump" behavior in Scene Chat.
                if SCENE_AUTO_LIFT_ENABLED:
                    try:
                        stage_now = _get_stage()
                        lifted = _lift_objects_above_table_if_needed(stage_now) if stage_now else []
                        if lifted:
                            print(f"[scene] auto-lifted objects for table clearance: {lifted}")
                    except Exception as _lift_exc:
                        print(f"[scene] WARNING: auto-lift failed: {_lift_exc}")
                # Clear selection to hide gizmo overlay in WebRTC stream
                try:
                    omni.usd.get_context().get_selection().clear_selected_prim_paths()
                except Exception:
                    pass
                for _ in range(5):
                    simulation_app.update()
                # Scene edits that remove/redefine prims can invalidate simulation/articulation views.
                # Rebuild world context after mutating code to keep collect/articulation stable.
                if stage_mutated:
                    was_physics_running = bool(PHYSICS_RUNNING)
                    if _recreate_world_for_open_stage("code_execute"):
                        if was_physics_running and world is not None:
                            try:
                                world.play()
                                PHYSICS_RUNNING = True
                                _state["physics"] = True
                            except Exception as _play_exc:
                                print(f"[world] WARNING: failed to resume physics after code_execute: {_play_exc}")
                    else:
                        print("[world] WARNING: failed to recreate world after code_execute mutation")
                _save_autosave_stage("code_execute")
                cmd["result"] = {"success": True, "output": stdout_capture.getvalue()}

            elif cmd_type == "collect_start":
                if _state["collecting"] or _collect_request is not None:
                    cmd["error"] = "Collection already running"
                else:
                    # Auto-play physics if needed
                    if not _state["physics"]:
                        try:
                            stage = _get_stage()
                            if stage is not None:
                                cleaned = _sanitize_franka_root_rigidbody(stage)
                                if cleaned:
                                    print(f"[physics] removed accidental robot root RigidBodyAPI: {cleaned}")
                            world.play()
                            PHYSICS_RUNNING = True
                            _state["physics"] = True
                            print("[interactive] Physics auto-PLAY for collect")
                        except Exception as _play_exc:
                            cmd["error"] = f"Failed to auto-play physics: {_play_exc}"
                            continue
                    try:
                        num_episodes = int(cmd.get("num_episodes", 10))
                        steps_per_segment = int(cmd.get("steps_per_segment", os.environ.get("COLLECT_STEPS_PER_SEGMENT_DEFAULT", "70")))
                        episode_timeout_sec = float(cmd.get("episode_timeout_sec", os.environ.get("COLLECT_EPISODE_TIMEOUT_SEC", "300")))
                    except Exception:
                        cmd["error"] = "num_episodes/steps_per_segment/episode_timeout_sec must be numeric"
                        continue
                    skill = str(cmd.get("skill", "pick_place") or "pick_place")
                    output_dir = _normalize_collect_output_dir(cmd.get("output_dir"), skill)
                    scene_mode = str(cmd.get("scene_mode", "auto") or "auto")
                    reset_mode = str(cmd.get("reset_mode", "full") or "full").strip().lower()
                    rounds_per_episode = int(cmd.get("rounds_per_episode", 1))
                    try:
                        object_position_noise = float(cmd.get("object_position_noise", 0.0))
                    except Exception:
                        object_position_noise = 0.0
                    target_objects = cmd.get("target_objects")
                    if num_episodes <= 0:
                        cmd["error"] = "num_episodes must be > 0"
                    elif steps_per_segment <= 0:
                        cmd["error"] = "steps_per_segment must be > 0"
                    elif episode_timeout_sec < 0:
                        cmd["error"] = "episode_timeout_sec must be >= 0"
                    else:
                        _collect_stop.clear()
                        now_ts = time.time()
                        _state["collecting"] = True
                        _state["collect_progress"] = {
                            "total": num_episodes,
                            "completed": 0,
                            "skill": skill,
                            "steps_per_segment": steps_per_segment,
                            "episode_timeout_sec": episode_timeout_sec,
                            "output_dir": output_dir,
                            "status": "running",
                            "scene_mode": scene_mode,
                            "target_objects": target_objects,
                            "started_at": now_ts,
                            "updated_at": now_ts,
                        }
                        _collect_request = {
                            "num_episodes": num_episodes,
                            "steps_per_segment": steps_per_segment,
                            "episode_timeout_sec": episode_timeout_sec,
                            "output_dir": output_dir,
                            "skill": skill,
                            "scene_mode": scene_mode,
                            "target_objects": target_objects,
                            "reset_mode": reset_mode,
                            "rounds_per_episode": rounds_per_episode,
                            "object_position_noise": object_position_noise,
                        }
                        cmd["result"] = {
                            "status": "started",
                            "num_episodes": num_episodes,
                            "steps_per_segment": steps_per_segment,
                            "episode_timeout_sec": episode_timeout_sec,
                            "output_dir": output_dir,
                            "skill": skill,
                            "scene_mode": scene_mode,
                            "target_objects": target_objects,
                            "reset_mode": reset_mode,
                            "rounds_per_episode": rounds_per_episode,
                            "object_position_noise": object_position_noise,
                        }
                        print(
                            f"[collect] queued main-thread collection: skill={skill}, "
                            f"scene_mode={scene_mode}, episodes={num_episodes}, "
                            f"steps_per_segment={steps_per_segment}, timeout={episode_timeout_sec}s, output={output_dir}"
                        )

            elif cmd_type == "replay_start":
                if _state["collecting"] or _collect_request is not None:
                    cmd["error"] = "Cannot replay while collecting"
                elif _replay_request is not None or _state.get("replaying"):
                    cmd["error"] = "Replay already running"
                else:
                    dataset_path = cmd["dataset_path"]
                    episode_index = int(cmd.get("episode_index", 0))
                    speed = float(cmd.get("speed", 1.0))
                    scene_usd_path = cmd.get("scene_usd_path", "")
                    camera_meta_path = str(cmd.get("camera_meta", "") or "").strip()

                    _replay_stop.clear()
                    _state["replaying"] = True
                    _state["replay_progress"] = {
                        "dataset_path": dataset_path,
                        "episode_index": episode_index,
                        "speed": speed,
                        "current_frame": 0,
                        "total_frames": 0,
                        "status": "loading",
                    }
                    _replay_request = {
                        "dataset_path": dataset_path,
                        "episode_index": episode_index,
                        "speed": speed,
                        "scene_usd_path": scene_usd_path,
                        "camera_meta": camera_meta_path,
                    }
                    cmd["result"] = {
                        "status": "started",
                        "dataset_path": dataset_path,
                        "episode_index": episode_index,
                        "speed": speed,
                    }
                    print(f"[replay] queued: dataset={dataset_path}, ep={episode_index}, speed={speed}")

            elif cmd_type == "replay_record":
                if _state["collecting"] or _collect_request is not None:
                    cmd["error"] = "Cannot record while collecting"
                elif _replay_request is not None or _state.get("replaying"):
                    cmd["error"] = "Replay already running"
                elif _replay_record_request is not None:
                    cmd["error"] = "Replay record already queued"
                else:
                    _replay_stop.clear()
                    _state["replaying"] = True
                    _state["replay_progress"] = {
                        "source_dataset": cmd["source_dataset"],
                        "output_dataset": cmd["output_dataset"],
                        "status": "loading",
                        "current_episode": 0,
                        "total_episodes": 0,
                        "current_frame": 0,
                        "total_frames": 0,
                    }
                    _replay_record_request = {
                        "source_dataset": cmd["source_dataset"],
                        "output_dataset": cmd["output_dataset"],
                        "episodes": cmd.get("episodes"),
                        "speed": float(cmd.get("speed", 1.0)),
                        "camera_meta": cmd.get("camera_meta"),
                    }
                    cmd["result"] = {"status": "started", "output_dataset": cmd["output_dataset"]}
                    print(f"[replay_record] queued: {cmd['source_dataset']} → {cmd['output_dataset']}")

            else:
                cmd["error"] = f"Unknown command: {cmd_type}"

        except Exception as exc:
            traceback.print_exc()
            cmd["error"] = str(exc)
        finally:
            cmd["event"].set()


def _run_pending_collection():
    """Run queued collection request on the main thread."""
    global _collect_request, _inf_dc, _inf_init_result
    if not _collect_request:
        return

    # Clear init_inference state to prevent MonitorPanel obs polling from
    # reading camera annotators during collection (causes segfault with cuRobo).
    if _inf_dc is not None:
        _inf_dc = None
        _inf_init_result = None
        print("[collect] cleared _inf_dc to avoid annotator conflict during collection")

    req = _collect_request
    _collect_request = None
    num_episodes = int(req.get("num_episodes", 10))
    steps_per_segment = int(req.get("steps_per_segment", os.environ.get("COLLECT_STEPS_PER_SEGMENT_DEFAULT", "70")))
    episode_timeout_sec = float(req.get("episode_timeout_sec", os.environ.get("COLLECT_EPISODE_TIMEOUT_SEC", "300")))
    skill = str(req.get("skill", "pick_place"))
    output_dir = _normalize_collect_output_dir(req.get("output_dir"), skill)
    scene_mode = str(req.get("scene_mode", "auto") or "auto")
    reset_mode = str(req.get("reset_mode", "full") or "full").strip().lower()
    rounds_per_episode = int(req.get("rounds_per_episode", 1))
    try:
        object_position_noise = float(req.get("object_position_noise", 0.0))
    except Exception:
        object_position_noise = 0.0
    target_objects = req.get("target_objects")
    scene_usd_path = None  # set early so finally can reference it

    try:
        # Collect can run after many scene-chat mutations. Recreate World to
        # clear stale scene registry wrappers (e.g. expired /World/Table
        # FixedCuboid handles) while preserving the currently opened USD stage.
        if not _recreate_world_for_open_stage("collect_start"):
            raise RuntimeError("Failed to recreate world before collection")
        if world is None:
            raise RuntimeError("World is unavailable before collection")

        os.makedirs(output_dir, exist_ok=True)

        # Auto-save current scene before collection starts
        scene_usd_path = None
        try:
            stage = _get_stage()
            if stage and stage.GetRootLayer():
                scene_usd_path = os.path.join(output_dir, "scene.usda")
                if stage.GetRootLayer().Export(scene_usd_path):
                    print(f"[collect] auto-saved scene: {scene_usd_path}")
                else:
                    scene_usd_path = None
                    print("[collect] WARNING: scene auto-save failed")
        except Exception as _save_exc:
            print(f"[collect] WARNING: scene auto-save error: {_save_exc}")

        print(
            f"[collect] start main-thread run: skill={skill}, scene_mode={scene_mode}, "
            f"episodes={num_episodes}, steps_per_segment={steps_per_segment}, "
            f"timeout={episode_timeout_sec}s, output={output_dir}, targets={target_objects}, "
            f"reset_mode={reset_mode}, rounds={rounds_per_episode}, obj_noise={object_position_noise}"
        )

        import importlib
        import lerobot_writer as _lrw_mod
        importlib.reload(_lrw_mod)

        # Select collector based on robot type in scene
        _active_robot_type = _state.get("active_robot_type", "").lower()
        if "openarm" in _active_robot_type:
            import openarm_pick_place_collector as _ipc_mod
            importlib.reload(_ipc_mod)
            from openarm_pick_place_collector import run_collection_in_process
            print(f"[collect] using OpenArm collector (robot_type={_active_robot_type})")
        else:
            import isaac_pick_place_collector as _ipc_mod
            importlib.reload(_ipc_mod)
            from isaac_pick_place_collector import run_collection_in_process
            print(f"[collect] using Franka collector (robot_type={_active_robot_type})")

        collect_kwargs = {
            "world": world,
            "simulation_app": simulation_app,
            "num_episodes": num_episodes,
            "output_dir": output_dir,
            "steps_per_segment": steps_per_segment,
            "stop_event": _collect_stop,
            "progress_callback": lambda ep: _state["collect_progress"].update({"completed": ep, "updated_at": time.time()}),
        }
        # Backward compatibility: older collector builds may not accept task_name.
        try:
            if "task_name" in inspect.signature(run_collection_in_process).parameters:
                collect_kwargs["task_name"] = skill
            if "scene_mode" in inspect.signature(run_collection_in_process).parameters:
                collect_kwargs["scene_mode"] = scene_mode
            if "target_objects" in inspect.signature(run_collection_in_process).parameters:
                collect_kwargs["target_objects"] = target_objects
            if "dataset_repo_id" in inspect.signature(run_collection_in_process).parameters:
                collect_kwargs["dataset_repo_id"] = f"local/{os.path.basename(output_dir.rstrip('/'))}"
            if "episode_timeout_sec" in inspect.signature(run_collection_in_process).parameters:
                collect_kwargs["episode_timeout_sec"] = episode_timeout_sec
            if "reset_mode" in inspect.signature(run_collection_in_process).parameters:
                collect_kwargs["reset_mode"] = reset_mode
            if "rounds_per_episode" in inspect.signature(run_collection_in_process).parameters:
                collect_kwargs["rounds_per_episode"] = rounds_per_episode
            if "object_position_noise" in inspect.signature(run_collection_in_process).parameters:
                collect_kwargs["object_position_noise"] = object_position_noise
        except Exception:
            pass

        result = run_collection_in_process(**collect_kwargs)

        # Inject scene_usd_path into dataset info.json
        if scene_usd_path:
            try:
                info_path = os.path.join(output_dir, "meta", "info.json")
                if os.path.exists(info_path):
                    with open(info_path) as f:
                        info = json.load(f)
                    info["scene_usd_path"] = scene_usd_path
                    with open(info_path, "w") as f:
                        json.dump(info, f, indent=2)
                    print(f"[collect] scene_usd_path written to info.json: {scene_usd_path}")
            except Exception as _info_exc:
                print(f"[collect] WARNING: failed to write scene to info.json: {_info_exc}")

        final_status = "stopped" if _collect_stop.is_set() else "done"
        try:
            completed_eps = int(result.get("completed", 0))
            success_eps = int(result.get("successful_episodes", completed_eps))
            if (not _collect_stop.is_set()) and completed_eps > 0 and success_eps < completed_eps:
                final_status = "done_with_failures"
        except Exception:
            pass
        _state["collect_progress"]["status"] = final_status
        _state["collect_progress"]["updated_at"] = time.time()
        _state["collect_progress"]["result"] = result
        print(
            f"[collect] finished: status={final_status}, "
            f"completed={_state['collect_progress'].get('completed', 0)}"
        )
    except Exception as exc:
        traceback.print_exc()
        _state["collect_progress"]["status"] = f"error: {exc}"
        _state["collect_progress"]["updated_at"] = time.time()
        print(f"[collect] ERROR: {exc}")
    finally:
        _state["collecting"] = False
        # Restore scene + NVCF streaming after collection (world.reset breaks it)
        try:
            _recreate_world_for_open_stage("collect_end")
            # Reload the auto-saved scene if available
            _saved_scene = scene_usd_path if scene_usd_path and os.path.exists(scene_usd_path) else None
            if not _saved_scene:
                _saved_scene = _state.get("scene", "")
            if _saved_scene and os.path.exists(_saved_scene):
                ctx = omni.usd.get_context()
                _ok = ctx.open_stage(_saved_scene)
                if _ok:
                    for _ in range(300):
                        simulation_app.update()
                        if ctx.get_stage_state() == omni.usd.StageState.OPENED:
                            break
                        time.sleep(0.05)
                    _recreate_world_for_open_stage("collect_scene_restore")
                    print(f"[collect] scene restored: {_saved_scene}")
            # Restore NVCF streaming readiness
            try:
                import omni.services.livestream.nvcf.services.api as _nvcf_api
                _nvcf_api.app_ready = True
                _nvcf_api.rtx_ready = True
                print("[collect] NVCF streaming restored")
            except Exception:
                pass
            # Pump a few frames to re-establish rendering
            for _ in range(10):
                simulation_app.update()
        except Exception as _restore_exc:
            print(f"[collect] WARNING: post-collection restore failed: {_restore_exc}")


def _run_pending_replay():
    """Run queued replay request on the main thread."""
    global _replay_request, PHYSICS_RUNNING, _inf_dc, _inf_init_result
    if not _replay_request:
        return

    req = _replay_request
    _replay_request = None
    dataset_path = req["dataset_path"]
    episode_index = int(req.get("episode_index", 0))
    speed = float(req.get("speed", 1.0))
    scene_usd_path = req.get("scene_usd_path", "")
    camera_meta_path = str(req.get("camera_meta", "") or "").strip()
    replay_cameras = {}  # {cam_name: (render_product, annotator)}

    # Invalidate monitor's cached articulation — scene reload makes it stale
    _inf_dc = None
    _inf_init_result = None

    try:
        import base64 as _b64
        import io as _obs_io
        import pandas as pd
        import numpy as np
        import yaml as pyyaml
        from PIL import Image as _PILImage

        # Reload scene to reset object positions (cube, bowl, etc.)
        if scene_usd_path and os.path.exists(scene_usd_path):
            print(f"[replay] reloading scene: {scene_usd_path}")
            ctx = omni.usd.get_context()
            ok = ctx.open_stage(scene_usd_path)
            if ok:
                for _ in range(300):
                    simulation_app.update()
                    if ctx.get_stage_state() == omni.usd.StageState.OPENED:
                        break
                    time.sleep(0.05)
                _recreate_world_for_open_stage("replay_scene_reload")
                PHYSICS_RUNNING = False
                _state["physics"] = False
                print(f"[replay] scene reloaded successfully")
            else:
                print(f"[replay] WARNING: scene reload failed, continuing with current state")

        # Find parquet file
        meta_path = os.path.join(dataset_path, "meta", "info.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Dataset meta not found: {meta_path}")

        with open(meta_path) as f:
            meta = json.load(f)

        fps = meta.get("fps", 30)
        chunks_size = meta.get("chunks_size", 1000)
        data_path_template = meta.get("data_path", "data")
        features = meta.get("features", {})

        # Extract joint names from features
        state_feature = features.get("observation.state", {})
        raw_names = state_feature.get("names", [])
        # LeRobot v3: names can be {"motors": [...]} or flat list
        if isinstance(raw_names, dict):
            joint_names = list(raw_names.get("motors", raw_names.get(list(raw_names.keys())[0], []))) if raw_names else []
        else:
            joint_names = list(raw_names) if raw_names else []

        chunk_index = episode_index // chunks_size
        if "{" in data_path_template:
            parquet_rel = data_path_template.format(chunk_index=chunk_index, file_index=0)
        else:
            parquet_rel = os.path.join(data_path_template, f"chunk-{chunk_index:03d}", "file-000.parquet")

        parquet_path = os.path.join(dataset_path, parquet_rel)
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        if "episode_index" in df.columns:
            df = df[df["episode_index"] == episode_index]

        if len(df) == 0:
            raise ValueError(f"Episode {episode_index} not found in {parquet_path}")

        # Get state arrays
        states = None
        if "observation.state" in df.columns:
            states = np.stack(df["observation.state"].values)
        elif "action" in df.columns:
            states = np.stack(df["action"].values)

        if states is None or len(states) == 0:
            raise ValueError("No observation.state or action data found")

        total_frames = len(states)
        print(f"[replay] loaded {total_frames} frames from ep {episode_index}, fps={fps}, joints={joint_names[:5]}...")

        _state["replay_progress"]["total_frames"] = total_frames
        _state["replay_progress"]["status"] = "playing"

        # Find robot articulation
        stage = _get_stage()
        if stage is None:
            raise RuntimeError("No USD stage")

        # Locate robot prims (look for articulation roots)
        from pxr import UsdPhysics
        robot_prims = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                robot_prims.append(prim)

        if not robot_prims:
            raise RuntimeError("No robot (ArticulationRootAPI) found in scene")

        # Ensure physics is running — Articulation.initialize() needs active physics
        if world is not None and not PHYSICS_RUNNING:
            print("[replay] starting physics for articulation init...")
            world.play()
            PHYSICS_RUNNING = True
            _state["physics"] = True
            # Step once so physics engine registers the articulation
            world.step(render=True)

        Articulation = _Articulation  # compat
        robot = Articulation(robot_prims[0].GetPath().pathString)
        robot.initialize()

        dof_names = list(robot.dof_names)  # ensure plain Python list
        print(f"[replay] robot DOFs: {dof_names}")

        # Build index mapping for OpenArm bimanual robot
        # Dataset records 8 joints: openarm_joint1-7 + openarm_finger_joint1
        # Sim right arm DOFs have different internal ordering
        # data_to_sim[sim_j] = data_j: for sim joint sim_j, read data column data_j
        data_to_sim = [0, 5, 1, 2, 3, 4, 6]  # joint reordering from all replay scripts
        negate_joint = {6}  # sim joint 6 (wrist) needs sign flip

        # Find right arm joint DOF indices in the articulation
        right_arm_idx = [i for i, n in enumerate(dof_names) if 'right' in n and 'joint' in n and 'finger' not in n]
        right_arm_idx.sort(key=lambda i: dof_names[i])  # sort by name → joint1, joint2, ...
        right_finger_idx = [i for i, n in enumerate(dof_names) if 'right' in n and 'finger' in n]

        # Split mapping: arm (PD) vs finger (kinematic)
        dof_map_arm = []
        dof_map_finger = []
        for sim_j, data_j in enumerate(data_to_sim):
            if sim_j < len(right_arm_idx) and data_j < states.shape[1]:
                dof_map_arm.append((data_j, right_arm_idx[sim_j], sim_j in negate_joint))
        for fidx in right_finger_idx:
            if states.shape[1] > 7:
                dof_map_finger.append((7, fidx, False))
        dof_map = dof_map_arm + dof_map_finger

        # Tune PD gains: arm needs damping to prevent oscillation,
        # finger needs high stiffness+damping for firm physical grip via friction
        ctrl = robot.get_articulation_controller()
        stiffness, damping = ctrl.get_gains()
        # Arm: add 10% damping to prevent oscillation
        damping = np.where(stiffness > 0, stiffness * 0.1, damping)
        # Finger: moderate stiffness for contact-aware grip (not too high = won't push cube away)
        for fidx in right_finger_idx:
            stiffness[fidx] = 100000.0   # firm but not aggressive
            damping[fidx] = 10000.0      # enough to prevent bounce
        ctrl.set_gains(stiffness, damping)
        arm_gain_dbg = (
            f"stiff={stiffness[right_arm_idx[0]]:.0f} damp={damping[right_arm_idx[0]]:.0f}"
            if right_arm_idx
            else "n/a"
        )
        finger_gain_dbg = (
            f"stiff={stiffness[right_finger_idx[0]]:.0f} damp={damping[right_finger_idx[0]]:.0f}"
            if right_finger_idx
            else "n/a"
        )
        print(f"[replay] PD gains: arm {arm_gain_dbg}, finger {finger_gain_dbg}")

        if not dof_map:
            print(f"[replay] WARNING: no DOF mapping found. joint_names={joint_names}, dof_names={dof_names}")
        else:
            jn = joint_names or [f"data[{i}]" for i in range(states.shape[1])]
            print(f"[replay] DOF mapping ({len(dof_map)} joints): " +
                  ", ".join(f"{jn[s] if s < len(jn) else f'data[{s}]'}→{dof_names[d]}{'(neg)' if neg else ''}" for s, d, neg in dof_map))

        # Diagnostic: check collision on finger and cube prims
        from pxr import UsdGeom, UsdShade, UsdPhysics
        # List prim tree to find collision geometry on robot
        # Find the openarm reference prim
        robot_path = robot_prims[0].GetPath().pathString
        # Go up to the openarm root (e.g. /World/OpenArm)
        parts = robot_path.split('/')
        openarm_root = '/'.join(parts[:3]) if len(parts) >= 3 else robot_path
        print(f"[replay] scanning prims under: {openarm_root}")
        prim_count = 0
        for p in stage.Traverse():
            path_str = str(p.GetPath())
            if not path_str.startswith(openarm_root):
                continue
            tp = p.GetTypeName()
            has_col = p.HasAPI(UsdPhysics.CollisionAPI)
            is_geom = p.IsA(UsdGeom.Gprim)
            depth = path_str[len(openarm_root):].count('/')
            # Show: all geometry, all collision, all joint-related, depth<=2
            if is_geom or has_col or 'Joint' in tp or depth <= 2:
                marker = "COL" if has_col else ("GEO" if is_geom else "   ")
                print(f"  {marker} {'  '*min(depth,4)}{p.GetName()} [{tp}]")
                prim_count += 1
                if prim_count > 80:
                    print("  ... (truncated)")
                    break
        print(f"[replay] total prims shown: {prim_count}")

        # Find cube and bowl prims for position tracking
        cube_prim = bowl_prim = None
        for p in stage.Traverse():
            name_lower = p.GetName().lower()
            if 'cube' in name_lower or 'cuboid' in name_lower:
                cube_prim = p
            elif 'bowl' in name_lower:
                bowl_prim = p
        if cube_prim:
            print(f"[replay] tracking cube: {cube_prim.GetPath()}")
        if bowl_prim:
            print(f"[replay] tracking bowl: {bowl_prim.GetPath()}")
            # Fix bowl collision: convex hull seals the concave opening.
            # Set child meshes to triangle mesh collision so cube can fall inside.
            for child in Usd.PrimRange(bowl_prim):
                if child.IsA(UsdGeom.Mesh):
                    if not child.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI.Apply(child)
                    if not child.HasAPI(UsdPhysics.MeshCollisionAPI):
                        UsdPhysics.MeshCollisionAPI.Apply(child)
                    child.GetAttribute("physics:approximation").Set("none")
            print(f"[replay] fixed bowl collision to triangle mesh")

        # Add very high-friction physics material to cube for firm grip
        if cube_prim:
            mat_path = str(cube_prim.GetPath()) + "/GripMaterial"
            mat_prim = stage.DefinePrim(mat_path, "Material")
            UsdPhysics.MaterialAPI.Apply(mat_prim)
            phys_mat = UsdPhysics.MaterialAPI(mat_prim)
            phys_mat.CreateStaticFrictionAttr(10.0)   # extreme friction for reliable grip
            phys_mat.CreateDynamicFrictionAttr(10.0)
            phys_mat.CreateRestitutionAttr(0.0)
            # Bind material to cube
            binding = UsdShade.MaterialBindingAPI.Apply(cube_prim)
            binding.Bind(UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")
            print(f"[replay] applied friction material to cube: static=10.0 dynamic=10.0")
            # Also reduce cube mass for easier grip
            if cube_prim.HasAPI(UsdPhysics.MassAPI):
                UsdPhysics.MassAPI(cube_prim).CreateMassAttr(0.01)  # 10g — very light
                print(f"[replay] reduced cube mass to 0.01 kg")

        # Create a scene-level physics material with high friction
        # (Can't modify instanceable robot prims directly)
        # PhysX uses max(friction_A, friction_B) by default for contact pairs
        # So high friction on cube alone should work for grip
        print("[replay] finger collision check:")
        for p in stage.Traverse():
            path_str = str(p.GetPath())
            if ('finger' in path_str.lower() or 'hand' in path_str.lower()) and 'collision' in path_str.lower():
                has_col = p.HasAPI(UsdPhysics.CollisionAPI)
                print(f"  {p.GetPath()} col={has_col} type={p.GetTypeName()}")

        # Setup wrist camera render products + annotators for replay MonitorPanel stream.
        camera_meta = {}
        camera_meta_candidates = []
        if camera_meta_path:
            camera_meta_candidates.append(camera_meta_path)
        camera_meta_candidates.extend(
            [
                os.path.join(os.path.dirname(__file__), "camera_meta.yaml"),
                "/data/embodied/asset/robots/openarm_bimanual/camera_meta.yaml",
            ]
        )
        camera_meta_used = None
        for candidate in camera_meta_candidates:
            if not candidate or not os.path.exists(candidate):
                continue
            try:
                with open(candidate) as f:
                    camera_meta = pyyaml.safe_load(f) or {}
                camera_meta_used = candidate
                break
            except Exception as meta_exc:
                print(f"[replay] WARNING: failed to load camera meta {candidate}: {meta_exc}")

        if camera_meta_used:
            print(f"[replay] camera_meta: {camera_meta_used}")
        else:
            print("[replay] WARNING: camera_meta.yaml not found, replay images disabled")

        cam_configs = {}
        if isinstance(camera_meta, dict):
            cam_configs = camera_meta.get("wrist_cameras") or camera_meta.get("cameras") or {}

        if cam_configs:
            import omni.replicator.core as rep

            for cam_name, cam_cfg in cam_configs.items():
                if not isinstance(cam_cfg, dict):
                    continue
                if "wrist" not in str(cam_name).lower():
                    continue

                mount_link = str(cam_cfg.get("mount_link", "") or "").strip()
                prim_name = str(cam_cfg.get("prim_name", "wrist_cam") or "wrist_cam").strip()
                cam_resolution_raw = cam_cfg.get("resolution", [640, 480])
                try:
                    # Use half resolution for replay monitoring (full res too slow)
                    cam_resolution = (
                        int(cam_resolution_raw[0]) // 2,
                        int(cam_resolution_raw[1]) // 2,
                    )
                except Exception:
                    cam_resolution = (320, 240)

                mount_prim = None
                if mount_link:
                    if mount_link.startswith("/"):
                        p = stage.GetPrimAtPath(mount_link)
                        if p.IsValid():
                            mount_prim = p
                    if mount_prim is None:
                        for p in stage.Traverse():
                            p_path = p.GetPath().pathString
                            if p.GetName() == mount_link or p_path.endswith("/" + mount_link):
                                mount_prim = p
                                break

                cam_prim_path = None
                if mount_prim is not None:
                    existing_cam = mount_prim.GetPath().pathString + f"/{prim_name}"
                    fallback_cam = mount_prim.GetPath().pathString + f"/{cam_name}"
                    if stage.GetPrimAtPath(existing_cam).IsValid():
                        cam_prim_path = existing_cam
                    elif stage.GetPrimAtPath(fallback_cam).IsValid():
                        cam_prim_path = fallback_cam

                if cam_prim_path is None:
                    for key in ("camera_prim", "camera_path", "prim_path", "path"):
                        candidate = str(cam_cfg.get(key, "") or "").strip()
                        if candidate and stage.GetPrimAtPath(candidate).IsValid():
                            cam_prim_path = candidate
                            break

                if cam_prim_path is None:
                    print(f"[replay] WARNING: wrist camera '{cam_name}' prim not found (mount_link={mount_link})")
                    continue

                try:
                    rp = rep.create.render_product(cam_prim_path, cam_resolution)
                    annot = rep.AnnotatorRegistry.get_annotator("rgb")
                    annot.attach([rp])
                    replay_cameras[cam_name] = (rp, annot)
                    print(f"[replay] camera attached: {cam_name} -> {cam_prim_path} @ {cam_resolution}")
                except Exception as cam_exc:
                    print(f"[replay] WARNING: camera setup failed for {cam_name}: {cam_exc}")

            if replay_cameras:
                # Warm-up frames so annotators start returning valid data.
                for _ in range(3):
                    if world and PHYSICS_RUNNING:
                        world.step(render=True)
                    else:
                        simulation_app.update()
        else:
            print("[replay] WARNING: no wrist camera configs found in camera_meta")

        # Replay loop — all PD control with high finger gains for physical grip
        ArticulationAction = _ArticulationAction  # compat
        frame_interval = 1.0 / (fps * speed) if fps > 0 and speed > 0 else 1.0 / 30.0
        log_every = max(1, int(fps))  # log every 1 second
        finger_contact = {}

        # Seed from an observed post-step state to avoid reading articulation before stepping.
        if world and PHYSICS_RUNNING:
            world.step(render=True)
        else:
            simulation_app.update()
        last_actual_positions = robot.get_joint_positions()
        if last_actual_positions is None:
            last_actual_positions = np.zeros(len(dof_names), dtype=np.float32)
        else:
            last_actual_positions = np.array(last_actual_positions, dtype=np.float32, copy=True)

        for frame_idx in range(total_frames):
            if _replay_stop.is_set():
                print(f"[replay] stopped at frame {frame_idx}/{total_frames}")
                break

            state_row = states[frame_idx]
            targets = np.array(last_actual_positions, dtype=np.float32, copy=True)
            for src_idx, dst_idx, negate in dof_map:
                if src_idx < len(state_row):
                    val = float(state_row[src_idx])
                    if negate:
                        val = -val
                    targets[dst_idx] = val

            # Contact-aware finger control:
            # When dataset says close (target < 0.03), gradually close finger.
            # If actual finger stops moving (hit object), hold at contact width + squeeze.
            for fidx in right_finger_idx:
                data_tgt = targets[fidx]
                if data_tgt < 0.03:  # dataset says grip
                    actual = float(last_actual_positions[fidx])
                    prev = finger_contact.get(fidx)
                    if prev is not None:
                        # Contact detection: actual stopped decreasing despite target decreasing
                        if actual > prev - 0.0005 and actual > data_tgt + 0.005:
                            # Hit something — hold at contact width, squeeze 2mm for grip
                            targets[fidx] = max(0.0, actual - 0.002)
                        else:
                            targets[fidx] = data_tgt  # still closing freely
                    else:
                        targets[fidx] = data_tgt  # first frame of closing
                    finger_contact[fidx] = actual
                else:
                    # Open — clear contact state
                    finger_contact.pop(fidx, None)

            action = ArticulationAction(joint_positions=np.array(targets))
            ctrl.apply_action(action)

            # Step sim to render
            if world and PHYSICS_RUNNING:
                world.step(render=True)
            else:
                simulation_app.update()

            _state["replay_progress"]["current_frame"] = frame_idx + 1

            # Read updated articulation state and camera frames after stepping.
            try:
                actual_pos = robot.get_joint_positions()
                if actual_pos is not None:
                    last_actual_positions = np.array(actual_pos, dtype=np.float32, copy=True)
            except Exception:
                pass

            # Camera: sample every 5 frames to reduce overhead (joints every frame)
            if replay_cameras and frame_idx % 5 == 0:
                _replay_images = {}
                for cam_name, (_rp, annot) in replay_cameras.items():
                    try:
                        rgba = annot.get_data()
                        if rgba is None:
                            continue
                        rgba = np.asarray(rgba)
                        if rgba.ndim != 3 or rgba.shape[-1] < 3 or rgba.size == 0:
                            continue
                        rgb = rgba[:, :, :3]
                        if rgb.dtype != np.uint8:
                            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                        img = _PILImage.fromarray(rgb)
                        buf = _obs_io.BytesIO()
                        img.save(buf, format="JPEG", quality=60)
                        _replay_images[cam_name] = _b64.b64encode(buf.getvalue()).decode("ascii")
                    except Exception as img_exc:
                        print(f"[replay] WARNING: image capture failed for {cam_name}: {img_exc}")
                # Store latest images for non-camera frames
                _state["_replay_latest_images"] = _replay_images

            # Update replay_obs for MonitorPanel (reads from Flask thread).
            try:
                _state["replay_obs"] = {
                    "positions": last_actual_positions.tolist() if last_actual_positions is not None else [],
                    "names": dof_names,
                    "images": _state.get("_replay_latest_images", {}),
                    "timestamp": time.time(),
                }
            except Exception:
                pass

            # Log cube/bowl positions + actual finger pos
            # Dense logging (every 5 frames) during grip phase (frame 230-350), else every second
            in_grip_phase = 230 <= frame_idx <= 350
            should_log = (in_grip_phase and frame_idx % 5 == 0) or (not in_grip_phase and frame_idx % log_every == 0)
            if should_log:
                parts = []
                finger_val = float(state_row[7]) if states.shape[1] > 7 else -1
                actual_finger = float(last_actual_positions[right_finger_idx[0]]) if right_finger_idx else -1
                if cube_prim:
                    xf = UsdGeom.Xformable(cube_prim)
                    mat = xf.ComputeLocalToWorldTransform(0)
                    cp = mat.ExtractTranslation()
                    parts.append(f"cube=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f})")
                if bowl_prim:
                    xf = UsdGeom.Xformable(bowl_prim)
                    mat = xf.ComputeLocalToWorldTransform(0)
                    bp = mat.ExtractTranslation()
                    parts.append(f"bowl=({bp[0]:.3f},{bp[1]:.3f},{bp[2]:.3f})")
                parts.append(f"finger_tgt={finger_val:.4f} actual={actual_finger:.4f}")
                print(f"[replay] f={frame_idx}/{total_frames} {' '.join(parts)}")

            # Throttle to match dataset FPS
            time.sleep(max(0, frame_interval - 0.001))

        final_status = "stopped" if _replay_stop.is_set() else "done"
        _state["replay_progress"]["status"] = final_status
        print(f"[replay] finished: {final_status}, frames={_state['replay_progress']['current_frame']}/{total_frames}")

    except Exception as exc:
        traceback.print_exc()
        _state["replay_progress"]["status"] = f"error: {exc}"
        print(f"[replay] ERROR: {exc}")
    finally:
        for cam_name, (rp, annot) in replay_cameras.items():
            try:
                annot.detach([rp])
            except Exception:
                pass
            try:
                rp.destroy()
            except Exception:
                pass
        if replay_cameras:
            print(f"[replay] cleaned up {len(replay_cameras)} replay camera render products")
        _state["replaying"] = False
        _state.pop("replay_obs", None)
        _state.pop("_replay_latest_images", None)
        # Clear monitor's cached articulation so MonitorPanel re-inits cleanly
        _inf_dc = None
        _inf_init_result = None
        print("[replay] cleared _inf_dc for monitor re-init")


def _run_pending_replay_record():
    """Replay all episodes with wrist camera recording → new LeRobot v3 dataset."""
    global _replay_record_request, PHYSICS_RUNNING
    if not _replay_record_request:
        return

    req = _replay_record_request
    _replay_record_request = None
    source_dataset = req["source_dataset"]
    output_dataset = req["output_dataset"]
    episode_list = req.get("episodes")
    speed = float(req.get("speed", 1.0))
    camera_meta_path = req.get("camera_meta")

    try:
        import numpy as np
        import pandas as pd
        import yaml as pyyaml

        # ── Load camera meta ──
        with open(camera_meta_path) as f:
            camera_meta = pyyaml.safe_load(f)
        cam_configs = camera_meta.get("wrist_cameras", {})
        camera_names = list(cam_configs.keys())
        first_cam = cam_configs[camera_names[0]]
        cam_resolution = tuple(first_cam.get("resolution", [640, 480]))
        print(f"[replay_record] cameras: {camera_names}, resolution: {cam_resolution}")

        # ── Load source dataset meta ──
        meta_path = os.path.join(source_dataset, "meta", "info.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Source dataset meta not found: {meta_path}")

        with open(meta_path) as f:
            meta = json.load(f)

        fps = meta.get("fps", 30)
        chunks_size = meta.get("chunks_size", 1000)
        data_path_template = meta.get("data_path", "data")
        features = meta.get("features", {})
        total_episodes = meta.get("total_episodes", 0)

        # Determine episodes to process
        if episode_list is not None:
            episodes = sorted(episode_list)
        else:
            episodes = list(range(total_episodes))
        if not episodes:
            # Fallback: scan parquet for episode indices
            chunk_0 = os.path.join(source_dataset, "data", "chunk-000", "file-000.parquet")
            if os.path.exists(chunk_0):
                df_all = pd.read_parquet(chunk_0)
                if "episode_index" in df_all.columns:
                    episodes = sorted(df_all["episode_index"].unique().tolist())

        print(f"[replay_record] {len(episodes)} episodes to record from {source_dataset}")
        _state["replay_progress"]["total_episodes"] = len(episodes)

        # Extract joint names
        state_feature = features.get("observation.state", {})
        raw_names = state_feature.get("names", [])
        if isinstance(raw_names, dict):
            joint_names = list(raw_names.get("motors", list(raw_names.values())[0] if raw_names else []))
        else:
            joint_names = list(raw_names) if raw_names else []
        state_dim = state_feature.get("shape", [8])[0] if state_feature else 8

        # ── Setup robot ──
        from pxr import UsdPhysics, UsdGeom, UsdShade
        stage = _get_stage()
        if stage is None:
            raise RuntimeError("No USD stage")

        robot_prims = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                robot_prims.append(prim)
        if not robot_prims:
            raise RuntimeError("No robot (ArticulationRootAPI) found in scene")

        if world is not None and not PHYSICS_RUNNING:
            print("[replay_record] starting physics...")
            world.play()
            PHYSICS_RUNNING = True
            _state["physics"] = True
            world.step(render=True)

        Articulation = _Articulation  # compat
        robot = Articulation(robot_prims[0].GetPath().pathString)
        robot.initialize()
        dof_names = list(robot.dof_names)
        print(f"[replay_record] robot DOFs: {dof_names}")

        # DOF mapping (same as replay)
        data_to_sim = [0, 5, 1, 2, 3, 4, 6]
        negate_joint = {6}
        right_arm_idx = sorted(
            [i for i, n in enumerate(dof_names) if 'right' in n and 'joint' in n and 'finger' not in n],
            key=lambda i: dof_names[i]
        )
        right_finger_idx = [i for i, n in enumerate(dof_names) if 'right' in n and 'finger' in n]

        dof_map_arm = []
        for sim_j, data_j in enumerate(data_to_sim):
            if sim_j < len(right_arm_idx):
                dof_map_arm.append((data_j, right_arm_idx[sim_j], sim_j in negate_joint))
        dof_map_finger = []
        for fidx in right_finger_idx:
            dof_map_finger.append((7, fidx, False))
        dof_map = dof_map_arm + dof_map_finger

        # PD gains (same as replay)
        ctrl = robot.get_articulation_controller()
        stiffness, damping = ctrl.get_gains()
        damping = np.where(stiffness > 0, stiffness * 0.1, damping)
        for fidx in right_finger_idx:
            stiffness[fidx] = 100000.0
            damping[fidx] = 10000.0
        ctrl.set_gains(stiffness, damping)

        # Friction on cube (same as replay)
        cube_prim = None
        for p in stage.Traverse():
            if 'cube' in p.GetName().lower() or 'cuboid' in p.GetName().lower():
                cube_prim = p
                break
        if cube_prim:
            mat_path = str(cube_prim.GetPath()) + "/GripMaterial"
            mat_prim = stage.DefinePrim(mat_path, "Material")
            UsdPhysics.MaterialAPI.Apply(mat_prim)
            phys_mat = UsdPhysics.MaterialAPI(mat_prim)
            phys_mat.CreateStaticFrictionAttr(10.0)
            phys_mat.CreateDynamicFrictionAttr(10.0)
            phys_mat.CreateRestitutionAttr(0.0)
            binding = UsdShade.MaterialBindingAPI.Apply(cube_prim)
            binding.Bind(UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")
            if cube_prim.HasAPI(UsdPhysics.MassAPI):
                UsdPhysics.MassAPI(cube_prim).CreateMassAttr(0.01)

        # ── Setup camera render products + annotators ──
        import omni.replicator.core as rep

        annotators = {}
        render_products = []
        robot_path = robot_prims[0].GetPath().pathString
        parts = robot_path.split('/')
        openarm_root = '/'.join(parts[:3]) if len(parts) >= 3 else robot_path

        for cam_name, cam_cfg in cam_configs.items():
            mount_link = cam_cfg["mount_link"]
            prim_name = cam_cfg.get("prim_name", "wrist_cam")
            cam_prim_path = f"{openarm_root}/{mount_link}/{prim_name}"

            # Verify camera prim exists
            cam_prim = stage.GetPrimAtPath(cam_prim_path)
            if not cam_prim.IsValid():
                print(f"[replay_record] WARNING: camera not found at {cam_prim_path}, creating...")
                from pxr import Gf
                cam = UsdGeom.Camera.Define(stage, cam_prim_path)
                xf = UsdGeom.Xformable(cam.GetPrim())
                t = cam_cfg["translate"]
                xf.AddTranslateOp().Set(Gf.Vec3d(*t))
                q = cam_cfg["orient_quat_xyzw"]
                xf.AddOrientOp().Set(Gf.Quatf(q[3], q[0], q[1], q[2]))  # w,x,y,z
                cam.GetFocalLengthAttr().Set(cam_cfg.get("focal_length", 24.0))
                clip = cam_cfg.get("clipping_range", [0.01, 2.0])
                cam.GetClippingRangeAttr().Set(Gf.Vec2f(*clip))

            rp = rep.create.render_product(cam_prim_path, cam_resolution)
            render_products.append(rp)

            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp])
            annotators[cam_name] = ann
            print(f"[replay_record] attached annotator to {cam_prim_path}")

        # Warm up: step a few frames so annotators initialize
        for _ in range(3):
            world.step(render=True)

        # ── Setup writer ──
        from lerobot_writer import SimLeRobotWriter
        writer = SimLeRobotWriter(
            output_dir=output_dataset,
            repo_id=f"local/{os.path.basename(output_dataset)}",
            fps=fps,
            robot_type="openarm",
            state_dim=state_dim,
            action_dim=state_dim,
            camera_names=camera_names,
            camera_resolution=(cam_resolution[1], cam_resolution[0]),  # (h, w)
            state_names=joint_names if joint_names else None,
            action_names=joint_names if joint_names else None,
        )

        # ── Replay each episode ──
        ArticulationAction = _ArticulationAction  # compat
        finger_contact = {}
        global_frame = 0

        for ep_seq, episode_index in enumerate(episodes):
            if _replay_stop.is_set():
                print(f"[replay_record] stopped before episode {episode_index}")
                break

            _state["replay_progress"]["current_episode"] = ep_seq
            _state["replay_progress"]["status"] = f"recording episode {episode_index}"

            # Load episode data
            chunk_index = episode_index // chunks_size
            if "{" in data_path_template:
                parquet_rel = data_path_template.format(chunk_index=chunk_index, file_index=0)
            else:
                parquet_rel = os.path.join(data_path_template, f"chunk-{chunk_index:03d}", "file-000.parquet")

            parquet_path = os.path.join(source_dataset, parquet_rel)
            df = pd.read_parquet(parquet_path)
            if "episode_index" in df.columns:
                df = df[df["episode_index"] == episode_index]
            if len(df) == 0:
                print(f"[replay_record] episode {episode_index} empty, skipping")
                continue

            states = None
            if "observation.state" in df.columns:
                states = np.stack(df["observation.state"].values)
            elif "action" in df.columns:
                states = np.stack(df["action"].values)
            if states is None or len(states) == 0:
                print(f"[replay_record] episode {episode_index} no state data, skipping")
                continue

            total_frames = len(states)
            _state["replay_progress"]["total_frames"] = total_frames
            _state["replay_progress"]["current_frame"] = 0

            # Reset cube position if possible (reload scene state)
            # For now we rely on the scene being in initial state

            finger_contact.clear()

            for frame_idx in range(total_frames):
                if _replay_stop.is_set():
                    break

                state_row = states[frame_idx]
                targets = robot.get_joint_positions().copy()
                for src_idx, dst_idx, negate in dof_map:
                    if src_idx < len(state_row):
                        val = float(state_row[src_idx])
                        if negate:
                            val = -val
                        targets[dst_idx] = val

                # Contact-aware finger control (same as replay)
                for fidx in right_finger_idx:
                    data_tgt = targets[fidx]
                    if data_tgt < 0.03:
                        actual = float(robot.get_joint_positions()[fidx])
                        prev = finger_contact.get(fidx)
                        if prev is not None:
                            if actual > prev - 0.0005 and actual > data_tgt + 0.005:
                                targets[fidx] = max(0.0, actual - 0.002)
                            else:
                                targets[fidx] = data_tgt
                        else:
                            targets[fidx] = data_tgt
                        finger_contact[fidx] = actual
                    else:
                        finger_contact.pop(fidx, None)

                action = ArticulationAction(joint_positions=np.array(targets))
                ctrl.apply_action(action)
                world.step(render=True)

                # ── Capture camera frames ──
                for cam_name, ann in annotators.items():
                    rgb_data = ann.get_data()
                    if rgb_data is not None:
                        if rgb_data.ndim == 3 and rgb_data.shape[-1] == 4:
                            rgb_data = rgb_data[..., :3]  # RGBA → RGB
                        writer.add_video_frame(cam_name, rgb_data)

                # ── Write tabular data ──
                obs_state = state_row.astype(np.float32)
                # Action = state of next frame (or same for last frame)
                if frame_idx + 1 < total_frames:
                    action_row = states[frame_idx + 1].astype(np.float32)
                else:
                    action_row = obs_state.copy()

                writer.add_frame(
                    episode_index=ep_seq,
                    frame_index=frame_idx,
                    observation_state=obs_state,
                    action=action_row,
                    timestamp=frame_idx / fps,
                    next_done=(frame_idx == total_frames - 1),
                )

                _state["replay_progress"]["current_frame"] = frame_idx + 1

                if frame_idx % fps == 0:
                    print(f"[replay_record] ep={episode_index} f={frame_idx}/{total_frames}")

            # Finish episode
            writer.finish_episode(
                episode_index=ep_seq,
                length=total_frames,
                task="pick_and_place",
                success=True,
            )
            print(f"[replay_record] episode {episode_index} done ({total_frames} frames)")

        # ── Finalize dataset ──
        writer.finalize()

        # Cleanup render products
        for rp in render_products:
            rp.destroy()

        final_status = "stopped" if _replay_stop.is_set() else "done"
        _state["replay_progress"]["status"] = final_status
        print(f"[replay_record] {final_status}: {len(episodes)} episodes → {output_dataset}")

    except Exception as exc:
        traceback.print_exc()
        _state["replay_progress"]["status"] = f"error: {exc}"
        print(f"[replay_record] ERROR: {exc}")
    finally:
        _state["replaying"] = False


def _apply_emergency_stop_if_requested():
    """Main-thread side effects for emergency stop (Isaac API calls)."""
    global PHYSICS_RUNNING
    if not _estop_event.is_set():
        return
    try:
        if world is not None:
            world.pause()
    except Exception as exc:
        print(f"[estop] WARNING: world.pause() failed: {exc}")
    PHYSICS_RUNNING = False
    _state["physics"] = False
    _state["collecting"] = False
    _estop_event.clear()
    print(f"[estop] Applied emergency stop (reason={_state.get('last_estop_reason')})")


def _run_deferred_inference_init():
    """Phased inference init — runs one phase per main-loop frame to avoid blocking.

    Phases:
      0 → start physics if needed, step a few frames
      1 → find articulation + initialize
      2 → setup cameras
      3 → warm up render products
      4 → done
    """
    global _inf_init_phase, _inf_init_pending, _inf_init_result, _inf_init_error
    global _inf_init_step_count, _inf_robot_prim, _inf_dc, _inf_cameras
    global PHYSICS_RUNNING

    _inf_init_step_count += 1

    try:
        if _inf_init_phase == 0:
            # Phase 0: ensure physics is playing + reset to create physics views
            print("[init_inference] Phase 0: starting physics...")
            stage = _get_stage()
            if stage:
                _sanitize_franka_root_rigidbody(stage)
            if not PHYSICS_RUNNING:
                world.play()
                PHYSICS_RUNNING = True
                _state["physics"] = True
            # Reset world to create fresh physics simulation views
            try:
                world.reset()
                print("[init_inference] world.reset() done")
                # Restore NVCF streaming readiness (world.reset can break it)
                try:
                    import omni.services.livestream.nvcf.services.api as _nvcf_api
                    _nvcf_api.app_ready = True
                    _nvcf_api.rtx_ready = True
                    print("[init_inference] NVCF readiness restored")
                except Exception:
                    pass
            except Exception as _reset_err:
                print(f"[init_inference] world.reset() failed: {_reset_err}")
            _inf_init_phase = 1
            _inf_init_step_count = 0
            return  # let main loop step the world

        elif _inf_init_phase == 1:
            # Phase 1: initialize articulation using Articulation class
            # (same approach as working collection code)
            if _inf_init_step_count < 5:
                return  # let physics step a few frames first
            print("[init_inference] Phase 1: initializing articulation...")
            art_prim = _find_robot_articulation()
            if art_prim is None:
                _inf_init_error = "No ArticulationRootAPI found in scene"
                _inf_init_pending = False
                print(f"[init_inference] ERROR: {_inf_init_error}")
                return
            _inf_robot_prim = art_prim
            prim_path = art_prim.GetPath().pathString
            print(f"[init_inference] Found ArticulationRoot: {prim_path}")
            # Step world once (same pattern as collection code line 2717)
            world.step(render=True)
            art = _Articulation(prim_path)
            art.initialize()
            _inf_dc = art
            names = [str(n) for n in art.dof_names] if art.dof_names is not None else []
            print(f"[init_inference] Articulation initialized: {len(names)} DOF")
            _inf_init_phase = 2
            _inf_init_step_count = 0
            return

        elif _inf_init_phase == 2:
            # Phase 2: wait a few frames, then setup cameras
            if _inf_init_step_count < 5:
                return
            print("[init_inference] Phase 2: setting up cameras...")
            _inf_cameras.clear()
            camera_names = []
            try:
                import omni.replicator.core as rep
                import yaml as _yaml
                meta_path = "/data/embodied/asset/robots/openarm_bimanual/camera_meta.yaml"
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = _yaml.safe_load(f)
                    stage = _get_stage()
                    cams = meta.get("wrist_cameras") or meta.get("cameras") or {}
                    for cam_name, cam_cfg in cams.items():
                        if "wrist" not in cam_name.lower():
                            continue
                        mount_link = cam_cfg.get("mount_link", "")
                        mount_prim = None
                        for p in stage.Traverse():
                            if p.GetName() == mount_link:
                                mount_prim = p
                                break
                        if mount_prim is None:
                            print(f"[init_inference] Camera {cam_name}: mount_link '{mount_link}' not found")
                            continue
                        # Prefer existing camera prim from sensor.usd (e.g. wrist_cam)
                        prim_name = cam_cfg.get("prim_name", cam_name)
                        existing_cam = mount_prim.GetPath().pathString + f"/{prim_name}"
                        cam_path = mount_prim.GetPath().pathString + f"/{cam_name}"
                        if stage.GetPrimAtPath(existing_cam).IsValid():
                            cam_path = existing_cam
                            print(f"[init_inference] Using existing camera prim: {cam_path}")
                        elif not stage.GetPrimAtPath(cam_path).IsValid():
                            cam_p = UsdGeom.Camera.Define(stage, cam_path)
                            translate = cam_cfg.get("translate", [0, 0, 0])
                            xf = UsdGeom.Xformable(cam_p.GetPrim())
                            xf.ClearXformOpOrder()
                            t_op = xf.AddTranslateOp()
                            t_op.Set(Gf.Vec3d(*translate))
                            cam_p.GetFocalLengthAttr().Set(cam_cfg.get("focal_length", 24.0))
                            clip = cam_cfg.get("clipping_range", [0.01, 10.0])
                            from pxr import Gf as _Gf
                            cam_p.GetClippingRangeAttr().Set(_Gf.Vec2f(*clip))
                            print(f"[init_inference] Created new camera: {cam_path}")
                        rp = rep.create.render_product(cam_path, _inf_cam_resolution)
                        annot = rep.AnnotatorRegistry.get_annotator("rgb")
                        annot.attach([rp])
                        _inf_cameras[cam_name] = (rp, annot)
                        camera_names.append(cam_name)
                        print(f"[init_inference] Camera: {cam_name} at {cam_path}")
            except Exception as e:
                logger.warning(f"Camera setup: {e}")
            _inf_init_phase = 3
            _inf_init_step_count = 0
            return

        elif _inf_init_phase == 3:
            # Phase 3: warm up render products (need enough frames for replicator to initialize)
            if _inf_init_step_count < 30:
                return
            names = [str(n) for n in _inf_dc.dof_names] if _inf_dc.dof_names is not None else []
            positions = _inf_dc.get_joint_positions()
            pos_list = positions.tolist() if positions is not None else []
            _inf_init_result = {
                "n_dof": len(names), "names": names,
                "positions": pos_list, "cameras": list(_inf_cameras.keys()),
            }
            _inf_init_phase = 4
            _inf_init_pending = False
            print(f"[init_inference] Complete: {len(names)} DOF, cameras={list(_inf_cameras.keys())}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        _inf_init_error = str(e)
        _inf_init_pending = False
        print(f"[init_inference] ERROR: {e}")


# ── Main render loop ─────────────────────────────────────────
step = 0
try:
    while simulation_app.is_running():
        _apply_emergency_stop_if_requested()
        _process_commands()
        if _inf_init_pending and _inf_init_phase < 4 and _inf_init_error is None:
            _run_deferred_inference_init()
        if _state["collecting"] and _collect_request is not None:
            _run_pending_collection()
        elif _state.get("replaying") and _replay_record_request is not None:
            _run_pending_replay_record()
        elif _state.get("replaying") and _replay_request is not None:
            _run_pending_replay()
        elif world and PHYSICS_RUNNING:
            world.step(render=True)
        else:
            simulation_app.update()
        step += 1
        _state["step"] = step
        if step % 300 == 0:
            mode = "physics" if PHYSICS_RUNNING else "viewer"
            print(f"[interactive] frame={step} mode={mode} scene={_state['scene']}")
except KeyboardInterrupt:
    print(f"[interactive] Stopped at frame {step}")
finally:
    if world:
        world.stop()
    simulation_app.close()
    print("[interactive] Done.")
