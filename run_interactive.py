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
import sys
import time
import threading
import re
import uuid
from pathlib import Path

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
EMBODIED_DATASETS_ROOT = os.environ.get("EMBODIED_DATASETS_ROOT", "/data/embodied/datasets")
LEGACY_COLLECT_OUTPUT_DIRS = {
    "/data/collections/latest",
    "/data/embodied/datasets/sim_collect_latest",
}

# ── Inject Kit args for WebRTC + Kit API ports BEFORE any Isaac imports ──
sys.argv.append(f"--kit_args=--/app/livestream/port={WEBRTC_PORT}")
sys.argv.append(f"--kit_args=--/exts/omni.services.transport.server.http/port={KIT_API_PORT}")
# Fix WebRTC media UDP port so NodePort can forward it
sys.argv.append(f"--kit_args=--/app/livestream/fixedHostPort={MEDIA_PORT}")
print(f"[interactive] WebRTC media UDP fixed to port {MEDIA_PORT}")

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
    # Fallback: try direct SimulationApp
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({
        "headless": True,
        "width": 1920,
        "height": 1080,
    })
    print("[interactive] Isaac Sim started via SimulationApp (fallback)")

# ── Imports available only after Kit starts ───────────────────
import omni.usd
import omni.kit.app
from pxr import Gf, UsdGeom, Usd

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
    ext_mgr.set_extension_enabled_immediate("omni.services.livestream.nvcf", True)
    print("[interactive] NVCF streaming extension enabled")
except Exception as exc:
    print(f"[interactive] WARNING: NVCF extension: {exc}")

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
    from omni.isaac.core import World

    w = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0)
    if add_ground_plane:
        try:
            w.scene.add_default_ground_plane()
        except Exception as exc:
            print(f"[world] WARNING: add_default_ground_plane failed: {exc}")
    return w


def _recreate_world_for_open_stage(reason: str) -> bool:
    """Rebuild World after open_stage(), which invalidates prior World handles."""
    global world, PHYSICS_RUNNING
    try:
        if world is not None:
            try:
                world.stop()
            except Exception:
                pass
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
from flask import Flask, request as flask_request, jsonify
import traceback

bridge = Flask(__name__)
bridge.config["JSONIFY_PRETTYPRINT_REGULAR"] = False


def _get_stage():
    ctx = omni.usd.get_context()
    return ctx.get_stage() if ctx else None


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
    helper_names = {"create_table", "create_franka", "create_mug"}
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
                r"\b(create_table|create_franka|create_mug)\b(?:\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?\s*,?\s*",
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
    from omni.isaac.core.utils.stage import add_reference_to_stage

    stage = stage or _get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available")

    candidates = []
    if usd_path:
        candidates.append(str(usd_path))

    try:
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
    from omni.isaac.core.utils.stage import add_reference_to_stage

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
        from omni.isaac.core.utils.nucleus import get_assets_root_path

        assets_root = get_assets_root_path()
        if assets_root:
            candidates.append(f"{assets_root}/{c1_rel}")
    except Exception:
        pass

    candidates.append(c1_http)
    candidates = _unique_preserve_order(candidates)

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
            for _ in range(8):
                simulation_app.update()
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
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
    px, py, pz = position
    xf.AddTranslateOp().Set(Gf.Vec3d(float(px), float(py), float(pz)))
    xf.AddScaleOp().Set(Gf.Vec3d(0.01, 0.01, 0.01))

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
    return _enqueue_cmd("scene_load", usd_path=usd_path)


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


# ── Robot endpoints ──────────────────────────────────────────

ROBOT_USD_MAP = {
    "franka": "/home/user/magicphysics/MagicPhysics/packages/MagicSim/Assets/Robots/franka_umi.usd",
    "franka_umi": "/home/user/magicphysics/MagicPhysics/packages/MagicSim/Assets/Robots/franka_umi.usd",
}

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
    result_event = threading.Event()
    cmd = {"type": cmd_type, "result": None, "error": None, "event": result_event, **kwargs}
    _cmd_queue.put(cmd)
    if not result_event.wait(timeout=_CMD_TIMEOUT):
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
    global _collect_request
    _collect_stop.set()
    _collect_request = None
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


# ── Data collection ──────────────────────────────────────────

_collect_stop = threading.Event()
_collect_request = None

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
        episode_timeout_sec = float(data.get("episode_timeout_sec", os.environ.get("COLLECT_EPISODE_TIMEOUT_SEC", "180")))
    except Exception:
        return jsonify({"error": "num_episodes/steps_per_segment/episode_timeout_sec must be numeric"}), 400
    skill = str(data.get("skill", "pick_place") or "pick_place")
    output_dir = _normalize_collect_output_dir(data.get("output_dir"), skill)
    scene_mode = str(data.get("scene_mode", "auto") or "auto")
    target_objects = data.get("target_objects")
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
print(f"[interactive] Ready. WebRTC port={WEBRTC_PORT}, Kit API=8011, Bridge={BRIDGE_PORT}")

# ── Main-thread command processor ────────────────────────────
def _process_commands():
    """Drain command queue and execute on main thread (called each frame)."""
    global PHYSICS_RUNNING, _collect_request
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
                ctx = omni.usd.get_context()
                print(f"[interactive] Loading scene: {usd_path}")

                # Guard against invalid paths. Some Isaac builds may shutdown app
                # after open_stage() failures on missing files.
                if not os.path.exists(usd_path):
                    cmd["error"] = f"Scene file not found: {usd_path}"
                else:
                    # open_stage returns bool (True=success), NOT a tuple
                    ok = ctx.open_stage(usd_path)
                    if not ok:
                        cmd["error"] = f"Failed to open stage: {usd_path}"
                    else:
                        for _ in range(300):
                            simulation_app.update()
                            if ctx.get_stage_state() == omni.usd.StageState.OPENED:
                                break
                            time.sleep(0.05)
                        if not _recreate_world_for_open_stage("scene_load"):
                            cmd["error"] = "Failed to recreate world after scene load"
                        else:
                            _state["scene"] = usd_path
                            _state["robots"] = {}
                            _state["physics"] = False
                            _save_autosave_stage("scene_load")
                            print(f"[interactive] Scene loaded: {usd_path}")
                            cmd["result"] = {"success": True, "scene": usd_path}

            elif cmd_type == "robot_spawn":
                stage = _get_stage()
                if not stage:
                    cmd["error"] = "No stage loaded"
                else:
                    from omni.isaac.core.utils.stage import add_reference_to_stage
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
                    _save_autosave_stage("robot_spawn")
                    print(f"[interactive] Spawned {cmd['robot_type']} at {cmd['prim_path']}")
                    cmd["result"] = {"success": True, "prim_path": cmd["prim_path"],
                                     "robot_type": cmd["robot_type"]}

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
                    from omni.isaac.core.utils.stage import add_reference_to_stage
                    _extra_globals["add_reference_to_stage"] = add_reference_to_stage
                except ImportError:
                    pass
                try:
                    from omni.isaac.core.utils.prims import create_prim
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
                # Keep objects visible on table: if a newly placed object intersects table top,
                # lift it slightly above the surface.
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
                elif not _state["physics"]:
                    cmd["error"] = "Physics must be running first (POST /physics/play)"
                else:
                    try:
                        num_episodes = int(cmd.get("num_episodes", 10))
                        steps_per_segment = int(cmd.get("steps_per_segment", os.environ.get("COLLECT_STEPS_PER_SEGMENT_DEFAULT", "70")))
                        episode_timeout_sec = float(cmd.get("episode_timeout_sec", os.environ.get("COLLECT_EPISODE_TIMEOUT_SEC", "180")))
                    except Exception:
                        cmd["error"] = "num_episodes/steps_per_segment/episode_timeout_sec must be numeric"
                        continue
                    skill = str(cmd.get("skill", "pick_place") or "pick_place")
                    output_dir = _normalize_collect_output_dir(cmd.get("output_dir"), skill)
                    scene_mode = str(cmd.get("scene_mode", "auto") or "auto")
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
                        }
                        print(
                            f"[collect] queued main-thread collection: skill={skill}, "
                            f"scene_mode={scene_mode}, episodes={num_episodes}, "
                            f"steps_per_segment={steps_per_segment}, timeout={episode_timeout_sec}s, output={output_dir}"
                        )

            else:
                cmd["error"] = f"Unknown command: {cmd_type}"

        except Exception as exc:
            traceback.print_exc()
            cmd["error"] = str(exc)
        finally:
            cmd["event"].set()


def _run_pending_collection():
    """Run queued collection request on the main thread."""
    global _collect_request
    if not _collect_request:
        return

    req = _collect_request
    _collect_request = None
    num_episodes = int(req.get("num_episodes", 10))
    steps_per_segment = int(req.get("steps_per_segment", os.environ.get("COLLECT_STEPS_PER_SEGMENT_DEFAULT", "70")))
    episode_timeout_sec = float(req.get("episode_timeout_sec", os.environ.get("COLLECT_EPISODE_TIMEOUT_SEC", "180")))
    skill = str(req.get("skill", "pick_place"))
    output_dir = _normalize_collect_output_dir(req.get("output_dir"), skill)
    scene_mode = str(req.get("scene_mode", "auto") or "auto")
    target_objects = req.get("target_objects")

    try:
        # Collect can run after many scene-chat mutations. Recreate World to
        # clear stale scene registry wrappers (e.g. expired /World/Table
        # FixedCuboid handles) while preserving the currently opened USD stage.
        if not _recreate_world_for_open_stage("collect_start"):
            raise RuntimeError("Failed to recreate world before collection")
        if world is None:
            raise RuntimeError("World is unavailable before collection")

        os.makedirs(output_dir, exist_ok=True)
        print(
            f"[collect] start main-thread run: skill={skill}, scene_mode={scene_mode}, "
            f"episodes={num_episodes}, steps_per_segment={steps_per_segment}, "
            f"timeout={episode_timeout_sec}s, output={output_dir}, targets={target_objects}"
        )

        from isaac_pick_place_collector import run_collection_in_process

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
        except Exception:
            pass

        result = run_collection_in_process(**collect_kwargs)

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


# ── Main render loop ─────────────────────────────────────────
step = 0
try:
    while simulation_app.is_running():
        _apply_emergency_stop_if_requested()
        _process_commands()
        if _state["collecting"] and _collect_request is not None:
            _run_pending_collection()
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
