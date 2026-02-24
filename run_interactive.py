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
import os
import sys
import time
import threading

# ── Env defaults ──────────────────────────────────────────────
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ.setdefault("PRIVACY_CONSENT", "Y")
os.environ.setdefault("LIVESTREAM", "2")  # NVCF streaming

HEALTH_PORT = int(os.environ.get("HEALTH_PORT", "5900"))
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "5800"))
WEBRTC_PORT = int(os.environ.get("WEBRTC_PORT", "49100"))
KIT_API_PORT = int(os.environ.get("KIT_API_PORT", "8011"))
MEDIA_PORT = int(os.environ.get("MEDIA_PORT", "47998"))

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

# ── World + physics (paused by default) ──────────────────────
try:
    from omni.isaac.core import World
    world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0)
    world.scene.add_default_ground_plane()
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
}

# ── Thread-safe command queue for main-thread execution ──────
# Isaac Sim API is NOT thread-safe — must execute on main thread.
# Flask handlers push commands here; main loop executes and signals done.
import queue
_cmd_queue = queue.Queue()
_CMD_TIMEOUT = 10  # seconds to wait for main thread to process

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
        return jsonify({"error": "Timeout waiting for main thread"}), 504
    if cmd["error"]:
        return jsonify({"error": cmd["error"]}), 500
    return jsonify(cmd["result"])


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
    data = flask_request.get_json(silent=True) or {}
    num_episodes = int(data.get("num_episodes", 10))
    output_dir = data.get("output_dir", "/data/collections/latest")
    skill = str(data.get("skill", "pick_place") or "pick_place")

    if num_episodes <= 0:
        return jsonify({"error": "num_episodes must be > 0"}), 400

    # Main-thread execution only (Isaac Sim API is not thread-safe).
    return _enqueue_cmd(
        "collect_start",
        num_episodes=num_episodes,
        output_dir=output_dir,
        skill=skill,
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


# ── Start Bridge API in daemon thread ────────────────────────
def _run_bridge():
    bridge.run(host="0.0.0.0", port=BRIDGE_PORT, threaded=True)

threading.Thread(target=_run_bridge, daemon=True, name="bridge-api").start()
print(f"[interactive] Bridge API on :{BRIDGE_PORT}")

# ── Warmup frames ────────────────────────────────────────────
print("[interactive] Running warmup frames...")
for _ in range(60):
    simulation_app.update()
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
                        _state["scene"] = usd_path
                        _state["robots"] = {}
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
                    print(f"[interactive] Spawned {cmd['robot_type']} at {cmd['prim_path']}")
                    cmd["result"] = {"success": True, "prim_path": cmd["prim_path"],
                                     "robot_type": cmd["robot_type"]}

            elif cmd_type == "code_execute":
                import io as _io
                import builtins as _builtins
                from pxr import UsdPhysics, UsdShade, Sdf, Vt
                stdout_capture = _io.StringIO()
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
                    "print": lambda *a, **kw: stdout_capture.write(" ".join(str(x) for x in a) + "\n"),
                    **_extra_globals,
                }
                try:
                    exec(cmd["code"], exec_globals)
                except Exception as _exec_exc:
                    _exc_str = str(_exec_exc)
                    # USD pxr precision mismatch is a warning, not a real error
                    if "pxrInternal" in _exc_str and "Proceeding" in _exc_str:
                        pass  # operation succeeded despite warning
                    else:
                        raise
                # Clear selection to hide gizmo overlay in WebRTC stream
                try:
                    omni.usd.get_context().get_selection().clear_selected_prim_paths()
                except Exception:
                    pass
                for _ in range(5):
                    simulation_app.update()
                cmd["result"] = {"success": True, "output": stdout_capture.getvalue()}

            elif cmd_type == "collect_start":
                if _state["collecting"]:
                    cmd["error"] = "Collection already running"
                elif not _state["physics"]:
                    cmd["error"] = "Physics must be running first (POST /physics/play)"
                else:
                    num_episodes = int(cmd.get("num_episodes", 10))
                    output_dir = str(cmd.get("output_dir", "/data/collections/latest"))
                    skill = str(cmd.get("skill", "pick_place") or "pick_place")
                    if num_episodes <= 0:
                        cmd["error"] = "num_episodes must be > 0"
                    else:
                        _collect_stop.clear()
                        _state["collecting"] = True
                        _state["collect_progress"] = {
                            "total": num_episodes,
                            "completed": 0,
                            "skill": skill,
                            "output_dir": output_dir,
                            "status": "running",
                        }
                        _collect_request = {
                            "num_episodes": num_episodes,
                            "output_dir": output_dir,
                            "skill": skill,
                        }
                        cmd["result"] = {
                            "status": "started",
                            "num_episodes": num_episodes,
                            "output_dir": output_dir,
                            "skill": skill,
                        }
                        print(
                            f"[collect] queued main-thread collection: skill={skill}, "
                            f"episodes={num_episodes}, output={output_dir}"
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
    output_dir = str(req.get("output_dir", "/data/collections/latest"))
    skill = str(req.get("skill", "pick_place"))

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[collect] start main-thread run: skill={skill}, episodes={num_episodes}, output={output_dir}")

        from isaac_pick_place_collector import run_collection_in_process

        result = run_collection_in_process(
            world=world,
            simulation_app=simulation_app,
            num_episodes=num_episodes,
            output_dir=output_dir,
            stop_event=_collect_stop,
            progress_callback=lambda ep: _state["collect_progress"].update({"completed": ep}),
            task_name=skill,
        )

        final_status = "stopped" if _collect_stop.is_set() else "done"
        _state["collect_progress"]["status"] = final_status
        _state["collect_progress"]["result"] = result
        print(
            f"[collect] finished: status={final_status}, "
            f"completed={_state['collect_progress'].get('completed', 0)}"
        )
    except Exception as exc:
        traceback.print_exc()
        _state["collect_progress"]["status"] = f"error: {exc}"
        print(f"[collect] ERROR: {exc}")
    finally:
        _state["collecting"] = False


# ── Main render loop ─────────────────────────────────────────
step = 0
try:
    while simulation_app.is_running():
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
