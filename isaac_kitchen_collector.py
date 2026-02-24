#!/usr/bin/env python3
"""Standalone Isaac Sim kitchen grasp data collector.

Run with Isaac Sim Python:
    /isaac-sim/python.sh /sim-service/isaac_kitchen_collector.py \
        --num-episodes 5 --output /tmp/test_collect
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lerobot_writer import SimLeRobotWriter

LOG = logging.getLogger("isaac-kitchen-collector")

STATE_DIM = 23
ACTION_DIM = 8
FPS = 30
CAMERA_NAMES = ["cam_high", "cam_wrist"]
CAMERA_RESOLUTION = (512, 512)
ROBOT_TYPE = "franka"
STATE_NAMES = [
    "joint_pos_0",
    "joint_pos_1",
    "joint_pos_2",
    "joint_pos_3",
    "joint_pos_4",
    "joint_pos_5",
    "joint_pos_6",
    "joint_vel_0",
    "joint_vel_1",
    "joint_vel_2",
    "joint_vel_3",
    "joint_vel_4",
    "joint_vel_5",
    "joint_vel_6",
    "gripper_pos_0",
    "gripper_pos_1",
    "eef_pos_x",
    "eef_pos_y",
    "eef_pos_z",
    "eef_quat_w",
    "eef_quat_x",
    "eef_quat_y",
    "eef_quat_z",
]
ACTION_NAMES = [
    "action_joint_0",
    "action_joint_1",
    "action_joint_2",
    "action_joint_3",
    "action_joint_4",
    "action_joint_5",
    "action_joint_6",
    "action_gripper",
]

HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
APPROACH = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.8, 0.785], dtype=np.float32)
GRASP_DOWN = np.array([0.0, 0.1, 0.0, -1.2, 0.0, 2.0, 0.785], dtype=np.float32)
LIFT = np.array([0.0, -0.5, 0.0, -1.8, 0.0, 1.6, 0.785], dtype=np.float32)

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
WAYPOINT_NOISE_RAD = 0.05

# Graspable object defaults
OBJECT_SIZE = 0.04  # 4cm cube fits in Franka gripper (max opening ~8cm)
OBJECT_MASS = 0.1   # 100g
OBJECT_COLOR = np.array([0.8, 0.1, 0.1])  # dark red

# YCB objects small enough for Franka gripper (~8cm max opening)
YCB_OBJECTS = [
    "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
    "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
]


def _find_clear_spot(stage: Any, table_z: float, robot_clearance: float = 0.3) -> tuple:
    """Scan prims near table height to find a clear area for the robot.

    Returns (x, y, min_dist) of the best clear spot on the table surface.
    """
    from pxr import UsdGeom as _UsdGeom

    occupied: list[tuple[float, float]] = []
    for prim in stage.Traverse():
        if not prim.IsA(_UsdGeom.Xformable):
            continue
        try:
            xf = _UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            pos = xf.ExtractTranslation()
            # Objects near table surface (within 0.4m above)
            if abs(pos[2] - table_z) < 0.4 and abs(pos[0]) < 1.5 and abs(pos[1]) < 1.5:
                occupied.append((float(pos[0]), float(pos[1])))
        except Exception:
            pass

    LOG.info("Found %d occupied positions near table z=%.2f", len(occupied), table_z)

    # Grid search: find position with maximum clearance from all occupied spots
    best_pos = (0.05, 0.05)
    best_min_dist = 0.0
    for x in np.arange(-0.8, 0.8, 0.05):
        for y in np.arange(-0.8, 0.8, 0.05):
            if not occupied:
                return (float(x), float(y), 999.0)
            min_dist = min(((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 for ox, oy in occupied)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = (float(x), float(y))

    return (*best_pos, best_min_dist)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Franka kitchen grasp data in Isaac Sim.")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to collect.")
    parser.add_argument(
        "--output",
        type=str,
        default="/sim-service/datasets/kitchen_grasp",
        help="Output LeRobot dataset directory.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run Isaac Sim headless (default: False for WebRTC monitoring).",
    )
    parser.add_argument(
        "--robot-prim",
        type=str,
        default="/World/Franka",
        help="Preferred robot prim path.",
    )
    parser.add_argument(
        "--camera-prim",
        type=str,
        default="/World/Camera",
        help="Preferred camera prim path.",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Dataset FPS.")
    parser.add_argument(
        "--scene-usd",
        type=str,
        default="/scenesmith/scenes/tabletop_scene.usda",
        help="Scene USD path (tabletop or kitchen).",
    )
    parser.add_argument(
        "--robot-pos",
        type=float,
        nargs=3,
        default=[0.05, 0.05, 0.902],
        help="Robot world position [x, y, z] (default: on counter).",
    )
    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=30,
        help="Interpolation steps for each waypoint transition.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for waypoint noise.")
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Enable NVCF WebRTC streaming for live monitoring.",
    )
    parser.add_argument("--streaming-port", type=int, default=49102, help="WebRTC signaling port.")
    return parser.parse_args()


def _to_numpy(value: Any, dtype: Any = np.float32) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=dtype)
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=dtype)


def _pad_or_trim(vec: np.ndarray, size: int, fill: float = 0.0) -> np.ndarray:
    out = np.full((size,), fill, dtype=np.float32)
    if vec.size:
        flat = vec.reshape(-1).astype(np.float32, copy=False)
        out[: min(size, flat.size)] = flat[:size]
    return out


def _prim_is_valid(stage: Any, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    return bool(prim) and prim.IsValid()


def _resolve_robot_prim(stage: Any, requested_path: str) -> str | None:
    candidates: list[str] = []
    for path in [
        requested_path,
        "/World/Franka",
        "/World/franka",
        "/World/envs/env_0/Franka",
        "/Franka",
    ]:
        if path and path not in candidates:
            candidates.append(path)

    for path in candidates:
        if _prim_is_valid(stage, path):
            if path != requested_path:
                LOG.warning("Robot prim '%s' not found, using fallback '%s'.", requested_path, path)
            else:
                LOG.info("Using robot prim '%s'.", path)
            return path

    discovered: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        path = prim.GetPath().pathString
        if "franka" in path.lower():
            discovered.append(path)
            if len(discovered) >= 10:
                break

    if discovered:
        LOG.warning(
            "Robot prim '%s' not found. Using discovered Franka-like prim '%s'.",
            requested_path,
            discovered[0],
        )
        return discovered[0]

    LOG.error("Unable to find Franka prim. Tried: %s", ", ".join(candidates))
    return None


def _resolve_camera_prim(stage: Any, requested_path: str, usd_geom: Any, gf: Any) -> str:
    def _is_camera(path: str) -> bool:
        prim = stage.GetPrimAtPath(path)
        return bool(prim) and prim.IsValid() and prim.IsA(usd_geom.Camera)

    for path in [requested_path, "/World/Camera", "/World/camera"]:
        if path and _is_camera(path):
            if path != requested_path:
                LOG.warning("Camera prim '%s' not found, using '%s'.", requested_path, path)
            else:
                LOG.info("Using camera prim '%s'.", path)
            return path

    for prim in stage.Traverse():
        if prim.IsValid() and prim.IsA(usd_geom.Camera):
            path = prim.GetPath().pathString
            LOG.warning("Camera prim '%s' not found, using first stage camera '%s'.", requested_path, path)
            return path

    create_path = requested_path or "/World/Camera"
    try:
        cam = usd_geom.Camera.Define(stage, create_path)
        xformable = usd_geom.Xformable(cam.GetPrim())
        xformable.AddTranslateOp().Set(gf.Vec3d(1.5, -1.5, 1.6))
        xformable.AddRotateXYZOp().Set(gf.Vec3f(60.0, 0.0, 45.0))
        LOG.warning("No camera prim found; created camera at '%s'.", create_path)
        return create_path
    except Exception:
        fallback_path = "/World/CollectorCamera"
        cam = usd_geom.Camera.Define(stage, fallback_path)
        xformable = usd_geom.Xformable(cam.GetPrim())
        xformable.AddTranslateOp().Set(gf.Vec3d(1.5, -1.5, 1.6))
        xformable.AddRotateXYZOp().Set(gf.Vec3f(60.0, 0.0, 45.0))
        LOG.warning("Created fallback camera at '%s'.", fallback_path)
        return fallback_path


def _resolve_eef_prim(stage: Any, robot_prim_path: str, get_prim_at_path: Any) -> str | None:
    candidates = [
        f"{robot_prim_path}/panda_hand",
        f"{robot_prim_path}/panda_hand_tcp",
        f"{robot_prim_path}/panda_link8",
    ]
    for path in candidates:
        prim = get_prim_at_path(path)
        if prim and prim.IsValid():
            return path

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        path = prim.GetPath().pathString
        if not path.startswith(robot_prim_path):
            continue
        lower = path.lower()
        if "hand" in lower or "gripper" in lower or "tcp" in lower:
            return path
    return None


def _get_eef_pose(
    stage: Any,
    eef_prim_path: str,
    get_prim_at_path: Any,
    usd: Any,
    usd_geom: Any,
) -> tuple[np.ndarray, np.ndarray]:
    eef_pos = np.zeros(3, dtype=np.float32)
    eef_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    prim = get_prim_at_path(eef_prim_path)
    if not prim or not prim.IsValid():
        prim = stage.GetPrimAtPath(eef_prim_path)
    if not prim or not prim.IsValid():
        return eef_pos, eef_quat

    xformable = usd_geom.Xformable(prim)
    tf = xformable.ComputeLocalToWorldTransform(usd.TimeCode.Default())
    translation = tf.ExtractTranslation()
    rotation = tf.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()

    eef_pos = np.array([translation[0], translation[1], translation[2]], dtype=np.float32)
    eef_quat = np.array(
        [rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
        dtype=np.float32,
    )
    return eef_pos, eef_quat


def _extract_state_vector(
    franka: Any,
    stage: Any,
    eef_prim_path: str,
    get_prim_at_path: Any,
    usd: Any,
    usd_geom: Any,
) -> np.ndarray:
    joint_positions_all = _to_numpy(franka.get_joint_positions())
    joint_velocities_all = _to_numpy(franka.get_joint_velocities())

    joint_pos_padded = _pad_or_trim(joint_positions_all, 9)
    joint_vel_padded = _pad_or_trim(joint_velocities_all, 7)

    joint_pos = joint_pos_padded[:7]
    joint_vel = joint_vel_padded[:7]
    gripper_pos = joint_pos_padded[7:9]
    eef_pos, eef_quat = _get_eef_pose(stage, eef_prim_path, get_prim_at_path, usd, usd_geom)

    state = np.concatenate([joint_pos, joint_vel, gripper_pos, eef_pos, eef_quat], axis=0)
    return _pad_or_trim(state, STATE_DIM)


def _set_joint_targets(
    franka: Any, arm_target: np.ndarray, gripper_target: float, physics_control: bool = True
) -> np.ndarray:
    current = _to_numpy(franka.get_joint_positions())
    dof = int(current.size) if current.size > 0 else 9
    if dof < 7:
        raise RuntimeError(f"Robot DOF is {dof}, expected at least 7.")

    targets = current.copy() if current.size > 0 else np.zeros((dof,), dtype=np.float32)
    targets[:7] = arm_target[:7]
    if dof > 7:
        targets[7] = float(gripper_target)
    if dof > 8:
        targets[8] = float(gripper_target)

    if physics_control:
        from omni.isaac.core.utils.types import ArticulationAction
        franka.apply_action(ArticulationAction(joint_positions=targets))
    else:
        franka.set_joint_positions(targets)
    return targets


def _capture_rgb(camera: Any, resolution: tuple[int, int]) -> np.ndarray:
    height, width = resolution
    black = np.zeros((height, width, 3), dtype=np.uint8)

    try:
        rgba = camera.get_rgba()
    except Exception as exc:
        LOG.warning("Camera capture failed: %s", exc)
        return black

    if rgba is None:
        return black

    image = np.asarray(rgba)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        return black

    if image.shape[-1] >= 4:
        image = image[..., :3]
    elif image.shape[-1] != 3:
        return black

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(np.max(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[0] != height or image.shape[1] != width:
        resized = np.zeros((height, width, 3), dtype=np.uint8)
        h = min(height, image.shape[0])
        w = min(width, image.shape[1])
        resized[:h, :w] = image[:h, :w]
        image = resized

    return image


def _make_noisy_waypoints(rng: np.random.Generator) -> list[tuple[str, np.ndarray, float]]:
    def noisy(base: np.ndarray) -> np.ndarray:
        noise = rng.uniform(-WAYPOINT_NOISE_RAD, WAYPOINT_NOISE_RAD, size=base.shape).astype(np.float32)
        return base + noise

    noisy_approach = noisy(APPROACH)
    noisy_grasp = noisy(GRASP_DOWN)
    noisy_lift = noisy(LIFT)

    return [
        ("HOME", HOME.copy(), GRIPPER_OPEN),
        ("APPROACH", noisy_approach, GRIPPER_OPEN),
        ("GRASP_DOWN", noisy_grasp, GRIPPER_OPEN),
        ("CLOSE_GRIPPER", noisy_grasp, GRIPPER_CLOSED),
        ("LIFT", noisy_lift, GRIPPER_CLOSED),
    ]


def _is_stage_loading(usd_context: Any) -> bool:
    is_loading = getattr(usd_context, "is_loading", None)
    if callable(is_loading):
        try:
            return bool(is_loading())
        except Exception:
            return False
    return False


def _wait_for_stage_ready(simulation_app: Any, usd_context: Any, timeout_s: float = 60.0) -> Any:
    start = time.time()
    stage = None
    while (time.time() - start) < timeout_s:
        simulation_app.update()
        stage = usd_context.get_stage()
        if stage and not _is_stage_loading(usd_context):
            return stage
        time.sleep(0.01)
    return stage


def _run_episode(
    episode_index: int,
    world: Any,
    franka: Any,
    cameras: dict[str, Any],
    stage: Any,
    eef_prim_path: str,
    get_prim_at_path: Any,
    usd: Any,
    usd_geom: Any,
    writer: SimLeRobotWriter,
    rng: np.random.Generator,
    fps: int,
    steps_per_segment: int,
    target_object: Any = None,
    robot_pos: tuple = (0.05, 0.05, 0.902),
) -> int:
    world.reset()
    for _ in range(10):
        world.step(render=True)

    # Remove previous grasp joint if exists
    from pxr import UsdPhysics as UsdPhysicsRT
    joint_prim = stage.GetPrimAtPath("/World/GraspJoint")
    if joint_prim and joint_prim.IsValid():
        stage.RemovePrim("/World/GraspJoint")

    # Reset graspable object to randomized position on shelf
    if target_object is not None:
        shelf_x = robot_pos[0] + 0.59
        shelf_y = robot_pos[1] + 0.05
        shelf_top_z = robot_pos[2] + 0.73  # shelf center + half thickness
        obj_x = shelf_x + rng.uniform(-0.03, 0.03)
        obj_y = shelf_y + rng.uniform(-0.03, 0.03)
        obj_z = shelf_top_z + OBJECT_SIZE / 2  # on top of shelf
        target_object.set_world_pose(
            position=np.array([obj_x, obj_y, obj_z], dtype=np.float32),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        LOG.info("Episode %d: object at (%.3f, %.3f, %.3f)", episode_index + 1, obj_x, obj_y, obj_z)

    waypoints = _make_noisy_waypoints(rng)
    _, home_arm, home_gripper = waypoints[0]
    # Teleport to HOME (kinematic, no physics delay)
    _set_joint_targets(franka, home_arm, home_gripper, physics_control=False)
    for _ in range(20):
        world.step(render=True)

    frame_index = 0
    transitions = list(zip(waypoints[:-1], waypoints[1:]))

    for transition_idx, (start_wp, end_wp) in enumerate(transitions):
        start_name, start_arm, start_gripper = start_wp
        end_name, end_arm, end_gripper = end_wp
        LOG.debug(
            "Episode %d transition %s -> %s (%d steps)",
            episode_index + 1,
            start_name,
            end_name,
            steps_per_segment,
        )

        for step in range(steps_per_segment):
            alpha = float(step + 1) / float(steps_per_segment)
            arm_target = ((1.0 - alpha) * start_arm + alpha * end_arm).astype(np.float32)
            gripper_target = float((1.0 - alpha) * start_gripper + alpha * end_gripper)

            _set_joint_targets(franka, arm_target, gripper_target, physics_control=False)
            world.step(render=True)

            state = _extract_state_vector(franka, stage, eef_prim_path, get_prim_at_path, usd, usd_geom)
            action = np.concatenate([arm_target, np.array([gripper_target], dtype=np.float32)], axis=0)
            action = _pad_or_trim(action, ACTION_DIM)
            is_last = transition_idx == (len(transitions) - 1) and step == (steps_per_segment - 1)
            writer.add_frame(
                episode_index=episode_index,
                frame_index=frame_index,
                observation_state=state,
                action=action,
                timestamp=frame_index / float(fps),
                next_done=is_last,
            )
            for cam_name, cam_obj in cameras.items():
                rgb = _capture_rgb(cam_obj, CAMERA_RESOLUTION)
                writer.add_video_frame(cam_name, rgb)
            frame_index += 1

        # Log EEF position at end of each transition for debugging
        eef_pos, _ = _get_eef_pose(stage, eef_prim_path, get_prim_at_path, usd, usd_geom)
        LOG.info(
            "Episode %d %s->%s done, EEF=(%.3f, %.3f, %.3f)",
            episode_index + 1, start_name, end_name,
            eef_pos[0], eef_pos[1], eef_pos[2],
        )

        # After gripper close: check proximity and attach object via physics joint
        if end_name == "CLOSE_GRIPPER" and target_object is not None:
            # Settling
            for _ in range(20):
                _set_joint_targets(franka, end_arm, end_gripper, physics_control=False)
                world.step(render=True)

            # Check if object is near gripper
            eef_pos_now, _ = _get_eef_pose(stage, eef_prim_path, get_prim_at_path, usd, usd_geom)
            obj_pose_now, _ = target_object.get_world_pose()
            xy_dist = float(np.linalg.norm(eef_pos_now[:2] - obj_pose_now[:2]))
            z_dist = abs(float(eef_pos_now[2]) - float(obj_pose_now[2]))
            LOG.info("Grasp check: EEF=(%.3f,%.3f,%.3f), obj=(%.3f,%.3f,%.3f), xy_dist=%.3f, z_dist=%.3f",
                     eef_pos_now[0], eef_pos_now[1], eef_pos_now[2],
                     obj_pose_now[0], obj_pose_now[1], obj_pose_now[2],
                     xy_dist, z_dist)

            if xy_dist < 0.08 and z_dist < 0.15:
                # Create fixed joint to attach object to hand (physics-based constraint)
                joint_path = "/World/GraspJoint"
                joint = UsdPhysicsRT.FixedJoint.Define(stage, joint_path)
                joint.GetBody0Rel().SetTargets([eef_prim_path])
                joint.GetBody1Rel().SetTargets(["/World/GraspTarget"])
                LOG.info("GRASP SUCCESS: attached object to hand via FixedJoint")
            else:
                LOG.warning("GRASP FAILED: object too far (xy=%.3f, z=%.3f)", xy_dist, z_dist)

    # Log object Z at end (check if lifted)
    if target_object is not None:
        obj_pose, _ = target_object.get_world_pose()
        LOG.info(
            "Episode %d end: object at (%.3f, %.3f, %.3f)",
            episode_index + 1, obj_pose[0], obj_pose[1], obj_pose[2],
        )

    return frame_index


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    if args.num_episodes <= 0:
        LOG.error("--num-episodes must be > 0")
        return 1
    if args.fps <= 0:
        LOG.error("--fps must be > 0")
        return 1
    if args.steps_per_segment <= 0:
        LOG.error("--steps-per-segment must be > 0")
        return 1

    LOG.info("Starting kitchen collector. headless=%s, episodes=%d", args.headless, args.num_episodes)
    LOG.info("Expected launcher: /isaac-sim/python.sh (current python: %s)", sys.executable)

    simulation_app = None
    writer: SimLeRobotWriter | None = None
    try:
        # Isaac Sim requirement: create SimulationApp before importing omni.* modules.
        from isaacsim import SimulationApp

        app_config = {
            "headless": bool(args.headless),
            "width": 640,
            "height": 480,
            "extra_args": [
                "--/rtx/sceneDb/maxTLASBuildSize=536870912",
                "--/renderer/multiGpu/enabled=False",
            ],
        }
        if args.streaming:
            app_config["livestream_port"] = args.streaming_port
        LOG.info("Creating SimulationApp with config: %s", app_config)
        simulation_app = SimulationApp(app_config)

        # Enable WebRTC streaming if requested
        if args.streaming:
            try:
                import omni.kit.app
                ext_manager = omni.kit.app.get_app().get_extension_manager()
                ext_manager.set_extension_enabled_immediate("omni.services.livestream.nvcf", True)
                import carb
                settings = carb.settings.get_settings()
                settings.set("/app/livestream/port", args.streaming_port)
                LOG.info("WebRTC streaming enabled on port %d", args.streaming_port)
            except Exception as exc:
                LOG.warning("Failed to enable streaming (continuing without): %s", exc)

        import omni.usd
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.prims import get_prim_at_path
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        from omni.isaac.sensor import Camera
        from pxr import Gf, Usd, UsdGeom

        usd_context = omni.usd.get_context()
        LOG.info("Opening scene USD: %s", args.scene_usd)
        if not usd_context.open_stage(args.scene_usd):
            LOG.error("Failed to open USD stage: %s", args.scene_usd)
            return 1

        stage = _wait_for_stage_ready(simulation_app, usd_context, timeout_s=120.0)
        if stage is None:
            LOG.error("Stage is not ready after timeout: %s", args.scene_usd)
            return 1

        LOG.info("Stage loaded. Running warmup frames for physics initialization...")
        for i in range(100):
            simulation_app.update()
            if i % 25 == 0:
                LOG.info("  warmup frame %d/100", i)
        LOG.info("Warmup complete.")

        # --- Robot position ---
        robot_prim_path = args.robot_prim
        rx, ry, rz = args.robot_pos
        stage = omni.usd.get_context().get_stage()
        LOG.info("Robot position: (%.3f, %.3f, %.3f)", rx, ry, rz)

        # --- Add Franka from S3 assets (replaces broken Nucleus references) ---
        assets_root = get_assets_root_path()
        if assets_root is None:
            LOG.error("Cannot resolve Isaac Sim assets root path. Need internet for S3.")
            return 1
        franka_usd = assets_root + "/Isaac/Robots/Franka/franka.usd"
        LOG.info("Adding Franka from: %s", franka_usd)

        add_reference_to_stage(usd_path=franka_usd, prim_path=robot_prim_path)
        franka_prim = stage.GetPrimAtPath(robot_prim_path)
        if franka_prim and franka_prim.IsValid():
            xformable = UsdGeom.Xformable(franka_prim)
            # Set existing translate op (S3 Franka already has xformOps)
            translate_attr = franka_prim.GetAttribute("xformOp:translate")
            if translate_attr and translate_attr.IsValid():
                translate_attr.Set(Gf.Vec3d(rx, ry, rz))
            else:
                xformable.AddTranslateOp().Set(Gf.Vec3d(rx, ry, rz))
            LOG.info("Franka positioned at (%.3f, %.3f, %.3f)", rx, ry, rz)
        else:
            LOG.error("Franka prim not valid at %s after adding reference.", robot_prim_path)
            return 1

        # Wait for Franka meshes to load from S3
        for i in range(100):
            simulation_app.update()
            if i % 25 == 0:
                LOG.info("  Franka mesh loading frame %d/100", i)

        # --- Create wrist camera on panda_hand ---
        wrist_cam_path = f"{robot_prim_path}/panda_hand/wrist_cam"
        wrist_cam_usd = UsdGeom.Camera.Define(stage, wrist_cam_path)
        wrist_prim = wrist_cam_usd.GetPrim()
        wrist_xform = UsdGeom.Xformable(wrist_prim)
        wrist_xform.ClearXformOpOrder()
        wrist_xform.AddTranslateOp().Set(Gf.Vec3d(0.05, 0.0, 0.04))
        wrist_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))
        LOG.info("Created wrist camera at %s", wrist_cam_path)

        # --- Create overhead camera: angled top-down view ---
        # Behind and above robot, tilted 25° from vertical toward +Y (workspace)
        cam_pos = Gf.Vec3d(rx + 0.3, ry - 0.3, rz + 1.3)

        overhead_cam_path = "/World/OverheadCam"
        overhead_cam_usd = UsdGeom.Camera.Define(stage, overhead_cam_path)
        overhead_prim = overhead_cam_usd.GetPrim()
        overhead_xform = UsdGeom.Xformable(overhead_prim)
        overhead_xform.ClearXformOpOrder()
        overhead_xform.AddTranslateOp().Set(cam_pos)
        overhead_xform.AddRotateXYZOp().Set(Gf.Vec3f(25.0, 0.0, 0.0))
        overhead_cam_usd.GetFocalLengthAttr().Set(14.0)
        overhead_cam_usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        LOG.info("Overhead camera at %s, tilt=25deg, focal=14mm", cam_pos)

        # --- Upgrade scene lighting ---
        from pxr import UsdLux
        # Use existing DomeLight if present, else create one
        existing_dome = stage.GetPrimAtPath("/World/DomeLight")
        if existing_dome and existing_dome.IsValid():
            dome_light = UsdLux.DomeLight(existing_dome)
            dome_light.GetIntensityAttr().Set(3000.0)
        else:
            dome_light = UsdLux.DomeLight.Define(stage, "/World/CollectorDomeLight")
            dome_light.GetIntensityAttr().Set(3000.0)
            dome_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
        LOG.info("Scene lighting configured")

        distant_light = UsdLux.DistantLight.Define(stage, "/World/CollectorKeyLight")
        distant_light.GetIntensityAttr().Set(5000.0)
        distant_light.GetAngleAttr().Set(1.0)
        dl_xform = UsdGeom.Xformable(distant_light.GetPrim())
        dl_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))
        LOG.info("Added distant key light")

        # Interior workspace light — sphere light above shelf area
        work_light = UsdLux.SphereLight.Define(stage, "/World/WorkLight")
        work_light.GetIntensityAttr().Set(15000.0)
        work_light.GetRadiusAttr().Set(0.15)
        work_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
        wl_xform = UsdGeom.Xformable(work_light.GetPrim())
        wl_xform.AddTranslateOp().Set(Gf.Vec3d(rx + 0.3, ry, rz + 1.5))
        LOG.info("Added workspace sphere light above shelf")

        # Fill light from opposite side to reduce harsh shadows
        fill_light = UsdLux.SphereLight.Define(stage, "/World/FillLight")
        fill_light.GetIntensityAttr().Set(8000.0)
        fill_light.GetRadiusAttr().Set(0.2)
        fill_light.GetColorAttr().Set(Gf.Vec3f(0.95, 0.95, 1.0))
        fl_xform = UsdGeom.Xformable(fill_light.GetPrim())
        fl_xform.AddTranslateOp().Set(Gf.Vec3d(rx - 0.3, ry + 0.5, rz + 1.2))
        LOG.info("Added fill light")

        for _ in range(20):
            simulation_app.update()

        # --- Create shelf + graspable object ---
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
        from pxr import UsdPhysics, UsdShade

        # Shelf: forward of robot, matching GRASP_DOWN EEF reach
        shelf_x = rx + 0.59
        shelf_y = ry + 0.05
        shelf_z = rz + 0.72  # top at rz+0.73

        shelf = FixedCuboid(
            prim_path="/World/Shelf",
            name="shelf",
            position=np.array([shelf_x, shelf_y, shelf_z], dtype=np.float32),
            size=1.0,
            scale=np.array([0.20, 0.20, 0.02]),
            color=np.array([0.6, 0.4, 0.2]),
        )
        LOG.info("Created shelf (FixedCuboid) at (%.3f, %.3f, %.3f)", shelf_x, shelf_y, shelf_z)

        # Graspable rigid body object on shelf
        obj_z = shelf_z + 0.01 + OBJECT_SIZE / 2
        cube_pos = np.array([shelf_x, shelf_y, obj_z], dtype=np.float32)
        grasp_target = DynamicCuboid(
            prim_path="/World/GraspTarget",
            name="grasp_target",
            position=cube_pos,
            size=OBJECT_SIZE,
            color=OBJECT_COLOR,
            mass=OBJECT_MASS,
        )

        # High-friction physics material
        grip_mat = UsdShade.Material.Define(stage, "/World/GripMaterial")
        physics_mat_api = UsdPhysics.MaterialAPI.Apply(grip_mat.GetPrim())
        physics_mat_api.CreateStaticFrictionAttr(1.5)
        physics_mat_api.CreateDynamicFrictionAttr(1.0)
        physics_mat_api.CreateRestitutionAttr(0.0)
        obj_prim = stage.GetPrimAtPath("/World/GraspTarget")
        if obj_prim and obj_prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(obj_prim).Bind(
                grip_mat, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
        LOG.info("Created grasp target at (%.3f, %.3f, %.3f)", cube_pos[0], cube_pos[1], cube_pos[2])

        if args.fps != FPS:
            LOG.warning(
                "Writer fps is set to %d, but world rendering_dt remains fixed at 1/30.",
                args.fps,
            )
        world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0)

        franka = Articulation(prim_path=robot_prim_path, name="franka_collector")
        world.scene.add(franka)

        # Create both cameras: overhead (cam_high) + wrist (cam_wrist)
        camera_high = Camera(
            prim_path=overhead_cam_path,
            name="cam_high",
            frequency=args.fps,
            resolution=CAMERA_RESOLUTION,
        )
        world.scene.add(camera_high)

        camera_wrist = Camera(
            prim_path=wrist_cam_path,
            name="cam_wrist",
            frequency=args.fps,
            resolution=CAMERA_RESOLUTION,
        )
        world.scene.add(camera_wrist)
        cameras = {"cam_high": camera_high, "cam_wrist": camera_wrist}

        world.scene.add(grasp_target)

        LOG.info("Running pre-reset warmup...")
        for _ in range(50):
            simulation_app.update()

        LOG.info("Calling world.reset() (this initializes physics)...")
        world.reset()
        LOG.info("world.reset() succeeded. Running post-reset warmup...")
        for _ in range(30):
            world.step(render=True)
        if hasattr(franka, "initialize"):
            try:
                franka.initialize()
            except Exception as exc:
                LOG.warning("Franka initialize() failed (continuing): %s", exc)
        for cam_name, cam_obj in cameras.items():
            if hasattr(cam_obj, "initialize"):
                try:
                    cam_obj.initialize()
                    LOG.info("Camera '%s' initialized.", cam_name)
                except Exception as exc:
                    LOG.warning("Camera '%s' initialize() failed (continuing): %s", cam_name, exc)

        eef_prim_path = _resolve_eef_prim(stage, robot_prim_path, get_prim_at_path)
        if eef_prim_path is None:
            eef_prim_path = f"{robot_prim_path}/panda_hand"
            LOG.warning("EEF prim not found; defaulting to %s", eef_prim_path)
        else:
            LOG.info("Using EEF prim '%s'.", eef_prim_path)

        writer = SimLeRobotWriter(
            output_dir=args.output,
            repo_id="local/franka_kitchen_grasp",
            fps=args.fps,
            robot_type=ROBOT_TYPE,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            camera_names=CAMERA_NAMES,
            camera_resolution=CAMERA_RESOLUTION,
            state_names=STATE_NAMES,
            action_names=ACTION_NAMES,
        )

        rng = np.random.default_rng(args.seed)
        for episode in range(args.num_episodes):
            LOG.info("Episode %d/%d", episode + 1, args.num_episodes)
            episode_frames = 0
            episode_start = time.time()
            try:
                episode_frames = _run_episode(
                    episode_index=episode,
                    world=world,
                    franka=franka,
                    cameras=cameras,
                    stage=stage,
                    eef_prim_path=eef_prim_path,
                    get_prim_at_path=get_prim_at_path,
                    usd=Usd,
                    usd_geom=UsdGeom,
                    writer=writer,
                    rng=rng,
                    fps=args.fps,
                    steps_per_segment=args.steps_per_segment,
                    target_object=grasp_target,
                    robot_pos=tuple(args.robot_pos),
                )
            except Exception as exc:
                LOG.exception("Episode %d failed: %s", episode + 1, exc)
            finally:
                writer.finish_episode(
                    episode_index=episode,
                    length=episode_frames,
                    task="kitchen grasp",
                )
                LOG.info(
                    "Episode %d finished with %d frames in %.2fs",
                    episode + 1,
                    episode_frames,
                    time.time() - episode_start,
                )

        writer.finalize()
        LOG.info("Collection complete: %s", args.output)
        return 0
    except Exception as exc:
        LOG.exception("Collector failed: %s", exc)
        return 1
    finally:
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception as exc:
                LOG.warning("Failed to close SimulationApp cleanly: %s", exc)


if __name__ == "__main__":
    raise SystemExit(main())
