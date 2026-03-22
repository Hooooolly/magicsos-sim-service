#!/usr/bin/env python3
"""Simplified OpenArm pick-and-place data collector.

This collector is designed for an already-loaded Isaac Sim scene containing:
- an OpenArm bimanual articulation
- a cube/cuboid grasp object
- a bowl receptacle
- wrist camera prims mounted on the robot

It records a single right-arm pick-and-place cycle per episode into a LeRobot
dataset using ``SimLeRobotWriter``.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lerobot_writer import SimLeRobotWriter

LOG = logging.getLogger("openarm-pick-place-collector")

STATE_DIM = 8
ACTION_DIM = 8
FPS = 30
STEPS_PER_SEGMENT = 24
CAMERA_NAMES = ["right_wrist_cam"]
CAMERA_RENDER_RESOLUTION = (640, 480)  # (width, height) for replicator
CAMERA_RESOLUTION = (480, 640)  # (height, width) for LeRobot metadata
ROBOT_TYPE = "openarm_bimanual"

STATE_NAMES = [
    "openarm_joint1",
    "openarm_joint3",
    "openarm_joint4",
    "openarm_joint5",
    "openarm_joint6",
    "openarm_joint2",
    "openarm_joint7",
    "openarm_finger",
]
ACTION_NAMES = list(STATE_NAMES)

RIGHT_ARM_INDICES = np.array([1, 3, 5, 7, 9, 11, 13], dtype=np.int64)
RIGHT_FINGER_INDICES = np.array([16, 17], dtype=np.int64)
DATA_TO_SIM = np.array([0, 5, 1, 2, 3, 4, 6], dtype=np.int64)
NEGATED_SIM_JOINTS = {6}  # sim joint 6 (wrist rotation) needs sign flip for dataset

HOME_FULL = np.array(
    [
        0.7854,
        -0.7854,
        0.0,
        0.0,
        0.0,
        0.0,
        1.5708,
        1.5708,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.7854,
        0.7854,
        0.0,
        0.0,
        0.04,
        0.04,
    ],
    dtype=np.float32,
)
HOME_RIGHT_ARM = HOME_FULL[RIGHT_ARM_INDICES].astype(np.float32, copy=True)

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
DEFAULT_OBJECT_POSITION_NOISE = 0.02
DEFAULT_EPISODE_TIMEOUT_SEC = 90.0
ATTACHMENT_JOINT_PATH = "/World/OpenArmCollectorGraspJoint"

DEFAULT_CUBE_POS = np.array([0.292, -0.190, 0.277], dtype=np.float32)
DEFAULT_BOWL_POS = np.array([0.400, 0.000, 0.252], dtype=np.float32)
DEFAULT_RIGHT_CAMERA_CFG = {
    "mount_link": "openarm_right_link7",
    "prim_name": "wrist_cam",
    "resolution": [640, 480],
    "focal_length": 15.0,
}

# Hand-tuned sim-order right-arm waypoints. Small per-episode offsets are added
# based on cube/bowl XY deviations so the collector stays simple and deterministic.
BASE_APPROACH_RIGHT = np.array([-0.56, 0.22, 0.08, 1.18, 0.02, 0.50, 0.78], dtype=np.float32)
BASE_GRASP_RIGHT = np.array([-0.58, 0.36, 0.14, 0.98, 0.02, 0.72, 0.80], dtype=np.float32)
BASE_LIFT_RIGHT = np.array([-0.56, 0.12, 0.08, 1.28, 0.02, 0.45, 0.80], dtype=np.float32)
BASE_BOWL_RIGHT = np.array([-0.20, 0.16, 0.00, 1.22, -0.02, 0.48, 0.50], dtype=np.float32)
BASE_PLACE_RIGHT = np.array([-0.12, 0.28, 0.02, 1.05, -0.02, 0.66, 0.50], dtype=np.float32)


@dataclass
class CollectorContext:
    stage: Any
    robot: Any
    robot_prim_path: str
    openarm_root: str
    right_eef_prim_path: str
    cube_prim_path: Optional[str]
    bowl_prim_path: Optional[str]
    right_wrist_camera_path: str
    right_wrist_render_product: Any
    right_wrist_annotator: Any
    cube_base_pos: np.ndarray
    bowl_base_pos: np.ndarray


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


def _dataset_repo_id(output_dir: str) -> str:
    base = os.path.basename(str(output_dir).rstrip("/")) or "openarm_pick_place"
    return f"local/{base}"


def _step_world(world: Any, simulation_app: Any, render: bool = True, steps: int = 1) -> None:
    for _ in range(max(1, int(steps))):
        if world is not None:
            world.step(render=render)
        elif simulation_app is not None:
            simulation_app.update()


def _world_reset(world: Any, simulation_app: Any) -> None:
    if world is not None and hasattr(world, "reset"):
        world.reset()
    _step_world(world, simulation_app, render=True, steps=8)


def _find_stage(simulation_app: Any) -> Any:
    import omni.usd

    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()
    if stage is None and simulation_app is not None:
        start = time.time()
        while stage is None and (time.time() - start) < 10.0:
            simulation_app.update()
            stage = usd_context.get_stage()
            time.sleep(0.01)
    if stage is None:
        raise RuntimeError("No USD stage available for OpenArm collection.")
    return stage


def _get_articulation_root_paths(stage: Any, usd_physics: Any) -> list[str]:
    paths: list[str] = []
    for prim in stage.Traverse():
        if prim.HasAPI(usd_physics.ArticulationRootAPI):
            paths.append(prim.GetPath().pathString)
    return paths


def _select_robot_prim(stage: Any) -> str:
    from pxr import UsdPhysics

    candidates = _get_articulation_root_paths(stage, UsdPhysics)
    if not candidates:
        raise RuntimeError("No articulation root found in scene.")

    def score(path: str) -> tuple[int, int]:
        low = path.lower()
        return (
            0 if "openarm" in low else 1,
            len(path),
        )

    return sorted(candidates, key=score)[0]


def _openarm_root_from_robot_path(robot_prim_path: str) -> str:
    parts = [part for part in str(robot_prim_path).split("/") if part]
    if len(parts) >= 2:
        return "/" + "/".join(parts[:2])
    return robot_prim_path


def _find_named_prim(stage: Any, tokens: Sequence[str]) -> Optional[str]:
    best_path: Optional[str] = None
    best_key: tuple[int, int] | None = None
    lowered = [str(token).lower() for token in tokens if token]
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        name_low = prim.GetName().lower()
        if not any(token in name_low for token in lowered):
            continue
        path = prim.GetPath().pathString
        key = (path.count("/"), len(path))
        if best_key is None or key < best_key:
            best_path = path
            best_key = key
    return best_path


def _get_prim_world_pose(stage: Any, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    xformable = UsdGeom.Xformable(prim)
    tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = tf.ExtractTranslation()
    quat = tf.ExtractRotationQuat()
    imag = quat.GetImaginary()
    return (
        np.array([translation[0], translation[1], translation[2]], dtype=np.float32),
        np.array([quat.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float32),
    )


def _ensure_translate_op(xformable: Any) -> Any:
    from pxr import UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xformable.AddTranslateOp()


def _set_prim_world_position(stage: Any, prim_path: str, world_pos: np.ndarray) -> bool:
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False

    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return False

    parent = prim.GetParent()
    local_pos = np.asarray(world_pos, dtype=np.float32)
    if parent and parent.IsValid():
        parent_tf = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        try:
            parent_inv = parent_tf.GetInverse()
            local = parent_inv.Transform(Gf.Vec3d(float(world_pos[0]), float(world_pos[1]), float(world_pos[2])))
            local_pos = np.array([local[0], local[1], local[2]], dtype=np.float32)
        except Exception:
            pass

    translate_op = _ensure_translate_op(xformable)
    translate_op.Set(Gf.Vec3d(float(local_pos[0]), float(local_pos[1]), float(local_pos[2])))
    return True


def _remove_prim_if_exists(stage: Any, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def _load_right_wrist_camera_cfg() -> dict[str, Any]:
    cfg = dict(DEFAULT_RIGHT_CAMERA_CFG)
    candidates = [
        SCRIPT_DIR / "camera_meta.yaml",
        Path("/data/embodied/asset/robots/openarm_bimanual/camera_meta.yaml"),
        SCRIPT_DIR / "robot_config.yaml",
    ]
    try:
        import yaml as pyyaml
    except Exception:
        pyyaml = None

    if pyyaml is None:
        return cfg

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with open(candidate) as f:
                data = pyyaml.safe_load(f) or {}
            camera_section = data.get("wrist_cameras") or data.get("cameras") or {}
            cam_cfg = camera_section.get("right_wrist_cam")
            if isinstance(cam_cfg, dict):
                cfg.update(cam_cfg)
                return cfg
        except Exception as exc:
            LOG.warning("Failed to load camera config from %s: %s", candidate, exc)
    return cfg


def _resolve_right_wrist_camera(stage: Any, openarm_root: str, cam_cfg: dict[str, Any]) -> tuple[str, str]:
    mount_link = str(cam_cfg.get("mount_link", DEFAULT_RIGHT_CAMERA_CFG["mount_link"]) or "").strip()
    prim_name = str(cam_cfg.get("prim_name", DEFAULT_RIGHT_CAMERA_CFG["prim_name"]) or "").strip()

    direct_candidates = []
    if mount_link:
        direct_candidates.extend(
            [
                f"{openarm_root}/{mount_link}/{prim_name}",
                f"{openarm_root}/{mount_link}/right_wrist_cam",
                f"{openarm_root}/{mount_link}/Camera",
            ]
        )
    for key in ("camera_prim", "camera_path", "prim_path", "path"):
        raw = str(cam_cfg.get(key, "") or "").strip()
        if raw:
            direct_candidates.append(raw)

    for candidate in direct_candidates:
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            eef_path = candidate.rsplit("/", 1)[0]
            return candidate, eef_path

    fallback_camera = None
    fallback_eef = None
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        path = prim.GetPath().pathString
        low = path.lower()
        if "right" in low and "wrist" in low and ("cam" in low or prim.GetName().lower() == prim_name.lower()):
            fallback_camera = path
            fallback_eef = path.rsplit("/", 1)[0]
            break
        if mount_link and path.endswith("/" + mount_link):
            fallback_eef = path

    if fallback_camera:
        return fallback_camera, fallback_eef or fallback_camera.rsplit("/", 1)[0]
    if fallback_eef:
        raise RuntimeError(f"Found right wrist mount {fallback_eef}, but no camera prim under it.")
    raise RuntimeError("Unable to resolve right wrist camera prim for OpenArm.")


def _setup_right_wrist_camera(stage: Any, camera_prim_path: str, cam_cfg: dict[str, Any]) -> tuple[Any, Any]:
    import omni.replicator.core as rep

    resolution_raw = cam_cfg.get("resolution", list(CAMERA_RENDER_RESOLUTION))
    try:
        render_resolution = (int(resolution_raw[0]), int(resolution_raw[1]))
    except Exception:
        render_resolution = CAMERA_RENDER_RESOLUTION

    rp = rep.create.render_product(camera_prim_path, render_resolution)
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    return rp, annot


def _capture_rgb(annotator: Any) -> np.ndarray:
    height, width = CAMERA_RESOLUTION
    black = np.zeros((height, width, 3), dtype=np.uint8)

    try:
        data = annotator.get_data()
    except Exception as exc:
        LOG.warning("Camera capture failed: %s", exc)
        return black

    if data is None:
        return black

    rgb = np.asarray(data)
    if rgb.ndim != 3 or rgb.shape[-1] < 3 or rgb.size == 0:
        return black
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating) and float(np.max(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if rgb.shape[0] != height or rgb.shape[1] != width:
        resized = np.zeros((height, width, 3), dtype=np.uint8)
        h = min(height, rgb.shape[0])
        w = min(width, rgb.shape[1])
        resized[:h, :w] = rgb[:h, :w]
        rgb = resized
    return rgb


def _sim_arm_to_dataset(arm_sim: np.ndarray) -> np.ndarray:
    arm_sim = np.asarray(arm_sim, dtype=np.float32).reshape(-1)
    if arm_sim.size < 7:
        raise ValueError(f"Expected 7 right-arm joints, got {arm_sim.size}")

    dataset_arm = np.zeros(7, dtype=np.float32)
    for sim_joint_idx, data_joint_idx in enumerate(DATA_TO_SIM):
        value = float(arm_sim[sim_joint_idx])
        if sim_joint_idx in NEGATED_SIM_JOINTS:
            value = -value
        dataset_arm[int(data_joint_idx)] = value
    return dataset_arm


def _full_to_dataset_state(full_joint_positions: np.ndarray) -> np.ndarray:
    full_joint_positions = np.asarray(full_joint_positions, dtype=np.float32).reshape(-1)
    if full_joint_positions.size < int(RIGHT_FINGER_INDICES[-1]) + 1:
        raise RuntimeError(
            f"OpenArm joint vector too small: got {full_joint_positions.size}, expected at least 18."
        )
    arm_sim = full_joint_positions[RIGHT_ARM_INDICES]
    finger = float(np.mean(full_joint_positions[RIGHT_FINGER_INDICES]))
    return np.concatenate([_sim_arm_to_dataset(arm_sim), np.array([finger], dtype=np.float32)], axis=0)


def _compose_full_target(current_full: np.ndarray, arm_sim_target: np.ndarray, finger_target: float) -> np.ndarray:
    targets = np.asarray(current_full, dtype=np.float32).copy()
    if targets.size < int(RIGHT_FINGER_INDICES[-1]) + 1:
        raise RuntimeError(f"Robot DOF is {targets.size}, expected at least 18 for OpenArm.")
    targets[RIGHT_ARM_INDICES] = np.asarray(arm_sim_target, dtype=np.float32)[:7]
    targets[RIGHT_FINGER_INDICES] = float(finger_target)
    return targets


def _apply_joint_targets(robot: Any, targets: np.ndarray, physics_control: bool) -> None:
    if physics_control:
        from omni.isaac.core.utils.types import ArticulationAction

        robot.apply_action(ArticulationAction(joint_positions=np.asarray(targets, dtype=np.float32)))
    else:
        robot.set_joint_positions(np.asarray(targets, dtype=np.float32))


def _configure_cube_physics(stage: Any, cube_prim_path: Optional[str]) -> None:
    if not cube_prim_path:
        return
    from pxr import UsdPhysics, UsdShade

    cube_prim = stage.GetPrimAtPath(cube_prim_path)
    if not cube_prim or not cube_prim.IsValid():
        return

    mat_path = f"{cube_prim_path}/GripMaterial"
    mat_prim = stage.DefinePrim(mat_path, "Material")
    UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat = UsdPhysics.MaterialAPI(mat_prim)
    phys_mat.CreateStaticFrictionAttr(10.0)
    phys_mat.CreateDynamicFrictionAttr(10.0)
    phys_mat.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(cube_prim).Bind(
        UsdShade.Material(mat_prim),
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )
    if cube_prim.HasAPI(UsdPhysics.MassAPI):
        UsdPhysics.MassAPI(cube_prim).CreateMassAttr(0.01)


def _configure_bowl_collision(stage: Any, bowl_prim_path: Optional[str]) -> None:
    if not bowl_prim_path:
        return
    from pxr import Usd, UsdGeom, UsdPhysics

    bowl_prim = stage.GetPrimAtPath(bowl_prim_path)
    if not bowl_prim or not bowl_prim.IsValid():
        return

    for child in Usd.PrimRange(bowl_prim):
        if child.IsA(UsdGeom.Mesh):
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(child)
            if not child.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(child)
            child.GetAttribute("physics:approximation").Set("none")


def _object_xy_adjust(delta_xy: np.ndarray) -> np.ndarray:
    dx = float(delta_xy[0])
    dy = float(delta_xy[1])
    return np.array(
        [
            -2.2 * dy,
            -2.4 * dx,
            1.2 * dy,
            -1.8 * dx,
            0.4 * dy,
            -1.1 * dy,
            0.0,
        ],
        dtype=np.float32,
    )


def _build_episode_waypoints(cube_pos: np.ndarray, bowl_pos: np.ndarray) -> list[tuple[str, np.ndarray, float, int]]:
    cube_delta = np.asarray(cube_pos[:2] - DEFAULT_CUBE_POS[:2], dtype=np.float32)
    bowl_delta = np.asarray(bowl_pos[:2] - DEFAULT_BOWL_POS[:2], dtype=np.float32)

    approach = BASE_APPROACH_RIGHT + _object_xy_adjust(cube_delta)
    grasp = BASE_GRASP_RIGHT + 1.1 * _object_xy_adjust(cube_delta)
    lift = BASE_LIFT_RIGHT + 0.7 * _object_xy_adjust(cube_delta)
    bowl = BASE_BOWL_RIGHT + 0.8 * _object_xy_adjust(bowl_delta)
    place = BASE_PLACE_RIGHT + 1.0 * _object_xy_adjust(bowl_delta)

    return [
        ("home_open", HOME_RIGHT_ARM.copy(), GRIPPER_OPEN, 12),
        ("approach", approach, GRIPPER_OPEN, 24),
        ("grasp", grasp, GRIPPER_OPEN, 18),
        ("close", grasp, GRIPPER_CLOSED, 10),
        ("lift", lift, GRIPPER_CLOSED, 18),
        ("move_to_bowl", bowl, GRIPPER_CLOSED, 26),
        ("place_open", place, GRIPPER_OPEN, 12),
        ("release_hold", place, GRIPPER_OPEN, 10),
        ("return_home", HOME_RIGHT_ARM.copy(), GRIPPER_OPEN, 24),
    ]


def _create_attachment_joint(stage: Any, eef_prim_path: str, cube_prim_path: str) -> bool:
    from pxr import UsdPhysics

    _remove_prim_if_exists(stage, ATTACHMENT_JOINT_PATH)
    joint = UsdPhysics.FixedJoint.Define(stage, ATTACHMENT_JOINT_PATH)
    joint.GetBody0Rel().SetTargets([eef_prim_path])
    joint.GetBody1Rel().SetTargets([cube_prim_path])
    return True


def _should_attach(stage: Any, eef_prim_path: str, cube_prim_path: Optional[str]) -> bool:
    if not cube_prim_path:
        return False
    eef_pos, _ = _get_prim_world_pose(stage, eef_prim_path)
    cube_pos, _ = _get_prim_world_pose(stage, cube_prim_path)
    xy_dist = float(np.linalg.norm(eef_pos[:2] - cube_pos[:2]))
    z_dist = abs(float(eef_pos[2] - cube_pos[2]))
    return xy_dist <= 0.10 and z_dist <= 0.16


def _episode_success(stage: Any, cube_prim_path: Optional[str], bowl_prim_path: Optional[str]) -> bool:
    if not cube_prim_path or not bowl_prim_path:
        return True
    cube_pos, _ = _get_prim_world_pose(stage, cube_prim_path)
    bowl_pos, _ = _get_prim_world_pose(stage, bowl_prim_path)
    xy_dist = float(np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))
    z_delta = float(cube_pos[2] - bowl_pos[2])
    return xy_dist <= 0.14 and z_delta <= 0.18


def _move_cube_for_episode(
    stage: Any,
    cube_prim_path: Optional[str],
    cube_base_pos: np.ndarray,
    rng: np.random.Generator,
    object_position_noise: float,
) -> np.ndarray:
    if cube_prim_path is None:
        noise = rng.uniform(-object_position_noise, object_position_noise, size=2).astype(np.float32)
        return cube_base_pos + np.array([noise[0], noise[1], 0.0], dtype=np.float32)

    cube_pos, _ = _get_prim_world_pose(stage, cube_prim_path)
    target = np.array(cube_base_pos, dtype=np.float32, copy=True)
    noise = rng.uniform(-object_position_noise, object_position_noise, size=2).astype(np.float32)
    target[0] += noise[0]
    target[1] += noise[1]
    if cube_pos.size >= 3:
        target[2] = float(cube_pos[2])
    _set_prim_world_position(stage, cube_prim_path, target)
    return target


def _setup_scene_context(
    world: Any,
    simulation_app: Any,
) -> CollectorContext:
    from omni.isaac.core.articulations import Articulation

    if world is not None and hasattr(world, "play"):
        try:
            world.play()
        except Exception:
            pass
    _step_world(world, simulation_app, render=True, steps=2)

    stage = _find_stage(simulation_app)
    robot_prim_path = _select_robot_prim(stage)
    robot = Articulation(robot_prim_path)
    if hasattr(robot, "initialize"):
        robot.initialize()
    _step_world(world, simulation_app, render=True, steps=2)

    openarm_root = _openarm_root_from_robot_path(robot_prim_path)
    camera_cfg = _load_right_wrist_camera_cfg()
    right_camera_path, right_eef_prim_path = _resolve_right_wrist_camera(stage, openarm_root, camera_cfg)

    cube_prim_path = _find_named_prim(stage, ("cube", "cuboid"))
    bowl_prim_path = _find_named_prim(stage, ("bowl",))

    cube_base_pos = _get_prim_world_pose(stage, cube_prim_path)[0] if cube_prim_path else DEFAULT_CUBE_POS.copy()
    bowl_base_pos = _get_prim_world_pose(stage, bowl_prim_path)[0] if bowl_prim_path else DEFAULT_BOWL_POS.copy()

    _configure_cube_physics(stage, cube_prim_path)
    _configure_bowl_collision(stage, bowl_prim_path)

    render_product, annotator = _setup_right_wrist_camera(stage, right_camera_path, camera_cfg)
    _step_world(world, simulation_app, render=True, steps=3)

    return CollectorContext(
        stage=stage,
        robot=robot,
        robot_prim_path=robot_prim_path,
        openarm_root=openarm_root,
        right_eef_prim_path=right_eef_prim_path,
        cube_prim_path=cube_prim_path,
        bowl_prim_path=bowl_prim_path,
        right_wrist_camera_path=right_camera_path,
        right_wrist_render_product=render_product,
        right_wrist_annotator=annotator,
        cube_base_pos=np.asarray(cube_base_pos, dtype=np.float32),
        bowl_base_pos=np.asarray(bowl_base_pos, dtype=np.float32),
    )


def _run_episode(
    episode_index: int,
    world: Any,
    simulation_app: Any,
    ctx: CollectorContext,
    writer: SimLeRobotWriter,
    fps: int,
    steps_per_segment: int,
    rng: np.random.Generator,
    stop_event: threading.Event,
    episode_timeout_sec: float,
    object_position_noise: float,
) -> tuple[int, bool, bool]:
    _remove_prim_if_exists(ctx.stage, ATTACHMENT_JOINT_PATH)
    _world_reset(world, simulation_app)

    cube_pos = _move_cube_for_episode(
        stage=ctx.stage,
        cube_prim_path=ctx.cube_prim_path,
        cube_base_pos=ctx.cube_base_pos,
        rng=rng,
        object_position_noise=object_position_noise,
    )
    bowl_pos = ctx.bowl_base_pos.copy()
    _step_world(world, simulation_app, render=True, steps=6)

    current = _to_numpy(ctx.robot.get_joint_positions())
    if current.size < 18:
        raise RuntimeError(f"Expected OpenArm articulation with 18 DOF, got {current.size}.")

    home_targets = _compose_full_target(current, HOME_RIGHT_ARM, GRIPPER_OPEN)
    _apply_joint_targets(ctx.robot, home_targets, physics_control=False)
    _step_world(world, simulation_app, render=True, steps=10)

    waypoints = _build_episode_waypoints(cube_pos=cube_pos, bowl_pos=bowl_pos)
    frame_index = 0
    attached = False
    episode_start = time.time()

    for transition_index, (start_wp, end_wp) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        start_name, start_arm, start_gripper, _ = start_wp
        end_name, end_arm, end_gripper, end_steps = end_wp
        seg_steps = max(
            1,
            int(round(float(end_steps) * float(steps_per_segment) / float(STEPS_PER_SEGMENT))),
        )

        for step in range(seg_steps):
            if stop_event.is_set():
                return frame_index, True, False
            if episode_timeout_sec > 0.0 and (time.time() - episode_start) > episode_timeout_sec:
                LOG.warning("Episode %d timed out after %.2fs", episode_index, time.time() - episode_start)
                return frame_index, False, False

            alpha = float(step + 1) / float(seg_steps)
            arm_target = ((1.0 - alpha) * start_arm + alpha * end_arm).astype(np.float32)
            gripper_target = float((1.0 - alpha) * start_gripper + alpha * end_gripper)
            current_full = _to_numpy(ctx.robot.get_joint_positions())
            full_target = _compose_full_target(current_full, arm_target, gripper_target)

            _apply_joint_targets(ctx.robot, full_target, physics_control=False)
            _step_world(world, simulation_app, render=True, steps=1)

            obs_state = _full_to_dataset_state(_to_numpy(ctx.robot.get_joint_positions()))
            action = np.concatenate(
                [_sim_arm_to_dataset(arm_target), np.array([gripper_target], dtype=np.float32)],
                axis=0,
            )
            is_last = transition_index == (len(waypoints) - 2) and step == (seg_steps - 1)
            writer.add_frame(
                episode_index=episode_index,
                frame_index=frame_index,
                observation_state=obs_state,
                action=action.astype(np.float32),
                timestamp=frame_index / float(fps),
                next_done=is_last,
            )
            writer.add_video_frame("right_wrist_cam", _capture_rgb(ctx.right_wrist_annotator))
            frame_index += 1

        if end_name == "close" and not attached and _should_attach(ctx.stage, ctx.right_eef_prim_path, ctx.cube_prim_path):
            if ctx.cube_prim_path:
                attached = _create_attachment_joint(ctx.stage, ctx.right_eef_prim_path, ctx.cube_prim_path)
                _step_world(world, simulation_app, render=True, steps=4)

        if end_name == "place_open" and attached:
            _remove_prim_if_exists(ctx.stage, ATTACHMENT_JOINT_PATH)
            attached = False
            _step_world(world, simulation_app, render=True, steps=4)

        LOG.info(
            "Episode %d transition %s -> %s done (%d frames)",
            episode_index + 1,
            start_name,
            end_name,
            seg_steps,
        )

    _remove_prim_if_exists(ctx.stage, ATTACHMENT_JOINT_PATH)
    _step_world(world, simulation_app, render=True, steps=6)
    return frame_index, False, _episode_success(ctx.stage, ctx.cube_prim_path, ctx.bowl_prim_path)


def run_collection_in_process(
    world: Any,
    simulation_app: Any,
    num_episodes: int,
    output_dir: str,
    stop_event: threading.Event,
    progress_callback: Callable[[int], None] | None = None,
    fps: int = FPS,
    steps_per_segment: int = STEPS_PER_SEGMENT,
    seed: int = 12345,
    task_name: str = "pick_place",
    scene_mode: str = "existing",
    target_objects: Optional[Sequence[str]] = None,
    dataset_repo_id: Optional[str] = None,
    episode_timeout_sec: float | None = None,
    reset_mode: str = "full",
    rounds_per_episode: int = 1,
    object_position_noise: float = DEFAULT_OBJECT_POSITION_NOISE,
) -> dict[str, Any]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if steps_per_segment <= 0:
        raise ValueError("steps_per_segment must be > 0")
    if stop_event is None:
        stop_event = threading.Event()
    if episode_timeout_sec is None:
        episode_timeout_sec = DEFAULT_EPISODE_TIMEOUT_SEC

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    ctx = _setup_scene_context(world=world, simulation_app=simulation_app)
    writer = SimLeRobotWriter(
        output_dir=output_dir,
        repo_id=(str(dataset_repo_id).strip() if dataset_repo_id else _dataset_repo_id(output_dir)),
        fps=fps,
        robot_type=ROBOT_TYPE,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        camera_names=CAMERA_NAMES,
        camera_resolution=CAMERA_RESOLUTION,
        state_names=STATE_NAMES,
        action_names=ACTION_NAMES,
    )

    if rounds_per_episode != 1:
        LOG.warning("OpenArm collector ignores rounds_per_episode=%s and always records one cycle per episode.", rounds_per_episode)
    if target_objects:
        LOG.info("OpenArm collector ignores target_objects=%s and uses the scene cube/bowl.", list(target_objects))
    if reset_mode not in {"full", "arm_only"}:
        LOG.warning("Unknown reset_mode=%s, continuing with simple full reset.", reset_mode)

    completed = 0
    successful_episodes = 0
    stopped = False

    try:
        for episode_index in range(num_episodes):
            if stop_event.is_set():
                stopped = True
                break

            LOG.info(
                "Episode %d/%d: cube=%s bowl=%s camera=%s scene_mode=%s",
                episode_index + 1,
                num_episodes,
                ctx.cube_prim_path,
                ctx.bowl_prim_path,
                ctx.right_wrist_camera_path,
                scene_mode,
            )

            episode_frames = 0
            episode_success = False
            try:
                episode_frames, episode_stopped, episode_success = _run_episode(
                    episode_index=episode_index,
                    world=world,
                    simulation_app=simulation_app,
                    ctx=ctx,
                    writer=writer,
                    fps=fps,
                    steps_per_segment=steps_per_segment,
                    rng=rng,
                    stop_event=stop_event,
                    episode_timeout_sec=float(max(0.0, episode_timeout_sec)),
                    object_position_noise=float(max(0.0, object_position_noise)),
                )
                stopped = stopped or episode_stopped
            finally:
                writer.finish_episode(
                    episode_index=episode_index,
                    length=episode_frames,
                    task=task_name,
                    success=episode_success,
                )

            completed += 1
            if episode_success:
                successful_episodes += 1
            if progress_callback is not None:
                progress_callback(completed)
            if stopped:
                break

        writer.finalize()
    except Exception:
        writer.finalize()
        raise
    finally:
        _remove_prim_if_exists(ctx.stage, ATTACHMENT_JOINT_PATH)
        try:
            ctx.right_wrist_render_product.destroy()
        except Exception:
            pass

    return {
        "completed": completed,
        "successful_episodes": successful_episodes,
        "failed_episodes": max(0, completed - successful_episodes),
        "requested": num_episodes,
        "output_dir": output_dir,
        "stopped": bool(stopped or stop_event.is_set()),
        "scene_mode": str(scene_mode or "existing"),
        "target_object": ctx.cube_prim_path,
        "episode_timeout_sec": float(episode_timeout_sec),
    }
