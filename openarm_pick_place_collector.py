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
import re
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
HOME_FULL[10] = -0.7  # L-J6: wrist yaw for left cam overview
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

# Waypoints extracted from successful replay data (ep0 of openarm_pick_sim_cam)
# Dataset joint order: [j1, j3, j4, j5, j6, j2, j7, finger]
WP_HOME = np.array([-0.0399, 0.0299, -0.0429, 0.0135, -0.4461, 0.0357, -1.4796, 0.0397], dtype=np.float32)
WP_APPROACH = np.array([-0.0937, -0.1349, 0.5159, 0.0505, -0.2317, 0.1345, -1.1534, 0.0396], dtype=np.float32)
WP_PRE_GRASP = np.array([-0.0055, -0.2199, 0.5159, 0.0517, -0.2241, 0.1375, -1.0065, 0.0396], dtype=np.float32)
WP_GRASP = np.array([-0.0055, -0.2470, 0.5159, 0.0525, -0.2237, 0.1379, -0.9657, 0.0212], dtype=np.float32)
WP_CLOSE = np.array([-0.0055, -0.2470, 0.5159, 0.0521, -0.2237, 0.1379, -0.9607, 0.0140], dtype=np.float32)
WP_LIFT = np.array([0.0673, -0.4568, 1.0771, 0.0559, -0.2241, 0.1520, -0.6456, 0.0134], dtype=np.float32)
WP_MOVE_BOWL = np.array([0.2470, -0.6949, 0.8532, 0.0689, -0.2058, 0.1287, -0.6464, 0.0395], dtype=np.float32)
WP_PLACE = np.array([-0.2424, -0.3904, 1.1427, 0.0711, -0.2165, 0.2211, -0.6666, 0.0398], dtype=np.float32)
WP_OPEN = np.array([-0.2806, -0.3325, 1.1427, 0.0746, -0.2169, 0.2207, -0.7414, 0.0398], dtype=np.float32)
WP_RETRACT = np.array([-0.4137, -0.2150, 1.0260, 0.1959, -0.2173, 0.2180, -0.9600, 0.0398], dtype=np.float32)

# Waypoint sequence: (name, dataset-order-8dim, steps_per_transition)
PICK_PLACE_WAYPOINTS = [
    ("HOME", WP_HOME, 30),
    ("APPROACH", WP_APPROACH, 50),
    ("PRE_GRASP", WP_PRE_GRASP, 40),
    ("GRASP", WP_GRASP, 30),
    ("CLOSE", WP_CLOSE, 40),        # gripper closing
    ("LIFT", WP_LIFT, 50),
    ("MOVE_BOWL", WP_MOVE_BOWL, 50),
    ("PLACE", WP_PLACE, 40),
    ("OPEN", WP_OPEN, 30),          # gripper opening
    ("RETRACT", WP_RETRACT, 30),
    ("HOME_END", WP_HOME, 40),
]

DEFAULT_CUROBO_CONFIG_PATH = Path("/data/embodied/asset/robots/openarm_bimanual/openarm.yml")
DEFAULT_CUROBO_URDF_PATH = Path("/data/embodied/asset/robots/openarm_bimanual/openarm_bimanual.urdf")
DEFAULT_CUROBO_TO_SIM_ARM = np.array([0, 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 13], dtype=np.int64)
RIGHT_ARM_CUROBO_SLICE = slice(7, 14)
LEFT_ARM_CUROBO_SLICE = slice(0, 7)
CUROBO_RIGHT_EE_LINK = "openarm_right_hand"
# Grasp orientation from replay data frame 260 (side-approach, not top-down)
# ee_link quat when gripper is at cube: wxyz = (0.0505, 0.7232, 0.1517, 0.6719)
GRASP_QUAT_WXYZ = np.array([0.0505, 0.7232, 0.1517, 0.6719], dtype=np.float32)
# Pre-grasp: offset back along approach direction (not straight up)
PRE_GRASP_OFFSET = np.array([-0.08, 0.0, 0.06], dtype=np.float32)  # back + up from cube
GRASP_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # at cube position
LIFT_OFFSET = np.array([0.0, 0.0, 0.15], dtype=np.float32)
BOWL_APPROACH_OFFSET = np.array([0.0, 0.0, 0.15], dtype=np.float32)
PLACE_LOWER_OFFSET = np.array([0.0, 0.0, 0.05], dtype=np.float32)
LINEAR_CARTESIAN_WAYPOINTS = 20
GRIPPER_CLOSE_STEPS = 30
GRIPPER_OPEN_STEPS = 15
PRE_GRASP_IK_WAYPOINTS = 14
LIFT_IK_WAYPOINTS = 12
PLACE_IK_WAYPOINTS = 12
CUROBO_SETTLE_STEPS = 6


@dataclass(frozen=True)
class PoseSample:
    position_world: np.ndarray
    orientation: np.ndarray


@dataclass
class CollectorContext:
    stage: Any
    robot: Any
    robot_prim_path: str
    openarm_root: str
    right_eef_prim_path: str
    cube_prim_path: Optional[str]
    bowl_prim_path: Optional[str]
    table_prim_path: Optional[str]
    right_wrist_camera_path: str
    right_wrist_render_product: Any
    right_wrist_annotator: Any
    cube_base_pos: np.ndarray
    bowl_base_pos: np.ndarray
    pose_samples: dict[str, PoseSample]
    curobo_state: Optional[dict[str, Any]]


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


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = _to_numpy(quat, dtype=np.float32).reshape(-1)
    if q.size < 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q4 = q[:4].astype(np.float32, copy=False)
    if not np.all(np.isfinite(q4)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    norm = float(np.linalg.norm(q4))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q4 / norm).astype(np.float32)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = _normalize_quat_wxyz(a)
    bw, bx, by, bz = _normalize_quat_wxyz(b)
    return _normalize_quat_wxyz(
        np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float32,
        )
    )


def _quat_to_rot_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = _normalize_quat_wxyz(q0).astype(np.float64)
    q1 = _normalize_quat_wxyz(q1).astype(np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return _normalize_quat_wxyz(result.astype(np.float32))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    result = (np.sin((1.0 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1
    return _normalize_quat_wxyz(result.astype(np.float32))


def _world_to_robot_frame(
    target_pos_world: np.ndarray,
    target_quat_world: np.ndarray,
    robot_base_pos: np.ndarray,
    robot_base_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rot_inv = _quat_to_rot_wxyz(robot_base_quat).T
    pos_robot = (rot_inv @ (_to_numpy(target_pos_world, dtype=np.float32)[:3] - robot_base_pos[:3])).astype(np.float32)
    base_q_inv = _normalize_quat_wxyz(robot_base_quat).copy()
    base_q_inv[1:] *= -1.0
    quat_robot = _quat_mul_wxyz(base_q_inv, target_quat_world)
    return pos_robot, quat_robot


def _compute_prim_bbox(stage: Any, prim_path: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None

    tc = Usd.TimeCode.Default()

    def _xform_extent_corners(lo: Any, hi: Any, tf: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
        corners = [
            Gf.Vec3d(lo[0], lo[1], lo[2]),
            Gf.Vec3d(hi[0], lo[1], lo[2]),
            Gf.Vec3d(lo[0], hi[1], lo[2]),
            Gf.Vec3d(lo[0], lo[1], hi[2]),
            Gf.Vec3d(hi[0], hi[1], lo[2]),
            Gf.Vec3d(hi[0], lo[1], hi[2]),
            Gf.Vec3d(lo[0], hi[1], hi[2]),
            Gf.Vec3d(hi[0], hi[1], hi[2]),
        ]
        pts = [tf.Transform(corner) for corner in corners]
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        zs = [float(p[2]) for p in pts]
        mn = np.array([min(xs), min(ys), min(zs)], dtype=np.float32)
        mx = np.array([max(xs), max(ys), max(zs)], dtype=np.float32)
        if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
            return None
        return mn, mx

    def _try_extent(prim_obj: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
        xform = UsdGeom.Xformable(prim_obj)
        wtf = xform.ComputeLocalToWorldTransform(tc)
        if prim_obj.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim_obj)
            size_val = cube.GetSizeAttr().Get(tc)
            if size_val is None:
                size_val = 2.0
            half = float(size_val) / 2.0
            return _xform_extent_corners(Gf.Vec3d(-half, -half, -half), Gf.Vec3d(half, half, half), wtf)
        extent_attr = prim_obj.GetAttribute("extent")
        if not extent_attr or not extent_attr.HasValue():
            return None
        extent_val = extent_attr.Get(tc)
        if extent_val is None or len(extent_val) < 2:
            return None
        lo = Gf.Vec3d(float(extent_val[0][0]), float(extent_val[0][1]), float(extent_val[0][2]))
        hi = Gf.Vec3d(float(extent_val[1][0]), float(extent_val[1][1]), float(extent_val[1][2]))
        return _xform_extent_corners(lo, hi, wtf)

    result = _try_extent(prim)
    if result is not None:
        return result

    merged_min: Optional[np.ndarray] = None
    merged_max: Optional[np.ndarray] = None
    for child in Usd.PrimRange(prim):
        if child == prim or not child.IsValid():
            continue
        child_bbox = _try_extent(child)
        if child_bbox is None:
            continue
        child_min, child_max = child_bbox
        merged_min = child_min if merged_min is None else np.minimum(merged_min, child_min)
        merged_max = child_max if merged_max is None else np.maximum(merged_max, child_max)
    if merged_min is None or merged_max is None:
        return None
    return merged_min.astype(np.float32), merged_max.astype(np.float32)


def _joint_side_and_index(name: str) -> tuple[Optional[str], Optional[int]]:
    low = str(name or "").lower()
    if "finger" in low:
        return None, None
    side: Optional[str] = None
    if "left" in low:
        side = "left"
    elif "right" in low:
        side = "right"
    match = re.search(r"joint[_]?(\d+)", low) or re.search(r"(\d+)$", low)
    index = int(match.group(1)) if match else None
    return side, index


def _joint_key(name: str) -> Optional[str]:
    side, index = _joint_side_and_index(name)
    if side is None or index is None:
        return None
    return f"{side}_{index}"


def _get_robot_dof_names(robot: Any) -> list[str]:
    names = getattr(robot, "dof_names", None)
    if names is None:
        return []
    return [str(name) for name in names]


def _resolve_openarm_arm_mapping(robot: Any, curobo_joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim_name_to_index: dict[str, int] = {}
    for index, name in enumerate(_get_robot_dof_names(robot)):
        key = _joint_key(name)
        if key is not None:
            sim_name_to_index[key] = index

    mapped: list[int] = []
    left_indices: list[int] = []
    right_indices: list[int] = []
    mapping_ok = True
    for curobo_index, name in enumerate(curobo_joint_names):
        key = _joint_key(str(name))
        sim_index = sim_name_to_index.get(key) if key is not None else None
        if sim_index is None:
            mapping_ok = False
            break
        mapped.append(sim_index)
        if key.startswith("left_"):
            left_indices.append(curobo_index)
        elif key.startswith("right_"):
            right_indices.append(curobo_index)

    if mapping_ok and mapped:
        return (
            np.asarray(mapped, dtype=np.int64),
            np.asarray(left_indices, dtype=np.int64),
            np.asarray(right_indices, dtype=np.int64),
        )

    LOG.warning("Falling back to default OpenArm sim↔cuRobo joint order mapping.")
    return (
        DEFAULT_CUROBO_TO_SIM_ARM.copy(),
        np.arange(LEFT_ARM_CUROBO_SLICE.start, LEFT_ARM_CUROBO_SLICE.stop, dtype=np.int64),
        np.arange(RIGHT_ARM_CUROBO_SLICE.start, RIGHT_ARM_CUROBO_SLICE.stop, dtype=np.int64),
    )


def _resolve_openarm_curobo_paths() -> tuple[Path, Optional[Path]]:
    config_candidates = [
        Path(os.environ.get("OPENARM_CUROBO_CONFIG", "")).expanduser() if os.environ.get("OPENARM_CUROBO_CONFIG") else None,
        DEFAULT_CUROBO_CONFIG_PATH,
        SCRIPT_DIR / "openarm.yml",
    ]
    config_path = next((path for path in config_candidates if path is not None and path.exists()), None)
    if config_path is None:
        raise FileNotFoundError(f"OpenArm cuRobo config not found. Tried: {[str(p) for p in config_candidates if p is not None]}")

    urdf_candidates = [
        Path(os.environ.get("OPENARM_CUROBO_URDF", "")).expanduser() if os.environ.get("OPENARM_CUROBO_URDF") else None,
        DEFAULT_CUROBO_URDF_PATH,
        config_path.parent / "openarm_bimanual.urdf",
        config_path.parent / "robot" / "openarm_config" / "openarm_bimanual.urdf",
    ]
    urdf_path = next((path for path in urdf_candidates if path is not None and path.exists()), None)
    return config_path, urdf_path


def _build_openarm_curobo_robot_cfg(config_path: Path, urdf_path: Optional[Path], left_lock_joints: dict[str, float]) -> dict[str, Any]:
    from curobo.util_file import load_yaml

    robot_yaml = load_yaml(str(config_path))
    robot_cfg = dict(robot_yaml.get("robot_cfg") or {})
    kin_cfg = dict(robot_cfg.get("kinematics") or {})
    kin_cfg["ee_link"] = CUROBO_RIGHT_EE_LINK
    kin_cfg["link_names"] = [CUROBO_RIGHT_EE_LINK]
    kin_cfg["lock_joints"] = {str(name): float(value) for name, value in left_lock_joints.items()}
    kin_cfg["asset_root_path"] = str(config_path.parent)
    if urdf_path is not None:
        kin_cfg["urdf_path"] = str(urdf_path)
    robot_cfg["kinematics"] = kin_cfg
    return robot_cfg


def _extract_curobo_world_config(
    stage: Any,
    robot_prim_path: str,
    table_prim_path: Optional[str],
    include_table: bool = True,
) -> Any:
    from curobo.geom.types import Cuboid, WorldConfig

    cuboids: list[Cuboid] = []
    robot_pos, robot_quat = _get_prim_world_pose(stage, robot_prim_path)
    rot_inv = _quat_to_rot_wxyz(robot_quat).T

    if include_table and table_prim_path:
        bbox = _compute_prim_bbox(stage, table_prim_path)
        if bbox is not None:
            mn, mx = bbox
            dims = (mx - mn).astype(np.float32)
            center_world = 0.5 * (mn + mx)
            center_robot = (rot_inv @ (center_world - robot_pos[:3])).astype(np.float32)
            center_robot[2] -= 0.02
            cuboids.append(
                Cuboid(
                    name="table",
                    pose=[
                        float(center_robot[0]),
                        float(center_robot[1]),
                        float(center_robot[2]),
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dims=[float(dims[0]), float(dims[1]), float(dims[2])],
                )
            )

    if not cuboids:
        cuboids.append(
            Cuboid(
                name="dummy",
                pose=[0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.0],
                dims=[0.01, 0.01, 0.01],
            )
        )
    return WorldConfig(cuboid=cuboids)


def _sample_pose_at_arm_target(
    robot: Any,
    stage: Any,
    eef_prim_path: str,
    world: Any,
    simulation_app: Any,
    arm_target: np.ndarray,
    finger_target: float = GRIPPER_OPEN,
) -> PoseSample:
    current_full = _to_numpy(robot.get_joint_positions())
    full_target = _compose_full_target(current_full, arm_target, finger_target)
    _apply_joint_targets(robot, full_target, physics_control=False)
    _step_world(world, simulation_app, render=True, steps=4)
    pos, quat = _get_prim_world_pose(stage, eef_prim_path)
    return PoseSample(position_world=pos.astype(np.float32), orientation=_normalize_quat_wxyz(quat))


def _sample_pose_references(
    world: Any,
    simulation_app: Any,
    robot: Any,
    stage: Any,
    eef_prim_path: str,
) -> dict[str, PoseSample]:
    current_full = _to_numpy(robot.get_joint_positions())
    samples = {
        "home": _sample_pose_at_arm_target(robot, stage, eef_prim_path, world, simulation_app, HOME_RIGHT_ARM, GRIPPER_OPEN),
        "pre_grasp": _sample_pose_at_arm_target(robot, stage, eef_prim_path, world, simulation_app, BASE_APPROACH_RIGHT, GRIPPER_OPEN),
        "grasp": _sample_pose_at_arm_target(robot, stage, eef_prim_path, world, simulation_app, BASE_GRASP_RIGHT, GRIPPER_OPEN),
        "lift": _sample_pose_at_arm_target(robot, stage, eef_prim_path, world, simulation_app, BASE_LIFT_RIGHT, GRIPPER_CLOSED),
        "bowl": _sample_pose_at_arm_target(robot, stage, eef_prim_path, world, simulation_app, BASE_BOWL_RIGHT, GRIPPER_CLOSED),
        "place": _sample_pose_at_arm_target(robot, stage, eef_prim_path, world, simulation_app, BASE_PLACE_RIGHT, GRIPPER_CLOSED),
    }
    _apply_joint_targets(robot, current_full, physics_control=False)
    _step_world(world, simulation_app, render=True, steps=4)
    return samples


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


def _compose_home_full_target(current_full: np.ndarray, finger_target: float = GRIPPER_OPEN) -> np.ndarray:
    targets = np.asarray(current_full, dtype=np.float32).copy()
    limit = min(targets.size, HOME_FULL.size)
    targets[:limit] = HOME_FULL[:limit]
    targets[RIGHT_FINGER_INDICES] = float(finger_target)
    return targets


def _record_frame(
    writer: SimLeRobotWriter,
    ctx: CollectorContext,
    world: Any,
    simulation_app: Any,
    episode_index: int,
    frame_index: int,
    arm_target: Optional[np.ndarray] = None,
    gripper_target: Optional[float] = None,
    next_done: bool = False,
    fps: int = FPS,
) -> int:
    del world, simulation_app
    current_full = _to_numpy(ctx.robot.get_joint_positions())
    obs_state = _full_to_dataset_state(current_full)
    if arm_target is None:
        arm_target = current_full[RIGHT_ARM_INDICES]
    if gripper_target is None:
        gripper_target = float(np.mean(current_full[RIGHT_FINGER_INDICES]))
    action = np.concatenate(
        [
            _sim_arm_to_dataset(np.asarray(arm_target, dtype=np.float32)),
            np.array([gripper_target], dtype=np.float32),
        ],
        axis=0,
    )
    writer.add_frame(
        episode_index=episode_index,
        frame_index=frame_index,
        observation_state=obs_state,
        action=action.astype(np.float32),
        timestamp=frame_index / float(fps),
        next_done=next_done,
    )
    writer.add_video_frame("right_wrist_cam", _capture_rgb(ctx.right_wrist_annotator))
    return frame_index + 1


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


def _build_episode_cartesian_targets(
    cube_pos: np.ndarray,
    bowl_pos: np.ndarray,
    pose_samples: dict[str, PoseSample],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    def _sample_offset(sample_name: str, anchor: np.ndarray) -> np.ndarray:
        return (pose_samples[sample_name].position_world - anchor).astype(np.float32)

    cube_anchor = np.asarray(cube_pos, dtype=np.float32)
    bowl_anchor = np.asarray(bowl_pos, dtype=np.float32)
    return {
        "pre_grasp": (
            cube_anchor + _sample_offset("pre_grasp", DEFAULT_CUBE_POS),
            pose_samples["pre_grasp"].orientation.copy(),
        ),
        "grasp": (
            cube_anchor + _sample_offset("grasp", DEFAULT_CUBE_POS),
            pose_samples["grasp"].orientation.copy(),
        ),
        "lift": (
            cube_anchor + _sample_offset("lift", DEFAULT_CUBE_POS),
            pose_samples["lift"].orientation.copy(),
        ),
        "bowl": (
            bowl_anchor + _sample_offset("bowl", DEFAULT_BOWL_POS),
            pose_samples["bowl"].orientation.copy(),
        ),
        "place": (
            bowl_anchor + _sample_offset("place", DEFAULT_BOWL_POS),
            pose_samples["place"].orientation.copy(),
        ),
    }


def _init_openarm_curobo(
    stage: Any,
    robot: Any,
    robot_prim_path: str,
    table_prim_path: Optional[str],
) -> Optional[dict[str, Any]]:
    try:
        import torch
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    except Exception as exc:
        LOG.warning("cuRobo unavailable, using waypoint fallback only: %s", exc)
        return None

    try:
        config_path, urdf_path = _resolve_openarm_curobo_paths()
        sim_names = _get_robot_dof_names(robot)
        left_lock_joints: dict[str, float] = {}
        for sim_index, name in enumerate(sim_names):
            side, joint_idx = _joint_side_and_index(name)
            if side == "left" and joint_idx is not None and sim_index < HOME_FULL.size:
                left_lock_joints[str(name)] = float(HOME_FULL[sim_index])

        tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
        world_cfg = _extract_curobo_world_config(
            stage=stage,
            robot_prim_path=robot_prim_path,
            table_prim_path=table_prim_path,
            include_table=True,
        )
        robot_cfg = _build_openarm_curobo_robot_cfg(config_path, urdf_path, left_lock_joints)

        mg_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            [world_cfg],
            tensor_args,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            use_cuda_graph=False,
            interpolation_dt=0.03,
            collision_cache={"obb": 16, "mesh": 2},
            collision_activation_distance=0.015,
            maximum_trajectory_dt=0.25,
            n_collision_envs=1,
        )
        motion_gen = MotionGen(mg_cfg)
        motion_gen.warmup(warmup_js_trajopt=False)

        ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            tensor_args=tensor_args,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            use_cuda_graph=False,
            num_seeds=32,
            self_collision_check=True,
            self_collision_opt=True,
            position_threshold=0.005,
            rotation_threshold=0.05,
        )
        ik_solver = IKSolver(ik_cfg)

        curobo_joint_names = [str(name) for name in motion_gen.kinematics.joint_names]
        curobo_to_sim_arm, left_curobo_indices, right_curobo_indices = _resolve_openarm_arm_mapping(
            robot=robot,
            curobo_joint_names=curobo_joint_names,
        )
        home_curobo = HOME_FULL[curobo_to_sim_arm].astype(np.float32, copy=True)

        state = {
            "motion_gen": motion_gen,
            "ik_solver": ik_solver,
            "plan_config": MotionGenPlanConfig(
                enable_graph=True,
                enable_graph_attempt=3,
                max_attempts=16,
                enable_finetune_trajopt=True,
            ),
            "tensor_args": tensor_args,
            "curobo_joint_names": curobo_joint_names,
            "curobo_to_sim_arm": curobo_to_sim_arm,
            "left_curobo_indices": left_curobo_indices,
            "right_curobo_indices": right_curobo_indices,
            "home_curobo": home_curobo,
            "config_path": str(config_path),
            "urdf_path": str(urdf_path) if urdf_path is not None else None,
        }
        LOG.info(
            "OpenArm cuRobo initialised: config=%s ee_link=%s joints=%s",
            config_path,
            CUROBO_RIGHT_EE_LINK,
            curobo_joint_names,
        )
        return state
    except Exception as exc:
        LOG.warning("OpenArm cuRobo init failed, using waypoint fallback: %s", exc)
        return None


def _update_openarm_curobo_world(
    curobo_state: dict[str, Any],
    stage: Any,
    robot_prim_path: str,
    table_prim_path: Optional[str],
) -> None:
    world_cfg = _extract_curobo_world_config(
        stage=stage,
        robot_prim_path=robot_prim_path,
        table_prim_path=table_prim_path,
        include_table=True,
    )
    curobo_state["motion_gen"].world_coll_checker.load_collision_model(world_cfg, env_idx=0)
    curobo_state["ik_solver"].world_coll_checker.load_collision_model(world_cfg, env_idx=0)


def _full_to_curobo_arm_target(full_joint_positions: np.ndarray, curobo_state: dict[str, Any]) -> np.ndarray:
    full_joint_positions = _to_numpy(full_joint_positions, dtype=np.float32).reshape(-1)
    return full_joint_positions[curobo_state["curobo_to_sim_arm"]].astype(np.float32, copy=True)


def _compose_full_target_from_curobo(
    current_full: np.ndarray,
    curobo_arm_target: np.ndarray,
    curobo_state: dict[str, Any],
    finger_target: float,
) -> np.ndarray:
    targets = _to_numpy(current_full, dtype=np.float32).copy()
    targets[curobo_state["curobo_to_sim_arm"]] = _to_numpy(curobo_arm_target, dtype=np.float32)[: len(curobo_state["curobo_to_sim_arm"])]
    targets[RIGHT_FINGER_INDICES] = float(finger_target)
    return targets


def _solve_openarm_curobo_ik(
    curobo_state: dict[str, Any],
    current_curobo: np.ndarray,
    target_pos_world: np.ndarray,
    target_quat_world: np.ndarray,
    robot_base_pos: np.ndarray,
    robot_base_quat: np.ndarray,
) -> Optional[np.ndarray]:
    from curobo.types.math import Pose

    solver = curobo_state.get("ik_solver")
    if solver is None:
        return None

    pos_robot, quat_robot = _world_to_robot_frame(
        target_pos_world,
        target_quat_world,
        robot_base_pos,
        robot_base_quat,
    )
    ta = curobo_state["tensor_args"]
    retract = ta.to_device(_to_numpy(current_curobo, dtype=np.float32).tolist()).view(1, -1)
    goal = Pose(
        position=ta.to_device(pos_robot.tolist()).view(1, 3),
        quaternion=ta.to_device(quat_robot.tolist()).view(1, 4),
    )
    try:
        result = solver.solve_single(
            goal,
            retract_config=retract,
            seed_config=retract.view(1, 1, -1),
            return_seeds=1,
            use_nn_seed=False,
        )
    except Exception as exc:
        LOG.warning("OpenArm cuRobo IK exception: %s", exc)
        return None

    success = getattr(result, "success", None)
    if success is None or not bool(_to_numpy(success).reshape(-1)[0]):
        return None

    solution = None
    js_solution = getattr(result, "js_solution", None)
    if js_solution is not None:
        solution = getattr(js_solution, "position", None)
    if solution is None:
        solution = getattr(result, "solution", None)
    solution_np = _to_numpy(solution, dtype=np.float32).reshape(-1)
    if solution_np.size < len(curobo_state["curobo_joint_names"]):
        return None
    return solution_np[: len(curobo_state["curobo_joint_names"])].astype(np.float32, copy=True)


def _plan_openarm_curobo_segment(
    curobo_state: dict[str, Any],
    current_full: np.ndarray,
    target_pos_world: np.ndarray,
    target_quat_world: np.ndarray,
    robot_base_pos: np.ndarray,
    robot_base_quat: np.ndarray,
) -> Optional[np.ndarray]:
    from curobo.types.math import Pose
    from curobo.types.state import JointState

    motion_gen = curobo_state["motion_gen"]
    ta = curobo_state["tensor_args"]
    current_curobo = _full_to_curobo_arm_target(current_full, curobo_state)
    current_curobo[curobo_state["left_curobo_indices"]] = curobo_state["home_curobo"][curobo_state["left_curobo_indices"]]
    pos_robot, quat_robot = _world_to_robot_frame(
        target_pos_world,
        target_quat_world,
        robot_base_pos,
        robot_base_quat,
    )
    joint_state = JointState(
        position=ta.to_device(current_curobo.tolist()).view(1, -1),
        velocity=ta.to_device((current_curobo * 0.0).tolist()).view(1, -1),
        acceleration=ta.to_device((current_curobo * 0.0).tolist()).view(1, -1),
        jerk=ta.to_device((current_curobo * 0.0).tolist()).view(1, -1),
        joint_names=curobo_state["curobo_joint_names"],
    )
    goal = Pose(
        position=ta.to_device(pos_robot.tolist()).view(1, 3),
        quaternion=ta.to_device(quat_robot.tolist()).view(1, 4),
    )
    try:
        result = motion_gen.plan_single(joint_state, goal, curobo_state["plan_config"].clone())
    except Exception as exc:
        LOG.warning("OpenArm cuRobo plan exception: %s", exc)
        return None
    if result.success is None or not bool(result.success.item()):
        LOG.warning("OpenArm cuRobo plan failed: status=%s", getattr(result, "status", None))
        return None
    traj = motion_gen.get_full_js(result.get_interpolated_plan()).position.cpu().numpy().astype(np.float32)
    traj[:, curobo_state["left_curobo_indices"]] = curobo_state["home_curobo"][curobo_state["left_curobo_indices"]]
    return traj[:, : len(curobo_state["curobo_joint_names"])]


def _plan_openarm_curobo_joint_segment(
    curobo_state: dict[str, Any],
    current_full: np.ndarray,
    target_full: np.ndarray,
) -> Optional[np.ndarray]:
    from curobo.types.state import JointState

    motion_gen = curobo_state["motion_gen"]
    ta = curobo_state["tensor_args"]

    current_curobo = _full_to_curobo_arm_target(current_full, curobo_state)
    target_curobo = _full_to_curobo_arm_target(target_full, curobo_state)
    current_curobo[curobo_state["left_curobo_indices"]] = curobo_state["home_curobo"][curobo_state["left_curobo_indices"]]
    target_curobo[curobo_state["left_curobo_indices"]] = curobo_state["home_curobo"][curobo_state["left_curobo_indices"]]

    current_state = JointState(
        position=ta.to_device(current_curobo.tolist()).view(1, -1),
        velocity=ta.to_device((current_curobo * 0.0).tolist()).view(1, -1),
        acceleration=ta.to_device((current_curobo * 0.0).tolist()).view(1, -1),
        jerk=ta.to_device((current_curobo * 0.0).tolist()).view(1, -1),
        joint_names=curobo_state["curobo_joint_names"],
    )
    goal_state = JointState(
        position=ta.to_device(target_curobo.tolist()).view(1, -1),
        velocity=ta.to_device((target_curobo * 0.0).tolist()).view(1, -1),
        acceleration=ta.to_device((target_curobo * 0.0).tolist()).view(1, -1),
        jerk=ta.to_device((target_curobo * 0.0).tolist()).view(1, -1),
        joint_names=curobo_state["curobo_joint_names"],
    )

    result = None
    for method_name in ("plan_single_js", "plan_js"):
        method = getattr(motion_gen, method_name, None)
        if not callable(method):
            continue
        try:
            result = method(current_state, goal_state, curobo_state["plan_config"].clone())
            break
        except Exception as exc:
            LOG.warning("OpenArm cuRobo %s exception: %s", method_name, exc)
    if result is None:
        LOG.warning("OpenArm cuRobo joint-space API unavailable, using linear joint interpolation for return-home.")
        return np.linspace(current_curobo, target_curobo, num=32, endpoint=True).astype(np.float32)

    success = getattr(result, "success", None)
    if success is None or not bool(_to_numpy(success).reshape(-1)[0]):
        LOG.warning("OpenArm cuRobo joint-space plan failed: status=%s", getattr(result, "status", None))
        return None

    traj = motion_gen.get_full_js(result.get_interpolated_plan()).position.cpu().numpy().astype(np.float32)
    traj[:, curobo_state["left_curobo_indices"]] = curobo_state["home_curobo"][curobo_state["left_curobo_indices"]]
    return traj[:, : len(curobo_state["curobo_joint_names"])]


def _plan_openarm_linear_ik_segment(
    curobo_state: dict[str, Any],
    current_full: np.ndarray,
    stage: Any,
    eef_prim_path: str,
    target_pos_world: np.ndarray,
    target_quat_world: np.ndarray,
    robot_base_pos: np.ndarray,
    robot_base_quat: np.ndarray,
    n_waypoints: int,
) -> Optional[np.ndarray]:
    start_pos, start_quat = _get_prim_world_pose(stage, eef_prim_path)
    prev_q = _full_to_curobo_arm_target(current_full, curobo_state)
    prev_q[curobo_state["left_curobo_indices"]] = curobo_state["home_curobo"][curobo_state["left_curobo_indices"]]
    waypoints: list[np.ndarray] = []
    max_joint_jump = 0.5
    for index in range(1, n_waypoints + 1):
        alpha = float(index) / float(n_waypoints)
        wp_pos = ((1.0 - alpha) * start_pos + alpha * _to_numpy(target_pos_world, dtype=np.float32)).astype(np.float32)
        wp_quat = _slerp_quat(start_quat, target_quat_world, alpha)
        q = _solve_openarm_curobo_ik(
            curobo_state=curobo_state,
            current_curobo=prev_q,
            target_pos_world=wp_pos,
            target_quat_world=wp_quat,
            robot_base_pos=robot_base_pos,
            robot_base_quat=robot_base_quat,
        )
        if q is None:
            LOG.warning("OpenArm cuRobo linear IK failed at waypoint %d/%d", index, n_waypoints)
            return None
        jump = float(np.max(np.abs(q - prev_q)))
        if jump > max_joint_jump:
            q = prev_q + np.clip(q - prev_q, -max_joint_jump, max_joint_jump)
        q[curobo_state["left_curobo_indices"]] = curobo_state["home_curobo"][curobo_state["left_curobo_indices"]]
        waypoints.append(q.astype(np.float32, copy=True))
        prev_q = q.astype(np.float32, copy=True)
    return np.stack(waypoints, axis=0) if waypoints else None


def _execute_openarm_curobo_trajectory(
    robot: Any,
    world: Any,
    simulation_app: Any,
    curobo_state: dict[str, Any],
    trajectory: np.ndarray,
    gripper_target: float,
    record_fn: Callable[[np.ndarray, float, bool], None],
    timeout_fn: Callable[[], bool],
    stop_event: threading.Event,
    settle_steps: int = CUROBO_SETTLE_STEPS,
    mark_last: bool = False,
) -> bool:
    final_full_target: Optional[np.ndarray] = None
    for index in range(trajectory.shape[0]):
        if stop_event.is_set() or timeout_fn():
            return False
        current_full = _to_numpy(robot.get_joint_positions())
        final_full_target = _compose_full_target_from_curobo(
            current_full=current_full,
            curobo_arm_target=trajectory[index],
            curobo_state=curobo_state,
            finger_target=gripper_target,
        )
        _apply_joint_targets(robot, final_full_target, physics_control=False)
        _step_world(world, simulation_app, render=True, steps=1)
        is_last = mark_last and index == (trajectory.shape[0] - 1) and settle_steps <= 0
        record_fn(final_full_target[RIGHT_ARM_INDICES], gripper_target, is_last)

    if final_full_target is None:
        return False
    for settle_index in range(settle_steps):
        if stop_event.is_set() or timeout_fn():
            return False
        _apply_joint_targets(robot, final_full_target, physics_control=False)
        _step_world(world, simulation_app, render=True, steps=1)
        is_last = mark_last and settle_index == (settle_steps - 1)
        record_fn(final_full_target[RIGHT_ARM_INDICES], gripper_target, is_last)
    return True


def _hold_current_pose(
    robot: Any,
    world: Any,
    simulation_app: Any,
    gripper_target: float,
    hold_steps: int,
    record_fn: Callable[[np.ndarray, float, bool], None],
    timeout_fn: Callable[[], bool],
    stop_event: threading.Event,
    mark_last: bool = False,
) -> bool:
    current_full = _to_numpy(robot.get_joint_positions())
    current_full[RIGHT_FINGER_INDICES] = float(gripper_target)
    for step in range(max(1, int(hold_steps))):
        if stop_event.is_set() or timeout_fn():
            return False
        _apply_joint_targets(robot, current_full, physics_control=False)
        _step_world(world, simulation_app, render=True, steps=1)
        is_last = mark_last and step == (max(1, int(hold_steps)) - 1)
        record_fn(current_full[RIGHT_ARM_INDICES], gripper_target, is_last)
    return True


def _run_waypoint_fallback(
    cube_pos: np.ndarray,
    bowl_pos: np.ndarray,
    ctx: CollectorContext,
    world: Any,
    simulation_app: Any,
    steps_per_segment: int,
    stop_event: threading.Event,
    timeout_fn: Callable[[], bool],
    record_fn: Callable[[np.ndarray, float, bool], None],
    attach_state: dict[str, bool],
) -> bool:
    waypoints = _build_episode_waypoints(cube_pos=cube_pos, bowl_pos=bowl_pos)
    current_full = _to_numpy(ctx.robot.get_joint_positions())
    current_arm = current_full[RIGHT_ARM_INDICES].astype(np.float32, copy=True)
    current_gripper = float(np.mean(current_full[RIGHT_FINGER_INDICES]))
    fallback_points = [("current", current_arm, current_gripper, 12)] + waypoints

    for transition_index, (start_wp, end_wp) in enumerate(zip(fallback_points[:-1], fallback_points[1:])):
        _, start_arm, start_gripper, _ = start_wp
        end_name, end_arm, end_gripper, end_steps = end_wp
        seg_steps = max(
            1,
            int(round(float(end_steps) * float(steps_per_segment) / float(STEPS_PER_SEGMENT))),
        )
        for step in range(seg_steps):
            if stop_event.is_set() or timeout_fn():
                return False
            alpha = float(step + 1) / float(seg_steps)
            arm_target = ((1.0 - alpha) * start_arm + alpha * end_arm).astype(np.float32)
            gripper_target = float((1.0 - alpha) * start_gripper + alpha * end_gripper)
            current_full = _to_numpy(ctx.robot.get_joint_positions())
            full_target = _compose_full_target(current_full, arm_target, gripper_target)
            _apply_joint_targets(ctx.robot, full_target, physics_control=False)
            _step_world(world, simulation_app, render=True, steps=1)
            is_last = transition_index == (len(fallback_points) - 2) and step == (seg_steps - 1)
            record_fn(arm_target, gripper_target, is_last)
        if end_name == "close" and not attach_state["attached"] and _should_attach(ctx.stage, ctx.right_eef_prim_path, ctx.cube_prim_path):
            if ctx.cube_prim_path:
                attach_state["attached"] = _create_attachment_joint(ctx.stage, ctx.right_eef_prim_path, ctx.cube_prim_path)
                _step_world(world, simulation_app, render=True, steps=4)
        if end_name == "place_open" and attach_state["attached"]:
            _remove_prim_if_exists(ctx.stage, ATTACHMENT_JOINT_PATH)
            attach_state["attached"] = False
            _step_world(world, simulation_app, render=True, steps=4)
    return True


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
    # Match Franka collector pattern: create Articulation, add to world scene,
    # reset world (creates PhysicsSimulationViews), step, then initialize.
    import time as _time_mod
    robot = Articulation(robot_prim_path, name=f"openarm_collect_{int(_time_mod.time())}")
    if world is not None:
        world.scene.add(robot)
        try:
            world.reset()
        except RuntimeError as _reset_exc:
            if "expired" in str(_reset_exc).lower():
                LOG.warning("world.reset() expired prim — clearing registry and retrying")
                reg = getattr(world.scene, "_scene_registry", None)
                if reg is not None:
                    for attr in list(vars(reg)):
                        v = getattr(reg, attr, None)
                        if isinstance(v, dict):
                            v.clear()
                world.reset()
            else:
                raise
        for _ in range(30):
            world.step(render=True)
        LOG.info("world.reset() + 30 steps done")
    else:
        for _ in range(30):
            simulation_app.update()

    if hasattr(robot, "initialize"):
        try:
            robot.initialize()
        except Exception as _init_exc:
            LOG.warning("robot.initialize() failed: %s", _init_exc)
    _step_world(world, simulation_app, render=True, steps=5)
    LOG.info("robot DOFs: %d names: %s", robot.num_dof, list(robot.dof_names)[:5])

    openarm_root = _openarm_root_from_robot_path(robot_prim_path)
    camera_cfg = _load_right_wrist_camera_cfg()
    right_camera_path, right_eef_prim_path = _resolve_right_wrist_camera(stage, openarm_root, camera_cfg)

    cube_prim_path = _find_named_prim(stage, ("cube", "cuboid"))
    bowl_prim_path = _find_named_prim(stage, ("bowl",))
    table_prim_path = _find_named_prim(stage, ("table", "desk", "bench", "work"))

    cube_base_pos = _get_prim_world_pose(stage, cube_prim_path)[0] if cube_prim_path else DEFAULT_CUBE_POS.copy()
    bowl_base_pos = _get_prim_world_pose(stage, bowl_prim_path)[0] if bowl_prim_path else DEFAULT_BOWL_POS.copy()

    _configure_cube_physics(stage, cube_prim_path)
    _configure_bowl_collision(stage, bowl_prim_path)

    render_product, annotator = _setup_right_wrist_camera(stage, right_camera_path, camera_cfg)
    _step_world(world, simulation_app, render=True, steps=3)
    pose_samples = _sample_pose_references(world, simulation_app, robot, stage, right_eef_prim_path)
    curobo_state = _init_openarm_curobo(
        stage=stage,
        robot=robot,
        robot_prim_path=robot_prim_path,
        table_prim_path=table_prim_path,
    )

    return CollectorContext(
        stage=stage,
        robot=robot,
        robot_prim_path=robot_prim_path,
        openarm_root=openarm_root,
        right_eef_prim_path=right_eef_prim_path,
        cube_prim_path=cube_prim_path,
        bowl_prim_path=bowl_prim_path,
        table_prim_path=table_prim_path,
        right_wrist_camera_path=right_camera_path,
        right_wrist_render_product=render_product,
        right_wrist_annotator=annotator,
        cube_base_pos=np.asarray(cube_base_pos, dtype=np.float32),
        bowl_base_pos=np.asarray(bowl_base_pos, dtype=np.float32),
        pose_samples=pose_samples,
        curobo_state=curobo_state,
    )


def _run_episode_simple(
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

    _move_cube_for_episode(
        stage=ctx.stage,
        cube_prim_path=ctx.cube_prim_path,
        cube_base_pos=ctx.cube_base_pos,
        rng=rng,
        object_position_noise=object_position_noise,
    )
    _step_world(world, simulation_app, render=True, steps=6)

    cube_pos = _get_prim_world_pose(ctx.stage, ctx.cube_prim_path)[0] if ctx.cube_prim_path else ctx.cube_base_pos.copy()
    bowl_pos = _get_prim_world_pose(ctx.stage, ctx.bowl_prim_path)[0] if ctx.bowl_prim_path else ctx.bowl_base_pos.copy()

    current = _to_numpy(ctx.robot.get_joint_positions())
    if current.size < 18:
        raise RuntimeError(f"Expected OpenArm articulation with 18 DOF, got {current.size}.")
    if ctx.curobo_state is None:
        LOG.error("Episode %d cannot run: cuRobo was not initialized.", episode_index + 1)
        return 0, False, False

    home_targets = _compose_home_full_target(current, finger_target=GRIPPER_OPEN)
    _apply_joint_targets(ctx.robot, home_targets, physics_control=False)
    _step_world(world, simulation_app, render=True, steps=10)

    frame_index = 0
    attach_state = {"attached": False}
    episode_start = time.time()
    robot_base_pos, robot_base_quat = _get_prim_world_pose(ctx.stage, ctx.robot_prim_path)
    grasp_quat = GRASP_QUAT_WXYZ.copy()
    pre_grasp_pos = (cube_pos + PRE_GRASP_OFFSET).astype(np.float32)
    grasp_pos = (cube_pos + GRASP_OFFSET).astype(np.float32)
    lift_pos = (grasp_pos + LIFT_OFFSET).astype(np.float32)
    bowl_approach_pos = (bowl_pos + BOWL_APPROACH_OFFSET).astype(np.float32)
    place_pos = (bowl_pos + PLACE_LOWER_OFFSET).astype(np.float32)

    def timeout_fn() -> bool:
        if episode_timeout_sec <= 0.0:
            return False
        return (time.time() - episode_start) > episode_timeout_sec

    def record_step(arm_target: np.ndarray, gripper_target: float, is_last: bool = False) -> None:
        nonlocal frame_index
        frame_index = _record_frame(
            writer=writer,
            ctx=ctx,
            world=world,
            simulation_app=simulation_app,
            episode_index=episode_index,
            frame_index=frame_index,
            arm_target=np.asarray(arm_target, dtype=np.float32),
            gripper_target=float(gripper_target),
            next_done=is_last,
            fps=fps,
        )
    try:
        _update_openarm_curobo_world(
            curobo_state=ctx.curobo_state,
            stage=ctx.stage,
            robot_prim_path=ctx.robot_prim_path,
            table_prim_path=ctx.table_prim_path,
        )

        traj_pre_grasp = _plan_openarm_curobo_segment(
            curobo_state=ctx.curobo_state,
            current_full=_to_numpy(ctx.robot.get_joint_positions()),
            target_pos_world=pre_grasp_pos,
            target_quat_world=grasp_quat,
            robot_base_pos=robot_base_pos,
            robot_base_quat=robot_base_quat,
        )
        if traj_pre_grasp is None:
            raise RuntimeError("home->pre_grasp planning failed")
        if not _execute_openarm_curobo_trajectory(
            robot=ctx.robot,
            world=world,
            simulation_app=simulation_app,
            curobo_state=ctx.curobo_state,
            trajectory=traj_pre_grasp,
            gripper_target=GRIPPER_OPEN,
            record_fn=record_step,
            timeout_fn=timeout_fn,
            stop_event=stop_event,
        ):
            raise RuntimeError("home->pre_grasp execution failed")

        traj_grasp = _plan_openarm_linear_ik_segment(
            curobo_state=ctx.curobo_state,
            current_full=_to_numpy(ctx.robot.get_joint_positions()),
            stage=ctx.stage,
            eef_prim_path=ctx.right_eef_prim_path,
            target_pos_world=grasp_pos,
            target_quat_world=grasp_quat,
            robot_base_pos=robot_base_pos,
            robot_base_quat=robot_base_quat,
            n_waypoints=LINEAR_CARTESIAN_WAYPOINTS,
        )
        if traj_grasp is None:
            raise RuntimeError("pre_grasp->grasp IK failed")
        if not _execute_openarm_curobo_trajectory(
            robot=ctx.robot,
            world=world,
            simulation_app=simulation_app,
            curobo_state=ctx.curobo_state,
            trajectory=traj_grasp,
            gripper_target=GRIPPER_OPEN,
            record_fn=record_step,
            timeout_fn=timeout_fn,
            stop_event=stop_event,
        ):
            raise RuntimeError("pre_grasp->grasp execution failed")

        if not _hold_current_pose(
            robot=ctx.robot,
            world=world,
            simulation_app=simulation_app,
            gripper_target=GRIPPER_CLOSED,
            hold_steps=GRIPPER_CLOSE_STEPS,
            record_fn=record_step,
            timeout_fn=timeout_fn,
            stop_event=stop_event,
        ):
            raise RuntimeError("gripper close failed")
        if ctx.cube_prim_path and not attach_state["attached"]:
            attach_state["attached"] = _create_attachment_joint(ctx.stage, ctx.right_eef_prim_path, ctx.cube_prim_path)
            _step_world(world, simulation_app, render=True, steps=2)

        traj_lift = _plan_openarm_linear_ik_segment(
            curobo_state=ctx.curobo_state,
            current_full=_to_numpy(ctx.robot.get_joint_positions()),
            stage=ctx.stage,
            eef_prim_path=ctx.right_eef_prim_path,
            target_pos_world=lift_pos,
            target_quat_world=grasp_quat,
            robot_base_pos=robot_base_pos,
            robot_base_quat=robot_base_quat,
            n_waypoints=LINEAR_CARTESIAN_WAYPOINTS,
        )
        if traj_lift is None:
            raise RuntimeError("lift IK failed")
        if not _execute_openarm_curobo_trajectory(
            robot=ctx.robot,
            world=world,
            simulation_app=simulation_app,
            curobo_state=ctx.curobo_state,
            trajectory=traj_lift,
            gripper_target=GRIPPER_CLOSED,
            record_fn=record_step,
            timeout_fn=timeout_fn,
            stop_event=stop_event,
        ):
            raise RuntimeError("lift execution failed")

        traj_bowl = _plan_openarm_curobo_segment(
            curobo_state=ctx.curobo_state,
            current_full=_to_numpy(ctx.robot.get_joint_positions()),
            target_pos_world=bowl_approach_pos,
            target_quat_world=grasp_quat,
            robot_base_pos=robot_base_pos,
            robot_base_quat=robot_base_quat,
        )
        if traj_bowl is None:
            raise RuntimeError("lift->bowl planning failed")
        if not _execute_openarm_curobo_trajectory(
            robot=ctx.robot,
            world=world,
            simulation_app=simulation_app,
            curobo_state=ctx.curobo_state,
            trajectory=traj_bowl,
            gripper_target=GRIPPER_CLOSED,
            record_fn=record_step,
            timeout_fn=timeout_fn,
            stop_event=stop_event,
        ):
            raise RuntimeError("lift->bowl execution failed")

        traj_place = _plan_openarm_linear_ik_segment(
            curobo_state=ctx.curobo_state,
            current_full=_to_numpy(ctx.robot.get_joint_positions()),
            stage=ctx.stage,
            eef_prim_path=ctx.right_eef_prim_path,
            target_pos_world=place_pos,
            target_quat_world=grasp_quat,
            robot_base_pos=robot_base_pos,
            robot_base_quat=robot_base_quat,
            n_waypoints=LINEAR_CARTESIAN_WAYPOINTS,
        )
        if traj_place is None:
            raise RuntimeError("bowl->place IK failed")
        if not _execute_openarm_curobo_trajectory(
            robot=ctx.robot,
            world=world,
            simulation_app=simulation_app,
            curobo_state=ctx.curobo_state,
            trajectory=traj_place,
            gripper_target=GRIPPER_CLOSED,
            record_fn=record_step,
            timeout_fn=timeout_fn,
            stop_event=stop_event,
        ):
            raise RuntimeError("bowl->place execution failed")

        if attach_state["attached"]:
            _remove_prim_if_exists(ctx.stage, ATTACHMENT_JOINT_PATH)
            attach_state["attached"] = False
            _step_world(world, simulation_app, render=True, steps=2)
        if not _hold_current_pose(
            robot=ctx.robot,
            world=world,
            simulation_app=simulation_app,
            gripper_target=GRIPPER_OPEN,
            hold_steps=GRIPPER_OPEN_STEPS,
            record_fn=record_step,
            timeout_fn=timeout_fn,
            stop_event=stop_event,
        ):
            raise RuntimeError("place release failed")

        traj_home = _plan_openarm_curobo_joint_segment(
            curobo_state=ctx.curobo_state,
            current_full=_to_numpy(ctx.robot.get_joint_positions()),
            target_full=_compose_home_full_target(_to_numpy(ctx.robot.get_joint_positions()), finger_target=GRIPPER_OPEN),
        )
        if traj_home is None:
            raise RuntimeError("return home planning failed")
        if not _execute_openarm_curobo_trajectory(
            robot=ctx.robot,
            world=world,
            simulation_app=simulation_app,
            curobo_state=ctx.curobo_state,
            trajectory=traj_home,
            gripper_target=GRIPPER_OPEN,
            record_fn=record_step,
            timeout_fn=timeout_fn,
            stop_event=stop_event,
            mark_last=True,
        ):
            raise RuntimeError("return home execution failed")
    except Exception as exc:
        LOG.warning("Episode %d GT pose-driven execution failed: %s", episode_index + 1, exc)
        return frame_index, bool(stop_event.is_set()), False

    if stop_event.is_set():
        return frame_index, True, False
    if timeout_fn():
        LOG.warning("Episode %d timed out after %.2fs", episode_index, time.time() - episode_start)
        return frame_index, False, False

    _remove_prim_if_exists(ctx.stage, ATTACHMENT_JOINT_PATH)
    _step_world(world, simulation_app, render=True, steps=6)
    return frame_index, False, _episode_success(ctx.stage, ctx.cube_prim_path, ctx.bowl_prim_path)


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
    """Simple waypoint interpolation episode using replay-derived joint angles."""
    _remove_prim_if_exists(ctx.stage, ATTACHMENT_JOINT_PATH)
    _world_reset(world, simulation_app)

    # Randomize cube position
    _move_cube_for_episode(
        stage=ctx.stage,
        cube_prim_path=ctx.cube_prim_path,
        cube_base_pos=ctx.cube_base_pos,
        rng=rng,
        object_position_noise=object_position_noise,
    )
    _step_world(world, simulation_app, render=True, steps=6)

    current = _to_numpy(ctx.robot.get_joint_positions())
    if current.size < 18:
        raise RuntimeError(f"Expected OpenArm articulation with 18 DOF, got {current.size}.")

    frame_index = 0
    episode_start = time.time()
    stopped = False

    def timeout_fn():
        return episode_timeout_sec > 0 and (time.time() - episode_start) > episode_timeout_sec

    # Convert dataset-order waypoint to 18-DOF sim target
    def wp_to_full(wp_data: np.ndarray) -> np.ndarray:
        current_full = _to_numpy(ctx.robot.get_joint_positions())
        arm_7 = np.zeros(7, dtype=np.float32)
        for sim_j in range(7):
            data_j = DATA_TO_SIM[sim_j]
            val = float(wp_data[data_j])
            if sim_j in NEGATED_SIM_JOINTS:
                val = -val
            arm_7[sim_j] = val
        target = current_full.copy()
        for i, idx in enumerate(RIGHT_ARM_INDICES):
            target[idx] = arm_7[i]
        finger_val = float(wp_data[7])
        for idx in RIGHT_FINGER_INDICES:
            target[idx] = finger_val
        # Set left arm to homing
        left_indices = [0, 2, 4, 6, 8, 10, 12, 14, 15]
        left_home = [HOME_FULL[i] for i in left_indices]
        for i, idx in enumerate(left_indices):
            target[idx] = left_home[i]
        return target

    # Execute waypoint transitions with joint interpolation
    prev_wp = PICK_PLACE_WAYPOINTS[0]
    prev_full = wp_to_full(prev_wp[1])
    _apply_joint_targets(ctx.robot, prev_full, physics_control=False)
    _step_world(world, simulation_app, render=True, steps=10)
    LOG.info("Episode %d: starting from HOME", episode_index + 1)

    for wp_idx in range(1, len(PICK_PLACE_WAYPOINTS)):
        if stopped or timeout_fn():
            break
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break

        wp_name, wp_data, n_steps = PICK_PLACE_WAYPOINTS[wp_idx]
        target_full = wp_to_full(wp_data)
        start_full = _to_numpy(ctx.robot.get_joint_positions())

        LOG.info("Episode %d: %s (%d steps)", episode_index + 1, wp_name, n_steps)

        for step in range(n_steps):
            if timeout_fn() or (stop_event is not None and stop_event.is_set()):
                stopped = True
                break

            alpha = float(step + 1) / float(n_steps)
            interp = ((1.0 - alpha) * start_full + alpha * target_full).astype(np.float32)

            from omni.isaac.core.utils.types import ArticulationAction
            ctrl = ctx.robot.get_articulation_controller()
            ctrl.apply_action(ArticulationAction(joint_positions=interp))
            world.step(render=True)

            # Record frame
            obs_state = _full_to_dataset_state(_to_numpy(ctx.robot.get_joint_positions()))
            act_state = _full_to_dataset_state(interp)
            is_last = (wp_idx == len(PICK_PLACE_WAYPOINTS) - 1 and step == n_steps - 1)

            # Camera image as extras
            extras = {}
            if ctx.right_wrist_annotator is not None:
                try:
                    rgba = ctx.right_wrist_annotator.get_data()
                    if rgba is not None:
                        rgba = np.asarray(rgba)
                        if rgba.ndim == 3 and rgba.shape[-1] >= 3:
                            rgb = rgba[:, :, :3]
                            if rgb.dtype != np.uint8:
                                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                            extras["observation.images.right_wrist_cam"] = rgb
                except Exception:
                    pass

            writer.add_frame(
                episode_index=episode_index,
                frame_index=frame_index,
                observation_state=obs_state,
                action=act_state,
                timestamp=time.time(),
                next_done=is_last,
                extras=extras if extras else None,
            )
            frame_index += 1

        # After CLOSE phase, create attachment joint
        if wp_name == "CLOSE" and ctx.cube_prim_path and not _prim_exists(ctx.stage, ATTACHMENT_JOINT_PATH):
            _create_attachment_joint(ctx.stage, ctx.right_eef_prim_path, ctx.cube_prim_path)
            _step_world(world, simulation_app, render=True, steps=2)
            LOG.info("Episode %d: attached cube", episode_index + 1)

        # After OPEN phase, remove attachment
        if wp_name == "OPEN":
            _remove_prim_if_exists(ctx.stage, ATTACHMENT_JOINT_PATH)
            _step_world(world, simulation_app, render=True, steps=2)
            LOG.info("Episode %d: released cube", episode_index + 1)

    LOG.info("Episode %d: done, %d frames", episode_index + 1, frame_index)
    _remove_prim_if_exists(ctx.stage, ATTACHMENT_JOINT_PATH)
    _step_world(world, simulation_app, render=True, steps=6)
    return frame_index, stopped, _episode_success(ctx.stage, ctx.cube_prim_path, ctx.bowl_prim_path)


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
