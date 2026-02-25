#!/usr/bin/env python3
"""Standalone Isaac Sim pick-and-place data collector.

Run with Isaac Sim Python:
    /isaac-sim/python.sh /sim-service/isaac_pick_place_collector.py \
        --num-episodes 5 --output /tmp/pick_place

This module also exposes `run_collection_in_process(...)` for `run_interactive.py`.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lerobot_writer import SimLeRobotWriter

LOG = logging.getLogger("isaac-pick-place-collector")

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
APPROACH_BASE = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.8, 0.785], dtype=np.float32)
DOWN_BASE = np.array([0.0, 0.1, 0.0, -1.2, 0.0, 2.0, 0.785], dtype=np.float32)

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
WAYPOINT_NOISE_RAD = 0.05

OBJECT_SIZE = 0.04
OBJECT_MASS = 0.1
MIN_PICK_PLACE_DIST = 0.1
STEPS_PER_SEGMENT = 50
TABLE_MARGIN = 0.08
MAX_GRIPPER_STEP = 0.008
MAX_IK_WAYPOINT_JUMP_RAD = 1.20

try:
    MAX_ARM_STEP_RAD = float(os.environ.get("COLLECT_MAX_ARM_STEP_RAD", "0.03"))
except Exception:
    MAX_ARM_STEP_RAD = 0.03
MAX_ARM_STEP_RAD = float(np.clip(MAX_ARM_STEP_RAD, 0.005, 0.10))

TABLE_X_RANGE = (0.3, 0.7)
TABLE_Y_RANGE = (-0.3, 0.3)
TABLE_Z = 0.41
IK_APPROACH_Z_OFFSET = 0.18
IK_GRASP_Z_OFFSET = 0.07
IK_PICK_APPROACH_EXTRA_Z = 0.12
IK_PLACE_APPROACH_EXTRA_Z = 0.10
IK_PICK_GRASP_HEIGHT_RATIO = 0.60
IK_PICK_MIN_CLEARANCE_FROM_TABLE = 0.01
IK_PLACE_MIN_CLEARANCE_FROM_TABLE = 0.02
IK_PRE_GRASP_OFFSET = 0.10
IK_PRE_GRASP_MIN_ABOVE_TABLE = 0.02
GRASP_MAX_ATTEMPTS = 3
REACH_BEFORE_CLOSE_MAX_OBJECT_EEF_DISTANCE = 0.16
REACH_BEFORE_CLOSE_MAX_OBJECT_EEF_XY_DISTANCE = 0.08
REACH_BEFORE_CLOSE_MAX_OBJECT_TIP_MID_DISTANCE = 0.03
REACH_BEFORE_CLOSE_MAX_OBJECT_TIP_MID_XY_DISTANCE = 0.022
REACH_BEFORE_CLOSE_MAX_ABS_TIP_MID_Z_DELTA = 0.03
REACH_REQUIRE_TIP_MID = True
CLOSE_HOLD_STEPS = 8
try:
    PRE_CLOSE_SETTLE_STEPS = int(os.environ.get("COLLECT_PRE_CLOSE_SETTLE_STEPS", "18"))
except Exception:
    PRE_CLOSE_SETTLE_STEPS = 18
PRE_CLOSE_SETTLE_STEPS = int(np.clip(PRE_CLOSE_SETTLE_STEPS, 0, 120))
VERIFY_CLOSE_MIN_GRIPPER_WIDTH = 0.0025
VERIFY_CLOSE_MAX_OBJECT_EEF_DISTANCE = 0.25
VERIFY_CLOSE_MAX_OBJECT_EEF_XY_DISTANCE = 0.08
VERIFY_RETRIEVAL_MIN_LIFT = 0.008
VERIFY_RETRIEVAL_MAX_OBJECT_EEF_DISTANCE = 0.30
ROBOT_REACH_RADIUS_X = 0.75
ROBOT_REACH_RADIUS_Y = 0.75
REACH_INTERSECTION_MIN_SPAN = 0.25
PICK_CLAMP_MAX_DELTA = 0.03
DEFAULT_EPISODE_TIMEOUT_SEC = 180.0
STATE_JOINT_VEL_CLIP_RAD_S = 20.0
# Use ROS camera frame pose for wrist camera to avoid USD axis confusion:
# - +Z forward, +Y up (camera_axes='ros').
# These values align with common IsaacLab hand-camera usage.
WRIST_CAM_LOCAL_POS_ROS = (0.025, 0.0, 0.0)
WRIST_CAM_LOCAL_QUAT_ROS = (0.7071068, 0.0, 0.0, 0.7071068)
# Fallback when Camera.set_local_pose(camera_axes=...) is unavailable.
WRIST_CAM_FALLBACK_TRANSLATE = (0.0, 0.0, 0.10)
WRIST_CAM_FALLBACK_ROTATE_XYZ = (0.0, 180.0, 0.0)
# MagicSim-style heuristic fallback when no annotation exists.
# Quaternion format: [w, x, y, z]
TOP_DOWN_FALLBACK_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
# Treat annotation translations as fingertip-midpoint (TCP) by default, then
# convert to panda_hand frame for IK target consistency.
_ANN_FRAME_RAW = os.environ.get("COLLECT_ANNOTATION_POSE_IS_TIP_MID_FRAME", "1").strip().lower()
ANNOTATION_POSE_IS_TIP_MID_FRAME = _ANN_FRAME_RAW not in {"0", "false", "no", "off"}

# Franka Panda joint limits (radians), used for command clipping + anti-knot safety.
FRANKA_ARM_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
FRANKA_ARM_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)
TRACKING_BBOX_MAX_EXTENT_M = 0.60
TRACKING_BBOX_MAX_ROOT_DELTA_M = 0.40
TRACKING_OBJECT_HEIGHT_MAX_M = 0.25

COLLECTOR_ROOT = "/World/PickPlaceCollector"
TABLE_PRIM_PATH = f"{COLLECTOR_ROOT}/Table"
ROBOT_PRIM_PATH = f"{COLLECTOR_ROOT}/Franka"
OVERHEAD_CAM_PATH = f"{COLLECTOR_ROOT}/OverheadCam"
WRIST_CAM_PATH = f"{ROBOT_PRIM_PATH}/panda_hand/wrist_cam"
OBJECT_PRIM_PATH = f"{COLLECTOR_ROOT}/PickObject"
MATERIAL_PRIM_PATH = f"{COLLECTOR_ROOT}/GripMaterial"

DEFAULT_GRASP_POSE_DIR = str(SCRIPT_DIR / "grasp_poses")
GRASP_POSE_SOURCE_IDS = {
    "none": 0.0,
    "annotation": 1.0,
    "bbox_center": 2.0,
    "root_pose": 3.0,
}
_FAILED_ANNOTATION_POSE_CACHE: dict[str, set[int]] = {}

try:
    ANNOTATION_IK_SCREEN_MAX_CANDIDATES = int(
        os.environ.get("COLLECT_ANNOTATION_IK_SCREEN_MAX_CANDIDATES", "64")
    )
except Exception:
    ANNOTATION_IK_SCREEN_MAX_CANDIDATES = 64
ANNOTATION_IK_SCREEN_MAX_CANDIDATES = int(
    np.clip(ANNOTATION_IK_SCREEN_MAX_CANDIDATES, 1, 512)
)

try:
    ANNOTATION_MIN_LOCAL_Z_PERCENTILE = float(
        os.environ.get("COLLECT_ANNOTATION_MIN_LOCAL_Z_PERCENTILE", "0.30")
    )
except Exception:
    ANNOTATION_MIN_LOCAL_Z_PERCENTILE = 0.30
ANNOTATION_MIN_LOCAL_Z_PERCENTILE = float(
    np.clip(ANNOTATION_MIN_LOCAL_Z_PERCENTILE, 0.0, 0.90)
)

try:
    ANNOTATION_BODY_MIN_LOCAL_Z_PERCENTILE = float(
        os.environ.get("COLLECT_ANNOTATION_BODY_MIN_LOCAL_Z_PERCENTILE", "0.45")
    )
except Exception:
    ANNOTATION_BODY_MIN_LOCAL_Z_PERCENTILE = 0.45
ANNOTATION_BODY_MIN_LOCAL_Z_PERCENTILE = float(
    np.clip(ANNOTATION_BODY_MIN_LOCAL_Z_PERCENTILE, 0.0, 0.95)
)

try:
    ANNOTATION_MIN_GROUP_KEEP = int(
        os.environ.get("COLLECT_ANNOTATION_MIN_GROUP_KEEP", "4")
    )
except Exception:
    ANNOTATION_MIN_GROUP_KEEP = 4
ANNOTATION_MIN_GROUP_KEEP = int(np.clip(ANNOTATION_MIN_GROUP_KEEP, 1, 32))

_ANN_VALIDATE_RAW = os.environ.get("COLLECT_ANNOTATION_VALIDATE_STRICT", "1").strip().lower()
ANNOTATION_VALIDATE_STRICT = _ANN_VALIDATE_RAW not in {"0", "false", "no", "off"}
try:
    ANNOTATION_MAX_POS_RADIUS_FACTOR = float(
        os.environ.get("COLLECT_ANNOTATION_MAX_POS_RADIUS_FACTOR", "2.8")
    )
except Exception:
    ANNOTATION_MAX_POS_RADIUS_FACTOR = 2.8
ANNOTATION_MAX_POS_RADIUS_FACTOR = float(np.clip(ANNOTATION_MAX_POS_RADIUS_FACTOR, 1.2, 20.0))

try:
    ANNOTATION_BBOX_MARGIN_FACTOR = float(
        os.environ.get("COLLECT_ANNOTATION_BBOX_MARGIN_FACTOR", "0.25")
    )
except Exception:
    ANNOTATION_BBOX_MARGIN_FACTOR = 0.25
ANNOTATION_BBOX_MARGIN_FACTOR = float(np.clip(ANNOTATION_BBOX_MARGIN_FACTOR, 0.01, 2.0))

try:
    ANNOTATION_MIN_VALID_KEEP_RATIO = float(
        os.environ.get("COLLECT_ANNOTATION_MIN_VALID_KEEP_RATIO", "0.30")
    )
except Exception:
    ANNOTATION_MIN_VALID_KEEP_RATIO = 0.30
ANNOTATION_MIN_VALID_KEEP_RATIO = float(np.clip(ANNOTATION_MIN_VALID_KEEP_RATIO, 0.05, 1.0))

FRAME_EXTRA_FEATURES: dict[str, dict[str, Any]] = {
    "observation.object_pose_world": {"dtype": "float32", "shape": [7]},
    "observation.grasp_target_pose_world": {"dtype": "float32", "shape": [7]},
    "observation.grasp_target_valid": {"dtype": "bool", "shape": [1]},
    "observation.grasp_target_source_id": {"dtype": "float32", "shape": [1]},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Franka pick-place data in Isaac Sim.")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to collect.")
    parser.add_argument(
        "--output",
        type=str,
        default="/sim-service/datasets/pick_place",
        help="Output LeRobot dataset directory.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run Isaac Sim headless.",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Dataset FPS.")
    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=STEPS_PER_SEGMENT,
        help="Interpolation steps for each waypoint transition.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Enable WebRTC streaming for monitoring.",
    )
    parser.add_argument("--streaming-port", type=int, default=49102, help="WebRTC signaling port.")
    parser.add_argument(
        "--use-ycb",
        action="store_true",
        default=False,
        help="Use YCB assets instead of a cube object.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="pick_place",
        help="Task label recorded in dataset episode metadata.",
    )
    parser.add_argument(
        "--episode-timeout-sec",
        type=float,
        default=_read_float_env("COLLECT_EPISODE_TIMEOUT_SEC", DEFAULT_EPISODE_TIMEOUT_SEC),
        help="Per-episode hard timeout in seconds (0 disables timeout).",
    )
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


def _default_dataset_repo_id(output_dir: str) -> str:
    dataset_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(output_dir).name.strip()).strip("._-")
    dataset_id = dataset_id or "sim_collect"
    return f"local/{dataset_id}"


def _read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        LOG.warning("Invalid float env %s=%r, fallback %.3f", name, raw, float(default))
        return float(default)


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    LOG.warning("Invalid bool env %s=%r, fallback %s", name, raw, bool(default))
    return bool(default)


def _resolve_collector_log_dir(output_dir: str | None = None) -> Path | None:
    candidates: list[Path] = []
    env_dir = os.environ.get("COLLECT_LOG_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    if output_dir:
        try:
            candidates.append(Path(output_dir).resolve() / "_logs")
        except Exception:
            candidates.append(Path(output_dir) / "_logs")
    sim_log_dir = os.environ.get("SIM_LOG_DIR", "").strip()
    if sim_log_dir:
        candidates.append(Path(sim_log_dir) / "collector")
    candidates.extend(
        [
            Path("/data/embodied/logs/sim-service/collector"),
            Path("/data/datasets/embodied/logs/sim-service/collector"),
            Path("/tmp/sim-service-logs/collector"),
        ]
    )
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            continue
    return None


def _configure_collector_logging(output_dir: str | None = None) -> str | None:
    level = getattr(logging, os.environ.get("COLLECT_LOG_LEVEL", "INFO").upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_path: str | None = None
    log_dir = _resolve_collector_log_dir(output_dir=output_dir)
    if log_dir is not None:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_file = log_dir / f"isaac_pick_place_collector_{ts}_{os.getpid()}.log"
        try:
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
            log_path = str(log_file)
        except Exception:
            pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    if log_path:
        LOG.info("Persistent collector log enabled: %s", log_path)
    else:
        LOG.warning("Collector file logging unavailable; using stdout only")
    return log_path


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = _to_numpy(quat, dtype=np.float32).reshape(-1)
    if q.size < 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q4 = q[:4].astype(np.float32, copy=False)
    if not np.all(np.isfinite(q4)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    n = float(np.linalg.norm(q4))
    if not np.isfinite(n) or n < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q4 / n).astype(np.float32)


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


def _transform_local_pose_to_world(
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    local_pos: np.ndarray,
    local_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    obj_p = _to_numpy(object_pos, dtype=np.float32).reshape(-1)
    obj_q = _normalize_quat_wxyz(object_quat)
    loc_p = _to_numpy(local_pos, dtype=np.float32).reshape(-1)
    loc_q = _normalize_quat_wxyz(local_quat)
    if obj_p.size < 3:
        obj_p = np.zeros((3,), dtype=np.float32)
    if loc_p.size < 3:
        loc_p = np.zeros((3,), dtype=np.float32)
    rot = _quat_to_rot_wxyz(obj_q)
    world_pos = obj_p[:3] + rot @ loc_p[:3]
    world_quat = _quat_mul_wxyz(obj_q, loc_q)
    return world_pos.astype(np.float32), world_quat.astype(np.float32)


def _convert_tip_mid_local_pose_to_hand_local(
    local_pos: np.ndarray,
    local_quat: np.ndarray,
    tip_mid_offset_in_hand: np.ndarray,
) -> np.ndarray:
    """Convert a TCP/tip-mid local translation to panda_hand local translation."""
    pos = _to_numpy(local_pos, dtype=np.float32).reshape(-1)
    if pos.size < 3:
        pos = np.zeros((3,), dtype=np.float32)
    quat = _normalize_quat_wxyz(local_quat)
    offset_h = _to_numpy(tip_mid_offset_in_hand, dtype=np.float32).reshape(-1)
    if offset_h.size < 3 or not np.all(np.isfinite(offset_h[:3])):
        return pos[:3].astype(np.float32)
    offset_world_from_hand = _quat_to_rot_wxyz(quat) @ offset_h[:3]
    return (pos[:3] - offset_world_from_hand.astype(np.float32)).astype(np.float32)


def _apply_wrist_camera_pose(
    camera_wrist: Any | None,
    stage: Any,
    usd_geom: Any,
    gf: Any,
    wrist_cam_path: Optional[str] = None,
) -> None:
    """Set a stable wrist-camera pose with ROS camera-axis semantics.

    Primary path uses Camera.set_local_pose(..., camera_axes='ros'), which avoids
    manual USD/OpenGL axis conversion mistakes. If unavailable, fallback to a
    conservative USD transform that looks outward from the hand.
    """
    cam_path = str(wrist_cam_path or getattr(camera_wrist, "prim_path", "") or "")
    ros_pos = np.array(WRIST_CAM_LOCAL_POS_ROS, dtype=np.float32)
    ros_quat = np.array(WRIST_CAM_LOCAL_QUAT_ROS, dtype=np.float32)

    if camera_wrist is not None and hasattr(camera_wrist, "set_local_pose"):
        try:
            camera_wrist.set_local_pose(ros_pos, ros_quat, camera_axes="ros")
            return
        except TypeError:
            # Older Isaac builds may not expose camera_axes arg.
            try:
                camera_wrist.set_local_pose(ros_pos, ros_quat)
                LOG.warning("collect: wrist camera set_local_pose fallback without camera_axes")
                return
            except Exception as exc:
                LOG.warning("collect: wrist camera set_local_pose fallback failed: %s", exc)
        except Exception as exc:
            LOG.warning("collect: wrist camera set_local_pose failed: %s", exc)

    if not cam_path or stage is None:
        return
    wrist_prim = stage.GetPrimAtPath(cam_path)
    if not wrist_prim or not wrist_prim.IsValid():
        return
    wrist_xform = usd_geom.Xformable(wrist_prim)
    wrist_xform.ClearXformOpOrder()
    wrist_xform.AddTranslateOp().Set(gf.Vec3d(*WRIST_CAM_FALLBACK_TRANSLATE))
    wrist_xform.AddRotateXYZOp().Set(gf.Vec3f(*WRIST_CAM_FALLBACK_ROTATE_XYZ))


def _object_token_candidates(object_prim_path: str) -> list[str]:
    raw = str(object_prim_path or "").lower()
    leaf = raw.split("/")[-1]
    tokens = set()
    if leaf:
        tokens.add(leaf)
    for piece in re.split(r"[^a-z0-9]+", raw):
        if piece:
            tokens.add(piece)

    aliases = {
        "mug": ("mug", "cup", "coffee", "025", "ycb"),
        "mustard": ("mustard", "006"),
        "sugar": ("sugar", "004"),
        "cracker": ("cracker", "003"),
        "tomato": ("tomato", "soup", "005"),
    }
    expanded = set(tokens)
    for canonical, words in aliases.items():
        if any(w in tokens for w in words):
            expanded.add(canonical)

    ordered = sorted(expanded)
    return [t for t in ordered if t]


def _iter_mug_annotation_candidate_paths() -> list[Path]:
    candidates = [
        SCRIPT_DIR / "grasp_poses" / "mug_grasp_pose.json",
        Path("/sim-service/grasp_poses/mug_grasp_pose.json"),
        Path("/code/grasp_poses/mug_grasp_pose.json"),
        Path.cwd() / "grasp_poses" / "mug_grasp_pose.json",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        try:
            key = str(cand.resolve())
        except Exception:
            key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def _resolve_grasp_annotation_path(stage: Any, object_prim_path: str) -> Optional[Path]:
    explicit = os.environ.get("COLLECT_GRASP_POSE_PATH", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        LOG.warning("collect: COLLECT_GRASP_POSE_PATH not found: %s", p)

    prim = stage.GetPrimAtPath(object_prim_path) if stage is not None else None
    if prim and prim.IsValid():
        for key in ("grasp_pose_path", "graspPosePath", "annotation_path"):
            try:
                raw = prim.GetCustomDataByKey(key)
            except Exception:
                raw = None
            if isinstance(raw, str) and raw.strip():
                p = Path(raw.strip()).expanduser()
                if not p.is_absolute():
                    p = (SCRIPT_DIR / p).resolve()
                if p.exists():
                    return p

    roots = []
    env_root = os.environ.get("COLLECT_GRASP_POSE_DIR", "").strip()
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend(
        [
            Path(DEFAULT_GRASP_POSE_DIR),
            SCRIPT_DIR / "grasp_pose",
            SCRIPT_DIR / "grasp_ops" / "assets",
            SCRIPT_DIR.parent / "grasp_ops" / "assets",
        ]
    )

    tokens = _object_token_candidates(object_prim_path)
    for root in roots:
        if not root.exists():
            continue
        for tok in tokens:
            for cand in (
                root / f"{tok}_grasp_pose.json",
                root / f"{tok}.json",
                root / tok / "grasp_pose.json",
                root / tok / f"{tok}_grasp_pose.json",
            ):
                if cand.exists():
                    return cand
            try:
                for cand in sorted(root.glob(f"*{tok}*grasp_pose.json")):
                    if cand.exists():
                        return cand
            except Exception:
                pass
    if "mug" in set(tokens):
        for cand in _iter_mug_annotation_candidate_paths():
            if cand.exists():
                LOG.info("collect: using mug annotation fallback path: %s", cand)
                return cand
    return None


def _load_grasp_pose_candidates(annotation_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        with annotation_path.open("r") as f:
            data = json.load(f)
    except Exception as exc:
        LOG.warning("collect: failed to read grasp annotation %s: %s", annotation_path, exc)
        return [], {}

    out: list[dict[str, Any]] = []
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    quat_order_raw = str(
        metadata.get(
            "quat_order",
            metadata.get("quaternion_order", metadata.get("quat_format", "wxyz")),
        )
    ).strip().lower()
    quat_order = "xyzw" if quat_order_raw in {"xyzw", "x,y,z,w", "xyz_w", "quat_xyzw"} else "wxyz"

    pos_scale_hint = 1.0
    for key in ("position_scale_hint", "position_scale", "position_scale_to_m", "local_pos_scale"):
        raw = metadata.get(key, None)
        try:
            v = float(raw)
            if np.isfinite(v) and v > 0.0:
                pos_scale_hint = float(v)
                break
        except Exception:
            pass
    if abs(float(pos_scale_hint) - 1.0) < 1e-8:
        unit = str(metadata.get("position_unit", metadata.get("position_units", ""))).strip().lower()
        if unit in {"cm", "centimeter", "centimeters"}:
            pos_scale_hint = 0.01
        elif unit in {"mm", "millimeter", "millimeters"}:
            pos_scale_hint = 0.001
        elif unit in {"m", "meter", "meters"}:
            pos_scale_hint = 1.0

    meta_info: dict[str, Any] = {
        "quat_order": quat_order,
        "position_scale_hint": float(pos_scale_hint),
        "source": str(metadata.get("source", "")),
    }

    def _append_pose(label: str, pose: Any, source: str) -> None:
        arr = _to_numpy(pose, dtype=np.float32).reshape(-1)
        if arr.size < 7:
            return
        pos = arr[:3].astype(np.float32, copy=False)
        raw_quat = arr[3:7].astype(np.float32, copy=False)
        if quat_order == "xyzw":
            raw_quat = np.asarray(
                [raw_quat[3], raw_quat[0], raw_quat[1], raw_quat[2]],
                dtype=np.float32,
            )
        quat = _normalize_quat_wxyz(raw_quat)
        if not np.all(np.isfinite(pos)):
            return
        approach_z = abs(float((-_quat_to_rot_wxyz(quat)[:, 2])[2]))
        label_l = str(label or "body").strip().lower()
        label_bonus = 0.0
        if "handle" in label_l:
            label_bonus = 2.0
        elif "topdown" in label_l or "top_down" in label_l:
            # Keep top-down candidates as explicit fallback below side/handle pinches.
            label_bonus = -0.2
        score = label_bonus + max(0.0, 1.0 - approach_z)
        out.append(
            {
                "label": label_l,
                "local_pos": pos,
                "local_quat": quat,
                "score": float(score),
                "source": source,
            }
        )

    functional = data.get("functional_grasp", {})
    if isinstance(functional, dict):
        for label, poses in functional.items():
            if isinstance(poses, list):
                src = f"functional_grasp.{str(label).strip().lower()}"
                for p in poses:
                    _append_pose(str(label), p, src)

    grasp = data.get("grasp", {})
    if isinstance(grasp, dict):
        body = grasp.get("body", [])
        if isinstance(body, list):
            for p in body:
                _append_pose("body", p, "grasp.body")

    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    meta_info["num_loaded"] = int(len(out))
    return out, meta_info


def _infer_annotation_position_scale(
    stage: Any,
    object_prim_path: str,
    candidates: Sequence[dict[str, Any]],
    usd: Any,
    usd_geom: Any,
    preset_pos_scale: float = 1.0,
) -> float:
    """Infer local translation scale correction for annotation poses.

    Some annotation files are authored in a different local scale than the prim.
    We compare median candidate translation norm with local mesh radius and only
    apply correction on clear mismatch.
    """
    if stage is None or not candidates:
        return 1.0
    try:
        prim = stage.GetPrimAtPath(object_prim_path)
        if prim is None or not prim.IsValid():
            return 1.0

        bbox_cache = usd_geom.BBoxCache(
            usd.TimeCode.Default(),
            [usd_geom.Tokens.default_],
            useExtentsHint=True,
        )
        bbox = bbox_cache.ComputeLocalBound(prim)
        rng = bbox.GetRange()
        min_pt = np.asarray(rng.GetMin(), dtype=np.float32)
        max_pt = np.asarray(rng.GetMax(), dtype=np.float32)
        half = 0.5 * np.abs(max_pt - min_pt)
        mesh_radius = float(np.max(half))
        if not np.isfinite(mesh_radius) or mesh_radius < 1e-6:
            return 1.0

        norms: list[float] = []
        preset = float(preset_pos_scale)
        if not np.isfinite(preset) or preset <= 0.0:
            preset = 1.0
        for cand in candidates:
            lp = _to_numpy(cand.get("local_pos"), dtype=np.float32).reshape(-1)
            if lp.size < 3 or not np.all(np.isfinite(lp[:3])):
                continue
            norms.append(float(np.linalg.norm(lp[:3] * preset)))
        if len(norms) < 3:
            return 1.0

        median_norm = float(np.median(np.asarray(norms, dtype=np.float32)))
        if not np.isfinite(median_norm) or median_norm < 1e-6:
            return 1.0

        # Only correct large mismatches (MagicSim-style conservative threshold).
        if median_norm > mesh_radius * 3.0 or median_norm < mesh_radius / 3.0:
            scale = float(np.clip(mesh_radius / median_norm, 1e-4, 1e4))
            return scale
        return 1.0
    except Exception as exc:
        LOG.debug("collect: annotation scale inference failed for %s: %s", object_prim_path, exc)
        return 1.0


def _sanitize_annotation_candidates(
    stage: Any,
    object_prim_path: str,
    candidates: Sequence[dict[str, Any]],
    usd: Any,
    usd_geom: Any,
    local_pos_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {
        "total": int(len(candidates)),
        "kept": 0,
        "dropped_non_finite": 0,
        "dropped_too_far": 0,
        "dropped_outside_bbox": 0,
        "rejected_file": False,
    }
    if not candidates:
        return [], report
    if stage is None:
        kept = list(candidates)
        report["kept"] = int(len(kept))
        return kept, report
    try:
        prim = stage.GetPrimAtPath(object_prim_path)
        if prim is None or not prim.IsValid():
            kept = list(candidates)
            report["kept"] = int(len(kept))
            return kept, report

        bbox_cache = usd_geom.BBoxCache(
            usd.TimeCode.Default(),
            [usd_geom.Tokens.default_],
            useExtentsHint=True,
        )
        bbox = bbox_cache.ComputeLocalBound(prim)
        rng = bbox.GetRange()
        min_pt = np.asarray(rng.GetMin(), dtype=np.float32)
        max_pt = np.asarray(rng.GetMax(), dtype=np.float32)
        half = 0.5 * np.abs(max_pt - min_pt)
        mesh_radius = float(np.max(half))
        if not np.isfinite(mesh_radius) or mesh_radius < 1e-6:
            kept = list(candidates)
            report["kept"] = int(len(kept))
            return kept, report

        margin = max(0.002, mesh_radius * float(ANNOTATION_BBOX_MARGIN_FACTOR))
        min_allowed = min_pt - margin
        max_allowed = max_pt + margin
        max_norm = mesh_radius * float(ANNOTATION_MAX_POS_RADIUS_FACTOR)
        scale = float(local_pos_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        kept: list[dict[str, Any]] = []
        for cand in candidates:
            lp = _to_numpy(cand.get("local_pos"), dtype=np.float32).reshape(-1)
            lq = _to_numpy(cand.get("local_quat"), dtype=np.float32).reshape(-1)
            if lp.size < 3 or lq.size < 4 or not np.all(np.isfinite(lp[:3])) or not np.all(np.isfinite(lq[:4])):
                report["dropped_non_finite"] = int(report["dropped_non_finite"]) + 1
                continue
            pos = lp[:3] * scale
            pos_norm = float(np.linalg.norm(pos))
            if not np.isfinite(pos_norm):
                report["dropped_non_finite"] = int(report["dropped_non_finite"]) + 1
                continue
            if pos_norm > max_norm:
                report["dropped_too_far"] = int(report["dropped_too_far"]) + 1
                continue
            if np.any(pos < min_allowed) or np.any(pos > max_allowed):
                report["dropped_outside_bbox"] = int(report["dropped_outside_bbox"]) + 1
                continue
            kept.append(cand)

        report["kept"] = int(len(kept))
        total = int(len(candidates))
        keep_ratio = float(len(kept) / max(1, total))
        report["keep_ratio"] = keep_ratio
        report["mesh_radius"] = float(mesh_radius)
        report["max_norm"] = float(max_norm)
        if ANNOTATION_VALIDATE_STRICT and total >= 6 and keep_ratio < float(ANNOTATION_MIN_VALID_KEEP_RATIO):
            report["rejected_file"] = True
            return [], report
        return kept, report
    except Exception as exc:
        LOG.warning("collect: annotation sanitize failed for %s: %s", object_prim_path, exc)
        kept = list(candidates)
        report["kept"] = int(len(kept))
        report["sanitize_error"] = str(exc)
        return kept, report


def _annotation_failed_pose_cache_key(
    object_prim_path: str,
    annotation_path: Optional[Path],
) -> str:
    ann = str(annotation_path) if annotation_path is not None else ""
    return f"{str(object_prim_path)}::{ann}"


def _get_failed_annotation_pose_ids(cache_key: str) -> set[int]:
    return set(_FAILED_ANNOTATION_POSE_CACHE.get(str(cache_key), set()))


def _mark_failed_annotation_pose_id(cache_key: str, pose_id: int) -> None:
    key = str(cache_key)
    failed = _FAILED_ANNOTATION_POSE_CACHE.setdefault(key, set())
    failed.add(int(pose_id))


def _clear_failed_annotation_pose_ids(cache_key: str) -> None:
    key = str(cache_key)
    if key in _FAILED_ANNOTATION_POSE_CACHE:
        del _FAILED_ANNOTATION_POSE_CACHE[key]


def _record_failed_annotation_target(
    cache_key: str,
    target: Optional[dict[str, Any]],
    reason: str,
    attempt: int,
) -> None:
    if not target or target.get("source") != "annotation":
        return
    idx_raw = target.get("index", None)
    try:
        idx = int(idx_raw)
    except Exception:
        return
    _mark_failed_annotation_pose_id(cache_key, idx)
    failed_ids = sorted(list(_get_failed_annotation_pose_ids(cache_key)))
    LOG.info(
        "collect: mark failed annotation pose id=%d reason=%s attempt=%d failed_ids=%s",
        idx,
        str(reason),
        int(attempt),
        failed_ids,
    )


def _select_annotation_grasp_target(
    stage: Any,
    object_prim_path: str,
    usd: Any,
    usd_geom: Any,
    candidates: Sequence[dict[str, Any]],
    attempt_index: int,
    ik_solver: Any | None = None,
    table_x_range: Optional[tuple[float, float]] = None,
    table_y_range: Optional[tuple[float, float]] = None,
    table_top_z: Optional[float] = None,
    failed_pose_ids: Optional[set[int]] = None,
    tip_mid_offset_in_hand: np.ndarray | None = None,
    annotation_pose_is_tip_mid_frame: bool = ANNOTATION_POSE_IS_TIP_MID_FRAME,
    local_pos_scale: float = 1.0,
) -> Optional[dict[str, Any]]:
    if not candidates:
        return None

    def _build_target_from_candidate(index: int) -> dict[str, Any]:
        cand = candidates[int(index)]
        local_pos = _to_numpy(cand.get("local_pos"), dtype=np.float32)[:3].astype(np.float32)
        if np.isfinite(float(local_pos_scale)) and abs(float(local_pos_scale) - 1.0) > 1e-6:
            local_pos = (local_pos * float(local_pos_scale)).astype(np.float32, copy=False)
        local_quat = _normalize_quat_wxyz(_to_numpy(cand.get("local_quat"), dtype=np.float32))
        converted = False
        if annotation_pose_is_tip_mid_frame and tip_mid_offset_in_hand is not None:
            local_pos = _convert_tip_mid_local_pose_to_hand_local(
                local_pos=local_pos,
                local_quat=local_quat,
                tip_mid_offset_in_hand=tip_mid_offset_in_hand,
            )
            converted = True
        obj_pos, obj_quat = _get_prim_world_pose(stage, object_prim_path, usd, usd_geom)
        target_pos, target_quat = _transform_local_pose_to_world(
            object_pos=obj_pos,
            object_quat=obj_quat,
            local_pos=local_pos,
            local_quat=local_quat,
        )
        return {
            "target_pos": target_pos.astype(np.float32),
            "target_quat": target_quat.astype(np.float32),
            "local_pos": local_pos,
            "local_quat": local_quat,
            "label": str(cand.get("label", "body")),
            "source": "annotation",
            "source_id": GRASP_POSE_SOURCE_IDS["annotation"],
            "index": int(index),
            "annotation_tip_mid_to_hand_converted": bool(converted),
            "ik_feasible": False,
            "ik_orientation_relaxed": False,
            "candidate_group": "body",
        }

    def _rotate(indices: list[int], shift_seed: int) -> list[int]:
        if not indices:
            return []
        shift = int(max(0, shift_seed)) % len(indices)
        if shift == 0:
            return indices
        return indices[shift:] + indices[:shift]

    def _local_z_for_candidate(index: int) -> float | None:
        arr = _to_numpy(candidates[int(index)].get("local_pos"), dtype=np.float32).reshape(-1)
        if arr.size < 3 or not np.isfinite(arr[2]):
            return None
        return float(arr[2])

    def _sort_indices(indices: list[int]) -> list[int]:
        def _key(idx: int) -> tuple[float, float]:
            cand = candidates[int(idx)]
            score = float(cand.get("score", 0.0))
            z = _local_z_for_candidate(int(idx))
            return score, (-1e9 if z is None else z)

        return sorted(indices, key=_key, reverse=True)

    def _compute_low_z_exclusion(
        indices: list[int],
        percentile: float,
    ) -> tuple[set[int], float | None]:
        if not indices:
            return set(), None
        vals: list[tuple[int, float]] = []
        for idx in indices:
            z = _local_z_for_candidate(idx)
            if z is None:
                continue
            vals.append((int(idx), float(z)))
        min_candidates_for_filter = max(8, ANNOTATION_MIN_GROUP_KEEP + 3)
        if len(vals) < min_candidates_for_filter:
            return set(), None
        z_values = np.array([z for _, z in vals], dtype=np.float32)
        floor = float(np.percentile(z_values, float(percentile) * 100.0))
        low = {idx for idx, z in vals if z < floor}
        if len(indices) - len(low) < ANNOTATION_MIN_GROUP_KEEP:
            keep_ids = {
                idx for idx, _ in sorted(vals, key=lambda x: x[1], reverse=True)[:ANNOTATION_MIN_GROUP_KEEP]
            }
            low = {idx for idx, _ in vals if idx not in keep_ids}
        return low, floor

    failed_ids = {int(x) for x in (failed_pose_ids or set()) if int(x) >= 0}
    handle_indices: list[int] = []
    body_indices: list[int] = []
    topdown_indices: list[int] = []
    for i, cand in enumerate(candidates):
        label = str(cand.get("label", "body")).lower()
        src = str(cand.get("source", "")).lower()
        if "handle" in label or src.endswith(".handle"):
            handle_indices.append(i)
        elif "topdown" in label or "top_down" in label or src.endswith(".topdown") or src.endswith(".topdown_side"):
            topdown_indices.append(i)
        else:
            body_indices.append(i)

    handle_indices = _sort_indices(handle_indices)
    body_indices = _sort_indices(body_indices)
    topdown_indices = _sort_indices(topdown_indices)
    all_indices = handle_indices + body_indices + topdown_indices
    global_low_z_ids, global_low_z_floor = _compute_low_z_exclusion(
        all_indices,
        ANNOTATION_MIN_LOCAL_Z_PERCENTILE,
    )
    body_low_z_ids, body_low_z_floor = _compute_low_z_exclusion(
        body_indices,
        ANNOTATION_BODY_MIN_LOCAL_Z_PERCENTILE,
    )

    ordered_groups: list[tuple[str, list[int]]] = [
        ("functional_handle", _rotate(handle_indices, attempt_index - 1)),
        ("body", _rotate(body_indices, attempt_index - 1)),
        ("topdown", _rotate(topdown_indices, attempt_index - 1)),
    ]

    filtered_groups: list[tuple[str, list[int]]] = []
    low_z_removed_summary: dict[str, int] = {}
    for group_name, indices in ordered_groups:
        if not indices:
            continue
        kept = indices
        if failed_ids and len(indices) > 1:
            survived = [idx for idx in indices if idx not in failed_ids]
            if survived:
                kept = survived
        low_z_ids = body_low_z_ids if group_name == "body" else global_low_z_ids
        if low_z_ids and len(kept) > ANNOTATION_MIN_GROUP_KEEP:
            survived = [idx for idx in kept if idx not in low_z_ids]
            if len(survived) >= ANNOTATION_MIN_GROUP_KEEP:
                removed = len(kept) - len(survived)
                if removed > 0:
                    low_z_removed_summary[group_name] = removed
                kept = survived
        filtered_groups.append((group_name, kept))

    if low_z_removed_summary:
        LOG.info(
            "collect: annotation low-z pruning removed=%s floors(global=%.4f, body=%.4f)",
            low_z_removed_summary,
            float(global_low_z_floor if global_low_z_floor is not None else float("nan")),
            float(body_low_z_floor if body_low_z_floor is not None else float("nan")),
        )

    if not filtered_groups:
        # Fallback when all candidates have failed before: allow retry from raw list.
        LOG.warning(
            "collect: annotation selector all candidates filtered by failed_ids=%s and low-z pruning; retrying raw ordered list",
            sorted(list(failed_ids)),
        )
        filtered_groups = ordered_groups

    workspace_margin = 0.04

    def _target_in_workspace(target: dict[str, Any]) -> bool:
        pos = _to_numpy(target.get("target_pos"), dtype=np.float32).reshape(-1)
        if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
            return False
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if (
            table_x_range is not None
            and not (float(table_x_range[0]) - workspace_margin <= x <= float(table_x_range[1]) + workspace_margin)
        ):
            return False
        if (
            table_y_range is not None
            and not (float(table_y_range[0]) - workspace_margin <= y <= float(table_y_range[1]) + workspace_margin)
        ):
            return False
        if table_top_z is not None and np.isfinite(float(table_top_z)):
            if z < float(table_top_z) - 0.03:
                return False
        return True

    first_fallback: Optional[dict[str, Any]] = None
    max_scan = int(ANNOTATION_IK_SCREEN_MAX_CANDIDATES)
    scanned_count = 0
    workspace_ok_count = 0

    for group_name, indices in filtered_groups:
        if not indices:
            continue

        strict_pass_targets: list[dict[str, Any]] = []
        for idx in indices[:max_scan]:
            scanned_count += 1
            target = _build_target_from_candidate(idx)
            target["candidate_group"] = group_name
            if first_fallback is None:
                first_fallback = target
            if not _target_in_workspace(target):
                continue
            workspace_ok_count += 1
            strict_pass_targets.append(target)

        if not strict_pass_targets:
            continue

        if ik_solver is None:
            return strict_pass_targets[0]

        # Pass 1: strict orientation (MagicSim-like annotation orientation fidelity).
        for target in strict_pass_targets:
            q = _solve_ik_arm_target(
                ik_solver=ik_solver,
                target_position=_to_numpy(target.get("target_pos"), dtype=np.float32),
                target_orientation=_to_numpy(target.get("target_quat"), dtype=np.float32),
            )
            if q is None:
                continue
            target["ik_feasible"] = True
            target["ik_orientation_relaxed"] = False
            target["ik_joint_positions"] = q.astype(np.float32, copy=False)
            return target

        # Pass 2: relax orientation only if strict pass had no solution.
        for target in strict_pass_targets:
            q = _solve_ik_arm_target(
                ik_solver=ik_solver,
                target_position=_to_numpy(target.get("target_pos"), dtype=np.float32),
                target_orientation=None,
            )
            if q is None:
                continue
            target["ik_feasible"] = True
            target["ik_orientation_relaxed"] = True
            target["ik_joint_positions"] = q.astype(np.float32, copy=False)
            return target

    if first_fallback is not None:
        LOG.warning(
            "collect: annotation selector fallback target idx=%s group=%s (scanned=%d, workspace_ok=%d, failed_ids=%s)",
            str(first_fallback.get("index", "na")),
            str(first_fallback.get("candidate_group", "body")),
            scanned_count,
            workspace_ok_count,
            sorted(list(failed_ids)),
        )
    return first_fallback


def _build_frame_extras(
    stage: Any,
    object_prim_path: str,
    usd: Any,
    usd_geom: Any,
    current_target: Optional[dict[str, Any]],
) -> dict[str, Any]:
    obj_pos, obj_quat = _get_object_tracking_pose(stage, object_prim_path, usd, usd_geom)
    object_pose = np.concatenate([obj_pos[:3], _normalize_quat_wxyz(obj_quat)], axis=0).astype(np.float32)

    source_id = GRASP_POSE_SOURCE_IDS["none"]
    target_pose = object_pose.copy()
    valid = False
    if current_target:
        if current_target.get("source") == "annotation":
            tgt_pos, tgt_quat = _transform_local_pose_to_world(
                object_pos=obj_pos,
                object_quat=obj_quat,
                local_pos=_to_numpy(current_target.get("local_pos"), dtype=np.float32),
                local_quat=_to_numpy(current_target.get("local_quat"), dtype=np.float32),
            )
            target_pose = np.concatenate([tgt_pos[:3], _normalize_quat_wxyz(tgt_quat)], axis=0).astype(np.float32)
            source_id = GRASP_POSE_SOURCE_IDS["annotation"]
            valid = True
        else:
            pos = _to_numpy(current_target.get("target_pos"), dtype=np.float32).reshape(-1)
            quat = _normalize_quat_wxyz(_to_numpy(current_target.get("target_quat"), dtype=np.float32))
            if pos.size >= 3 and np.all(np.isfinite(pos[:3])):
                target_pose = np.concatenate([pos[:3], quat], axis=0).astype(np.float32)
                source_id = float(current_target.get("source_id", GRASP_POSE_SOURCE_IDS["none"]))
                valid = bool(current_target.get("valid", True))

    return {
        "observation.object_pose_world": object_pose.tolist(),
        "observation.grasp_target_pose_world": target_pose.tolist(),
        "observation.grasp_target_valid": bool(valid),
        "observation.grasp_target_source_id": float(source_id),
    }


def _compute_live_grasp_target_pose(
    stage: Any,
    object_prim_path: str,
    usd: Any,
    usd_geom: Any,
    current_target: Optional[dict[str, Any]],
    z_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve grasp target pose from live object ground truth.

    For annotation targets, keep local grasp pose attached to the object's live pose.
    For fallback targets, track live object center in XY and preserve target Z/orientation when available.
    """
    obj_pos, obj_quat = _get_object_tracking_pose(stage, object_prim_path, usd, usd_geom)
    tgt_pos = obj_pos.astype(np.float32, copy=True)
    tgt_quat = _normalize_quat_wxyz(obj_quat)

    if current_target:
        if current_target.get("source") == "annotation":
            tgt_pos, tgt_quat = _transform_local_pose_to_world(
                object_pos=obj_pos,
                object_quat=obj_quat,
                local_pos=_to_numpy(current_target.get("local_pos"), dtype=np.float32),
                local_quat=_to_numpy(current_target.get("local_quat"), dtype=np.float32),
            )
            tgt_pos = _to_numpy(tgt_pos, dtype=np.float32)[:3]
            tgt_quat = _normalize_quat_wxyz(_to_numpy(tgt_quat, dtype=np.float32))
        else:
            raw_pos = _to_numpy(current_target.get("target_pos"), dtype=np.float32).reshape(-1)
            raw_quat = _to_numpy(current_target.get("target_quat"), dtype=np.float32).reshape(-1)
            if raw_pos.size >= 3 and np.isfinite(raw_pos[2]):
                tgt_pos[2] = float(raw_pos[2])
            if raw_quat.size >= 4:
                tgt_quat = _normalize_quat_wxyz(raw_quat[:4])

    if z_override is not None and np.isfinite(float(z_override)):
        tgt_pos[2] = float(z_override)

    return tgt_pos.astype(np.float32, copy=False), tgt_quat.astype(np.float32, copy=False)


def _select_ik_frame_name(frame_names: Sequence[str], eef_prim_path: str) -> Optional[str]:
    if not frame_names:
        return None
    available = set(frame_names)
    leaf = str(eef_prim_path or "").split("/")[-1]
    # Keep IK target frame aligned with collector pose/metric frame.
    # Prefer hand-link style frames before custom "right_gripper" aliases.
    preferred = [leaf, "panda_hand", "panda_hand_tcp", "right_gripper", "panda_link8", "panda_link7"]
    for name in preferred:
        if name and name in available:
            return name
    for name in frame_names:
        low = str(name).lower()
        if "gripper" in low or "hand" in low:
            return name
    return frame_names[0]


def _create_franka_ik_solver(
    franka: Any,
    stage: Any,
    robot_prim_path: str,
    eef_prim_path: str,
    usd: Any,
    usd_geom: Any,
) -> Any | None:
    try:
        from omni.isaac.motion_generation import (
            ArticulationKinematicsSolver,
            LulaKinematicsSolver,
            interface_config_loader,
        )

        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        lula = LulaKinematicsSolver(**cfg)
        base_pos, base_quat = _get_prim_world_pose(stage, robot_prim_path, usd, usd_geom)
        lula.set_robot_base_pose(base_pos, base_quat)
        frame_name = _select_ik_frame_name(lula.get_all_frame_names(), eef_prim_path)
        if not frame_name:
            LOG.warning("IK init skipped: no valid end-effector frame")
            return None
        solver = ArticulationKinematicsSolver(franka, lula, frame_name)
        leaf = str(eef_prim_path or "").split("/")[-1]
        if leaf and frame_name != leaf:
            LOG.warning(
                "IK frame differs from eef prim leaf: eef_prim=%s leaf=%s ik_frame=%s",
                eef_prim_path,
                leaf,
                frame_name,
            )
        LOG.info("IK solver ready: frame=%s robot=%s", frame_name, robot_prim_path)
        return solver
    except Exception as exc:
        LOG.warning("IK solver init failed: %s", exc)
        return None


def _solve_ik_arm_target(
    ik_solver: Any,
    target_position: np.ndarray,
    target_orientation: np.ndarray | None = None,
) -> np.ndarray | None:
    if ik_solver is None:
        return None
    try:
        orient = None
        if target_orientation is not None:
            orient = np.asarray(target_orientation, dtype=np.float32).reshape(-1)
            if orient.size >= 4:
                orient = orient[:4]
                norm = float(np.linalg.norm(orient))
                if np.isfinite(norm) and norm > 1e-6:
                    orient = (orient / norm).astype(np.float32)
                else:
                    orient = None
            else:
                orient = None
        action, success = ik_solver.compute_inverse_kinematics(
            target_position=np.asarray(target_position, dtype=np.float32),
            target_orientation=orient,
            position_tolerance=0.01,
        )
        if not success or action is None:
            return None
        joints = _to_numpy(getattr(action, "joint_positions", None))
        if joints.size < 7:
            return None
        return np.clip(
            joints[:7].astype(np.float32, copy=False),
            FRANKA_ARM_LOWER,
            FRANKA_ARM_UPPER,
        )
    except Exception as exc:
        LOG.warning("IK solve failed for target %s: %s", np.asarray(target_position).tolist(), exc)
        return None


def _prim_is_valid(stage: Any, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    return prim is not None and prim.IsValid()


def _looks_like_franka_root(stage: Any, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    has_hand = False
    has_link0 = False
    for sub in _iter_descendants(prim):
        name = sub.GetName().lower()
        if "panda_hand" in name:
            has_hand = True
        if "panda_link0" in name:
            has_link0 = True
        if has_hand and has_link0:
            return True
    return False


def _has_articulation_root(stage: Any, prim_path: str, usd_physics: Any) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    try:
        if prim.HasAPI(usd_physics.ArticulationRootAPI):
            return True
    except Exception:
        pass
    for sub in _iter_descendants(prim):
        if not sub or not sub.IsValid():
            continue
        try:
            if sub.HasAPI(usd_physics.ArticulationRootAPI):
                return True
        except Exception:
            continue
    return False


def _resolve_articulation_root_prim_path(stage: Any, prim_path: str, usd_physics: Any) -> Optional[str]:
    """Return concrete articulation root prim path for a Franka hierarchy."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    try:
        if prim.HasAPI(usd_physics.ArticulationRootAPI):
            return prim.GetPath().pathString
    except Exception:
        pass
    for sub in _iter_descendants(prim):
        if not sub or not sub.IsValid():
            continue
        try:
            if sub.HasAPI(usd_physics.ArticulationRootAPI):
                return sub.GetPath().pathString
        except Exception:
            continue
    return None


def _resolve_robot_prim(stage: Any) -> Optional[str]:
    candidates = [
        "/World/FrankaRobot",
        "/World/Franka",
        "/World/franka",
        f"{COLLECTOR_ROOT}/Franka",
    ]
    for path in candidates:
        if _prim_is_valid(stage, path) and _looks_like_franka_root(stage, path):
            return path

    discovered: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        path = prim.GetPath().pathString
        if not path.startswith("/World/") or path.count("/") != 2:
            continue
        if "franka" in path.lower() and _looks_like_franka_root(stage, path):
            discovered.append(path)
    return discovered[0] if discovered else None


def _franka_usd_candidates(assets_root: str) -> list[str]:
    root = str(assets_root or "").rstrip("/")
    candidates = [
        # Prefer known-good public Isaac asset URLs first (used by scene-chat path).
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Robots/Franka/franka.usd",
    ]
    if root:
        candidates.extend(
            [
                root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                root + "/Isaac/Robots/Franka/franka.usd",
            ]
        )
    # Keep order but remove duplicates.
    out: list[str] = []
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _add_franka_reference(
    stage: Any,
    simulation_app: Any,
    add_reference_to_stage: Any,
    assets_root: str,
    usd_geom: Any,
    gf: Any,
    usd_physics: Any,
    prim_path: str = "/World/FrankaRobot",
) -> str:
    added = False
    for franka_usd in _franka_usd_candidates(assets_root):
        add_reference_to_stage(usd_path=franka_usd, prim_path=prim_path)
        for _ in range(8):
            simulation_app.update()
        if _looks_like_franka_root(stage, prim_path) and _has_articulation_root(stage, prim_path, usd_physics):
            added = True
            break
        _remove_prim_if_exists(stage, prim_path)
    if not added:
        raise RuntimeError("collect cannot add Franka: no valid Franka asset path available")

    franka_prim = stage.GetPrimAtPath(prim_path)
    if franka_prim and franka_prim.IsValid():
        xf = usd_geom.Xformable(franka_prim)
        tx = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                tx = op
                break
        if tx is None:
            tx = xf.AddTranslateOp()
        tx.Set(gf.Vec3d(0.40, 0.00, 0.77))
    return prim_path


def _wait_articulation_ready(franka: Any, world: Any, steps: int = 30) -> None:
    """Block until articulation exposes joint state, else raise clear error."""
    last_exc: Optional[Exception] = None
    for _ in range(max(1, int(steps))):
        try:
            joints = _to_numpy(franka.get_joint_positions())
            if joints.size > 0:
                return
        except Exception as exc:
            last_exc = exc
        try:
            world.step(render=False)
        except Exception:
            pass
    if last_exc is not None:
        raise RuntimeError(f"Franka articulation not ready: {last_exc}")
    raise RuntimeError("Franka articulation not ready: no joint states available")


def _iter_world_root_prims(stage: Any) -> list[Any]:
    roots = []
    world = stage.GetPrimAtPath("/World")
    if not world or not world.IsValid():
        return roots
    for child in world.GetChildren():
        if child and child.IsValid():
            roots.append(child)
    return roots


def _iter_descendants(prim: Any):
    stack = list(prim.GetChildren()) if prim and prim.IsValid() else []
    while stack:
        node = stack.pop()
        yield node
        children = list(node.GetChildren()) if node and node.IsValid() else []
        if children:
            stack.extend(children)


def _is_geom_prim(prim: Any, usd_geom: Any) -> bool:
    if not prim or not prim.IsValid():
        return False
    try:
        if prim.IsA(usd_geom.Gprim):
            return True
    except Exception:
        pass
    try:
        if prim.IsA(usd_geom.Mesh):
            return True
    except Exception:
        pass
    return False


def _root_has_mesh_descendant(prim: Any, usd_geom: Any) -> bool:
    if not prim or not prim.IsValid():
        return False
    if _is_geom_prim(prim, usd_geom):
        return True
    for sub in _iter_descendants(prim):
        if sub and sub.IsValid() and _is_geom_prim(sub, usd_geom):
            return True
    return False


def _resolve_table_prim(stage: Any, usd_geom: Any) -> Optional[str]:
    explicit_candidates = [
        "/World/Table",
        "/World/table",
        f"{TABLE_PRIM_PATH}",
    ]
    for path in explicit_candidates:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and _root_has_mesh_descendant(prim, usd_geom):
            return path

    skip_tokens = {
        "defaultgroundplane",
        "ground",
        "physicsscene",
        "light",
        "camera",
        "franka",
        "looks",
        "render",
    }
    name_tokens = ("table", "desk", "counter", "workbench")

    best_path: Optional[str] = None
    best_score = -1.0
    for root in _iter_world_root_prims(stage):
        if not root or not root.IsValid():
            continue
        name = root.GetName().lower()
        if any(tok in name for tok in skip_tokens):
            continue
        if not _root_has_mesh_descendant(root, usd_geom):
            continue
        bbox = _compute_prim_bbox(stage, root.GetPath().pathString, usd_geom)
        if not bbox:
            continue
        mn, mx = bbox
        sx = float(mx[0] - mn[0])
        sy = float(mx[1] - mn[1])
        sz = float(mx[2] - mn[2])
        top_z = float(mx[2])
        if not np.isfinite(sx) or not np.isfinite(sy) or not np.isfinite(sz):
            continue
        # Keep candidates near typical tabletop geometry.
        if sx < 0.35 or sy < 0.35:
            continue
        if top_z < 0.45 or top_z > 1.35:
            continue

        area = sx * sy
        flat_ratio = min(sx, sy) / max(sz, 0.02)
        score = area
        if any(tok in name for tok in name_tokens):
            score += 5.0
        score += min(flat_ratio, 30.0) * 0.05
        if score > best_score:
            best_score = score
            best_path = root.GetPath().pathString

    return best_path


def _root_is_graspable_candidate(prim: Any, usd_geom: Any) -> bool:
    if not prim or not prim.IsValid():
        return False
    name = prim.GetName().lower()
    if name in {"defaultgroundplane", "ground", "physicsscene"}:
        return False
    if "table" in name or "franka" in name or "camera" in name:
        return False
    if "light" in name or "looks" in name or "render" in name:
        return False
    if name.startswith("omniversekit_"):
        return False

    if prim.IsA(usd_geom.Camera):
        return False
    if _is_geom_prim(prim, usd_geom):
        return True
    # Require at least one mesh descendant.
    for sub in _iter_descendants(prim):
        if sub and sub.IsValid() and _is_geom_prim(sub, usd_geom):
            return True
    return False


def _normalize_name_token(value: str) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def _match_target_hint(name: str, hints: Sequence[str]) -> bool:
    if not hints:
        return False
    n = _normalize_name_token(name)
    for hint in hints:
        h = _normalize_name_token(str(hint))
        if not h:
            continue
        if h in n or n in h:
            return True
    return False


def _prim_matches_target_hint_deep(prim: Any, hints: Sequence[str]) -> bool:
    """Match hints against root + descendant names/paths.

    This helps disambiguate assets like different mug variants where the root
    prim may be generic (/World/Mug) but children contain SM_Mug_C1/B1 tokens.
    """
    if prim is None or not prim.IsValid() or not hints:
        return False
    try:
        if _match_target_hint(prim.GetName(), hints):
            return True
        if _match_target_hint(prim.GetPath().pathString, hints):
            return True
    except Exception:
        pass

    for sub in _iter_descendants(prim):
        if sub is None or not sub.IsValid():
            continue
        try:
            if _match_target_hint(sub.GetName(), hints):
                return True
            if _match_target_hint(sub.GetPath().pathString, hints):
                return True
        except Exception:
            continue
    return False


def _resolve_target_object_prim(
    stage: Any,
    usd_geom: Any,
    target_hints: Optional[Sequence[str]] = None,
    table_x_range: Optional[tuple[float, float]] = None,
    table_y_range: Optional[tuple[float, float]] = None,
    table_top_z: Optional[float] = None,
) -> Optional[str]:
    roots = _iter_world_root_prims(stage)
    candidates: list[tuple[Any, np.ndarray, np.ndarray]] = []
    for root in roots:
        if not _root_is_graspable_candidate(root, usd_geom):
            continue
        bbox = _compute_prim_bbox(stage, root.GetPath().pathString, usd_geom)
        if not bbox:
            continue
        mn, mx = bbox
        size = mx - mn
        max_dim = float(np.max(size))
        min_dim = float(np.min(size))
        # Prefer handheld tabletop items, not large furniture.
        if max_dim > 0.45 or min_dim < 0.01:
            continue
        candidates.append((root, mn, mx))
    if not candidates:
        return None

    def _score_candidate(prim: Any, mn: np.ndarray, mx: np.ndarray) -> float:
        path = prim.GetPath().pathString
        name = prim.GetName().lower()
        center = 0.5 * (mn + mx)
        size = mx - mn
        max_dim = float(np.max(size))
        min_dim = float(np.min(size))
        score = 0.0

        # Strongly prefer the explicitly requested object family when hints exist.
        if target_hints and _prim_matches_target_hint_deep(prim, target_hints):
            score += 8.0

        # Prefer common tabletop manipulands when no explicit hint is supplied.
        preferred_tokens = [
            "mug",
            "mustard",
            "sugar",
            "cracker",
            "tomato",
            "pickobject",
            "object",
        ]
        for rank, token in enumerate(preferred_tokens):
            if token in name:
                score += max(0.5, 4.0 - 0.35 * float(rank))
                break

        if (
            table_x_range is not None
            and table_y_range is not None
            and np.all(np.isfinite(center[:2]))
        ):
            margin = 0.05
            in_xy = (
                (float(table_x_range[0]) - margin)
                <= float(center[0])
                <= (float(table_x_range[1]) + margin)
                and (float(table_y_range[0]) - margin)
                <= float(center[1])
                <= (float(table_y_range[1]) + margin)
            )
            score += 3.0 if in_xy else -2.0

        if table_top_z is not None and np.isfinite(float(table_top_z)):
            bottom_z = float(mn[2])
            top_z = float(mx[2])
            z_gap = abs(bottom_z - float(table_top_z))
            if z_gap <= 0.08:
                score += 4.0
            elif z_gap <= 0.20:
                score += 2.0
            else:
                score -= min(8.0, (z_gap - 0.20) * 20.0)
            if top_z < float(table_top_z) - 0.02:
                score -= 8.0

        # Bias toward realistic handheld dimensions.
        score -= abs(max_dim - 0.09) * 3.0
        if min_dim < 0.012:
            score -= 1.5

        # Prefer assets that already have grasp annotations.
        ann_path = _resolve_grasp_annotation_path(stage=stage, object_prim_path=path)
        if ann_path is not None:
            score += 1.5
        return score

    filtered = candidates
    if target_hints:
        hinted: list[tuple[Any, np.ndarray, np.ndarray]] = []
        for prim, mn, mx in candidates:
            if _prim_matches_target_hint_deep(prim, target_hints):
                hinted.append((prim, mn, mx))
        if hinted:
            filtered = hinted

    best_path: Optional[str] = None
    best_score = -1e9
    for prim, mn, mx in filtered:
        score = _score_candidate(prim, mn, mx)
        if score > best_score:
            best_score = score
            best_path = prim.GetPath().pathString
    return best_path


def _compute_prim_bbox(stage: Any, prim_path: str, usd_geom: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    try:
        cache = usd_geom.BBoxCache(0, ["default", "render"])
        rng = cache.ComputeWorldBound(prim).GetRange()
        mn = np.array([float(rng.GetMin()[0]), float(rng.GetMin()[1]), float(rng.GetMin()[2])], dtype=np.float32)
        mx = np.array([float(rng.GetMax()[0]), float(rng.GetMax()[1]), float(rng.GetMax()[2])], dtype=np.float32)
        if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
            return None
        return mn, mx
    except Exception:
        return None


def _bbox_center_if_reasonable(
    mn: np.ndarray,
    mx: np.ndarray,
    root_pos: np.ndarray,
) -> Optional[np.ndarray]:
    if not (np.all(np.isfinite(mn)) and np.all(np.isfinite(mx))):
        return None
    ext = (mx - mn).astype(np.float32, copy=False)
    if np.any(ext <= 1e-5):
        return None
    if float(np.max(ext)) > float(TRACKING_BBOX_MAX_EXTENT_M):
        return None
    center = (0.5 * (mn + mx)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(center)):
        return None
    root = _to_numpy(root_pos, dtype=np.float32).reshape(-1)
    if root.size < 3 or not np.all(np.isfinite(root[:3])):
        return center
    if float(np.linalg.norm(center[:3] - root[:3])) > float(TRACKING_BBOX_MAX_ROOT_DELTA_M):
        return None
    return center


def _infer_table_workspace(
    stage: Any,
    usd_geom: Any,
    table_prim_path: Optional[str] = None,
) -> tuple[tuple[float, float], tuple[float, float], float, tuple[float, float]]:
    table_candidates: list[str] = []
    if table_prim_path:
        table_candidates.extend([f"{table_prim_path}/Top", table_prim_path])
    table_candidates.extend(
        [
            "/World/Table/Top",
            "/World/Table",
            f"{TABLE_PRIM_PATH}/Top",
            TABLE_PRIM_PATH,
        ]
    )

    seen = set()
    for p in table_candidates:
        if p in seen:
            continue
        seen.add(p)
        bbox = _compute_prim_bbox(stage, p, usd_geom)
        if not bbox:
            continue
        mn, mx = bbox
        size_x = float(mx[0] - mn[0])
        size_y = float(mx[1] - mn[1])
        top_z = float(mx[2])
        # Reject tiny / malformed table candidates (common in imported scenes
        # where an empty Xform or decorative mesh sits at /World/Table).
        if size_x < 0.30 or size_y < 0.30:
            continue
        if (not np.isfinite(top_z)) or top_z < 0.25 or top_z > 1.60:
            continue
        x_min = float(mn[0] + TABLE_MARGIN)
        x_max = float(mx[0] - TABLE_MARGIN)
        y_min = float(mn[1] + TABLE_MARGIN)
        y_max = float(mx[1] - TABLE_MARGIN)
        if x_max <= x_min or y_max <= y_min:
            continue
        center = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5)
        return (x_min, x_max), (y_min, y_max), top_z, center

    # Fallback for generated scene defaults.
    center = ((TABLE_X_RANGE[0] + TABLE_X_RANGE[1]) * 0.5, (TABLE_Y_RANGE[0] + TABLE_Y_RANGE[1]) * 0.5)
    return TABLE_X_RANGE, TABLE_Y_RANGE, TABLE_Z, center


def _adjust_workspace_for_robot_reach(
    stage: Any,
    usd_geom: Any,
    robot_prim_path: str,
    table_x_range: tuple[float, float],
    table_y_range: tuple[float, float],
    table_top_z: float,
    work_center: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], float, tuple[float, float]]:
    robot_bbox = _compute_prim_bbox(stage, robot_prim_path, usd_geom)
    if robot_bbox:
        rmn, rmx = robot_bbox
        robot_xy = (float((rmn[0] + rmx[0]) * 0.5), float((rmn[1] + rmx[1]) * 0.5))
    else:
        robot_xy = (float(work_center[0]), float(work_center[1]))

    top_z = float(table_top_z)
    if (not np.isfinite(top_z)) or top_z < 0.30 or top_z > 1.35:
        LOG.warning("collect: abnormal table_top_z=%.3f, fallback to 0.770", top_z)
        top_z = 0.77

    # Keep sampled points inside table bounds first. Then intersect with a
    # broad Franka reach disk if valid; otherwise keep full table range.
    rx, ry = robot_xy
    reach_x = (rx - ROBOT_REACH_RADIUS_X, rx + ROBOT_REACH_RADIUS_X)
    reach_y = (ry - ROBOT_REACH_RADIUS_Y, ry + ROBOT_REACH_RADIUS_Y)

    x_min = max(float(table_x_range[0]), float(reach_x[0]))
    x_max = min(float(table_x_range[1]), float(reach_x[1]))
    y_min = max(float(table_y_range[0]), float(reach_y[0]))
    y_max = min(float(table_y_range[1]), float(reach_y[1]))

    if (x_max - x_min) >= REACH_INTERSECTION_MIN_SPAN and (y_max - y_min) >= REACH_INTERSECTION_MIN_SPAN:
        x_range = (float(x_min), float(x_max))
        y_range = (float(y_min), float(y_max))
    else:
        LOG.warning(
            "collect: robot/table reach intersection too small (robot=(%.3f, %.3f), table_x=(%.3f, %.3f), "
            "table_y=(%.3f, %.3f)); using full table workspace",
            rx,
            ry,
            float(table_x_range[0]),
            float(table_x_range[1]),
            float(table_y_range[0]),
            float(table_y_range[1]),
        )
        x_range = (float(table_x_range[0]), float(table_x_range[1]))
        y_range = (float(table_y_range[0]), float(table_y_range[1]))
    center = (float((x_range[0] + x_range[1]) * 0.5), float((y_range[0] + y_range[1]) * 0.5))
    return x_range, y_range, top_z, center


def _get_prim_world_pose(stage: Any, prim_path: str, usd: Any, usd_geom: Any) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros(3, dtype=np.float32)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return pos, quat
    try:
        xf = usd_geom.Xformable(prim)
        tf = xf.ComputeLocalToWorldTransform(usd.TimeCode.Default())
        t = tf.ExtractTranslation()
        q = tf.ExtractRotationQuat()
        imag = q.GetImaginary()
        pos = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)
        quat = np.array([float(q.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])], dtype=np.float32)
    except Exception:
        pass
    return pos, quat


def _get_object_tracking_pose(
    stage: Any,
    object_prim_path: str,
    usd: Any,
    usd_geom: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Stable object pose for logging/verification.

    For many referenced assets, root pivot is not the visual/physical center.
    Use bbox center for position tracking, while keeping root quaternion.
    """
    pos, quat = _get_prim_world_pose(stage, object_prim_path, usd, usd_geom)
    bbox = _compute_prim_bbox(stage, object_prim_path, usd_geom)
    if bbox:
        mn, mx = bbox
        center = _bbox_center_if_reasonable(mn=mn, mx=mx, root_pos=pos)
        if center is not None:
            pos = center
    return pos, quat


def _set_prim_world_position(stage: Any, prim_path: str, position: np.ndarray, usd_geom: Any, gf: Any) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Target prim not found: {prim_path}")
    xf = usd_geom.Xformable(prim)
    tx = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            tx = op
            break
    if tx is None:
        tx = xf.AddTranslateOp()
    tx.Set(gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))


def _set_prim_upright_yaw_deg(stage: Any, prim_path: str, yaw_deg: float, usd_geom: Any, gf: Any) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Target prim not found: {prim_path}")
    xf = usd_geom.Xformable(prim)
    rotate_xyz = None
    rotate_z = None
    for op in xf.GetOrderedXformOps():
        name = op.GetOpName()
        if name == "xformOp:rotateXYZ":
            rotate_xyz = op
        elif name == "xformOp:rotateZ":
            rotate_z = op
        elif name in {"xformOp:rotateX", "xformOp:rotateY"}:
            try:
                op.Set(0.0)
            except Exception:
                pass

    if rotate_xyz is not None:
        rotate_xyz.Set(gf.Vec3f(0.0, 0.0, float(yaw_deg)))
        return
    if rotate_z is None:
        rotate_z = xf.AddRotateZOp()
    rotate_z.Set(float(yaw_deg))


def _align_robot_base_for_collect(
    stage: Any,
    robot_prim_path: str,
    usd_geom: Any,
    gf: Any,
    table_x_range: tuple[float, float],
    table_y_range: tuple[float, float],
    table_top_z: float,
    work_center: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Move robot base to a stable tabletop edge pose and face table center."""
    span_x = float(table_x_range[1] - table_x_range[0])
    margin_x = min(0.10, max(0.04, span_x * 0.10))
    target_x = float(np.clip(work_center[0] + span_x * 0.42, table_x_range[0] + margin_x, table_x_range[1] - margin_x))
    target_y = float(np.clip(work_center[1], table_y_range[0] + 0.05, table_y_range[1] - 0.05))
    target_z = float(table_top_z)
    _set_prim_world_position(
        stage=stage,
        prim_path=robot_prim_path,
        position=np.array([target_x, target_y, target_z], dtype=np.float32),
        usd_geom=usd_geom,
        gf=gf,
    )
    yaw = float(math.degrees(math.atan2(float(work_center[1]) - target_y, float(work_center[0]) - target_x)))
    _set_prim_upright_yaw_deg(
        stage=stage,
        prim_path=robot_prim_path,
        yaw_deg=yaw,
        usd_geom=usd_geom,
        gf=gf,
    )
    return target_x, target_y, target_z, yaw


def _estimate_object_height(stage: Any, object_prim_path: str, usd_geom: Any, fallback: float = OBJECT_SIZE) -> float:
    bbox = _compute_prim_bbox(stage, object_prim_path, usd_geom)
    if not bbox:
        return fallback
    mn, mx = bbox
    h = float(mx[2] - mn[2])
    if not np.isfinite(h) or h <= 0.0:
        return fallback
    if h > float(TRACKING_OBJECT_HEIGHT_MAX_M):
        return float(max(0.01, fallback))
    return max(h, 0.01)


def _query_object_pick_xy(
    stage: Any,
    object_prim_path: str,
    usd: Any,
    usd_geom: Any,
) -> tuple[float, float, str]:
    """Pick target XY from live object pose, with bbox center only when reasonable."""
    obj_pos, _ = _get_prim_world_pose(stage, object_prim_path, usd, usd_geom)
    bbox = _compute_prim_bbox(stage, object_prim_path, usd_geom)
    if bbox:
        mn, mx = bbox
        center = _bbox_center_if_reasonable(mn=mn, mx=mx, root_pos=obj_pos)
        if center is not None and center.size >= 3:
            cx = float(center[0])
            cy = float(center[1])
            if np.isfinite(cx) and np.isfinite(cy):
                return cx, cy, "bbox_center"

    rx = float(obj_pos[0]) if obj_pos.size >= 1 and np.isfinite(obj_pos[0]) else 0.0
    ry = float(obj_pos[1]) if obj_pos.size >= 2 and np.isfinite(obj_pos[1]) else 0.0
    return rx, ry, "root_pose"


def _ensure_object_within_workspace(
    stage: Any,
    object_prim_path: str,
    usd: Any,
    usd_geom: Any,
    gf: Any,
    table_x_range: tuple[float, float],
    table_y_range: tuple[float, float],
    table_top_z: float,
    work_center: tuple[float, float],
) -> Optional[dict[str, float]]:
    """If object is clearly out-of-bounds, recenter it on the tabletop."""
    raw_x, raw_y, _ = _query_object_pick_xy(stage, object_prim_path, usd, usd_geom)
    bbox = _compute_prim_bbox(stage, object_prim_path, usd_geom)
    raw_bottom_z = float(bbox[0][2]) if bbox else float("nan")
    raw_top_z = float(bbox[1][2]) if bbox else float("nan")
    margin = 0.15
    in_xy = (
        np.isfinite(raw_x)
        and np.isfinite(raw_y)
        and (table_x_range[0] - margin) <= raw_x <= (table_x_range[1] + margin)
        and (table_y_range[0] - margin) <= raw_y <= (table_y_range[1] + margin)
    )
    # Z guard: objects can fall below/inside table while keeping XY inside table footprint.
    # When that happens collect appears as "arm flailing with no mug in view".
    z_ok = (
        np.isfinite(raw_bottom_z)
        and np.isfinite(raw_top_z)
        and (raw_bottom_z >= float(table_top_z) - 0.02)
        and (raw_bottom_z <= float(table_top_z) + 0.30)
    )
    if in_xy and z_ok:
        return None

    obj_height = _estimate_object_height(stage, object_prim_path, usd_geom, fallback=OBJECT_SIZE)
    target_x = float(work_center[0])
    target_y = float(work_center[1])
    target_z = float(table_top_z) + max(float(obj_height) * 0.5, 0.02) + 0.003
    _set_prim_world_position(
        stage=stage,
        prim_path=object_prim_path,
        position=np.array([target_x, target_y, target_z], dtype=np.float32),
        usd_geom=usd_geom,
        gf=gf,
    )
    return {
        "raw_x": float(raw_x),
        "raw_y": float(raw_y),
        "raw_bottom_z": float(raw_bottom_z) if np.isfinite(raw_bottom_z) else float("nan"),
        "raw_top_z": float(raw_top_z) if np.isfinite(raw_top_z) else float("nan"),
        "target_x": target_x,
        "target_y": target_y,
        "target_z": target_z,
        "reason_xy": 0.0 if in_xy else 1.0,
        "reason_z": 0.0 if z_ok else 1.0,
    }


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


def _resolve_finger_prim_paths(
    stage: Any,
    robot_prim_path: str,
    get_prim_at_path: Any,
) -> tuple[str | None, str | None]:
    left_candidates = [
        f"{robot_prim_path}/panda_leftfinger",
        f"{robot_prim_path}/leftfinger",
        f"{robot_prim_path}/left_finger",
    ]
    right_candidates = [
        f"{robot_prim_path}/panda_rightfinger",
        f"{robot_prim_path}/rightfinger",
        f"{robot_prim_path}/right_finger",
    ]

    def _first_valid(paths: Sequence[str]) -> str | None:
        for p in paths:
            prim = get_prim_at_path(p)
            if prim and prim.IsValid():
                return p
        return None

    left = _first_valid(left_candidates)
    right = _first_valid(right_candidates)
    if left and right:
        return left, right

    # Fallback: discover from names under robot hierarchy.
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        path = prim.GetPath().pathString
        if not path.startswith(robot_prim_path):
            continue
        low = path.lower()
        if left is None and ("leftfinger" in low or "left_finger" in low):
            left = path
        if right is None and ("rightfinger" in low or "right_finger" in low):
            right = path
        if left and right:
            break
    return left, right


def _compute_tip_mid_offset_in_hand(
    stage: Any,
    eef_prim_path: str,
    finger_left_prim_path: str | None,
    finger_right_prim_path: str | None,
    get_prim_at_path: Any,
    usd: Any,
    usd_geom: Any,
) -> np.ndarray | None:
    """Measure fingertip-midpoint offset expressed in the current hand frame."""
    if not eef_prim_path or not finger_left_prim_path or not finger_right_prim_path:
        return None
    hand_pos, hand_quat = _get_eef_pose(stage, eef_prim_path, get_prim_at_path, usd, usd_geom)
    left_pos, _ = _get_prim_world_pose(stage, finger_left_prim_path, usd, usd_geom)
    right_pos, _ = _get_prim_world_pose(stage, finger_right_prim_path, usd, usd_geom)
    if not (
        np.all(np.isfinite(hand_pos[:3]))
        and np.all(np.isfinite(hand_quat[:4]))
        and np.all(np.isfinite(left_pos[:3]))
        and np.all(np.isfinite(right_pos[:3]))
    ):
        return None
    tip_mid = 0.5 * (left_pos[:3] + right_pos[:3])
    offset_world = (tip_mid - hand_pos[:3]).astype(np.float32)
    rot_hand = _quat_to_rot_wxyz(hand_quat)
    offset_hand = rot_hand.T @ offset_world
    if not np.all(np.isfinite(offset_hand)):
        return None
    if float(np.linalg.norm(offset_hand)) < 1e-6:
        return None
    return offset_hand.astype(np.float32)


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
    joint_vel = np.nan_to_num(joint_vel_padded[:7], nan=0.0, posinf=0.0, neginf=0.0)
    joint_vel = np.clip(joint_vel, -STATE_JOINT_VEL_CLIP_RAD_S, STATE_JOINT_VEL_CLIP_RAD_S)
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


def _step_toward_joint_targets(
    franka: Any,
    arm_target: np.ndarray,
    gripper_target: float,
    max_arm_step: float = MAX_ARM_STEP_RAD,
    max_gripper_step: float = MAX_GRIPPER_STEP,
) -> tuple[np.ndarray, float]:
    """Rate-limit joint command updates to avoid teleport-like oscillation."""
    def _shortest_angular_delta(target: np.ndarray, current: np.ndarray) -> np.ndarray:
        # Keep revolute motion on shortest path to avoid large wraps / knotting.
        return ((target - current + np.pi) % (2.0 * np.pi)) - np.pi

    current = _to_numpy(franka.get_joint_positions())
    if current.size < 7:
        return (
            np.clip(arm_target.astype(np.float32), FRANKA_ARM_LOWER, FRANKA_ARM_UPPER),
            float(gripper_target),
        )

    arm_current = current[:7].astype(np.float32, copy=False)
    target = np.clip(arm_target[:7].astype(np.float32, copy=False), FRANKA_ARM_LOWER, FRANKA_ARM_UPPER)
    arm_delta = _shortest_angular_delta(target, arm_current)
    arm_delta = np.clip(arm_delta, -float(max_arm_step), float(max_arm_step))
    arm_cmd = np.clip(arm_current + arm_delta, FRANKA_ARM_LOWER, FRANKA_ARM_UPPER).astype(np.float32)

    if current.size >= 9:
        gr_cur = float(0.5 * (current[7] + current[8]))
    elif current.size == 8:
        gr_cur = float(current[7])
    else:
        gr_cur = float(gripper_target)
    gr_delta = float(np.clip(float(gripper_target) - gr_cur, -float(max_gripper_step), float(max_gripper_step)))
    gr_cmd = float(gr_cur + gr_delta)

    return arm_cmd, gr_cmd


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


def _remove_prim_if_exists(stage: Any, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)

def _pos_to_joint_offset(
    target_x: float,
    target_y: float,
    center_x: float = 0.5,
    center_y: float = 0.0,
) -> np.ndarray:
    dx = target_x - center_x
    dy = target_y - center_y
    j0_offset = float(np.clip(np.arctan2(dy, 0.5) * 0.8, -0.35, 0.35))
    j1_offset = float(np.clip(-dx * 0.5, -0.25, 0.25))
    return np.array([j0_offset, j1_offset, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _make_pick_place_waypoints(
    pick_pos: tuple[float, float],
    place_pos: tuple[float, float],
    rng: np.random.Generator,
    work_center: tuple[float, float],
    home_arm: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray, float]]:
    def add_noise(base: np.ndarray) -> np.ndarray:
        noise = rng.uniform(-WAYPOINT_NOISE_RAD, WAYPOINT_NOISE_RAD, size=base.shape).astype(np.float32)
        return (base + noise).astype(np.float32)

    pick_offset = _pos_to_joint_offset(pick_pos[0], pick_pos[1], center_x=work_center[0], center_y=work_center[1])
    place_offset = _pos_to_joint_offset(place_pos[0], place_pos[1], center_x=work_center[0], center_y=work_center[1])

    approach_pick = add_noise(APPROACH_BASE + pick_offset)
    down_pick = add_noise(DOWN_BASE + pick_offset)
    move_place = add_noise(APPROACH_BASE + place_offset)
    down_place = add_noise(DOWN_BASE + place_offset)

    home = _pad_or_trim(_to_numpy(home_arm), 7) if home_arm is not None else HOME.copy()

    return [
        ("HOME", home.copy(), GRIPPER_OPEN),
        ("APPROACH_PICK", approach_pick, GRIPPER_OPEN),
        ("DOWN_PICK", down_pick, GRIPPER_OPEN),
        ("CLOSE", down_pick, GRIPPER_CLOSED),
        ("LIFT", approach_pick, GRIPPER_CLOSED),
        ("MOVE_TO_PLACE", move_place, GRIPPER_CLOSED),
        ("DOWN_PLACE", down_place, GRIPPER_CLOSED),
        ("OPEN", down_place, GRIPPER_OPEN),
        ("RETRACT", move_place, GRIPPER_OPEN),
        ("HOME_END", home.copy(), GRIPPER_OPEN),
    ]


def _make_pick_place_waypoints_ik(
    ik_solver: Any,
    pick_pos: tuple[float, float],
    place_pos: tuple[float, float],
    pick_grasp_z: float,
    pick_approach_z: float,
    place_grasp_z: float,
    place_approach_z: float,
    table_top_z: float,
    rng: np.random.Generator,
    target_orientation: np.ndarray | None = None,
    home_arm: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray, float]] | None:
    if ik_solver is None:
        return None

    def _shortest_angular_delta(target: np.ndarray, current: np.ndarray) -> np.ndarray:
        return ((target - current + np.pi) % (2.0 * np.pi)) - np.pi

    pick_grasp_xyz = np.array([pick_pos[0], pick_pos[1], float(pick_grasp_z)], dtype=np.float32)
    pick_pregrasp_xyz = np.array([pick_pos[0], pick_pos[1], float(pick_approach_z)], dtype=np.float32)
    if target_orientation is not None:
        try:
            q = _normalize_quat_wxyz(target_orientation)
            rot = _quat_to_rot_wxyz(q)
            grasp_dir = rot[:, 2].astype(np.float32, copy=False)
            norm = float(np.linalg.norm(grasp_dir))
            if np.isfinite(norm) and norm > 1e-6:
                grasp_dir = grasp_dir / norm
                pick_pregrasp_xyz = pick_grasp_xyz - float(IK_PRE_GRASP_OFFSET) * grasp_dir
                min_z = float(table_top_z) + float(IK_PRE_GRASP_MIN_ABOVE_TABLE)
                pick_pregrasp_xyz[2] = float(max(float(pick_pregrasp_xyz[2]), min_z))
        except Exception:
            pass

    pose_targets = {
        "APPROACH_PICK": np.array([pick_pos[0], pick_pos[1], float(pick_approach_z)], dtype=np.float32),
        "PRE_GRASP": pick_pregrasp_xyz.astype(np.float32, copy=False),
        "DOWN_PICK": np.array([pick_pos[0], pick_pos[1], float(pick_grasp_z)], dtype=np.float32),
        "MOVE_TO_PLACE": np.array([place_pos[0], place_pos[1], float(place_approach_z)], dtype=np.float32),
        "DOWN_PLACE": np.array([place_pos[0], place_pos[1], float(place_grasp_z)], dtype=np.float32),
    }

    solved: dict[str, np.ndarray] = {}
    prev_q: np.ndarray | None = None
    for name in ("APPROACH_PICK", "PRE_GRASP", "DOWN_PICK", "MOVE_TO_PLACE", "DOWN_PLACE"):
        q = _solve_ik_arm_target(ik_solver, pose_targets[name], target_orientation=target_orientation)
        if q is None and target_orientation is not None:
            # Orientation constraints improve top-down behavior but can fail near singularities.
            q = _solve_ik_arm_target(ik_solver, pose_targets[name], target_orientation=None)
            if q is not None:
                LOG.warning("IK waypoint '%s' fallback to unconstrained orientation", name)
        if q is None:
            LOG.warning("IK waypoint '%s' failed, falling back to joint-offset planner", name)
            return None
        if prev_q is not None:
            jump = _shortest_angular_delta(q, prev_q)
            max_jump = float(np.max(np.abs(jump)))
            if max_jump > MAX_IK_WAYPOINT_JUMP_RAD:
                LOG.warning(
                    "IK waypoint '%s' jump %.3f rad too large; clamping for continuity",
                    name,
                    max_jump,
                )
                q = prev_q + np.clip(jump, -MAX_IK_WAYPOINT_JUMP_RAD, MAX_IK_WAYPOINT_JUMP_RAD)
                q = np.clip(q, FRANKA_ARM_LOWER, FRANKA_ARM_UPPER).astype(np.float32, copy=False)
        noise = rng.uniform(-0.01, 0.01, size=q.shape).astype(np.float32)
        solved[name] = np.clip(q + noise, FRANKA_ARM_LOWER, FRANKA_ARM_UPPER).astype(np.float32)
        prev_q = solved[name]

    home = _pad_or_trim(_to_numpy(home_arm), 7) if home_arm is not None else HOME.copy()

    return [
        ("HOME", home.copy(), GRIPPER_OPEN),
        ("APPROACH_PICK", solved["APPROACH_PICK"], GRIPPER_OPEN),
        ("PRE_GRASP", solved["PRE_GRASP"], GRIPPER_OPEN),
        ("DOWN_PICK", solved["DOWN_PICK"], GRIPPER_OPEN),
        ("CLOSE", solved["DOWN_PICK"], GRIPPER_CLOSED),
        ("LIFT", solved["APPROACH_PICK"], GRIPPER_CLOSED),
        ("MOVE_TO_PLACE", solved["MOVE_TO_PLACE"], GRIPPER_CLOSED),
        ("DOWN_PLACE", solved["DOWN_PLACE"], GRIPPER_CLOSED),
        ("OPEN", solved["DOWN_PLACE"], GRIPPER_OPEN),
        ("RETRACT", solved["MOVE_TO_PLACE"], GRIPPER_OPEN),
        ("HOME_END", home.copy(), GRIPPER_OPEN),
    ]


def _compute_pick_place_z_targets(
    stage: Any,
    object_prim_path: str,
    usd_geom: Any,
    table_top_z: float,
    fallback_object_height: float,
) -> dict[str, float]:
    bbox = _compute_prim_bbox(stage, object_prim_path, usd_geom)
    if not bbox:
        obj_h = max(float(fallback_object_height), 0.01)
        pick_grasp_z = float(table_top_z + max(IK_GRASP_Z_OFFSET, obj_h * 0.6))
        pick_approach_z = float(pick_grasp_z + IK_APPROACH_Z_OFFSET)
        place_grasp_z = float(table_top_z + max(IK_PLACE_MIN_CLEARANCE_FROM_TABLE + 0.5 * obj_h, 0.04))
        place_approach_z = float(place_grasp_z + IK_PLACE_APPROACH_EXTRA_Z)
        return {
            "object_height": obj_h,
            "pick_grasp_z": pick_grasp_z,
            "pick_approach_z": pick_approach_z,
            "place_grasp_z": place_grasp_z,
            "place_approach_z": place_approach_z,
        }

    mn, mx = bbox
    obj_h = max(float(mx[2] - mn[2]), 0.01)
    obj_top_z = float(mx[2])
    obj_bottom_z = float(mn[2])
    pick_grasp_z = float(obj_bottom_z + IK_PICK_GRASP_HEIGHT_RATIO * obj_h)
    pick_grasp_z = max(pick_grasp_z, float(table_top_z) + IK_PICK_MIN_CLEARANCE_FROM_TABLE)
    pick_grasp_z = min(pick_grasp_z, obj_top_z - 0.004)
    if pick_grasp_z <= float(table_top_z) + IK_PICK_MIN_CLEARANCE_FROM_TABLE:
        pick_grasp_z = float(table_top_z) + IK_PICK_MIN_CLEARANCE_FROM_TABLE + 0.005

    pick_approach_z = max(obj_top_z + IK_PICK_APPROACH_EXTRA_Z, pick_grasp_z + 0.08)
    place_grasp_z = max(float(table_top_z) + IK_PLACE_MIN_CLEARANCE_FROM_TABLE + 0.5 * obj_h, float(table_top_z) + 0.04)
    place_approach_z = place_grasp_z + IK_PLACE_APPROACH_EXTRA_Z

    return {
        "object_height": obj_h,
        "pick_grasp_z": float(pick_grasp_z),
        "pick_approach_z": float(pick_approach_z),
        "place_grasp_z": float(place_grasp_z),
        "place_approach_z": float(place_approach_z),
    }


def _sample_place_from_pick(
    rng: np.random.Generator,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    pick_pos: tuple[float, float],
) -> tuple[float, float]:
    place_x = float(pick_pos[0])
    place_y = float(pick_pos[1])
    for _ in range(100):
        place_x = float(rng.uniform(*x_range))
        place_y = float(rng.uniform(*y_range))
        dist = float(np.linalg.norm(np.array([place_x - pick_pos[0], place_y - pick_pos[1]], dtype=np.float32)))
        if dist >= MIN_PICK_PLACE_DIST:
            break

    return place_x, place_y


def _compute_grasp_metrics(
    franka: Any,
    stage: Any,
    eef_prim_path: str,
    get_prim_at_path: Any,
    usd: Any,
    usd_geom: Any,
    object_prim_path: str,
    table_top_z: float,
    object_height: float,
    finger_left_prim_path: str | None = None,
    finger_right_prim_path: str | None = None,
) -> dict[str, float]:
    joints = _to_numpy(franka.get_joint_positions())
    if joints.size >= 9:
        gripper_width = float(joints[7] + joints[8])
    elif joints.size == 8:
        gripper_width = float(2.0 * joints[7])
    else:
        gripper_width = 0.0

    obj_pos, _ = _get_object_tracking_pose(stage, object_prim_path, usd, usd_geom)
    eef_pos, _ = _get_eef_pose(stage, eef_prim_path, get_prim_at_path, usd, usd_geom)
    delta = obj_pos - eef_pos
    obj_eef_dist = float(np.linalg.norm(delta))
    obj_eef_xy_dist = float(np.linalg.norm(delta[:2]))
    tip_mid_metrics: dict[str, float] = {}
    if finger_left_prim_path and finger_right_prim_path:
        left_pos, _ = _get_prim_world_pose(stage, finger_left_prim_path, usd, usd_geom)
        right_pos, _ = _get_prim_world_pose(stage, finger_right_prim_path, usd, usd_geom)
        if np.all(np.isfinite(left_pos[:3])) and np.all(np.isfinite(right_pos[:3])):
            tip_mid = 0.5 * (left_pos[:3] + right_pos[:3])
            tip_delta = obj_pos - tip_mid
            tip_mid_metrics = {
                "object_tip_mid_distance": float(np.linalg.norm(tip_delta)),
                "object_tip_mid_xy_distance": float(np.linalg.norm(tip_delta[:2])),
                "object_tip_mid_z_delta": float(tip_delta[2]),
            }
    lift_height = float(obj_pos[2] - (float(table_top_z) + 0.5 * float(object_height)))
    out = {
        "gripper_width": gripper_width,
        "object_eef_distance": obj_eef_dist,
        "object_eef_xy_distance": obj_eef_xy_dist,
        "lift_height": lift_height,
    }
    out.update(tip_mid_metrics)
    return out


def _verify_after_close(metrics: dict[str, float]) -> bool:
    return bool(
        metrics.get("gripper_width", 0.0) >= VERIFY_CLOSE_MIN_GRIPPER_WIDTH
        and metrics.get("object_eef_distance", 1e9) <= VERIFY_CLOSE_MAX_OBJECT_EEF_DISTANCE
        and metrics.get("object_eef_xy_distance", 1e9) <= VERIFY_CLOSE_MAX_OBJECT_EEF_XY_DISTANCE
    )


def _verify_after_retrieval(metrics: dict[str, float]) -> bool:
    return bool(
        metrics.get("gripper_width", 0.0) >= VERIFY_CLOSE_MIN_GRIPPER_WIDTH
        and metrics.get("lift_height", -1e9) >= VERIFY_RETRIEVAL_MIN_LIFT
        and metrics.get("object_eef_distance", 1e9) <= VERIFY_RETRIEVAL_MAX_OBJECT_EEF_DISTANCE
    )


def _verify_reach_before_close(metrics: dict[str, float]) -> bool:
    eef_ok = bool(
        metrics.get("object_eef_distance", 1e9) <= REACH_BEFORE_CLOSE_MAX_OBJECT_EEF_DISTANCE
        and metrics.get("object_eef_xy_distance", 1e9) <= REACH_BEFORE_CLOSE_MAX_OBJECT_EEF_XY_DISTANCE
    )
    tip_dist = float(metrics.get("object_tip_mid_distance", np.nan))
    tip_xy = float(metrics.get("object_tip_mid_xy_distance", np.nan))
    tip_dz = float(metrics.get("object_tip_mid_z_delta", np.nan))
    tip_available = np.isfinite(tip_dist) and np.isfinite(tip_xy) and np.isfinite(tip_dz)
    if not tip_available:
        return eef_ok
    tip_ok = bool(
        tip_dist <= REACH_BEFORE_CLOSE_MAX_OBJECT_TIP_MID_DISTANCE
        and tip_xy <= REACH_BEFORE_CLOSE_MAX_OBJECT_TIP_MID_XY_DISTANCE
        and abs(tip_dz) <= REACH_BEFORE_CLOSE_MAX_ABS_TIP_MID_Z_DELTA
    )
    if REACH_REQUIRE_TIP_MID:
        return tip_ok
    return bool(eef_ok or tip_ok)


def _run_pick_place_episode(
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
    object_prim_path: str,
    ik_solver: Any,
    table_x_range: tuple[float, float],
    table_y_range: tuple[float, float],
    table_top_z: float,
    work_center: tuple[float, float],
    gf: Any,
    stop_event: threading.Event | None = None,
    episode_timeout_sec: float = DEFAULT_EPISODE_TIMEOUT_SEC,
    finger_left_prim_path: str | None = None,
    finger_right_prim_path: str | None = None,
) -> tuple[int, bool, bool]:
    world.reset()
    for _ in range(10):
        world.step(render=True)

    for cam_obj in cameras.values():
        if hasattr(cam_obj, "initialize"):
            try:
                cam_obj.initialize()
            except Exception as exc:
                LOG.warning("Camera initialize() failed after reset: %s", exc)
    _apply_wrist_camera_pose(
        camera_wrist=cameras.get("cam_wrist"),
        stage=stage,
        usd_geom=usd_geom,
        gf=gf,
        wrist_cam_path=getattr(cameras.get("cam_wrist"), "prim_path", None),
    )
    current_grasp_target: dict[str, Any] | None = None

    def _current_arm_target() -> np.ndarray:
        joints = _to_numpy(franka.get_joint_positions())
        return _pad_or_trim(joints[:7], 7)

    def _current_gripper_target() -> float:
        joints = _to_numpy(franka.get_joint_positions())
        if joints.size >= 9:
            return float(0.5 * (joints[7] + joints[8]))
        if joints.size == 8:
            return float(joints[7])
        return float(GRIPPER_OPEN)

    def _record_frame(arm_target: Optional[np.ndarray] = None, gripper_target: Optional[float] = None) -> None:
        nonlocal frame_index
        arm_cmd = _current_arm_target() if arm_target is None else _pad_or_trim(_to_numpy(arm_target), 7)
        grip_cmd = _current_gripper_target() if gripper_target is None else float(gripper_target)
        state = _extract_state_vector(franka, stage, eef_prim_path, get_prim_at_path, usd, usd_geom)
        action = np.concatenate([arm_cmd, np.array([grip_cmd], dtype=np.float32)], axis=0)
        action = _pad_or_trim(action, ACTION_DIM)
        frame_extras = _build_frame_extras(
            stage=stage,
            object_prim_path=object_prim_path,
            usd=usd,
            usd_geom=usd_geom,
            current_target=current_grasp_target,
        )
        writer.add_frame(
            episode_index=episode_index,
            frame_index=frame_index,
            observation_state=state,
            action=action,
            timestamp=frame_index / float(fps),
            next_done=False,
            extras=frame_extras,
        )
        for cam_name, cam_obj in cameras.items():
            rgb = _capture_rgb(cam_obj, CAMERA_RESOLUTION)
            writer.add_video_frame(cam_name, rgb)
        frame_index += 1

    frame_index = 0
    stopped = False
    place_pos: tuple[float, float] | None = None
    last_pick_pos: tuple[float, float] = (float(work_center[0]), float(work_center[1]))
    grasp_succeeded = False
    attempt_used = 0
    ik_target_orientation: np.ndarray | None = None
    episode_started = time.monotonic()
    timeout_sec = max(0.0, float(episode_timeout_sec))
    object_tokens = set(_object_token_candidates(object_prim_path))
    is_mug_target = "mug" in object_tokens
    require_mug_annotation = _read_bool_env("COLLECT_REQUIRE_MUG_ANNOTATION", True)
    enforce_mug_annotation_target = _read_bool_env("COLLECT_ENFORCE_MUG_ANNOTATION_TARGET", True)

    annotation_path = _resolve_grasp_annotation_path(stage=stage, object_prim_path=object_prim_path)
    annotation_candidates: list[dict[str, Any]] = []
    annotation_meta: dict[str, Any] = {}
    annotation_local_pos_scale = 1.0
    if annotation_path is not None:
        annotation_candidates, annotation_meta = _load_grasp_pose_candidates(annotation_path)
        if annotation_candidates:
            loaded_annotation_candidates = list(annotation_candidates)
            hint_scale = float(annotation_meta.get("position_scale_hint", 1.0))
            infer_scale = _infer_annotation_position_scale(
                stage=stage,
                object_prim_path=object_prim_path,
                candidates=annotation_candidates,
                usd=usd,
                usd_geom=usd_geom,
                preset_pos_scale=hint_scale,
            )
            annotation_local_pos_scale = float(hint_scale * infer_scale)
            annotation_candidates, sanitize_report = _sanitize_annotation_candidates(
                stage=stage,
                object_prim_path=object_prim_path,
                candidates=annotation_candidates,
                usd=usd,
                usd_geom=usd_geom,
                local_pos_scale=annotation_local_pos_scale,
            )
            if sanitize_report.get("rejected_file", False):
                LOG.warning(
                    "collect: reject grasp annotation %s for %s by strict validation report=%s",
                    annotation_path,
                    object_prim_path,
                    sanitize_report,
                )
            elif int(sanitize_report.get("kept", 0)) != int(sanitize_report.get("total", 0)):
                LOG.info(
                    "collect: sanitize grasp annotation %s for %s report=%s",
                    annotation_path,
                    object_prim_path,
                    sanitize_report,
                )
            allow_unsanitized_fallback = _read_bool_env("COLLECT_ALLOW_UNSANITIZED_ANNOTATION_FALLBACK", True)
            force_unsanitized_for_required_mug = bool(
                require_mug_annotation and is_mug_target and loaded_annotation_candidates
            )
            if not annotation_candidates and (allow_unsanitized_fallback or force_unsanitized_for_required_mug):
                annotation_candidates = loaded_annotation_candidates
                annotation_local_pos_scale = float(hint_scale)
                fallback_reason = (
                    "required_mug_annotation" if force_unsanitized_for_required_mug else "allow_unsanitized_env"
                )
                LOG.warning(
                    "collect: sanitize dropped all candidates; fallback to unsanitized annotation set "
                    "(count=%d, path=%s, local_pos_scale=%.5f, reason=%s)",
                    len(annotation_candidates),
                    annotation_path,
                    float(annotation_local_pos_scale),
                    fallback_reason,
                )
            if annotation_candidates:
                LOG.info(
                    "collect: loaded %d grasp annotations for %s from %s "
                    "(quat_order=%s, hint_scale=%.5f, infer_scale=%.5f, local_pos_scale=%.5f)",
                    len(annotation_candidates),
                    object_prim_path,
                    annotation_path,
                    str(annotation_meta.get("quat_order", "wxyz")),
                    float(hint_scale),
                    float(infer_scale),
                    float(annotation_local_pos_scale),
                )
            else:
                LOG.warning(
                    "collect: grasp annotation disabled after validation: %s for %s",
                    annotation_path,
                    object_prim_path,
                )
        else:
            LOG.warning("collect: grasp annotation file had no usable poses: %s", annotation_path)
    if require_mug_annotation and is_mug_target and not annotation_candidates:
        for mug_fallback_path in _iter_mug_annotation_candidate_paths():
            if not mug_fallback_path.exists():
                continue
            try:
                if annotation_path is not None and mug_fallback_path.resolve() == annotation_path.resolve():
                    continue
            except Exception:
                if annotation_path is not None and str(mug_fallback_path) == str(annotation_path):
                    continue
            fallback_candidates, fallback_meta = _load_grasp_pose_candidates(mug_fallback_path)
            if not fallback_candidates:
                continue
            annotation_path = mug_fallback_path
            annotation_meta = fallback_meta
            annotation_candidates = list(fallback_candidates)
            annotation_local_pos_scale = float(annotation_meta.get("position_scale_hint", 1.0))
            LOG.warning(
                "collect: recovered mug grasp annotation from fallback path=%s count=%d local_pos_scale=%.5f",
                mug_fallback_path,
                len(annotation_candidates),
                float(annotation_local_pos_scale),
            )
            break
    if annotation_path is not None and annotation_meta:
        if str(annotation_meta.get("quat_order", "wxyz")) == "xyzw":
            LOG.warning(
                "collect: converted annotation quat order xyzw->wxyz for %s",
                annotation_path,
            )
    annotation_failed_pose_cache_key = _annotation_failed_pose_cache_key(
        object_prim_path=object_prim_path,
        annotation_path=annotation_path,
    )
    existing_failed_ids = sorted(list(_get_failed_annotation_pose_ids(annotation_failed_pose_cache_key)))
    if existing_failed_ids:
        LOG.info(
            "collect: carry failed annotation pose cache for %s -> %s",
            object_prim_path,
            existing_failed_ids,
        )
    no_annotation_available = len(annotation_candidates) == 0
    if require_mug_annotation and is_mug_target and no_annotation_available:
        raise RuntimeError(
            "Mug grasp annotation required but none found. "
            "Expected grasp_poses/mug_grasp_pose.json (or set COLLECT_GRASP_POSE_PATH)."
        )
    force_topdown_grasp = _read_bool_env("COLLECT_FORCE_TOPDOWN_GRASP", False)
    if force_topdown_grasp:
        LOG.warning(
            "collect: COLLECT_FORCE_TOPDOWN_GRASP enabled; bypassing annotation orientation and using top-down grasp"
        )
    annotation_tip_mid_offset_hand = _compute_tip_mid_offset_in_hand(
        stage=stage,
        eef_prim_path=eef_prim_path,
        finger_left_prim_path=finger_left_prim_path,
        finger_right_prim_path=finger_right_prim_path,
        get_prim_at_path=get_prim_at_path,
        usd=usd,
        usd_geom=usd_geom,
    )
    if ANNOTATION_POSE_IS_TIP_MID_FRAME:
        if annotation_tip_mid_offset_hand is not None:
            LOG.info(
                "collect: annotation tip-mid->hand enabled, offset_hand=(%.4f, %.4f, %.4f), norm=%.4f",
                float(annotation_tip_mid_offset_hand[0]),
                float(annotation_tip_mid_offset_hand[1]),
                float(annotation_tip_mid_offset_hand[2]),
                float(np.linalg.norm(annotation_tip_mid_offset_hand)),
            )
        else:
            LOG.warning(
                "collect: annotation tip-mid->hand enabled but finger offset unavailable; using raw annotation poses"
            )

    def _timeout_triggered() -> bool:
        nonlocal stopped
        if timeout_sec <= 0.0:
            return False
        if (time.monotonic() - episode_started) < timeout_sec:
            return False
        LOG.error(
            "collect: episode %d exceeded timeout %.1fs; requesting stop",
            episode_index + 1,
            timeout_sec,
        )
        if stop_event is not None:
            stop_event.set()
        stopped = True
        return True

    def _execute_transitions(
        transitions: list[tuple[Any, Any]],
        live_target_z_by_waypoint: Optional[dict[str, float]] = None,
        enable_close_hold: bool = True,
    ) -> None:
        nonlocal frame_index, stopped, current_grasp_target
        close_anchor_arm: np.ndarray | None = None

        def _resolve_live_arm_target(waypoint_name: str, fallback_arm: np.ndarray) -> np.ndarray:
            if not live_target_z_by_waypoint or waypoint_name not in live_target_z_by_waypoint:
                return fallback_arm
            if ik_solver is None:
                return fallback_arm
            live_pos, live_quat = _compute_live_grasp_target_pose(
                stage=stage,
                object_prim_path=object_prim_path,
                usd=usd,
                usd_geom=usd_geom,
                current_target=current_grasp_target,
                z_override=float(live_target_z_by_waypoint[waypoint_name]),
            )
            # Keep live target inside current workspace envelope.
            live_pos = live_pos.copy()
            live_pos[0] = float(np.clip(live_pos[0], table_x_range[0], table_x_range[1]))
            live_pos[1] = float(np.clip(live_pos[1], table_y_range[0], table_y_range[1]))
            current_target_orientation = ik_target_orientation if ik_target_orientation is not None else live_quat
            q = _solve_ik_arm_target(
                ik_solver,
                target_position=live_pos,
                target_orientation=current_target_orientation,
            )
            if q is None and current_target_orientation is not None:
                q = _solve_ik_arm_target(
                    ik_solver,
                    target_position=live_pos,
                    target_orientation=None,
                )
            if q is None:
                return fallback_arm

            # Keep fallback targets synchronized with live world target for logging/action extras.
            if current_grasp_target is not None and current_grasp_target.get("source") != "annotation":
                current_grasp_target["target_pos"] = live_pos.astype(np.float32, copy=False)
                current_grasp_target["target_quat"] = _normalize_quat_wxyz(current_target_orientation)
                current_grasp_target["valid"] = True
            return q.astype(np.float32, copy=False)

        for start_wp, end_wp in transitions:
            _, start_arm, start_gripper = start_wp
            end_name, end_arm, end_gripper = end_wp
            early_reach_halt = False
            resolved_end_arm = (
                _resolve_live_arm_target(end_name, end_arm)
                if live_target_z_by_waypoint and end_name in live_target_z_by_waypoint
                else end_arm
            )
            if end_name == "CLOSE" and close_anchor_arm is not None:
                resolved_end_arm = close_anchor_arm.copy()
            segment_steps = 1 if end_name in {"CLOSE", "OPEN"} else steps_per_segment

            for step in range(segment_steps):
                if _timeout_triggered():
                    return
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    return

                alpha = float(step + 1) / float(segment_steps)
                # Lock to a single planned pose for this transition (no per-frame mesh chasing).
                if end_name == "CLOSE" and close_anchor_arm is not None:
                    # If DOWN_PICK halted early, close at the current stabilized arm pose.
                    arm_target = resolved_end_arm.astype(np.float32, copy=False)
                else:
                    arm_target = ((1.0 - alpha) * start_arm + alpha * resolved_end_arm).astype(np.float32)

                # Keep arm/gripper phases decoupled (MagicSim-style): only close/open in explicit hold loops.
                if end_name in {"CLOSE", "OPEN"}:
                    gripper_target = float(start_gripper)
                else:
                    gripper_target = float((1.0 - alpha) * start_gripper + alpha * end_gripper)

                arm_cmd, gr_cmd = _step_toward_joint_targets(franka, arm_target, gripper_target)
                _set_joint_targets(franka, arm_cmd, gr_cmd, physics_control=True)
                world.step(render=True)
                _record_frame(arm_target=arm_target, gripper_target=gripper_target)

                # Borrowed from MagicSim grasp phase behavior:
                # if reach gate is already satisfied while descending, halt early and close.
                if end_name == "DOWN_PICK" and step >= 4:
                    reach_metrics = _compute_grasp_metrics(
                        franka=franka,
                        stage=stage,
                        eef_prim_path=eef_prim_path,
                        get_prim_at_path=get_prim_at_path,
                        usd=usd,
                        usd_geom=usd_geom,
                        object_prim_path=object_prim_path,
                        table_top_z=table_top_z,
                        object_height=max(float(object_height), 0.01),
                        finger_left_prim_path=finger_left_prim_path,
                        finger_right_prim_path=finger_right_prim_path,
                    )
                    if _verify_reach_before_close(reach_metrics):
                        close_anchor_arm = _current_arm_target().astype(np.float32, copy=False)
                        LOG.info(
                            "collect: attempt %d early halt before close at %s step=%d/%d metrics=%s close_anchor=%s",
                            attempt,
                            end_name,
                            step + 1,
                            segment_steps,
                            reach_metrics,
                            np.round(close_anchor_arm, 4).tolist(),
                        )
                        early_reach_halt = True
                        break

            if stopped:
                return
            if early_reach_halt:
                # Stop descending/chasing once we already reached the close gate.
                continue

            if end_name == "CLOSE" and enable_close_hold:
                close_arm = (
                    close_anchor_arm.astype(np.float32, copy=False)
                    if close_anchor_arm is not None
                    else _resolve_live_arm_target(end_name, end_arm)
                )
                for _ in range(20):
                    if _timeout_triggered():
                        return
                    arm_cmd, gr_cmd = _step_toward_joint_targets(franka, close_arm, end_gripper)
                    _set_joint_targets(franka, arm_cmd, gr_cmd, physics_control=True)
                    world.step(render=True)
                    _record_frame(arm_target=close_arm, gripper_target=end_gripper)
                close_anchor_arm = None

            if end_name == "OPEN":
                for _ in range(10):
                    if _timeout_triggered():
                        return
                    arm_cmd, gr_cmd = _step_toward_joint_targets(franka, end_arm, end_gripper)
                    _set_joint_targets(franka, arm_cmd, gr_cmd, physics_control=True)
                    world.step(render=True)
                    _record_frame(arm_target=end_arm, gripper_target=end_gripper)

    def _prepare_replan_from_current(lift_arm: np.ndarray) -> None:
        nonlocal stopped
        for _ in range(20):
            if _timeout_triggered():
                break
            arm_cmd, gr_cmd = _step_toward_joint_targets(franka, lift_arm, GRIPPER_OPEN)
            _set_joint_targets(franka, arm_cmd, gr_cmd, physics_control=True)
            world.step(render=True)
            _record_frame(arm_target=lift_arm, gripper_target=GRIPPER_OPEN)
        if stopped:
            return
        for _ in range(8):
            if _timeout_triggered():
                break
            cur_arm = _current_arm_target()
            arm_cmd, gr_cmd = _step_toward_joint_targets(franka, cur_arm, GRIPPER_OPEN)
            _set_joint_targets(franka, arm_cmd, gr_cmd, physics_control=True)
            world.step(render=True)
            _record_frame(arm_target=cur_arm, gripper_target=GRIPPER_OPEN)

    for attempt in range(1, GRASP_MAX_ATTEMPTS + 1):
        if _timeout_triggered():
            break
        attempt_used = attempt
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        annotation_target = _select_annotation_grasp_target(
            stage=stage,
            object_prim_path=object_prim_path,
            usd=usd,
            usd_geom=usd_geom,
            candidates=annotation_candidates,
            attempt_index=attempt,
            ik_solver=ik_solver,
            table_x_range=table_x_range,
            table_y_range=table_y_range,
            table_top_z=table_top_z,
            failed_pose_ids=_get_failed_annotation_pose_ids(annotation_failed_pose_cache_key),
            tip_mid_offset_in_hand=annotation_tip_mid_offset_hand,
            annotation_pose_is_tip_mid_frame=ANNOTATION_POSE_IS_TIP_MID_FRAME,
            local_pos_scale=annotation_local_pos_scale,
        )
        if force_topdown_grasp and annotation_target is not None:
            annotation_target = None
        if annotation_target is not None:
            raw_pick_x = float(annotation_target["target_pos"][0])
            raw_pick_y = float(annotation_target["target_pos"][1])
            pick_source = "annotation"
            current_grasp_target = annotation_target
            try:
                obj_pos_dbg, _ = _get_object_tracking_pose(stage, object_prim_path, usd, usd_geom)
                tgt_pos_dbg = _to_numpy(annotation_target.get("target_pos"), dtype=np.float32).reshape(-1)
                d = tgt_pos_dbg[:3] - obj_pos_dbg[:3]
                LOG.info(
                    "collect: attempt %d annotation_pick idx=%s group=%s label=%s "
                    "converted=%s ik_feasible=%s relaxed=%s target=(%.4f, %.4f, %.4f) "
                    "obj_delta=(%.4f, %.4f, %.4f)",
                    attempt,
                    str(annotation_target.get("index", "na")),
                    str(annotation_target.get("candidate_group", "body")),
                    str(annotation_target.get("label", "body")),
                    bool(annotation_target.get("annotation_tip_mid_to_hand_converted", False)),
                    bool(annotation_target.get("ik_feasible", False)),
                    bool(annotation_target.get("ik_orientation_relaxed", False)),
                    float(tgt_pos_dbg[0]),
                    float(tgt_pos_dbg[1]),
                    float(tgt_pos_dbg[2]),
                    float(d[0]),
                    float(d[1]),
                    float(d[2]),
                )
            except Exception:
                pass
        else:
            if is_mug_target and enforce_mug_annotation_target:
                raise RuntimeError(
                    "Mug grasp annotation target selection failed; refusing fallback to bbox/root pose."
                )
            raw_pick_x, raw_pick_y, pick_source = _query_object_pick_xy(
                stage=stage,
                object_prim_path=object_prim_path,
                usd=usd,
                usd_geom=usd_geom,
            )
            current_grasp_target = {
                "target_pos": np.array([raw_pick_x, raw_pick_y, float(table_top_z)], dtype=np.float32),
                "target_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "source": pick_source,
                "source_id": GRASP_POSE_SOURCE_IDS.get(pick_source, GRASP_POSE_SOURCE_IDS["none"]),
                "valid": pick_source in {"bbox_center", "root_pose"},
            }
        pick_x = float(np.clip(raw_pick_x, table_x_range[0], table_x_range[1]))
        pick_y = float(np.clip(raw_pick_y, table_y_range[0], table_y_range[1]))
        clamp_delta = float(np.linalg.norm(np.array([pick_x - raw_pick_x, pick_y - raw_pick_y], dtype=np.float32)))
        if clamp_delta > PICK_CLAMP_MAX_DELTA and np.isfinite(raw_pick_x) and np.isfinite(raw_pick_y):
            LOG.warning(
                "collect: attempt %d workspace mismatch (%s raw=(%.3f, %.3f), clamped=(%.3f, %.3f), delta=%.3f) -> using CLAMPED live target",
                attempt,
                pick_source,
                raw_pick_x,
                raw_pick_y,
                pick_x,
                pick_y,
                clamp_delta,
            )
            # Keep target inside robot-table reachable workspace for physically valid approach.
            if current_grasp_target is not None:
                if current_grasp_target.get("source") == "annotation":
                    if is_mug_target and enforce_mug_annotation_target:
                        raise RuntimeError(
                            "Mug annotation target projected outside workspace; refusing fallback to bbox target."
                        )
                    # Annotation can point to unreachable local handles; switch to live bbox target.
                    current_grasp_target = {
                        "target_pos": np.array([pick_x, pick_y, float(table_top_z)], dtype=np.float32),
                        "target_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        "source": "bbox_center",
                        "source_id": GRASP_POSE_SOURCE_IDS["bbox_center"],
                        "valid": True,
                    }
                else:
                    pos = _to_numpy(current_grasp_target.get("target_pos"), dtype=np.float32).reshape(-1)
                    if pos.size < 3:
                        pos = np.array([pick_x, pick_y, float(table_top_z)], dtype=np.float32)
                    else:
                        pos = pos.copy()
                        pos[0] = pick_x
                        pos[1] = pick_y
                    current_grasp_target["target_pos"] = pos.astype(np.float32, copy=False)
        last_pick_pos = (pick_x, pick_y)
        if abs(pick_x - raw_pick_x) > 1e-3 or abs(pick_y - raw_pick_y) > 1e-3:
            LOG.warning(
                "collect: attempt %d pick pose clamped (%s): raw=(%.3f, %.3f), clamped=(%.3f, %.3f)",
                attempt,
                pick_source,
                raw_pick_x,
                raw_pick_y,
                pick_x,
                pick_y,
            )
        if place_pos is None:
            place_pos = _sample_place_from_pick(rng, table_x_range, table_y_range, last_pick_pos)
        object_height = _estimate_object_height(stage, object_prim_path, usd_geom, fallback=OBJECT_SIZE)
        z_targets = _compute_pick_place_z_targets(
            stage=stage,
            object_prim_path=object_prim_path,
            usd_geom=usd_geom,
            table_top_z=table_top_z,
            fallback_object_height=object_height,
        )
        object_height = float(z_targets["object_height"])
        if annotation_target is not None:
            ann_z = float(annotation_target["target_pos"][2])
            min_z = float(table_top_z) + IK_PICK_MIN_CLEARANCE_FROM_TABLE
            max_z = max(min_z + 0.01, float(z_targets["pick_approach_z"]) - 0.04)
            z_targets["pick_grasp_z"] = float(np.clip(ann_z, min_z, max_z))
            z_targets["pick_approach_z"] = max(
                float(z_targets["pick_approach_z"]),
                float(z_targets["pick_grasp_z"]) + 0.08,
            )
            ik_target_orientation = _normalize_quat_wxyz(annotation_target["target_quat"])
        elif force_topdown_grasp:
            ik_target_orientation = TOP_DOWN_FALLBACK_QUAT.copy()
            if current_grasp_target is not None:
                current_grasp_target["target_quat"] = ik_target_orientation.copy()
        elif no_annotation_available:
            ik_target_orientation = TOP_DOWN_FALLBACK_QUAT.copy()
            if current_grasp_target is not None:
                current_grasp_target["target_quat"] = ik_target_orientation.copy()
            LOG.info("collect: attempt %d using top-down fallback orientation (no annotation)", attempt)
        elif ik_target_orientation is None:
            _, cand_quat = _get_eef_pose(stage, eef_prim_path, get_prim_at_path, usd, usd_geom)
            cand_quat = np.asarray(cand_quat, dtype=np.float32).reshape(-1)
            if cand_quat.size >= 4 and np.all(np.isfinite(cand_quat[:4])):
                norm = float(np.linalg.norm(cand_quat[:4]))
                if np.isfinite(norm) and norm > 1e-6:
                    ik_target_orientation = (cand_quat[:4] / norm).astype(np.float32)

        if current_grasp_target is not None and current_grasp_target.get("source") != "annotation":
            pos = _to_numpy(current_grasp_target.get("target_pos"), dtype=np.float32).reshape(-1)
            if pos.size >= 3:
                pos = pos.copy()
                pos[2] = float(z_targets["pick_grasp_z"])
                current_grasp_target["target_pos"] = pos

        for _ in range(8):
            world.step(render=True)
            _record_frame()

        # First attempt replans from current arm; later retries bias back to canonical HOME to unwind knots.
        attempt_home_arm = _current_arm_target() if attempt == 1 else HOME.copy()
        waypoints = _make_pick_place_waypoints_ik(
            ik_solver=ik_solver,
            pick_pos=last_pick_pos,
            place_pos=place_pos,
            pick_grasp_z=float(z_targets["pick_grasp_z"]),
            pick_approach_z=float(z_targets["pick_approach_z"]),
            place_grasp_z=float(z_targets["place_grasp_z"]),
            place_approach_z=float(z_targets["place_approach_z"]),
            table_top_z=float(table_top_z),
            rng=rng,
            target_orientation=ik_target_orientation,
            home_arm=attempt_home_arm,
        )
        if waypoints is None:
            waypoints = _make_pick_place_waypoints(
                last_pick_pos,
                place_pos,
                rng,
                work_center=work_center,
                home_arm=attempt_home_arm,
            )

        names = [wp[0] for wp in waypoints]
        if "CLOSE" not in names or "LIFT" not in names:
            LOG.warning("collect: invalid waypoint list without CLOSE/LIFT, skipping episode")
            break
        close_idx = names.index("CLOSE")
        lift_idx = names.index("LIFT")
        if close_idx <= 0 or lift_idx <= close_idx:
            LOG.warning(
                "collect: invalid waypoint order (close_idx=%d lift_idx=%d), skipping episode",
                close_idx,
                lift_idx,
            )
            break

        _, home_arm, home_gripper = waypoints[0]
        for _ in range(30):
            if _timeout_triggered():
                break
            arm_cmd, gr_cmd = _step_toward_joint_targets(franka, home_arm, home_gripper)
            _set_joint_targets(franka, arm_cmd, gr_cmd, physics_control=True)
            world.step(render=True)
            _record_frame(arm_target=home_arm, gripper_target=home_gripper)
        if stopped:
            break

        pick_transitions = list(zip(waypoints[:close_idx], waypoints[1 : close_idx + 1]))
        lift_transitions = list(zip(waypoints[close_idx:lift_idx], waypoints[close_idx + 1 : lift_idx + 1]))
        place_transitions = list(zip(waypoints[lift_idx:-1], waypoints[lift_idx + 1 :]))

        _execute_transitions(
            pick_transitions,
            # Lock to per-attempt planned pose to avoid chasing disturbed mesh during descent.
            live_target_z_by_waypoint=None,
            # Reach-before-close gate should be evaluated before physically closing gripper.
            enable_close_hold=False,
        )
        if stopped:
            break

        # Settle near close pose with gripper open before reach gate (MagicSim-style).
        if PRE_CLOSE_SETTLE_STEPS > 0:
            close_arm_pre = waypoints[close_idx][1]
            for _ in range(PRE_CLOSE_SETTLE_STEPS):
                if _timeout_triggered():
                    break
                arm_cmd, gr_cmd = _step_toward_joint_targets(franka, close_arm_pre, GRIPPER_OPEN)
                _set_joint_targets(franka, arm_cmd, gr_cmd, physics_control=True)
                world.step(render=True)
                _record_frame(arm_target=close_arm_pre, gripper_target=GRIPPER_OPEN)
        if stopped:
            break

        reach_metrics = _compute_grasp_metrics(
            franka=franka,
            stage=stage,
            eef_prim_path=eef_prim_path,
            get_prim_at_path=get_prim_at_path,
            usd=usd,
            usd_geom=usd_geom,
            object_prim_path=object_prim_path,
            table_top_z=table_top_z,
            object_height=object_height,
            finger_left_prim_path=finger_left_prim_path,
            finger_right_prim_path=finger_right_prim_path,
        )
        reach_ok = _verify_reach_before_close(reach_metrics)
        if not reach_ok:
            LOG.warning(
                "collect: grasp retry %d/%d reach_before_close failed metrics=%s",
                attempt,
                GRASP_MAX_ATTEMPTS,
                reach_metrics,
            )
            _record_failed_annotation_target(
                cache_key=annotation_failed_pose_cache_key,
                target=current_grasp_target,
                reason="reach_verify_failed",
                attempt=attempt,
            )
            lift_arm = waypoints[lift_idx][1]
            _prepare_replan_from_current(lift_arm=lift_arm)
            if stopped:
                break
            continue

        if CLOSE_HOLD_STEPS > 0:
            close_arm = waypoints[close_idx][1]
            for _ in range(CLOSE_HOLD_STEPS):
                if _timeout_triggered():
                    break
                arm_cmd, gr_cmd = _step_toward_joint_targets(franka, close_arm, GRIPPER_CLOSED)
                _set_joint_targets(franka, arm_cmd, gr_cmd, physics_control=True)
                world.step(render=True)
                _record_frame(arm_target=close_arm, gripper_target=GRIPPER_CLOSED)
        if stopped:
            break

        close_metrics = _compute_grasp_metrics(
            franka=franka,
            stage=stage,
            eef_prim_path=eef_prim_path,
            get_prim_at_path=get_prim_at_path,
            usd=usd,
            usd_geom=usd_geom,
            object_prim_path=object_prim_path,
            table_top_z=table_top_z,
            object_height=object_height,
            finger_left_prim_path=finger_left_prim_path,
            finger_right_prim_path=finger_right_prim_path,
        )
        close_ok = _verify_after_close(close_metrics)
        if not close_ok:
            LOG.warning(
                "collect: grasp retry %d/%d close_verify failed metrics=%s",
                attempt,
                GRASP_MAX_ATTEMPTS,
                close_metrics,
            )
            _record_failed_annotation_target(
                cache_key=annotation_failed_pose_cache_key,
                target=current_grasp_target,
                reason="close_verify_failed",
                attempt=attempt,
            )
            lift_arm = waypoints[lift_idx][1]
            _prepare_replan_from_current(lift_arm=lift_arm)
            if stopped:
                break
            continue

        _execute_transitions(lift_transitions)
        if stopped:
            break

        retrieval_metrics = _compute_grasp_metrics(
            franka=franka,
            stage=stage,
            eef_prim_path=eef_prim_path,
            get_prim_at_path=get_prim_at_path,
            usd=usd,
            usd_geom=usd_geom,
            object_prim_path=object_prim_path,
            table_top_z=table_top_z,
            object_height=object_height,
            finger_left_prim_path=finger_left_prim_path,
            finger_right_prim_path=finger_right_prim_path,
        )
        retrieval_ok = _verify_after_retrieval(retrieval_metrics)

        if retrieval_ok:
            grasp_succeeded = True
            _clear_failed_annotation_pose_ids(annotation_failed_pose_cache_key)
            _execute_transitions(place_transitions)
            break

        LOG.warning(
            "collect: grasp retry %d/%d retrieval_verify failed metrics=%s",
            attempt,
            GRASP_MAX_ATTEMPTS,
            retrieval_metrics,
        )
        _record_failed_annotation_target(
            cache_key=annotation_failed_pose_cache_key,
            target=current_grasp_target,
            reason="retrieval_verify_failed",
            attempt=attempt,
        )

        lift_arm = waypoints[lift_idx][1]
        _prepare_replan_from_current(lift_arm=lift_arm)
        if stopped:
            break

    if not grasp_succeeded and not stopped and attempt_used >= GRASP_MAX_ATTEMPTS:
        LOG.warning(
            "collect: episode %d marked failed after %d grasp attempts",
            episode_index + 1,
            GRASP_MAX_ATTEMPTS,
        )

    # Ensure failed episodes are still visible in dataset metadata/data tables.
    # If no frame was emitted due early-return branches, write one snapshot frame.
    if frame_index <= 0 and not stopped:
        LOG.warning(
            "collect: episode %d produced 0 frames; writing fallback snapshot frame",
            episode_index + 1,
        )
        state = _extract_state_vector(franka, stage, eef_prim_path, get_prim_at_path, usd, usd_geom)
        action = np.zeros((ACTION_DIM,), dtype=np.float32)
        frame_extras = _build_frame_extras(
            stage=stage,
            object_prim_path=object_prim_path,
            usd=usd,
            usd_geom=usd_geom,
            current_target=current_grasp_target,
        )
        writer.add_frame(
            episode_index=episode_index,
            frame_index=0,
            observation_state=state,
            action=action,
            timestamp=0.0,
            next_done=True,
            extras=frame_extras,
        )
        for cam_name, cam_obj in cameras.items():
            rgb = _capture_rgb(cam_obj, CAMERA_RESOLUTION)
            writer.add_video_frame(cam_name, rgb)
        frame_index = 1
    else:
        # Mark this episode's final frame as done.
        for i in range(len(writer.all_frames) - 1, -1, -1):
            frame = writer.all_frames[i]
            if int(frame.get("episode_index", -1)) == episode_index:
                frame["next.done"] = True
                break

    if not stopped:
        obj_pos, _ = _get_prim_world_pose(stage, object_prim_path, usd, usd_geom)
        place_vec = np.array(place_pos if place_pos is not None else last_pick_pos, dtype=np.float32)
        place_dist = float(np.linalg.norm(obj_pos[:2] - place_vec))
        LOG.info(
            "Episode %d done: success=%s attempts=%d pick=(%.3f, %.3f) place=(%.3f, %.3f) final_obj=(%.3f, %.3f, %.3f) dist=%.3f",
            episode_index + 1,
            grasp_succeeded,
            attempt_used,
            last_pick_pos[0],
            last_pick_pos[1],
            place_vec[0],
            place_vec[1],
            obj_pos[0],
            obj_pos[1],
            obj_pos[2],
            place_dist,
        )
    else:
        LOG.info("Episode %d interrupted by stop event", episode_index + 1)

    # Ensure each episode ends in a predictable home/open pose unless an explicit stop is active.
    if not stopped and not (stop_event is not None and stop_event.is_set()):
        try:
            for _ in range(45):
                arm_cmd, gr_cmd = _step_toward_joint_targets(franka, HOME, GRIPPER_OPEN)
                _set_joint_targets(franka, arm_cmd, gr_cmd, physics_control=True)
                world.step(render=True)
                _record_frame(arm_target=HOME, gripper_target=GRIPPER_OPEN)
            LOG.info("Episode %d returned to HOME pose", episode_index + 1)
        except Exception as exc:
            LOG.warning("Episode %d failed to return HOME pose: %s", episode_index + 1, exc)

    return frame_index, stopped, grasp_succeeded


def _setup_pick_place_scene_template(
    world: Any,
    simulation_app: Any,
    fps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    import omni.usd
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
    from omni.isaac.core.utils.prims import get_prim_at_path
    from omni.isaac.sensor import Camera
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available")

    _remove_prim_if_exists(stage, COLLECTOR_ROOT)
    UsdGeom.Xform.Define(stage, COLLECTOR_ROOT)

    table = FixedCuboid(
        prim_path=TABLE_PRIM_PATH,
        name=f"table_{int(time.time() * 1000) % 100000}",
        position=np.array([0.5, 0.0, 0.4], dtype=np.float32),
        size=1.0,
        scale=np.array([0.6, 0.8, 0.02], dtype=np.float32),
        color=np.array([0.6, 0.4, 0.2], dtype=np.float32),
    )

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Cannot resolve Isaac Sim assets root path")
    added = False
    for franka_usd in _franka_usd_candidates(assets_root):
        add_reference_to_stage(usd_path=franka_usd, prim_path=ROBOT_PRIM_PATH)
        simulation_app.update()
        if _looks_like_franka_root(stage, ROBOT_PRIM_PATH):
            added = True
            break
        _remove_prim_if_exists(stage, ROBOT_PRIM_PATH)
    if not added:
        raise RuntimeError("Failed to add Franka robot from available asset paths")

    franka_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not franka_prim or not franka_prim.IsValid():
        raise RuntimeError(f"Franka prim missing at {ROBOT_PRIM_PATH}")

    try:
        translate_attr = franka_prim.GetAttribute("xformOp:translate")
        if translate_attr and translate_attr.IsValid():
            translate_attr.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        else:
            xformable = UsdGeom.Xformable(franka_prim)
            xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    except Exception as exc:
        LOG.warning("Ignoring Franka translate warning: %s", exc)

    wrist_cam_usd = UsdGeom.Camera.Define(stage, WRIST_CAM_PATH)
    wrist_prim = wrist_cam_usd.GetPrim()
    wrist_xform = UsdGeom.Xformable(wrist_prim)
    wrist_xform.ClearXformOpOrder()
    # Conservative fallback USD pose; precise ROS-frame pose is applied later
    # via Camera.set_local_pose(..., camera_axes='ros') after initialization.
    wrist_xform.AddTranslateOp().Set(Gf.Vec3d(*WRIST_CAM_FALLBACK_TRANSLATE))
    wrist_xform.AddRotateXYZOp().Set(Gf.Vec3f(*WRIST_CAM_FALLBACK_ROTATE_XYZ))
    wrist_cam_usd.GetFocalLengthAttr().Set(14.0)
    wrist_cam_usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))

    overhead_cam_usd = UsdGeom.Camera.Define(stage, OVERHEAD_CAM_PATH)
    overhead_prim = overhead_cam_usd.GetPrim()
    overhead_xform = UsdGeom.Xformable(overhead_prim)
    overhead_xform.ClearXformOpOrder()
    overhead_xform.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 1.6))
    overhead_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    overhead_cam_usd.GetFocalLengthAttr().Set(14.0)
    overhead_cam_usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))

    dome = UsdLux.DomeLight.Define(stage, f"{COLLECTOR_ROOT}/DomeLight")
    dome.GetIntensityAttr().Set(3000.0)

    key = UsdLux.DistantLight.Define(stage, f"{COLLECTOR_ROOT}/KeyLight")
    key.GetIntensityAttr().Set(5000.0)
    key_xf = UsdGeom.Xformable(key.GetPrim())
    key_xf.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

    work = UsdLux.SphereLight.Define(stage, f"{COLLECTOR_ROOT}/WorkLight")
    work.GetIntensityAttr().Set(15000.0)
    work.GetRadiusAttr().Set(0.15)
    work_xf = UsdGeom.Xformable(work.GetPrim())
    work_xf.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 1.2))

    target_object = DynamicCuboid(
        prim_path=OBJECT_PRIM_PATH,
        name=f"pick_object_{int(time.time() * 1000) % 100000}",
        position=np.array([0.5, 0.0, TABLE_Z + OBJECT_SIZE / 2.0], dtype=np.float32),
        size=OBJECT_SIZE,
        color=rng.uniform(0.2, 0.9, size=3).astype(np.float32),
        mass=OBJECT_MASS,
    )

    grip_mat = UsdShade.Material.Define(stage, MATERIAL_PRIM_PATH)
    physics_api = UsdPhysics.MaterialAPI.Apply(grip_mat.GetPrim())
    physics_api.CreateStaticFrictionAttr(1.5)
    physics_api.CreateDynamicFrictionAttr(1.0)
    physics_api.CreateRestitutionAttr(0.0)

    obj_prim = stage.GetPrimAtPath(OBJECT_PRIM_PATH)
    if obj_prim and obj_prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(obj_prim).Bind(
            grip_mat,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )

    for _ in range(80):
        simulation_app.update()

    articulation_prim_path = _resolve_articulation_root_prim_path(stage, ROBOT_PRIM_PATH, UsdPhysics) or ROBOT_PRIM_PATH
    franka = Articulation(prim_path=articulation_prim_path, name=f"franka_pick_place_{int(time.time())}")
    world.scene.add(franka)

    camera_high = Camera(
        prim_path=OVERHEAD_CAM_PATH,
        name=f"cam_high_{int(time.time() * 1000) % 100000}",
        frequency=fps,
        resolution=CAMERA_RESOLUTION,
    )
    camera_wrist = Camera(
        prim_path=WRIST_CAM_PATH,
        name=f"cam_wrist_{int(time.time() * 1000) % 100000}",
        frequency=fps,
        resolution=CAMERA_RESOLUTION,
    )
    world.scene.add(camera_high)
    world.scene.add(camera_wrist)

    world.reset()
    for _ in range(30):
        world.step(render=True)

    if hasattr(franka, "initialize"):
        try:
            franka.initialize()
        except Exception as exc:
            LOG.warning("Franka initialize() failed: %s", exc)
    _wait_articulation_ready(franka, world, steps=40)

    for cam in (camera_high, camera_wrist):
        if hasattr(cam, "initialize"):
            try:
                cam.initialize()
            except Exception as exc:
                LOG.warning("Camera initialize() failed: %s", exc)
    _apply_wrist_camera_pose(
        camera_wrist=camera_wrist,
        stage=stage,
        usd_geom=UsdGeom,
        gf=Gf,
        wrist_cam_path=WRIST_CAM_PATH,
    )

    eef_prim_path = _resolve_eef_prim(stage, ROBOT_PRIM_PATH, get_prim_at_path)
    if eef_prim_path is None:
        eef_prim_path = f"{ROBOT_PRIM_PATH}/panda_hand"
    finger_left_prim_path, finger_right_prim_path = _resolve_finger_prim_paths(
        stage=stage,
        robot_prim_path=ROBOT_PRIM_PATH,
        get_prim_at_path=get_prim_at_path,
    )
    if not (finger_left_prim_path and finger_right_prim_path):
        LOG.warning(
            "collect: fingertip prims not resolved under %s; reach gate falls back to panda_hand distance",
            ROBOT_PRIM_PATH,
        )
    ik_solver = _create_franka_ik_solver(
        franka=franka,
        stage=stage,
        robot_prim_path=ROBOT_PRIM_PATH,
        eef_prim_path=eef_prim_path,
        usd=Usd,
        usd_geom=UsdGeom,
    )

    table_x_range, table_y_range, table_top_z, work_center = _infer_table_workspace(
        stage,
        UsdGeom,
        table_prim_path=TABLE_PRIM_PATH,
    )
    table_x_range, table_y_range, table_top_z, work_center = _adjust_workspace_for_robot_reach(
        stage=stage,
        usd_geom=UsdGeom,
        robot_prim_path=ROBOT_PRIM_PATH,
        table_x_range=table_x_range,
        table_y_range=table_y_range,
        table_top_z=table_top_z,
        work_center=work_center,
    )

    return {
        "stage": stage,
        "franka": franka,
        "articulation_prim_path": articulation_prim_path,
        "cameras": {"cam_high": camera_high, "cam_wrist": camera_wrist},
        "ik_solver": ik_solver,
        "object_prim_path": OBJECT_PRIM_PATH,
        "robot_prim_path": ROBOT_PRIM_PATH,
        "eef_prim_path": eef_prim_path,
        "finger_left_prim_path": finger_left_prim_path,
        "finger_right_prim_path": finger_right_prim_path,
        "get_prim_at_path": get_prim_at_path,
        "usd": Usd,
        "usd_geom": UsdGeom,
        "gf": Gf,
        "table_x_range": table_x_range,
        "table_y_range": table_y_range,
        "table_top_z": table_top_z,
        "work_center": work_center,
    }


def _setup_pick_place_scene_reuse_or_patch(
    world: Any,
    simulation_app: Any,
    fps: int,
    rng: np.random.Generator,
    target_objects: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    import omni.usd
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
    from omni.isaac.sensor import Camera
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available")

    table_added = False
    table_root = _resolve_table_prim(stage, UsdGeom)
    if table_root is None:
        if _prim_is_valid(stage, "/World/Table"):
            table_root = "/World/Table"
            LOG.info("collect: fallback reusing existing /World/Table")
        else:
            table_root = "/World/Table"
            table = FixedCuboid(
                prim_path=table_root,
                name=f"table_collect_{int(time.time() * 1000) % 100000}",
                position=np.array([0.0, 0.0, 0.75], dtype=np.float32),
                size=1.0,
                scale=np.array([1.2, 0.8, 0.04], dtype=np.float32),
                color=np.array([0.6, 0.4, 0.2], dtype=np.float32),
            )
            world.scene.add(table)
            table_added = True
            LOG.info("collect: missing table -> added %s", table_root)
    else:
        LOG.info("collect: reusing table at %s", table_root)

    robot_prim_path = _resolve_robot_prim(stage)
    if robot_prim_path is None:
        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError("Cannot resolve Isaac Sim assets root path for Franka")
        robot_prim_path = _add_franka_reference(
            stage=stage,
            simulation_app=simulation_app,
            add_reference_to_stage=add_reference_to_stage,
            assets_root=assets_root,
            usd_geom=UsdGeom,
            gf=Gf,
            usd_physics=UsdPhysics,
            prim_path="/World/FrankaRobot",
        )
        LOG.info("collect: missing Franka -> added %s", robot_prim_path)
    else:
        # Existing Franka prim can be an unloaded/incomplete reference in interactive scenes.
        # If articulation root is not ready, re-add a known-good Franka reference once.
        for _ in range(6):
            simulation_app.update()
        if not _has_articulation_root(stage, robot_prim_path, UsdPhysics):
            LOG.warning("collect: existing Franka not articulation-ready at %s, reloading", robot_prim_path)
            _remove_prim_if_exists(stage, robot_prim_path)
            assets_root = get_assets_root_path()
            if not assets_root:
                raise RuntimeError("Cannot resolve Isaac Sim assets root path for Franka reload")
            robot_prim_path = _add_franka_reference(
                stage=stage,
                simulation_app=simulation_app,
                add_reference_to_stage=add_reference_to_stage,
                assets_root=assets_root,
                usd_geom=UsdGeom,
                gf=Gf,
                usd_physics=UsdPhysics,
                prim_path="/World/FrankaRobot",
            )
            LOG.info("collect: reloaded Franka at %s", robot_prim_path)

    articulation_prim_path = _resolve_articulation_root_prim_path(stage, robot_prim_path, UsdPhysics)
    if not articulation_prim_path:
        raise RuntimeError(f"collect cannot find articulation root under {robot_prim_path}")

    table_x_range, table_y_range, table_top_z, work_center = _infer_table_workspace(
        stage,
        UsdGeom,
        table_prim_path=table_root,
    )
    table_x_range, table_y_range, table_top_z, work_center = _adjust_workspace_for_robot_reach(
        stage=stage,
        usd_geom=UsdGeom,
        robot_prim_path=robot_prim_path,
        table_x_range=table_x_range,
        table_y_range=table_y_range,
        table_top_z=table_top_z,
        work_center=work_center,
    )

    object_prim_path = _resolve_target_object_prim(
        stage,
        UsdGeom,
        target_hints=target_objects,
        table_x_range=table_x_range,
        table_y_range=table_y_range,
        table_top_z=table_top_z,
    )
    if object_prim_path is None:
        obj_x = float((table_x_range[0] + table_x_range[1]) * 0.5)
        obj_y = float((table_y_range[0] + table_y_range[1]) * 0.5)
        object_prim_path = "/World/PickObject"
        _remove_prim_if_exists(stage, object_prim_path)
        UsdGeom.Xform.Define(stage, object_prim_path)
        cube = UsdGeom.Cube.Define(stage, f"{object_prim_path}/Body")
        cube.GetSizeAttr().Set(1.0)
        bx = UsdGeom.Xformable(cube.GetPrim())
        bx.ClearXformOpOrder()
        bx.AddScaleOp().Set(Gf.Vec3d(OBJECT_SIZE, OBJECT_SIZE, OBJECT_SIZE))
        root = stage.GetPrimAtPath(object_prim_path)
        rx = UsdGeom.Xformable(root)
        rx.ClearXformOpOrder()
        rx.AddTranslateOp().Set(Gf.Vec3d(obj_x, obj_y, table_top_z + OBJECT_SIZE / 2.0 + 0.003))
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        try:
            UsdPhysics.MeshCollisionAPI.Apply(cube.GetPrim())
        except Exception:
            pass
        LOG.info("collect: missing grasp object -> added %s", object_prim_path)
    else:
        try:
            bbox = _compute_prim_bbox(stage, object_prim_path, UsdGeom)
            if bbox:
                mn, mx = bbox
                LOG.info(
                    "collect: target object=%s bbox_min=(%.3f, %.3f, %.3f) bbox_max=(%.3f, %.3f, %.3f)",
                    object_prim_path,
                    float(mn[0]),
                    float(mn[1]),
                    float(mn[2]),
                    float(mx[0]),
                    float(mx[1]),
                    float(mx[2]),
                )
            else:
                LOG.info("collect: target object=%s (bbox unavailable)", object_prim_path)
        except Exception:
            LOG.info("collect: target object=%s", object_prim_path)

    # Ensure chosen object has basic rigid/collision physics for pure physical grasp.
    obj_root = stage.GetPrimAtPath(object_prim_path)
    if obj_root and obj_root.IsValid() and not obj_root.HasAPI(UsdPhysics.RigidBodyAPI):
        try:
            UsdPhysics.RigidBodyAPI.Apply(obj_root)
        except Exception:
            pass
    if obj_root and obj_root.IsValid():
        for p in _iter_descendants(obj_root):
            if not p or not p.IsValid() or not p.IsA(UsdGeom.Mesh):
                continue
            try:
                if not p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(p)
                if not p.HasAPI(UsdPhysics.MeshCollisionAPI):
                    UsdPhysics.MeshCollisionAPI.Apply(p)
            except Exception:
                pass

    table_x_range, table_y_range, table_top_z, work_center = _infer_table_workspace(
        stage,
        UsdGeom,
        table_prim_path=table_root,
    )
    table_x_range, table_y_range, table_top_z, work_center = _adjust_workspace_for_robot_reach(
        stage=stage,
        usd_geom=UsdGeom,
        robot_prim_path=robot_prim_path,
        table_x_range=table_x_range,
        table_y_range=table_y_range,
        table_top_z=table_top_z,
        work_center=work_center,
    )
    # Keep user-arranged scene intact: do not auto-move robot base or object here.

    overhead_cam_path = "/World/OverheadCam"
    overhead_prim = stage.GetPrimAtPath(overhead_cam_path)
    if not overhead_prim or not overhead_prim.IsValid() or not overhead_prim.IsA(UsdGeom.Camera):
        over_usd = UsdGeom.Camera.Define(stage, overhead_cam_path)
        LOG.info("collect: missing overhead camera -> added %s", overhead_cam_path)
    over_prim = stage.GetPrimAtPath(overhead_cam_path)
    if over_prim and over_prim.IsValid():
        over_usd = UsdGeom.Camera(over_prim)
        over_xf = UsdGeom.Xformable(over_prim)
        over_xf.ClearXformOpOrder()
        over_xf.AddTranslateOp().Set(Gf.Vec3d(float(work_center[0]), float(work_center[1]), 1.6))
        over_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        try:
            over_usd.GetFocalLengthAttr().Set(14.0)
            over_usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
        except Exception:
            pass

    wrist_cam_path = f"{robot_prim_path}/panda_hand/wrist_cam"
    wrist_prim = stage.GetPrimAtPath(wrist_cam_path)
    if not wrist_prim or not wrist_prim.IsValid() or not wrist_prim.IsA(UsdGeom.Camera):
        wrist_usd = UsdGeom.Camera.Define(stage, wrist_cam_path)
        LOG.info("collect: missing wrist camera -> added %s", wrist_cam_path)
    wrist_prim = stage.GetPrimAtPath(wrist_cam_path)
    if wrist_prim and wrist_prim.IsValid():
        wrist_usd = UsdGeom.Camera(wrist_prim)
        wrist_xf = UsdGeom.Xformable(wrist_prim)
        wrist_xf.ClearXformOpOrder()
        wrist_xf.AddTranslateOp().Set(Gf.Vec3d(*WRIST_CAM_FALLBACK_TRANSLATE))
        wrist_xf.AddRotateXYZOp().Set(Gf.Vec3f(*WRIST_CAM_FALLBACK_ROTATE_XYZ))
        try:
            wrist_usd.GetFocalLengthAttr().Set(14.0)
            wrist_usd.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
        except Exception:
            pass

    for _ in range(40 if table_added else 20):
        simulation_app.update()

    franka = Articulation(prim_path=articulation_prim_path, name=f"franka_collect_{int(time.time())}")
    world.scene.add(franka)
    camera_high = Camera(
        prim_path=overhead_cam_path,
        name=f"cam_high_{int(time.time() * 1000) % 100000}",
        frequency=fps,
        resolution=CAMERA_RESOLUTION,
    )
    camera_wrist = Camera(
        prim_path=wrist_cam_path,
        name=f"cam_wrist_{int(time.time() * 1000) % 100000}",
        frequency=fps,
        resolution=CAMERA_RESOLUTION,
    )
    world.scene.add(camera_high)
    world.scene.add(camera_wrist)

    world.reset()
    for _ in range(20):
        world.step(render=True)

    if hasattr(franka, "initialize"):
        try:
            franka.initialize()
        except Exception as exc:
            LOG.warning("Franka initialize() failed: %s", exc)
    _wait_articulation_ready(franka, world, steps=40)
    for cam in (camera_high, camera_wrist):
        if hasattr(cam, "initialize"):
            try:
                cam.initialize()
            except Exception as exc:
                LOG.warning("Camera initialize() failed: %s", exc)
    _apply_wrist_camera_pose(
        camera_wrist=camera_wrist,
        stage=stage,
        usd_geom=UsdGeom,
        gf=Gf,
        wrist_cam_path=wrist_cam_path,
    )

    from omni.isaac.core.utils.prims import get_prim_at_path

    eef_prim_path = _resolve_eef_prim(stage, robot_prim_path, get_prim_at_path) or f"{robot_prim_path}/panda_hand"
    finger_left_prim_path, finger_right_prim_path = _resolve_finger_prim_paths(
        stage=stage,
        robot_prim_path=robot_prim_path,
        get_prim_at_path=get_prim_at_path,
    )
    if not (finger_left_prim_path and finger_right_prim_path):
        LOG.warning(
            "collect: fingertip prims not resolved under %s; reach gate falls back to panda_hand distance",
            robot_prim_path,
        )
    ik_solver = _create_franka_ik_solver(
        franka=franka,
        stage=stage,
        robot_prim_path=robot_prim_path,
        eef_prim_path=eef_prim_path,
        usd=Usd,
        usd_geom=UsdGeom,
    )

    return {
        "stage": stage,
        "franka": franka,
        "articulation_prim_path": articulation_prim_path,
        "cameras": {"cam_high": camera_high, "cam_wrist": camera_wrist},
        "ik_solver": ik_solver,
        "object_prim_path": object_prim_path,
        "robot_prim_path": robot_prim_path,
        "eef_prim_path": eef_prim_path,
        "finger_left_prim_path": finger_left_prim_path,
        "finger_right_prim_path": finger_right_prim_path,
        "get_prim_at_path": get_prim_at_path,
        "usd": Usd,
        "usd_geom": UsdGeom,
        "gf": Gf,
        "table_x_range": table_x_range,
        "table_y_range": table_y_range,
        "table_top_z": table_top_z,
        "work_center": work_center,
        "table_prim_path": table_root,
    }


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
    scene_mode: str = "auto",
    target_objects: Optional[Sequence[str]] = None,
    dataset_repo_id: Optional[str] = None,
    episode_timeout_sec: float | None = None,
) -> dict[str, Any]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if steps_per_segment <= 0:
        raise ValueError("steps_per_segment must be > 0")
    task_name = str(task_name or "pick_place")
    scene_mode = str(scene_mode or "auto").lower()
    if episode_timeout_sec is None:
        episode_timeout_sec = _read_float_env("COLLECT_EPISODE_TIMEOUT_SEC", DEFAULT_EPISODE_TIMEOUT_SEC)
    episode_timeout_sec = max(0.0, float(episode_timeout_sec))

    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    if scene_mode in {"auto", "reuse", "existing", "patch", "preserve"}:
        ctx = _setup_pick_place_scene_reuse_or_patch(
            world=world,
            simulation_app=simulation_app,
            fps=fps,
            rng=rng,
            target_objects=target_objects,
        )
        actual_scene_mode = "reuse_or_patch"
    elif scene_mode in {"template", "generated"}:
        ctx = _setup_pick_place_scene_template(world=world, simulation_app=simulation_app, fps=fps, rng=rng)
        actual_scene_mode = "template"
    else:
        raise ValueError(f"Unsupported scene_mode: {scene_mode}")

    writer = SimLeRobotWriter(
        output_dir=output_dir,
        repo_id=(str(dataset_repo_id).strip() if dataset_repo_id else _default_dataset_repo_id(output_dir)),
        fps=fps,
        robot_type=ROBOT_TYPE,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        camera_names=CAMERA_NAMES,
        camera_resolution=CAMERA_RESOLUTION,
        state_names=STATE_NAMES,
        action_names=ACTION_NAMES,
        extra_features=FRAME_EXTRA_FEATURES,
    )

    completed = 0
    successful_episodes = 0
    try:
        for episode in range(num_episodes):
            if stop_event.is_set():
                LOG.info("Stop requested before episode %d", episode + 1)
                break

            LOG.info("Episode %d/%d", episode + 1, num_episodes)
            episode_start = time.time()
            episode_frames = 0
            stopped = False
            episode_success = False
            try:
                episode_frames, stopped, episode_success = _run_pick_place_episode(
                    episode_index=episode,
                    world=world,
                    franka=ctx["franka"],
                    cameras=ctx["cameras"],
                    stage=ctx["stage"],
                    eef_prim_path=ctx["eef_prim_path"],
                    get_prim_at_path=ctx["get_prim_at_path"],
                    usd=ctx["usd"],
                    usd_geom=ctx["usd_geom"],
                    writer=writer,
                    rng=rng,
                    fps=fps,
                    steps_per_segment=steps_per_segment,
                    object_prim_path=ctx["object_prim_path"],
                    ik_solver=ctx.get("ik_solver"),
                    table_x_range=ctx["table_x_range"],
                    table_y_range=ctx["table_y_range"],
                    table_top_z=ctx["table_top_z"],
                    work_center=ctx["work_center"],
                    gf=ctx["gf"],
                    stop_event=stop_event,
                    episode_timeout_sec=episode_timeout_sec,
                    finger_left_prim_path=ctx.get("finger_left_prim_path"),
                    finger_right_prim_path=ctx.get("finger_right_prim_path"),
                )
            finally:
                writer.finish_episode(
                    episode_index=episode,
                    length=episode_frames,
                    task=task_name,
                )
                LOG.info(
                    "Episode %d finished with %d frames in %.2fs",
                    episode + 1,
                    episode_frames,
                    time.time() - episode_start,
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

    return {
        "completed": completed,
        "successful_episodes": successful_episodes,
        "failed_episodes": max(0, completed - successful_episodes),
        "requested": num_episodes,
        "output_dir": output_dir,
        "stopped": bool(stop_event.is_set()),
        "scene_mode": actual_scene_mode,
        "target_object": ctx["object_prim_path"],
        "episode_timeout_sec": episode_timeout_sec,
    }


def main() -> int:
    args = _parse_args()
    _configure_collector_logging(output_dir=args.output)

    if args.num_episodes <= 0:
        LOG.error("--num-episodes must be > 0")
        return 1
    if args.fps <= 0:
        LOG.error("--fps must be > 0")
        return 1
    if args.steps_per_segment <= 0:
        LOG.error("--steps-per-segment must be > 0")
        return 1

    LOG.info("Starting pick-place collector. headless=%s episodes=%d", args.headless, args.num_episodes)
    LOG.info("Expected launcher: /isaac-sim/python.sh (current python: %s)", sys.executable)

    simulation_app = None
    try:
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

        simulation_app = SimulationApp(app_config)

        if args.streaming:
            try:
                import omni.kit.app
                import carb

                ext_manager = omni.kit.app.get_app().get_extension_manager()
                ext_manager.set_extension_enabled_immediate("omni.services.livestream.nvcf", True)
                settings = carb.settings.get_settings()
                settings.set("/app/livestream/port", args.streaming_port)
                LOG.info("WebRTC streaming enabled on port %d", args.streaming_port)
            except Exception as exc:
                LOG.warning("Failed to enable streaming (continuing): %s", exc)

        from omni.isaac.core import World

        world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0)
        world.scene.add_default_ground_plane()

        stop_event = threading.Event()
        result = run_collection_in_process(
            world=world,
            simulation_app=simulation_app,
            num_episodes=args.num_episodes,
            output_dir=args.output,
            stop_event=stop_event,
            progress_callback=lambda ep: LOG.info("Progress: %d/%d", ep, args.num_episodes),
            fps=args.fps,
            steps_per_segment=args.steps_per_segment,
            seed=args.seed,
            task_name=args.task_name,
            episode_timeout_sec=args.episode_timeout_sec,
        )

        LOG.info("Collection complete: %s", result)
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
