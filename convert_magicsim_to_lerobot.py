"""Convert MagicSim TestOutput format to LeRobot v3.0 dataset.

Reads trajectories from MagicSim's native TestOutput directory structure
and writes a LeRobot v3.0 dataset using SimLeRobotWriter.

MagicSim TestOutput layout (per trajectory):
    {trajectory_id}/
        action/action_0000.json ... action_NNNN.json
        env/info/env_0_step_0000.json ... env_0_step_NNNN.json
        env/camera/cam_0/rgb/step_0000.png ... step_NNNN.png

LeRobot v3.0 output:
    {dataset_id}/
        meta/info.json, stats.json, tasks.jsonl, episodes/...
        data/chunk-000/file-000.parquet
        videos/observation.images.cam_high/chunk-000/episode_XXXXX.mp4

Usage:
    python convert_magicsim_to_lerobot.py \
        --input /path/to/TestOutput \
        --output /path/to/lerobot_dataset \
        --dataset-id local/franka_grasp
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure the script directory is importable for lerobot_writer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lerobot_writer import SimLeRobotWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
LOG = logging.getLogger("magicsim-to-lerobot")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# State: 7 arm joint_pos + 7 arm joint_vel + 2 gripper_pos + 3 eef_pos + 4 eef_quat
STATE_DIM = 23
# Action: 7 arm joints + 1 gripper
ACTION_DIM = 8
FPS = 30
ROBOT_TYPE = "franka"
CAMERA_RESOLUTION = (512, 512)
CAMERA_NAMES = ["cam_high"]

STATE_NAMES = [
    "joint_pos_0", "joint_pos_1", "joint_pos_2", "joint_pos_3",
    "joint_pos_4", "joint_pos_5", "joint_pos_6",
    "joint_vel_0", "joint_vel_1", "joint_vel_2", "joint_vel_3",
    "joint_vel_4", "joint_vel_5", "joint_vel_6",
    "gripper_pos_0", "gripper_pos_1",
    "eef_pos_x", "eef_pos_y", "eef_pos_z",
    "eef_quat_w", "eef_quat_x", "eef_quat_y", "eef_quat_z",
]
ACTION_NAMES = [
    "action_joint_0", "action_joint_1", "action_joint_2", "action_joint_3",
    "action_joint_4", "action_joint_5", "action_joint_6",
    "action_gripper",
]

# Camera mapping: MagicSim cam_0 -> LeRobot cam_high
CAM_MAPPING = {"cam_0": "cam_high"}


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _numeric_sort_key(path: Path) -> int:
    """Extract the trailing integer from a filename for numeric sorting.

    Handles patterns like: action_0000.json, env_0_step_0042.json, step_0042.png
    """
    match = re.search(r"(\d+)", path.stem.split("_")[-1])
    if match:
        return int(match.group(1))
    # Fallback: try the full stem
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else 0


def discover_trajectories(input_dir: Path) -> list[Path]:
    """Find and return trajectory directories sorted numerically.

    Trajectory directories are numbered subdirectories (0, 1, 2, ...) inside
    the TestOutput folder.
    """
    trajectories = []
    for child in input_dir.iterdir():
        if not child.is_dir():
            continue
        # Accept directories whose name is a non-negative integer
        try:
            int(child.name)
            trajectories.append(child)
        except ValueError:
            continue

    trajectories.sort(key=lambda p: int(p.name))
    return trajectories


def list_sorted_files(directory: Path, glob_pattern: str) -> list[Path]:
    """List files matching glob_pattern inside directory, sorted numerically."""
    files = list(directory.glob(glob_pattern))
    files.sort(key=_numeric_sort_key)
    return files


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _get_robot_data(env_info: dict) -> dict | None:
    """Navigate to robot data dict inside env_info JSON.

    Path: obs -> policy_obs -> robot_state -> [0] -> {RobotName} -> {...}
    """
    obs = env_info.get("obs", {})
    policy_obs = obs.get("policy_obs", {})
    robot_state = policy_obs.get("robot_state")

    if not isinstance(robot_state, list) or not robot_state:
        return None

    robot_map = robot_state[0]
    if not isinstance(robot_map, dict) or not robot_map:
        return None

    robot_data = next(iter(robot_map.values()))
    return robot_data if isinstance(robot_data, dict) else None


def _safe_array(data: dict, key: str, expected_len: int, take: int) -> np.ndarray:
    """Extract a float32 sub-array from robot_data, zero-padded if missing."""
    raw = data.get(key)
    if raw is None:
        return np.zeros(take, dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    out = np.zeros(take, dtype=np.float32)
    n = min(take, arr.size)
    out[:n] = arr[:n]
    return out


def extract_state_from_env_info(env_info: dict) -> np.ndarray:
    """Extract 23D state: joint_pos[7] + joint_vel[7] + gripper_pos[2] + eef_pos[3] + eef_quat[4]."""
    robot_data = _get_robot_data(env_info)
    if robot_data is None:
        return np.zeros(STATE_DIM, dtype=np.float32)

    joint_pos = _safe_array(robot_data, "joint_pos", 9, 7)
    joint_vel = _safe_array(robot_data, "joint_vel", 9, 7)
    gripper_pos = _safe_array(robot_data, "gripper_pos", 2, 2)
    eef_pos = _safe_array(robot_data, "eef_pos", 3, 3)
    eef_quat = _safe_array(robot_data, "eef_quat", 4, 4)

    return np.concatenate([joint_pos, joint_vel, gripper_pos, eef_pos, eef_quat])


def extract_action_from_action_json(action_data: dict) -> np.ndarray:
    """Extract 8D action: 7 arm joints + 1 gripper from robot_action[8]."""
    robot_action = action_data.get("robot_action")
    if robot_action is None:
        return np.zeros(ACTION_DIM, dtype=np.float32)

    arr = np.asarray(robot_action, dtype=np.float32).reshape(-1)
    output = np.zeros(ACTION_DIM, dtype=np.float32)
    n = min(ACTION_DIM, arr.size)
    output[:n] = arr[:n]
    return output


# ---------------------------------------------------------------------------
# Single trajectory processing
# ---------------------------------------------------------------------------

def process_trajectory(
    traj_dir: Path,
    episode_index: int,
    writer: SimLeRobotWriter,
    fps: int = FPS,
) -> int:
    """Process one MagicSim trajectory directory and write to the LeRobot writer.

    Returns the number of frames written (0 if trajectory is empty/corrupt).
    """
    action_dir = traj_dir / "action"
    env_info_dir = traj_dir / "env" / "info"
    rgb_dir = traj_dir / "env" / "camera" / "cam_0" / "rgb"

    # Discover action files — the primary count reference
    action_files = list_sorted_files(action_dir, "action_*.json")
    if not action_files:
        LOG.warning("Trajectory %s: no action files, skipping.", traj_dir.name)
        return 0

    # Discover env info files — MagicSim names them env_{env_id}_step_*.json
    # where env_id varies (0-3 for 4 parallel envs). Find whichever exists.
    env_info_files = list_sorted_files(env_info_dir, "env_*_step_*.json")
    rgb_files = list_sorted_files(rgb_dir, "step_*.png")

    # Use action count as the authoritative step count.
    # Env info and RGB may have one extra step (the initial observation at t=0
    # before any action). We align by using min(action_count, other_count).
    num_steps = len(action_files)

    if len(env_info_files) < num_steps:
        LOG.warning(
            "Trajectory %s: %d actions but only %d env_info files. "
            "Using %d steps.",
            traj_dir.name, num_steps, len(env_info_files), len(env_info_files),
        )
        num_steps = len(env_info_files)

    if num_steps == 0:
        LOG.warning("Trajectory %s: 0 usable steps after alignment, skipping.", traj_dir.name)
        return 0

    has_rgb = len(rgb_files) >= num_steps

    if not has_rgb:
        LOG.warning(
            "Trajectory %s: %d RGB images for %d steps. "
            "Video will be incomplete.",
            traj_dir.name, len(rgb_files), num_steps,
        )

    for step in range(num_steps):
        # --- Read action ---
        with open(action_files[step]) as f:
            action_data = json.load(f)
        action_vec = extract_action_from_action_json(action_data)

        # --- Read observation state ---
        with open(env_info_files[step]) as f:
            env_info_data = json.load(f)
        state_vec = extract_state_from_env_info(env_info_data)

        # --- Write tabular frame ---
        is_last = (step == num_steps - 1)
        writer.add_frame(
            episode_index=episode_index,
            frame_index=step,
            observation_state=state_vec,
            action=action_vec,
            timestamp=step / fps,
            next_done=is_last,
        )

        # --- Read and write RGB frame ---
        if step < len(rgb_files):
            rgb = np.array(Image.open(rgb_files[step]).convert("RGB"), dtype=np.uint8)
            writer.add_video_frame(CAM_MAPPING["cam_0"], rgb)

    return num_steps


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def convert(
    input_dir: Path,
    output_dir: Path,
    dataset_id: str,
    task: str,
    fps: int = FPS,
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
) -> None:
    """Run the full MagicSim -> LeRobot v3.0 conversion."""
    t_start = time.time()
    LOG.info("Input:      %s", input_dir)
    LOG.info("Output:     %s", output_dir)
    LOG.info("Dataset ID: %s", dataset_id)
    LOG.info("Task:       %s", task)
    LOG.info("FPS:        %d", fps)
    LOG.info("State dim:  %d, Action dim: %d", state_dim, action_dim)
    LOG.info("-" * 60)

    trajectories = discover_trajectories(input_dir)
    if not trajectories:
        LOG.error("No trajectory directories found in %s", input_dir)
        sys.exit(1)

    LOG.info("Found %d trajectory directories.", len(trajectories))

    writer = SimLeRobotWriter(
        output_dir=str(output_dir),
        repo_id=dataset_id,
        fps=fps,
        robot_type=ROBOT_TYPE,
        state_dim=state_dim,
        action_dim=action_dim,
        camera_names=CAMERA_NAMES,
        camera_resolution=CAMERA_RESOLUTION,
        state_names=STATE_NAMES,
        action_names=ACTION_NAMES,
    )

    total_frames = 0
    episodes_written = 0
    skipped = 0

    for traj_path in trajectories:
        traj_id = int(traj_path.name)
        LOG.info(
            "Processing trajectory %d (episode %d)...",
            traj_id, episodes_written,
        )

        num_frames = process_trajectory(traj_path, episodes_written, writer, fps=fps)

        if num_frames == 0:
            skipped += 1
            LOG.info("  -> Skipped (empty/corrupt).")
            continue

        writer.finish_episode(
            episode_index=episodes_written,
            length=num_frames,
            task=task,
        )
        total_frames += num_frames
        episodes_written += 1
        LOG.info("  -> %d frames written.", num_frames)

    if episodes_written == 0:
        LOG.error("No valid episodes produced. Output directory is empty.")
        sys.exit(1)

    writer.finalize()

    elapsed = time.time() - t_start

    # Compute output size
    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    size_mb = total_bytes / (1024 * 1024)

    LOG.info("=" * 60)
    LOG.info("Conversion complete.")
    LOG.info("  Episodes:   %d written, %d skipped", episodes_written, skipped)
    LOG.info("  Frames:     %d total", total_frames)
    LOG.info("  Output:     %s (%.1f MB)", output_dir, size_mb)
    LOG.info("  Time:       %.1f s", elapsed)
    LOG.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MagicSim TestOutput to LeRobot v3.0 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="MagicSim TestOutput directory (contains numbered trajectory dirs).",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        type=Path,
        help="Output directory for the LeRobot v3.0 dataset.",
    )
    parser.add_argument(
        "--dataset-id",
        default="local/franka_collection",
        help="Dataset repo ID string (default: local/franka_collection).",
    )
    parser.add_argument(
        "--task",
        default="grasp object",
        help="Task description for the episodes (default: 'grasp object').",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help=f"Frames per second (default: {FPS}).",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=STATE_DIM,
        help=f"Observation state dimension (default: {STATE_DIM}).",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=ACTION_DIM,
        help=f"Action dimension (default: {ACTION_DIM}).",
    )

    args = parser.parse_args()

    if not args.input.is_dir():
        LOG.error("Input directory does not exist: %s", args.input)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    convert(
        input_dir=args.input,
        output_dir=args.output,
        dataset_id=args.dataset_id,
        task=args.task,
        fps=args.fps,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )


if __name__ == "__main__":
    main()
