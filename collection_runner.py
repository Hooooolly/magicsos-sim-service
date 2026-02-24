"""K8s Job entry point for MagicSim data collection.

Reads job config from /config/job.json (K8s ConfigMap).
Pipeline: Isaac Sim init → scene load → camera setup → robot spawn
          → AtomicSkill collection → LeRobot3 + RecordManager output.

Usage: /isaac-sim/python.sh -u collection_runner.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
LOG = logging.getLogger("collection-runner")

SCRIPT_DIR = Path(__file__).resolve().parent


class _MagicLoggerAdapter:
    """Minimal logger adapter for MagicSim components."""

    def info(self, *message: Any) -> None:
        LOG.info("%s", " ".join(str(m) for m in message))

    def debug(self, *message: Any) -> None:
        LOG.debug("%s", " ".join(str(m) for m in message))

    def warning(self, *message: Any) -> None:
        LOG.warning("%s", " ".join(str(m) for m in message))

    def error(self, *message: Any) -> None:
        LOG.error("%s", " ".join(str(m) for m in message))


def _normalize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_")


def _canonical_skill_name(skill: str) -> str:
    mapping = {
        "reach": "Reach",
        "grasp": "Grasp",
        "place": "Place",
        "push": "Push",
        "wave": "Wave",
        "retractreach": "RetractReach",
        "mobilereach": "MobileReach",
        "robotgoto": "RobotGoTo",
    }
    skill_name = str(skill).strip()
    return mapping.get(skill_name.lower(), skill_name)


def _infer_camera_names(camera_config: Any) -> list[str]:
    if camera_config is None:
        return ["cam_high", "cam_wrist"]
    names = []
    for key in camera_config.keys():
        if key in ("enable_tiled", "colorize_depth"):
            continue
        names.append(str(key))
    return names or ["cam_high", "cam_wrist"]


def _pick_target(scene_mgr: Any, target_objects: list[str], scene_state: dict) -> tuple[str, str, int]:
    rigid_keys = []
    geometry_keys = []
    try:
        rigid_keys = list(scene_mgr.rigid_objects[0].keys())
    except Exception:
        rigid_keys = []
    try:
        geometry_keys = list(scene_mgr.geometry_objects[0].keys())
    except Exception:
        geometry_keys = []

    candidate_names = list(target_objects or [])
    objects = scene_state.get("objects", {})
    if isinstance(objects, dict):
        for _, obj in objects.items():
            if isinstance(obj, dict) and obj.get("name"):
                candidate_names.append(str(obj.get("name")))

    normalized_rigid = {_normalize_name(k): k for k in rigid_keys}
    normalized_geo = {_normalize_name(k): k for k in geometry_keys}
    for candidate in candidate_names:
        norm = _normalize_name(candidate)
        if norm in normalized_rigid:
            return "rigid", normalized_rigid[norm], 0
        if norm in normalized_geo:
            return "geometry", normalized_geo[norm], 0

    if rigid_keys:
        return "rigid", rigid_keys[0], 0
    if geometry_keys:
        return "geometry", geometry_keys[0], 0
    raise RuntimeError("No target objects found in SceneManager.")


def _build_skill_command(skill: str, obj_type: str, obj_name: str, obj_id: int) -> list[Any]:
    skill_name = _canonical_skill_name(skill)
    if skill_name in ("Reach", "RetractReach", "Push", "MobileReach", "RobotGoTo"):
        return [skill_name, 0, 0, obj_type, obj_name, int(obj_id)]
    return [skill_name, obj_type, obj_name, int(obj_id)]


def load_job_config() -> dict:
    """Load job config from K8s ConfigMap mount or env var."""
    config_path = os.environ.get("JOB_CONFIG", "/config/job.json")
    LOG.info("Loading job config from %s", config_path)
    with open(config_path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Phase 1: Isaac Sim
# ------------------------------------------------------------------

def init_isaac_sim(device: str = "cuda:0"):
    """Start Isaac Sim headless. Returns MagicLauncher instance."""
    LOG.info("Phase 1 — Initializing Isaac Sim (headless, device=%s)...", device)
    from magicsim.Launch.MagicLauncher import MagicLauncher

    launcher = MagicLauncher(headless=True, enable_cameras=True, device=device)
    LOG.info("Isaac Sim ready.")
    return launcher


# ------------------------------------------------------------------
# Phase 2: Scene
# ------------------------------------------------------------------

def load_scene(scene_dir: str) -> tuple[str, str, dict]:
    """Generate MagicSim YAML from SceneSmith scene_state.json.

    Returns (yaml_path, yaml_text, scene_state dict).
    """
    LOG.info("Phase 2 — Loading scene from %s", scene_dir)
    sys.path.insert(0, str(SCRIPT_DIR))
    from scene_loader import generate_magicsim_yaml, load_scene_state

    yaml_path = str(Path(scene_dir) / "magicsim_scene.yaml")
    yaml_text = generate_magicsim_yaml(scene_dir, output_path=yaml_path)
    scene_state = load_scene_state(scene_dir)

    obj_map = scene_state.get("objects", {})
    obj_count = len(obj_map) if isinstance(obj_map, dict) else 0
    LOG.info("Scene YAML generated: %d objects → %s", obj_count, yaml_path)
    return yaml_path, yaml_text, scene_state


# ------------------------------------------------------------------
# Phase 3: Cameras
# ------------------------------------------------------------------

def load_camera_config() -> dict | None:
    """Load camera YAML from same directory."""
    config_path = SCRIPT_DIR / "camera_config.yaml"
    if not config_path.exists():
        LOG.warning("camera_config.yaml not found at %s", config_path)
        return None

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(config_path))
    LOG.info("Phase 3 — Camera config loaded: %s", list(cfg.keys()))
    return cfg


# ------------------------------------------------------------------
# Phase 4: Collection loop
# ------------------------------------------------------------------

def collect_episodes(
    launcher,
    yaml_path: str,
    scene_state: dict,
    camera_config,
    robot_type: str,
    skill: str,
    target_objects: list[str],
    num_episodes: int,
    dataset_dir: str,
    trajectory_dir: str,
    max_episode_steps: int = 200,
) -> None:
    """Run the MagicSim collection loop and write LeRobot3 + RecordManager."""
    import numpy as np
    from lerobot_writer import SimLeRobotWriter

    callback_url = os.environ.get("CALLBACK_URL", "").strip() or None
    callback_error_logged = False

    def emit_status(status: str, **extra: Any) -> None:
        nonlocal callback_error_logged
        if not callback_url:
            return
        payload = {
            "status": status,
            "timestamp": time.time(),
            "phase": "collection",
        }
        payload.update(extra)
        request = urllib.request.Request(
            callback_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5):
                pass
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            if not callback_error_logged:
                LOG.warning("CALLBACK_URL post failed (%s): %s", callback_url, exc)
                callback_error_logged = True

    camera_names = _infer_camera_names(camera_config)
    writer = SimLeRobotWriter(
        output_dir=dataset_dir,
        repo_id=f"local/collection_{Path(dataset_dir).name[:8]}",
        fps=30,
        robot_type=robot_type,
        camera_names=camera_names,
    )

    def _to_numpy(value: Any):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        try:
            return np.asarray(value)
        except Exception:
            return None

    def _first_bool(value: Any) -> bool:
        array = _to_numpy(value)
        if array is None:
            return False
        if array.shape == ():
            return bool(array.item())
        if array.size == 0:
            return False
        return bool(array.reshape(-1)[0])

    def _extract_joint_positions(obs: dict[str, Any]) -> np.ndarray:
        policy_obs = obs.get("policy_obs", {}) if isinstance(obs, dict) else {}
        robot_state = policy_obs.get("robot_state")
        if not isinstance(robot_state, list) or not robot_state:
            return np.zeros(7, dtype=np.float32)
        robot_map = robot_state[0]
        if not isinstance(robot_map, dict) or not robot_map:
            return np.zeros(7, dtype=np.float32)
        robot_data = next(iter(robot_map.values()))
        if not isinstance(robot_data, dict):
            return np.zeros(7, dtype=np.float32)
        for key in ("joint_positions", "joint_position", "joint_pos", "joint"):
            if key not in robot_data:
                continue
            array = _to_numpy(robot_data.get(key))
            if array is None:
                continue
            flat = array.reshape(-1).astype(np.float32, copy=False)
            if flat.size == 0:
                continue
            output = np.zeros(7, dtype=np.float32)
            output[: min(7, flat.size)] = flat[:7]
            return output
        return np.zeros(7, dtype=np.float32)

    def _extract_action_vector(action: Any) -> np.ndarray:
        array = _to_numpy(action)
        if array is None:
            return np.zeros(7, dtype=np.float32)
        if array.ndim > 1:
            array = array[0]
        flat = array.reshape(-1).astype(np.float32, copy=False)
        output = np.zeros(7, dtype=np.float32)
        output[: min(7, flat.size)] = flat[:7]
        return output

    def _extract_rgb_frames(obs: dict[str, Any]) -> dict[str, np.ndarray]:
        rgb_frames = {}
        policy_obs = obs.get("policy_obs", {}) if isinstance(obs, dict) else {}
        camera_info = policy_obs.get("camera_info")
        if not isinstance(camera_info, list):
            return rgb_frames
        for cam_idx, cam_name in enumerate(camera_names):
            if cam_idx >= len(camera_info):
                break
            cam_data = camera_info[cam_idx]
            if not isinstance(cam_data, dict):
                continue
            rgb_by_env = cam_data.get("rgb")
            if not isinstance(rgb_by_env, (list, tuple)) or not rgb_by_env:
                continue
            rgb = _to_numpy(rgb_by_env[0])
            if rgb is None:
                continue
            if rgb.ndim == 4:
                rgb = rgb[0]
            if rgb.ndim != 3:
                continue
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            if rgb.dtype != np.uint8:
                if np.issubdtype(rgb.dtype, np.floating) and float(np.max(rgb)) <= 1.0:
                    rgb = rgb * 255.0
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb_frames[cam_name] = rgb
        return rgb_frames

    def _write_placeholder_episode(episode_index: int, steps: int) -> int:
        LOG.warning(
            "Episode %d using degraded placeholder mode (%d steps, zero state/action)",
            episode_index + 1,
            steps,
        )
        for step in range(steps):
            writer.add_frame(
                episode_index=episode_index,
                frame_index=step,
                observation_state=np.zeros(7, dtype=np.float32),
                action=np.zeros(7, dtype=np.float32),
                timestamp=step / 30.0,
                next_done=(step == steps - 1),
            )
        return steps

    task_env = None
    scene_mgr = None
    skill_mgr = None
    global_planner_mgr = None
    record_mgr = None
    is_magicsim_ready = False

    emit_status(
        "running",
        event="collection_start",
        mode="initializing",
        episodes=int(num_episodes),
        max_episode_steps=int(max_episode_steps),
    )
    try:
        import gymnasium as gym
        import magicsim
        import magicsim.Env
        import magicsim.StardardEnv.Robot
        import magicsim.Task.TableTop.Env
        import torch
        from omegaconf import OmegaConf
        from magicsim.Collect.AtomicSkill.AtomicSkillManager import AtomicSkillManager
        from magicsim.Collect.GlobalPlanner.GlobalPlannerManager import GlobalPlannerManager
        from magicsim.Collect.Record.RecordManager import RecordManager

        magicsim_root = Path(magicsim.__file__).resolve().parent
        tabletop_scene_conf = magicsim_root / "Task" / "TableTop" / "Conf" / "Scene"
        tabletop_collect_conf = magicsim_root / "Collect" / "Task" / "TableTop" / "Conf"
        sim_cfg = OmegaConf.load(str(tabletop_scene_conf / "sim" / "sim_config.yaml"))
        robot_cfg = OmegaConf.load(str(tabletop_scene_conf / "robot" / "curobo_franka.yaml"))
        scene_cfg = OmegaConf.load(str(yaml_path))
        camera_cfg = camera_config
        if camera_cfg is None:
            camera_cfg = OmegaConf.load(str(tabletop_scene_conf / "camera" / "grasp_camera.yaml"))

        preferred_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        sim_cfg.device = preferred_device
        if preferred_device != "cpu":
            sim_cfg.use_fabric = True
        sim_cfg.scene.num_envs = 1

        env_cfg = OmegaConf.create(
            {
                "timeout_steps": int(max_episode_steps),
                "auto_reset": False,
                "Scene": {
                    "sim": OmegaConf.to_container(sim_cfg, resolve=True),
                    "scene": OmegaConf.to_container(scene_cfg, resolve=True),
                    "camera": OmegaConf.to_container(camera_cfg, resolve=True),
                    "robot": OmegaConf.to_container(robot_cfg, resolve=True),
                },
            }
        )
        if target_objects:
            env_cfg.target_object_name = _normalize_name(target_objects[0])
            env_cfg.target_category = _normalize_name(target_objects[0])

        canonical_skill = _canonical_skill_name(skill)
        env_by_skill = {
            "reach": "ReachEnv-V0",
            "push": "PushEnv-V0",
            "wave": "WaveEnv-V0",
            "grasp": "GraspEnv-V0",
            "place": "GraspEnv-V0",
        }
        env_id = env_by_skill.get(canonical_skill.lower(), "GraspEnv-V0")
        task_env = gym.make(
            env_id,
            config=env_cfg,
            cli_args=None,
            logger=_MagicLoggerAdapter(),
        )
        if hasattr(task_env, "unwrapped"):
            task_env = task_env.unwrapped

        scene_mgr = task_env.scene.scene_manager

        atomic_cfg = OmegaConf.load(str(tabletop_collect_conf / "atomic_skill" / "default.yaml"))
        global_planner_cfg = OmegaConf.load(str(tabletop_collect_conf / "global_planner" / "default.yaml"))
        record_cfg = OmegaConf.load(str(tabletop_collect_conf / "record" / "default.yaml"))
        record_cfg.output_dir = trajectory_dir

        manager_logger = _MagicLoggerAdapter()
        manager_device = torch.device(task_env.device)
        skill_mgr = AtomicSkillManager(
            task_env,
            1,
            atomic_cfg,
            device=manager_device,
            logger=manager_logger,
        )
        global_planner_mgr = GlobalPlannerManager(
            task_env,
            1,
            global_planner_cfg,
            device=manager_device,
            logger=manager_logger,
        )
        record_mgr = RecordManager(
            task_env,
            1,
            record_cfg,
            device=manager_device,
            logger=manager_logger,
        )

        is_magicsim_ready = True
        LOG.info(
            "MagicSim integration initialized (env=%s, device=%s, cameras=%s)",
            env_id,
            preferred_device,
            camera_names,
        )
    except Exception as exc:
        LOG.warning("MagicSim init failed (%s). Using degraded placeholder mode.", exc)
        emit_status("running", event="fallback_mode", mode="placeholder", warning=str(exc))

    LOG.info(
        "Starting collection: %d episodes of %s (max_steps=%d, mode=%s)",
        num_episodes,
        skill,
        max_episode_steps,
        "magicsim" if is_magicsim_ready else "placeholder",
    )
    emit_status(
        "running",
        event="collection_ready",
        mode="magicsim" if is_magicsim_ready else "placeholder",
    )

    for episode in range(num_episodes):
        t0 = time.time()
        LOG.info("Episode %d/%d", episode + 1, num_episodes)
        emit_status(
            "running",
            event="episode_start",
            episode=int(episode + 1),
            total_episodes=int(num_episodes),
        )
        episode_steps = 0
        placeholder_episode = not is_magicsim_ready
        try:
            if not is_magicsim_ready:
                episode_steps = _write_placeholder_episode(episode, int(max_episode_steps))
            else:
                reset_obs, reset_info = task_env.reset()
                if scene_mgr is not None:
                    scene_mgr.reset(soft=False)
                if record_mgr is not None:
                    record_mgr.reset({"env_info": (reset_obs, reset_info)})

                obj_type, obj_name, obj_id = _pick_target(scene_mgr, target_objects, scene_state)
                command = _build_skill_command(canonical_skill, obj_type, obj_name, obj_id)

                skill_mgr.create_atomic_skill(canonical_skill, env_id=0)
                skill_mgr.atomic_skill_list[0].reset(command)

                frame_index = 0
                for _ in range(max_episode_steps):
                    commands = [command]
                    atomic_actions, valid_env_ids, failed_env_ids = skill_mgr.step(commands, [0])

                    planner_actions = None
                    planner_valid_env_ids = list(valid_env_ids)
                    planner_failed_env_ids = []
                    if planner_valid_env_ids:
                        planner_actions, planner_valid_env_ids, planner_failed_env_ids = (
                            global_planner_mgr.step(atomic_actions, planner_valid_env_ids)
                        )

                    all_failed_env_ids = list(failed_env_ids) + list(planner_failed_env_ids)
                    env_action = planner_actions if planner_valid_env_ids else None
                    env_ids = planner_valid_env_ids if planner_valid_env_ids else None

                    obs, reward, terminated, truncated, info, pending_env_ids = task_env.step(
                        action=env_action,
                        env_ids=env_ids,
                        failed_env_ids=all_failed_env_ids,
                    )

                    record_payload = {"env_info": (obs, reward, terminated, truncated, info)}
                    global_info = global_planner_mgr.update(record_payload)
                    record_payload["global_planner_info"] = global_info
                    atomic_info = skill_mgr.update(record_payload)
                    record_payload["atomic_skill_info"] = atomic_info
                    record_payload["auto_collect_info"] = atomic_info

                    pending_list = []
                    if pending_env_ids is not None:
                        if hasattr(pending_env_ids, "detach"):
                            pending_list = pending_env_ids.detach().cpu().tolist()
                        else:
                            pending_list = list(pending_env_ids)
                    pending_set = {int(v) for v in pending_list}
                    ready_env_ids = [
                        int(v) for v in (planner_valid_env_ids or []) if int(v) not in pending_set
                    ]
                    ready_env_ids.extend(int(v) for v in all_failed_env_ids)

                    if ready_env_ids and record_mgr is not None:
                        record_mgr.step(record_payload, ready_env_ids=ready_env_ids)
                        record_mgr.update(record_payload)

                    joint_positions = _extract_joint_positions(obs)
                    action_vec = _extract_action_vector(planner_actions)
                    atomic_state = ""
                    atomic_finished = False
                    atomic_truncated = False
                    if isinstance(atomic_info, list) and atomic_info and atomic_info[0] is not None:
                        atomic_state = str(atomic_info[0].get("state", ""))
                        atomic_finished = bool(atomic_info[0].get("finished", False))
                        atomic_truncated = int(atomic_info[0].get("truncated", 0)) > 0
                    done = (
                        _first_bool(terminated)
                        or _first_bool(truncated)
                        or atomic_finished
                        or atomic_truncated
                        or atomic_state.startswith(("success", "failed", "truncated"))
                        or bool(all_failed_env_ids)
                    )

                    writer.add_frame(
                        episode_index=episode,
                        frame_index=frame_index,
                        observation_state=joint_positions,
                        action=action_vec,
                        timestamp=frame_index / 30.0,
                        next_done=done,
                    )
                    episode_steps = frame_index + 1
                    for cam_name, rgb in _extract_rgb_frames(obs).items():
                        writer.add_video_frame(cam_name, rgb)

                    frame_index += 1
                    if done:
                        break

                if episode_steps == 0:
                    placeholder_episode = True
                    episode_steps = _write_placeholder_episode(episode, int(max_episode_steps))
        except Exception as exc:
            LOG.exception("Episode %d failed: %s", episode + 1, exc)
            emit_status(
                "running",
                event="episode_failed",
                episode=int(episode + 1),
                error=str(exc),
            )
            if episode_steps == 0:
                placeholder_episode = True
                episode_steps = _write_placeholder_episode(episode, int(max_episode_steps))

        writer.finish_episode(episode, length=episode_steps, task=f"{skill} target object")
        LOG.info(
            "Episode %d done (%.1fs, %d steps, mode=%s)",
            episode + 1,
            time.time() - t0,
            episode_steps,
            "placeholder" if placeholder_episode else "magicsim",
        )
        emit_status(
            "running",
            event="episode_done",
            episode=int(episode + 1),
            steps=int(episode_steps),
            mode="placeholder" if placeholder_episode else "magicsim",
        )

    writer.finalize()
    LOG.info("LeRobot3 dataset written to %s", dataset_dir)
    emit_status("succeeded", event="collection_complete", dataset_dir=dataset_dir)
    if task_env is not None and hasattr(task_env, "close"):
        try:
            task_env.close()
        except Exception as exc:
            LOG.warning("Failed to close task env cleanly: %s", exc)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    config = load_job_config()
    job_id = config["job_id"]
    scene_dir = config["scene_dir"]
    robot_type = config.get("robot_type", "franka")
    skill = config.get("skill", "Grasp")
    target_objects = config.get("target_objects", [])
    num_episodes = config.get("num_episodes", 100)
    max_episode_steps = int(config.get("max_episode_steps", config.get("max_steps", 200)))
    output_dir = config.get("output_dir", f"/data/trajectories/{job_id}")
    dataset_dir = f"/data/datasets/{job_id}"

    LOG.info("=" * 60)
    LOG.info("Collection Job %s", job_id)
    LOG.info("  scene_dir:  %s", scene_dir)
    LOG.info("  robot:      %s", robot_type)
    LOG.info("  skill:      %s", skill)
    LOG.info("  targets:    %s", target_objects)
    LOG.info("  episodes:   %d", num_episodes)
    LOG.info("  max_steps:  %d", max_episode_steps)
    LOG.info("  dataset:    %s", dataset_dir)
    LOG.info("  trajectory: %s", output_dir)
    LOG.info("=" * 60)

    # Phase 1: Isaac Sim
    launcher = init_isaac_sim()

    # Phase 2: Scene
    yaml_path, yaml_text, scene_state = load_scene(scene_dir)

    # Phase 3: Cameras
    camera_config = load_camera_config()

    # Phase 4: Collection
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    collect_episodes(
        launcher=launcher,
        yaml_path=yaml_path,
        scene_state=scene_state,
        camera_config=camera_config,
        robot_type=robot_type,
        skill=skill,
        target_objects=target_objects,
        num_episodes=num_episodes,
        dataset_dir=dataset_dir,
        trajectory_dir=output_dir,
        max_episode_steps=max_episode_steps,
    )

    LOG.info("=== Job %s complete ===", job_id)
    launcher.app.close()


if __name__ == "__main__":
    main()
