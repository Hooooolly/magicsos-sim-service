"""LeRobot v3.0 writer for MagicSim data collection (Franka).

Writes joint states + actions to parquet, camera RGB to mp4.
Produces a dataset that RLInf loads directly via dataset_path.

Output structure:
    {output_dir}/
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   ├── tasks.jsonl
    │   └── episodes/chunk-000/file-000.parquet
    ├── data/chunk-000/file-000.parquet
    └── videos/observation.images.{cam}/chunk-000/episode_{idx:05d}.mp4
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger("lerobot-writer")


class SimLeRobotWriter:
    """Write LeRobot v3.0 dataset from Isaac Sim collection data."""

    def __init__(
        self,
        output_dir: str,
        repo_id: str = "local/franka_collection",
        fps: int = 30,
        robot_type: str = "franka",
        state_dim: int = 7,
        action_dim: int = 7,
        camera_names: Optional[List[str]] = None,
        camera_resolution: tuple[int, int] = (224, 224),
        state_names: Optional[List[str]] = None,
        action_names: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.repo_id = repo_id
        self.fps = fps
        self.robot_type = robot_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.camera_names = camera_names or ["cam_high", "cam_wrist"]
        self.camera_resolution = camera_resolution
        self.state_names = state_names
        self.action_names = action_names

        # Create directory structure
        (self.output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in self.camera_names:
            (self.output_dir / "videos" / f"observation.images.{cam}" / "chunk-000").mkdir(
                parents=True, exist_ok=True
            )

        self.all_frames: list[dict] = []
        self.episode_info: list[dict] = []
        self.global_frame_idx = 0

        # Per-episode video frame buffers: {camera_name: [np.ndarray, ...]}
        self._video_buffers: dict[str, list[np.ndarray]] = {c: [] for c in self.camera_names}

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------

    def add_frame(
        self,
        episode_index: int,
        frame_index: int,
        observation_state: np.ndarray,
        action: np.ndarray,
        timestamp: float,
        next_done: bool = False,
    ) -> None:
        """Append one frame of tabular data (joint state + action)."""
        self.all_frames.append(
            {
                "index": self.global_frame_idx,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "timestamp": float(timestamp),
                "observation.state": observation_state.astype(np.float32).tolist(),
                "action": action.astype(np.float32).tolist(),
                "next.done": bool(next_done),
            }
        )
        self.global_frame_idx += 1

    def add_video_frame(self, camera_name: str, rgb: np.ndarray) -> None:
        """Append one RGB frame (H, W, 3) uint8 for a camera."""
        if camera_name not in self._video_buffers:
            LOG.warning("Unknown camera %s, skipping frame", camera_name)
            return
        self._video_buffers[camera_name].append(rgb.astype(np.uint8))

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def finish_episode(
        self,
        episode_index: int,
        length: int,
        task: str = "grasp object",
    ) -> None:
        """Flush video buffers for this episode and record metadata."""
        chunk = episode_index // 1000

        for cam in self.camera_names:
            frames = self._video_buffers[cam]
            if frames:
                self._write_episode_video(episode_index, cam, frames, chunk)
            self._video_buffers[cam] = []

        self.episode_info.append(
            {
                "episode_index": episode_index,
                "length": length,
                "task": task,
                "task_index": 0,
                "video_chunk_index": chunk,
                "video_file_index": 0,
                "data_chunk_index": chunk,
                "data_file_index": 0,
            }
        )

    # ------------------------------------------------------------------
    # Video writing
    # ------------------------------------------------------------------

    def _write_episode_video(
        self,
        episode_index: int,
        camera_name: str,
        frames: list[np.ndarray],
        chunk: int = 0,
    ) -> None:
        """Write a list of RGB frames to an mp4 file."""
        video_dir = (
            self.output_dir
            / "videos"
            / f"observation.images.{camera_name}"
            / f"chunk-{chunk:03d}"
        )
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"episode_{episode_index:05d}.mp4"

        try:
            import imageio.v3 as iio

            iio.imwrite(
                video_path,
                np.stack(frames),
                fps=self.fps,
                codec="libx264",
                plugin="pyav",
            )
        except Exception:
            try:
                import imageio

                writer = imageio.get_writer(
                    str(video_path), fps=self.fps, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"],
                )
                for f in frames:
                    writer.append_data(f)
                writer.close()
            except Exception:
                # Last resort: save PNGs, then try ffmpeg subprocess
                png_dir = video_dir / f"episode_{episode_index:05d}"
                png_dir.mkdir(parents=True, exist_ok=True)
                for i, f in enumerate(frames):
                    from PIL import Image

                    Image.fromarray(f).save(png_dir / f"frame_{i:05d}.png")
                try:
                    import subprocess

                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-framerate", str(self.fps),
                            "-i", str(png_dir / "frame_%05d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            str(video_path),
                        ],
                        capture_output=True, timeout=120,
                    )
                    if video_path.exists() and video_path.stat().st_size > 0:
                        import shutil
                        shutil.rmtree(png_dir)
                    else:
                        LOG.warning("ffmpeg produced empty file for %s ep %d — keeping PNGs", camera_name, episode_index)
                except Exception as exc:
                    LOG.warning("Video write failed for %s ep %d: %s — PNGs saved", camera_name, episode_index, exc)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Write all parquet + metadata files."""
        LOG.info("Finalizing LeRobot v3.0 dataset (%d frames, %d episodes)...",
                 self.global_frame_idx, len(self.episode_info))
        self._write_data_parquet()
        self._write_episode_parquet()
        self._write_info_json()
        self._write_stats_json()
        self._write_tasks_jsonl()
        LOG.info("Dataset saved to %s", self.output_dir)

    def _write_data_parquet(self) -> None:
        if not self.all_frames:
            return
        df = pd.DataFrame(self.all_frames)
        path = self.output_dir / "data" / "chunk-000" / "file-000.parquet"
        df.to_parquet(path, index=False)

    def _write_episode_parquet(self) -> None:
        if not self.episode_info:
            return
        df = pd.DataFrame(self.episode_info)
        path = self.output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        df.to_parquet(path, index=False)

    def _write_info_json(self) -> None:
        h, w = self.camera_resolution
        features: dict = {
            "observation.state": {
                "dtype": "float32",
                "shape": [self.state_dim],
                "names": {"motors": self.state_names} if self.state_names else None,
            },
            "action": {
                "dtype": "float32",
                "shape": [self.action_dim],
                "names": {"motors": self.action_names} if self.action_names else None,
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
        }
        for cam in self.camera_names:
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
                "video_info": {"fps": self.fps, "codec": "libx264"},
            }

        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "fps": self.fps,
            "total_episodes": len(self.episode_info),
            "total_frames": self.global_frame_idx,
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:05d}.mp4",
            "features": features,
            "splits": {"train": f"0:{len(self.episode_info)}"},
        }

        with open(self.output_dir / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def _write_stats_json(self) -> None:
        if not self.all_frames:
            return
        states = np.array([f["observation.state"] for f in self.all_frames])
        actions = np.array([f["action"] for f in self.all_frames])

        stats = {}
        for name, arr in [("observation.state", states), ("action", actions)]:
            stats[name] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": np.maximum(arr.std(axis=0), 1e-8).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
            }

        with open(self.output_dir / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    def _write_tasks_jsonl(self) -> None:
        tasks = sorted({ep["task"] for ep in self.episode_info})
        with open(self.output_dir / "meta" / "tasks.jsonl", "w") as f:
            for i, task in enumerate(tasks):
                f.write(json.dumps({"task_index": i, "task": task}) + "\n")
