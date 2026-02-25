#!/usr/bin/env python3
"""Generate manual grasp annotation for YCB sugar_box (004).

Output pose format: [x, y, z, qw, qx, qy, qz] (wxyz).
All local positions are in meters (position_scale_hint=1.0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


TOP_DOWN_QUAT_WXYZ = [0.0, 1.0, 0.0, 0.0]

# Measured in running Isaac stage with Axis_Aligned sugar_box + scale 0.01:
# size ~= (0.09268, 0.17625, 0.04513) m.
DEFAULT_HALF_EXTENTS_M = (0.04634, 0.08813, 0.02257)


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-8:
        return fallback.astype(np.float64, copy=True)
    return (v / n).astype(np.float64, copy=False)


def _quat_from_approach_and_finger(
    approach_dir_world: np.ndarray,
    finger_axis_world: np.ndarray,
) -> list[float]:
    # Local -Z points to approach direction; local X aligns with finger axis.
    a = _normalize(approach_dir_world, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    x = _normalize(finger_axis_world, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    x = x - float(np.dot(x, a)) * a
    x = _normalize(x, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    y = np.cross(a, x)
    y = _normalize(y, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    z = -a
    rot = np.column_stack([x, y, z])
    if np.linalg.det(rot) < 0.0:
        rot[:, 1] = -rot[:, 1]
    q_xyzw = Rotation.from_matrix(rot).as_quat()
    q_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
    if q_wxyz[0] < 0.0:
        q_wxyz = [-v for v in q_wxyz]
    return q_wxyz


def _build_manual_sugar_pose(hx: float, hy: float, hz: float) -> dict[str, Any]:
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    y_offsets = [-0.040, -0.018, 0.018, 0.040]
    z_levels = [-0.004, 0.004]

    body_points: list[list[float]] = []
    for side in (-1.0, 1.0):
        x = float(side * hx * 0.90)
        for yy in y_offsets:
            y = float(np.clip(yy, -hy * 0.82, hy * 0.82))
            for zz in z_levels:
                z = float(np.clip(zz, -hz * 0.75, hz * 0.75))
                pos = np.array([x, y, z], dtype=np.float64)
                approach = pos - center
                approach[2] = 0.0  # side pinch first
                quat = _quat_from_approach_and_finger(
                    approach_dir_world=approach,
                    finger_axis_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                )
                body_points.append([float(pos[0]), float(pos[1]), float(pos[2])] + quat)

    topdown_points: list[list[float]] = []
    topdown_z = float(hz * 0.92)
    for x in (-hx * 0.45, hx * 0.45):
        for y in (-hy * 0.32, 0.0, hy * 0.32):
            topdown_points.append([float(x), float(y), topdown_z] + list(TOP_DOWN_QUAT_WXYZ))

    grasp_body = body_points + topdown_points
    return {
        "functional_grasp": {
            "body": body_points,
            "topdown_side": topdown_points,
        },
        "grasp": {"body": grasp_body},
        "metadata": {
            "source": "manual_sugar_box_template_v1",
            "quaternion_order": "wxyz",
            "position_unit": "meters",
            "position_scale_hint": 1.0,
            "num_total": len(grasp_body),
            "num_body_side": len(body_points),
            "num_top_down": len(topdown_points),
            "half_extents_m": [float(hx), float(hy), float(hz)],
        },
    }


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate manual sugar_box grasp annotation.")
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "grasp_poses" / "sugar_grasp_pose.json",
        help="Output grasp pose JSON path.",
    )
    parser.add_argument(
        "--also-write",
        type=Path,
        default=None,
        help="Optional second output path.",
    )
    parser.add_argument("--hx", type=float, default=DEFAULT_HALF_EXTENTS_M[0], help="half extent x in meters")
    parser.add_argument("--hy", type=float, default=DEFAULT_HALF_EXTENTS_M[1], help="half extent y in meters")
    parser.add_argument("--hz", type=float, default=DEFAULT_HALF_EXTENTS_M[2], help="half extent z in meters")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = _build_manual_sugar_pose(hx=float(args.hx), hy=float(args.hy), hz=float(args.hz))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)

    if args.also_write is not None:
        args.also_write.parent.mkdir(parents=True, exist_ok=True)
        with args.also_write.open("w") as f:
            json.dump(out, f, indent=2)

    md = out.get("metadata", {})
    print(
        "wrote",
        str(args.output),
        "total=",
        md.get("num_total"),
        "body=",
        md.get("num_body_side"),
        "topdown=",
        md.get("num_top_down"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
