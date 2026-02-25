#!/usr/bin/env python3
"""Generate manual (non-VLM) mug grasp poses from mesh geometry.

This script builds three pose groups:
1) functional_grasp.handle
2) functional_grasp.body
3) functional_grasp.topdown_side

Pose format: [x, y, z, qw, qx, qy, qz] (wxyz quaternion order).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


TOP_DOWN_QUAT_WXYZ = [0.0, 1.0, 0.0, 0.0]


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-8:
        return fallback.astype(np.float64)
    return (v / n).astype(np.float64)


def _quat_from_approach_and_finger(
    approach_dir_world: np.ndarray,
    finger_axis_world: np.ndarray,
) -> list[float]:
    """Build quaternion where local -Z points to approach, local X is finger axis."""
    a = _normalize(approach_dir_world, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    x = _normalize(finger_axis_world, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    # Force orthogonality: remove approach component from finger axis.
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


def _detect_handle_dir(vertices: np.ndarray, center: np.ndarray) -> np.ndarray:
    xy = vertices[:, :2] - center[:2]
    r = np.linalg.norm(xy, axis=1)
    q = float(np.percentile(r, 96.0))
    idx = r >= q
    if int(np.sum(idx)) < 16:
        q = float(np.percentile(r, 92.0))
        idx = r >= q
    handle_vec = np.mean(xy[idx], axis=0) if int(np.sum(idx)) > 0 else np.array([1.0, 0.0])
    return _normalize(handle_vec, np.array([1.0, 0.0], dtype=np.float64))


def _load_vlm_handle_points(vlm_json: Path | None) -> np.ndarray | None:
    if vlm_json is None or not vlm_json.exists():
        return None
    with vlm_json.open("r") as f:
        data = json.load(f)
    handle = data.get("functional_grasp", {}).get("handle", [])
    pts: list[list[float]] = []
    for p in handle:
        if not isinstance(p, list) or len(p) < 3:
            continue
        xyz = [float(p[0]), float(p[1]), float(p[2])]
        if all(np.isfinite(xyz)):
            pts.append(xyz)
    if len(pts) < 4:
        return None
    return np.asarray(pts, dtype=np.float64)


def _estimate_widest_body_z(
    verts: np.ndarray,
    center: np.ndarray,
    handle_dir_2d: np.ndarray,
    z_min: float,
    z_max: float,
) -> float:
    """Estimate body widest-height (away from handle side)."""
    z_span = max(1e-6, z_max - z_min)
    xy = verts[:, :2] - center[:2]
    r = np.linalg.norm(xy, axis=1)
    z = verts[:, 2]
    handle_side_score = xy @ handle_dir_2d

    # Exclude clear handle-side vertices to avoid picking handle loop as "widest".
    side_cut = np.percentile(handle_side_score, 65.0)
    # Search in mid-upper cup wall but avoid the very rim.
    z_band_lo = z_min + 0.42 * z_span
    z_band_hi = z_min + 0.72 * z_span
    mask = (handle_side_score <= side_cut) & (z >= z_band_lo) & (z <= z_band_hi)
    if int(np.sum(mask)) < 100:
        mask = (z >= z_min + 0.35 * z_span) & (z <= z_min + 0.75 * z_span)

    z_sel = z[mask]
    r_sel = r[mask]
    if z_sel.size == 0:
        return float(z_min + 0.40 * z_span)

    n_bins = 32
    edges = np.linspace(float(np.min(z_sel)), float(np.max(z_sel)), n_bins + 1)
    best_z = float(np.median(z_sel))
    best_score = -1.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        bmask = (z_sel >= lo) & (z_sel < hi)
        if int(np.sum(bmask)) < 20:
            continue
        # Use 70th percentile to represent local cup-body width robustly.
        score = float(np.percentile(r_sel[bmask], 70.0))
        if score > best_score:
            best_score = score
            best_z = 0.5 * (lo + hi)
    return float(best_z)


def _build_manual_mug_pose(mesh: trimesh.Trimesh, vlm_handle_points: np.ndarray | None = None) -> dict[str, Any]:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    center = np.asarray(mesh.centroid, dtype=np.float64)
    z_min = float(np.min(verts[:, 2]))
    z_max = float(np.max(verts[:, 2]))
    z_span = max(1e-6, z_max - z_min)

    handle_dir_2d = _detect_handle_dir(verts, center)
    if vlm_handle_points is not None and len(vlm_handle_points) >= 6:
        hv = vlm_handle_points[:, :2] - center[:2]
        handle_dir_2d = _normalize(np.mean(hv, axis=0), handle_dir_2d)
    body_dir_2d = np.array([-handle_dir_2d[1], handle_dir_2d[0]], dtype=np.float64)
    handle_dir = np.array([handle_dir_2d[0], handle_dir_2d[1], 0.0], dtype=np.float64)
    body_dir = np.array([body_dir_2d[0], body_dir_2d[1], 0.0], dtype=np.float64)

    xy = verts[:, :2] - center[:2]
    r = np.linalg.norm(xy, axis=1)
    z = verts[:, 2]
    z_lo = z_min + 0.30 * z_span
    z_hi = z_min + 0.82 * z_span
    mid_mask = (z >= z_lo) & (z <= z_hi)

    handle_side_score = xy @ handle_dir_2d
    # Keep away from handle side when estimating body radius.
    body_radius_candidates = r[mid_mask & (handle_side_score <= np.percentile(handle_side_score, 60.0))]
    if body_radius_candidates.size == 0:
        body_radius_candidates = r[mid_mask]
    if body_radius_candidates.size == 0:
        body_radius_candidates = r

    body_radius = float(np.percentile(body_radius_candidates, 62.0))
    body_radius = max(body_radius, 0.16 * max(np.ptp(verts[:, 0]), np.ptp(verts[:, 1])))

    handle_radius = float(np.percentile(r, 97.5))
    handle_radius = max(handle_radius, body_radius + 0.10 * max(np.ptp(verts[:, 0]), np.ptp(verts[:, 1])))

    widest_z = _estimate_widest_body_z(verts, center, handle_dir_2d, z_min, z_max)
    # Keep points off the rim (too high) and off the table plane (too low).
    handle_z_levels = [z_min + 0.30 * z_span, z_min + 0.38 * z_span, z_min + 0.46 * z_span, z_min + 0.54 * z_span]
    body_half_band = 0.03 * z_span
    body_z_levels = [
        max(z_min + 0.44 * z_span, min(z_min + 0.70 * z_span, widest_z - body_half_band)),
        max(z_min + 0.44 * z_span, min(z_min + 0.70 * z_span, widest_z + body_half_band)),
    ]
    topdown_z_levels = [
        z_min + 0.56 * z_span,
        z_min + 0.64 * z_span,
    ]

    tangent_offsets = [-0.28 * body_radius, -0.14 * body_radius, 0.0, 0.14 * body_radius, 0.28 * body_radius]
    radial_scales = [1.0, 0.90]
    used_vlm_handle_hint = False
    if vlm_handle_points is not None and len(vlm_handle_points) >= 6:
        hv = vlm_handle_points[:, :2] - center[:2]
        tcoord = hv @ body_dir_2d
        t_lo, t_hi = float(np.percentile(tcoord, 10.0)), float(np.percentile(tcoord, 90.0))
        if np.isfinite(t_lo) and np.isfinite(t_hi) and (t_hi - t_lo) > (0.06 * body_radius):
            tangent_offsets = np.linspace(t_lo, t_hi, 5).tolist()

        rcoord = hv @ handle_dir_2d
        sign = 1.0 if float(np.median(rcoord)) >= 0.0 else -1.0
        rcoord = sign * rcoord
        r_inner = float(np.percentile(rcoord, 40.0))
        r_outer = float(np.percentile(rcoord, 85.0))
        if np.isfinite(r_outer) and r_outer > 0.0:
            handle_radius = max(handle_radius, r_outer)
            inner_scale = np.clip(r_inner / handle_radius, 0.78, 0.96)
            outer_scale = np.clip(r_outer / handle_radius, 0.90, 1.02)
            radial_scales = sorted({float(inner_scale), float(outer_scale)}, reverse=True)

        hz = vlm_handle_points[:, 2]
        hz_q = np.percentile(hz, [20.0, 40.0, 60.0, 80.0]).tolist()
        handle_z_levels = [
            max(z_min + 0.28 * z_span, min(z_min + 0.64 * z_span, float(v)))
            for v in hz_q
        ]
        used_vlm_handle_hint = True

    handle_points: list[list[float]] = []
    body_points: list[list[float]] = []
    topdown_points: list[list[float]] = []

    # Handle strip: cover full handle arc (outer+inner bands, multiple arc samples).
    tangential = body_dir_2d
    for zz in handle_z_levels:
        for radial_scale in radial_scales:
            for tangent_offset in tangent_offsets:
                xy_pos = center[:2] + handle_dir_2d * (handle_radius * radial_scale) + tangent_offset * tangential
                pos = np.array([xy_pos[0], xy_pos[1], float(zz)], dtype=np.float64)
                # Handle approach points outward to avoid gripper body scraping cup wall first.
                approach = pos - center
                approach[2] = 0.0
                quat = _quat_from_approach_and_finger(approach, np.array([0.0, 0.0, 1.0], dtype=np.float64))
                handle_points.append([float(pos[0]), float(pos[1]), float(pos[2])] + quat)

    # Body side: symmetric left/right sides away from handle axis.
    for zz in body_z_levels:
        for side in (-1.0, 1.0):
            pos = center + side * body_radius * body_dir
            pos[2] = float(zz)
            # Same convention as handle: make pose z-axis point inward to object,
            # so collector pre-grasp offset lands outside.
            approach = pos - center
            approach[2] = 0.0
            quat = _quat_from_approach_and_finger(approach, np.array([0.0, 0.0, 1.0], dtype=np.float64))
            body_points.append([float(pos[0]), float(pos[1]), float(pos[2])] + quat)
            # Add a second nearby sample for denser side coverage.
            pos2 = pos.copy()
            pos2[2] = float(zz + 0.04 * z_span)
            body_points.append([float(pos2[0]), float(pos2[1]), float(pos2[2])] + quat)

    # Topdown side-wall fallback: from above on the same symmetric side pair.
    for zz in topdown_z_levels:
        for side in (-1.0, 1.0):
            pos = center + side * body_radius * body_dir
            pos[2] = float(zz)
            topdown_points.append([float(pos[0]), float(pos[1]), float(pos[2])] + list(TOP_DOWN_QUAT_WXYZ))

    grasp_body = handle_points + body_points + topdown_points

    return {
        "functional_grasp": {
            "handle": handle_points,
            "body": body_points,
            "topdown_side": topdown_points,
        },
        "grasp": {"body": grasp_body},
        "metadata": {
            "source": "manual_mug_template_v3",
            "quaternion_order": "wxyz",
            "position_unit": "mesh_local_units",
            "num_total": len(grasp_body),
            "num_handle": len(handle_points),
            "num_body_side": len(body_points),
            "num_top_down": len(topdown_points),
            "geometry": {
                "center": [float(center[0]), float(center[1]), float(center[2])],
                "z_min": z_min,
                "z_max": z_max,
                "body_radius": body_radius,
                "handle_radius": handle_radius,
                "widest_body_z": widest_z,
                "handle_dir_xy": [float(handle_dir_2d[0]), float(handle_dir_2d[1])],
                "body_dir_xy": [float(body_dir_2d[0]), float(body_dir_2d[1])],
                "used_vlm_handle_hint": bool(used_vlm_handle_hint),
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate manual mug grasp annotation.")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("/tmp/SM_Mug_C1_extracted.obj"),
        help="Mug mesh path (OBJ/GLB/GLTF).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "grasp_poses" / "mug_grasp_pose.json",
        help="Output grasp pose JSON path.",
    )
    parser.add_argument(
        "--also-write",
        type=Path,
        default=None,
        help="Optional second output path.",
    )
    parser.add_argument(
        "--vlm-json",
        type=Path,
        default=None,
        help="Optional raw VLM grasp_pose.json used only as handle-region hint.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.mesh.exists():
        raise FileNotFoundError(f"Mesh not found: {args.mesh}")

    mesh = trimesh.load(str(args.mesh))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh type: {type(mesh)}")

    vlm_handle_points = _load_vlm_handle_points(args.vlm_json)
    out = _build_manual_mug_pose(mesh, vlm_handle_points=vlm_handle_points)

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
        "handle=",
        md.get("num_handle"),
        "body=",
        md.get("num_body_side"),
        "topdown=",
        md.get("num_top_down"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
