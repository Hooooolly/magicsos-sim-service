#!/usr/bin/env python3
"""Post-process mug VLM grasps into symmetry-priority annotation.

Priority rules:
1) functional_grasp.handle: handle strip first (with symmetric orientation variants)
2) functional_grasp.body: side pinches on symmetric body pairs
3) functional_grasp.topdown_side: top-down side-wall pinch fallback
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


TOP_DOWN_QUAT_WXYZ = [0.0, 1.0, 0.0, 0.0]
YAW_180_QUAT_WXYZ = [0.0, 0.0, 0.0, 1.0]

HANDLE_MIN_Z_FRAC = 0.18
BODY_MIN_Z_FRAC = 0.35
BODY_MAX_Z_FRAC = 0.85
TOPDOWN_LOW_Z_FRAC = 0.55
TOPDOWN_HIGH_Z_FRAC = 0.70
HANDLE_OFFSET_MIN_Q = 0.45
BODY_OFFSET_CORE_Q = 0.60
BODY_OFFSET_SIDE_Q = 0.72
HANDLE_SIDE_BODY_GAP_FRAC = 0.08
HANDLE_RADIUS_MIN_Q = 0.45
BODY_RADIUS_MAX_Q = 0.85
BODY_HANDLE_CLEARANCE_FRAC = 0.04


def _is_finite_list(values: list[float]) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def _normalize_quat_wxyz(q: list[float]) -> list[float]:
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if not math.isfinite(n) or n < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    out = [float(v) / n for v in q]
    if out[0] < 0.0:
        out = [-v for v in out]
    return out


def _quat_mul_wxyz(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _yaw_flip_quat_wxyz(q: list[float]) -> list[float]:
    return _normalize_quat_wxyz(_quat_mul_wxyz(YAW_180_QUAT_WXYZ, q))


def _safe_median(values: list[float], fallback: float = 0.0) -> float:
    if not values:
        return float(fallback)
    try:
        return float(statistics.median(values))
    except Exception:
        return float(fallback)


def _safe_percentile(values: list[float], q: float, fallback: float = 0.0) -> float:
    if not values:
        return float(fallback)
    qq = max(0.0, min(100.0, float(q)))
    ordered = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not ordered:
        return float(fallback)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (qq / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    t = rank - lo
    return float(ordered[lo] * (1.0 - t) + ordered[hi] * t)


def _quantile_pick_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [n // 2]
    out: list[int] = []
    used = set()
    for i in range(k):
        idx = int(round(i * (n - 1) / (k - 1)))
        idx = max(0, min(n - 1, idx))
        if idx not in used:
            used.add(idx)
            out.append(idx)
    return sorted(out)


def _dedupe_poses(poses: list[list[float]], ndigits: int = 5) -> list[list[float]]:
    out: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    for pose in poses:
        if len(pose) < 7:
            continue
        pos = [float(v) for v in pose[:3]]
        quat = _normalize_quat_wxyz([float(v) for v in pose[3:7]])
        packed = pos + quat
        key = tuple(round(v, ndigits) for v in packed)
        if key in seen:
            continue
        seen.add(key)
        out.append(packed)
    return out


def _to_pose_list(raw_poses: Any) -> list[list[float]]:
    out: list[list[float]] = []
    if not isinstance(raw_poses, list):
        return out
    for p in raw_poses:
        if not isinstance(p, list) or len(p) < 7:
            continue
        arr = [float(v) for v in p[:7]]
        if not _is_finite_list(arr):
            continue
        arr[3:7] = _normalize_quat_wxyz(arr[3:7])
        out.append(arr)
    return out


def _sample_by_z(poses: list[list[float]], count: int) -> list[list[float]]:
    if not poses or count <= 0:
        return []
    ordered = sorted(poses, key=lambda p: float(p[2]))
    indices = _quantile_pick_indices(len(ordered), count)
    return [ordered[i] for i in indices]


def _axis_offset(pose: list[float], axis: int, center: float) -> float:
    return float(pose[axis]) - float(center)


def _radial_xy(pose: list[float], cx: float, cy: float) -> float:
    dx = float(pose[0]) - float(cx)
    dy = float(pose[1]) - float(cy)
    return math.sqrt(dx * dx + dy * dy)


def _build_symmetric_mug_pose(raw: dict[str, Any]) -> dict[str, Any]:
    functional = raw.get("functional_grasp", {})
    grasp = raw.get("grasp", {})
    handle_raw = _to_pose_list(functional.get("handle", []))
    rim_raw = _to_pose_list(functional.get("rim", []))
    body_raw = _to_pose_list(functional.get("body", []))
    grasp_body_raw = _to_pose_list(grasp.get("body", []))

    # Prefer semantic body/rim labels. The merged grasp.body bucket often
    # contains handle-biased points and should be fallback-only.
    body_pool = body_raw + rim_raw
    if len(body_pool) < 6:
        body_pool = body_pool + grasp_body_raw
    all_pool = handle_raw + body_pool
    if not all_pool:
        raise RuntimeError("No valid grasp poses found in input JSON")

    xs = [p[0] for p in all_pool]
    ys = [p[1] for p in all_pool]
    zs = [p[2] for p in all_pool]
    cx = _safe_median(xs)
    cy = _safe_median(ys)
    z_min = min(zs)
    z_max = max(zs)
    z_span = max(1e-6, z_max - z_min)
    handle_zs = [p[2] for p in handle_raw]
    body_zs = [p[2] for p in body_pool]

    # Pick mirror axis from handle displacement (fallback: x).
    if handle_raw:
        hx = [abs(p[0] - cx) for p in handle_raw]
        hy = [abs(p[1] - cy) for p in handle_raw]
        mirror_axis = 0 if _safe_median(hx) >= _safe_median(hy) else 1
    else:
        mirror_axis = 0
    mirror_center = cx if mirror_axis == 0 else cy
    body_axis = 1 if mirror_axis == 0 else 0
    body_center = cx if body_axis == 0 else cy
    axis_values = [float(p[mirror_axis]) for p in all_pool]
    axis_extent = max(1e-6, max(axis_values) - min(axis_values))
    body_axis_values = [float(p[body_axis]) for p in all_pool]
    body_axis_extent = max(1e-6, max(body_axis_values) - min(body_axis_values))
    radial_extent = max(1e-6, max(max(xs) - min(xs), max(ys) - min(ys)))

    handle_floor = max(
        z_min + HANDLE_MIN_Z_FRAC * z_span,
        _safe_percentile(handle_zs, 35.0, fallback=z_min),
    )
    body_floor = max(
        z_min + BODY_MIN_Z_FRAC * z_span,
        _safe_percentile(body_zs, 45.0, fallback=z_min),
    )
    body_ceil = min(
        z_min + BODY_MAX_Z_FRAC * z_span,
        _safe_percentile(body_zs, 90.0, fallback=z_max),
    )
    topdown_z = [
        _safe_percentile(body_zs, 65.0, fallback=z_min + TOPDOWN_LOW_Z_FRAC * z_span),
        _safe_percentile(body_zs, 80.0, fallback=z_min + TOPDOWN_HIGH_Z_FRAC * z_span),
    ]

    handle_offsets = [_axis_offset(p, mirror_axis, mirror_center) for p in handle_raw]
    body_offsets = [_axis_offset(p, body_axis, body_center) for p in body_pool]
    body_handle_axis_offsets = [_axis_offset(p, mirror_axis, mirror_center) for p in body_pool]
    handle_abs = [abs(v) for v in handle_offsets]
    body_abs = [abs(v) for v in body_offsets]
    handle_sign = 1.0
    if handle_offsets:
        m = _safe_median(handle_offsets, fallback=0.0)
        if abs(m) > 1e-8:
            handle_sign = 1.0 if m >= 0.0 else -1.0
        else:
            max_v = max(handle_offsets, key=lambda x: abs(x))
            handle_sign = 1.0 if max_v >= 0.0 else -1.0
    handle_side_offsets = [handle_sign * v for v in handle_offsets]
    body_handle_side_offsets = [handle_sign * v for v in body_handle_axis_offsets]

    handle_offset_min = _safe_percentile(handle_abs, HANDLE_OFFSET_MIN_Q * 100.0, fallback=0.0)
    body_offset_core = _safe_percentile(body_abs, BODY_OFFSET_CORE_Q * 100.0, fallback=0.0)
    body_offset_side = _safe_percentile(body_abs, BODY_OFFSET_SIDE_Q * 100.0, fallback=0.0)
    body_side_hi = _safe_percentile(body_handle_side_offsets, 85.0, fallback=0.0)
    handle_side_lo = _safe_percentile(
        [v for v in handle_side_offsets if math.isfinite(v)],
        35.0,
        fallback=0.0,
    )
    handle_offset_min = max(
        handle_offset_min,
        handle_side_lo,
        body_side_hi + HANDLE_SIDE_BODY_GAP_FRAC * axis_extent,
        body_offset_core + 0.05 * axis_extent,
    )
    body_handle_exclusion = min(
        _safe_percentile([abs(v) for v in body_handle_axis_offsets], BODY_OFFSET_SIDE_Q * 100.0, fallback=axis_extent),
        0.55 * handle_offset_min,
        handle_offset_min - BODY_HANDLE_CLEARANCE_FRAC * axis_extent,
    )
    if body_handle_exclusion <= 0.0:
        body_handle_exclusion = max(0.0, 0.75 * handle_offset_min)

    handle_radii = [_radial_xy(p, cx, cy) for p in handle_raw]
    body_radii = [_radial_xy(p, cx, cy) for p in body_pool]
    handle_radius_min = max(
        _safe_percentile(handle_radii, HANDLE_RADIUS_MIN_Q * 100.0, fallback=0.0),
        _safe_percentile(body_radii, 80.0, fallback=0.0) + 0.02 * radial_extent,
    )
    body_radius_cap = _safe_percentile(body_radii, BODY_RADIUS_MAX_Q * 100.0, fallback=radial_extent)

    # 1) Handle strip: keep along vertical strip, add symmetric orientation variant.
    handle_filtered = [
        p
        for p in handle_raw
        if p[2] >= handle_floor
        and (handle_sign * _axis_offset(p, mirror_axis, mirror_center)) >= handle_offset_min
        and _radial_xy(p, cx, cy) >= handle_radius_min
    ]
    if len(handle_filtered) < 4:
        handle_filtered = [
            p
            for p in handle_raw
            if p[2] >= handle_floor
            and (handle_sign * _axis_offset(p, mirror_axis, mirror_center)) > 0.0
            and _radial_xy(p, cx, cy) >= (0.90 * handle_radius_min)
        ]
    if len(handle_filtered) < 4:
        handle_filtered = list(handle_raw)
    base_handle = _sample_by_z(handle_filtered, min(8, max(4, len(handle_filtered))))
    handle_out: list[list[float]] = []
    for p in base_handle:
        pos = p[:3]
        quat = p[3:7]
        handle_out.append(pos + quat)
        handle_out.append(pos + _yaw_flip_quat_wxyz(quat))
    handle_out = _dedupe_poses(handle_out)

    # 2) Body side pinch: symmetric pair around mug centerline.
    body_filtered = [
        p
        for p in body_pool
        if body_floor <= p[2] <= body_ceil
        and abs(_axis_offset(p, body_axis, body_center)) <= body_offset_side
        and (handle_sign * _axis_offset(p, mirror_axis, mirror_center)) <= body_handle_exclusion
        and _radial_xy(p, cx, cy) <= body_radius_cap
    ]
    if len(body_filtered) < 6:
        body_filtered = [
            p
            for p in body_pool
            if p[2] >= body_floor
            and (handle_sign * _axis_offset(p, mirror_axis, mirror_center)) < handle_offset_min
            and abs(_axis_offset(p, body_axis, body_center)) <= body_offset_side
            and _radial_xy(p, cx, cy) <= body_radius_cap
        ]
    if len(body_filtered) < 4:
        body_filtered = [
            p
            for p in body_pool
            if body_floor <= p[2] <= body_ceil
            and _radial_xy(p, cx, cy) <= body_radius_cap
        ]
    if len(body_filtered) < 4:
        body_filtered = [
            p
            for p in body_pool
            if _radial_xy(p, cx, cy) <= body_radius_cap
        ]
    if len(body_filtered) < 4:
        body_filtered = list(body_pool)

    # Hard de-bias pass: remove points too close to the handle centroid in XY.
    if handle_raw and len(body_filtered) >= 4:
        hx = _safe_median([p[0] for p in handle_raw], fallback=cx)
        hy = _safe_median([p[1] for p in handle_raw], fallback=cy)
        handle_dist = [math.hypot(float(p[0]) - hx, float(p[1]) - hy) for p in body_filtered]
        handle_dist = [d for d in handle_dist if math.isfinite(d)]
        if handle_dist:
            handle_keep_min = max(_safe_percentile(handle_dist, 45.0, fallback=0.0), 0.12 * radial_extent)
            body_filtered = [
                p
                for p in body_filtered
                if math.hypot(float(p[0]) - hx, float(p[1]) - hy) >= handle_keep_min
            ] or body_filtered
    body_seed = _sample_by_z(body_filtered, min(4, max(3, len(body_filtered))))
    body_side_candidates = [
        abs(_axis_offset(p, body_axis, body_center))
        for p in body_filtered
        if body_floor <= p[2] <= body_ceil
    ]
    body_side_candidates = [v for v in body_side_candidates if math.isfinite(v)]
    body_side_radius = max(
        _safe_percentile(body_side_candidates, 70.0, fallback=0.0),
        0.18 * body_axis_extent,
    )
    body_side_radius = min(body_side_radius, 0.48 * body_axis_extent)
    body_side_radius = max(body_side_radius, 1e-5)
    body_out: list[list[float]] = []
    for p in body_seed:
        pos = [float(p[0]), float(p[1]), float(p[2])]
        quat = p[3:7]
        pos_pos = list(pos)
        pos_neg = list(pos)
        pos_pos[body_axis] = body_center + body_side_radius
        pos_neg[body_axis] = body_center - body_side_radius
        body_out.append(pos_pos + quat)
        body_out.append(pos_neg + _yaw_flip_quat_wxyz(quat))
    body_out = _dedupe_poses(body_out)

    # 3) Top-down side-wall fallback.
    side_offsets = [
        abs(_axis_offset(p, body_axis, body_center))
        for p in body_filtered
        if body_floor <= p[2] <= body_ceil
    ]
    side_offsets = [v for v in side_offsets if math.isfinite(v)]
    if side_offsets:
        ordered_offsets = sorted(side_offsets)
        idx = int(round(0.65 * (len(ordered_offsets) - 1)))
        side_radius = max(0.0, float(ordered_offsets[max(0, min(len(ordered_offsets) - 1, idx))]))
    else:
        side_radius = float(body_offset_core)
    side_radius = max(side_radius, 0.95 * body_side_radius, 0.18 * body_axis_extent)
    if side_radius < 1e-6:
        extent = max(max(xs) - min(xs), max(ys) - min(ys))
        side_radius = 0.20 * float(extent)
    side_radius = max(side_radius, 1e-5)

    topdown_out: list[list[float]] = []
    for z in topdown_z:
        for sign in (-1.0, 1.0):
            pos = [cx, cy, float(z)]
            pos[body_axis] = body_center + sign * side_radius
            topdown_out.append(pos + list(TOP_DOWN_QUAT_WXYZ))
    topdown_out = _dedupe_poses(topdown_out)

    grasp_body = _dedupe_poses(handle_out + body_out + topdown_out)

    metadata = dict(raw.get("metadata", {}))
    metadata.update(
        {
            "source": f"{metadata.get('source', 'vlm')}_mug_symmetry_postprocess_v2",
            "quaternion_order": "wxyz",
            "position_scale_hint": 1.0,
            "position_unit": "mesh_local_units",
            "num_total": len(grasp_body),
            "num_handle": len(handle_out),
            "num_body_side": len(body_out),
            "num_top_down": len(topdown_out),
            "priority_order": [
                "functional_grasp.handle",
                "functional_grasp.body",
                "functional_grasp.topdown_side",
            ],
            "symmetry_postprocess": {
                "version": "mug_symmetry_v2",
                "mirror_axis": "x" if mirror_axis == 0 else "y",
                "mirror_center": mirror_center,
                "body_axis": "x" if body_axis == 0 else "y",
                "body_center": body_center,
                "z_filters": {
                    "handle_min_frac": HANDLE_MIN_Z_FRAC,
                    "body_min_frac": BODY_MIN_Z_FRAC,
                    "body_max_frac": BODY_MAX_Z_FRAC,
                    "topdown_fracs": [TOPDOWN_LOW_Z_FRAC, TOPDOWN_HIGH_Z_FRAC],
                    "handle_offset_min_q": HANDLE_OFFSET_MIN_Q,
                    "body_offset_core_q": BODY_OFFSET_CORE_Q,
                    "body_offset_side_q": BODY_OFFSET_SIDE_Q,
                },
                "offset_partition": {
                    "handle_sign": float(handle_sign),
                    "handle_offset_min": float(handle_offset_min),
                    "body_offset_core": float(body_offset_core),
                    "body_offset_side": float(body_offset_side),
                    "body_handle_exclusion": float(body_handle_exclusion),
                    "handle_radius_min": float(handle_radius_min),
                    "body_radius_cap": float(body_radius_cap),
                    "body_side_radius": float(body_side_radius),
                    "body_axis_extent": float(body_axis_extent),
                },
            },
        }
    )

    return {
        "functional_grasp": {
            "handle": handle_out,
            "body": body_out,
            "topdown_side": topdown_out,
        },
        "grasp": {"body": grasp_body},
        "metadata": metadata,
    }


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build symmetry-priority mug grasp pose JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/tmp/mug_vlm_rerun/mug_c1_vlm_raw.json"),
        help="Raw VLM grasp JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "grasp_poses" / "mug_grasp_pose.json",
        help="Output mug grasp pose JSON path.",
    )
    parser.add_argument(
        "--also-write",
        type=Path,
        default=None,
        help="Optional second output path (artifact copy).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    with args.input.open("r") as f:
        raw = json.load(f)

    out = _build_symmetric_mug_pose(raw)

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
