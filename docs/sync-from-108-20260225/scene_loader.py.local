#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None
USD_SUFFIXES = {".usd", ".usda", ".usdc"}
def _clean_float(v: float) -> float:
    return 0.0 if abs(v) < 1e-10 else round(v, 6)
def quat_wxyz_to_euler_degrees(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert Drake quaternion (w,x,y,z) to XYZ Euler degrees."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        raise ValueError("Quaternion has zero norm")
    w, x, y, z = (w / n, x / n, y / n, z / n)
    rx = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    ry = math.asin(sinp)
    rz = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (_clean_float(math.degrees(rx)), _clean_float(math.degrees(ry)), _clean_float(math.degrees(rz)))
def _is_usd(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in USD_SUFFIXES
def _collect_usd_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if _is_usd(p)])
def _score_room_usd(path: Path) -> float:
    lower = path.as_posix().lower()
    name = path.name.lower()
    score = 0.0
    if name in ("scene.usd", "scene.usda", "scene.usdc"):
        score += 100.0
    if "scene" in path.stem.lower():
        score += 20.0
    if "house" in path.stem.lower():
        score += 10.0
    if "/mujoco/usd/" in lower:
        score += 8.0
    if "/payload/" in lower:
        score -= 25.0
    if "materialslibrary" in name:
        score -= 40.0
    return score - len(path.parts) * 0.05
def discover_room_usd(scene_dir: Path) -> Path | None:
    """Bridge-compatible room USD discovery."""
    candidates: list[Path] = []
    for root in [scene_dir / "mujoco" / "usd", scene_dir / "mujoco", scene_dir]:
        candidates.extend(_collect_usd_files(root))
    return max(candidates, key=_score_room_usd) if candidates else None
def _find_named_usd(root: Path, name_prefix: str, recursive: bool) -> list[Path]:
    if not root.exists() or not name_prefix:
        return []
    it = root.rglob("*") if recursive else root.glob("*")
    pref = name_prefix.lower()
    return sorted([p for p in it if _is_usd(p) and p.stem.lower().startswith(pref)])
def find_usd_for_object(
    scene_dir: Path | str,
    obj: dict[str, Any],
    *,
    room_root: Path | None = None,
    scene_state_path: Path | None = None,
) -> str | None:
    """Find object USD using SceneSmith search order; fallback to geometry_path."""
    scene_root = Path(scene_dir).resolve()
    room_root = room_root.resolve() if room_root is not None else None
    scene_state_path = scene_state_path.resolve() if scene_state_path is not None else None
    name = str(obj.get("name") or obj.get("object_id") or "").strip()
    search: list[tuple[Path, bool]] = []
    bases: list[Path] = []
    for base in [room_root, scene_root]:
        if isinstance(base, Path) and base not in bases:
            bases.append(base)
    for base in bases:
        search.extend(
            [
                (base / "mujoco" / "usd" / "Payload", False),
                (base / "mujoco" / "usd", False),
                (base / "generated_assets", True),
            ]
        )
    for root, recursive in search:
        matches = _find_named_usd(root, name, recursive)
        if matches:
            return str(matches[0].resolve())
    geom = obj.get("geometry_path")
    if isinstance(geom, str) and geom.strip():
        p = Path(geom)
        candidates: list[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            if room_root is not None:
                candidates.append(room_root / p)
            if scene_state_path is not None:
                # scene_state.json is usually under room_*/scene_states/final_scene/.
                # Try resolving relative paths against both state dir and room dir.
                state_dir = scene_state_path.parent
                candidates.extend([state_dir / p, state_dir.parent / p, state_dir.parent.parent / p])
            candidates.append(scene_root / p)
        seen: set[str] = set()
        for cand in candidates:
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            if cand.exists():
                return str(cand.resolve())
    return None
def find_room_usd(scene_dir: Path | str) -> str | None:
    scene_root = Path(scene_dir).resolve()
    roots: list[Path] = [scene_root]
    for child in sorted(scene_root.glob("room_*")):
        if child.is_dir():
            roots.append(child.resolve())
    seen: set[str] = set()
    unique_roots: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(root)
    for root in unique_roots:
        preferred = root / "mujoco" / "usd" / "scene.usd"
        if preferred.is_file():
            return str(preferred.resolve())
    for root in unique_roots:
        found = discover_room_usd(root)
        if found:
            return str(found.resolve())
    return None
def _scene_after_idx(path: Path) -> int:
    m = re.search(r"scene_after_(\d+)$", path.name)
    return int(m.group(1)) if m else -1
def _discover_scene_state(scene_root: Path) -> tuple[Path, Path]:
    room_roots: list[Path] = []
    direct_states = scene_root / "scene_states"
    if direct_states.is_dir():
        room_roots.append(scene_root)
    for room in sorted(scene_root.glob("room_*")):
        if room.is_dir() and (room / "scene_states").is_dir():
            room_roots.append(room.resolve())
    if not room_roots:
        raise FileNotFoundError(f"No scene_states directory found under {scene_root} or scene_root/room_*")
    for room_root in room_roots:
        final_state = room_root / "scene_states" / "final_scene" / "scene_state.json"
        if final_state.is_file():
            return final_state, room_root
    after_candidates: list[tuple[int, float, Path, Path]] = []
    for room_root in room_roots:
        states = room_root / "scene_states"
        for after_dir in states.glob("scene_after_*"):
            if not after_dir.is_dir():
                continue
            state_path = after_dir / "scene_state.json"
            if state_path.is_file():
                after_candidates.append((_scene_after_idx(after_dir), state_path.stat().st_mtime, state_path, room_root))
    if after_candidates:
        after_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, _, state_path, room_root = after_candidates[0]
        return state_path, room_root
    searched = ", ".join(str(root / "scene_states") for root in room_roots)
    raise FileNotFoundError(f"No scene_state.json found under: {searched}")
def load_scene_state_with_meta(scene_dir: str) -> tuple[dict[str, Any], Path, Path]:
    scene_root = Path(scene_dir).resolve()
    state_path, room_root = _discover_scene_state(scene_root)
    data = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"scene_state.json at {state_path} is not a JSON object")
    return data, state_path, room_root
def load_scene_state(scene_dir: str) -> dict[str, Any]:
    data, _, _ = load_scene_state_with_meta(scene_dir)
    return data
def physics_type_for_object(obj: dict[str, Any]) -> str:
    kind = str(obj.get("object_type", "")).strip().lower()
    return {
        "furniture": "geometry",
        "wall_mounted": "geometry",
        "ceiling_mounted": "geometry",
        "manipuland": "dynamic",
    }.get(kind, "geometry")
def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "object"
def _unique(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    out = f"{base}_{i}"
    used.add(out)
    return out
def _vec3(v: Any, d: tuple[float, float, float]) -> list[float]:
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return [_clean_float(float(v[0])), _clean_float(float(v[1])), _clean_float(float(v[2]))]
        except Exception:
            pass
    return [_clean_float(d[0]), _clean_float(d[1]), _clean_float(d[2])]
def _scale3(v: Any) -> list[float]:
    if isinstance(v, (list, tuple)):
        return _vec3(v, (1.0, 1.0, 1.0))
    try:
        s = _clean_float(float(v))
    except Exception:
        s = 1.0
    return [s, s, s]
def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(str(v))
def _manual_yaml_dump(v: Any, indent: int = 0) -> str:
    sp = " " * indent
    if isinstance(v, dict):
        lines: list[str] = []
        for k, item in v.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}{k}:")
                lines.append(_manual_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{sp}{k}: {_yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(v, list):
        if not v:
            return f"{sp}[]"
        lines = []
        for item in v:
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(_manual_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{sp}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{sp}{_yaml_scalar(v)}"
def generate_magicsim_yaml(scene_dir: str, output_path: str | None = None) -> str:
    scene_root = Path(scene_dir).resolve()
    scene_state, state_path, room_root = load_scene_state_with_meta(str(scene_root))
    if os.environ.get("SCENE_LOADER_DEBUG", "").lower() not in ("", "0", "false", "no"):
        print(f"[scene_loader] debug: scene_root={scene_root}")
        print(f"[scene_loader] debug: room_root={room_root}")
        print(f"[scene_loader] debug: scene_state={state_path}")
    room_usd = find_room_usd(scene_root)
    doc: dict[str, Any] = {"objects": {}}
    if room_usd:
        doc["room"] = {
            "scenesmith_room": {
                "usd": [str(Path(room_usd).resolve())],
                "random": False,
                "scenesmith_room_1": {"pos": [0.0, 0.0, 0.0], "ori": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
            }
        }
    else:
        print(f"[scene_loader] warning: no room USD found in {scene_root}, skipping room section")
    objects = scene_state.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError("scene_state.json field 'objects' must be a dictionary")
    used: set[str] = set()
    for object_id, obj in objects.items():
        if not isinstance(obj, dict):
            continue
        usd_path = find_usd_for_object(scene_root, obj, room_root=room_root, scene_state_path=state_path)
        if usd_path is None:
            print(f"[scene_loader] warning: skipping '{obj.get('name') or object_id}' (no USD/geometry path found)")
            continue
        if Path(usd_path).suffix.lower() not in USD_SUFFIXES:
            print(
                f"[scene_loader] warning: '{obj.get('name') or object_id}' resolved to non-USD asset: {usd_path}"
            )
        transform = obj.get("transform") if isinstance(obj.get("transform"), dict) else {}
        pos = _vec3(transform.get("translation"), (0.0, 0.0, 0.0))
        quat = transform.get("rotation_wxyz")
        qw, qx, qy, qz = (1.0, 0.0, 0.0, 0.0)
        if isinstance(quat, (list, tuple)) and len(quat) >= 4:
            qw, qx, qy, qz = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        ori = list(quat_wxyz_to_euler_degrees(qw, qx, qy, qz))
        key = _unique(_slug(str(obj.get("name") or object_id)), used)
        inst = f"{key}_1"
        doc["objects"][key] = {
            "usd": [str(Path(usd_path).resolve())],
            "random": False,
            "num_per_env": 1,
            inst: {
                "pos": pos,
                "ori": ori,
                "visual": {"scale": _scale3(obj.get("scale_factor", 1.0))},
                "physics": {"type": physics_type_for_object(obj), "collision": True},
            },
        }
    yaml_text = yaml.dump(doc, sort_keys=False) if yaml is not None else _manual_yaml_dump(doc) + "\n"
    if output_path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml_text, encoding="utf-8")
    return yaml_text
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MagicSim YAML from SceneSmith scene_state.json")
    p.add_argument("--scene-dir", required=True, type=str, help="SceneSmith scene directory")
    p.add_argument("--output", type=str, default=None, help="Output YAML path")
    return p.parse_args()
if __name__ == "__main__":
    args = _parse_args()
    print(generate_magicsim_yaml(args.scene_dir, args.output))
