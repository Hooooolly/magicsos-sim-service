# Mug Grasp Symmetry Update (2026-02-25)

## Goal
Rebuild mug grasp annotation to match priority:
1. Handle symmetric strip first (entire handle line can be pinched)
2. Body symmetric side pinch second (two symmetric sides on cup wall)
3. Top-down side-wall pinch fallback

## Code changes
- `/Users/holly/Documents/code/sim/sim-service/mug_grasp_pose_postprocess.py`
  - New postprocess script to convert raw VLM result into symmetry-priority mug annotation.
  - Outputs:
    - `functional_grasp.handle`
    - `functional_grasp.body`
    - `functional_grasp.topdown_side`
    - `grasp.body` (merged ordered list)

- `/Users/holly/Documents/code/sim/sim-service/isaac_pick_place_collector.py`
  - Fixed annotation grouping logic:
    - `functional_grasp.handle` only -> handle priority group
    - body labels -> body group
    - `topdown/top_down` labels -> topdown group (last fallback)
  - Prevented all `functional_grasp.*` labels from being treated as handle.
  - Added annotation import guardrails:
    - metadata parsing for `quaternion_order` (`wxyz` / `xyzw`) and `position_scale_hint`
    - strict local-bbox/radius validation after scale inference
    - reject annotation file when valid keep ratio is too low (configurable by env)

## Generated annotation
- Main runtime file:
  - `/Users/holly/Documents/code/sim/sim-service/grasp_poses/mug_grasp_pose.json`
- Artifact copy:
  - `/Users/holly/Documents/code/sim/sim-service/docs/artifacts/mug-vlm-2026-02-25/mug_c1_vlm_symmetry_v2.json`

Current counts:
- handle: 10
- body side: 12
- topdown: 4
- total merged: 25
- metadata now explicitly includes:
  - `quaternion_order: wxyz`
  - `position_scale_hint: 1.0`
  - `position_unit: mesh_local_units`

## Reproduce command
```bash
cd /Users/holly/Documents/code/sim/sim-service
python3 mug_grasp_pose_postprocess.py \
  --input /tmp/mug_vlm_rerun/mug_c1_vlm_raw.json \
  --output grasp_poses/mug_grasp_pose.json \
  --also-write docs/artifacts/mug-vlm-2026-02-25/mug_c1_vlm_symmetry_v2.json
```

## Validation
```bash
cd /Users/holly/Documents/code/sim/sim-service
python3 -m py_compile isaac_pick_place_collector.py mug_grasp_pose_postprocess.py run_interactive.py
```

## New VLM run for visual inspection (2026-02-25 11:14)
- Run directory:
  - `/Users/holly/Documents/code/sim/sim-service/docs/artifacts/mug-vlm-2026-02-25/run_20260225_111441`
- Inputs:
  - mesh: `/tmp/SM_Mug_C1_extracted.obj`
  - model: `/data/models/Qwen/Qwen3-VL-32B-Instruct`
  - endpoint: `http://magics-hgx.cs.northwestern.edu:30096/v1`
- Outputs:
  - raw VLM result: `grasp_pose.json`
  - symmetry-processed result: `grasp_pose_symmetry_v2.json`
  - comparison images:
    - `1_raw_views.png`
    - `2_raw_overlay_views.png`
    - `3_symmetry_overlay_views.png`
    - `4_summary_raw_vs_symmetry.png`
- Symmetry result counts:
  - handle: 16
  - body: 12
  - topdown_side: 4
  - total: 28

## HGX server run (2026-02-25 11:22)
- Reason: run directly on server (not local), and build 3 preference candidates.
- Server output root:
  - `/data/models/grasp_pref_mug_20260225_112210`
- Base VLM output:
  - `/data/models/grasp_pref_mug_20260225_112210/base/grasp_pose.json`
- Symmetry output:
  - `/data/models/grasp_pref_mug_20260225_112210/sym/grasp_pose_symmetry_v2.json`
- Preference candidates:
  - `/data/models/grasp_pref_mug_20260225_112210/preference/handle_only.json`
  - `/data/models/grasp_pref_mug_20260225_112210/preference/body_symmetric_only.json`
  - `/data/models/grasp_pref_mug_20260225_112210/preference/topdown_wall_only.json`
- Preference visualizations:
  - `/data/models/grasp_pref_mug_20260225_112210/preference/handle_only_overlay.png`
  - `/data/models/grasp_pref_mug_20260225_112210/preference/body_symmetric_only_overlay.png`
  - `/data/models/grasp_pref_mug_20260225_112210/preference/topdown_wall_only_overlay.png`
  - `/data/models/grasp_pref_mug_20260225_112210/preference/preference_summary.png`
- Manifest:
  - `/data/models/grasp_pref_mug_20260225_112210/preference/manifest.json`
  - counts: handle_only=16, body_symmetric_only=12, topdown_wall_only=4

## Server-side grasp format check (MagicSim/Omni style)
- On HGX, current available grasp JSONs are under `/data/models/grasp_test_v*/grasp_pose.json` and new `grasp_pref_mug_*`.
- Sample file inspected:
  - `/data/models/grasp_test_v8/grasp_pose.json`
- Structure:
  - top-level keys: `functional_grasp`, `grasp`, `metadata`
  - `functional_grasp` labels: e.g. `handle`, `body`
  - `grasp.body`: list of 7D pose arrays `[x, y, z, qw, qx, qy, qz]`

## Server-side geometric sanity check
- Mesh: `/tmp/SM_Mug_C1_extracted.obj` (HGX)
- Validation window:
  - `max_pos_radius_factor=2.8`
  - `bbox_margin_factor=0.25`
- Results:
  - symmetry output `/data/models/grasp_pref_mug_20260225_112210/sym/grasp_pose_symmetry_v2.json`
    - keep_ratio = `1.0` (30/30 valid)
  - raw base output `/data/models/grasp_pref_mug_20260225_112210/base/grasp_pose.json`
    - keep_ratio = `0.793` (23/29 valid)

## HGX iterative reruns with image-based review (2026-02-25 12:00~12:06)
- Reason: tighten geometric partition for `handle/body/topdown` and validate by overlay images.
- Code updated:
  - `/Users/holly/Documents/code/sim/sim-service/mug_grasp_pose_postprocess.py`
  - Added stricter handle-vs-body geometric gating (axis/radius thresholds).
  - Added explicit body-axis mirroring (separate from handle-axis).
  - Added mesh-size lower bound for body/topdown side radius.
- Server outputs:
  - `/data/models/grasp_pref_mug_20260225_115958_tight`
  - `/data/models/grasp_pref_mug_20260225_120331_tight2`
  - `/data/models/grasp_pref_mug_20260225_120606_tight3`
- Local pulled artifacts (for quick inspection):
  - `/Users/holly/Documents/code/sim/sim-service/docs/artifacts/server-mug-pref-20260225_115958_tight/`
  - `/Users/holly/Documents/code/sim/sim-service/docs/artifacts/server-mug-pref-20260225_120331_tight2/`
  - `/Users/holly/Documents/code/sim/sim-service/docs/artifacts/server-mug-pref-20260225_120606_tight3/`
- Final (`tight3`) counts:
  - handle_only=16
  - body_symmetric_only=8
  - topdown_wall_only=4
- Image review verdict (`tight3`):
  - `topdown_wall_only`: usable as fallback (points are stable and no collapse).
  - `handle_only`: still has spillover toward cup body in several views.
  - `body_symmetric_only`: still biased near handle-side region; not yet ideal bilateral body pinch coverage.

## Manual annotation pass (non-VLM, 2026-02-25 12:15~12:18)
- Request: stop using VLM outputs for mug and annotate by deterministic geometry.
- New generator script:
  - `/Users/holly/Documents/code/sim/sim-service/mug_grasp_pose_manual.py`
  - Method:
    - detect handle direction from XY radial outliers,
    - generate handle strip points along handle direction,
    - generate body points on two symmetric side walls (perpendicular to handle),
    - generate top-down side-wall fallback points.
- HGX outputs:
  - v1: `/data/models/mug_manual_20260225_121546_v1`
  - v2: `/data/models/mug_manual_20260225_121741_v2`
- Local pulled artifacts:
  - `/Users/holly/Documents/code/sim/sim-service/docs/artifacts/mug-manual-20260225_121741_v2/`
  - key files:
    - `handle_only_overlay.png`
    - `body_symmetric_only_overlay.png`
    - `topdown_wall_only_overlay.png`
    - `preference_summary.png`
    - `grasp_pose_manual.json`
- Final accepted set (v2):
  - handle=8, body=8, topdown=4, total=20
  - updated runtime file:
    - `/Users/holly/Documents/code/sim/sim-service/grasp_poses/mug_grasp_pose.json`
  - metadata source: `manual_mug_template_v1`

## Manual tuning (v3/v4, 2026-02-25 12:23~12:28)
- User feedback incorporated:
  - body points should be around the widest body band (middle-upper), not rim-only.
  - handle points should cover the full handle arc.
  - handle approach should point outward to reduce cup-wall pre-collision.
- Changes in `/Users/holly/Documents/code/sim/sim-service/mug_grasp_pose_manual.py`:
  - Added widest-band estimator with z-band constraints.
  - Expanded handle arc sampling from 2 offsets to 3 offsets.
  - Switched handle approach direction to outward (`pos - center`).
- Artifacts:
  - v3: `/data/models/mug_manual_20260225_122335_v3`
  - v4: `/data/models/mug_manual_20260225_122701_v4`
  - local:
    - `/Users/holly/Documents/code/sim/sim-service/docs/artifacts/mug-manual-20260225_122701_v4/`
- v4 counts:
  - handle=12
  - body=8
  - topdown=4
  - total=24
- Runtime file switched to v4:
  - `/Users/holly/Documents/code/sim/sim-service/grasp_poses/mug_grasp_pose.json`
  - metadata source: `manual_mug_template_v3`
