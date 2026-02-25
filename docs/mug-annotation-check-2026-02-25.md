# Mug Annotation Check (2026-02-25)

## Goal
- Verify whether the mug created by `create_mug(...)` has matching grasp annotation.
- Run our VLM grasp annotator on the corresponding mug asset and inspect visualization.
- Compare with MagicSim `grasp_pose.json` format/usage.

## What `create_mug(...)` loads
- Runtime helper: `sim-service/run_interactive.py` `_runtime_create_mug(...)`
- Asset path candidates include:
  - `.../Isaac/Props/Mugs/SM_Mug_C1.usd` (and A2/B1/D1 variants)
- Therefore, the runtime mug is Isaac `SM_Mug_*`, not the test `coffee_mug.gltf`.

## Current repo annotation status
- File in collector search path: `sim-service/grasp_poses/mug_grasp_pose.json`
- It exists and is valid, but metadata indicates:
  - `mesh_path: coffee_mug.gltf`
  - `functional_grasp: {}`
  - `grasp.body: 54`
- Conclusion: there is an annotation file, but it is not guaranteed to match `SM_Mug_C1` geometry.

## VLM run on `SM_Mug_C1`
- Source USD fetched from Omniverse asset URL used by runtime.
- USD converted to USDA (`usdcat`), mesh extracted to OBJ:
  - `/tmp/SM_Mug_C1_extracted.obj`
- VLM command:
  - `_wt_a142/tools/vlm_grasp_annotator.py /tmp/SM_Mug_C1_extracted.obj --vlm-url http://magics-hgx.cs.northwestern.edu:30096/v1 --vlm-model /data/models/Qwen/Qwen3-VL-32B-Instruct --critique-rounds 2`
- Output:
  - `functional_grasp`: `handle` + `body`
  - total grasps: 20 (`guided=16`, `top_down=4`)
  - critic final score: 8/10

## MagicSim generator comparison
- Baseline run:
  - `MagicSim/tools/generate_grasp_poses.py /tmp/SM_Mug_C1_extracted.obj ...`
- Output:
  - `functional_grasp: {}`
  - `grasp.body: 4` (top-down only, antipodal=0 on this extracted mesh)
- Interpretation:
  - Format is compatible, but simple antipodal generator is weaker for this mug mesh than VLM-guided labeling.

## Artifacts
- `sim-service/docs/artifacts/mug-vlm-2026-02-25/mug_c1_vlm_grasp_pose.json`
- `sim-service/docs/artifacts/mug-vlm-2026-02-25/mug_c1_magicsim_antipodal.json`
- `sim-service/docs/artifacts/mug-vlm-2026-02-25/mug_c1_views_sheet.png`
- `sim-service/docs/artifacts/mug-vlm-2026-02-25/mug_c1_grasp_overlay_sheet.png`

## Format notes
- Collector and MagicSim both consume pose as:
  - `[x, y, z, qw, qx, qy, qz]` (wxyz quaternion order).
- `functional_grasp.handle` is preferred by MagicSim `Grasp` skill, then fallback to `grasp.body`.
