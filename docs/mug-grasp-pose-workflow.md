# Mug Grasp Pose Workflow (VLM -> Collector -> Per-frame GT)

Current workspace already contains a generated mug grasp file:
- `/Users/holly/Documents/code/sim/sim-service/grasp_poses/mug_grasp_pose.json`

## 1) Generate mug grasp_pose.json with VLM

Use the existing VLM annotator script (current workspace path):

```bash
python /Users/holly/Documents/code/sim/_wt_a142/tools/vlm_grasp_annotator.py \
  /path/to/mug_mesh.glb \
  --vlm-url http://<vlm-host>:<port>/v1 \
  --vlm-model /data/models/Qwen/Qwen3-VL-32B-Instruct \
  --api-key empty \
  -o /tmp/mug_grasp_pose.json
```

Output format should include:
- `functional_grasp` (optional)
- `grasp.body` (list of `[x,y,z,qw,qx,qy,qz]`)

## 2) Put annotation where collector can auto-find it

Recommended path:

```bash
mkdir -p /sim-service/grasp_poses
cp /tmp/mug_grasp_pose.json /sim-service/grasp_poses/mug_grasp_pose.json
```

Collector lookup priority:
1. `COLLECT_GRASP_POSE_PATH` (exact file)
2. `COLLECT_GRASP_POSE_DIR` + token matching
3. default local folders (including `grasp_ops/assets`)

## 3) Run collect with annotation enabled

If you want explicit path:

```bash
export COLLECT_GRASP_POSE_PATH=/sim-service/grasp_poses/mug_grasp_pose.json
```

Or directory mode:

```bash
export COLLECT_GRASP_POSE_DIR=/sim-service/grasp_poses
```

Then run collect as usual (`skill=pick_place`).

## 4) Verify per-frame grasp GT in dataset parquet

Collector now exports these extra columns per frame:
- `observation.object_pose_world` (7D)
- `observation.grasp_target_pose_world` (7D)
- `observation.grasp_target_valid` (bool)
- `observation.grasp_target_source_id` (float id)

Source id:
- `0`: none
- `1`: annotation
- `2`: bbox_center
- `3`: root_pose

Quick check:

```bash
python - <<'PY'
import pandas as pd
p = '/data/embodied/datasets/<dataset_id>/data/chunk-000/file-000.parquet'
df = pd.read_parquet(p)
print(df[[
    'observation.grasp_target_valid',
    'observation.grasp_target_source_id'
]].head())
PY
```
