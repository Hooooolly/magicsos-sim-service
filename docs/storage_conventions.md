# 场景与资产存储规范

## 核心原则

所有场景和资产文件存储在 hgx NFS: `/data/embodied/`。
Pod 通过 PVC `embodied-nfs-pvc` 挂载到同路径 `/data/embodied/`。

**禁止事项**：
- ❌ 不要存到 108 本地 `/data/sim_scenes/` (已废弃)
- ❌ 不要存到 `/home/magics/` 下的任何目录
- ❌ 不要在代码里硬编码路径，用环境变量

## 目录结构

```
/data/embodied/
├── scene/
│   ├── library/           ← 可加载的 .usda 场景文件
│   └── runs/              ← SceneSmith pipeline 中间产物
├── asset/
│   ├── manipuland/        ← 可抓取物体 (USD + grasp annotations)
│   ├── furniture/         ← 碰撞体家具 USD
│   └── articulated/       ← 关节体 USD (门/抽屉)
└── dataset/
    └── sim/               ← 采集数据输出
```

## 环境变量

| 变量 | 值 | 用途 |
|------|----|------|
| `SIM_SCENE_LIBRARY` | `/data/embodied/scene/library` | scene_save 输出目录 |
| `SCENESMITH_OUTPUTS_ROOT` | `/data/embodied/scene/runs` | SceneSmith pipeline 输出 |
| `SIM_ASSET_ROOT` | `/data/embodied/asset` | Asset 存储根目录 |
| `EMBODIED_DATASETS_ROOT` | `/data/embodied/dataset/sim` | 采集数据输出 |

## 场景保存

### 方式1: sim-service 直接保存 (推荐)

1. Isaac Sim 编辑好场景
2. 调用 `POST /scene/save` → 写入 `/data/embodied/scene/library/xxx.usda`
3. infra-backend 创建 SavedScene 记录，`environment.usd_path` 指向 NFS 路径

### 方式2: SceneSmith pipeline

1. SceneSmith 生成 → .blend 输出到 `/data/embodied/scene/runs/run_N/`
2. USD 导出 → 同目录生成 `scene.usda`
3. 保存到 Library → 创建 SavedScene 记录

## 场景加载

1. 前端/Luban 调用 `POST /bridge/scene/load` (带 `job_id` 或 `usd_path`)
2. infra-backend 解析 USD 路径（三级 fallback）：
   - `environment.usd_path` 直接路径
   - SceneSmith job output_dir 搜索
   - Legacy `/scenesmith/scenes/{id}/scene.usda`
3. 转发给 sim-service Bridge → `ctx.open_stage(usd_path)`

## Asset 分类

| 类型 | 目录 | USD 特征 | 导入处理 |
|------|------|----------|----------|
| manipuland | `asset/manipuland/` | rigid body | RigidBodyAPI + CollisionAPI + grasp annotations |
| furniture | `asset/furniture/` | static collider | CollisionAPI (mesh/convex), 无 RigidBody |
| articulated | `asset/articulated/` | joint hierarchy | ArticulationRootAPI + joint definitions |

每个 asset 独立文件夹：
```
asset/manipuland/apple_001/
├── Object.usd
├── textures/
│   └── Scan.jpg
└── grasp_pose.json     ← 仅 manipuland
```

## K8s Volume 挂载

Pod manifest 中两个存储挂载：

| Volume | 挂载路径 | 来源 |
|--------|----------|------|
| `data-storage` | `/data` | hostPath (108 本地) |
| `embodied-shared` | `/data/embodied` | PVC `embodied-nfs-pvc` (hgx NFS) |

NFS 挂载会 overlay hostPath 的 `/data/embodied` 子目录。
场景和资产通过 NFS 存取，所有 pod 共享同一份数据。
