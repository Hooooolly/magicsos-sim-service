# Isaac Sim 物理踩坑记录

## 1. Bowl 凹面碰撞 — Cube 掉不进碗里 (2026-03-21)

**Problem**: Replay 后 cube 放到碗上停在碗边缘漂浮，不掉进碗内部。

**Root Cause**: PhysX **dynamic rigid body 不支持 triangle mesh collision**，自动 fallback 到 convex hull。Convex hull 把碗口封死了。

**Fix**: Bowl 设为 **static**（去掉 `RigidBodyAPI`），子 mesh 用 triangle mesh collision：

```usda
over "Bowl" (
    prepend apiSchemas = ["PhysicsCollisionAPI"]   # 去掉 PhysicsRigidBodyAPI!
)
...
    over "ball" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    ) {
        uniform token physics:approximation = "none"   # triangle mesh，不是 convex hull
    }
```

**Key Rules**:
- Dynamic body → 只能用 convex hull / convex decomposition / SDF mesh
- **Static body → 支持 triangle mesh**（`approximation = "none"`）
- 碗不需要动，设 static 最简单
- `approximation = "none"` 在 dynamic body 上**无效**，会 fallback 到 convex hull
- 如果一定要 dynamic concave collision，用 `approximation = "sdfMesh"`（SDF 更贵但支持动态凹面）

**References**:
- [Isaac Sim Physics Fundamentals](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html)
- [Isaac Sim Forum - Triangle Mesh Collision](https://forums.developer.nvidia.com/t/collider-and-rigid-body-preset-error-physicsusd-parse-collision-triangle-mesh-collision/244203)

---

## 2. Cube 漂浮 / PhysX-USD 脱耦 (2026-03-21)

**Problem**: Replay 后 cube 停在半空中不掉，即使有 RigidBodyAPI + gravity + mass。

**Root Cause**: Replay 代码或 code_execute 用 USD `xformOp:translate` 直接设置 cube 位置。PhysX body 和 USD prim **脱耦**了 — PhysX solver 不会 pick up USD translate 变化。

**Fix**: 不要直接改 USD xformOp 移动有 RigidBodyAPI 的 prim。
- 用 Isaac Sim API: `rigid_prim.set_world_pose(position, orientation)`
- 或 delete + recreate: `delete_prim()` 然后 `DynamicCuboid()`
- 或 reload scene

**Key Rule**: 永远不要用 `xformOp` 移动有 RigidBodyAPI 的 prim。PhysX 和 USD 的位置同步只在特定 API 调用时发生。

---

## 3. Save Scene 陷阱 (2026-03-21)

**Problem**: 通过 code_execute 改了 cube 位置后 save scene，下次 load 时 cube 在错误位置。

**Root Cause**: `stage.GetRootLayer().Export()` 把**当前 runtime 状态**（robot 姿势、cube 位置、velocity 等）全部写入 USD 文件。

**Key Rule**:
- **不要在 replay/inference 后 save scene** — robot 和 objects 在非初始位置
- 只在场景搭建完成、确认初始状态正确时 save
- save 前先 reload scene 确保回到初始状态

---

## 4. Scene 备份记录 (2026-03-21)

**Working scene（碗能接住 cube，物理正确）**: `openarm-cube-bowl_WORKING_20260321.usda`

| 位置 | 路径 |
|------|------|
| 服务器备份 | `/data/embodied/scene/library/openarm-cube-bowl_WORKING_BACKUP_20260321.usda` |
| 本地备份 | `/Users/holly/Documents/code/bot/openarm-cube-bowl_WORKING_20260321.usda` |
| 运行中 | `/data/embodied/scene/library/openarm-cube-bowl_1773675345.usda` |
| 运行中(副本) | `/data/embodied/scene/library/openarm-cube-bowl_1772686918.usda` |

**Scene 内容**:
- Robot: openarm bimanual (sensor.usd, 有 wrist camera)
- Cube: 0.04×0.03×0.05m, mass=0.05kg, 初始位置 (0.292, -0.190, 0.277)
- Bowl: static, triangle mesh collision, 位置 (0.400, 0.000, 0.252)
- Table: static collision, 位置 (0.392, -0.095, 0.126), scale (0.25, 0.35, 0.252)

---

## 5. Camera Render Product 全黑 (2026-03-21)

**Problem**: init_inference 创建的 camera render product 返回全黑图片（avg_pixel=0）。

**Root Cause**: init_inference 用 `cam_name`（如 `left_wrist_cam`）创建新 camera prim，但新 prim 没有正确的 orient — 朝向 robot 内部，被 mesh 遮挡。

**Fix**: 优先使用 sensor.usd 中已有的 camera prim（`prim_name: wrist_cam`），它有正确的 translate 和 orient。

```python
# camera_meta.yaml 中 prim_name 和 cam_name 不同:
#   cam_name = "left_wrist_cam" (dataset 中的名字)
#   prim_name = "wrist_cam" (USD 中已有的 camera prim)
prim_name = cam_cfg.get("prim_name", cam_name)
existing_cam = mount_prim.GetPath().pathString + f"/{prim_name}"
if stage.GetPrimAtPath(existing_cam).IsValid():
    cam_path = existing_cam  # 用已有的，有正确 orient
```

---

## 6. Inference Action Clip 截断 (2026-03-21)

**Problem**: ACT model 输出的 action 被 clip 到 [-1, 1]，但训练数据 joint 范围超过 ±1。

**Details**:
- joint7 (wrist rotation): 训练范围 [-1.71, 0.13]，37% 的 action 被 clip
- joint4: 训练范围 [-0.10, 1.36]，17% 的 action 被 clip

**Fix**: 把 `self._action_clip` 从 `(-1.0, 1.0)` 改为 `(-3.14, 3.14)`。

---

## 7. Replay 物理设置 (参考)

Replay 代码中动态设置的物理参数（不在 scene 文件里）：
- Cube friction: static=10.0, dynamic=10.0（极高，确保夹住）
- Cube mass: 0.01 kg（从 scene 的 0.05 减到 0.01，更轻更容易抓）
- Finger PD: 高 stiffness + damping

---

## 8. Inference Homing Pose (2026-03-21)

Replay ep0 结束后的 arm 姿势，可作为 inference 的 homing/ready position：

**Right arm (sim joint order, radians)**:
| Joint | Degrees | Radians |
|-------|---------|---------|
| joint1 | -24.6° | -0.4292 |
| joint2 | 12.2° | 0.2138 |
| joint3 | -12.6° | -0.2208 |
| joint4 | 53.3° | 0.9299 |
| joint5 | 14.9° | 0.2602 |
| joint6 | -11.8° | -0.2055 |
| joint7 | 57.6° | 1.0057 |

**Dataset order** [j1,j3,j4,j5,j6,j2,j7,finger]:
`[-0.4292, -0.2208, 0.9299, 0.2602, -0.2055, 0.2138, -1.0057, 0.04]`

**18-DOF (for InferencePanel TRAIN_INIT_18DOF)**:
`[0, -0.4292, 0, 0.2138, 0, -0.2208, 0, 0.9299, 0, 0.2602, 0, -0.2055, 0, 1.0057, 0, 0, 0.04, 0.04]`

注意：这是 replay 完 pick-and-place 后的姿势（手臂在桌面上方放开 cube 后的状态），不是初始 rest 位。更适合作为 inference 的 ready pose 因为 camera 对着桌面。

---

## 9. Replay 第二次不 Reset Scene (2026-03-21)

**Problem**: 第一次 replay 后 cube 停在碗里/桌上。第二次 replay 不 reload scene，cube 从上次终态开始，位置错误。

**Root Cause**: 之前 backend 用两个独立 HTTP 调用：先 `/scene/load` 再 `/replay/start`。两个命令分别入队到 bridge 的 `_cmd_queue`。如果 scene_load 超时或 replay 太快开始，scene 可能没有完全 reset。

**Fix**: `_run_pending_replay()` 内部原子化处理 — 收到 `scene_usd_path` 后直接 `ctx.open_stage()` reload scene，再执行 replay。不再依赖单独的 scene_load 命令。

同时在 replay 的 `finally` block 中清除 `_inf_dc` 和 `_inf_init_result`，让 MonitorPanel 自动 re-init。

**Key Rule**: Replay 必须 reload scene 才能保证 object 初始位置正确。不能复用上次 replay 后的 stage 状态。

---

## 10. MonitorPanel Replay 后卡死 (2026-03-21)

**Problem**: Replay 运行时 main thread 被阻塞在 frame-by-frame 循环里，`_process_commands()` 不执行。MonitorPanel 发的 `init_inference` 请求排队但无法执行。Replay 结束后 `_inf_dc` 已 stale，Monitor 卡在 "Initializing"。

**Fix**:
1. Replay `finally` block 中清除 `_inf_dc = None`
2. MonitorPanel 检测到 obs 400（之前是 ready）时，threshold 从 3 降到 1，立即发 force re-init
3. 检测到 obs 200 但 positions=[] 时，threshold 从 5 降到 2

**Key Rule**: 任何改变 stage/articulation 的操作（replay、scene_load）都必须清除 `_inf_dc`，否则 MonitorPanel 会用 stale handle 读不到数据。

---

## 11. Replay 期间 MonitorPanel obs 504 Timeout (2026-03-21)

**Problem**: Replay 运行时 MonitorPanel 的 `/robot/obs` 返回 504 timeout。有 joint data 也读不到。

**Root Cause**: `/robot/obs` 用 `_enqueue_cmd("inf_obs")` 把读取操作排队到主线程。但 replay 期间主线程被阻塞在 frame loop 里，`_process_commands()` 不执行，命令超时返回 504。

尝试直接把 replay 的 `Articulation` 对象赋给 `_inf_dc` 让 obs 直接读也不行 — Flask 线程调 `robot.get_joint_positions()` 和主线程的 `world.step()` 争 PhysX 锁，同样 504。

**Fix**: 共享 dict 方案：
1. Replay loop 每帧 `world.step()` 之后，在主线程上调 `robot.get_joint_positions()` 写入 `_state["replay_obs"]`
2. `/robot/obs` endpoint 检测 `_state["replaying"]` 时直接返回 `_state["replay_obs"]`，不走 `_enqueue_cmd`
3. Replay `finally` 中清除 `_state["replay_obs"]`

**Key Rule**: Replay 期间不能用 `_enqueue_cmd` 或从 Flask 线程调任何 Isaac Sim API — 主线程被阻塞，PhysX 锁会冲突。必须用主线程写 → Flask 线程读的共享 dict 模式。

---

## 12. Replay 期间 MonitorPanel Camera 全黑 (2026-03-21)

**Problem**: Replay 期间 MonitorPanel 的 camera feed 全黑，joint 数据有但 images 为空 `{}`。

**Root Cause**: Replay 循环没有 setup render product + annotator。init_inference 会做 camera setup，但 replay 的 `_run_pending_replay()` 是独立流程，不走 init_inference。

**Fix**: 在 `_run_pending_replay()` 的 frame loop 开始前：
1. 从 `camera_meta.yaml` 读取 wrist camera 配置（mount_link、prim_name、resolution）
2. 找到已有的 camera prim（如 `/World/OpenArm/openarm_right_link7/wrist_cam`）
3. `rep.create.render_product(cam_prim_path, resolution)` + `rgb` annotator
4. 3 帧 warmup
5. 每帧 `world.step(render=True)` 之后读 `annot.get_data()` → RGBA → JPEG base64 → 写入 `_state["replay_obs"]["images"]`
6. `finally` 中 `annot.detach()` + `rp.destroy()` 清理

**Key Rule**: 任何需要 camera 图像的流程（replay、inference、collect）都必须独立 setup render product + annotator。不能假设别的流程已经 setup 好了。Camera meta 路径不能 hardcode，用 `os.path.dirname(__file__)/camera_meta.yaml` 或 `/data/embodied/asset/robots/...` fallback。
