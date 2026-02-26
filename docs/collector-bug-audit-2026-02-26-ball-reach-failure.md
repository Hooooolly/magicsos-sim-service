# Collector Bug Audit (2026-02-26): Ball Reach Failure Root Cause

> Author: Claude Code (Opus 4.6)
> Date: 2026-02-26
> Log source: `sim-interactive-1_collect_20260226_0822/pod_full.log`

## Summary

Ball collect 连续 5/5 attempt `reach_before_close` failed，arm 停在距 target 25-30cm 外。
根因是三个 bug 叠加：curobo 未安装、IK 帧选错、tip_mid offset 计算错误。

---

## Bug 1: Curobo 未安装（Pod 环境）

**现象**
```
collect: PLANNER_BACKEND=curobo (raw='curobo')
Curobo init failed, falling back to IK: No module named 'curobo'
```

**原因**: Pod 容器镜像缺少 `curobo` Python 包。环境变量 `COLLECT_PLANNER_BACKEND=curobo` 已设置，
但 `import curobo` 失败后 fallback 到纯 IK 路径。

**影响**: `_curobo_full_approach()` 完全不执行。所有 attempt 走 IK rate-limited stepping，
无法处理 body-grasp 标注的大关节跳变。

**修复**: 在 pod 容器镜像中安装 curobo（`pip install nvidia-curobo` 或从 MagicSim 镜像复制）。

---

## Bug 2: IK 帧选用 panda_hand 而非 TCP（right_gripper）

**现象**: IK solver 控制 `panda_hand` 帧（link8 级别），而标注描述的是指尖接触位置。

```
IK solver ready: frame=panda_hand robot=/World/FrankaRobot
```

**URDF 帧层次**（`lula_franka_gen.urdf`）:
```
panda_link8 / panda_hand (same position, -45° yaw)
    ├── panda_leftfinger    xyz=(0, 0, 0.0584)   finger base, 5.84cm
    │   └── panda_leftfingertip  xyz=(0, 0, 0.045)   +4.5cm = total 10.34cm
    └── right_gripper       xyz=(0, 0, 0.1)       TCP frame, 10cm
```

`right_gripper` 是标准 TCP 帧（10cm from hand ≈ 指尖中点），已存在于 URDF 中且在
`_select_ik_frame_name` 候选列表里（L1394），但优先级低于 `panda_hand`，从未被选中。

**影响**: IK target 必须经过 `tip_mid→hand` 转换才能正确定位 panda_hand。
如果转换有误（见 Bug 3），指尖位置就会偏。

**修复方案 A（推荐）**: IK 帧改用 `right_gripper`，关闭 `ANNOTATION_POSE_IS_TIP_MID_FRAME`。
标注 position 直接作为 IK 目标，无需转换。

**修复方案 B**: 保持 panda_hand 帧，修正 Bug 3 中的 offset 值。

---

## Bug 3: tip_mid offset 用了 finger base 而非 fingertip

**现象**:
```
collect: annotation tip-mid->hand enabled, offset_hand=(0.0000, 0.0000, 0.0584), norm=0.0584
```

**原因**: `_compute_tip_mid_offset_in_hand()` (L2364) 调用 `_resolve_finger_prim_paths()` (L2319),
后者解析到 `panda_leftfinger` / `panda_rightfinger`（finger base，5.84cm from hand）。
但 URDF 中实际指尖是 `panda_leftfingertip` / `panda_rightfingertip`（再往下 4.5cm，total 10.34cm）。

```python
# _resolve_finger_prim_paths 候选列表 (L2324-2330)
left_candidates = [
    f"{robot_prim_path}/panda_leftfinger",      # ← finger base, 5.84cm
    f"{robot_prim_path}/leftfinger",
    f"{robot_prim_path}/left_finger",
]
# 缺少 panda_leftfingertip (10.34cm)
```

**数学后果**:

ContactGraspNet 标注定义的是 TCP（指尖）位置。ball 标注 scaled position = 1.44cm from center（球表面）。

`tip_mid→hand` 转换: `hand_pos = tip_pos - R(quat) × offset`

| | offset | hand target (from ball center) | 指尖实际位置 |
|---|--------|-------------------------------|------------|
| 当前 (错误) | 5.84cm | 1.44 + 5.84 = **7.28cm** | 7.28 - 10.34 = **-3.06cm** (穿过球心) |
| 正确 | 10.34cm | 1.44 + 10.34 = **11.78cm** | 11.78 - 10.34 = **1.44cm** (球表面) ✓ |

当前 offset 导致指尖目标穿过球心 3cm 到另一侧。物理上不可能达到，arm 被卡在错误位置。

**MagicSim 对比**:

MagicSim `FrankaFrameCfg`（`Env/Robot/Cfg/Manipulator/Franka.py:176-203`）在 finger prim 上
额外加了 `offset=(0, 0, 0.046)` 来补偿 finger base → fingertip 的距离：

```python
# MagicSim FrankaFrameCfg.target_frames
[0] panda_hand,        offset=(0, 0, 0)       # hand 本身
[1] panda_rightfinger,  offset=(0, 0, 0.046)   # finger base + 4.6cm = fingertip ✓
[2] panda_leftfinger,   offset=(0, 0, 0.046)   # finger base + 4.6cm = fingertip ✓
```

tip_mid 计算（`Grasp.py:586`）:
```python
tip_mid = 0.5 * (right_pos + left_pos)   # right/left 已含 4.6cm offset
offset_world = tip_mid - hand_pos         # ≈ 5.84 + 4.6 = 10.44cm
```

MagicSim 的 tip_mid offset ≈ **10.44cm**（接近 URDF 的 10.34cm），
我们的 tip_mid offset = **5.84cm**（少了 4.6cm 的 fingertip 延伸）。

**修复（已选方案 B）**: 在 `_compute_tip_mid_offset_in_hand` 中对 finger base 位置加上
fingertip offset `(0, 0, 0.046)`（与 MagicSim 一致），使 tip_mid 代表真正的指尖中点。

---

## Bug 4: IK 路径无法执行 body-grasp 朝向

**现象**:
```
IK waypoint 'PRE_GRASP' fallback to unconstrained orientation
IK waypoint 'PRE_GRASP' jump 1.770 rad
IK waypoint 'DOWN_PICK' jump 1.799 rad
```

5 次 attempt 的 eef_to_target xy 距离（pre_close_gate 时刻）:

| Attempt | eef_to_target xy (m) | 说明 |
|---------|---------------------|------|
| 1 | 0.221 | arm 基本没移动到位 |
| 2 | 0.230 | |
| 3 | 0.226 | |
| 4 | 0.247 | |
| 5 | 0.265 | |

**原因**: OmniObject3D ball 标注包含 498 个 body-grasp（侧面、斜面抓取），
这些朝向产生的 IK 关节解与 home 姿态差 1.7+ rad。
rate-limited stepping（0.03 rad/step × ~50 steps = max 1.5 rad）无法遍历这些跳变，
arm 停在中途。

**影响**: 纯 IK 路径下 ball body-grasp 标注 100% 失败。

**修复**: 安装 curobo（Bug 1 修复）后，`_curobo_full_approach()` 从当前关节直接规划平滑轨迹，
不受 0.03 rad/step 限制。临时绕过方案：`COLLECT_FORCE_TOPDOWN_GRASP=1`（丢弃标注朝向用顶抓）。

---

## Bug 5: Wrist Camera 贴合手腕，看不到指尖

**现象**: 用户报告 wrist camera 太贴合手腕，视角被遮挡无法看到指尖区域。

**原因**:
```python
WRIST_CAM_LOCAL_POS_ROS = (0.025, 0.0, 0.0)   # 仅 2.5cm 偏移
```
相机几乎贴着 panda_hand 表面，无法俯视指尖。

**修复**: 增大 position offset 使相机凸出，加入俯角使光轴朝向指尖区域。
需要在 sim 中实际验证具体数值（position 和 quaternion 组合）。

---

## 附录: 日志中的关键数值

### 标注 Scale

```
annotation scale inferred 0.08445 (mesh_radius=0.0144, median_norm=0.1701, preset=1.0000)
annotation_local_pos_scale=0.08445 (hint=1.0000 * infer=0.0845)
loaded 498 grasp annotations for /World/Ball
```

Scale 推断正确。raw annotation norm 0.17m × 0.08445 = 0.0144m ≈ ball radius 0.014m。

### obj_to_target（球心到 panda_hand target）

| Attempt | dx (cm) | dy (cm) | dz (cm) | 3D (cm) |
|---------|---------|---------|---------|---------|
| 1 | +3.8 | -6.1 | -1.5 | 7.3 |
| 2 | -2.6 | +0.8 | +6.7 | 7.3 |
| 3 | -5.2 | -1.0 | -5.0 | 7.3 |
| 4 | -4.6 | -3.9 | -4.1 | 7.3 |
| 5 | -0.8 | -7.2 | +0.9 | 7.3 |

所有 attempt 3D 距离恒定 7.3cm = scaled_annotation_norm(1.44cm) + tip_mid_offset(5.84cm)。
验证了 Bug 3 中 offset 计算链路。

### URDF 帧结构

```
panda_link8
  ├── panda_hand            rpy=(0, 0, -0.785)  xyz=(0, 0, 0)        same position, -45° yaw
  │   ├── panda_leftfinger  rpy=(0, 0, 0)        xyz=(0, 0, 0.0584)  finger base
  │   │   └── panda_leftfingertip               xyz=(0, 0, 0.045)    fingertip
  │   └── panda_rightfinger rpy=(0, 0, 0)        xyz=(0, 0, 0.0584)  finger base
  │       └── panda_rightfingertip              xyz=(0, 0, 0.045)    fingertip
  └── right_gripper         rpy=(0, 0, 2.356)   xyz=(0, 0, 0.1)     TCP (≈ fingertip midpoint)
```

---

## 修复优先级

| 优先级 | Bug | 修复 | 依赖 |
|--------|-----|------|------|
| P0 | Bug 1: curobo 未安装 | pip install nvidia-curobo | 容器镜像 |
| P0 | Bug 3: tip_mid offset 少 4.6cm | finger pos + (0,0,0.046) fingertip offset（与 MagicSim 一致） | 代码改动 |
| P1 | Bug 4: IK body-grasp 失败 | Bug 1 修复后自动解决 | Bug 1 |
| P2 | Bug 5: wrist camera 偏移 | 调整 position + quaternion | 视觉验证 |

Bug 3 修复方案：在 `_compute_tip_mid_offset_in_hand` 中对 finger base 位置加 `(0, 0, 0.046)`
fingertip offset（与 MagicSim `FrankaFrameCfg` 一致）。修复后 tip_mid offset 从 5.84cm → 10.44cm，
tip_mid→hand 转换正确，IK target 和 Curobo target 都受益。保持 IK frame = panda_hand 不变。
