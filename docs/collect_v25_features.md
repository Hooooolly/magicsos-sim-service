# Pick-Place Collector v25q/v25r 新功能

## 目的

为 ACT (Action Chunking with Transformers) 训练采集多样化的 pick-place 演示数据。核心需求：

1. **连续 pick-place**：抓了放下后，从放下位置再抓（而非每次从同一位置开始）
2. **位置随机化**：每个 episode 球的初始位置略有变化，提升策略泛化能力

---

## 新参数一览

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `reset_mode` | str | `"full"` | `COLLECT_RESET_MODE` | `full`: 每个 episode 重置所有物体；`arm_only`: 只重置机械臂，球留在上次放置位置 |
| `rounds_per_episode` | int | `1` | `COLLECT_ROUNDS_PER_EPISODE` | 单个 episode 内 pick→place 循环次数 |
| `object_position_noise` | float | `0.0` | `COLLECT_OBJECT_POSITION_NOISE` | reset 后对球施加 ±noise 米的随机 x,y 偏移 |
| `episode_timeout_sec` | float | `300.0` | `COLLECT_EPISODE_TIMEOUT_SEC` | 单个 episode 超时（从 180s 提升到 300s） |

所有参数通过 `/collect/start` API 的 JSON body 传递，也支持环境变量 fallback。

---

## 功能详解

### 1. reset_mode: arm_only（v25q）

默认行为（`full`）：每个 episode 开始时 `world.reset()` 把所有物体归位。球永远从同一初始位置被抓。

`arm_only` 模式：episode 间不 reset 世界，只把机械臂移回 HOME。球留在上一个 episode 的放置位置，下一个 episode 从新位置抓取。

```
Episode 1: 球在 (0, 0) → pick → place at (-0.3, -0.1) → HOME
Episode 2: 球在 (-0.3, -0.1) → pick → place at (0.1, 0.2) → HOME
Episode 3: 球在 (0.1, 0.2) → pick → ...
```

注意：`episode_index == 0` 时始终 full reset，保证首个 episode 从干净状态开始。

### 2. rounds_per_episode: 多轮循环（v25q）

单个 episode 内执行多轮 pick→place→HOME 循环，帧计数连续。

```
Episode 1, Round 1/2: pick ball at A → place at B → HOME (recorded)
Episode 1, Round 2/2: pick ball at B → place at C → HOME (recorded)
→ 一个 episode 包含 ~1500 帧（两轮各 ~750 帧）
```

轮间变量重置：`place_pos`, `grasp_succeeded`, `attempt_used`, `ik_target_orientation` 等。
跨轮保持：`frame_index`（连续递增）、`annotation_candidates`。

如果某轮抓取失败，episode 提前结束，不进入下一轮。

### 3. object_position_noise: 位置随机化（v25r）

每个 episode reset 后，对目标物体施加 ±noise 的随机 x,y 偏移。用于 ACT 训练数据多样性。

```
Episode 1: noise=(+0.023, -0.009) → 球在 (0.006, -0.020)
Episode 2: noise=(-0.017, +0.004) → 球在 (-0.034, -0.007)
Episode 3: noise=(+0.005, +0.018) → 球在 (-0.012, 0.008)
```

仅在 `reset_mode=full` 时生效（`arm_only` 模式下球自然在不同位置，无需额外 noise）。

---

## API 调用示例

### 基础采集（默认行为，兼容旧版）

```bash
curl -X POST http://localhost:6805/collect/start \
  -H 'Content-Type: application/json' \
  -d '{"num_episodes": 10, "scene_mode": "reuse", "target_objects": ["/World/Ball"]}'
```

### ACT 训练数据采集（推荐）

```bash
curl -X POST http://localhost:6805/collect/start \
  -H 'Content-Type: application/json' \
  -d '{
    "num_episodes": 50,
    "scene_mode": "reuse",
    "target_objects": ["/World/Ball"],
    "object_position_noise": 0.02,
    "episode_timeout_sec": 600
  }'
```

### 连续 pick-place（arm_only + 多轮）

```bash
curl -X POST http://localhost:6805/collect/start \
  -H 'Content-Type: application/json' \
  -d '{
    "num_episodes": 5,
    "scene_mode": "reuse",
    "target_objects": ["/World/Ball"],
    "reset_mode": "arm_only",
    "rounds_per_episode": 2
  }'
```

---

## ACT 训练数据策略

| 方案 | 参数 | 数据特征 | 泛化效果 |
|------|------|----------|----------|
| 固定位置 | noise=0, episodes=50 | 球始终在同一位置 | 仅对该位置有效 |
| 微偏移 | noise=0.02, episodes=50 | 球在中心 ±2cm | 泛化到 ~4cm 范围 |
| 中偏移 | noise=0.05, episodes=100 | 球在中心 ±5cm | 泛化到 ~10cm |
| 连续轨迹 | arm_only, rounds=3 | 球位置逐步变化 | 学习从任意位置抓取 |

建议起步配置：`object_position_noise=0.02, num_episodes=50`，训练后根据效果调整。

---

## 代码版本

- **v25q** (2026-03-03): `reset_mode` + `rounds_per_episode`
- **v25r** (2026-03-03): `object_position_noise` + timeout 提升到 300s

## 关键文件

- `isaac_pick_place_collector.py` — collector 核心逻辑
- `run_interactive.py` — API 层参数解析和透传
