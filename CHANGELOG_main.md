# main.py 修改日志（指标扩展版）
10月15
## 概述
为满足“用**碰撞率**替代 PDR 展示、引入 **SU 与 PU 竞争时的成功率**、用 **平均吞吐量**替代总吞吐展示”的需求，更新了训练脚本 `main.py`，新增底层采样指标、派生统计指标，并扩展了 CSV 输出字段，便于后续画图脚本直接读取并出图。

---

## 变更详情

### 1) 环境计数器（Env.metrics）扩展
**位置**：`MultiAccessEnv._zero_metrics()`、`MultiAccessEnv.step()`

- **新增字段**
  - `su_tx_compete_pu`：SU 在与 PU 竞争同一信道时的**尝试次数**
  - `su_success_compete_pu`：上述竞争场景下，SU 的**成功次数**
- **统计逻辑**
  - 在 `step()` 中根据每个信道是否有 PU 参与（`pu_present[k]`）来判断“SU 是否在与 PU 竞争”。
  - 若 SU 在该信道发起传输且 `pu_present[k] == True`，则 `su_tx_compete_pu += 1`。若当拍 SU 成为赢家，则 `su_success_compete_pu += 1`。
- 其余原有计数（`tx/success/collision/wait/sum_throughput`、各类 collision 分解、PU/SU 的 tx 和 success 等）保持不变。

> 目的：支持计算 **SU 在与 PU 竞争时的成功率**，作为公平性/接入能力的重要指标。

---

### 2) 训练回合后的派生指标（train_one）
**位置**：`train_one()`（每个 episode 结束时）

新增计算并记录以下派生指标：
- `pdr_total = success / tx`（保留原有“整体成功率”，便于对比）
- `coll_rate = collision / tx`（**碰撞率**，用于替代 PDR 的对外展示）
- `wait_percent = wait / (episode_len * num_users) * 100`（等待占比，单位 %）
- `coll_per_100 = collision / (episode_len * num_users) * 100`（每百步碰撞数）
- `su_compete_success_rate = su_success_compete_pu / su_tx_compete_pu`（SU 对 PU 竞争成功率；无竞争时定义为 0.0）
- `throughput = sum_throughput / (episode_len * num_users)`（**平均吞吐量**，用于替代总吞吐展示；原始 `sum_throughput` 仍保留以兼容历史）

> 目的：让画图脚本无需再推导，直接读取即可作图；同时保留关键原始量，便于审计与二次分析。

---

### 3) CSV 输出字段扩展
**位置**：`train_one()` 保存 `metrics_multi_roles.csv` 处

新增或调整后的 CSV 表头（顺序固定，兼容画图脚本）：
```
episode, return,
tx, success, collision, wait,
coll_incumbent, coll_multi, coll_preempt_su,
tx_pu, success_pu, tx_su, success_su,
pdr_total,          # 总体成功率
coll_rate,          # 碰撞率（后续展示主指标）
wait_percent,       # 等待占比（%）
coll_per_100,       # 每百步碰撞
sum_throughput,     # 原总吞吐（保留）
throughput,         # 新：平均吞吐量（每槽）
su_tx_compete_pu, su_success_compete_pu, su_compete_success_rate
```

> 兼容性：旧列名（如 `sum_throughput`）仍保留；新增 `throughput` 作为推荐展示项。画图脚本可直接使用新列。

---

### 4) 训练日志打印调整
**位置**：`train_one()` 中 `if ep % log_every == 0:` 的打印

- 增加了 `SuccRate`（= `pdr_total`）与 `CollRate`（= `coll_rate`）的并行打印，方便对照。
- 替换展示项：用 `Throughput`（平均吞吐量）替代原“SumThr”，但为避免信息丢失，仍打印 `Coll/100`、`Wait%` 等。

**示例输出**
```
[base] Ep XXXX | Return XXXXX | AvgLoss X.XXXX | Eps X.XXX |
SuccRate 0.8XX | CollRate 0.0XX | Coll XXXX | Wait XXXX |
Coll/100 X.X | Wait% XX.X | Throughput 0.XXXX
```

---

### 5) 评估流程（evaluate）
**位置**：`evaluate()`  
- 评估逻辑本身未改变；依然基于 best ckpt 做 20 回合纯贪心评估并打印汇总（便于纵向对比）。  
- 本次改动集中在训练时的指标采样与导出，不影响评估接口。

---

### 6) 路径与产物
- 训练产物统一写到：`outputs/sc_<tag>/`
  - 模型：`dqn_multi_roles.pth`、`dqn_multi_roles_best.pth`
  - 指标：`metrics_multi_roles.csv`（**本次有扩展**）
  - 元信息：`run_meta.json`（用于记录 `episode_len/num_users/num_channels/...`）

> 画图脚本（新版 `plot_metrics.py`）会自动读取 `metrics_multi_roles.csv`；若未提供路径，会在工作目录下递归搜索最新的一个。

---

## 设计取舍 / 注意事项

1. **除零保护**
   - 当 `tx == 0` 或 `su_tx_compete_pu == 0` 时，对应比率定义为 `0.0`，避免 NaN/Inf 污染曲线。

2. **平均吞吐量**
   - 出图推荐使用 `throughput = sum_throughput / (episode_len * num_users)`，利于不同参数/负载下横向对比。
   - 原始 `sum_throughput` 仍保留，方便核对/审计。

3. **兼容旧流程**
   - 保留了旧的关键字段（如 `sum_throughput`、`pdr_total`），即使你暂时不更新画图脚本也能继续用；但推荐升级到新画图脚本以直接展示“碰撞率/吞吐量/SU 竞争成功率”。

4. **对算法行为的影响**
   - 本次改动**不改变**智能体训练逻辑与奖励定义（仅在**统计口径**与**导出**上扩展）；训练曲线趋势应与旧版一致。

---

10月16
# **_存在问题：su不能成功接入**_
本人想法：可以共享频谱资源，策略：频分复用技术；
# 变更日志（CHANGELOG）

> 项目：**Multi-Channel Multi-User DQN with PU/SU Roles (+ V2X guard window & PU duty)**  
> 版本：`v2x-guard-duty-integration`  
> 日期：2025-10-16  
> 作者：@robertswaters8219 & GPT-5 Thinking

---

## ✨ 新增功能（Feature Additions）

### A) SU 保护窗口（V2X Guard Window）
- 在 V2X 第三段 `rsu_suggest` 中引入**SU 保护掩码**：每拍选择一定比例的信道作为“建议=保护”信道。
- **行为**：
  - **SU** 在“建议信道”上获得正向偏置（Q 值加分）。
  - **PU** 在“建议信道”上获得负向偏置（Q 值减分）。
- **目标**：给 SU 创造更稳定的接入窗口、降低与 PU 的正面冲突。

### B) PU 占空比（PU Duty / `pu_tx_prob`）
- 即便 PU **选择了发射动作**，也仅以 `pu_tx_prob` 的概率**真正尝试**；否则“静默”（等价等待）。
- **目标**：降低 PU 对 SU 的持续压制，为 SU 提供更多可学习的成功样本。

---

## 🧩 新增/调整的配置项

### `EnvConfig`
- `su_guard_frac: float = 0.0` — 每拍“保护信道”占比（0~1），如 `0.25` 表示 25% 信道受保护。
- `su_guard_rotate: bool = True` — 是否**轮动**保护窗口，以保证**公平性**。
- `pu_tx_prob: float = 1.0` — PU 真实发射概率（小于 1 时生效）。
- （保留）`use_v2x, v2x_noise_std, v2x_latency_slots`。

### `DQNConfig`
- `v2x_suggest_pu_malus: float = 1.0` — PU 在**保护信道**上的减分强度（与 `softmask_v2x_suggest` 对 SU 的加分对偶）。
- （保留）`softmask_v2x_suggest, softmask_v2x_avoid_winner, sticky_bias` 等。

---

## 🏗 环境层（Environment）改动要点

1. **占空比作用于“有效动作”**：
   - 在 `step()` 中将输入 `actions` 映射为 `effective_actions`：对 PU 以 `pu_tx_prob` 决定是否静默。
   - 后续**选择统计、胜者判定、奖励**全部基于 `effective_actions`。

2. **返回“生效动作”**：
   - `info["effective_actions"] = effective_actions` 供训练/评估使用，避免把 PU 的“静默”误计为尝试。

3. **V2X 第三段变更**：
   - `_build_v2x()`：当 `su_guard_frac > 0` 时，用**保护窗口掩码**覆盖 `rsu_suggest`；支持**轮动**或**随机抽样**。

4. **随机数安全**：
   - 保护窗口随机抽样使用**局部生成器**，不污染全局 RNG 状态，保证实验复现性。

---

## 🧠 DQN 决策逻辑改动

1. **角色化 V2X 偏置**：
   - 在 `_apply_softmask_and_tiebreak()` 和 `select_batch()` 中：
     - SU：`+ softmask_v2x_suggest * rsu_suggest`
     - PU：`- v2x_suggest_pu_malus * rsu_suggest`
   - 继续保留：**忙信道软惩罚**、**唯一赢家信道惩罚（可选）**、**粘滞性**与**打散**。

2. **探索率按角色缩放（已有）**：
   - `scale_pu=0.6, scale_su=1.4`，使 SU 更愿意探索。

---

## 🔁 训练与评估流程改动

- 统一使用 `info["effective_actions"]`：
  - `agent.update_sticky(effective_actions, winners)`
  - 正则项中“尝试数/碰撞率”的计算
  - Replay 写入时使用 `effective_actions[u]`（与奖励匹配）

---

## 🛠 Bug 修复

1. **探索率衰减笔误**：`c.e_end` → `c.eps_end`，修复 `DQN._eps()` 报错。  
2. **打印语句**：`print(f(...))` → `print(f"...")`，避免 `TypeError`。  
3. **随机数状态污染**：V2X 保护窗口抽样使用局部 RNG。

---

## 🧪 默认行为与兼容性

- 默认关闭新增策略：`su_guard_frac=0.0`、`pu_tx_prob=1.0`，行为与改造前一致。  
- 所有原有指标与导出文件保持向后兼容；`run_meta.json` 增补记录新参数。

---

## 🚀 快速试运行（建议起点）

```python
EnvConfig(
    num_channels=4, num_users=8, pu_ratio=0.5, p_idle=0.7,
    use_v2x=True, v2x_latency_slots=1, v2x_noise_std=0.02,
    su_guard_frac=0.25,   # 25% 信道保护 SU
    su_guard_rotate=True,
    pu_tx_prob=0.9        # PU 9 成概率发射
)
# 可配合：DQNConfig.softmask_v2x_suggest=1.0~1.5, v2x_suggest_pu_malus=1.0~1.5
```

观察指标：
- `success_su / tx_su`（SU 成功率）、`su_compete_success_rate`（与 PU 竞争时 SU 的胜率）  
- `coll_rate`（碰撞率）、`throughput`（平均吞吐）

---

## 📊 指标与输出文件

- 日志打印：`SuccRate`、`CollRate`、`Throughput`、`Coll/100`、`Wait%` 等。
- `outputs/sc_<tag>/metrics_multi_roles.csv`：记录每回合的主要指标（含新增 SU 竞争指标）。
- `outputs/sc_<tag>/run_meta.json`：记录本次运行的关键环境与新参数。

---

## ⚠️ 注意事项（Practical Notes）

- **保护窗口不是硬规则**：仅体现在 V2X 建议与 Q 值偏置，不会强行禁止 PU。实际约束仍由奖励/偏置引导学习。  
- **占空比过低**会让 PU 过“仁慈”，可视化对整体吞吐与时延的影响，按需折中 `pu_tx_prob`。  
- 在**高 `p_miss`/低 `p_idle`** 下，适度放大 `softmask_busy_tx` 以避免盲目碰撞。

---

## 🔁 复现实验建议（A/B Ablation）

在 `SCENARIOS` 中准备三组：
1) **Baseline**：`su_guard_frac=0.0, pu_tx_prob=1.0`  
2) **Only A**：`su_guard_frac=0.25, pu_tx_prob=1.0`  
3) **A+B**：`su_guard_frac=0.25, pu_tx_prob=0.9`  

比较 `success_su`、`su_compete_success_rate`、`coll_rate`、`throughput` 的提升幅度。

---

## 📝 版本记录（Summary of Diffs）

- **EnvConfig**：新增 `su_guard_frac`, `su_guard_rotate`, `pu_tx_prob`。  
- **DQNConfig**：新增 `v2x_suggest_pu_malus`。  
- **Env.step**：引入 `effective_actions`；计数/赢家/奖励基于有效动作；返回 `effective_actions`。  
- **_build_v2x**：第三段 `rsu_suggest` 改为 SU 保护掩码（可轮动/抽样）。  
- **DQN**：在 Q 值上叠加**角色化 V2X 偏置**；修正 `_eps()`。  
- **Train/Eval**：统一使用 `effective_actions`；修复打印。

---

如需我把 **三组场景模板** 直接写进 `SCENARIOS` 帮你跑对比，请告诉我你的 GPU 显存与偏好的训练回合数。

