# QMIX 基线（负责人 D）

本目录承担两件事：**说明 QMIX 怎么接入**，以及**把外部日志并进统一格式**。
仓库内的 QMIX 实现不在这个目录，在 `src/bacmarl/`：

| 文件 | 作用 |
|---|---|
| `src/bacmarl/algos/qmix.py` | 个体 Q（GRU + agent_id one-hot）、单调混合网络、Double-Q、n 步回报 |
| `src/bacmarl/algos/replay.py` | 环形回放与 `NStepWindow`（把单步转移折叠成 n 步回报） |
| `src/bacmarl/runners/qmix.py` | 训练/评测入口，与 `runners/train.py` 共用配置、运行名、评测信道与日志列 |
| `configs/algo/qmix.yaml` | 全部 QMIX 超参（只在 `algo.*` 下，不动共用的 `train.*`） |
| `baselines/qmix_benchmarl/convert.py` | BenchMARL/torchrl 导出的 csv → 本仓库 `metrics.csv` |

## 为什么没有直接接 BenchMARL

开题报告写的是「QMIX 采用 BenchMARL（TorchRL 后端）」。落地时发现 BenchMARL 的 MPE2
任务包的是原版 `simple_formation`：没有局部观测半径、没有碰撞惩罚、没有本项目的团队
奖励重算与完成判定。用它和 IPPO/MAPPO 对比，比的是环境差异而不是算法差异，而且拿不到
`success_rate / formation_err / collisions` 这些本项目要求的指标。

所以主路径改为仓库内实现（跑同一个 `FormationCommEnv`），BenchMARL 保留为**日志来源**
之一：外部跑出来的 csv 用 `convert.py` 转成同一套列，作图与汇总脚本对两种来源一视同仁。
详细权衡与调参记录见 [`docs/baselines.md`](../../docs/baselines.md)。

## 用法

```bash
# 训练（与策略梯度入口同一套配置口径）
python -m bacmarl.runners.qmix env=formation algo=qmix seed=0

# 训练完顺带按丢包率网格复测，写 robust.csv（契约 5 列）
python -m bacmarl.runners.qmix env=formation algo=qmix seed=0 --robust

# 把 BenchMARL 的日志转进来（列名映射见脚本 docstring）
python baselines/qmix_benchmarl/convert.py \
    --in benchmarl_qmix.csv \
    --out results/main/formation_6n_qmix_L0_K3_B0.0_s0
```

注意：QMIX 不通信，`comm/bits`、`speak_rate` 恒为 0，`L0/B0.0` 是正确口径；
`--robust` 的丢包网格对它必然是平线（策略对信道免疫），报告里要说明它不参与
「比特压缩」比较。

动作空间需为离散——MPE2 默认 5 个离散动作，正好匹配。
