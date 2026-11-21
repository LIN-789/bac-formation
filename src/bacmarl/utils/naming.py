"""运行名与结果目录口径（contracts 第 6 节，唯一实现）。

`train.py`、批量驱动 `scripts/run_experiments.py` 以及作图/汇总脚本必须共用这一份
实现，避免「训练写一个名字、排班算另一个名字」的漂移。

Issue #11：连续消息基线（IC3Net / TarMAC）在配置里 `msg_bits=0`（只给信道采样占位），
真实的比特宽度是 `32 × msg_dim = 1024`。按 `msg_bits` 拼名字会写出 `L0_B1e9`，
与实测 `comm/bits`（597~1024）自相矛盾，汇总表会读成「零比特、无预算」。
因此这里统一按「名字里的 L 就是通信模块上报的单条消息比特宽度 `bits_per_msg`」计算
（平均每步比特数是 `comm/bits = 发言率 × L`，两者别混）：

* 二值消息（BAC）：L = `algo.msg_bits`（`L4`，配 `B2.0` 时平均比特数 ≈ 2.0）；
* 连续消息（IC3Net / TarMAC）：L = `algo.bits_per_msg`（缺省 `32 × algo.msg_dim`）；
* 无通信（MAPPO / IPPO / QMIX）：L = `algo.msg_bits = 0`。

预算段：`budget_bits` 是有上限的实数时沿用旧格式（`B2.0`），无上限（≥ `INF_BUDGET`）
写 `Binf`，避免目录名出现 `B1000000000.0`。
"""

from __future__ import annotations

INF_BUDGET = 1e6  # 视为「无预算上限」的阈值：1e9 的占位值与真实预算差着量级

# 连续消息算法：每个实数维度按 32 bit 计（fp32），与 comm/*.py 的 bits_per_msg 同口径
CONTINUOUS_COMMS = ("ic3net", "tarmac")
BITS_PER_DIM = 32


def bits_per_msg_for_name(algo: dict) -> int:
    """运行名里 L 段的数值：与通信模块 `bits_per_msg` 必须一致。

    显式配置 `algo.bits_per_msg` 优先（给之后换精度/换维度的实验留口子），
    否则连续消息按 32 × msg_dim 换算，二值/无通信按 msg_bits。
    """
    algo = algo or {}
    if algo.get("bits_per_msg") is not None:
        return int(algo["bits_per_msg"])
    comm = str(algo.get("comm", "none"))
    if comm in CONTINUOUS_COMMS:
        return BITS_PER_DIM * int(algo.get("msg_dim", 8))
    return int(algo.get("msg_bits", 0) or 0)


def budget_for_name(algo: dict) -> str:
    """预算段的字符串：有限预算沿用旧写法，无上限写 `inf`。"""
    algo = algo or {}
    budget = algo.get("budget_bits", 0.0)
    try:
        value = float(budget)
    except (TypeError, ValueError):
        return str(budget)
    return "inf" if value >= INF_BUDGET else str(budget)


def _channel_off(value) -> bool:
    """标量 0 视为关闭信道；训练区间 [a, b] 不算（列表转 float 会失败）。"""
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def variant_suffix(cfg: dict) -> str:
    """消融变体短后缀；默认配置返回空串，既有目录名不受影响（contracts 第 6 节）。

    消融只改 `aggregate` / `use_gate` / 信道，运行名的六个字段全同，不加后缀会
    让 mean_agg 与 no_gate 写进同一个目录、后者静默覆盖前者。
    """
    a, e = cfg.get("algo") or {}, cfg.get("env") or {}
    parts = []

    aggregate = str(a.get("aggregate", "softmax"))
    if aggregate != "softmax":
        parts.append(f"agg{aggregate}")

    if a.get("use_gate", True) is not True:
        parts.append("nogate")

    channel = e.get("channel") or {}
    drop_off = _channel_off(channel.get("drop_p", 0.0))
    flip_off = _channel_off(channel.get("flip_p", 0.0))
    if drop_off and flip_off:
        parts.append("nonoise")
    elif drop_off:
        parts.append("drop0")
    elif flip_off:
        parts.append("flip0")

    return ("_" + "-".join(parts)) if parts else ""


def run_name(cfg: dict) -> str:
    """`{env}_{N}n_{algo}_L{bits}_K{recv_k}_B{budget}_s{seed}{变体后缀}`。"""
    e, a = cfg["env"], cfg["algo"]
    return (
        f"{e.get('scene', 'formation')}_{e['n_agents']}n_{a.get('name', a['comm'])}"
        f"_L{bits_per_msg_for_name(a)}_K{e.get('recv_k', 1)}"
        f"_B{budget_for_name(a)}_s{cfg['seed']}{variant_suffix(cfg)}"
    )
