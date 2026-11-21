"""把 BenchMARL/torchrl 导出的训练日志转成本仓库的 `metrics.csv` 格式（负责人 D）。

开题报告 §3.1 写的是「QMIX 采用 BenchMARL（TorchRL 后端）实现」。仓库内已经有一份
与统一环境同源的 QMIX（`src/bacmarl/runners/qmix.py`），因为它比 BenchMARL 更容易
保证「同一环境、同一奖励、同一指标」。这个脚本保留 BenchMARL 这条路：外部跑出来的
csv 用同一套列名并进 `results/`，两种来源在作图与汇总时一视同仁。

用法：

    python baselines/qmix_benchmarl/convert.py \
        --in benchmarl_qmix.csv \
        --out results/main/formation_6n_qmix_L0_K3_B0.0_s0

支持的输入列（大小写与常见别名都认，取第一个命中的）：

    env_steps     : env_steps, step, steps, frame, total_frames
    train/return  : train/return, return, episode_reward, episode_return, rollout/episode_reward
    train/success_rate : success_rate, train/success_rate
    train/formation_err: formation_err, train/formation_err
    train/collisions   : collisions, train/collisions

缺列会补 `nan`（数值列）或 0（通信列）；`comm/*` 一律 0，因为 QMIX 不通信。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

# 输出列与 `RunLogger` 的列集合保持同构（顺序不重要，作图按列名取）
OUT_COLUMNS = (
    "env_steps",
    "train/return",
    "train/success_rate",
    "train/formation_err",
    "train/collisions",
    "train/episode_length",
    "train/episodes",
    "comm/bits",
    "comm/speak_rate",
    "comm/beta",
)

ALIASES = {
    "env_steps": ("env_steps", "step", "steps", "frame", "total_frames", "global_step"),
    "train/return": (
        "train/return",
        "return",
        "episode_reward",
        "episode_return",
        "rollout/episode_reward",
        "reward",
    ),
    "train/success_rate": ("train/success_rate", "success_rate", "success"),
    "train/formation_err": ("train/formation_err", "formation_err"),
    "train/collisions": ("train/collisions", "collisions"),
    "train/episode_length": ("train/episode_length", "episode_length", "length"),
    "train/episodes": ("train/episodes", "episodes", "episode"),
}

_NUMERIC_FILL = {
    "train/return": math.nan,
    "train/success_rate": math.nan,
    "train/formation_err": math.nan,
    "train/collisions": math.nan,
    "train/episode_length": math.nan,
    "train/episodes": math.nan,
    "comm/bits": 0.0,       # QMIX 不通信
    "comm/speak_rate": 0.0,  # QMIX 无门控
    "comm/beta": 0.0,        # 无对偶变量
}


def _pick(row: dict, key: str):
    for name in ALIASES[key]:
        if name in row and row[name] not in ("", None):
            return row[name]
    return None


def _to_float(value):
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def convert_rows(rows: list[dict]) -> list[dict]:
    """把外部日志行映射成契约列；丢不掉的行按 env_steps 升序返回。"""
    out = []
    for row in rows:
        step = _to_float(_pick(row, "env_steps"))
        if step is None:
            continue
        rec = {"env_steps": int(step)}
        for key in OUT_COLUMNS:
            if key == "env_steps":
                continue
            value = _to_float(_pick(row, key)) if key in ALIASES else None
            rec[key] = _NUMERIC_FILL[key] if value is None else value
        out.append(rec)
    out.sort(key=lambda r: r["env_steps"])
    # 同一步多行时保留最后一条（断点续跑常见），避免图中出现重复 x
    dedup: dict[int, dict] = {}
    for rec in out:
        dedup[rec["env_steps"]] = rec
    return [dedup[k] for k in sorted(dedup)]


def convert_file(src: Path, dst: Path) -> int:
    with open(src, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    records = convert_rows(rows)
    if not records:
        raise SystemExit(f"[ERR] {src} 里没有可用的 env_steps 列，检查表头")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUT_COLUMNS))
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BenchMARL 日志 → metrics.csv")
    ap.add_argument("--in", dest="src", required=True, help="BenchMARL 导出的 csv")
    ap.add_argument(
        "--out",
        dest="dst",
        required=True,
        help="目标 run 目录（会写成 <dir>/metrics.csv）",
    )
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    dst = Path(args.dst)
    if dst.suffix.lower() != ".csv":
        dst = dst / "metrics.csv"
    n = convert_file(Path(args.src), dst)
    print(f"[DONE] {n} 行 → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
