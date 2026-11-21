"""训练入口共用的运行时小工具：评测信道口径与配置落盘。

策略梯度入口（`runners/train.py`）与值分解入口（`runners/qmix.py`）必须用**同一份**
评测信道规则，否则两条学习范式的 `eval/*` 不可比；配置快照也按同一格式落盘，
截图/回放时能直接对照。
"""

from __future__ import annotations

from pathlib import Path


def eval_channel(env_dict: dict, train_cfg: dict) -> dict:
    """评测信道取固定值：优先 `train.eval_drop_p/eval_flip_p`，否则取训练区间中点。"""
    channel = (env_dict or {}).get("channel") or {}

    def fixed(key: str, override: str) -> float:
        if (train_cfg or {}).get(override) is not None:
            return float(train_cfg[override])
        value = channel.get(key, 0.0)
        if isinstance(value, (list, tuple)):
            return float(0.5 * (float(value[0]) + float(value[-1])))
        return float(value)

    return {
        "drop_p": fixed("drop_p", "eval_drop_p"),
        "flip_p": fixed("flip_p", "eval_flip_p"),
    }


def dump_yaml(cfg: dict, path: str | Path) -> None:
    """把合并后的配置写到运行目录；缺 PyYAML 时退回 JSON，不因此中断训练。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    except Exception:
        import json

        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
