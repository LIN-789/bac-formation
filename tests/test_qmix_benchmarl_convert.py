"""BenchMARL 日志转换脚本的测试（负责人 D，不依赖 torch/benchmarl）。"""

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines" / "qmix_benchmarl"))

import convert  # noqa: E402


def test_maps_common_benchmarl_columns():
    rows = [
        {"step": "1000", "episode_reward": "-42.5", "success_rate": "0.25"},
        {"step": "500", "episode_reward": "-43.5", "success_rate": "0.125"},
    ]
    out = convert.convert_rows(rows)
    assert [r["env_steps"] for r in out] == [500, 1000]  # 按步数排序
    assert out[-1]["train/return"] == pytest.approx(-42.5)
    assert out[-1]["train/success_rate"] == pytest.approx(0.25)
    # QMIX 不通信：通信列必须是 0 而不是 nan
    assert out[-1]["comm/bits"] == 0.0
    assert out[-1]["comm/speak_rate"] == 0.0
    # 外部日志没有的列补 nan，表头保持契约同构
    assert out[-1]["train/collisions"] != out[-1]["train/collisions"]  # nan


def test_dedupes_repeated_steps_and_drops_rows_without_steps():
    rows = [
        {"env_steps": "8", "return": "1.0"},
        {"env_steps": "8", "return": "2.0"},
        {"return": "3.0"},  # 没有步数的行丢掉
    ]
    out = convert.convert_rows(rows)
    assert len(out) == 1
    assert out[0]["train/return"] == pytest.approx(2.0)  # 保留最后一条


def test_file_roundtrip_writes_contract_columns(tmp_path: Path):
    src = tmp_path / "benchmarl.csv"
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "episode_reward"])
        w.writeheader()
        w.writerow({"step": 256, "episode_reward": -10.0})
        w.writerow({"step": 512, "episode_reward": -9.0})
    dst = tmp_path / "run" / "metrics.csv"
    n = convert.convert_file(src, dst)
    assert n == 2
    with open(dst, newline="", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert header == list(convert.OUT_COLUMNS)


def test_empty_input_fails_loudly(tmp_path: Path):
    src = tmp_path / "empty.csv"
    src.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        convert.convert_file(src, tmp_path / "out" / "metrics.csv")
