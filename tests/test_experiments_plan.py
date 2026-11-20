"""批量实验驱动的排班测试（负责人 E）。

只测「计划」这一层：展开、撞名、评测信道对齐、与 train.run_name 口径一致。
不涉及训练本身，缺 torch 也能跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in ("scripts", "src"):
    p = str(ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

import run_experiments as rx  # noqa: E402


def test_ablation_gets_its_own_group_and_never_collides_with_main():
    jobs = rx.plan(["main", "ablation"], seeds=[0, 1], total_env_steps=2000)
    keys = [(j["exp_name"], j["run_name"]) for j in jobs]
    assert len(keys) == len(set(keys)), "消融与主实验不能落在同一个 results 目录"
    assert not rx.find_collisions(jobs)
    assert {j["exp_name"] for j in jobs if j["group"] == "ablation"} == {
        "ablation_no_gate",
        "ablation_mean_agg",
        "ablation_no_noise",
    }
    # 主实验仍在 main 组
    assert {j["exp_name"] for j in jobs if j["group"] == "main"} == {"main"}


def test_no_gate_pins_dual_step_and_keeps_run_name():
    job = next(
        j for j in rx.plan(["ablation"], seeds=[0], total_env_steps=1000) if j["variant"] == "no_gate"
    )
    assert job["cfg"]["algo"]["use_gate"] is False
    assert job["cfg"]["algo"]["beta_lr"] == 0.0
    # run_name 不含开关，只是落在不同实验组，名字本身与主实验一致
    assert job["run_name"].startswith("formation_6n_bac_L4_K3_B2.0_s0")


def test_mean_agg_switches_aggregate_only():
    job = next(
        j for j in rx.plan(["ablation"], seeds=[0], total_env_steps=1000) if j["variant"] == "mean_agg"
    )
    assert job["cfg"]["algo"]["aggregate"] == "mean"
    assert job["cfg"]["algo"]["use_gate"] is True


def test_no_noise_keeps_eval_channel_at_training_midpoint():
    job = next(
        j for j in rx.plan(["ablation"], seeds=[0], total_env_steps=1000) if j["variant"] == "no_noise"
    )
    env_channel = job["cfg"]["env"]["channel"]
    assert env_channel["drop_p"] == 0.0 and env_channel["flip_p"] == 0.0
    # 训练区间 [0, 0.3] / [0, 0.1] 的中点，否则评测也在零噪声下跑
    assert job["cfg"]["train"]["eval_drop_p"] == pytest.approx(0.15)
    assert job["cfg"]["train"]["eval_flip_p"] == pytest.approx(0.05)


def test_sweep_is_single_factor_and_splits_by_axis():
    jobs = rx.plan(["sweep"], seeds=[0], total_env_steps=1000)
    # 每轴只扫非基线的点：msg_bits 3 点 + recv_k 3 点（默认 K=3 不在列表里）
    # + budget_bits 2 点 = 8，不是 4×3×3=36 的笛卡尔积；基线由 main 组提供
    assert len(jobs) == 8
    assert {j["exp_name"] for j in jobs} == {
        "sweep_algo_msg_bits",
        "sweep_env_recv_k",
        "sweep_algo_budget_bits",
    }
    assert not rx.find_collisions(jobs)


def test_include_baseline_doubles_back_the_default_points():
    jobs = rx.plan(["sweep"], seeds=[0], total_env_steps=1000, include_baseline=True)
    assert len(jobs) == 10
    assert not rx.find_collisions(jobs)


def test_collision_guard_flags_same_experiment_and_run_name():
    jobs = [
        {"exp_name": "main", "run_name": "formation_6n_bac_L4_K3_B2.0_s0", "variant": "main"},
        {"exp_name": "main", "run_name": "formation_6n_bac_L4_K3_B2.0_s0", "variant": "mean_agg"},
    ]
    problems = rx.find_collisions(jobs)
    assert problems and "撞名" in problems[0]


def test_seeds_expand_every_variant():
    jobs = rx.plan(["ablation"], seeds=[0, 1, 2], total_env_steps=1000)
    assert len(jobs) == 9
    assert sorted({j["seed"] for j in jobs}) == [0, 1, 2]


def test_total_env_steps_override_reaches_config():
    job = rx.plan(["main"], seeds=[0], total_env_steps=2000)[0]
    assert job["cfg"]["train"]["total_env_steps"] == 2000
    assert "train.total_env_steps=2000" in job["tokens"]


def test_unknown_group_is_rejected():
    with pytest.raises(ValueError):
        rx.plan(["nope"], seeds=[0])


def test_run_name_matches_train_module():
    tr = pytest.importorskip("bacmarl.runners.train")
    job = rx.plan(["main"], seeds=[7], total_env_steps=1000)[0]
    assert rx.run_name(job["cfg"]) == tr.run_name(job["cfg"])
