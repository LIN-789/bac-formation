"""运行名与消融变体后缀测试（contracts 第 6 节）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bacmarl.runners.train import run_name  # noqa: E402
from bacmarl.utils.config import load_config  # noqa: E402


def _name(overrides: list[str]) -> str:
    return run_name(load_config(["env=formation", "algo=bac", *overrides]))


def test_default_name_has_no_variant_suffix():
    assert _name([]) == "formation_6n_bac_L4_K3_B2.0_s0"


def test_aggregate_variant_gets_suffix():
    assert _name(["algo.aggregate=mean"]).endswith("_s0_aggmean")


def test_no_gate_variant_gets_suffix():
    assert _name(["algo.use_gate=false"]).endswith("_s0_nogate")


def test_no_noise_variant_gets_suffix():
    assert _name(["env.channel.drop_p=0", "env.channel.flip_p=0"]).endswith(
        "_s0_nonoise"
    )


def test_single_channel_knob_gets_suffix():
    assert _name(["env.channel.drop_p=0"]).endswith("_s0_drop0")
    assert _name(["env.channel.flip_p=0"]).endswith("_s0_flip0")


def test_variants_never_share_a_directory():
    """#10：mean_agg 与 no_gate 曾写进同一目录并静默覆盖。"""
    names = {
        _name(["algo.aggregate=mean"]),
        _name(["algo.use_gate=false"]),
        _name(["env.channel.drop_p=0", "env.channel.flip_p=0"]),
        _name([]),
    }
    assert len(names) == 4


def test_combined_variants_are_still_distinct():
    assert _name(["algo.aggregate=mean", "algo.use_gate=false"]).endswith(
        "_s0_aggmean-nogate"
    )


@pytest.mark.parametrize("name", ["bac", "ippo", "ic3net", "tarmac", "qmix"])
def test_run_name_algo_component_matches_config(name):
    cfg = load_config(["env=formation", f"algo={name}"])
    assert f"_{name}_" in run_name(cfg)


# ---- Issue #11：连续消息基线的 L 段必须写真实比特宽度 ----

@pytest.mark.parametrize("name", ["ic3net", "tarmac"])
def test_continuous_message_baselines_name_real_bit_width(name):
    """ic3net/tarmac 的 msg_bits=0 只是占位，名字里不能写 L0。"""
    cfg = load_config(["env=formation", f"algo={name}"])
    text = run_name(cfg)
    assert "_L1024_" in text, text
    assert "_L0_" not in text, text
    assert text.endswith("_s0"), text


@pytest.mark.parametrize("name", ["ic3net", "tarmac"])
def test_unbounded_budget_is_written_as_inf(name):
    assert "_Binf_" in run_name(load_config(["env=formation", f"algo={name}"]))


def test_no_communication_baselines_stay_at_l0_b0():
    """MAPPO/IPPO 确实不通信，L0/B0.0 是正确口径，不能被连续消息的修法带跑。"""
    for name in ("mappo", "ippo"):
        text = run_name(load_config(["env=formation", f"algo={name}"]))
        assert "_L0_K3_B0.0_" in text, text


def test_bit_width_matches_comm_module():
    """名字里的 L 必须等于通信模块上报的 bits_per_msg（防口径再次漂移）。"""
    pytest.importorskip("torch")
    from bacmarl.comm.base import build_comm
    from bacmarl.utils.naming import bits_per_msg_for_name

    for name in ("bac", "ic3net", "tarmac", "mappo"):
        algo = load_config(["env=formation", f"algo={name}"])["algo"]
        comm = build_comm(algo["comm"], hidden=8, cfg=algo)
        assert bits_per_msg_for_name(algo) == comm.bits_per_msg, name


def test_explicit_bits_per_msg_wins_over_derivation():
    """以后换精度/换维度时可以直接写 algo.bits_per_msg，不必改代码。"""
    from bacmarl.utils.naming import bits_per_msg_for_name

    algo = {"comm": "tarmac", "msg_dim": 32, "bits_per_msg": 512}
    assert bits_per_msg_for_name(algo) == 512

