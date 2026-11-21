"""QMIX 基线测试（负责人 D）。

覆盖：形状、混合网络的单调性、Double-Q 目标、ε 退火、replay 环形覆盖、TD 可下降、
与统一评测入口（`evaluate_policy`）的接口兼容，以及运行名口径。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from bacmarl.algos.qmix import (  # noqa: E402
    AgentQNet,
    MixingNet,
    QMixAgent,
    epsilon_at,
    qmix_td_loss,
    td_target,
)
from bacmarl.algos.replay import NStepWindow, TransitionReplay  # noqa: E402
from bacmarl.envs.formation_comm import EnvConfig  # noqa: E402
from bacmarl.runners.evaluate import evaluate_policy  # noqa: E402
from bacmarl.runners.qmix import GreedyQPolicy, build_agent  # noqa: E402
from bacmarl.utils.config import load_config  # noqa: E402
from bacmarl.utils.naming import run_name  # noqa: E402

B, N, OBS_DIM, N_OBS, STATE, H, A = 2, 3, 6, 2, 10, 8, 5


def _agent() -> QMixAgent:
    return QMixAgent(
        obs_dim=OBS_DIM, n_obs=N_OBS, n_actions=A, hidden=H, n_agents=N,
        state_dim=STATE, mixer_embed=8, hypernet_embed=16,
    )


def _batch(k: int = 8) -> dict:
    return {
        "obs": torch.randn(k, N, OBS_DIM),
        "obs_mask": torch.rand(k, N, N_OBS) > 0.5,
        "action": torch.randint(0, A, (k, N)),
        "reward": torch.randn(k),
        "done": (torch.rand(k) > 0.8).float(),
        "state": torch.randn(k, STATE),
        "h_prev": torch.randn(k, N, H),
        "next_obs": torch.randn(k, N, OBS_DIM),
        "next_obs_mask": torch.rand(k, N, N_OBS) > 0.5,
        "next_h": torch.randn(k, N, H),
        "next_state": torch.randn(k, STATE),
    }


# ---- 结构 ----

def test_agent_q_shapes():
    q, h = _agent().q_values(torch.randn(B, N, OBS_DIM), torch.ones(B, N, N_OBS, dtype=torch.bool),
                            torch.zeros(B, N, H))
    assert q.shape == (B, N, A)
    assert h.shape == (B, N, H)


def test_agent_id_one_hot_breaks_symmetry():
    """没有 agent_id 时同构无人机 Q 完全相同，贪心动作会整队一致（编队会漂）。"""
    obs = torch.zeros(B, N, OBS_DIM)
    mask = torch.ones(B, N, N_OBS, dtype=torch.bool)
    h = torch.zeros(B, N, H)

    plain = QMixAgent(obs_dim=OBS_DIM, n_obs=N_OBS, n_actions=A, hidden=H, n_agents=N,
                      state_dim=STATE, mixer_embed=8, hypernet_embed=16,
                      use_agent_id=False)
    q_plain, _ = plain.q_values(obs, mask, h)
    assert torch.allclose(q_plain[0, 0], q_plain[0, 1])

    with_id = QMixAgent(obs_dim=OBS_DIM, n_obs=N_OBS, n_actions=A, hidden=H, n_agents=N,
                        state_dim=STATE, mixer_embed=8, hypernet_embed=16,
                        use_agent_id=True)
    q_id, _ = with_id.q_values(obs, mask, h)
    assert not torch.allclose(q_id[0, 0], q_id[0, 1])  # 不同 agent 可区分
    assert torch.allclose(q_id[0, 0], q_id[1, 0])      # 同一 agent 跨 batch 一致



def test_q_tot_shapes_and_action_selection():
    agent = _agent()
    batch = _batch(4)
    q, q_tot, h = agent(batch["obs"], batch["obs_mask"], batch["h_prev"], batch["state"],
                        batch["action"])
    assert q.shape == (4, N, A)
    assert q_tot.shape == (4,)
    # 用同一批数据只换动作，Q_tot 必须随所选动作的 Q 变化（而不是恒定）
    other = agent(batch["obs"], batch["obs_mask"], batch["h_prev"], batch["state"],
                  (batch["action"] + 1) % A)[1]
    assert not torch.allclose(q_tot, other)


def test_mixing_is_monotonic_in_each_agent_q():
    """∂Q_tot/∂Q_i ≥ 0：个体贪心动作在联合 Q 上仍单调不降（值分解的前提）。"""
    torch.manual_seed(0)
    mixer = MixingNet(n_agents=N, state_dim=STATE, embed=8, hypernet_embed=16)
    state = torch.randn(1, STATE)
    base = torch.zeros(1, N)
    value = mixer(base, state).item()
    for i in range(N):
        bumped = base.clone()
        bumped[0, i] = 1.0
        assert mixer(bumped, state).item() >= value - 1e-6


def test_mixing_weights_follow_state():
    """混合权重由全局状态经超网络生成，不是一组与状态无关的常数。"""
    torch.manual_seed(1)
    mixer = MixingNet(n_agents=N, state_dim=STATE, embed=8, hypernet_embed=16)
    qs = torch.randn(1, N)
    a = mixer(qs, torch.zeros(1, STATE)).item()
    b = mixer(qs, torch.ones(1, STATE)).item()
    assert a != pytest.approx(b)


# ---- 目标与调度 ----

def test_td_target_ignores_bootstrap_on_done():
    r = torch.tensor([1.0, 1.0])
    done = torch.tensor([0.0, 1.0])
    q_next = torch.tensor([10.0, 10.0])
    y = td_target(r, done, q_next, gamma=0.9)
    assert y[0].item() == pytest.approx(1.0 + 0.9 * 10.0)
    assert y[1].item() == pytest.approx(1.0)  # 终止转移只保留 r


def test_epsilon_anneals_linearly_and_clamps():
    assert epsilon_at(0, 1.0, 0.05, 100) == pytest.approx(1.0)
    assert epsilon_at(50, 1.0, 0.05, 100) == pytest.approx(0.525)
    assert epsilon_at(100, 1.0, 0.05, 100) == pytest.approx(0.05)
    assert epsilon_at(10_000, 1.0, 0.05, 100) == pytest.approx(0.05)
    assert epsilon_at(0, 1.0, 0.05, 0) == pytest.approx(0.05)


# ---- replay ----

def test_replay_sample_shapes_and_keys():
    replay = TransitionReplay(64, OBS_DIM, N_OBS, STATE, H, N)
    rng = np.random.default_rng(0)
    for _ in range(3):
        replay.push_batch(
            obs=np.zeros((B, N, OBS_DIM), dtype=np.float32),
            obs_mask=np.ones((B, N, N_OBS), dtype=bool),
            action=np.zeros((B, N), dtype=np.int64),
            reward=np.ones(B, dtype=np.float32),
            done=np.zeros(B, dtype=np.float32),
            state=np.zeros((B, STATE), dtype=np.float32),
            h_prev=np.zeros((B, N, H), dtype=np.float32),
            next_obs=np.zeros((B, N, OBS_DIM), dtype=np.float32),
            next_obs_mask=np.ones((B, N, N_OBS), dtype=bool),
            next_h=np.zeros((B, N, H), dtype=np.float32),
            next_state=np.zeros((B, STATE), dtype=np.float32),
        )
    assert len(replay) == 6
    sample = replay.sample(4, rng)
    assert sample["obs"].shape == (4, N, OBS_DIM)
    assert sample["action"].shape == (4, N)
    assert sample["reward"].shape == (4,)
    assert sample["obs_mask"].dtype == torch.bool


def test_replay_wraps_around_and_keeps_capacity():
    replay = TransitionReplay(5, OBS_DIM, N_OBS, STATE, H, N)
    rng = np.random.default_rng(0)
    for step in range(12):
        replay.push_batch(
            obs=np.full((1, N, OBS_DIM), float(step), dtype=np.float32),
            obs_mask=np.ones((1, N, N_OBS), dtype=bool),
            action=np.zeros((1, N), dtype=np.int64),
            reward=np.zeros(1, dtype=np.float32),
            done=np.zeros(1, dtype=np.float32),
            state=np.zeros((1, STATE), dtype=np.float32),
            h_prev=np.zeros((1, N, H), dtype=np.float32),
            next_obs=np.zeros((1, N, OBS_DIM), dtype=np.float32),
            next_obs_mask=np.ones((1, N, N_OBS), dtype=bool),
            next_h=np.zeros((1, N, H), dtype=np.float32),
            next_state=np.zeros((1, STATE), dtype=np.float32),
        )
    assert len(replay) == 5
    sample = replay.sample(64, rng)
    # 只保留最近 5 条（7..11），不会读到被覆盖的旧数据
    assert sample["obs"].max().item() <= 11.0
    assert sample["obs"].min().item() >= 7.0


# ---- 学习与集成 ----

def test_td_loss_decreases_on_a_fixed_batch():
    torch.manual_seed(0)
    agent = _agent()
    target = agent.target_copy()
    opt = torch.optim.Adam(agent.parameters(), lr=5e-3)
    batch = _batch(32)
    first = None
    for _ in range(40):
        loss, stats = qmix_td_loss(agent, target, batch, gamma=0.99)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = stats.loss if first is None else first
    assert stats.loss < first, (first, stats.loss)
    assert np.isfinite(stats.loss)


def test_greedy_policy_is_compatible_with_evaluate_policy():
    """QMIX 必须能走统一评测入口，否则 eval/* 与 robust.csv 不可比。"""
    from bacmarl.envs.formation_comm import FormationCommEnv

    cfg = load_config(["env=formation", "algo=qmix"])
    env_cfg = EnvConfig.from_dict({**cfg["env"], "backend": "mock", "n_agents": 3,
                                   "max_steps": 5})
    probe = FormationCommEnv(env_cfg)
    try:
        agent = QMixAgent(
            obs_dim=probe.obs_dim, n_obs=env_cfg.n_obs, n_actions=A, hidden=H,
            n_agents=env_cfg.n_agents, state_dim=probe.state_dim,
            mixer_embed=8, hypernet_embed=16,
        )
    finally:
        probe.close()
    metrics = evaluate_policy(GreedyQPolicy(agent), env_cfg, episodes=1, seed=0)
    assert set(metrics) >= {"return", "success_rate", "formation_err", "collisions"}
    assert metrics["bits"] == 0.0
    assert metrics["speak_rate"] == 0.0


def test_qmix_run_name_has_zero_bits_and_budget():
    cfg = load_config(["env=formation", "algo=qmix"])
    assert run_name(cfg) == "formation_6n_qmix_L0_K3_B0.0_s0"


def test_qmix_agent_builds_from_config():
    from bacmarl.envs.formation_comm import FormationCommEnv

    cfg = load_config(["env=formation", "algo=qmix"])
    env_cfg = EnvConfig.from_dict(cfg["env"])
    probe = FormationCommEnv(env_cfg)
    try:
        obs_dim, state_dim = probe.obs_dim, probe.state_dim
    finally:
        probe.close()

    class _Vec:  # 只需要构造网络时用到的三个属性
        pass

    _Vec.obs_dim = obs_dim
    _Vec.state_dim = state_dim
    _Vec.cfg = env_cfg
    agent = build_agent(cfg, _Vec())
    assert agent.agent.q_head[-1].out_features == A
    assert agent.mixer.n_agents == env_cfg.n_agents


# ---- n 步回报 ----

def _item(reward: float, done: bool, obs_id: int = 0) -> dict:
    return {
        "obs": np.full((N, OBS_DIM), float(obs_id), dtype=np.float32),
        "obs_mask": np.ones((N, N_OBS), dtype=bool),
        "action": np.zeros(N, dtype=np.int64),
        "reward": reward,
        "done": done,
        "state": np.zeros(STATE, dtype=np.float32),
        "h_prev": np.zeros((N, H), dtype=np.float32),
    }


def test_n_step_accumulates_discounted_reward():
    win = NStepWindow(n_step=3, gamma=0.5, n_envs=1)
    out = []
    for step, r in enumerate([1.0, 2.0, 4.0, 8.0]):
        win.set_tail(0, np.full((N, OBS_DIM), step + 1.0, dtype=np.float32),
                     np.ones((N, N_OBS), dtype=bool), np.zeros((N, H)), np.zeros(STATE))
        out.extend(win.push(0, _item(r, False, step)))
    assert len(out) == 2  # 4 步、n=3 → 前两步凑满后各吐一条（窗口重叠）
    assert out[0]["reward"] == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 4.0)
    assert out[0]["done"] == 0.0
    assert out[1]["reward"] == pytest.approx(2.0 + 0.5 * 4.0 + 0.25 * 8.0)
    # 窗口右端之后的那一步：第 3 次 push 时 tail 是 obs=3
    assert out[0]["next_obs"][0, 0] == pytest.approx(3.0)
    assert out[1]["next_obs"][0, 0] == pytest.approx(4.0)


def test_n_step_truncates_at_terminal_and_skips_bootstrap():
    win = NStepWindow(n_step=5, gamma=0.9, n_envs=1)
    win.set_tail(0, np.ones((N, OBS_DIM), dtype=np.float32),
                 np.ones((N, N_OBS), dtype=bool), np.zeros((N, H)), np.zeros(STATE))
    win.push(0, _item(1.0, False, 0))
    win.set_tail(0, np.ones((N, OBS_DIM), dtype=np.float32),
                 np.ones((N, N_OBS), dtype=bool), np.zeros((N, H)), np.zeros(STATE))
    out = win.push(0, _item(2.0, True, 1))
    # 终止把窗口切碎：先吐「1 → 终止」这一条，再吐终止步自己那一条
    assert len(out) == 2
    assert out[0]["reward"] == pytest.approx(1.0 + 0.9 * 2.0)  # 不越过终止步
    assert out[0]["done"] == 1.0  # 终止转移不 bootstrap
    assert out[1]["reward"] == pytest.approx(2.0)
    assert out[1]["done"] == 1.0


def test_n_step_one_is_plain_td():
    win = NStepWindow(n_step=1, gamma=0.99, n_envs=1)
    win.set_tail(0, np.ones((N, OBS_DIM), dtype=np.float32),
                 np.ones((N, N_OBS), dtype=bool), np.zeros((N, H)), np.zeros(STATE))
    out = win.push(0, _item(-1.5, False))
    assert len(out) == 1
    assert out[0]["reward"] == pytest.approx(-1.5)
    assert out[0]["done"] == 0.0

