"""QMIX 能不能学：一个可判定的玩具任务（负责人 D）。

真实编队任务在 20k 步内学不出来是实验结论（预算/难度问题），但「学不出来」和
「实现写错」必须分开。这里用一个判定明确的玩具环境：每个智能体选动作 1 得 0 分、
选其它得 -1（奖励按个体可分离）。跑几百步后贪心策略必须稳定选中动作 1，否则说明
TD / 混合网络 / 回放 / n 步这些环节里有真 bug。

玩具环境实现 `VecFormationEnv` 的最小接口，不依赖 mpe2，CI 里秒级完成。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")

from bacmarl.runners.qmix import _train, build_agent, load_qmix_checkpoint  # noqa: E402
from bacmarl.utils.config import load_config  # noqa: E402
from bacmarl.utils.seeding import set_seed  # noqa: E402

OPTIMAL_ACTION = 1


class ToyVecEnv:
    """每个智能体选 1 得 0 分、选其它得 -1（奖励按个体可分离，不考协同探索）。

    可分离是有意的：这条测试要验证「TD 目标 / n 步展开 / 回放 / 混合 / 贪心执行」这条
    链路能不能把动作价值学出来，而不是再考一次 QMIX 的协同探索能力（那是 §5.2 的实验
    结论）。若奖励要求「全员同时选 1」，随机探索撞上正样本的概率极低，测试会随执行顺序飘。
    """

    def __init__(self, n_envs: int = 4, n_agents: int = 2, max_steps: int = 6):
        self.n_envs = n_envs
        self.cfg = SimpleNamespace(n_agents=n_agents, n_obs=2)
        self.obs_dim = 6
        self.state_dim = 2 * n_agents
        self.max_steps = max_steps
        self.envs = [SimpleNamespace(backend="toy")]
        self._t = 0
        self.reset()

    def reset(self):
        self._t = 0
        return self._batch(0.0)

    def _batch(self, obs_value) -> dict:
        b, n = self.n_envs, self.cfg.n_agents
        return {
            "obs": np.full((b, n, self.obs_dim), float(obs_value), dtype=np.float32),
            "obs_mask": np.ones((b, n, self.cfg.n_obs), dtype=bool),
            "state": np.zeros((b, self.state_dim), dtype=np.float32),
            "comm_mask": np.zeros((b, n, n), dtype=bool),
            "channel": {"drop_p": np.zeros(b, dtype=np.float32),
                        "flip_p": np.zeros(b, dtype=np.float32)},
            "reward": np.zeros((b, n), dtype=np.float32),
            "done": np.zeros(b, dtype=bool),
            "ep": [None] * b,
        }

    def step(self, actions: np.ndarray) -> dict:
        b, n = self.n_envs, self.cfg.n_agents
        actions = np.asarray(actions)
        reward = np.where(actions == OPTIMAL_ACTION, 0.0, -1.0).astype(np.float32)
        self._t += 1
        done = np.full(b, self._t >= self.max_steps)
        batch = self._batch(0.0)
        batch["reward"] = reward
        batch["done"] = done
        if done.any():
            batch["ep"] = [
                {"ret": float(reward[k].mean()) * self.max_steps, "len": self._t,
                 "collisions": 0, "success": False, "formation_err": 0.0}
                if done[k] else None
                for k in range(b)
            ]
        return batch


def _cfg():
    return load_config(
        [
            "env=formation",
            "algo=qmix",
            "train.total_env_steps=600",
            "train.n_envs=4",
            "train.hidden=8",
            "train.eval_every=0",
            "algo.train_start=64",
            "algo.train_every=1",
            "algo.updates_per_train=4",
            "algo.batch_size=32",
            "algo.target_update=50",
            "algo.replay_size=2000",
            "algo.eps_anneal_steps=300",
            "algo.eps_end=0.05",
            "algo.n_step=1",
            "algo.reward_scale=1.0",
            "algo.use_agent_id=true",
            "experiment.name=pytest_qmix_toy",
            "seed=0",
        ]
    )


def test_qmix_learns_a_deterministic_toy_task():
    cfg = _cfg()
    # 直接调 _train 绕过了 main() 的 set_seed：整套测试一起跑时 RNG 状态会不同，
    # 不在这里显式播种的话这条测试会随执行顺序飘。
    set_seed(0)
    vec = ToyVecEnv()
    assert _train(cfg, vec) == 0

    run_dir = ROOT / "results" / "pytest_qmix_toy" / "formation_6n_qmix_L0_K3_B0.0_s0"
    agent = build_agent(cfg, vec)
    load_qmix_checkpoint(run_dir / "ckpt" / "last.ckpt", agent)
    agent.eval()

    obs = torch.zeros(1, vec.cfg.n_agents, vec.obs_dim)
    mask = torch.ones(1, vec.cfg.n_agents, vec.cfg.n_obs, dtype=torch.bool)
    with torch.no_grad():
        q, _ = agent.q_values(obs, mask, agent.init_hidden(1, vec.cfg.n_agents))
    picked = q.argmax(-1)[0].tolist()
    assert picked == [OPTIMAL_ACTION] * vec.cfg.n_agents, picked
