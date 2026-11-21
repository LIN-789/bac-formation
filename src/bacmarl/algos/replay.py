"""QMIX 的转移回放（对比基线，负责人 D）。

只做「均匀随机采样 + 环形覆盖」这一层，不做优先经验回放：正式实验里 QMIX 与其它
方法在同一环境、同一总步数下比较，PER 会引入额外的调参维度，报告里说不清增益来自
值分解还是来自采样策略。

与 episode 级回放的差异（写在 docstring 里，审阅时容易对照）：这里存的是**转移**，
隐状态 `h_prev` / `next_h` 由采样期记录，更新时不再跨步反传（见 `algos/qmix.py`）。
好处是采样 i.i.d.、内存固定；代价是不做全轨迹 BPTT——与 repo 里 PPO 缓存 h_prev
的口径一致。
"""

from __future__ import annotations

import numpy as np
import torch


class TransitionReplay:
    """固定容量的环形缓冲；每条转移含下一状态，终止标记按 float [0/1] 存。"""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        n_obs: int,
        state_dim: int,
        hidden: int,
        n_agents: int,
    ):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.n_obs = int(n_obs)
        self.state_dim = int(state_dim)
        self.hidden = int(hidden)
        self.n_agents = int(n_agents)
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.obs_mask = np.zeros((capacity, n_agents, n_obs), dtype=bool)
        self.action = np.zeros((capacity, n_agents), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.h_prev = np.zeros((capacity, n_agents, hidden), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_obs_mask = np.zeros((capacity, n_agents, n_obs), dtype=bool)
        self.next_h = np.zeros((capacity, n_agents, hidden), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        obs,
        obs_mask,
        action,
        reward,
        done,
        state,
        h_prev,
        next_obs,
        next_obs_mask,
        next_h,
        next_state,
    ) -> None:
        """写入单条转移（n 步回报由调用方展开后再写，见 runners/qmix.py）。"""
        i = self.ptr
        self.obs[i] = np.asarray(obs, dtype=np.float32)
        self.obs_mask[i] = np.asarray(obs_mask, dtype=bool)
        self.action[i] = np.asarray(action, dtype=np.int64)
        self.reward[i] = float(reward)
        self.done[i] = float(done)
        self.state[i] = np.asarray(state, dtype=np.float32)
        self.h_prev[i] = np.asarray(h_prev, dtype=np.float32)
        self.next_obs[i] = np.asarray(next_obs, dtype=np.float32)
        self.next_obs_mask[i] = np.asarray(next_obs_mask, dtype=bool)
        self.next_h[i] = np.asarray(next_h, dtype=np.float32)
        self.next_state[i] = np.asarray(next_state, dtype=np.float32)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(
        self,
        obs,
        obs_mask,
        action,
        reward,
        done,
        state,
        h_prev,
        next_obs,
        next_obs_mask,
        next_h,
        next_state,
    ) -> None:
        """写入一批并行环境的转移（逐条环形写入）。"""
        obs = np.asarray(obs, dtype=np.float32)
        for k in range(obs.shape[0]):
            self.push(
                obs[k],
                np.asarray(obs_mask[k]),
                np.asarray(action[k]),
                float(reward[k]),
                float(done[k]),
                np.asarray(state[k]),
                np.asarray(h_prev[k]),
                np.asarray(next_obs[k]),
                np.asarray(next_obs_mask[k]),
                np.asarray(next_h[k]),
                np.asarray(next_state[k]),
            )

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        """均匀采样一个批，返回 torch 张量字典（键与 `qmix_td_loss` 对齐）。"""
        if self.size < 1:
            raise ValueError("replay is empty")
        idx = rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "obs_mask": torch.as_tensor(self.obs_mask[idx]),
            "action": torch.as_tensor(self.action[idx]),
            "reward": torch.as_tensor(self.reward[idx]),
            "done": torch.as_tensor(self.done[idx]),
            "state": torch.as_tensor(self.state[idx]),
            "h_prev": torch.as_tensor(self.h_prev[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "next_obs_mask": torch.as_tensor(self.next_obs_mask[idx]),
            "next_h": torch.as_tensor(self.next_h[idx]),
            "next_state": torch.as_tensor(self.next_state[idx]),
        }


class NStepWindow:
    """把逐环境的单步转移折叠成 n 步回报的转移，终止时提前截断。

    为什么需要 n 步：团队回报是每步 -e_t 的累加，100 步回合的 Q 量级在 -100 附近，
    而 1 步 TD 每次更新只把视野推进一步，200k 步预算内根本传不到那么远（实测 20k 步
    时 Q 只收敛到约 7 步的量级）。n=5 让每次更新传播 5 步视野，是本项目里 QMIX 能
    在与其他方法相同的步数预算内学到东西的前提。

    `algo.n_step=1` 时退化为普通 1 步 TD（与旧实现完全一致）。
    """

    def __init__(self, n_step: int, gamma: float, n_envs: int):
        if int(n_step) < 1:
            raise ValueError("n_step must be >= 1")
        self.n = int(n_step)
        self.gamma = float(gamma)
        self.pending: list[list[dict]] = [[] for _ in range(int(n_envs))]
        self.tail: list[dict | None] = [None] * int(n_envs)

    def set_tail(self, k: int, obs, obs_mask, h, state) -> None:
        """记录环境 k 最新一步之后的快照（n 步窗口右端要拿它当 next state）。"""
        self.tail[k] = {
            "obs": obs,
            "obs_mask": obs_mask,
            "h": h,  # 下一步的输入隐状态
            "state": state,
        }

    def push(self, k: int, item: dict) -> list[dict]:
        """压入环境 k 的一步；返回所有已凑满或已终止、可以写入 replay 的转移。"""
        buf = self.pending[k]
        buf.append(item)
        out: list[dict] = []
        while True:
            if len(buf) >= self.n:
                out.append(self._emit(k, self.n))
            elif buf and buf[-1]["done"]:
                out.append(self._emit(k, len(buf)))
            else:
                break
        return out

    def _emit(self, k: int, width: int) -> dict:
        buf = self.pending[k]
        total, cut = 0.0, None
        for j in range(width):
            total += (self.gamma**j) * float(buf[j]["reward"])
            if buf[j]["done"]:
                cut = j
                break
        if cut is not None:  # 窗口内有终止：截断且不 bootstrap
            width = cut + 1
            done = 1.0
        else:
            done = 0.0
        nxt = buf[width] if width < len(buf) else self.tail[k]
        if nxt is None:  # 还没设过 tail（不该发生），退化为用当前步的观测
            nxt = {**buf[min(width, len(buf) - 1)]}
            nxt["h"] = nxt["h_prev"]
        first = buf[0]
        rec = {
            "obs": first["obs"],
            "obs_mask": first["obs_mask"],
            "action": first["action"],
            "reward": total,
            "done": done,
            "state": first["state"],
            "h_prev": first["h_prev"],
            "next_obs": nxt["obs"],
            "next_obs_mask": nxt["obs_mask"],
            "next_h": nxt["h"],
            "next_state": nxt["state"],
        }
        buf.pop(0)
        return rec
