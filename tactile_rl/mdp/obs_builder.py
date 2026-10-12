"""The student's observation (tactile_sim.control.obs_spec) batched over environments: the torch mirror of
ResidualHandL2.fast_sample / sample / observe.

Every millisecond (`sample`, then `push`) the fast inputs are scaled into one row, as the testbed's 1 kHz sample does:
taxel codes / (levels - 1); per patch the features sum / 100, CoP row / (256 rows) and column / (256 columns) (-1
while the patch reads no load), area / taxels, peak / 1000, rate / 100; per board the age of its newest frame in 5 ms
units (1 before any) and the frames received since the previous sample; the hand's encoder angles; the motor currents
over the rated currents. A ring keeps the newest `stack` rows (newest first). `observe` lays them out by block (each
fast block holds its `stack` samples, newest first), then the policy-rate blocks (last action, the base's hand target
and its feedforward over the rated torques now, the arm's lookahead, the forecast, the task, trigger_enable) and
zeroes the masked blocks.
"""

from __future__ import annotations

import numpy as np
import torch

from tactile_sim.control.obs_spec import BOARDS, N_JOINTS, OBS_BLOCKS, TASKS, obs_dim, obs_slices

FAST_BLOCKS = [(name, d) for name, d, fast, _ in OBS_BLOCKS if fast]
FAST_DIM = sum(d for _, d in FAST_BLOCKS)


class StudentObs:
    def __init__(self, num_envs: int, stack: int, grids: list[tuple[int, int]], levels: int,
                 mask_blocks: tuple[str, ...] = (), task: str = "hammer", device: str | torch.device = "cpu",
                 tau_rated=None):
        self.n_env, self.stack, self.dev = int(num_envs), int(stack), torch.device(device)
        # the hand joints' rated torques (the base_hand_tau block's scale; obs_spec.base_tau_obs)
        self.tau_rated = torch.as_tensor(np.ones(N_JOINTS) if tau_rated is None else np.asarray(tau_rated, float),
                                         dtype=torch.float32, device=self.dev)
        self.levels = int(levels)
        self.rows = torch.as_tensor([g[0] for g in grids], dtype=torch.float64, device=self.dev)
        self.cols = torch.as_tensor([g[1] for g in grids], dtype=torch.float64, device=self.dev)
        self.P = len(grids)
        slices = obs_slices(self.stack)
        unknown = set(mask_blocks) - set(slices)
        if unknown:
            raise ValueError(f"unknown observation blocks to mask: {sorted(unknown)}")
        self.mask_blocks = tuple(mask_blocks)
        self.masked = [slices[b] for b in mask_blocks]
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        self.task_idx = torch.full((self.n_env,), TASKS.index(task), dtype=torch.long, device=self.dev)
        self._eye = torch.eye(len(TASKS), device=self.dev)
        self.dim = obs_dim(self.stack)
        self.ring = torch.zeros(self.n_env, self.stack, FAST_DIM, device=self.dev)
        self.last_seq = torch.full((self.n_env, BOARDS), -1, dtype=torch.long, device=self.dev)
        edges = np.cumsum([0] + [d for _, d in FAST_BLOCKS])
        self.fast_slices = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]

    def set_task(self, env_ids, task: str) -> None:
        """These envs' task (the one-hot block; the C3 mix runs hammer and load-generator episodes side by side)."""
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        self.task_idx[env_ids] = TASKS.index(task)

    def reset(self, env_ids=None) -> None:
        """As ResidualHandL2.reset: an empty ring and no frame seen yet."""
        ids = slice(None) if env_ids is None else env_ids
        self.ring[ids] = 0.0
        self.last_seq[ids] = -1

    def sample(self, t: torch.Tensor, codes: torch.Tensor, feats: torch.Tensor, board_seq: torch.Tensor,
               board_avail: torch.Tensor, q: torch.Tensor, cur_ratio: torch.Tensor,
               update: torch.Tensor | None = None) -> torch.Tensor:
        """One 1 kHz row (N, FAST_DIM) at time t (N,): codes (N, taxels), features (N, P, 6) int, the boards' newest
        frame (seq (N, B), t_avail (N, B)), encoder angles (N, 20), currents / rated (N, 20). The boards' frame
        counters advance where `update` (default all), as the testbed's sample does."""
        f = feats.double()
        loaded = f[..., 0] > 0
        neg = torch.full_like(f[..., 0], -1.0)
        fe = torch.stack([f[..., 0] / 100.0, torch.where(loaded, f[..., 1] / (256.0 * self.rows), neg),
                          torch.where(loaded, f[..., 2] / (256.0 * self.cols), neg),
                          f[..., 3] / (self.rows * self.cols), f[..., 4] / 1000.0, f[..., 5] / 100.0], dim=-1)
        has = board_seq >= 0
        age = torch.where(has, (t[:, None] - board_avail) / 0.005, torch.ones_like(board_avail))
        new = torch.where(has, (board_seq - self.last_seq).double(), torch.zeros_like(board_avail))
        upd = torch.ones(self.n_env, dtype=torch.bool, device=self.dev) if update is None else update
        self.last_seq = torch.where(upd[:, None], board_seq, self.last_seq)
        status = torch.stack([age, new], dim=-1).reshape(self.n_env, 2 * BOARDS)
        row = torch.cat([codes.double() / (self.levels - 1), fe.reshape(self.n_env, -1), status, q.double(),
                         cur_ratio.double()], dim=1)
        return row.float()

    def push(self, row: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """The newest row in front, the oldest out, where `mask` (default all)."""
        rolled = torch.cat([row[:, None], self.ring[:, :-1]], dim=1)
        self.ring = rolled if mask is None else torch.where(mask[:, None, None], rolled, self.ring)

    def observe(self, last_action: torch.Tensor, base: torch.Tensor, lookahead: torch.Tensor,
                forecast: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
        """The student's observation (N, obs_dim(stack)), obs_spec order, from the base's MIT command (N, 5, J) and the
        arm's lookahead (N, 7 x len(LOOKAHEAD))."""
        parts = [self.ring[:, :, sl].reshape(self.n_env, -1) for sl in self.fast_slices]
        base_tau = base[:, 4].float() / self.tau_rated
        obs = torch.cat(parts + [last_action.float(), base[:, 0].float(), base_tau, lookahead.float(),
                                 forecast.float(), self._eye[self.task_idx], trigger.float().reshape(-1, 1)], dim=1)
        if obs.shape[1] != self.dim:
            raise ValueError(f"observation has {obs.shape[1]} entries, the spec {self.dim}")
        for sl in self.masked:
            obs[:, sl] = 0.0
        return obs


assert FAST_DIM == 448 + 6 * 11 + 2 * BOARDS + 2 * N_JOINTS
