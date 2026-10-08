"""The WUJI Hand 2 readback through the SDK (tactile_sim.sensors.hand_joints) batched over environments: the encoder
angle (`hand_q`) and the motor current (`hand_current`) per joint, as the testbed's RateLimitedSensor produces them
with decimation "zoh" and no band limit:

    every `every` physics steps from the reset (the SDK rate; the first sample on the first step) the present value
    is sampled -> it becomes visible `latency` later (when t_sample + latency <= t + dt / 2, in order) -> + white
    noise (noise_std x noise_scale) -> quantized to `quant` (round half to even) -> latest() holds it.

Each environment has its own sample period, latency, noise and quantization (per-env randomization). Values are
float64, as on the testbed. `noise_fn` (tests) replaces the noise draws: f(seq (N,) long, mask (N,) bool) -> (N, dim).
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class TorchReadback:
    def __init__(self, num_envs: int, dim: int, physics_dt: float, every: int = 4, latency: float = 0.0005,
                 noise_std: float = 0.0, quant: float = 0.0, device: str | torch.device = "cpu", depth: int = 8,
                 generator: torch.Generator | None = None):
        self.n_env, self.dim, self.dt, self.L = int(num_envs), int(dim), float(physics_dt), int(depth)
        self.device = torch.device(device)
        self.generator = generator
        self.noise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        N, f64, dev = self.n_env, torch.float64, self.device
        self.every = torch.full((N,), int(every), dtype=torch.long, device=dev)
        self.latency = torch.full((N,), float(latency), dtype=f64, device=dev)
        self.noise_std = torch.full((N, dim), float(noise_std), dtype=f64, device=dev)
        self.noise_scale = torch.ones(N, dtype=f64, device=dev)
        self.quant = torch.full((N,), float(quant), dtype=f64, device=dev)
        self._noisy = float(noise_std) > 0  # host-side: no synchronization per step
        self.k = torch.zeros(N, dtype=torch.long, device=dev)
        self.ring_t = torch.zeros(N, self.L, dtype=f64, device=dev)
        self.ring_v = torch.zeros(N, self.L, dim, dtype=f64, device=dev)
        self.head = torch.zeros(N, dtype=torch.long, device=dev)
        self.count = torch.zeros(N, dtype=torch.long, device=dev)
        self.value = torch.zeros(N, dim, dtype=f64, device=dev)
        self.t_sample = torch.zeros(N, dtype=f64, device=dev)
        self.t_avail = torch.zeros(N, dtype=f64, device=dev)
        self.seq = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.overflow = torch.zeros(N, dtype=torch.long, device=dev)
        self._ar = torch.arange(N, device=dev)
        self._slots = torch.arange(self.L, device=dev)

    def set_params(self, env_ids=None, every=None, latency=None, noise_std=None, noise_scale=None, quant=None) -> None:
        """Per-environment parameters (scalars, (n,) per env, or for noise_std (n, dim) / (dim,))."""
        ids = slice(None) if env_ids is None else env_ids
        for name, v in (("every", every), ("latency", latency), ("noise_scale", noise_scale), ("quant", quant)):
            if v is not None:
                getattr(self, name)[ids] = torch.as_tensor(v, dtype=getattr(self, name).dtype, device=self.device)
        if noise_std is not None:
            self.noise_std[ids] = torch.as_tensor(noise_std, dtype=torch.float64, device=self.device)
            self._noisy = bool((self.noise_std > 0).any())

    def reset(self, env_ids=None, t0: float = 0.0) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.k[ids] = 0
        self.head[ids] = 0
        self.count[ids] = 0
        self.value[ids] = 0.0
        self.t_sample[ids] = t0
        self.t_avail[ids] = t0
        self.seq[ids] = -1

    def step(self, t: torch.Tensor, raw: torch.Tensor) -> None:
        """One physics step, after MuJoCo stepped: `t` (N,) float64 the time now, `raw` (N, dim) the present value."""
        due = (self.k % self.every) == 0
        self.k += 1
        slot = (self.head + self.count) % self.L
        self.ring_t[self._ar, slot] = torch.where(due, t, self.ring_t[self._ar, slot])
        self.ring_v[self._ar, slot] = torch.where(due[:, None], raw.double(), self.ring_v[self._ar, slot])
        self.overflow += (due & (self.count >= self.L)).long()
        self.count = torch.clamp(self.count + due.long(), max=self.L)
        # samples whose latency has passed, oldest first (sample times are monotone: the ready ones lead the ring)
        order = (self.head[:, None] + self._slots[None]) % self.L
        ready = (self._slots[None] < self.count[:, None]) & \
            (self.ring_t.gather(1, order) + self.latency[:, None] <= (t + 0.5 * self.dt)[:, None])
        m = ready.sum(1)
        new = m > 0
        last = (self.head + m - 1) % self.L
        ts = self.ring_t[self._ar, last]
        v = self.ring_v[self._ar, last]
        seq = self.seq + m
        if self.noise_fn is not None:
            v = v + self.noise_fn(seq, new).double() * self.noise_std * self.noise_scale[:, None]
        elif self._noisy:
            z = torch.randn(self.n_env, self.dim, dtype=torch.float64, device=self.device, generator=self.generator)
            v = v + z * self.noise_std * self.noise_scale[:, None]
        q = self.quant[:, None]
        v = torch.where(q > 0, torch.round(v / torch.where(q > 0, q, 1.0)) * q, v)
        self.value = torch.where(new[:, None], v, self.value)
        self.t_sample = torch.where(new, ts, self.t_sample)
        self.t_avail = torch.where(new, ts + self.latency, self.t_avail)
        self.seq = seq
        self.head = (self.head + m) % self.L
        self.count = self.count - m

    def get_state(self, env_ids=None) -> dict[str, torch.Tensor]:
        ids = slice(None) if env_ids is None else env_ids
        return {k: getattr(self, k)[ids].clone() for k in ("k", "ring_t", "ring_v", "head", "count", "value",
                                                           "t_sample", "t_avail", "seq")}
