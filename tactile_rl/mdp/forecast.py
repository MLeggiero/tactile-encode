"""The load-regime forecast batched over environments: the torch mirror of tactile_sim.behaviors.forecast.ForecastModel
(the reference). Per environment and episode: a mask (no forecast at all), a bias on the predicted event time and a
regime confusion; updates on the update_rate grid (the first call at or after each grid point issues one), each lost
with probability `dropout`, with per-update jitter on the event time and log-normal errors on the frequency and the
amplitude; an update becomes visible `latency` after it was issued; "no event ahead" is issued explicitly. Output:
obs_spec.forecast_vector (13-D), zero when masked or with no event ahead.

Every ForecastCfg field is a per-env tensor (set_params), so the forecast's error model can be randomized. `draw_fn`
(tests) replaces the random draws: f(name, mask (N,) bool) -> (N,) with name in "mask", "bias", "confusion", "dropout",
"freq", "amp", "jitter" (uniforms for the first three and "dropout", standard normals for the rest).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from tactile_sim.config import ForecastCfg
from tactile_sim.control.obs_spec import FORECAST_DIM, REGIMES

FIELDS = ("mask_prob", "dropout", "bias_sigma", "jitter_sigma", "class_confusion", "freq_err", "amp_err",
          "update_rate", "latency")


class TorchForecastModel:
    def __init__(self, num_envs: int, cfg: ForecastCfg | None = None, device: str | torch.device = "cpu",
                 depth: int = 8, generator: torch.Generator | None = None):
        self.n_env, self.D, self.dev = int(num_envs), int(depth), torch.device(device)
        self.generator = generator
        self.draw_fn: Callable[[str, torch.Tensor], torch.Tensor] | None = None
        N, f64 = self.n_env, dict(dtype=torch.float64, device=self.dev)
        cfg = cfg or ForecastCfg()
        for k in FIELDS:
            setattr(self, k, torch.full((N,), float(getattr(cfg, k)), **f64))
        self.R = len(REGIMES)
        self.masked = torch.zeros(N, dtype=torch.bool, device=self.dev)
        self.bias = torch.zeros(N, **f64)
        self.confusion = torch.zeros(N, **f64)
        self.next_update = torch.full((N,), -math.inf, **f64)
        # pending updates (ring): issue time, regime probabilities + frequency + amplitude, event time (NaN: none)
        self.p_t = torch.zeros(N, self.D, **f64)
        self.p_v = torch.zeros(N, self.D, self.R + 2, **f64)
        self.p_e = torch.zeros(N, self.D, **f64)
        self.head = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.count = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.vis = torch.zeros(N, dtype=torch.bool, device=self.dev)
        self.v_t = torch.zeros(N, **f64)
        self.v_v = torch.zeros(N, self.R + 2, **f64)
        self.v_e = torch.zeros(N, **f64)
        self.overflow = torch.zeros(N, dtype=torch.long, device=self.dev)
        self._ar = torch.arange(N, device=self.dev)

    def set_params(self, env_ids=None, **values) -> None:
        ids = slice(None) if env_ids is None else env_ids
        for k, v in values.items():
            if k not in FIELDS:
                raise KeyError(k)
            getattr(self, k)[ids] = torch.as_tensor(v, dtype=torch.float64, device=self.dev)

    def _draw(self, name: str, mask: torch.Tensor, normal: bool) -> torch.Tensor:
        if self.draw_fn is not None:
            return self.draw_fn(name, mask).double()
        if normal:
            return torch.randn(self.n_env, dtype=torch.float64, device=self.dev, generator=self.generator)
        return torch.rand(self.n_env, dtype=torch.float64, device=self.dev, generator=self.generator)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """A new episode for env_ids: the mask, the bias and the confusion are drawn (ForecastModel.reset)."""
        ids = torch.arange(self.n_env, device=self.dev) if env_ids is None else env_ids
        sel = torch.zeros(self.n_env, dtype=torch.bool, device=self.dev)
        sel[ids] = True
        masked = self._draw("mask", sel, False) < self.mask_prob
        bias = self._draw("bias", sel, True) * self.bias_sigma
        conf = self._draw("confusion", sel, False) * self.class_confusion
        self.masked = torch.where(sel, masked, self.masked)
        self.bias = torch.where(sel, bias, self.bias)
        self.confusion = torch.where(sel, conf, self.confusion)
        self.next_update[ids] = -math.inf
        self.head[ids] = 0
        self.count[ids] = 0
        self.vis[ids] = False

    def __call__(self, t: torch.Tensor, valid: torch.Tensor, t_event: torch.Tensor, regime: torch.Tensor,
                 f_dom: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
        """The (N, 13) forecast at time t (N,) given the truth as of now: valid (N,) bool, event time (N,) (absolute,
        the env's clock), regime index (N,) long, dominant frequency (Hz) and amplitude (N)."""
        t = t.double()
        due = ~self.masked & (t >= self.next_update - 1e-12)
        # issue: "no event ahead" when the truth has none; otherwise a noisy forecast unless the update is lost
        lost = self._draw("dropout", due & valid, False) < self.dropout
        push = due & (~valid | ~lost)
        n = self.R
        p = (self.confusion / (n - 1))[:, None].expand(-1, n).clone()
        k = regime.clamp(0, n - 1)
        p[self._ar, k] = 1.0 - self.confusion
        f = f_dom.double() * torch.exp(self._draw("freq", due & valid & ~lost, True) * self.freq_err)
        a = amp.double() * torch.exp(self._draw("amp", due & valid & ~lost, True) * self.amp_err)
        te = t_event.double() + self.bias + self._draw("jitter", due & valid & ~lost, True) * self.jitter_sigma
        v = torch.cat([p, f[:, None], a[:, None]], dim=1)
        v = torch.where(valid[:, None], v, torch.zeros_like(v))
        te = torch.where(valid, te, torch.full_like(te, float("nan")))
        slot = (self.head + self.count) % self.D
        self.overflow += (push & (self.count >= self.D)).long()
        self.p_t[self._ar, slot] = torch.where(push, t, self.p_t[self._ar, slot])
        self.p_v[self._ar, slot] = torch.where(push[:, None], v, self.p_v[self._ar, slot])
        self.p_e[self._ar, slot] = torch.where(push, te, self.p_e[self._ar, slot])
        self.count = torch.clamp(self.count + push.long(), max=self.D)
        nxt = (torch.floor(t * self.update_rate + 1e-9) + 1.0) / self.update_rate
        self.next_update = torch.where(due, nxt, self.next_update)
        # pop the updates whose latency has passed, in order; the newest is what the residual sees
        for _ in range(self.D):
            h = self.head
            ready = (self.count > 0) & (self.p_t[self._ar, h] + self.latency <= t + 1e-12)
            self.vis = self.vis | ready
            self.v_t = torch.where(ready, self.p_t[self._ar, h], self.v_t)
            self.v_v = torch.where(ready[:, None], self.p_v[self._ar, h], self.v_v)
            self.v_e = torch.where(ready, self.p_e[self._ar, h], self.v_e)
            self.head = torch.where(ready, (h + 1) % self.D, h)
            self.count = self.count - ready.long()
        show = ~self.masked & self.vis & ~torch.isnan(self.v_e)
        s_t = torch.hypot(self.bias_sigma, self.jitter_sigma)
        fv, av = self.v_v[:, n], self.v_v[:, n + 1]
        out = torch.cat([torch.ones_like(t)[:, None], torch.clamp(self.v_e - t, -1.0, 1.0)[:, None], s_t[:, None],
                         self.v_v[:, :n], (fv / 100.0)[:, None], (self.freq_err * fv / 100.0)[:, None],
                         (av / 100.0)[:, None], (self.amp_err * av / 100.0)[:, None],
                         torch.clamp(t - self.v_t, max=1.0)[:, None]], dim=1)
        assert out.shape[1] == FORECAST_DIM
        return torch.where(show[:, None], out, torch.zeros_like(out))
