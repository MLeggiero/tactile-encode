"""The load generator batched over environments: the torch mirror of tactile_sim.model.loadgen (the reference).

Each env's episode holds up to K events (slots in time order), drawn at reset as sample_schedule draws them: a Poisson
stream at LoadGenCfg.rate from t_start, each event's kind from p_kind and its parameters from the kind's ranges, the
next gap counted from the end of the previous event. The kinds (impulse half-sine along a cone about the strike
axis, periodic force along the feed or torque about the long axis with a drifting amplitude, breakthrough or
appearing step, bind torque spike about the long axis or any tool axis, steady push, torque impulse about any tool
axis) and `value(t)` are LoadEvent's; a torque's signed axis is `tax` (LoadEvent.axis, +x by default). `amp_scale`
scales every amplitude (the curriculum's low-amplitude stage).

Every physics substep (`apply`, before MuJoCo steps) the summed wrench is applied at the tool's working point through
xfrc_applied: the force at the hammer face, the torque about its axis. The tool frame is LoadEvent's: x
along the handle, z the strike axis pointing into the head (a blow's reaction on the face), y = z x x.

`truth(t)` is forecast_truth: the next event or the one under way (a breakthrough step: when its push vanishes), its
regime (obs_spec.REGIMES), frequency and amplitude; the forecast model turns it into the residual's forecast.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from tactile_sim.config import LoadGenCfg
from tactile_sim.control.obs_spec import REGIMES
from tactile_sim.model.loadgen import KINDS, REGIME_OF, kind_probs

IMPULSE, PERIODIC, STEP, BIND, PUSH, TORQUE_IMPULSE = range(6)
assert KINDS == ("impulse", "periodic", "step", "bind", "push", "torque_impulse")
DRIFT_DT = 0.01  # s, LoadEvent.drift's sample period


def _pick(kind: torch.Tensor, vals: list[torch.Tensor]) -> torch.Tensor:
    """Per env, the value of its event's kind (vals[k] for kind k)."""
    out = torch.zeros_like(vals[0])
    for k, v in enumerate(vals):
        out = torch.where(kind == k, v, out)
    return out


class TorchLoadGen:
    def __init__(self, cfg: LoadGenCfg, num_envs: int, device: str | torch.device, slots: int = 16,
                 generator: torch.Generator | None = None):
        self.cfg = cfg
        self.n_env = N = int(num_envs)
        self.K = K = int(slots)
        self.dev = dev = torch.device(device)
        self.generator = generator
        self.D = int(math.ceil(cfg.periodic_s[1] / DRIFT_DT)) + 1
        f64 = dict(dtype=torch.float64, device=dev)
        self.valid = torch.zeros(N, K, dtype=torch.bool, device=dev)
        self.kind = torch.zeros(N, K, dtype=torch.long, device=dev)
        self.t0 = torch.zeros(N, K, **f64)
        self.dur = torch.zeros(N, K, **f64)
        self.amp = torch.zeros(N, K, **f64)
        self.dir = torch.zeros(N, K, 3, **f64)
        self.dir[..., 2] = 1.0
        self.torque = torch.zeros(N, K, dtype=torch.bool, device=dev)
        self.tax = torch.zeros(N, K, 3, **f64)  # a torque's signed unit axis, tool frame
        self.tax[..., 0] = 1.0
        self.freq = torch.zeros(N, K, **f64)
        self.phase = torch.zeros(N, K, **f64)
        self.sign = torch.ones(N, K, **f64)
        self.drift = torch.ones(N, K, self.D, **f64)
        self.n_drift = torch.ones(N, K, dtype=torch.long, device=dev)
        self.amp_scale = torch.ones(N, **f64)
        self.overflow = torch.zeros(N, dtype=torch.long, device=dev)  # events beyond K slots (dropped)
        reg = [REGIMES.index(REGIME_OF[k]) for k in KINDS]
        self.regime_of_kind = torch.as_tensor(reg, dtype=torch.long, device=dev)
        self.p_kind = torch.as_tensor(kind_probs(cfg), **f64)
        if cfg.bind_axes not in ("x", "xyz"):
            raise ValueError(f"bind_axes {cfg.bind_axes!r} (expected 'x' or 'xyz')")

    # ------------------------------------------------------------------ draws
    def _u(self, n: int, lohi, shape=()) -> torch.Tensor:
        lo, hi = float(lohi[0]), float(lohi[1])
        return lo + (hi - lo) * torch.rand(n, *shape, dtype=torch.float64, device=self.dev, generator=self.generator)

    def _cone(self, n: int, half_angle: float) -> torch.Tensor:
        c = self._u(n, (math.cos(half_angle), 1.0))
        phi = self._u(n, (0.0, 2 * math.pi))
        s = torch.sqrt((1 - c * c).clamp(min=0.0))
        return torch.stack([s * torch.cos(phi), s * torch.sin(phi), c], dim=-1)

    def _axis(self, n: int) -> torch.Tensor:
        """(n, 3) a tool axis per row, either sign (loadgen._axis)."""
        i = torch.randint(0, 3, (n,), device=self.dev, generator=self.generator)
        sign = torch.where(torch.rand(n, dtype=torch.float64, device=self.dev, generator=self.generator) < 0.5,
                           1.0, -1.0).double()
        return torch.nn.functional.one_hot(i, 3).double() * sign[:, None]

    def sample(self, env_ids: torch.Tensor, t_end: float | torch.Tensor, t_start: float = 0.2,
               amp_scale: float | torch.Tensor = 1.0) -> None:
        """New schedules for env_ids from t_start to t_end (sample_schedule, batched)."""
        c, dev, K = self.cfg, self.dev, self.K
        n = len(env_ids)
        f64 = dict(dtype=torch.float64, device=dev)
        t = torch.full((n,), float(t_start), **f64)
        t_end = torch.as_tensor(t_end, **f64).expand(n)
        cols = {k: [] for k in ("valid", "kind", "t0", "dur", "amp", "dir", "torque", "tax", "freq", "phase", "sign",
                                "drift", "n_drift")}
        open_ = torch.ones(n, dtype=torch.bool, device=dev)
        for _ in range(K):
            t = t + (-torch.log(torch.rand(n, **f64, generator=self.generator).clamp(min=1e-300))) / c.rate
            ok = open_ & (t < t_end)
            open_ = ok
            kind = torch.multinomial(self.p_kind.expand(n, -1).float(), 1, generator=self.generator).squeeze(1)
            # every kind's parameters, then pick per env
            imp_dur = self._u(n, c.impulse_ms) * 1e-3
            imp_amp = self._u(n, c.impulse_n)
            imp_dir = self._cone(n, math.radians(c.cone_deg))
            per_dur = self._u(n, c.periodic_s)
            per_torque = torch.rand(n, **f64, generator=self.generator) < 0.5
            per_amp = torch.where(per_torque, self._u(n, c.periodic_nm), self._u(n, c.periodic_n))
            per_f = torch.exp(self._u(n, (math.log(c.periodic_hz[0]), math.log(c.periodic_hz[1]))))
            per_ph = self._u(n, (0.0, 2 * math.pi))
            walk = torch.cumsum(torch.randn(n, self.D, **f64, generator=self.generator) * c.drift * math.sqrt(DRIFT_DT),
                                dim=1)
            per_drift = torch.clamp(1.0 + walk, 0.2, 2.0)
            per_nd = torch.ceil(per_dur / DRIFT_DT).long() + 1
            hold = self._u(n, c.step_hold_s)
            st_sign = torch.where(torch.rand(n, **f64, generator=self.generator) < 0.7, 1.0, -1.0).double()
            st_amp = self._u(n, c.step_n)
            bind_dur = self._u(n, c.bind_ms) * 1e-3
            bind_amp = self._u(n, c.bind_nm)
            push_dur = self._u(n, c.push_s)
            push_amp = self._u(n, c.push_n)
            ti_dur = self._u(n, c.torque_impulse_ms) * 1e-3
            ti_amp = self._u(n, c.torque_impulse_nm)
            axis = self._axis(n)
            dur = _pick(kind, [imp_dur, per_dur, hold, bind_dur, push_dur, ti_dur])
            amp = _pick(kind, [imp_amp, per_amp, st_amp, bind_amp, push_amp, ti_amp])
            z = torch.zeros(n, 3, **f64)
            z[:, 2] = 1.0
            direction = torch.where((kind == IMPULSE)[:, None], imp_dir, z)
            torque = ((kind == PERIODIC) & per_torque) | (kind == BIND) | (kind == TORQUE_IMPULSE)
            ex = torch.zeros(n, 3, **f64)
            ex[:, 0] = 1.0
            any_axis = (kind == TORQUE_IMPULSE) | ((kind == BIND) & (c.bind_axes == "xyz"))
            tax = torch.where(any_axis[:, None], axis, ex)
            freq = torch.where(kind == PERIODIC, per_f, torch.zeros_like(per_f))
            phase = torch.where(kind == PERIODIC, per_ph, torch.zeros_like(per_ph))
            sign = torch.where(kind == STEP, st_sign, torch.ones_like(st_sign))
            t0 = torch.where((kind == STEP) & (st_sign > 0), t + hold, t)
            drift = torch.where((kind == PERIODIC)[:, None], per_drift, torch.ones_like(per_drift))
            n_drift = torch.where(kind == PERIODIC, per_nd, torch.ones_like(per_nd))
            end = torch.where((kind == STEP) & (sign > 0), t0, t0 + dur)
            for k, v in (("valid", ok), ("kind", kind), ("t0", t0), ("dur", dur), ("amp", amp), ("dir", direction),
                         ("torque", torque), ("tax", tax), ("freq", freq), ("phase", phase), ("sign", sign),
                         ("drift", drift), ("n_drift", n_drift)):
                cols[k].append(v)
            t = torch.where(ok, torch.maximum(t, end), t)
        # events past the K slots are dropped (counted)
        t_next = t + (-torch.log(torch.rand(n, **f64, generator=self.generator).clamp(min=1e-300))) / c.rate
        self.overflow[env_ids] += (open_ & (t_next < t_end)).long()
        for k, v in cols.items():
            getattr(self, k)[env_ids] = torch.stack(v, dim=1)
        self.amp_scale[env_ids] = torch.as_tensor(amp_scale, **f64).expand(n)

    def set_events(self, env: int, events) -> None:
        """Load a CPU schedule (a list of tactile_sim.model.loadgen.LoadEvent) into env `env` (parity tests)."""
        if len(events) > self.K:
            raise ValueError(f"{len(events)} events, {self.K} slots")
        self.valid[env] = False
        for k, ev in enumerate(events):
            self.valid[env, k] = True
            self.kind[env, k] = KINDS.index(ev.kind)
            self.t0[env, k] = ev.t0
            self.dur[env, k] = ev.duration
            self.amp[env, k] = ev.amp
            self.dir[env, k] = torch.as_tensor(np.asarray(ev.direction, float), dtype=torch.float64)
            self.torque[env, k] = bool(ev.torque)
            ax = np.array([1.0, 0.0, 0.0]) if ev.axis is None else np.asarray(ev.axis, float)
            self.tax[env, k] = torch.as_tensor(ax, dtype=torch.float64)
            self.freq[env, k] = ev.freq
            self.phase[env, k] = ev.phase
            self.sign[env, k] = ev.sign
            if ev.drift is not None:
                d = torch.as_tensor(np.asarray(ev.drift, float), dtype=torch.float64)
                self.drift[env, k, :len(d)] = d[:self.D]
                self.n_drift[env, k] = min(len(d), self.D)
        self.amp_scale[env] = 1.0

    # ------------------------------------------------------------------ the loads
    def end(self) -> torch.Tensor:
        """(N, K) when each event is over (a breakthrough step: when its push vanishes)."""
        return torch.where((self.kind == STEP) & (self.sign > 0), self.t0, self.t0 + self.dur)

    def values(self, t: torch.Tensor) -> torch.Tensor:
        """(N, K) every slot's signed magnitude at time t (N,) (LoadEvent.value), amplitude-scaled."""
        tau = t.double()[:, None] - self.t0
        k = self.kind
        live = (tau >= 0) & (tau < self.dur)
        s = torch.sin(math.pi * tau / self.dur.clamp(min=1e-12))
        idx = torch.minimum(torch.floor(tau / DRIFT_DT).long().clamp(min=0), self.n_drift - 1)
        m = self.drift.gather(2, idx[..., None]).squeeze(-1)
        per = self.amp * m * torch.sin(2 * math.pi * self.freq * tau + self.phase)
        v = torch.where((k == IMPULSE) | (k == TORQUE_IMPULSE), self.amp * s,
                        torch.where(k == BIND, self.amp * s * s,
                                    torch.where(k == PERIODIC, per, self.amp)))
        v = torch.where(live, v, torch.zeros_like(v))
        step_v = torch.where(self.sign > 0, torch.where((tau >= -self.dur) & (tau < 0), self.amp, 0.0),
                             torch.where((tau >= 0) & (tau < self.dur), self.amp, 0.0))
        v = torch.where(k == STEP, step_v, v)
        v = torch.where(self.valid, v, torch.zeros_like(v))
        return v * self.amp_scale[:, None]

    def wrench(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Force (N, 3) and torque (N, 3) on the tool at time t, tool frame (loadgen.wrench)."""
        v = self.values(t)
        f = (torch.where(self.torque, torch.zeros_like(v), v)[..., None] * self.dir).sum(1)
        tq = (torch.where(self.torque, v, torch.zeros_like(v))[..., None] * self.tax).sum(1)
        return f, tq

    def truth(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """forecast_truth: the first event not over at t."""
        ahead = self.valid & (self.end() > t.double()[:, None])
        valid = ahead.any(1)
        k = torch.argmax(ahead.long(), dim=1)
        ar = torch.arange(self.n_env, device=self.dev)
        kind = self.kind[ar, k]
        z = torch.zeros_like(t, dtype=torch.float64)
        return {"valid": valid, "t_event": torch.where(valid, self.t0[ar, k], z),
                "regime": torch.where(valid, self.regime_of_kind[kind], torch.zeros_like(kind)),
                "f": torch.where(valid, self.freq[ar, k], z),
                "amp": torch.where(valid, self.amp[ar, k] * self.amp_scale, z), "slot": k}

    def active(self, t: torch.Tensor, settle: float = 0.05) -> torch.Tensor:
        """(N, K) events under way at t or within `settle` after their end (breakthrough steps: from their push's
        start to settle after it vanished)."""
        tau = t.double()[:, None]
        start = torch.where((self.kind == STEP) & (self.sign > 0), self.t0 - self.dur, self.t0)
        return self.valid & (tau >= start) & (tau < self.end() + settle)


class LoadApplicator:
    """Applies a TorchLoadGen's wrench to the tool body through xfrc_applied: the force at the working point (a site
    on the tool) and the torque, both turned from the tool frame (R_tb: the tool axes in the body frame) to the world;
    the force's moment about the body's centre of mass is added (xfrc_applied acts at xipos)."""

    def __init__(self, sim, body: int, site: int, R_tb: np.ndarray, device):
        self.sim = sim
        self.body, self.site = int(body), int(site)
        self.R_tb = torch.as_tensor(np.asarray(R_tb, float), dtype=torch.float32, device=device)

    def apply(self, f_tool: torch.Tensor, tq_tool: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.sim.data
        R_b = d.xmat[:][:, self.body].reshape(-1, 3, 3)
        R = R_b @ self.R_tb
        F = torch.einsum("nij,nj->ni", R, f_tool.float())
        Tq = torch.einsum("nij,nj->ni", R, tq_tool.float())
        p = d.site_xpos[:][:, self.site]
        com = d.xipos[:][:, self.body]
        Tq = Tq + torch.cross(p - com, F, dim=-1)
        x = d.xfrc_applied
        x[:, self.body, :3] = F
        x[:, self.body, 3:] = Tq
        return F, Tq

    def clear(self) -> None:
        self.sim.data.xfrc_applied[:, self.body] = 0.0


def tool_frame(m, body: int, face_site: int, head_site: int) -> np.ndarray:
    """The tool frame in the body frame (columns x, y, z): x along the handle (the body's x axis), z from the face into
    the head (the reaction of a blow on the face), y = z x x."""
    face = np.asarray(m.site_pos[face_site], float)
    head = np.asarray(m.site_pos[head_site], float)
    z = head - face
    x = np.array([1.0, 0.0, 0.0])
    z = z - (z @ x) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
