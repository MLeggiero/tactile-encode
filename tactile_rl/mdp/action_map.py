"""The residual's action map and the grip synergy in torch, batched over environments: mirrors of
tactile_sim.control.obs_spec (offsets, slew, compose), ResidualHandL2._trigger_guard and WrapSynergy.mit_command.

Shapes: actions (N, 40) squashed to [-1, 1]; offsets (N, 2, J) rows dq, dtau; MIT commands (N, 5, J) rows q, qd, kp,
kd, tau (the gains and qd are the base's: configuration, never the residual's). With a = 0 the composed command
equals the base command bit for bit.
"""

from __future__ import annotations

import numpy as np
import torch

from tactile_sim.config import ResidualCfg
from tactile_sim.control.hand import WrapSynergy
from tactile_sim.control.obs_spec import N_JOINTS, ActionSpec, JointParams


class TorchActionMap:
    def __init__(self, spec: ActionSpec, jp: JointParams, device: str | torch.device = "cpu",
                 close_dir: np.ndarray | None = None, trigger_idx: np.ndarray | None = None,
                 residual: ResidualCfg | None = None):
        f = dict(dtype=torch.float32, device=device)
        self.spec = spec
        self.dq_max = torch.as_tensor(np.asarray(spec.dq_max, float)[jp.group], **f)
        self.tau_rated = torch.as_tensor(jp.tau_rated, **f)
        self.rate = torch.stack([torch.full((N_JOINTS,), spec.dq_rate, **f),
                                 spec.tau_rate * self.tau_rated])  # (2, J) per second
        # drill-trigger interlock (ResidualHandL2._trigger_guard); no trigger joints = no interlock
        r = residual or ResidualCfg()
        self.trig = torch.as_tensor(np.asarray(trigger_idx if trigger_idx is not None else [], dtype=int),
                                    dtype=torch.long, device=device)
        cd = np.zeros(N_JOINTS) if close_dir is None else np.asarray(close_dir, float)
        self.trig_dir = torch.as_tensor(cd, **f)[self.trig]
        self.trig_dq = float(r.trigger_dq_max)
        self.trig_tau = (float(r.trigger_tau[0]), float(r.trigger_tau[1]))

    def offsets(self, a_sq: torch.Tensor) -> torch.Tensor:
        """(N, 2, J) residual offsets [dq, dtau] for squashed actions (N, 40) (obs_spec.offsets)."""
        a = a_sq.reshape(a_sq.shape[0], 2, N_JOINTS)
        dq = self.dq_max * a[:, 0]
        dtau = self.spec.tau_frac * self.tau_rated * a[:, 1]
        return torch.stack([dq, dtau], dim=1)

    def slew(self, prev: torch.Tensor, target: torch.Tensor, dt: float) -> torch.Tensor:
        """Per-second slew limit on the offsets over a policy period dt (obs_spec.slew)."""
        lim = self.rate * dt
        return prev + torch.minimum(torch.maximum(target - prev, -lim), lim)

    def compose(self, base: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        """Final MIT command (N, 5, J) from the base (N, 5, J) and the offsets (N, 2, J) (obs_spec.compose)."""
        q, qd, kp, kd, tau_b = base.unbind(1)
        return torch.stack([q + off[:, 0], qd, kp, kd, tau_b + off[:, 1]], dim=1)

    def trigger_guard(self, off: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
        """ResidualHandL2._trigger_guard: residual on the trigger joints capped, and only opening while the base
        disables the trigger (`trigger` (N,) bool)."""
        if self.trig.numel() == 0:
            return off
        out = off.clone()
        j, d = self.trig, self.trig_dir
        dq = off[:, 0, j].clamp(-self.trig_dq, self.trig_dq)
        tau_close = (d * off[:, 1, j]).clamp(self.trig_tau[0], self.trig_tau[1])
        off_trig = ~trigger[:, None]
        dq = torch.where(off_trig & (d * dq > 0), torch.zeros_like(dq), dq)
        tau_close = torch.where(off_trig, torch.clamp(tau_close, max=0.0), tau_close)
        out[:, 0, j] = dq
        out[:, 1, j] = d * tau_close
        return out


class TorchSynergy:
    """WrapSynergy.mit_command for every environment: (s (N,), q (N, J)[, s_trigger (N,)]) -> (N, 5, J)."""

    def __init__(self, syn: WrapSynergy, device: str | torch.device = "cpu"):
        f = dict(dtype=torch.float32, device=device)
        n = len(syn.joints)
        w = np.ones(n) if syn.weight is None else np.asarray(syn.weight, float)
        shaping = np.asarray(syn.close_dir) == 0
        self.tau_unit = torch.as_tensor(np.asarray(syn.close_dir, float) * w * np.asarray(syn.tau_max, float), **f)
        self.shaping = torch.as_tensor(shaping, device=device)
        self.q_hold = torch.as_tensor(np.asarray(syn.q_hold, float), **f)
        kp = np.where(shaping, syn.kp, 0.0)
        coupled = list(syn.coupled or ())
        for i, _ in coupled:
            kp[i] = syn.coupling_kp
        self.kp = torch.as_tensor(kp, **f)
        self.kd = torch.full((n,), float(syn.kd), **f)
        self.coupled_i = torch.as_tensor([i for i, _ in coupled], dtype=torch.long, device=device)
        self.coupled_j = torch.as_tensor([j for _, j in coupled], dtype=torch.long, device=device)
        self.ratio = float(syn.coupling_ratio)
        # a trigger finger (WrapSynergy.trigger): its joints close at their own level (WrapSynergy.levels)
        self.trigger = torch.as_tensor(list(getattr(syn, "trigger", ()) or ()), dtype=torch.long, device=device)

    def mit_command(self, s: torch.Tensor, q: torch.Tensor, s_trigger: torch.Tensor | None = None) -> torch.Tensor:
        lv = torch.clamp(s, 0.0, 1.0)[:, None].expand(-1, self.tau_unit.shape[0])
        if s_trigger is not None and self.trigger.numel():
            lv = lv.clone()
            lv[:, self.trigger] = torch.clamp(s_trigger, 0.0, 1.0).to(lv.dtype)[:, None]
        tau = lv * self.tau_unit
        q_t = torch.where(self.shaping, self.q_hold, q)
        if self.coupled_i.numel():
            q_t = q_t.clone()
            q_t[:, self.coupled_i] = self.ratio * q[:, self.coupled_j]
        n_env = q.shape[0]
        return torch.stack([q_t, torch.zeros_like(q), self.kp.expand(n_env, -1), self.kd.expand(n_env, -1), tau], 1)
