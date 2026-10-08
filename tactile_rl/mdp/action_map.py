"""The residual's action map and the grip synergy in torch, batched over environments: mirrors of
tactile_sim.control.obs_spec (offsets, slew, compose), ResidualHandL2._trigger_guard and WrapSynergy.mit_command.

Shapes: actions (N, 60) squashed to [-1, 1]; offsets (N, 3, J) rows dq, dKp, dtau; MIT commands (N, 5, J) rows q,
qd, kp, kd, tau. With a = 0 the composed command equals the base command bit for bit.
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
        self.kp_nom = torch.as_tensor(jp.kp_nom, **f)
        self.tau_rated = torch.as_tensor(jp.tau_rated, **f)
        self.kp_max = spec.kp_max_ratio * self.kp_nom
        self.damp = 2.0 * spec.zeta * torch.sqrt(torch.as_tensor(jp.i_eff, **f))
        self.rate = torch.stack([torch.full((N_JOINTS,), spec.dq_rate, **f), spec.kp_rate * self.kp_nom,
                                 spec.tau_rate * self.tau_rated])  # (3, J) per second
        # drill-trigger interlock (ResidualHandL2._trigger_guard); no trigger joints = no interlock
        r = residual or ResidualCfg()
        self.trig = torch.as_tensor(np.asarray(trigger_idx if trigger_idx is not None else [], dtype=int),
                                    dtype=torch.long, device=device)
        cd = np.zeros(N_JOINTS) if close_dir is None else np.asarray(close_dir, float)
        self.trig_dir = torch.as_tensor(cd, **f)[self.trig]
        self.trig_dq = float(r.trigger_dq_max)
        self.trig_tau = (float(r.trigger_tau[0]), float(r.trigger_tau[1]))

    def offsets(self, a_sq: torch.Tensor) -> torch.Tensor:
        """(N, 3, J) residual offsets [dq, dKp, dtau] for squashed actions (N, 60) (obs_spec.offsets)."""
        a = a_sq.reshape(a_sq.shape[0], 3, N_JOINTS)
        dq = self.dq_max * a[:, 0]
        dkp = self.kp_nom * (torch.pow(self.spec.kp_gain, a[:, 1]) - 1.0)
        dtau = self.spec.tau_frac * self.tau_rated * a[:, 2]
        return torch.stack([dq, dkp, dtau], dim=1)

    def slew(self, prev: torch.Tensor, target: torch.Tensor, dt: float) -> torch.Tensor:
        """Per-second slew limit on the offsets over a policy period dt (obs_spec.slew)."""
        lim = self.rate * dt
        return prev + torch.minimum(torch.maximum(target - prev, -lim), lim)

    def compose(self, base: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        """Final MIT command (N, 5, J) from the base (N, 5, J) and the offsets (N, 3, J) (obs_spec.compose)."""
        q, qd, kp_b, kd_b, tau_b = base.unbind(1)
        kp = torch.minimum(torch.clamp(kp_b + off[:, 1], min=0.0), self.kp_max)
        kd = torch.clamp(kd_b + self.damp * (torch.sqrt(kp) - torch.sqrt(torch.clamp(kp_b, min=0.0))), min=0.0)
        return torch.stack([q + off[:, 0], qd, kp, kd, tau_b + off[:, 2]], dim=1)

    def trigger_guard(self, off: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
        """ResidualHandL2._trigger_guard: residual on the trigger joints capped, and only opening while the base
        disables the trigger (`trigger` (N,) bool)."""
        if self.trig.numel() == 0:
            return off
        out = off.clone()
        j, d = self.trig, self.trig_dir
        dq = off[:, 0, j].clamp(-self.trig_dq, self.trig_dq)
        tau_close = (d * off[:, 2, j]).clamp(self.trig_tau[0], self.trig_tau[1])
        off_trig = ~trigger[:, None]
        dq = torch.where(off_trig & (d * dq > 0), torch.zeros_like(dq), dq)
        tau_close = torch.where(off_trig, torch.clamp(tau_close, max=0.0), tau_close)
        out[:, 0, j] = dq
        out[:, 2, j] = d * tau_close
        return out


class TorchSynergy:
    """WrapSynergy.mit_command for every environment: (s (N,), q (N, J)) -> (N, 5, J)."""

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

    def mit_command(self, s: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        s = torch.clamp(s, 0.0, 1.0)
        tau = s[:, None] * self.tau_unit
        q_t = torch.where(self.shaping, self.q_hold, q)
        if self.coupled_i.numel():
            q_t = q_t.clone()
            q_t[:, self.coupled_i] = self.ratio * q[:, self.coupled_j]
        n_env = q.shape[0]
        return torch.stack([q_t, torch.zeros_like(q), self.kp.expand(n_env, -1), self.kd.expand(n_env, -1), tau], 1)
