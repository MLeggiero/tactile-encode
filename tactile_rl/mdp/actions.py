"""Action terms of the tactile residual env.

- `ResidualHandAction` (60-D), once per policy step, as the testbed's ResidualHandL2.tick: a non-finite action vector
  counts as zero; the actions are squashed by tanh (the exported policy carries the tanh) and become the residual's
  offsets (obs_spec.offsets); on the drill task the trigger interlock caps the index finger's offsets (only opening
  while the base disables the trigger; no interlock elsewhere: residual_l2.trigger_indices); if any input is stale
  (the TaxelScan boards' newest frames older than stale_taxel, the joint readback older than stale_joint) the offsets
  decay toward zero with decay_tau instead; the offsets' slew is limited per second over the policy period
  (obs_spec.slew); the result is composed with the base hand command (obs_spec.compose) into the MIT command the host
  sends the WUJI drives. The base: with a base-trajectory command (the hammer tasks), the grip synergy at the
  library's scalar from the joint angles the host last read (the readback); without one (the hold tasks of milestone
  1), the synergy at the constant hold scalar World.reset ends with, from the joints' true angles, and no sensors.
- `SubstepHooks` (0-D), every physics substep before MuJoCo steps (mjlab runs every action term's apply_actions
  there): the arm's servo targets (the base's 100 Hz stream, or the hover pose + gravity droop), the plant's pre-step
  hook (the tactile runtime's) and one step of the WUJI drives (TorchWujiDrive.step), whose torque the hand actuator
  then writes to ctrl.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from tactile_rl.actuators.wuji_mit import TorchWujiDrive, WujiMitActuator
from tactile_rl.mdp.action_map import TorchActionMap, TorchSynergy
from tactile_sim.control.obs_spec import N_JOINTS, ActionSpec, JointParams
from tactile_sim.control.residual_l2 import trigger_indices
from tactile_sim.sim.training_parts import TrainingParts, training_parts

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def hand_drive(entity) -> TorchWujiDrive:
    acts = [a for a in entity.actuators if isinstance(a, WujiMitActuator)]
    if len(acts) != 1:
        raise RuntimeError(f"expected one WujiMitActuator on the entity, found {len(acts)}")
    return acts[0].drive


def joint_ids(entity, names: tuple[str, ...], device) -> torch.Tensor:
    """Entity-local indices of `names`, in that order."""
    idx = {n: i for i, n in enumerate(entity.joint_names)}
    return torch.tensor([idx[n] for n in names], dtype=torch.long, device=device)


def base_command_term(env: ManagerBasedRlEnv):
    """The env's base command term (the hammer's library swings, the load generator's hold), or None (the hold tasks
    of milestone 1). A base term has `tactile_base = True` and provides the runtime (`rt`), the base hand command and
    its scalar (`base`, `s_now`), the lookahead, the forecast and its truth, `trigger`, `ref_rel`, `arm_target(t)`
    and `param_vector()`."""
    cm = getattr(env, "command_manager", None)
    for name in getattr(cm, "active_terms", []) or []:
        term = cm.get_term(name)
        if getattr(term, "tactile_base", False):
            return term
    return None


@dataclass(kw_only=True)
class ResidualHandActionCfg(ActionTermCfg):
    scene_key: str
    task: str = "hammer"

    def build(self, env: ManagerBasedRlEnv) -> ResidualHandAction:
        return ResidualHandAction(self, env)


class ResidualHandAction(ActionTerm):
    cfg: ResidualHandActionCfg

    def __init__(self, cfg: ResidualHandActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        parts: TrainingParts = training_parts(cfg.scene_key)
        self.parts = parts
        dev = self.device
        self.drive = hand_drive(self._entity)
        if list(self.drive.joint_names) != list(parts.hand_joints):
            raise RuntimeError("the hand actuator's joint order differs from the testbed's")
        self.hand_ids = joint_ids(self._entity, parts.hand_joints, dev)
        sim_cfg = parts.cfg
        self.spec = ActionSpec()
        jp = JointParams.for_joints(list(parts.hand_joints), parts.hand_tau_rated, sim_cfg.wuji.kp_nom, parts.i_eff)
        res = sim_cfg.residual
        self.res = res
        self.task = cfg.task
        trig = trigger_indices(type(res)(**{**res.__dict__, "task": cfg.task}), list(parts.hand_joints))
        self.map = TorchActionMap(self.spec, jp, dev, close_dir=parts.synergy.close_dir, trigger_idx=trig,
                                  residual=res)
        self.synergy = TorchSynergy(parts.synergy, dev)
        self.policy_dt = env.step_dt
        self.decay = math.exp(-self.policy_dt / res.decay_tau)
        n = self.num_envs
        self.s = torch.full((n,), parts.s_hold, dtype=torch.float32, device=dev)
        self.trigger = torch.zeros(n, dtype=torch.bool, device=dev)
        self._raw = torch.zeros(n, self.action_dim, device=dev)
        self.a_sq = torch.zeros(n, self.action_dim, device=dev)  # the squashed action the policy took (last_action)
        self.off = torch.zeros(n, 3, N_JOINTS, device=dev)
        self.base = torch.zeros(n, 5, N_JOINTS, device=dev)
        self.command = torch.zeros(n, 5, N_JOINTS, device=dev)
        self.n_nonfinite = torch.zeros(n, dtype=torch.long, device=dev)
        self.n_stale = torch.zeros(n, dtype=torch.long, device=dev)
        self.base_cmd = base_command_term(env)
        self.rt = getattr(env, "_tactile_rt", None)

    @property
    def action_dim(self) -> int:
        return self.spec.dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def base_command(self) -> torch.Tensor:
        """The base MIT command now (see the module doc)."""
        if self.base_cmd is not None:
            # a drill base also sets the trigger finger's own level (WrapSynergy.levels)
            return self.synergy.mit_command(self.base_cmd.s_now, self.rt.rb_q.value.float(),
                                            getattr(self.base_cmd, "s_trig_now", None))
        q = self._entity.data.joint_pos[:, self.hand_ids]
        return self.synergy.mit_command(self.s, q)

    def process_actions(self, actions: torch.Tensor) -> None:
        if self.rt is not None:
            self.rt.begin_step()  # the reward integrands cover this policy step
        self._raw[:] = actions
        bad = ~torch.isfinite(actions).all(dim=1)
        self.n_nonfinite += bad.long()
        a = torch.where(bad[:, None], torch.zeros_like(actions), actions)
        self.a_sq = torch.clamp(torch.tanh(a), -1.0, 1.0)
        if self.base_cmd is not None:
            self.trigger = self.base_cmd.trigger
        target = self.map.trigger_guard(self.map.offsets(self.a_sq), self.trigger)
        if self.rt is not None:
            stale = self.rt.stale()
            self.n_stale += stale.long()
            target = torch.where(stale[:, None, None], self.off * self.decay, target)
        self.off = self.map.trigger_guard(self.map.slew(self.off, target, self.policy_dt), self.trigger)
        self.base = self.base_command()
        self.command = self.map.compose(self.base, self.off)
        self.drive.set_host(self.command)

    def apply_actions(self) -> None:
        pass

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self.a_sq[ids] = 0.0
        self.off[ids] = 0.0
        self.n_nonfinite[ids] = 0
        self.n_stale[ids] = 0


@dataclass(kw_only=True)
class SubstepHooksCfg(ActionTermCfg):
    scene_key: str

    def build(self, env: ManagerBasedRlEnv) -> SubstepHooks:
        return SubstepHooks(self, env)


class SubstepHooks(ActionTerm):
    """Per-physics-substep hooks before MuJoCo steps (see module doc)."""

    cfg: SubstepHooksCfg

    def __init__(self, cfg: SubstepHooksCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        parts = training_parts(cfg.scene_key)
        dev = self.device
        self.drive = hand_drive(self._entity)
        self.hand_ids = joint_ids(self._entity, parts.hand_joints, dev)
        self.arm_ids = joint_ids(self._entity, parts.arm_joints, dev)
        self.arm_target = torch.as_tensor(parts.arm_target, dtype=torch.float32, device=dev)
        self._raw = torch.zeros(self.num_envs, 0, device=dev)
        self.base_cmd = base_command_term(env)
        self.rt = getattr(env, "_tactile_rt", None)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        del actions

    def apply_actions(self) -> None:
        d = self._entity.data
        if self.base_cmd is not None:
            d.joint_pos_target[:, self.arm_ids] = self.base_cmd.arm_target(self.rt.t_pre)
        else:
            d.joint_pos_target[:, self.arm_ids] = self.arm_target
        if self.rt is not None:
            self.rt.pre_substep()
        self.drive.step(d.joint_pos[:, self.hand_ids], d.joint_vel[:, self.hand_ids])
