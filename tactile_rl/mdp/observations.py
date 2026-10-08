"""Observation terms.

- `student_obs`: the student's observation exactly as obs_spec lays it out (ResidualHandL2.observe): the 1 kHz fast
  blocks from the robot's sensors (TaxelScan codes, patch features and board status, the readback's angles and
  currents), stacked `stack` deep for a policy slower than 1 kHz, then the last squashed action, the base's hand target
  now, its lookahead, the forecast, the task and trigger_enable; masked blocks read zero.
- privileged terms for the teacher and the critic (each scaled to O(1)): the tool in the hand (pose change since the
  episode's start and its twist relative to the hand), per-patch slip velocity and true normal force, the true event
  timing and regime, the nail, the drives' internals and the randomization the env runs with.
- `hand_joint_pos`, `hand_current`: milestone 1's placeholder terms (true angles, currents), kept for the hold tasks.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from tactile_rl.mdp.actions import base_command_term, hand_drive, joint_ids
from tactile_rl.scene.entities import ENTITY
from tactile_sim.control.obs_spec import REGIMES
from tactile_sim.sim.training_parts import training_parts

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class hand_joint_pos:  # noqa: N801 - mjlab term naming
    """Hand joint angles (rad), (N, 20) in the testbed's joint order."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        robot = env.scene[ENTITY]
        self.robot = robot
        self.ids = joint_ids(robot, training_parts(cfg.params["scene_key"]).hand_joints, env.device)

    def __call__(self, env: ManagerBasedRlEnv, scene_key: str) -> torch.Tensor:
        del scene_key
        return self.robot.data.joint_pos[:, self.ids]


def hand_current(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Hand motor currents as a fraction of each joint's rated current (obs_spec "hand_current"), (N, 20)."""
    drive = hand_drive(env.scene[ENTITY])
    return drive.i / drive.i_rated


def _parts(env: ManagerBasedRlEnv):
    cmd = base_command_term(env)
    if cmd is None:
        raise RuntimeError("the tactile observations need the base-trajectory command (BaseTrajectoryCommandCfg)")
    return cmd, cmd.rt


def student_obs(env: ManagerBasedRlEnv, action_name: str = "residual") -> torch.Tensor:
    """obs_spec's student observation (N, obs_dim(stack))."""
    cmd, rt = _parts(env)
    act = env.action_manager.get_term(action_name)
    return rt.obs.observe(act.a_sq, cmd.base[:, 0], cmd.lookahead, cmd.forecast, cmd.trigger)


def tool_in_hand(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 12): the hammer's pose in the hand relative to the episode's start (mm, degrees / 10) and its twist relative
    to the hand in the TCP frame (linear 0.1 m/s, angular rad/s units)."""
    cmd, rt = _parts(env)
    rel = rt.hammer_in_hand() - cmd.ref_rel
    d = rt.sim.data
    Rt = d.site_xmat[:][:, rt.tcp_site].reshape(-1, 3, 3)
    p = d.site_xpos[:][:, rt.ref_site]
    v = rt.point_velocity(rt.hammer_body, p) - rt.point_velocity(rt.hand_body, p)
    w = rt.angular_velocity(rt.hammer_body) - rt.angular_velocity(rt.hand_body)
    v_t = torch.einsum("nji,nj->ni", Rt, v)
    w_t = torch.einsum("nji,nj->ni", Rt, w)
    return torch.cat([rel[:, :3] * 1e3, torch.rad2deg(rel[:, 3:]) / 10.0, v_t / 0.1, w_t], dim=1)


def patch_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 4 P): per patch the true normal force (10 N) and the tool's slip velocity over the skin at the patch's
    contacts (force-weighted point; the relative velocity's part tangent to the skin, TCP frame, 0.1 m/s)."""
    _, rt = _parts(env)
    load = rt.load
    f = torch.where(load.contact_ok, load.contact_f, torch.zeros_like(load.contact_f))  # (N, P, K)
    fs = f.sum(-1)
    c = (f[..., None] * load.contact_pos).sum(2) / fs.clamp(min=1e-9)[..., None]  # (N, P, 3) centroid
    v_tool = rt.point_velocity(rt.hammer_body, c)
    v_patch = rt.point_velocity(rt.patch_body, c)
    v = v_tool - v_patch
    d = rt.sim.data
    # the skin's normal at the centroid: the patch body's mean taxel normal (body frame -> world)
    xmat = d.xmat[:][:, rt.patch_body].reshape(rt.N, -1, 3, 3)
    n_w = torch.einsum("npij,pj->npi", xmat, rt.patch_normal_b)
    n_w = n_w / n_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    v_t = v - (v * n_w).sum(-1, keepdim=True) * n_w
    Rt = d.site_xmat[:][:, rt.tcp_site].reshape(-1, 3, 3)
    v_t = torch.einsum("nji,npj->npi", Rt, v_t)
    v_t = torch.where((fs > 0)[..., None], v_t, torch.zeros_like(v_t))
    return torch.cat([load.total / 10.0, (v_t / 0.1).reshape(rt.N, -1)], dim=1)


def event_truth(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 10): the base's true plan (valid, time to its predicted contact, regime one-hot) and the true contact
    (touching now, time since the last blow's onset, its peak force so far)."""
    cmd, rt = _parts(env)
    tr = cmd.truth
    t = rt.t
    valid = tr["valid"].float()
    dt_ev = torch.clamp(tr["t_event"] - t, -1.0, 1.0).float() * valid
    reg = torch.zeros(rt.N, len(REGIMES), device=rt.dev)
    reg.scatter_(1, tr["regime"][:, None], 1.0)
    reg = reg * valid[:, None]
    touching = (rt.nail.force > 0).float() if rt.nail is not None else torch.zeros(rt.N, device=rt.dev)
    since = torch.clamp(t - rt.t_onset, max=1.0).float()
    return torch.cat([valid[:, None], dt_ev[:, None], reg, touching[:, None], since[:, None],
                      (rt.f_peak / 100.0)[:, None]], dim=1)


def nail_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 5): depth (cm), resistance (100 N), face gap (5 cm units, clipped), approach speed (m/s), contact force
    (100 N)."""
    _, rt = _parts(env)
    nail = rt.nail
    if nail is None:
        return torch.zeros(rt.N, 5, device=rt.dev)
    gap = torch.clamp(nail.gap, -0.05, 0.05) / 0.05
    return torch.stack([nail.depth() * 100.0, nail.resistance().float() / 100.0, gap.float(), nail.v_axis.float(),
                        nail.force / 100.0], dim=1)


def drive_internals(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 44): the drives' currents / rated, winding temperatures above ambient (50 K units), any fault, command
    latency (ms), SDK low-pass and current-loop coefficients."""
    drive = hand_drive(env.scene[ENTITY])
    dT = ((drive.T - drive.t_amb[:, None]) / 50.0).float()
    return torch.cat([drive.i / drive.i_rated, dT, drive.fault.any(1, keepdim=True).float(),
                      (drive.cmd_latency * 1e3).float()[:, None], drive.lpf_a[:, None], drive.i_a[:, None]], dim=1)


def randomization(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(N, 25): the randomization the env runs with (BaseTrajectoryCommand.param_vector)."""
    cmd, _ = _parts(env)
    return cmd.param_vector()


PRIVILEGED = {"tool_in_hand": 12, "patch_contact": 44, "event_truth": 10, "nail_state": 5, "drive_internals": 44,
              "randomization": 25}
assert math.isfinite(sum(PRIVILEGED.values()))
