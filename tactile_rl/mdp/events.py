"""Resets from a bank of settled states.

At startup every GPU world runs the CPU testbed's reset (World.reset_pose, then World.settle): the arm at the hover
pose on its servos, the hand at the grasp keyframe closing on the hammer welded to it while the grip ramps to its
hold over the first half of 0.15 s, then the weld lets go and the hammer seats under gravity for 0.25 s; the drives'
clocks are rebased to t = 0. The settled states (qpos, qvel, the solver's warm start and the drives' full state,
commands in flight included) form the bank; a reset copies a random entry into each resetting env.

The bank's diversity comes from the drives' command jitter (and, once randomization is on, from the per-world
parameters); `refresh_state_bank` rebuilds it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

from tactile_rl.mdp.action_map import TorchSynergy
from tactile_rl.mdp.actions import hand_drive, joint_ids
from tactile_rl.mdp.kinematics import hammer_in_hand
from tactile_rl.scene.entities import ENTITY
from tactile_sim.sim.training_parts import training_parts

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass
class StateBank:
    qpos: torch.Tensor  # (B, nq)
    qvel: torch.Tensor  # (B, nv)
    qacc_warmstart: torch.Tensor  # (B, nv)
    drive: dict[str, torch.Tensor]  # TorchWujiDrive.get_state(), (B, ...)
    hammer_rel: torch.Tensor  # (B, 6) hammer pose in the TCP frame: position, rotation vector

    @property
    def size(self) -> int:
        return self.qpos.shape[0]


def tactile_state(env: ManagerBasedRlEnv) -> TactileState:
    """Per-env bookkeeping the tactile terms share, kept on the env: the bank and each env's reference hammer pose."""
    st = getattr(env, "_tactile", None)
    if st is None:
        st = TactileState(bank=None, ref_rel=torch.zeros(env.num_envs, 6, device=env.device))
        env._tactile = st
    return st


@dataclass
class TactileState:
    bank: StateBank | None
    ref_rel: torch.Tensor  # (N, 6) each env's settled hammer pose in the hand (reward / termination reference)
    settle_peaks: CapacityPeaks | None = None


def weld_id(env: ManagerBasedRlEnv, name: str) -> int:
    return env.sim.mj_model.equality(f"{ENTITY}/{name}").id


class CapacityPeaks:
    """Peak use of MuJoCo Warp's contact and constraint buffers, accumulated on the GPU (no host sync per step):
    broadphase candidates and contacts over all worlds (both must fit nconmax x nworld), constraint rows in the
    fullest world (must fit njmax), and any overflow flag raised on the way (MuJoCo Warp clears it every step)."""

    def __init__(self, device):
        self.peaks = torch.zeros(4, dtype=torch.long, device=device)

    def update(self, env: ManagerBasedRlEnv) -> None:
        d = env.sim.data
        now = torch.stack([d.ncollision[0].long(), d.nacon[0].long(), d.nefc.max().long(), d.overflow.max().long()])
        torch.maximum(self.peaks[:3], now[:3], out=self.peaks[:3])
        self.peaks[3] |= now[3]

    @property
    def overflow(self) -> int:
        return int(self.peaks[3])

    def report(self, num_envs: int) -> dict[str, float]:
        ncol, ncon, nefc, overflow = (int(x) for x in self.peaks.cpu())
        return {"broadphase_per_world": ncol / num_envs, "contacts_per_world": ncon / num_envs, "nefc_max": nefc,
                "overflow": overflow}


def settle_worlds(env: ManagerBasedRlEnv, scene_key: str) -> StateBank:
    """World.reset in every world at once (see module doc); returns the settled states."""
    parts = training_parts(scene_key)
    sim, robot = env.sim, env.scene[ENTITY]
    data, dev, dt = sim.data, env.device, env.physics_dt
    if sim.mj_model.nq != parts.qpos_start.shape[0]:
        raise RuntimeError("the GPU model's qpos differs from the testbed's")
    drive = hand_drive(robot)
    hand_ids = joint_ids(robot, parts.hand_joints, dev)
    arm_ids = joint_ids(robot, parts.arm_joints, dev)
    arm_target = torch.as_tensor(parts.arm_target, dtype=torch.float32, device=dev)
    synergy = TorchSynergy(parts.synergy, dev)
    weld = weld_id(env, parts.weld)
    wp.to_torch(sim.wp_model.eq_data)[:, weld] = torch.as_tensor(parts.weld_eq_data, dtype=torch.float32, device=dev)

    sim.reset()
    data.qpos[:] = torch.as_tensor(parts.qpos_start, dtype=torch.float32, device=dev)
    data.qvel[:] = 0.0
    data.eq_active[:, weld] = True
    sim.forward()
    drive.reset()
    n_weld = int(round(parts.settle_weld / dt))
    n_free = int(round(parts.settle_free / dt))
    s = torch.zeros(env.num_envs, device=dev)
    peaks = tactile_state(env).settle_peaks = CapacityPeaks(dev)  # the welded phase loads the buffers most
    for i in range(n_weld + n_free):
        if i == n_weld:
            data.eq_active[:, weld] = False
        s.fill_(parts.s_settle(i, n_weld) if i < n_weld else parts.s_hold)
        q, qd = robot.data.joint_pos[:, hand_ids], robot.data.joint_vel[:, hand_ids]
        drive.set_host(synergy.mit_command(s, q))
        robot.data.joint_pos_target[:, arm_ids] = arm_target
        drive.step(q, qd)
        robot.write_data_to_sim()
        sim.step()
        peaks.update(env)
    if peaks.overflow:
        p = peaks.report(env.num_envs)
        raise RuntimeError(f"MuJoCo Warp buffers overflowed while settling (flags {p['overflow']:#x}; peaks: "
                           f"{p['broadphase_per_world']:.1f} broadphase candidates and {p['contacts_per_world']:.1f} "
                           f"contacts per world, {p['nefc_max']} constraint rows): raise SimulationCfg.nconmax / njmax")
    drive.shift_time(-drive.t.clone())  # time restarts at 0 (World.settle)
    data.time[:] = 0.0
    sim.forward()
    return StateBank(qpos=data.qpos.clone(), qvel=data.qvel.clone(), qacc_warmstart=data.qacc_warmstart.clone(),
                     drive=drive.get_state(), hammer_rel=hammer_in_hand(env, scene_key))


def build_state_bank(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, scene_key: str) -> None:
    """Startup event: settle every world and keep the result as the reset bank."""
    del env_ids
    tactile_state(env).bank = settle_worlds(env, scene_key)


def refresh_state_bank(env: ManagerBasedRlEnv, scene_key: str) -> None:
    """Rebuild the bank (e.g. after the randomization ranges change); every env is reset afterwards by the caller."""
    tactile_state(env).bank = settle_worlds(env, scene_key)


def reset_from_state_bank(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, scene_key: str) -> None:
    """Reset event: each resetting env starts from a random settled state."""
    st = tactile_state(env)
    bank = st.bank
    if bank is None:
        raise RuntimeError("no state bank: add the build_state_bank startup event")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    parts = training_parts(scene_key)
    robot = env.scene[ENTITY]
    idx = torch.randint(0, bank.size, (len(env_ids),), device=env.device)
    data = env.sim.data
    data.qpos[env_ids] = bank.qpos[idx]
    data.qvel[env_ids] = bank.qvel[idx]
    data.qacc_warmstart[env_ids] = bank.qacc_warmstart[idx]
    data.time[env_ids] = 0.0
    hand_drive(robot).set_state({k: v[idx] for k, v in bank.drive.items()}, env_ids)
    arm_ids = joint_ids(robot, parts.arm_joints, env.device)
    robot.data.joint_pos_target[env_ids[:, None], arm_ids] = torch.as_tensor(parts.arm_target, dtype=torch.float32,
                                                                              device=env.device)
    st.ref_rel[env_ids] = bank.hammer_rel[idx]
