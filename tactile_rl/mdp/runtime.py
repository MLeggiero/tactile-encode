"""The per-env state the tactile terms share, created once per env (`tactile_runtime`): the substep clock, the robot's
sensors (taxel load, TaxelScan rev3, WUJI readback), the nail plant, the student's 1 kHz input ring, and the truth the
privileged observations and the rewards read.

Clock: each env counts physics substeps since its reset (`k`); before a substep its time is k dt, after it (k + 1) dt,
float64. Resets happen between env steps and every task's decimation is a multiple of the substeps per millisecond,
so every env's millisecond boundaries fall on the same substeps.

Per substep, in the testbed's order: before MuJoCo steps (SubstepHooks.apply_actions) the arm targets, the plant's
pre_step and the drives; after it (`post_substep`, through the scene's TactileHub sensor) the taxel loads, the scan,
the readback, the plant's measurements and, on millisecond boundaries, the student's 1 kHz sample.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from tactile_rl.mdp.actions import hand_drive, joint_ids
from tactile_rl.mdp.kinematics import rotvec_from_matrix
from tactile_rl.mdp.obs_builder import StudentObs
from tactile_rl.plants.nail import TorchNailPlant
from tactile_rl.scene.entities import ENTITY
from tactile_rl.sensors.hub import HUB
from tactile_rl.sensors.readback import TorchReadback
from tactile_rl.sensors.taxel_load import TaxelLoadModel
from tactile_rl.sensors.taxelscan import TorchTaxelScan
from tactile_sim.control.obs_spec import BOARDS, stack_for
from tactile_sim.sim.training_parts import training_parts

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

PREFIX = f"{ENTITY}/"


class TactileRuntime:
    def __init__(self, env: ManagerBasedRlEnv, scene_key: str, task: str = "hammer",
                 mask_blocks: tuple[str, ...] = (), compiled: bool = True):
        self.env = env
        self.parts = parts = training_parts(scene_key)
        self.cfg = cfg = parts.cfg
        self.sim = sim = env.sim
        self.robot = robot = env.scene[ENTITY]
        self.dev = dev = env.device
        self.N = N = env.num_envs
        self.dt = float(env.physics_dt)
        self.steps_per_ms = int(round(1e-3 / self.dt))
        if abs(self.steps_per_ms * self.dt - 1e-3) > 1e-12 or env.cfg.decimation % self.steps_per_ms:
            raise ValueError("the substep clock needs a whole number of physics steps per millisecond and per policy "
                             "step")
        self.k = torch.zeros(N, dtype=torch.long, device=dev)
        self._g = 0  # substeps since construction (host side): its millisecond phase is every env's
        self.drive = hand_drive(robot)
        self.hand_ids = joint_ids(robot, parts.hand_joints, dev)
        self.arm_ids = joint_ids(robot, parts.arm_joints, dev)
        m = sim.mj_model
        # MuJoCo Warp gives a capsule/mesh pair one contact: at most two per patch (a distal pad has two hulls)
        self.load = TaxelLoadModel(parts, m, N, dev, prefix=PREFIX, max_per_patch=4, compiled=compiled)
        self.scan = TorchTaxelScan(parts.skin, cfg.sensors, self.dt, N, dev)
        w = cfg.wuji
        every = int(round(1.0 / (w.sdk_rate * self.dt)))
        self.rb_q = TorchReadback(N, len(parts.hand_joints), self.dt, every, w.fb_latency, 0.0,
                                  2.0 * math.pi / 2**w.enc_bits, dev)
        self.rb_i = TorchReadback(N, len(parts.hand_joints), self.dt, every, w.fb_latency, w.i_noise, w.i_quant, dev)
        self.nail = TorchNailPlant(parts, sim, N, dev, PREFIX) if parts.nail is not None else None
        self.rate = 1.0 / env.step_dt
        self.stack = stack_for(self.rate)
        self.task = task
        self.obs = StudentObs(N, self.stack, [p.grid for p in parts.skin], self.scan.codec.levels, mask_blocks, task,
                              dev)
        self.res = cfg.residual
        # truth
        self.tcp_site = m.site(PREFIX + parts.tcp_site).id
        self.ref_site = m.site(PREFIX + parts.hammer_site).id
        self.hammer_body = m.body(PREFIX + parts.hammer_body).id
        self.hand_body = m.body(PREFIX + "hand").id
        self.patch_body = torch.as_tensor([m.body(PREFIX + p.body).id for p in parts.skin], device=dev)
        nb = np.stack([p.taxel_normal.mean(0) for p in parts.skin])
        self.patch_normal_b = torch.as_tensor(nb / np.linalg.norm(nb, axis=1, keepdims=True), dtype=torch.float32,
                                              device=dev)  # each skin's mean outward normal, its body's frame
        self.body_root_np = np.asarray(m.body_rootid).astype(int)
        self.body_root = torch.as_tensor(self.body_root_np, device=dev)
        self.t_onset = torch.full((N,), -math.inf, dtype=torch.float64, device=dev)  # the last blow's onset
        self.t_touch = torch.full((N,), -math.inf, dtype=torch.float64, device=dev)  # the face last touched the nail
        self.f_peak = torch.zeros(N, device=dev)  # largest face/nail force since reset
        self.n_blows = torch.zeros(N, dtype=torch.long, device=dev)
        self.recorders: list = []  # f(runtime) after every substep (parity scripts, diagnostics)
        hub = env.scene.sensors.get(HUB) if hasattr(env.scene.sensors, "get") else None
        if hub is None:
            raise RuntimeError(f"the tactile runtime needs the {HUB!r} sensor in the scene (TactileHubCfg)")
        hub.post = self.post_substep
        hub.on_reset = self.reset
        self.reset(None)

    # ------------------------------------------------------------------ clock
    @property
    def t_pre(self) -> torch.Tensor:
        """Each env's time before the coming substep (s, float64)."""
        return self.k.double() * self.dt

    t = t_pre  # between substeps (at policy ticks and observations) the env's time

    # ------------------------------------------------------------------ resets
    def reset(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.N, device=self.dev)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.N, device=self.dev)[env_ids]
        self.k[env_ids] = 0
        self.load.reset(env_ids)
        self.scan.reset(env_ids)
        self.rb_q.reset(env_ids)
        self.rb_i.reset(env_ids)
        if self.nail is not None:
            self.nail.reset(env_ids)
        self.obs.reset(env_ids)
        self.t_onset[env_ids] = -math.inf
        self.t_touch[env_ids] = -math.inf
        self.f_peak[env_ids] = 0.0
        self.n_blows[env_ids] = 0
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        mask[env_ids] = True
        self.obs.push(self.sample(update=mask), mask)  # the 1 kHz sample at t = 0, from the reset sensors

    # ------------------------------------------------------------------ the substep hooks
    def pre_substep(self) -> None:
        """Before MuJoCo steps (called by SubstepHooks after the arm targets, before the drives)."""
        if self.nail is not None:
            self.nail.pre_step(self.t_pre)

    def post_substep(self) -> None:
        t = (self.k + 1).double() * self.dt
        data = self.sim.data
        self.scan.step(t, self.load.step(data, t))
        self.rb_q.step(t, self.robot.data.joint_pos[:, self.hand_ids])
        self.rb_i.step(t, self.drive.i)
        if self.nail is not None:
            self.nail.post_step(data)
            f = self.nail.force
            touching = f > 0
            onset = touching & (t - self.t_touch > 0.005)  # a new blow, not chatter (NailPlant's 5 ms)
            self.t_onset = torch.where(onset, t, self.t_onset)
            self.t_touch = torch.where(touching, t, self.t_touch)
            self.n_blows += onset.long()
            self.f_peak = torch.maximum(self.f_peak, f)
        self.k += 1
        self._g += 1
        if self._g % self.steps_per_ms == 0:  # a millisecond boundary for every env (see the module doc)
            self.obs.push(self.sample(t))
        for fn in self.recorders:
            fn(self)

    def sample(self, t: torch.Tensor | None = None, update: torch.Tensor | None = None) -> torch.Tensor:
        """The student's 1 kHz row now (ResidualHandL2.fast_sample)."""
        t = self.t_pre if t is None else t
        seq, avail = self.scan.board_status()
        return self.obs.sample(t, self.scan.codes, self.scan.features, seq, avail, self.rb_q.value,
                               self.rb_i.value / self.drive.i_rated.double(), update)

    # ------------------------------------------------------------------ the residual's view
    def stale(self, t: torch.Tensor | None = None) -> torch.Tensor:
        """ResidualHandL2.stale: some board's newest frame is older than stale_taxel (or none has come), or the joint
        readback is older than stale_joint (or none has come)."""
        t = self.t if t is None else t
        seq, avail = self.scan.board_status()
        age = torch.where(seq >= 0, t[:, None] - avail, torch.full_like(avail, math.inf))
        q = self.rb_q
        return (age.max(1).values > self.res.stale_taxel) | (q.seq < 0) | (t - q.t_avail > self.res.stale_joint)

    # ------------------------------------------------------------------ truth
    def hammer_in_hand(self) -> torch.Tensor:
        """(N, 6) the hammer's grasp frame in the TCP frame (World.hammer_in_hand), from the current kinematics."""
        d = self.sim.data
        xpos, xmat = d.site_xpos[:], d.site_xmat[:]
        Rt = xmat[:, self.tcp_site].reshape(-1, 3, 3)
        Rh = xmat[:, self.ref_site].reshape(-1, 3, 3)
        p = torch.einsum("nji,nj->ni", Rt, xpos[:, self.ref_site] - xpos[:, self.tcp_site])
        return torch.cat([p, rotvec_from_matrix(Rt.transpose(1, 2) @ Rh)], dim=-1)

    def point_velocity(self, body: torch.Tensor | int, point: torch.Tensor) -> torch.Tensor:
        """Linear velocity of `body`'s material point at world `point` (mj_objectVelocity's transform): body an id
        and point (N, 3) or (N, P, 3), or body (P,) ids and point (N, P, 3)."""
        d = self.sim.data
        cvel = d.cvel[:][:, body]
        root = int(self.body_root_np[body]) if isinstance(body, int) else self.body_root[body]  # no device scalar
        com = d.subtree_com[:][:, root]
        while cvel.ndim < point.ndim:
            cvel, com = cvel[:, None], com[:, None]
        return cvel[..., 3:] - torch.cross(point - com, cvel[..., :3].expand_as(point), dim=-1)

    def angular_velocity(self, body) -> torch.Tensor:
        return self.sim.data.cvel[:][:, body][..., :3]

    @property
    def board_count(self) -> int:
        return BOARDS


def tactile_runtime(env: ManagerBasedRlEnv, scene_key: str | None = None, **kw) -> TactileRuntime:
    """The env's runtime, created by the first term that asks (with the scene key)."""
    rt = getattr(env, "_tactile_rt", None)
    if rt is None:
        if scene_key is None:
            raise RuntimeError("the tactile runtime is created with the scene key")
        rt = TactileRuntime(env, scene_key, **kw)
        env._tactile_rt = rt
    return rt
