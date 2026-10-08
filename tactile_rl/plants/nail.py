"""The nail plant batched over environments: the GPU mirror of tactile_sim.model.plants.nail.NailPlant.

Before every physics substep (`pre_step`, as NailPlant.pre_step before mj_step), per world:

- the nail joint's friction loss (its Coulomb resistance) grows with depth: resistance_0 + resistance_per_m x depth,
  written into the per-world `dof_frictionloss`;
- velocity-dependent restitution: while the face is not touching the nail head and is within 1 cm of it, the face/nail
  pair's damping ratio follows the approach speed, zeta = min(zeta_max, zeta0 + zeta1 v), written into the per-world
  `pair_solref` in the direct form (-k_face, -2 zeta sqrt(k_face)); during contact it stays frozen.

The plant reads what the testbed's pre_step reads: the contacts, the face and nail-head site positions and the face's
velocity of the previous physics step (MuJoCo's data after mj_step, before integration). `post_step` takes them from
MuJoCo Warp right after each step, so mjlab's per-env-step forward() does not shift them. It also keeps the truth the
rewards and the strike metrics need: the face/nail contact force, the blow's onset, the damping ratio at onset.
"""

from __future__ import annotations

import math

import torch


class TorchNailPlant:
    def __init__(self, parts, sim, num_envs: int, device: str | torch.device, prefix: str = "robot/"):
        nail = parts.nail
        if nail is None:
            raise ValueError("the scene has no nail plant")
        m = sim.mj_model
        self.sim = sim
        self.dev = dev = torch.device(device)
        self.n_env = N = int(num_envs)
        j = m.joint(prefix + nail.joint)
        self.qadr, self.dof = int(j.qposadr[0]), int(j.dofadr[0])
        self.pair = m.pair(prefix + nail.face_pair).id
        self.face_site = m.site(prefix + nail.face_site).id
        self.head_site = m.site(prefix + nail.head_site).id
        self.face_geom = m.geom(prefix + nail.face_geom).id
        self.nail_geom = m.geom(prefix + nail.nail_geom).id
        b = m.body(prefix + parts.hammer_body).id
        self.hammer_body, self.hammer_root = b, int(m.body_rootid[b])
        self.axis = torch.as_tensor(nail.axis, dtype=torch.float32, device=dev)
        self.dmax, self.face_zeta = float(nail.dmax), float(nail.face_zeta)
        p = parts.cfg.plant
        f64 = dict(dtype=torch.float64, device=dev)
        self.r0 = torch.full((N,), float(p.resistance_0), **f64)
        self.r_per_m = torch.full((N,), float(p.resistance_per_m), **f64)
        self.zeta0 = torch.full((N,), float(p.vdr_zeta0), **f64)
        self.zeta1 = torch.full((N,), float(p.vdr_zeta1), **f64)
        self.zeta_max = torch.full((N,), float(p.vdr_zeta_max), **f64)
        self.k_face = torch.full((N,), float(nail.k_face), **f64)
        self.vdr = bool(p.vdr_enabled)
        self.drive_target = float(p.drive_target)
        # state (NailPlant: _in_contact, _last_contact_t, last_zeta, contact_zeta) and what post_step measured
        self.in_contact = torch.zeros(N, dtype=torch.bool, device=dev)
        self.last_contact_t = torch.full((N,), -math.inf, **f64)
        self.last_zeta = self.zeta0.clone()
        self.contact_zeta = torch.full((N,), float("nan"), **f64)
        self.active = torch.zeros(N, dtype=torch.bool, device=dev)  # face/nail contact (dist < 0) last step
        self.gap = torch.full((N,), math.inf, **f64)
        self.v_axis = torch.zeros(N, **f64)
        self.force = torch.zeros(N, dtype=torch.float32, device=dev)  # face/nail normal force last step (truth)
        self.onset = torch.zeros(N, dtype=torch.bool, device=dev)  # a new blow began last step

    def set_params(self, env_ids=None, resistance_0=None, resistance_per_m=None, vdr_zeta0=None, face_tc=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        for name, v in (("r0", resistance_0), ("r_per_m", resistance_per_m), ("zeta0", vdr_zeta0)):
            if v is not None:
                getattr(self, name)[ids] = torch.as_tensor(v, dtype=torch.float64, device=self.dev)
        if face_tc is not None:
            tc = torch.as_tensor(face_tc, dtype=torch.float64, device=self.dev)
            self.k_face[ids] = 1.0 / (self.dmax**2 * tc**2 * self.face_zeta**2)

    def reset(self, env_ids=None) -> None:
        """As NailPlant.bind + reset, the nail's depth aside (the env's state sets it): the face/nail pair back at
        zeta0, no contact."""
        ids = torch.arange(self.n_env, device=self.dev) if env_ids is None else env_ids
        self.in_contact[ids] = False
        self.last_contact_t[ids] = -math.inf
        self.contact_zeta[ids] = float("nan")
        self.active[ids] = False
        self.gap[ids] = math.inf
        self.v_axis[ids] = 0.0
        self.force[ids] = 0.0
        self.onset[ids] = False
        if self.vdr:
            self._set_zeta(ids, self.zeta0[ids])

    def _set_zeta(self, ids, zeta: torch.Tensor) -> None:
        k = self.k_face[ids]
        solref = self.sim.model.pair_solref
        solref[ids, self.pair, 0] = (-k).float()
        solref[ids, self.pair, 1] = (-2.0 * zeta * torch.sqrt(k)).float()
        self.last_zeta[ids] = zeta

    def depth(self) -> torch.Tensor:
        return self.sim.data.qpos[:, self.qadr]

    def resistance(self) -> torch.Tensor:
        return self.r0 + self.r_per_m * torch.clamp(self.depth().double(), min=0.0)

    # ------------------------------------------------------------------ hooks
    def pre_step(self, t: torch.Tensor) -> None:
        """Before MuJoCo steps, at time t (N,) float64."""
        self.sim.model.dof_frictionloss[:, self.dof] = self.resistance().float()
        if not self.vdr:
            return
        in_contact = self.active
        approach = ~in_contact & (self.gap < 0.01)
        zeta = torch.minimum(self.zeta_max, self.zeta0 + self.zeta1 * torch.clamp(self.v_axis, min=0.0))
        k = self.k_face
        solref = self.sim.model.pair_solref
        z = torch.where(approach, zeta, self.last_zeta)
        solref[:, self.pair, 0] = (-k).float()
        solref[:, self.pair, 1] = (-2.0 * z * torch.sqrt(k)).float()
        self.last_zeta = z
        new_blow = in_contact & ~self.in_contact & (t - self.last_contact_t > 0.005)
        self.contact_zeta = torch.where(new_blow, self.last_zeta, self.contact_zeta)
        self.last_contact_t = torch.where(in_contact, t, self.last_contact_t)
        self.onset = new_blow
        self.in_contact = in_contact

    def post_step(self, data) -> None:
        """After MuJoCo stepped: the face/nail contact, the gap along the strike axis and the face's approach speed,
        from the step's own (pre-integration) kinematics."""
        N, dev = self.n_env, self.dev
        con = data.contact
        geom = con.geom[:]
        C = geom.shape[0]
        g1, g2 = geom[:, 0].long(), geom[:, 1].long()
        pair = ((g1 == self.face_geom) & (g2 == self.nail_geom)) | ((g1 == self.nail_geom) & (g2 == self.face_geom))
        valid = (torch.arange(C, device=dev) < data.nacon[:].long()) & pair
        wid = con.worldid[:].long().clamp(0, N - 1)
        active = torch.zeros(N + 1, dtype=torch.bool, device=dev)
        active.scatter_(0, torch.where(valid & (con.dist[:] < 0), wid, torch.full_like(wid, N)),
                        torch.ones(C, dtype=torch.bool, device=dev))
        self.active = active[:N]
        adr = con.efc_address[:]
        adr = (adr[:, 0] if adr.ndim > 1 else adr).long()
        f = data.efc.force[:][wid, adr.clamp(min=0)]
        f = torch.where(valid & (adr >= 0) & (f > 0), f, torch.zeros_like(f))
        force = torch.zeros(N + 1, dtype=torch.float32, device=dev)
        force.scatter_add_(0, torch.where(valid, wid, torch.full_like(wid, N)), f.float())
        self.force = force[:N]
        xs = data.site_xpos[:]
        face, head = xs[:, self.face_site], xs[:, self.head_site]
        self.gap = ((head - face) @ self.axis).double()
        cvel = data.cvel[:][:, self.hammer_body]  # [angular; linear at the root's subtree com]
        com = data.subtree_com[:][:, self.hammer_root]
        v_lin = cvel[:, 3:] - torch.cross(face - com, cvel[:, :3], dim=-1)  # mju_transformSpatial
        self.v_axis = (v_lin @ self.axis).double()

    def done(self) -> torch.Tensor:
        return self.depth() >= self.drive_target
