"""Hand runtime interfaces: what the World, grip loop, sensors and episode need from a hand.

- `FrankaHandIO`: two parallel fingers on one force-controlled drive; a taxel patch on each pad (L, R).
  Grip force = mean normal force per pad.
- `WujiHandIO`: WUJI Hand 2 in a power wrap; taxel patches where the grasp loads the hand (palm, thumb).
  Grip force = summed normal force over the patches. The grip command sets the wrap synergy's scalar,
  and the hand's own 1 kHz joint law (`tick`) turns it into joint torques within each joint's rating.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.control.hand import WrapSynergy


@dataclass
class Patch:
    name: str
    body: int
    site: int  # taxel frame: x = rows, y = columns, z = outward normal; None for the Franka pads
    half: tuple[float, float]
    geoms: frozenset[int]  # hand geoms that load this patch
    franka_pad: bool = False


class HandIO:
    kind = "base"
    patches: list[Patch]
    grip_force_max: float  # largest grip-force command (N)
    grip_hold_max: float  # largest continuous hold the swing may ask for (N)
    grip_pre_max: float  # largest pre-impact squeeze the swing may ask for (N)

    def __init__(self, world):
        self.world = world
        m = world.model
        self.tool_geoms = frozenset(g for g in range(m.ngeom) if m.geom_bodyid[g] == world.hammer_body)

    @property
    def accel_names(self) -> list[str]:
        return [f"pad_acc_{p.name}" for p in self.patches]

    @property
    def pressure_names(self) -> list[str]:
        return [f"pressure_{p.name}" for p in self.patches]

    # --- to implement
    def init_pose(self) -> None: ...
    def set_grip(self, f: float) -> None: ...
    def tick(self, t: float) -> None:  # hand-level control at the hand's own rate
        pass

    def grip_from_patches(self, forces: np.ndarray) -> float:
        raise NotImplementedError

    # --- shared
    def patch_forces(self) -> np.ndarray:
        """Ground-truth normal force on each patch from the tool (N): the taxels' total without the spread."""
        w = self.world
        m, d = w.model, w.data
        out = np.zeros(len(self.patches))
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.tool_geoms:
                g1, g2 = g2, g1
            elif g2 not in self.tool_geoms:
                continue
            for k, p in enumerate(self.patches):
                if g1 not in p.geoms or not self._inside(p, c.pos):
                    continue
                mujoco.mj_contactForce(m, d, i, f6)
                out[k] += max(f6[0], 0.0)
        return out

    def _inside(self, p: Patch, pos: np.ndarray, spread: float = 0.003) -> bool:
        if p.franka_pad:
            return True
        d = self.world.data
        q = d.site_xmat[p.site].reshape(3, 3).T @ (pos - d.site_xpos[p.site])
        return abs(q[2]) <= 0.006 and abs(q[0]) <= p.half[0] + spread and abs(q[1]) <= p.half[1] + spread

    def grip_truth(self) -> float:
        return self.grip_from_patches(self.patch_forces())

    def taxels(self, k: int, spread: float = 0.003) -> np.ndarray:
        """Normal force per taxel (row-major, rows along the patch's first axis).

        MuJoCo reduces a contact patch to a few points; a rubber skin spreads each load over its contact
        patch. Each contact's force is shared among the taxels with Gaussian weights (sigma = `spread`,
        about the rubber layer's thickness) evaluated at the cell centres; the weights are normalised so the
        total force is exact. Contacts are placed by their position in the patch frame; on a dexterous
        hand, contacts outside the patch (plus the spread) do not load it.
        """
        w = self.world
        m, d = w.model, w.data
        p = self.patches[k]
        nr, nc = w.cfg.sensors.taxel_grid
        hu, hv = p.half
        grid = self._grids.get(k)
        if grid is None:
            us = -hu + (2 * np.arange(nr) + 1) * hu / nr
            vs = -hv + (2 * np.arange(nc) + 1) * hv / nc
            grid = np.stack(np.meshgrid(us, vs, indexing="ij"), axis=-1).reshape(-1, 2)
            self._grids[k] = grid
        out = np.zeros(nr * nc)
        f6 = np.zeros(6)
        if p.franka_pad:
            R, x0 = d.xmat[p.body].reshape(3, 3), d.xpos[p.body]
        else:
            R, x0 = d.site_xmat[p.site].reshape(3, 3), d.site_xpos[p.site]
        for i in range(d.ncon):
            c = d.contact[i]
            if c.efc_address < 0:
                continue
            g1, g2 = c.geom1, c.geom2
            if not ((g1 in p.geoms and g2 in self.tool_geoms) or (g2 in p.geoms and g1 in self.tool_geoms)):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            if f6[0] <= 0:
                continue
            q = R.T @ (c.pos - x0)
            if p.franka_pad:
                uv = np.array([q[0], q[2]])  # pad face: x along the handle, z along the finger
            else:
                if not self._inside(p, c.pos, spread):
                    continue
                uv = q[:2]
            dx = grid - np.clip(uv, [-hu, -hv], [hu, hv])
            wts = np.exp(-0.5 * np.sum(dx * dx, axis=1) / spread**2)
            out += f6[0] * wts / wts.sum()
        return out


class FrankaHandIO(HandIO):
    kind = "franka"

    def __init__(self, world):
        super().__init__(world)
        m = world.model
        g = world.cfg.gripper
        self.finger_qadr = np.array([m.joint(n).qposadr[0] for n in names.FINGER_JOINTS])
        self.grip_act = m.actuator(names.GRIP_MOTOR).id
        self.grip_force_max = g.grip_force_max
        self.grip_hold_max = 0.5 * g.grip_force_max  # Franka Hand: 70 N continuous, 140 N peak
        self.grip_pre_max = 120.0
        self._grids = {}
        pad_geoms = [m.geom(n).id for n in names.PAD_GEOMS]
        pad_bodies = [m.body(n).id for n in names.PAD_BODIES]
        self.patches = [Patch(tag, pad_bodies[k], -1, (g.pad_half[0], g.pad_half[2]), frozenset([pad_geoms[k]]),
                              franka_pad=True) for k, tag in enumerate("LR")]
        self.tool_geoms = frozenset([m.geom(names.HAMMER_HANDLE_GEOM).id])

    def init_pose(self) -> None:
        from tactile_sim.model.gripper import finger_q_touch
        from tactile_sim.model.tool_hammer import hammer_geometry

        w = self.world
        w.data.qpos[self.finger_qadr] = finger_q_touch(w.cfg.gripper, hammer_geometry(w.cfg.hammer).grip_half_width)

    def set_grip(self, f: float) -> None:
        """Squeeze force per pad (N); positive closes."""
        g = self.grip_force_max
        self.world.data.ctrl[self.grip_act] = -float(np.clip(f, -g, g))

    def grip_from_patches(self, forces: np.ndarray) -> float:
        return float(np.mean(forces))


class WujiHandIO(HandIO):
    kind = "wuji2"

    def __init__(self, world, info: dict):
        super().__init__(world)
        m = world.model
        g = world.cfg.gripper
        self.info = info
        self.grasp = info["grasp"]
        joints = info["joints"]
        self.joint_names = [j for j, _ in joints]
        self.tau_max = np.array([lim for _, lim in joints])
        self.qadr = np.array([m.joint(j).qposadr[0] for j in self.joint_names])
        self.dofs = np.array([m.joint(j).dofadr[0] for j in self.joint_names])
        self.jids = np.array([m.joint(j).id for j in self.joint_names])
        self.act = np.array([m.actuator(f"m_{j}").id for j in self.joint_names])
        self.synergy = WrapSynergy.for_wuji(joints, g.thumb_close, g.thumb_preshape)
        self._grids = {}
        self.patches = []
        for p in info["patches"]:
            b = m.body(p.body).id
            geoms = frozenset(gid for gid in range(m.ngeom) if m.geom_bodyid[gid] == b and m.geom_group[gid] == 3)
            self.patches.append(Patch(p.name, b, m.site(f"patch_{p.name}").id, p.half, geoms))
        # grip command (N of summed patch force) -> synergy scalar: the grasp keyframe's patch force at s = 1
        self.f_full = float(sum(self.grasp.contact_force.get(p.body, 0.0) for p in info["patches"]))
        self.grip_force_max = self.f_full
        self.grip_hold_max = 0.7 * self.f_full
        self.grip_pre_max = 0.85 * self.f_full
        self.s = 0.0
        self.self_locking = g.lock_mode == "self_locking"
        if g.lock_mode not in ("backdrivable", "self_locking"):
            raise ValueError(f"unknown lock_mode {g.lock_mode!r}")
        self._range0 = m.jnt_range[self.jids].copy()

    def init_pose(self) -> None:
        self.world.data.qpos[self.qadr] = self.grasp.q
        self.world.model.jnt_range[self.jids] = self._range0  # unlock

    def _lock(self, q: np.ndarray, tol: float = 1e-3) -> None:
        """Self-locking drives as a ratchet: each closing joint's opening-side limit follows it closed and never
        backs off; shaping joints are held where they are."""
        r = self.world.model.jnt_range
        d = self.synergy.close_dir
        lo, hi = r[self.jids, 0], r[self.jids, 1]
        lo = np.where(d > 0, np.maximum(lo, q - tol), lo)
        hi = np.where(d < 0, np.minimum(hi, q + tol), hi)
        shaping = d == 0
        lo = np.where(shaping, np.maximum(lo, np.minimum(q - tol, hi)), lo)
        hi = np.where(shaping, np.minimum(hi, np.maximum(q + tol, lo)), hi)
        r[self.jids, 0] = np.maximum(lo, self._range0[:, 0])
        r[self.jids, 1] = np.minimum(hi, self._range0[:, 1])

    def set_grip(self, f: float) -> None:
        self.s = float(np.clip(f / self.f_full, 0.0, 1.0))

    def tick(self, t: float) -> None:
        d = self.world.data
        q = d.qpos[self.qadr]
        d.ctrl[self.act] = self.synergy.torque(self.s, q, d.qvel[self.dofs])
        if self.self_locking and self.locked:
            self._lock(q)

    locked = False  # set once the grasp has settled (World.reset)

    def lock_engaged(self) -> None:
        self.locked = True
        self._q_locked = self.world.data.qpos[self.qadr].copy()

    def closed_further(self) -> float:
        """Mean angle the closing joints have advanced since the grasp seated (rad). A tool can only leave the
        wrap if the fingers close into the space it vacates (backdrivable: the synergy drives them shut;
        self-locking: the ratchet lets them close but never open)."""
        if getattr(self, "_q_locked", None) is None:
            return 0.0
        d = self.synergy.close_dir
        sel = d != 0
        dq = (self.world.data.qpos[self.qadr] - self._q_locked) * d
        return float(np.mean(dq[sel]))

    def grip_from_patches(self, forces: np.ndarray) -> float:
        return float(np.sum(forces))


def make_hand_io(world) -> HandIO:
    if world.cfg.gripper.hand == "franka":
        return FrankaHandIO(world)
    return WujiHandIO(world, world.spec.hand_info)
