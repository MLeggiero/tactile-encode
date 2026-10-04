"""Hand runtime interfaces: what the World, grip loop, sensors and episode need from a hand.

- `FrankaHandIO`: two parallel fingers on one force-controlled drive; a taxel patch on each pad (L, R).
  Grip force = mean normal force per pad.
- `WujiHandIO`: WUJI Hand 2 in a power wrap; flat taxel patches where the grasp loads the hand (palm, thumb), or
  the TaxelScan skins: taxels on the curved palmar surface of the palm and of each finger's middle and distal
  segment. Grip force = summed normal force over the patches. The grip command sets the wrap synergy's scalar,
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
    grid: tuple[int, int] | None = None  # rows, columns; None = cfg.sensors.taxel_grid
    taxel_pos: np.ndarray | None = None  # conforming skin: (n, 3) taxel centres in the body frame
    taxel_normal: np.ndarray | None = None  # conforming skin: (n, 3) outward unit normals in the body frame
    taxel_area: float = 0.0  # one taxel's area (m^2)
    accel: bool = True  # carries a patch accelerometer
    reach: float = 0.0  # conforming skin: a contact farther than this from every taxel does not load it (m)


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
        return [f"pad_acc_{p.name}" for p in self.patches if p.accel]

    def grid(self, k: int) -> tuple[int, int]:
        p = self.patches[k]
        return tuple(p.grid) if p.grid is not None else tuple(self.world.cfg.sensors.taxel_grid)

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
        if self._skin_by_geom:
            for k, t in enumerate(self.skin_taxels()):
                if t is not None:
                    out[k] = t.sum()
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.tool_geoms:
                g1, g2 = g2, g1
            elif g2 not in self.tool_geoms:
                continue
            for k, p in enumerate(self.patches):
                if p.taxel_pos is not None or g1 not in p.geoms or not self._inside(p, c.pos):
                    continue
                mujoco.mj_contactForce(m, d, i, f6)
                out[k] += max(f6[0], 0.0)
        return out

    def _inside(self, p: Patch, pos: np.ndarray, spread: float = 0.003) -> bool:
        if p.franka_pad:
            return True
        if p.taxel_pos is not None:
            d = self.world.data
            q = d.xmat[p.body].reshape(3, 3).T @ (pos - d.xpos[p.body])
            return float(np.min(np.sum((p.taxel_pos - q) ** 2, axis=1))) <= p.reach**2
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
        if p.taxel_pos is not None:
            return self.skin_taxels()[k]
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


    def skin_taxels(self) -> list[np.ndarray]:
        """Normal force per taxel on every conforming patch, computed once per physics step.

        MuJoCo reduces a tool resting on a link to a few contact points; a skin under, say, a 28 mm handle is
        loaded along a band. The force is therefore distributed as on an elastic foundation: from each taxel near
        the contacts a ray along its normal finds the gap to the touching tool geoms, and a taxel carries load in
        proportion to how much further than the closest taxel's the tool presses into it, up to the skin's
        compressible depth (ts_skin_depth). That shape is refreshed every ts_shape_dt and scaled every physics
        step so the patch's taxels share exactly the contacts' total normal force. Where no ray finds the tool,
        each contact is spread with Gaussian weights (sigma ts_spread) about its point instead. A contact farther
        than `reach` from every taxel (on a link's side or back, past the skin's edge) loads none."""
        w = self.world
        key = (w.step_count, w.data.time)
        if self._skin_key == key:
            return self._skin
        m, d = w.model, w.data
        sc = w.cfg.sensors
        sig = sc.ts_spread
        out = [np.zeros(len(p.taxel_pos)) if p.taxel_pos is not None else None for p in self.patches]
        gauss = [None if o is None else o.copy() for o in out]
        load: dict[int, list] = {}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.efc_address < 0:
                continue
            g1, g2 = c.geom1, c.geom2
            if g1 in self.tool_geoms:
                g1, g2 = g2, g1
            elif g2 not in self.tool_geoms:
                continue
            ks = self._skin_by_geom.get(g1)
            if not ks:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            if f6[0] <= 0:
                continue
            for k in ks:
                p = self.patches[k]
                q = d.xmat[p.body].reshape(3, 3).T @ (c.pos - d.xpos[p.body])
                r2 = np.sum((p.taxel_pos - q) ** 2, axis=1)
                if r2.min() > p.reach**2:
                    continue
                wts = np.exp(-0.5 * (r2 - r2.min()) / sig**2)
                gauss[k] += f6[0] * wts / wts.sum()
                load.setdefault(k, []).append((f6[0], q, g2))
        for k, cs in load.items():
            total = sum(f for f, _, _ in cs)
            geoms = frozenset(g for _, _, g in cs)
            t_s, g_s, shape = self._shape.get(k, (-np.inf, None, None))
            if d.time - t_s >= sc.ts_shape_dt - 1e-12 or g_s != geoms:
                shape = self._foundation(k, [q for _, q, _ in cs], geoms | self._pair_tool.get(k, frozenset()))
                self._shape[k] = (d.time, geoms, shape)
            out[k] = total * shape if shape is not None else gauss[k]
        for k in self._shape.keys() - load.keys():
            del self._shape[k]
        self._skin, self._skin_key = out, key
        return out

    def _foundation(self, k: int, contacts: list[np.ndarray], tool: frozenset[int]) -> np.ndarray | None:
        """Elastic-foundation load shape on patch k (sums to 1), or None if no taxel ray meets the tool."""
        w = self.world
        m, d = w.model, w.data
        sc = w.cfg.sensors
        p = self.patches[k]
        cq = np.array(contacts)
        near = np.min(np.sum((p.taxel_pos[:, None, :] - cq[None]) ** 2, axis=2), axis=1) <= sc.ts_shape_radius**2
        idx = np.flatnonzero(near)
        if idx.size == 0:
            return None
        R, x0 = d.xmat[p.body].reshape(3, 3), d.xpos[p.body]
        pw = p.taxel_pos[idx] @ R.T + x0
        nw = p.taxel_normal[idx] @ R.T
        back = 0.003  # start inside the skin so a tool already pressing in is still found
        gap = np.full(idx.size, np.inf)
        for g in tool:
            mesh = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            for j in range(idx.size):
                o = pw[j] - back * nw[j]
                if mesh:
                    h = mujoco.mj_rayMesh(m, d, g, o, nw[j])
                else:
                    h = mujoco.mju_rayGeom(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g], o, nw[j],
                                           int(m.geom_type[g]))
                if h >= 0:
                    gap[j] = min(gap[j], h - back)
        if not np.isfinite(gap).any():
            return None
        wts = np.clip(sc.ts_skin_depth - (gap - gap[np.isfinite(gap)].min()), 0.0, None)
        wts[~np.isfinite(gap)] = 0.0
        shape = np.zeros(len(p.taxel_pos))
        shape[idx] = wts
        return shape / shape.sum()

    _skin_key = None
    _skin_by_geom: dict = {}


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
        self._skin_by_geom = {}
        self._shape = {}  # patch -> (time, tool geoms, load shape) of the elastic-foundation model
        self._pair_tool = {}  # patch -> tool geoms it has contact pairs with (all are tested, not just touching)
        for p in info["patches"]:
            b = m.body(p.body).id
            bids = {m.body(n).id for n in getattr(p, "bodies", (p.body,))}
            geoms = frozenset(gid for gid in range(m.ngeom) if m.geom_bodyid[gid] in bids and m.geom_group[gid] == 3)
            site = m.site(f"patch_{p.name}").id
            if hasattr(p, "pos"):  # TaxelScan skin
                k = len(self.patches)
                self.patches.append(Patch(p.name, b, site, p.half, geoms, grid=tuple(p.grid),
                                          taxel_pos=np.asarray(p.pos), taxel_normal=np.asarray(p.normal),
                                          taxel_area=p.pitch[0] * p.pitch[1],
                                          accel=p.accel, reach=max(p.pitch) + 0.002))
                # reach: a pitch plus the hull-to-mesh offset
                for gid in geoms:
                    self._skin_by_geom.setdefault(gid, []).append(k)
                pg = {int(b) if int(a) in geoms else int(a) for a, b in zip(m.pair_geom1, m.pair_geom2, strict=True)
                      if (int(a) in geoms) != (int(b) in geoms) and (int(a) in self.tool_geoms or
                                                                    int(b) in self.tool_geoms)}
                self._pair_tool[k] = frozenset(pg)
            else:
                self.patches.append(Patch(p.name, b, site, p.half, geoms))
        # grip command (N of summed patch force) -> synergy scalar: the grasp keyframe's patch force at s = 1
        self.f_full = float(sum(self.grasp.contact_force.get(b, 0.0) for p in info["patches"]
                                for b in getattr(p, "bodies", (p.body,))))
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
