"""Task A plant: a board and a pre-started nail.

The nail is a body on a slide joint pointing into the board. Penetration resistance is Coulomb
friction on that joint (`frictionloss`), which MuJoCo solves as a constraint: the nail moves only
while the driving force exceeds the resistance, and no energy is stored (a spring would push the
nail back out). Resistance grows linearly with depth; viscous damping models rate-dependent
crushing of the wood. MuJoCo has no restitution coefficient, so velocity-dependent restitution is
emulated by setting the hammer/nail pair's damping from the approach speed just before contact and
freezing it for the duration of the pulse. MuJoCo's positive solref couples damping ratio and
stiffness, so the pair is switched to the direct form solref = (-k, -b): k stays fixed at the value
implied by the configured (timeconst, dampratio) and only b = 2 * zeta(v) * sqrt(k) changes.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.model.plants.base import Plant
from tactile_sim.model.xmlutil import sub

PAIR_FACE_NAIL = "pair_face_nail"
PAIR_HEAD_BOARD = "pair_head_board"


class NailPlant(Plant):
    def add_mjcf(self, root: ET.Element, worldbody: ET.Element, target: np.ndarray, axis: np.ndarray) -> None:
        """`target`: struck surface of the nail head at the start; `axis`: unit strike direction."""
        p, h = self.cfg.plant, self.cfg.hammer
        axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, axis)  # body z = strike axis (into the board)
        surface = np.asarray(target, dtype=float) + axis * (p.proud + 2 * p.head_half_h)
        bx, bt, bh = p.board_half
        board = sub(worldbody, "body", name=names.BOARD_BODY, pos=surface + axis * bt, quat=quat)
        sub(board, "geom", name=names.BOARD_GEOM, type="box", size=(bx, bh, bt), rgba=(0.76, 0.6, 0.42, 1),
            contype=0, conaffinity=0)
        nail = sub(worldbody, "body", name=names.NAIL_BODY, pos=surface, quat=quat)
        sub(nail, "joint", name=names.NAIL_JOINT, type="slide", axis=(0, 0, 1),
            range=(0, p.proud - 0.001), limited="true", frictionloss=p.resistance_0, damping=p.damping,
            solreflimit=p.limit_solref, solreffriction=p.friction_solref, armature=1e-4)
        zc = -(p.proud + p.head_half_h)
        sub(nail, "geom", name=names.NAIL_HEAD_GEOM, type="cylinder", size=(p.head_radius, p.head_half_h),
            pos=(0, 0, zc), mass=p.nail_mass, rgba=(0.7, 0.7, 0.75, 1), contype=0, conaffinity=0)
        sub(nail, "geom", type="capsule", fromto=(0, 0, zc, 0, 0, zc + 0.06), size=p.shank_radius,
            rgba=(0.7, 0.7, 0.75, 1), contype=0, conaffinity=0, mass=0, group=2)
        sub(nail, "site", name=names.NAIL_HEAD_SITE, pos=(0, 0, zc - p.head_half_h), size=0.002, group=4)

        contact = root.find("contact")
        if contact is None:
            contact = sub(root, "contact")
        sub(contact, "pair", name=PAIR_FACE_NAIL, geom1=names.HAMMER_FACE_GEOM, geom2=names.NAIL_HEAD_GEOM,
            condim=3, friction=(0.3, 0.3, 0.005, 0.0001, 0.0001), solref=h.face_solref, solimp=h.face_solimp,
            margin=0.0)
        sub(contact, "pair", name=PAIR_HEAD_BOARD, geom1=names.HAMMER_HEAD_GEOM, geom2=names.BOARD_GEOM,
            condim=3, friction=(0.5, 0.5, 0.005, 0.0001, 0.0001), solref=h.board_solref, margin=0.0)
        self.axis = axis

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.m, self.d = model, data
        j = model.joint(names.NAIL_JOINT)
        self.qadr = int(j.qposadr[0])
        self.dadr = int(j.dofadr[0])
        self.pair_id = model.pair(PAIR_FACE_NAIL).id
        self.face_site = model.site(names.HAMMER_FACE_SITE).id
        self.head_site = model.site(names.NAIL_HEAD_SITE).id
        self.face_geom = model.geom(names.HAMMER_FACE_GEOM).id
        self.nail_geom = model.geom(names.NAIL_HEAD_GEOM).id
        self.hammer_body = model.body(names.HAMMER_BODY).id
        self.resistance_0 = float(self.cfg.plant.resistance_0)
        self._vel = np.zeros(6)
        self._in_contact = False
        self._last_contact_t = -np.inf
        tc, zeta = self.cfg.hammer.face_solref
        dmax = float(model.pair_solimp[self.pair_id, 1])
        self.k_face = 1.0 / (dmax**2 * tc**2 * zeta**2)
        self.last_zeta = float(zeta)
        self.contact_zeta = float("nan")  # damping ratio frozen at the last contact onset
        if self.cfg.plant.vdr_enabled:
            self.set_face_damping_ratio(self.cfg.plant.vdr_zeta0)

    # ---- state ----
    @property
    def depth(self) -> float:
        """Nail depth driven since the start of the episode (m)."""
        return float(self.d.qpos[self.qadr])

    def resistance(self, depth: float | None = None) -> float:
        dep = self.depth if depth is None else depth
        return self.resistance_0 + self.cfg.plant.resistance_per_m * max(dep, 0.0)

    def gap(self) -> float:
        """Distance from the hammer face to the nail head along the strike axis (positive = short of it)."""
        return float((self.d.site_xpos[self.head_site] - self.d.site_xpos[self.face_site]) @ self.axis)

    def face_velocity(self) -> np.ndarray:
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, self.face_site, self._vel, 0)
        return self._vel[3:].copy()

    def contact_active(self) -> bool:
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if {c.geom1, c.geom2} == {self.face_geom, self.nail_geom} and c.dist < 0:
                return True
        return False

    def set_face_damping_ratio(self, zeta: float) -> None:
        """Direct-form contact: fixed stiffness, damping ratio `zeta`."""
        self.m.pair_solref[self.pair_id, 0] = -self.k_face
        self.m.pair_solref[self.pair_id, 1] = -2.0 * zeta * np.sqrt(self.k_face)
        self.last_zeta = float(zeta)

    # ---- hooks ----
    def reset(self, rng=None) -> None:
        self.d.qpos[self.qadr] = 0.0
        self.d.qvel[self.dadr] = 0.0
        self._in_contact = False
        self._last_contact_t = -np.inf

    def pre_step(self) -> None:
        m = self.m
        m.dof_frictionloss[self.dadr] = self.resistance()
        p = self.cfg.plant
        if not p.vdr_enabled:
            return
        in_contact = self.contact_active()
        if not in_contact and self.gap() < 0.01:
            v_approach = max(0.0, float(self.face_velocity() @ self.axis))
            self.set_face_damping_ratio(min(p.vdr_zeta_max, p.vdr_zeta0 + p.vdr_zeta1 * v_approach))
        t = float(self.d.time)
        if in_contact:
            if not self._in_contact and t - self._last_contact_t > 0.005:  # new blow, not chatter
                self.contact_zeta = self.last_zeta
            self._last_contact_t = t
        self._in_contact = in_contact

    def truth(self) -> dict[str, float]:
        return {"nail_depth": self.depth, "nail_resistance": self.resistance(), "face_gap": self.gap()}

    def done(self) -> bool:
        return self.depth >= self.cfg.plant.drive_target
