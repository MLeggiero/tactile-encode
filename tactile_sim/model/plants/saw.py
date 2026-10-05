"""Saw plant: a board lying on a table, crosscut through its thickness by a hand saw.

MuJoCo cannot remove material, so the cut is a state, the kerf depth c, and the blade's interaction with the wood is
a set of forces applied to the saw at the two ends of the stretch of teeth over the board:

- support: the kerf bottom (the board surface before the cut starts) pushes back with k_normal * penetration plus
  damping, never pulling;
- cutting: a tangential force mu_cut * F_n (plus a small drag) opposing the stroke;
- removal: on the cutting stroke (push for a western saw) the kerf deepens at F_n * |stroke speed| / k_cut, i.e. a
  fixed depth per metre of stroke per newton of push; a knot multiplies k_cut over a depth range;
- kerf walls: once the kerf is started, a blade point below the surface that strays sideways more than the side
  clearance is pushed back (k_lateral), and the wall force adds binding friction mu_bind * |F_lateral|.

The loads at the grip are periodic (they reverse with the stroke), with binding as the untimed event.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.model.plants.base import Plant, add_table, apply_point_forces, body_point_velocity
from tactile_sim.model.xmlutil import sub

SAW_BOARD = "saw_board"
KERF_GEOM = "saw_kerf"
V_EPS = 0.02  # m/s: tanh regularisation of the stroke-direction friction


def _frame_quat(u: np.ndarray, n: np.ndarray, f: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([u, n, f]).reshape(-1))
    return q


class SawPlant(Plant):
    def add_mjcf(self, root: ET.Element, worldbody: ET.Element, target: np.ndarray, axis: np.ndarray) -> None:
        """`target`: the middle of the teeth at the hover pose; `axis`: the cut direction (into the board)."""
        from tactile_sim.model.builder import tcp_rotation

        s = self.cfg.saw
        R = tcp_rotation(self.cfg)
        self.u = R[:, 0]  # stroke (along the blade)
        self.f = np.asarray(axis, float) / np.linalg.norm(axis)  # into the cut (tool y)
        self.n = np.cross(self.f, self.u)  # across the blade (kerf normal), completes (u, n, f) right-handed
        self.surface = np.asarray(target, float) + self.f * s.clearance  # board top, under the middle of the teeth
        T = s.board_thickness
        center = self.surface + self.f * 0.5 * T
        board = sub(worldbody, "body", name=SAW_BOARD, pos=center, quat=_frame_quat(self.u, self.n, self.f))
        sub(board, "geom", name=names.BOARD_GEOM, type="box", size=(0.5 * s.board_width, 0.5 * s.board_length, 0.5 * T),
            rgba=(0.80, 0.66, 0.46, 1), contype=0, conaffinity=0)
        sub(board, "geom", name=KERF_GEOM, type="box", size=(0.5 * s.board_width + 0.0005, 0.5 * s.kerf_width, 1e-5),
            pos=(0, 0, -0.5 * T), rgba=(0.25, 0.18, 0.10, 1), contype=0, conaffinity=0)
        if self.cfg.scene.table and self.f[2] < -np.cos(np.radians(30)):
            under = self.surface + self.f * T
            add_table(worldbody, self.cfg, under, float(under[2]))

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        from tactile_sim.model.tools import SAW_TEETH_A, SAW_TEETH_B

        self.m, self.d = model, data
        self.tool = model.body(names.HAMMER_BODY).id
        self.site_a = model.site(SAW_TEETH_A).id
        self.site_b = model.site(SAW_TEETH_B).id
        self.kerf_geom = model.geom(KERF_GEOM).id
        self._buf = np.zeros(6)
        self.reset()

    # ---- model ----
    def removal_resistance(self, depth: float) -> float:
        """k_cut at this depth (a knot makes the wood harder to remove over its depth range)."""
        s = self.cfg.saw
        lo, hi, mult = s.knot
        return s.k_cut * (mult if lo <= depth < hi else 1.0)

    def cutting_stroke(self, v_stroke: float) -> bool:
        return v_stroke > 0 if self.cfg.saw.cut_on == "push" else v_stroke < 0

    def _engaged_points(self) -> list[np.ndarray]:
        """Ends of the stretch of teeth over the board (world), or [] when the teeth are off the board."""
        s = self.cfg.saw
        pa, pb = self.d.site_xpos[self.site_a].copy(), self.d.site_xpos[self.site_b].copy()
        ua, ub = (pa - self.surface) @ self.u, (pb - self.surface) @ self.u
        half = 0.5 * s.board_width
        if abs(ub - ua) < 1e-9:
            return [pa, pb] if abs(ua) <= half else []
        lam = sorted([(-half - ua) / (ub - ua), (half - ua) / (ub - ua)])
        l0, l1 = max(lam[0], 0.0), min(lam[1], 1.0)
        if l1 <= l0:
            return []
        return [pa + l0 * (pb - pa), pa + l1 * (pb - pa)]

    # ---- hooks ----
    def reset(self, rng=None) -> None:
        self.depth = 0.0
        self.kerf_n: float | None = None
        self.F_normal = self.F_cut = self.F_lateral = self.v_stroke = 0.0
        self.binding = False
        self.strokes = 0
        self._last_sign = 0
        self._update_kerf_visual()

    def _update_kerf_visual(self) -> None:
        if not hasattr(self, "m"):
            return
        s = self.cfg.saw
        T = s.board_thickness
        c = max(self.depth, 1e-5)
        self.m.geom_size[self.kerf_geom, 2] = 0.5 * c
        self.m.geom_pos[self.kerf_geom, 2] = -0.5 * T + 0.5 * c - 1e-4
        if self.kerf_n is not None:
            self.m.geom_pos[self.kerf_geom, 1] = self.kerf_n

    def pre_step(self) -> None:
        s, d, m = self.cfg.saw, self.d, self.m
        pts = self._engaged_points()
        forces = []
        Fn_tot = Ft_tot = Fl_tot = 0.0
        v_s_mean = 0.0
        gap = 0.5 * (s.kerf_width - s.blade_thickness)
        for p in pts:
            v = body_point_velocity(m, d, self.tool, p, self._buf)
            v_s = float(v @ self.u)
            v_s_mean += v_s / len(pts)
            e = float((p - self.surface) @ self.f)  # depth of this tooth point below the board surface
            pen = e - self.depth
            F = np.zeros(3)
            Fn = 0.0
            if pen > 0:
                Fn = max(0.0, 0.5 * (s.k_normal * pen + s.d_normal * float(v @ self.f)))
                F -= Fn * self.f
            Fl = 0.0
            if self.kerf_n is not None and e > 0:
                delta = float((p - self.surface) @ self.n) - self.kerf_n
                excess = np.sign(delta) * max(abs(delta) - gap, 0.0)
                w = 0.5 * min(e / 0.01, 1.0)  # wall engagement grows with the blade's depth in the kerf
                if excess != 0.0:
                    Fl = -w * (s.k_lateral * excess + s.d_lateral * float(v @ self.n))
                    F += Fl * self.n
            elif Fn > 0:  # teeth skating on the surface before the kerf is started
                F -= 0.3 * Fn * np.tanh(float(v @ self.n) / V_EPS) * self.n
            Ft = s.mu_cut * Fn + (0.5 * s.drag if pen > 0 else 0.0) + s.mu_bind * abs(Fl)
            F -= Ft * np.tanh(v_s / V_EPS) * self.u
            forces.append((p, F))
            Fn_tot += Fn
            Ft_tot += Ft
            Fl_tot += abs(Fl)
            if Fn > 0 and self.cutting_stroke(v_s):
                self.depth += Fn * abs(v_s) / self.removal_resistance(self.depth) * m.opt.timestep
        self.depth = min(self.depth, s.board_thickness)
        if self.kerf_n is None and self.depth > 0.0005 and pts:
            self.kerf_n = float(np.mean([(p - self.surface) @ self.n for p in pts]))
        if pts:
            sign = int(np.sign(v_s_mean)) if abs(v_s_mean) > 0.05 else 0
            if sign and sign != self._last_sign:
                self.strokes += 1
                self._last_sign = sign
        apply_point_forces(d, self.tool, forces)
        self.F_normal, self.F_cut, self.F_lateral, self.v_stroke = Fn_tot, Ft_tot, Fl_tot, v_s_mean
        self.binding = Fl_tot > 2.0
        self._update_kerf_visual()

    def truth(self) -> dict[str, float]:
        return {"cut_depth": self.depth, "saw_normal_force": self.F_normal, "saw_cut_force": self.F_cut,
                "saw_lateral_force": self.F_lateral, "stroke_speed": self.v_stroke, "binding": float(self.binding),
                "strokes": float(self.strokes)}

    def done(self) -> bool:
        return self.depth >= self.cfg.saw.cut_target

    @property
    def progress(self) -> float:
        return self.depth
