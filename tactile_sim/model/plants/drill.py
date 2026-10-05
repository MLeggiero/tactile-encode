"""Drill plant: an inline cordless driver over a board on a table, driving a pre-started wood screw ("screw") or
drilling a hole through ("hole").

Phenomenological, like the nail: the bit's interaction with the work is applied to the tool as forces at the bit
tip and a torque about the bit, from a small state model.

- Motor: tau_m = stall * (u - w / w_free) on a rotor of inertia J, u following the trigger with a soft start, an
  electronic brake of at most brake_torque on release; the housing carries the reaction to the output torque plus
  the rotor's acceleration (the startup kick).
- Contact: the tip is supported axially by the screw head (or the hole bottom) with k_axial; in the screw recess, or
  in a started hole, a lateral spring keeps the tip on the axis.
- Screw: engaged when the tip is pressed into the recess within engage_radius of the axis. The screw turns with the
  spindle and advances pitch per turn against a torque that grows with depth and rises steeply once the head
  seats. The clutch slips above clutch_torque: the screw stops and the clutch ratchets (torque ripple at
  clutch_detents per spindle-to-motor revolution). Cam-out: when the output torque exceeds cam_ratio * axial push
  (less with bit misalignment) the bit rides out of the recess for cam_time, kicked back by cam_kick, with ripple;
  after strip_after cam-outs the recess is stripped. States: free, engaged, cam_out, seated, stripped.
- Hole: thrust above thrust_f0 sets the feed per revolution; torque grows with the feed; over the last exit_len
  the support fades and the bit grabs (catch_gain), pulling itself through at catch_feed per turn or more; once
  through, the tip loses its support and the tool lunges.
  States: free, drilling, catch, through.

Loads at the grip: a steady reaction torque with untimed jerks (cam-out, clutch slip, catch, breakthrough).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.model.plants.base import Plant, add_table, apply_point_forces, body_point_velocity
from tactile_sim.model.xmlutil import sub

DRILL_BOARD = "drill_board"
SCREW_BODY = "screw"
HOLE_GEOM = "drill_hole"
STATES = ("free", "engaged", "cam_out", "seated", "stripped", "drilling", "catch", "through")
HEAD_RADIUS = 0.004


class DrillPlant(Plant):
    def add_mjcf(self, root: ET.Element, worldbody: ET.Element, target: np.ndarray, axis: np.ndarray) -> None:
        """`target`: the screw head (screw mode) or the board surface (hole mode) on the bit axis; `axis`: the bit
        direction (into the board)."""
        dc = self.cfg.drill
        if dc.mode not in ("screw", "hole"):
            raise ValueError(f"unknown drill mode {dc.mode!r}")
        self.a = np.asarray(axis, float) / np.linalg.norm(axis)
        self.head0 = np.asarray(target, float)
        self.surface = self.head0 + (self.a * dc.screw_len if dc.mode == "screw" else 0.0)
        q = np.zeros(4)
        mujoco.mju_quatZ2Vec(q, self.a)
        T = dc.board_thickness
        if self.cfg.scene.table and self.a[2] < -np.cos(np.radians(30)):
            under = self.surface + self.a * T
            add_table(worldbody, self.cfg, under, float(under[2]))
        board = sub(worldbody, "body", name=DRILL_BOARD, pos=self.surface + self.a * 0.5 * T, quat=q)
        sub(board, "geom", name=names.BOARD_GEOM, type="box", size=(dc.board_half[0], dc.board_half[1], 0.5 * T),
            rgba=(0.80, 0.66, 0.46, 1), contype=0, conaffinity=0)
        sub(board, "geom", name=HOLE_GEOM, type="cylinder", size=(0.0035, 1e-5), pos=(0, 0, -0.5 * T),
            rgba=(0.25, 0.18, 0.10, 1), contype=0, conaffinity=0)
        if dc.mode == "screw":
            sb = sub(worldbody, "body", name=SCREW_BODY, mocap="true", pos=self.head0, quat=q)
            sub(sb, "geom", type="cylinder", size=(HEAD_RADIUS, 0.0012), pos=(0, 0, 0.0012), rgba=(0.75, 0.70, 0.45, 1),
                contype=0, conaffinity=0)
            sub(sb, "geom", type="box", size=(0.0028, 0.0005, 0.0003), pos=(0, 0, -0.0001), rgba=(0.2, 0.2, 0.2, 1),
                contype=0, conaffinity=0)
            sub(sb, "geom", type="capsule", fromto=(0, 0, 0.002, 0, 0, 0.032), size=0.0018, rgba=(0.75, 0.70, 0.45, 1),
                contype=0, conaffinity=0)

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        from tactile_sim.model.tools import BIT_TIP_SITE

        self.m, self.d = model, data
        self.tool = model.body(names.HAMMER_BODY).id
        self.tip_site = model.site(BIT_TIP_SITE).id
        self.hole_geom = model.geom(HOLE_GEOM).id
        self.screw_mocap = (int(model.body(SCREW_BODY).mocapid[0]) if self.cfg.drill.mode == "screw" else -1)
        self._buf = np.zeros(6)
        self.trigger = 0.0
        self.reset()

    # ---- model ----
    def screw_torque(self, depth: float) -> float:
        """Torque to keep turning the screw at this depth (rises steeply once the head is flush)."""
        dc = self.cfg.drill
        tau = dc.screw_torque0 + dc.screw_torque_per_m * depth
        if depth > dc.screw_len:
            tau += dc.seat_stiffness * (depth - dc.screw_len)
        return tau

    def cam_limit(self, F_axial: float, misalign: float) -> float:
        """Largest torque the recess transmits before the bit cams out."""
        dc = self.cfg.drill
        return dc.cam_ratio * max(F_axial, 0.0) * max(0.0, 1.0 - misalign / dc.cam_max_angle)

    def feed_per_rev(self, F_axial: float) -> float:
        dc = self.cfg.drill
        return max(F_axial - dc.thrust_f0, 0.0) / dc.feed_stiffness

    def motor_torque(self, trigger: float, w: float) -> float:
        """Drive torque at this trigger and speed; with the trigger released, the electronic brake."""
        dc = self.cfg.drill
        w_free = dc.free_speed if dc.mode == "screw" else dc.hole_free_speed
        if trigger <= 0.0:
            return -float(np.clip(dc.stall_torque * w / w_free, -dc.brake_torque, dc.brake_torque))
        return dc.stall_torque * (trigger - w / w_free)

    # ---- hooks ----
    def reset(self, rng=None) -> None:
        self.depth = 0.0  # screw advance (screw) or hole depth (hole)
        self.w = 0.0
        self.theta = 0.0
        self.state = "free"
        self.cam_outs = 0
        self._cam_until = -np.inf
        self.tau_out = self.F_axial = self.misalign = 0.0
        self.trigger = 0.0
        self.u = 0.0
        self.clutch_slipping = False
        self.completed = False  # latched once the screw seats / the bit goes through
        self._update_visual()

    def _update_visual(self) -> None:
        if not hasattr(self, "m"):
            return
        dc = self.cfg.drill
        if dc.mode == "screw":
            q = np.zeros(4)
            mujoco.mju_quatZ2Vec(q, self.a)
            spin = np.array([np.cos(0.5 * self.theta), 0.0, 0.0, np.sin(0.5 * self.theta)])
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, q, spin)
            self.d.mocap_pos[self.screw_mocap] = self.head0 + self.a * self.depth
            self.d.mocap_quat[self.screw_mocap] = out
        else:
            h = max(self.depth, 1e-5)
            T = dc.board_thickness
            self.m.geom_size[self.hole_geom, 1] = 0.5 * h
            self.m.geom_pos[self.hole_geom, 2] = -0.5 * T + 0.5 * h - 1e-4

    def pre_step(self) -> None:
        dc, d, m = self.cfg.drill, self.d, self.m
        dt = m.opt.timestep
        t = float(d.time)
        tip = d.site_xpos[self.tip_site].copy()
        a_t = d.xmat[self.tool].reshape(3, 3)[:, 2]  # bit axis (tool z)
        self.misalign = float(np.arccos(np.clip(a_t @ self.a, -1.0, 1.0)))
        v = body_point_velocity(m, d, self.tool, tip, self._buf)
        screw = dc.mode == "screw"
        seat = (self.head0 + self.a * self.depth) if screw else (self.surface + self.a * self.depth)
        rel = tip - seat
        ax = float(rel @ self.a)
        lat = rel - ax * self.a
        r_lat = float(np.linalg.norm(lat))
        v_ax = float(v @ self.a)
        v_lat = v - v_ax * self.a
        T = dc.board_thickness

        # axial support at the tip
        k_ax = dc.k_axial
        if not screw:
            k_ax *= float(np.clip((T - self.depth) / dc.exit_len, 0.0, 1.0)) if self.depth > T - dc.exit_len else 1.0
            if self.state == "through":
                k_ax = 0.0
        on_head = (r_lat < HEAD_RADIUS) if screw else True
        d_ax = dc.d_axial * k_ax / dc.k_axial  # the support's damping fades with it
        F_a = max(0.0, k_ax * ax + d_ax * v_ax) if (ax > 0 and on_head and k_ax > 0) else 0.0
        if screw and self.state == "stripped" and ax > 0:
            F_a = max(0.0, dc.k_axial * ax + dc.d_axial * v_ax)  # pressing on a stripped head still bears load
        F_tip = -F_a * self.a
        # lateral: recess or hole wall keeps the tip on the axis
        in_recess = screw and ax > -0.0005 and r_lat < 2 * dc.engage_radius and self.state != "stripped"
        in_hole = (not screw) and self.depth > 0.002 and ax > -0.002 and self.state != "through"
        if in_recess or in_hole:
            F_tip += -dc.k_lateral * lat - dc.d_lateral * v_lat
        elif F_a > 0:
            F_tip += -0.3 * F_a * np.tanh(np.linalg.norm(v_lat) / 0.01) * (v_lat / max(np.linalg.norm(v_lat), 1e-9))

        # spindle and load; the trigger soft-starts (the speed command slews)
        self.u = self.trigger if self.trigger <= 0 else min(self.trigger, self.u + dc.trigger_slew * dt)
        tau_m = self.motor_torque(self.u, self.w)
        tau_load = dc.spindle_drag * self.w
        ripple = 0.0
        advance = False
        self.clutch_slipping = False
        if screw:
            if self.state == "cam_out" and t >= self._cam_until:
                self.state = "stripped" if self.cam_outs >= dc.strip_after else "free"
            if self.state in ("free", "engaged", "seated"):
                engaged = F_a > 2.0 and r_lat < dc.engage_radius
                self.state = ("seated" if self.state == "seated" else "engaged") if engaged else "free"
            if self.state in ("engaged", "seated"):
                need = self.screw_torque(self.depth)
                if need >= dc.clutch_torque and tau_m > 0:
                    # the screw holds; the clutch ratchets and the motor spins through it
                    self.clutch_slipping = True
                    tau_load += dc.clutch_torque * (0.6 + 0.4 * np.sin(dc.clutch_detents * self.theta))
                    ripple = 0.4 * dc.clutch_torque * np.sin(dc.clutch_detents * self.theta)
                    if self.depth >= dc.screw_len:
                        self.state = "seated"
                else:
                    tau_load += need
                    advance = True
                if tau_load - dc.spindle_drag * self.w > self.cam_limit(F_a, self.misalign) and self.trigger > 0:
                    self.state = "cam_out"
                    self.cam_outs += 1
                    self._cam_until = t + dc.cam_time
                    advance = False
            if self.state == "cam_out":
                tau_load = dc.spindle_drag * self.w + 0.05
                ripple = 0.3 * self.screw_torque(self.depth) * np.sin(4 * self.theta)
                F_tip -= dc.cam_kick * self.a
        else:
            if self.state != "through":
                if F_a > 0 and self.w > 0:
                    catch = self.depth > T - dc.exit_len
                    f_rev = self.feed_per_rev(F_a)
                    if catch:  # the flutes grab and pull the bit through whatever the thrust
                        f_rev = max(f_rev, dc.catch_feed)
                    gain = dc.catch_gain if catch else 1.0
                    tau_load += (dc.drill_torque0 + dc.drill_torque_per_feed * f_rev) * gain
                    self.state = "catch" if catch else "drilling"
                    self.depth += f_rev * self.w / (2 * np.pi) * dt
                    if self.depth >= T:
                        self.depth = T
                        self.state = "through"
                else:
                    self.state = "free" if self.state != "through" else "through"
        # rotor (the spindle, geared): J dw = tau_m - tau_load; a stalled screw clamps the spindle at zero speed
        w_new = self.w + (tau_m - tau_load) / dc.rotor_inertia * dt
        if advance and w_new < 0:
            w_new = 0.0
        dw = w_new - self.w
        self.w = w_new
        self.theta += self.w * dt
        if advance and self.w > 0:
            self.depth += dc.pitch * self.w / (2 * np.pi) * dt
        # the housing carries the output torque and the rotor's acceleration: -(tau_load + J dw/dt), plus the
        # clutch / cam-out ripple the gear train passes to the housing
        tau_housing = -(tau_load + dc.rotor_inertia * dw / dt) - ripple
        apply_point_forces(d, self.tool, [(tip, F_tip)], torque=tau_housing * a_t)
        self.tau_out = tau_load
        self.F_axial = F_a
        self.completed |= self.state == ("seated" if screw else "through")
        self._update_visual()

    def truth(self) -> dict[str, float]:
        return {"drill_depth": self.depth, "spindle_speed": self.w, "drill_torque": self.tau_out,
                "drill_axial_force": self.F_axial, "drill_state": float(STATES.index(self.state)),
                "cam_outs": float(self.cam_outs), "clutch_slipping": float(self.clutch_slipping),
                "bit_misalign": self.misalign}

    def done(self) -> bool:
        return self.completed

    @property
    def progress(self) -> float:
        return self.depth
