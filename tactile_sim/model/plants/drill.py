"""Drill plant: a 12 V cordless drill/driver on low gear over a board on a table, driving a pre-started Torx wood
screw ("screw") or drilling a small hole through ("hole") with the clutch engaged. The previous model is
plants/drill_v2.py (DrillCfg.model = "v2").

Phenomenological at the work, like the nail: the bit's interaction with the screw or the wood is applied to the
tool as forces at the bit tip and a torque about the bit. The tool's own mechanics are the drivetrain
(plants/drivetrain.py): a brushed motor on a reflected spindle inertia, a ball-detent clutch, and the housing
reaction from the internals' angular momentum. Its nominal values reproduce a 12 V drill/driver's published ratings
in the virtual dyno (python -m tactile_sim.calibrate drill): no-load speed, ISO 5393 soft- and hard-joint torque.

- Contact: the tip is supported axially by the screw head (or the hole bottom) with k_axial; in the recess, or in a
  started hole, a lateral spring keeps the tip on the axis (offset by the bit's runout as the chuck turns).
- Screw: a Torx bit engages when pressed into the recess within engage_radius of the axis, then turns free up to
  lobe_gap until the lobes catch. The screw turns with the chuck and advances pitch per turn against a torque that
  grows with depth and rises steeply once the head seats; the clutch then slips and ratchets ("seated"). A Torx bit
  leaves the recess only tilted past torx_max_angle or with no push under load. The wood strips out
  at strip_torque (the screw spins in place, "spun") and the screw twists off at screw_twist_off ("snapped").
  drive = "phillips": the v2 cam-out law (cam_ratio * axial push, less with tilt), stripping after strip_after.
- Hole (bit_diameter, a brad-point bit): thrust F = thrust_chisel + thrust_k D f sets the feed f per revolution; the
  torque is cut_energy D^2 f / 8 plus flute friction growing with depth; over the last exit_len the support fades
  and the bit grabs (catch_gain, at least catch_feed per turn) until the clutch slips; through, the tool lunges.
- Vibration: the motor's unbalance (a lateral force turning at the motor's speed), the bit's runout, and a
  band-limited noise on the load (grain, thread forming) that the drivetrain passes on like any load. Their sizes
  are estimates (a rotor balance grade, a runout), randomized widely: unlike the drivetrain they have no published
  rating to match, a drill's declared vibration only bounding them from above.

Loads at the grip: a steady reaction torque with untimed jerks (the lobes catching, clutch slip, catch,
breakthrough), the clutch's ratchet at a rate set by the motor's speed under the clutch's load.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.model.plants.base import Plant, add_table, apply_point_forces, body_point_velocity
from tactile_sim.model.plants.drivetrain import Drivetrain, DrivetrainParams
from tactile_sim.model.tools import trigger_level
from tactile_sim.model.xmlutil import sub

DRILL_BOARD = "drill_board"
SCREW_BODY = "screw"
HOLE_GEOM = "drill_hole"
# v2's states, then the new ones (the numeric drill_state of the old ones is unchanged)
STATES = ("free", "engaged", "cam_out", "seated", "stripped", "drilling", "catch", "through", "spun", "snapped")
FAILED = ("stripped", "spun", "snapped")  # the screw can no longer be driven
HEAD_RADIUS = 0.004


class DrillPlant(Plant):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dtp = DrivetrainParams.from_cfg(cfg.drill)
        self.drive = Drivetrain(self.dtp)

    def add_mjcf(self, root: ET.Element, worldbody: ET.Element, target: np.ndarray, axis: np.ndarray) -> None:
        """`target`: the screw head (screw mode) or the board surface (hole mode) on the bit axis; `axis`: the bit
        direction (into the board)."""
        dc = self.cfg.drill
        if dc.mode not in ("screw", "hole"):
            raise ValueError(f"unknown drill mode {dc.mode!r}")
        if dc.drive not in ("torx", "phillips"):
            raise ValueError(f"unknown screw drive {dc.drive!r}")
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
        sub(board, "geom", name=HOLE_GEOM, type="cylinder", size=(0.5 * dc.bit_diameter, 1e-5), pos=(0, 0, -0.5 * T),
            rgba=(0.25, 0.18, 0.10, 1), contype=0, conaffinity=0)
        if dc.mode == "screw":
            sb = sub(worldbody, "body", name=SCREW_BODY, mocap="true", pos=self.head0, quat=q)
            sub(sb, "geom", type="cylinder", size=(HEAD_RADIUS, 0.0012), pos=(0, 0, 0.0012), rgba=(0.75, 0.70, 0.45, 1),
                contype=0, conaffinity=0)
            if dc.drive == "torx":  # the star recess, drawn as its outline circle
                sub(sb, "geom", type="cylinder", size=(dc.engage_radius, 0.0003), pos=(0, 0, -0.0001),
                    rgba=(0.2, 0.2, 0.2, 1), contype=0, conaffinity=0)
            else:
                sub(sb, "geom", type="box", size=(0.0028, 0.0005, 0.0003), pos=(0, 0, -0.0001),
                    rgba=(0.2, 0.2, 0.2, 1), contype=0, conaffinity=0)
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
        # a trigger part (DrillCfg.trigger_mode "finger"): the motor reads its travel; else the base sets self.trigger
        self.trigger_qadr = (int(model.joint(names.TRIGGER_JOINT).qposadr[0])
                             if self.cfg.drill.form == "pistol" and self.cfg.drill.trigger_mode == "finger" else -1)
        # the bit's direction in the tool frame: the inline driver's +z, the pistol driver's -y; and two directions
        # across it (the runout's and the unbalance's plane)
        self.bit_local = np.array([0.0, -1.0, 0.0]) if self.cfg.drill.form == "pistol" else np.array([0.0, 0.0, 1.0])
        self.cross_local = (np.array([1.0, 0.0, 0.0]), np.cross(self.bit_local, [1.0, 0.0, 0.0]))
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
        """Largest torque the recess transmits before the bit leaves it: a Phillips bit cams out above
        cam_ratio * push (less tilted); a Torx bit only when tilted past torx_max_angle or not pressed in."""
        dc = self.cfg.drill
        if dc.drive == "torx":
            return np.inf if (misalign < dc.torx_max_angle and F_axial >= dc.torx_min_push) else dc.torx_pop_torque
        return dc.cam_ratio * max(F_axial, 0.0) * max(0.0, 1.0 - misalign / dc.cam_max_angle)

    def feed_per_rev(self, F_axial: float) -> float:
        """The bit's feed per revolution under this thrust: F = thrust_chisel + thrust_k D f."""
        dc = self.cfg.drill
        return max(F_axial - dc.thrust_chisel, 0.0) / (dc.thrust_k * dc.bit_diameter)

    def hole_torque(self, f_rev: float, depth: float) -> float:
        """Torque to turn the bit cutting f_rev per revolution at this depth: the lips' cutting energy plus the
        flutes' friction on the chips (grows with the hole's depth)."""
        dc = self.cfg.drill
        return dc.bit_drag + dc.cut_energy * dc.bit_diameter ** 2 * f_rev / 8.0 + dc.flute_friction_per_m * depth

    def motor_torque(self, trigger: float, w: float) -> float:
        """Drive torque at the chuck at this trigger and speed; with the trigger released, the dynamic brake."""
        return self.drive.motor_torque(trigger, w)

    # ---- hooks ----
    def reset(self, rng=None) -> None:
        dc = self.cfg.drill
        self.rng = rng if rng is not None else np.random.default_rng(dc.noise_seed)
        self.drive.reset()
        self.depth = 0.0  # screw advance (screw) or hole depth (hole)
        self.state = "free"
        self.cam_outs = 0
        self._cam_until = -np.inf
        self.slack = 0.0  # the Torx lobes' free turn left before they catch
        self.noise = 0.0  # the load's band-limited relative noise
        self.tau_out = self.F_axial = self.misalign = self.tau_hand = 0.0
        self.tip_lat, self.tip_ax = np.inf, -np.inf  # the bit tip off the axis, and along it past the head/surface (m)
        self.trigger = 0.0
        self.completed = False  # latched once the screw seats / the bit goes through
        self._update_visual()

    @property
    def w(self) -> float:
        return self.drive.w

    @property
    def clutch_slipping(self) -> bool:
        return self.drive.slipping

    @property
    def failed(self) -> bool:
        return self.state in FAILED

    def _update_visual(self) -> None:
        if not hasattr(self, "m"):
            return
        dc = self.cfg.drill
        if dc.mode == "screw":
            q = np.zeros(4)
            mujoco.mju_quatZ2Vec(q, self.a)
            th = self.drive.out_angle
            spin = np.array([np.cos(0.5 * th), 0.0, 0.0, np.sin(0.5 * th)])
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, q, spin)
            self.d.mocap_pos[self.screw_mocap] = self.head0 + self.a * self.depth
            self.d.mocap_quat[self.screw_mocap] = out
        else:
            h = max(self.depth, 1e-5)
            T = dc.board_thickness
            self.m.geom_size[self.hole_geom, 1] = 0.5 * h
            self.m.geom_pos[self.hole_geom, 2] = -0.5 * T + 0.5 * h - 1e-4

    def _step_noise(self, dt: float) -> float:
        """A first-order (Ornstein-Uhlenbeck) noise of unit RMS and vib_noise_hz bandwidth."""
        dc = self.cfg.drill
        if dc.vib_noise <= 0.0:
            return 0.0
        k = 2 * np.pi * dc.vib_noise_hz * dt
        self.noise += -k * self.noise + np.sqrt(2 * k) * float(self.rng.standard_normal())
        return dc.vib_noise * self.noise

    def pre_step(self) -> None:
        dc, d, m = self.cfg.drill, self.d, self.m
        dt = m.opt.timestep
        t = float(d.time)
        if self.trigger_qadr >= 0:
            self.trigger = trigger_level(dc, float(d.qpos[self.trigger_qadr]))
        R = d.xmat[self.tool].reshape(3, 3)
        tip = d.site_xpos[self.tip_site].copy()
        a_t = R @ self.bit_local  # bit axis
        e1, e2 = R @ self.cross_local[0], R @ self.cross_local[1]
        self.misalign = float(np.arccos(np.clip(a_t @ self.a, -1.0, 1.0)))
        v = body_point_velocity(m, d, self.tool, tip, self._buf)
        omega_tool = self._buf[:3].copy()
        screw = dc.mode == "screw"
        seat = (self.head0 + self.a * self.depth) if screw else (self.surface + self.a * self.depth)
        rel = tip - seat
        ax = float(rel @ self.a)
        lat = rel - ax * self.a
        r_lat = float(np.linalg.norm(lat))
        self.tip_lat, self.tip_ax = r_lat, ax
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
        if screw and self.state in FAILED and ax > 0:
            F_a = max(0.0, dc.k_axial * ax + dc.d_axial * v_ax)  # pressing on a ruined head still bears load
        F_tip = -F_a * self.a
        # lateral: recess or hole wall keeps the tip on the axis; the bit's runout turns with the chuck
        psi = self.drive.out_angle
        runout = dc.vib_runout * (np.cos(psi) * e1 + np.sin(psi) * e2)
        runout -= (runout @ self.a) * self.a
        in_recess = screw and ax > -0.0005 and r_lat < 2 * dc.engage_radius and self.state not in FAILED
        in_hole = (not screw) and self.depth > 0.002 and ax > -0.002 and self.state != "through"
        if in_recess or in_hole:
            F_tip += -dc.k_lateral * (lat - runout) - dc.d_lateral * v_lat
        elif F_a > 0:
            F_tip += -0.3 * F_a * np.tanh(np.linalg.norm(v_lat) / 0.01) * (v_lat / max(np.linalg.norm(v_lat), 1e-9))

        # the load the chuck must turn
        noise = self._step_noise(dt)
        need = 0.0
        f_rev = 0.0
        if screw:
            if self.state == "cam_out" and t >= self._cam_until:
                self.state = "stripped" if (dc.drive == "phillips" and self.cam_outs >= dc.strip_after) else "free"
            if self.state in ("free", "engaged", "seated"):
                engaged = F_a > 2.0 and r_lat < dc.engage_radius
                if engaged and self.state == "free":
                    self.slack = float(self.rng.uniform(0.0, dc.lobe_gap)) if dc.drive == "torx" else 0.0
                self.state = ("seated" if self.state == "seated" else "engaged") if engaged else "free"
            if self.state in ("engaged", "seated") and self.slack <= 0.0:
                need = self.screw_torque(self.depth) * (1.0 + noise)
            elif self.state == "spun":
                need = dc.strip_residual * dc.strip_torque * (1.0 + noise)
            elif self.state == "cam_out":  # the bit chatters on the recess's flanks on its way out
                need = 0.3 * self.screw_torque(self.depth) * max(np.sin(4 * self.drive.theta), 0.0)
        elif self.state != "through" and F_a > 0 and (self.drive.w > 0 or self.trigger > 0):
            catch = self.depth > T - dc.exit_len
            f_rev = self.feed_per_rev(F_a)
            if catch:  # the brad point's spurs grab the exit fibres and pull the bit through
                f_rev = max(f_rev, dc.catch_feed)
            need = self.hole_torque(f_rev, self.depth) * (dc.catch_gain if catch else 1.0) * (1.0 + noise)
            self.state = "catch" if catch else "drilling"
        elif self.state != "through":
            self.state = "free"
        need = max(need, 0.0)

        turn = self.drive.step(self.trigger, need, dt)
        tau_c = self.drive.tau_chuck
        if screw:
            if self.state in ("engaged", "seated"):
                if self.slack > 0.0:
                    self.slack -= max(turn, 0.0)
                elif not self.drive.slipping:
                    self.depth += dc.pitch * max(turn, 0.0) / (2 * np.pi)
                if self.drive.slipping and self.depth >= dc.screw_len:
                    self.state = "seated"
                if tau_c >= min(dc.screw_twist_off, dc.strip_torque):
                    self.state = "snapped" if dc.screw_twist_off <= dc.strip_torque else "spun"
                elif tau_c > self.cam_limit(F_a, self.misalign) and self.trigger > 0:
                    self.state = "cam_out"
                    self.cam_outs += 1
                    self._cam_until = t + dc.cam_time
            if self.state == "cam_out":
                F_tip -= dc.cam_kick * self.a
        elif self.state in ("drilling", "catch"):
            self.depth += f_rev * max(turn, 0.0) / (2 * np.pi)
            if self.depth >= T:
                self.depth = T
                self.state = "through"
        # physical bounds on the tip's loads (a bit knocked off the axis must not blow the contact springs up)
        f_ax = float(F_tip @ self.a)
        f_lat = F_tip - f_ax * self.a
        n_lat = float(np.linalg.norm(f_lat))
        if n_lat > dc.max_lateral:
            f_lat *= dc.max_lateral / n_lat
        F_tip = np.clip(f_ax, -dc.max_axial, dc.max_axial) * self.a + f_lat
        # the motor's unbalance: a lateral force at the chuck turning with the rotor
        phi = self.dtp.gear_ratio * self.drive.theta
        f_unb = dc.vib_unbalance * (self.dtp.gear_ratio * self.drive.w) ** 2
        chuck = tip - a_t * dc.bit_len
        F_unb = f_unb * (np.cos(phi) * e1 + np.sin(phi) * e2)
        # the housing: the screw's (or the wood's) reaction through the gears plus the internals' angular momentum
        # change, J_am dw/dt along the bit and (gyroscopic) its turn with the housing
        self.tau_hand = self.drive.tau_housing
        L = self.dtp.momentum_inertia * self.drive.w * a_t
        torque = -self.tau_hand * a_t - np.cross(omega_tool, L)
        apply_point_forces(d, self.tool, [(tip, F_tip), (chuck, F_unb)], torque=torque)
        self.tau_out = tau_c
        self.F_axial = F_a
        self.completed |= self.state == ("seated" if screw else "through")
        self._update_visual()

    def truth(self) -> dict[str, float]:
        return {"drill_depth": self.depth, "spindle_speed": self.drive.w, "drill_torque": self.tau_out,
                "hand_torque": self.tau_hand, "drill_axial_force": self.F_axial,
                "drill_state": float(STATES.index(self.state)), "cam_outs": float(self.cam_outs),
                "clutch_slipping": float(self.drive.slipping), "clutch_phase": self.drive.phase,
                "bit_misalign": self.misalign}

    def event_truth(self, t: float):
        """The next load event the hand will feel, as a forecast truth (behaviors.forecast.ForecastTruth): while a
        screw goes in, its seating, after which the clutch ratchets (periodic, at its click rate); while a hole is
        drilled, the breakthrough, where the support fades and the bit grabs (a step); a running clutch or a bit
        already through is the event under way. Untimed events (the lobes catching, a pop-out) are not forecast.
        amp is in N m for the clutch's ratchet (half its swing) and N for the breakthrough's step."""
        from tactile_sim.behaviors.forecast import ForecastTruth

        dc, p = self.cfg.drill, self.dtp
        trig = max(self.trigger, self.drive.u)
        # spindle revolutions per second: the speed the motor settles to at this trigger and load, not the transient
        # of a spin-up
        rev = max(self.drive.w, p.steady_speed(trig, self.tau_out)) / (2 * np.pi)
        # the clutch's mean torque over a detent, and the spindle speed it slips at
        t_slip = dc.clutch_torque * dc.clutch_duty * 0.5 * (1.0 + dc.clutch_low)
        f_rip = p.click_rate(self.drive.w if self.drive.slipping else p.steady_speed(trig, t_slip))
        amp = 0.5 * dc.clutch_torque
        if dc.mode == "screw":
            if self.drive.slipping or self.state == "seated":
                return ForecastTruth(True, t, "periodic", f_rip, amp)
            if self.state == "engaged" and rev > 0:
                remaining = max(dc.screw_len - self.depth, 0.0)
                return ForecastTruth(True, t + remaining / (dc.pitch * rev), "periodic", f_rip, amp)
            return ForecastTruth(False)
        T = dc.board_thickness
        if self.state == "through":
            return ForecastTruth(True, t, "step", 0.0, self.F_axial)
        if self.state in ("drilling", "catch") and rev > 0 and self.F_axial > 0:
            f_rev = max(self.feed_per_rev(self.F_axial), 1e-6)
            remaining = max(T - dc.exit_len - self.depth, 0.0)
            return ForecastTruth(True, t + remaining / (f_rev * rev), "step", 0.0, self.F_axial)
        return ForecastTruth(False)

    def done(self) -> bool:
        return self.completed

    @property
    def progress(self) -> float:
        return self.depth
