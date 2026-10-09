"""The 12 V drill plant batched over environments: the GPU mirror of tactile_sim.model.plants.drill.DrillPlant.

`TorchDrillLaws` is the plant's pre_step without MuJoCo: from the tool's kinematics (its rotation, the bit tip's
position and velocity, the tool's angular velocity), the trigger's travel and the time, per env: the tip's support
along the work axis and its lateral spring (the recess or the hole wall, offset by the bit's runout turning with the
chuck) or friction, the screw's state machine (free, engaged with the Torx lobes' free turn, seated, cam-out,
stripped, spun, snapped) or the hole's (drilling, catch, through), the load the chuck must turn (with the load's
band-limited noise), the drivetrain (tactile_rl.plants.drivetrain), the screw's or the bit's advance, the bounds on
the tip's loads, the motor's unbalance at the chuck and the housing's reaction torque. It returns the wrench on the
tool as DrillPlant applies it (point forces at the tip and the chuck plus a free torque), float64.

The random draws (the lobes' free turn on engaging, the noise's normal increments) come from a torch generator, or
from the caller (`draws`) for parity against the CPU plant's own generator.
"""

from __future__ import annotations

import math

import torch

from tactile_rl.plants.drivetrain import TorchDrivetrain
from tactile_sim.model.plants.drill import FAILED, HEAD_RADIUS, STATES
from tactile_sim.model.plants.drivetrain import DrivetrainParams

S = {name: k for k, name in enumerate(STATES)}
FAILED_CODES = tuple(S[n] for n in FAILED)
# the plant's per-env randomized parameters beside the drivetrain's (sim.randomize._drill_dr)
PLANT_PARAMS = ("strip_torque", "screw_twist_off", "cut_energy", "exit_len")


def trigger_level(dc, travel: torch.Tensor) -> torch.Tensor:
    """tactile_sim.model.tools.trigger_level: the motor's command (0..1) from the trigger's travel (m)."""
    x = travel / dc.trigger_travel
    return torch.clamp((x - dc.trigger_dead) / (1.0 - dc.trigger_dead), 0.0, 1.0)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1)


class TorchDrillLaws:
    def __init__(self, dc, num_envs: int, axis, head0, device: str | torch.device = "cpu",
                 generator: torch.Generator | None = None):
        if dc.model != "12v":
            raise ValueError("the GPU drill is the 12 V model (DrillCfg.model = '12v')")
        if dc.mode not in ("screw", "hole") or dc.drive not in ("torx", "phillips"):
            raise ValueError(f"unknown drill mode / drive {dc.mode!r} / {dc.drive!r}")
        self.dc = dc
        self.n_env, self.dev = int(num_envs), torch.device(device)
        f64 = dict(dtype=torch.float64, device=self.dev)
        self.screw = dc.mode == "screw"
        self.torx = dc.drive == "torx"
        a = torch.as_tensor(axis, **f64).reshape(-1, 3).expand(self.n_env, 3)
        self.a = a / a.norm(dim=-1, keepdim=True)
        self.head0 = torch.as_tensor(head0, **f64).reshape(-1, 3).expand(self.n_env, 3).clone()
        self.surface = self.head0 + (self.a * dc.screw_len if self.screw else 0.0)
        self.bit_local = torch.as_tensor([0.0, -1.0, 0.0] if dc.form == "pistol" else [0.0, 0.0, 1.0], **f64)
        c0 = torch.tensor([1.0, 0.0, 0.0], **f64)
        self.cross_local = (c0, torch.linalg.cross(self.bit_local, c0))
        self.drive = TorchDrivetrain(DrivetrainParams.from_cfg(dc), self.n_env, self.dev)
        for k in PLANT_PARAMS:
            setattr(self, k, torch.full((self.n_env,), float(getattr(dc, k)), **f64))
        self.gen = generator
        # state
        self.depth = torch.zeros(self.n_env, **f64)
        self.state = torch.zeros(self.n_env, dtype=torch.long, device=self.dev)
        self.cam_outs = torch.zeros(self.n_env, dtype=torch.long, device=self.dev)
        self.cam_until = torch.full((self.n_env,), -math.inf, **f64)
        self.slack = torch.zeros(self.n_env, **f64)
        self.noise = torch.zeros(self.n_env, **f64)
        self.completed = torch.zeros(self.n_env, dtype=torch.bool, device=self.dev)
        self.trigger = torch.zeros(self.n_env, **f64)
        self.tau_out = torch.zeros(self.n_env, **f64)
        self.F_axial = torch.zeros(self.n_env, **f64)
        self.misalign = torch.zeros(self.n_env, **f64)
        self.tau_hand = torch.zeros(self.n_env, **f64)
        self.tip_lat = torch.full((self.n_env,), math.inf, **f64)
        self.tip_ax = torch.full((self.n_env,), -math.inf, **f64)

    # ------------------------------------------------------------------ parameters and resets
    def set_params(self, env_ids=None, **values) -> None:
        """Per-env values: the drivetrain's (DrivetrainParams names) and PLANT_PARAMS."""
        ids = slice(None) if env_ids is None else env_ids
        drive = {k: v for k, v in values.items() if k not in PLANT_PARAMS}
        if drive:
            self.drive.set_params(env_ids, **drive)
        for k in PLANT_PARAMS:
            if k in values:
                getattr(self, k)[ids] = torch.as_tensor(values[k], dtype=torch.float64, device=self.dev)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.drive.reset(env_ids)
        for t in (self.depth, self.slack, self.noise, self.trigger, self.tau_out, self.F_axial, self.misalign,
                  self.tau_hand):
            t[ids] = 0.0
        self.state[ids] = S["free"]
        self.cam_outs[ids] = 0
        self.cam_until[ids] = -math.inf
        self.completed[ids] = False
        self.tip_lat[ids] = math.inf
        self.tip_ax[ids] = -math.inf

    # ------------------------------------------------------------------ the model's laws
    def screw_torque(self, depth: torch.Tensor) -> torch.Tensor:
        dc = self.dc
        tau = dc.screw_torque0 + dc.screw_torque_per_m * depth
        return tau + torch.where(depth > dc.screw_len, dc.seat_stiffness * (depth - dc.screw_len),
                                 torch.zeros_like(depth))

    def cam_limit(self, F_axial: torch.Tensor, misalign: torch.Tensor) -> torch.Tensor:
        dc = self.dc
        if self.torx:
            ok = (misalign < dc.torx_max_angle) & (F_axial >= dc.torx_min_push)
            return torch.where(ok, torch.full_like(F_axial, math.inf), torch.full_like(F_axial, dc.torx_pop_torque))
        return dc.cam_ratio * torch.clamp(F_axial, min=0.0) * torch.clamp(1.0 - misalign / dc.cam_max_angle, min=0.0)

    def feed_per_rev(self, F_axial: torch.Tensor) -> torch.Tensor:
        dc = self.dc
        return torch.clamp(F_axial - dc.thrust_chisel, min=0.0) / (dc.thrust_k * dc.bit_diameter)

    def hole_torque(self, f_rev: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        dc = self.dc
        return dc.bit_drag + self.cut_energy * dc.bit_diameter ** 2 * f_rev / 8.0 + dc.flute_friction_per_m * depth

    def _normal(self) -> torch.Tensor:
        return torch.randn(self.n_env, dtype=torch.float64, device=self.dev, generator=self.gen)

    def _uniform(self) -> torch.Tensor:
        return torch.rand(self.n_env, dtype=torch.float64, device=self.dev, generator=self.gen)

    def step(self, t: torch.Tensor, dt: float, R: torch.Tensor, tip: torch.Tensor, v: torch.Tensor,
             omega: torch.Tensor, trigger: torch.Tensor | None = None, trigger_travel: torch.Tensor | None = None,
             draws: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """One physics step before it is taken (DrillPlant.pre_step): t (N,) the time, R (N, 3, 3) the tool's
        rotation, tip (N, 3) the bit tip, v (N, 3) its velocity, omega (N, 3) the tool's angular velocity (world),
        the trigger as a level (0..1) or the trigger part's travel (m). `draws`: optional "normal" (N,) for the
        noise and "uniform" (N,) for a lobe engagement (else the generator's). Returns the wrench's parts:
        F_tip at tip, F_unb at chuck, the free torque (world, float64)."""
        dc = self.dc
        f64 = torch.float64
        t, R, tip, v, omega = t.to(f64), R.to(f64), tip.to(f64), v.to(f64), omega.to(f64)
        if trigger_travel is not None:
            self.trigger = trigger_level(dc, trigger_travel.to(f64))
        elif trigger is not None:
            self.trigger = trigger.to(f64)
        a = self.a
        a_t = R @ self.bit_local
        e1, e2 = R @ self.cross_local[0], R @ self.cross_local[1]
        self.misalign = torch.arccos(torch.clamp(_dot(a_t, a), -1.0, 1.0))
        screw = self.screw
        st = self.state
        seat = (self.head0 if screw else self.surface) + a * self.depth[:, None]
        rel = tip - seat
        ax = _dot(rel, a)
        lat = rel - ax[:, None] * a
        r_lat = lat.norm(dim=-1)
        self.tip_lat, self.tip_ax = r_lat, ax
        v_ax = _dot(v, a)
        v_lat = v - v_ax[:, None] * a
        T = dc.board_thickness
        failed = torch.zeros_like(st, dtype=torch.bool)
        for c in FAILED_CODES:
            failed |= st == c

        # axial support at the tip
        k_ax = torch.full_like(ax, dc.k_axial)
        if not screw:
            fade = torch.clamp((T - self.depth) / self.exit_len, 0.0, 1.0)
            k_ax = torch.where(self.depth > T - self.exit_len, k_ax * fade, k_ax)
            k_ax = torch.where(st == S["through"], torch.zeros_like(k_ax), k_ax)
        on_head = (r_lat < HEAD_RADIUS) if screw else torch.ones_like(ax, dtype=torch.bool)
        d_ax = dc.d_axial * k_ax / dc.k_axial
        F_a = torch.where((ax > 0) & on_head & (k_ax > 0), torch.clamp(k_ax * ax + d_ax * v_ax, min=0.0),
                          torch.zeros_like(ax))
        if screw:
            F_a = torch.where(failed & (ax > 0), torch.clamp(dc.k_axial * ax + dc.d_axial * v_ax, min=0.0), F_a)
        F_tip = -F_a[:, None] * a
        # lateral: the recess or the hole wall keeps the tip on the axis; the bit's runout turns with the chuck
        psi = self.drive.out_angle[:, None]
        runout = dc.vib_runout * (torch.cos(psi) * e1 + torch.sin(psi) * e2)
        runout = runout - _dot(runout, a)[:, None] * a
        if screw:
            held = (ax > -0.0005) & (r_lat < 2 * dc.engage_radius) & ~failed
        else:
            held = (self.depth > 0.002) & (ax > -0.002) & (st != S["through"])
        spring = -dc.k_lateral * (lat - runout) - dc.d_lateral * v_lat
        n_vl = v_lat.norm(dim=-1, keepdim=True)
        rub = -0.3 * F_a[:, None] * torch.tanh(n_vl / 0.01) * (v_lat / torch.clamp(n_vl, min=1e-9))
        F_tip = F_tip + torch.where(held[:, None], spring, torch.where((F_a > 0)[:, None], rub, torch.zeros_like(rub)))

        # the load's band-limited noise (Ornstein-Uhlenbeck, unit RMS)
        if dc.vib_noise > 0.0:
            k = 2 * math.pi * dc.vib_noise_hz * dt
            z = draws["normal"] if draws is not None and "normal" in draws else self._normal()
            self.noise = self.noise - k * self.noise + math.sqrt(2 * k) * z.to(f64)
            noise = dc.vib_noise * self.noise
        else:
            noise = torch.zeros_like(ax)

        # the load the chuck must turn
        zero = torch.zeros_like(ax)
        need = zero
        f_rev = zero
        trig = self.trigger
        if screw:
            done_cam = (st == S["cam_out"]) & (t >= self.cam_until)
            stripped = done_cam & (self.cam_outs >= dc.strip_after) & (not self.torx)  # Phillips only
            st = torch.where(stripped, torch.full_like(st, S["stripped"]),
                             torch.where(done_cam, torch.full_like(st, S["free"]), st))
            live = (st == S["free"]) | (st == S["engaged"]) | (st == S["seated"])
            engaged = (F_a > 2.0) & (r_lat < dc.engage_radius)
            fresh = live & engaged & (st == S["free"])
            if self.torx:
                u = draws["uniform"] if draws is not None and "uniform" in draws else self._uniform()
                self.slack = torch.where(fresh, u.to(f64) * dc.lobe_gap, self.slack)
            else:
                self.slack = torch.where(fresh, zero, self.slack)
            new = torch.where(engaged, torch.where(st == S["seated"], st, torch.full_like(st, S["engaged"])),
                              torch.full_like(st, S["free"]))
            st = torch.where(live, new, st)
            turning = ((st == S["engaged"]) | (st == S["seated"])) & (self.slack <= 0.0)
            need = torch.where(turning, self.screw_torque(self.depth) * (1.0 + noise), need)
            need = torch.where(st == S["spun"], self.strip_residual_torque() * (1.0 + noise), need)
            cam = 0.3 * self.screw_torque(self.depth) * torch.clamp(torch.sin(4 * self.drive.theta), min=0.0)
            need = torch.where(st == S["cam_out"], cam, need)
        else:
            cutting = (st != S["through"]) & (F_a > 0) & ((self.drive.w > 0) | (trig > 0))
            catch = self.depth > T - self.exit_len
            f_rev = self.feed_per_rev(F_a)
            f_rev = torch.where(catch, torch.clamp(f_rev, min=dc.catch_feed), f_rev)
            hole = self.hole_torque(f_rev, self.depth) * torch.where(catch, torch.full_like(ax, dc.catch_gain),
                                                                     torch.ones_like(ax)) * (1.0 + noise)
            need = torch.where(cutting, hole, need)
            f_rev = torch.where(cutting, f_rev, zero)
            st = torch.where(cutting, torch.where(catch, torch.full_like(st, S["catch"]),
                                                  torch.full_like(st, S["drilling"])),
                             torch.where(st != S["through"], torch.full_like(st, S["free"]), st))
        need = torch.clamp(need, min=0.0)

        turn = self.drive.step(trig, need, dt)
        tau_c = self.drive.tau_chuck
        if screw:
            drv = (st == S["engaged"]) | (st == S["seated"])
            spin_free = drv & (self.slack > 0.0)
            self.slack = torch.where(spin_free, self.slack - torch.clamp(turn, min=0.0), self.slack)
            advance = drv & ~spin_free & ~self.drive.slipping
            self.depth = torch.where(advance, self.depth + dc.pitch * torch.clamp(turn, min=0.0) / (2 * math.pi),
                                     self.depth)
            st = torch.where(drv & self.drive.slipping & (self.depth >= dc.screw_len), torch.full_like(st, S["seated"]),
                             st)
            lim = torch.minimum(self.screw_twist_off, self.strip_torque)
            broke = drv & (tau_c >= lim)
            st_break = torch.where(self.screw_twist_off <= self.strip_torque, torch.full_like(st, S["snapped"]),
                                   torch.full_like(st, S["spun"]))
            cam_out = drv & ~broke & (tau_c > self.cam_limit(F_a, self.misalign)) & (trig > 0)
            st = torch.where(broke, st_break, torch.where(cam_out, torch.full_like(st, S["cam_out"]), st))
            self.cam_outs = self.cam_outs + cam_out.long()
            self.cam_until = torch.where(cam_out, t + dc.cam_time, self.cam_until)
            F_tip = F_tip - torch.where((st == S["cam_out"])[:, None], dc.cam_kick * a, torch.zeros_like(a))
        else:
            cut = (st == S["drilling"]) | (st == S["catch"])
            self.depth = torch.where(cut, self.depth + f_rev * torch.clamp(turn, min=0.0) / (2 * math.pi), self.depth)
            through = cut & (self.depth >= T)
            self.depth = torch.where(through, torch.full_like(self.depth, T), self.depth)
            st = torch.where(through, torch.full_like(st, S["through"]), st)
        # physical bounds on the tip's loads
        f_ax = _dot(F_tip, a)
        f_lat = F_tip - f_ax[:, None] * a
        n_lat = f_lat.norm(dim=-1)
        shrink = (dc.max_lateral / n_lat.clamp(min=1e-12))[:, None]
        f_lat = torch.where((n_lat > dc.max_lateral)[:, None], f_lat * shrink, f_lat)
        F_tip = torch.clamp(f_ax, -dc.max_axial, dc.max_axial)[:, None] * a + f_lat
        # the motor's unbalance at the chuck, turning with the rotor
        gr = self.drive.gear_ratio
        phi = (gr * self.drive.theta)[:, None]
        f_unb = (dc.vib_unbalance * (gr * self.drive.w) ** 2)[:, None]
        chuck = tip - a_t * dc.bit_len
        F_unb = f_unb * (torch.cos(phi) * e1 + torch.sin(phi) * e2)
        # the housing: the load's reaction through the gears plus the internals' angular momentum change and its turn
        self.tau_hand = self.drive.tau_housing
        L = (self.drive.momentum_inertia * self.drive.w)[:, None] * a_t
        torque = -self.tau_hand[:, None] * a_t - torch.linalg.cross(omega, L)
        self.tau_out = tau_c
        self.F_axial = F_a
        self.state = st
        self.completed = self.completed | (st == (S["seated"] if screw else S["through"]))
        return {"F_tip": F_tip, "tip": tip, "F_unb": F_unb, "chuck": chuck, "torque": torque}

    def strip_residual_torque(self) -> torch.Tensor:
        return self.dc.strip_residual * self.strip_torque

    @staticmethod
    def wrench_about(com: torch.Tensor, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """apply_point_forces: the parts as one wrench (N, 6) about the body's centre of mass (world frame)."""
        F = parts["F_tip"] + parts["F_unb"]
        Tq = (parts["torque"] + torch.linalg.cross(parts["tip"] - com, parts["F_tip"])
              + torch.linalg.cross(parts["chuck"] - com, parts["F_unb"]))
        return torch.cat([F, Tq], dim=-1)

    def truth(self) -> dict[str, torch.Tensor]:
        return {"drill_depth": self.depth, "spindle_speed": self.drive.w, "drill_torque": self.tau_out,
                "hand_torque": self.tau_hand, "drill_axial_force": self.F_axial, "drill_state": self.state.double(),
                "cam_outs": self.cam_outs.double(), "clutch_slipping": self.drive.slipping.double(),
                "clutch_phase": self.drive.phase, "bit_misalign": self.misalign}


class TorchDrillPlant:
    """The drill plant on MuJoCo Warp: TorchDrillLaws fed what DrillPlant.pre_step reads, its wrench into the tool's
    xfrc_applied before every physics step.

    The kinematics a CPU pre_step sees are those of the step before (MuJoCo's data after mj_step, positions and
    velocities from before its integration): `post_step` takes them right after each MuJoCo Warp step, so mjlab's
    per-env-step forward() does not shift them; an env just reset reads the live (forwarded) data once. The trigger
    part's travel is read live (qpos, as the CPU reads it)."""

    def __init__(self, parts, sim, num_envs: int, device: str | torch.device, prefix: str = "robot/", seed: int = 0):
        dp = parts.drill
        if dp is None:
            raise ValueError("the scene has no drill plant")
        m = sim.mj_model
        self.sim = sim
        self.n_env, self.dev = int(num_envs), torch.device(device)
        self.dt = float(m.opt.timestep)
        self.tool = m.body(prefix + parts.hammer_body).id
        self.tool_root = int(m.body_rootid[self.tool])
        self.tip_site = m.site(prefix + dp.tip_site).id
        self.trig_qadr = int(m.joint(prefix + dp.trigger_joint).qposadr[0]) if dp.trigger_joint else -1
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(int(seed))
        self.laws = TorchDrillLaws(parts.cfg.drill, self.n_env, dp.axis, dp.head0, self.dev, generator=gen)
        f64 = dict(dtype=torch.float64, device=self.dev)
        N = self.n_env
        self.R = torch.eye(3, **f64).expand(N, 3, 3).clone()
        self.tip = torch.zeros(N, 3, **f64)
        self.v = torch.zeros(N, 3, **f64)
        self.omega = torch.zeros(N, 3, **f64)
        self.com = torch.zeros(N, 3, **f64)
        self.fresh = torch.ones(N, dtype=torch.bool, device=self.dev)
        self.trigger_cmd = torch.zeros(N, **f64)  # the base's level when the tool has no trigger part

    def _read(self, data) -> tuple[torch.Tensor, ...]:
        N = self.n_env
        R = data.xmat[:][:, self.tool].reshape(N, 3, 3).double()
        tip = data.site_xpos[:][:, self.tip_site].double()
        cvel = data.cvel[:][:, self.tool].double()
        com_root = data.subtree_com[:][:, self.tool_root].double()
        v = cvel[:, 3:] - torch.linalg.cross(tip - com_root, cvel[:, :3])
        return R, tip, v, cvel[:, :3], data.xipos[:][:, self.tool].double()

    def post_step(self, data) -> None:
        self.R, self.tip, self.v, self.omega, self.com = self._read(data)
        self.fresh = torch.zeros_like(self.fresh)

    def pre_step(self, t: torch.Tensor) -> None:
        d = self.sim.data
        if bool(self.fresh.any()):
            live = self._read(d)
            f = self.fresh
            self.R = torch.where(f[:, None, None], live[0], self.R)
            self.tip, self.v, self.omega, self.com = (torch.where(f[:, None], a, b) for a, b in
                                                      zip(live[1:], (self.tip, self.v, self.omega, self.com),
                                                          strict=True))
        kw = ({"trigger_travel": d.qpos[:][:, self.trig_qadr].double()} if self.trig_qadr >= 0
              else {"trigger": self.trigger_cmd})
        parts = self.laws.step(t, self.dt, self.R, self.tip, self.v, self.omega, **kw)
        d.xfrc_applied[:, self.tool] = TorchDrillLaws.wrench_about(self.com, parts).float()

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.laws.reset(env_ids)
        self.fresh[ids] = True
        self.trigger_cmd[ids] = 0.0
        self.sim.data.xfrc_applied[ids, self.tool] = 0.0

    def done(self) -> torch.Tensor:
        return self.laws.completed
