"""World: owns MjModel/MjData, index lookups, the grasp-settle reset and physics stepping."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.config import SimConfig
from tactile_sim.control.kinematics import site_jacobian, site_pose, solve_ik
from tactile_sim.model.builder import build_scene, tcp_rotation
from tactile_sim.sim.hands import make_hand_io


def full_inertia(m: mujoco.MjModel, d: mujoco.MjData, out: np.ndarray) -> np.ndarray:
    """Dense joint-space inertia. MuJoCo >= 3.3 takes (m, d, dst); older releases took (m, dst, qM)."""
    if hasattr(d, "qM"):
        mujoco.mj_fullM(m, out, d.qM)
    else:
        mujoco.mj_fullM(m, d, out)
    return out


@dataclass
class ArmState:
    t: float
    q: np.ndarray
    qd: np.ndarray
    tcp_pos: np.ndarray
    tcp_R: np.ndarray
    tcp_vel: np.ndarray  # [linear; angular] world frame
    J: np.ndarray  # 6 x 7, arm columns
    M: np.ndarray  # 7 x 7 arm block of the joint-space inertia
    bias: np.ndarray  # qfrc_bias on arm dofs (gravity + Coriolis)
    passive: np.ndarray  # qfrc_passive on arm dofs (damping, springs)
    tau_cmd: np.ndarray  # last commanded motor torque


class World:
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        spec = build_scene(self.cfg)
        self.spec = spec
        self.arm_source = spec.arm_source
        self.hand_source = spec.hand_source
        self.hammer_source = spec.hammer_source
        self.model = mujoco.MjModel.from_xml_string(spec.xml)
        self.data = mujoco.MjData(self.model)
        self.plant = spec.plant
        self.tcp_R_nominal = tcp_rotation(self.cfg)  # tool orientation at the nominal contact
        from tactile_sim.model.builder import strike_geometry

        self.geom = strike_geometry(self.cfg)
        self.hover_tcp, self.hover_R = self.geom.pose(0.0)  # = spec.hover_tcp, R nominal for a straight strike
        self._index()
        self.hand = make_hand_io(self)
        self.plant.bind(self.model, self.data)
        self._Mfull = np.zeros((self.model.nv, self.model.nv))
        self._vel6 = np.zeros(6)
        self.q_hover: np.ndarray | None = None
        self.step_count = 0

    # ------------------------------------------------------------------ indices
    def _index(self) -> None:
        m = self.model
        from tactile_sim.model.robots import arm_spec

        self.arm_spec = spec = arm_spec(self.cfg.arm)
        jid = [m.joint(n).id for n in spec.joints]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.arm_dofs = np.array([m.jnt_dofadr[j] for j in jid])
        self.arm_act = np.array([m.actuator(n).id for n in spec.motors])
        self.tau_limit = spec.torque.copy()
        self.position_arm = spec.interface == "position"
        if self.position_arm:
            from tactile_sim.model.robots import vega_hold_pose

            # every joint of the robot other than the striking arm holds a pose on its own servo
            self.held = {n: v for n, v in vega_hold_pose(self.cfg.arm).items()}
            self.held_qadr = np.array([m.joint(n).qposadr[0] for n in self.held])
            self.held_dofs = np.array([m.joint(n).dofadr[0] for n in self.held])
            self.held_act = np.array([m.actuator(f"m_{n}").id for n in self.held])
            self.held_q = np.array(list(self.held.values()))
            self.servo_kp = m.actuator_gainprm[self.arm_act, 0].copy()
            self.servo_kd = -m.actuator_biasprm[self.arm_act, 2].copy()
            self.left_hand_act = np.array([m.actuator(i).id for i in range(m.nu)
                                           if m.actuator(i).name.startswith("m_l_")], dtype=int)
            # relaxed, half-curled fingers; abduction and the thumb at their zero
            self.left_hand_q = np.array([0.0 if ("abd" in m.actuator(int(a)).name or "thumb" in m.actuator(int(a)).name)
                                         else 0.35 for a in self.left_hand_act])
        hj = m.joint("hammer_free")
        self.hammer_qadr = int(hj.qposadr[0])
        self.hammer_dofadr = int(hj.dofadr[0])
        self.hammer_body = m.body(names.HAMMER_BODY).id
        self.hand_body = m.body(names.HAND_BODY).id
        self.weld_id = m.equality(names.GRASP_WELD).id
        self.site = {n: m.site(n).id for n in (names.TCP_SITE, names.FT_SITE, names.HAMMER_REF_SITE,
                                                names.HAMMER_FACE_SITE, names.HAMMER_IMU_SITE, names.NAIL_HEAD_SITE)}
        self.tcp_site = self.site[names.TCP_SITE]
        self.wrist_dofs = np.array([m.joint(n).dofadr[0] for n in names.WRIST_FLEX_JOINTS
                                    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0], dtype=int)
        self.sensor_slices: dict[str, slice] = {}
        for i in range(m.nsensor):
            s = m.sensor(i)
            self.sensor_slices[s.name] = slice(int(s.adr[0]), int(s.adr[0] + s.dim[0]))
        self.handle_geom = m.geom(names.HAMMER_HANDLE_GEOM).id
        if self.cfg.gripper.hand == "franka":
            self.finger_qadr = np.array([m.joint(n).qposadr[0] for n in names.FINGER_JOINTS])
            self.pad_geoms = [m.geom(n).id for n in names.PAD_GEOMS]

    # ------------------------------------------------------------------ helpers
    @property
    def t(self) -> float:
        return float(self.data.time)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def sensor(self, name: str) -> np.ndarray:
        return self.data.sensordata[self.sensor_slices[name]]

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return site_pose(self.data, self.tcp_site)

    def site_velocity(self, sid: int) -> np.ndarray:
        """[linear; angular] velocity of a site in the world frame."""
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, sid, self._vel6, 0)
        return np.concatenate([self._vel6[3:], self._vel6[:3]])

    def arm_state(self) -> ArmState:
        m, d = self.model, self.data
        full_inertia(m, d, self._Mfull)
        dofs = self.arm_dofs
        p, R = self.tcp_pose()
        J = site_jacobian(m, d, self.tcp_site, dofs)
        return ArmState(
            t=self.t, q=d.qpos[self.arm_qadr].copy(), qd=d.qvel[dofs].copy(), tcp_pos=p, tcp_R=R,
            tcp_vel=self.site_velocity(self.tcp_site), J=J, M=self._Mfull[np.ix_(dofs, dofs)].copy(),
            bias=d.qfrc_bias[dofs].copy(), passive=d.qfrc_passive[dofs].copy(),
            tau_cmd=self.arm_torque())

    def arm_torque(self) -> np.ndarray:
        """Joint torque the striking arm's drives apply this step."""
        return self.data.actuator_force[self.arm_act].copy()

    def gravity_torque(self, q_arm: np.ndarray) -> np.ndarray:
        """Static torque the striking arm needs at `q_arm` (other joints as they are), tool included."""
        m, d = self.model, self.data
        if not hasattr(self, "_dg"):
            self._dg = mujoco.MjData(m)
        dg = self._dg
        dg.qpos[:] = d.qpos
        dg.qpos[self.arm_qadr] = q_arm
        dg.qvel[:] = 0.0
        mujoco.mj_forward(m, dg)
        tau = dg.qfrc_bias[self.arm_dofs].copy()
        mass, com = self.hammer_payload()
        p, R = site_pose(dg, self.tcp_site)
        jp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, dg, jp, None, p + R @ com, self.hand_body)
        return tau + jp[:, self.arm_dofs].T @ (-mass * m.opt.gravity)

    def set_arm_position(self, q: np.ndarray, qd: np.ndarray | None = None, droop: np.ndarray | None = None) -> None:
        """Position-interface arm: send joint targets (and velocity feedforward) to the drives' PD servos.

        The servo applies kp (target - q) + kd (qd_ff - qd), so the target is offset by droop = g(q) / kp to
        land on q under gravity, and by kd / kp * qd_ff to carry the velocity feedforward."""
        tgt = np.asarray(q, float).copy()
        if droop is not None:
            tgt += droop
        if qd is not None:
            tgt += self.servo_kd / self.servo_kp * np.asarray(qd, float)
        self.data.ctrl[self.arm_act] = tgt

    def held_droop(self) -> np.ndarray:
        """Static servo droop of the held joints at the current posture (gravity torque / kp)."""
        m, d = self.model, self.data
        if not hasattr(self, "_dh"):
            self._dh = mujoco.MjData(m)
        dh = self._dh
        dh.qpos[:] = d.qpos
        dh.qpos[self.held_qadr] = self.held_q
        dh.qvel[:] = 0.0
        mujoco.mj_forward(m, dh)
        return dh.qfrc_bias[self.held_dofs] / m.actuator_gainprm[self.held_act, 0]

    def hold_others(self, update: bool = True) -> None:
        """Held joints (torso, head, left arm) and the idle left hand keep their poses; their targets carry the
        droop of the current posture (the striking arm moves the torso's load)."""
        d = self.data
        if update:
            self._held_droop = self.held_droop()
        d.ctrl[self.held_act] = self.held_q + self._held_droop
        if len(self.left_hand_act):
            d.ctrl[self.left_hand_act] = self.left_hand_q

    def set_arm_torque(self, tau: np.ndarray) -> np.ndarray:
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)
        self.data.ctrl[self.arm_act] = tau
        return tau

    def set_grip_force(self, f: float) -> None:
        """Grip command (N): squeeze per pad on the Franka Hand, summed patch force on a dexterous hand."""
        self.hand.set_grip(f)

    def grip_truth(self) -> float:
        """Ground-truth grip force in the grip command's units."""
        return self.hand.grip_from_patches(self.pad_normal_forces())

    def _hold_arm(self) -> None:
        if self.position_arm:
            self.set_arm_position(self.q_hover, droop=self._hover_droop)
            self.hold_others(update=False)
        else:
            self.set_arm_torque(self.hold_torque(self.q_hover))

    def hold_torque(self, q_ref: np.ndarray) -> np.ndarray:
        a = self.cfg.arm
        d = self.data
        q = d.qpos[self.arm_qadr]
        qd = d.qvel[self.arm_dofs]
        return np.asarray(a.hold_kp) * (q_ref - q) - np.asarray(a.hold_kd) * qd + d.qfrc_bias[self.arm_dofs]

    def hammer_in_hand(self) -> tuple[np.ndarray, np.ndarray]:
        """Hammer grasp-frame pose relative to the TCP frame: (position, rotation vector)."""
        pt, Rt = self.tcp_pose()
        ph, Rh = site_pose(self.data, self.site[names.HAMMER_REF_SITE])
        p_rel = Rt.T @ (ph - pt)
        R_rel = Rt.T @ Rh
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R_rel.reshape(-1))
        v = np.zeros(3)
        mujoco.mju_quat2Vel(v, q, 1.0)
        return p_rel, v

    def pad_normal_forces(self) -> np.ndarray:
        """Ground-truth normal force on each pad (taxel patch) from the handle (N)."""
        if self.hand.kind != "franka":
            return self.hand.patch_forces()
        out = np.zeros(2)
        f6 = np.zeros(6)
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if self.handle_geom not in (c.geom1, c.geom2):
                continue
            for k, pg in enumerate(self.pad_geoms):
                if pg in (c.geom1, c.geom2):
                    mujoco.mj_contactForce(self.model, d, i, f6)
                    out[k] += f6[0]
        return out

    def joint_friction(self) -> np.ndarray:
        """Generalized dry-friction force MuJoCo applied on each arm dof in the last step (a constraint)."""
        d = self.data
        out = np.zeros(len(self.arm_dofs))
        nefc = d.nefc
        if nefc == 0:
            return out
        typ = d.efc_type[:nefc]
        ids = d.efc_id[:nefc]
        frc = d.efc_force[:nefc]
        mask = typ == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
        for k, dof in enumerate(self.arm_dofs):
            sel = mask & (ids == dof)
            if np.any(sel):
                out[k] = float(frc[sel].sum())
        return out

    def payload_torque(self, mass: float, com_tcp: np.ndarray) -> np.ndarray:
        """Joint torques that hold a payload of `mass` whose CoM sits at `com_tcp` in the TCP frame."""
        m, d = self.model, self.data
        p, R = self.tcp_pose()
        point = p + R @ com_tcp
        jp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, None, point, self.hand_body)
        return jp[:, self.arm_dofs].T @ (-mass * m.opt.gravity)

    def payload_mass_matrix(self, mass: float, com_tcp: np.ndarray, inertia_tcp: np.ndarray) -> np.ndarray:
        """Joint-space inertia the payload adds to the arm (rigidly attached at the TCP)."""
        m, d = self.model, self.data
        p, R = self.tcp_pose()
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, p + R @ com_tcp, self.hand_body)
        Jp, Jr = jp[:, self.arm_dofs], jr[:, self.arm_dofs]
        Iw = R @ inertia_tcp @ R.T
        return mass * Jp.T @ Jp + Jr.T @ Iw @ Jr

    def hammer_inertia_tcp(self) -> np.ndarray:
        """Nominal tool inertia about its CoM, in the grasp (= TCP) frame."""
        b = self.hammer_body
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, self.model.body_iquat[b])
        Rq = Rq.reshape(3, 3)
        return Rq @ np.diag(self.model.body_inertia[b]) @ Rq.T

    def hammer_payload(self) -> tuple[float, np.ndarray]:
        """Nominal tool mass and CoM in the grasp (= TCP) frame, from the model."""
        b = self.hammer_body
        return float(self.model.body_mass[b]), self.model.body_ipos[b].copy()

    def pad_taxels(self, side: int, spread: float = 0.003) -> np.ndarray:
        """Normal force per taxel on one pad / patch (row-major); see HandIO.taxels."""
        return self.hand.taxels(side, spread)

    # ------------------------------------------------------------------ stepping
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.plant.pre_step()
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1

    def ik(self, p_des: np.ndarray, R_des: np.ndarray | None = None, q_seed: np.ndarray | None = None,
           ) -> tuple[np.ndarray, float]:
        """Two passes: damped least squares with a pull toward the seed posture, then an exact polish."""
        R_des = self.tcp_R_nominal if R_des is None else R_des
        seed = np.asarray(self.cfg.arm.q_seed if q_seed is None else q_seed, dtype=float)
        base = self.data.qpos.copy()
        q, _ = solve_ik(self.model, self.tcp_site, p_des, R_des, self.arm_qadr, self.arm_dofs, seed,
                        qpos_base=base, q_rest=seed, rest_gain=0.05, iters=1500, tol=1e-7)
        return solve_ik(self.model, self.tcp_site, p_des, R_des, self.arm_qadr, self.arm_dofs, q,
                        qpos_base=base, rest_gain=0.0, iters=500, tol=1e-9)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None, settle_weld: float = 0.15, settle_free: float = 0.25) -> None:
        """Put the arm at the hover pose with the hammer grasped and settled; time restarts at 0."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.position_arm:
            d.qpos[self.held_qadr] = self.held_q
            mujoco.mj_forward(m, d)
        if self.q_hover is None:
            q, res = self.ik(self.hover_tcp, self.hover_R)
            if res > 1e-4:
                raise RuntimeError(f"IK for the hover pose failed (residual {res:.2e})")
            self.q_hover = q
        d.qpos[self.arm_qadr] = self.q_hover
        if self.position_arm:
            self._hover_droop = self.gravity_torque(self.q_hover) / self.servo_kp
            self._held_droop = self.held_droop()
        self.hand.init_pose()
        mujoco.mj_kinematics(m, d)
        pt, Rt = self.tcp_pose()
        qt = np.zeros(4)
        mujoco.mju_mat2Quat(qt, Rt.reshape(-1))
        d.qpos[self.hammer_qadr:self.hammer_qadr + 3] = pt
        d.qpos[self.hammer_qadr + 3:self.hammer_qadr + 7] = qt
        self.plant.reset()
        mujoco.mj_forward(m, d)
        # weld hammer to hand at the current relative pose while the pads load up
        ph, Rh = d.xpos[self.hand_body].copy(), d.xmat[self.hand_body].reshape(3, 3).copy()
        pb, Rb = d.xpos[self.hammer_body].copy(), d.xmat[self.hammer_body].reshape(3, 3).copy()
        rel_p = Rh.T @ (pb - ph)
        rel_q = np.zeros(4)
        mujoco.mju_mat2Quat(rel_q, (Rh.T @ Rb).reshape(-1))
        eq = m.eq_data[self.weld_id]
        eq[0:3] = 0.0
        eq[3:6] = rel_p
        eq[6:10] = rel_q
        eq[10] = 1.0
        d.eq_active[self.weld_id] = 1
        hold = self.cfg.controller.grip_hold
        n_weld = int(round(settle_weld / self.dt))
        for i in range(n_weld):
            self.set_grip_force(hold * min(1.0, 2.0 * i / n_weld))
            self.hand.tick(self.t)
            self._hold_arm()
            self.step()
        d.eq_active[self.weld_id] = 0
        self.hand.locked = False
        for i in range(int(round(settle_free / self.dt))):
            if i == int(round(0.5 * settle_free / self.dt)):
                if hasattr(self.hand, "lock_engaged"):
                    self.hand.lock_engaged()  # self-locking drives engage once the wrap has seated
            self.set_grip_force(hold)
            self.hand.tick(self.t)
            self._hold_arm()
            self.step()
        d.time = 0.0
        self.step_count = 0
