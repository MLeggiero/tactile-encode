"""World: owns MjModel/MjData, index lookups, the grasp-settle reset and physics stepping."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.config import SimConfig
from tactile_sim.control.kinematics import site_jacobian, site_pose, solve_ik
from tactile_sim.model.builder import build_scene, tcp_rotation


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
        self.model = mujoco.MjModel.from_xml_string(spec.xml)
        self.data = mujoco.MjData(self.model)
        self.plant = spec.plant
        self.hover_tcp = spec.hover_tcp.copy()
        self.tcp_R_nominal = tcp_rotation()
        self._index()
        self.plant.bind(self.model, self.data)
        self._Mfull = np.zeros((self.model.nv, self.model.nv))
        self._vel6 = np.zeros(6)
        self.q_hover: np.ndarray | None = None
        self.step_count = 0

    # ------------------------------------------------------------------ indices
    def _index(self) -> None:
        m = self.model
        jid = [m.joint(n).id for n in names.ARM_JOINTS]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.arm_dofs = np.array([m.jnt_dofadr[j] for j in jid])
        self.arm_act = np.array([m.actuator(n).id for n in names.ARM_MOTORS])
        self.tau_limit = m.actuator_ctrlrange[self.arm_act, 1].copy()
        self.finger_qadr = np.array([m.joint(n).qposadr[0] for n in names.FINGER_JOINTS])
        self.grip_act = m.actuator(names.GRIP_MOTOR).id
        hj = m.joint("hammer_free")
        self.hammer_qadr = int(hj.qposadr[0])
        self.hammer_dofadr = int(hj.dofadr[0])
        self.hammer_body = m.body(names.HAMMER_BODY).id
        self.hand_body = m.body(names.HAND_BODY).id
        self.weld_id = m.equality(names.GRASP_WELD).id
        self.site = {n: m.site(n).id for n in (names.TCP_SITE, names.FT_SITE, names.HAMMER_REF_SITE,
                                                names.HAMMER_FACE_SITE, names.HAMMER_IMU_SITE, names.NAIL_HEAD_SITE,
                                                *names.PAD_IMU_SITES)}
        self.tcp_site = self.site[names.TCP_SITE]
        self.wrist_dofs = np.array([m.joint(n).dofadr[0] for n in names.WRIST_FLEX_JOINTS
                                    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0], dtype=int)
        self.sensor_slices: dict[str, slice] = {}
        for i in range(m.nsensor):
            s = m.sensor(i)
            self.sensor_slices[s.name] = slice(int(s.adr[0]), int(s.adr[0] + s.dim[0]))
        self.pad_geoms = [m.geom(n).id for n in names.PAD_GEOMS]
        self.handle_geom = m.geom(names.HAMMER_HANDLE_GEOM).id

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
            tau_cmd=d.ctrl[self.arm_act].copy())

    def set_arm_torque(self, tau: np.ndarray) -> np.ndarray:
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)
        self.data.ctrl[self.arm_act] = tau
        return tau

    def set_grip_force(self, f: float) -> None:
        """Squeeze force per pad (N); positive closes."""
        g = self.cfg.gripper.grip_force_max
        self.data.ctrl[self.grip_act] = -float(np.clip(f, -g, g))

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
        """Ground-truth normal force on each pad from the handle (N)."""
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
        if self.q_hover is None:
            q, res = self.ik(self.hover_tcp)
            if res > 1e-4:
                raise RuntimeError(f"IK for the hover pose failed (residual {res:.2e})")
            self.q_hover = q
        d.qpos[self.arm_qadr] = self.q_hover
        g, h = self.cfg.gripper, self.cfg.hammer
        d.qpos[self.finger_qadr] = h.handle_radius + 2 * g.pad_half[1]
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
            self.set_arm_torque(self.hold_torque(self.q_hover))
            self.step()
        d.eq_active[self.weld_id] = 0
        for _ in range(int(round(settle_free / self.dt))):
            self.set_grip_force(hold)
            self.set_arm_torque(self.hold_torque(self.q_hover))
            self.step()
        d.time = 0.0
        self.step_count = 0
