"""Feasibility of planned arm motions on the CPU testbed, checked on a scratch copy of the model before an episode is
simulated (vla.variation), and the arm's speed and acceleration caps along a strike path (behaviors.swing).

ArmProbe holds a scratch MjData at the live configuration. It poses the arm at joint angles or solves damped least
squares IK for the controlled frame (as PositionL1 servoes it), carries the free-jointed tool rigidly with the hand,
and measures the clearance between the moving side (the striking arm's links, its hand, the tool) and the obstacles
(everything welded to the world: table, board, floor; and the robot's own torso, head and other arm). The robot's
geoms have collisions disabled in the scene (model/robots.py), so physics would never catch an elbow through the table.
The probe runs MuJoCo's collision pipeline on its own copy of the model in which only moving-vs-obstacle pairs collide,
with a CLEARANCE margin on the moving geoms; a contact means the pair is closer than CLEARANCE. (mj_geomDistance
reports zero for some mesh pairs tens of cm apart on this scene, MuJoCo 3.11.) Pairs in contact at the start
(adjacent links, the hand on the tool) are not checked.

path_caps generalises ScriptedSwing._arm_caps from the hover pose to samples along a path: the joint rates per unit
path speed from the Jacobian's pseudo-inverse, the joint torques per unit path acceleration from the arm's inertia plus
the tool's (World.payload_mass_matrix), and the 60 % torque margin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from tactile_sim.control.kinematics import pose_error

IK_TOL_POS = 1e-3  # m
IK_TOL_ROT = 1e-2  # rad
JOINT_MARGIN = 0.05  # rad from either end of a joint's range
MAX_STEP = 0.1  # rad per IK iteration
MAX_JUMP = 0.35  # rad, any joint, between neighbouring path samples (a jump means an IK branch flip)
CLEARANCE = 0.01  # m between the moving side and an obstacle


def strike_speed(dist: float, v_target: float, v_cap: float, a_cap: float, sw) -> float:
    """The contact speed ScriptedSwing._start_swing settles on for a stroke of `dist` (s) from rest, under the caps and
    SwingCfg `sw`: the braking quintic's speed, lowered until it fits; 0 when none fits above 0.2 m/s (the swing would
    fall back to an unbraked crawl). A single quintic cannot cruise, so long strokes at a low cap do not fit."""
    from tactile_sim.behaviors.trajectories import strike_profile

    v_max = sw.v_margin * v_cap
    a_max = min(sw.a_max, a_cap)
    v = min(v_target, v_max)
    while sw.brake_decel > 0 and v > 0.2:
        a_b = min(sw.brake_decel * v / sw.v_strike, 0.8 * a_max)
        if strike_profile(dist, v, a_b, a_max, min(v_max, 1.25 * v)) is not None:
            return v
        v *= 0.97
    return 0.0


@dataclass
class Verdict:
    ok: bool
    reason: str = "ok"  # unreachable | joint_margin | discontinuous | collision | bad_shape | slow_strike | ok
    where: str = ""
    worst: dict = field(default_factory=dict)


class ArmProbe:
    def __init__(self, world):
        self.w = w = world
        self.m = m = world.model
        self.d = mujoco.MjData(m)
        self.d.qpos[:] = world.data.qpos
        mujoco.mj_forward(m, self.d)
        jnt = [int(m.dof_jntid[i]) for i in w.arm_dofs]
        self.lo, self.hi = m.jnt_range[jnt, 0].copy(), m.jnt_range[jnt, 1].copy()
        # the tool rides with the hand: its pose relative to the hand body, as the grasp holds it now
        d0 = world.data
        R_h = d0.xmat[w.hand_body].reshape(3, 3)
        R_b = d0.xmat[w.hammer_body].reshape(3, 3)
        self._p_rel = R_h.T @ (d0.xpos[w.hammer_body] - d0.xpos[w.hand_body])
        self._R_rel = R_h.T @ R_b
        self.moving, self.obstacles = self._geom_sets()
        self._mc: mujoco.MjModel | None = None  # the collision copy, built on first use

    def _collision_model(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        if self._mc is None:
            import copy

            mc = copy.deepcopy(self.m)
            mc.geom_contype[:] = 0
            mc.geom_conaffinity[:] = 0
            mc.geom_contype[self.moving] = 1
            mc.geom_conaffinity[self.obstacles] = 1
            mc.geom_margin[:] = 0.0
            mc.geom_gap[:] = 0.0
            mc.geom_margin[self.moving] = CLEARANCE
            self._mc, self._dc = mc, mujoco.MjData(mc)
            self._moving_set, self._obstacle_set = set(self.moving), set(self.obstacles)
            self._dc.qpos[:] = self.w.data.qpos
            self._ignore = {pair for pair, _ in self._contacts()}  # in contact by design at the start
        return self._mc, self._dc

    def _contacts(self) -> list[tuple[tuple[int, int], float]]:
        mc, dc = self._mc, self._dc
        mujoco.mj_kinematics(mc, dc)
        mujoco.mj_collision(mc, dc)
        out = []
        for c in dc.contact[:dc.ncon]:
            a, b = int(c.geom1), int(c.geom2)
            if a in self._obstacle_set:
                a, b = b, a
            if a in self._moving_set and b in self._obstacle_set:
                out.append(((a, b), float(c.dist)))
        return out

    # ---------------------------------------------------------------- geometry sets
    def _geom_sets(self) -> tuple[list[int], list[int]]:
        m, w = self.m, self.w
        first = int(m.jnt_bodyid[int(m.dof_jntid[w.arm_dofs[0]])])
        arm = set()
        for b in range(m.nbody):
            p = b
            while p > 0 and p != first:
                p = int(m.body_parentid[p])
            if p == first:
                arm.add(b)
        tool = {b for b in range(m.nbody) if self._descends(b, w.hammer_body)}
        robot_root = int(m.body_rootid[first])
        moving, obstacles = [], []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue  # the floor: far below anything the arm reaches
            if b in arm or b in tool:
                moving.append(g)
            elif int(m.body_weldid[b]) == 0 or int(m.body_rootid[b]) == robot_root:
                obstacles.append(g)
        return moving, obstacles

    def _descends(self, b: int, ancestor: int) -> bool:
        while b > 0:
            if b == ancestor:
                return True
            b = int(self.m.body_parentid[b])
        return False

    # ---------------------------------------------------------------- posing
    def set_q(self, q: np.ndarray) -> None:
        """The arm at q, the tool carried in the hand, kinematics and inertia updated on the scratch data."""
        m, d, w = self.m, self.d, self.w
        d.qpos[w.arm_qadr] = q
        mujoco.mj_kinematics(m, d)
        R_h = d.xmat[w.hand_body].reshape(3, 3)
        p = d.xpos[w.hand_body] + R_h @ self._p_rel
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (R_h @ self._R_rel).reshape(-1))
        d.qpos[w.hammer_qadr:w.hammer_qadr + 3] = p
        d.qpos[w.hammer_qadr + 3:w.hammer_qadr + 7] = quat
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    def ik(self, p: np.ndarray, R: np.ndarray, q_seed: np.ndarray, iters: int = 200) -> tuple[np.ndarray, float, float]:
        """Damped least squares for the controlled frame from q_seed, steps capped at MAX_STEP: (q, position error,
        rotation error)."""
        w = self.w
        q = np.array(q_seed, float)
        e = np.zeros(6)
        for _ in range(iters):
            self.set_q(q)
            pc, Rc = w.ctrl_pose(self.d)
            e = pose_error(p, R, pc, Rc)
            if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4:
                break
            J = w.ctrl_jacobian(w.arm_dofs, self.d)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            n = float(np.max(np.abs(dq)))
            if n > MAX_STEP:
                dq *= MAX_STEP / n
            q = np.clip(q + dq, self.lo + 1e-3, self.hi - 1e-3)
        self.set_q(q)
        pc, Rc = w.ctrl_pose(self.d)
        e = pose_error(p, R, pc, Rc)
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))

    def min_clearance(self) -> tuple[float, str]:
        """The smallest distance below CLEARANCE between a moving geom and an obstacle at the probe's configuration
        (inf when every pair is clear), and the pair's names."""
        _, dc = self._collision_model()
        dc.qpos[:] = self.d.qpos
        best, where = np.inf, ""
        for (a, b), dist in self._contacts():
            if (a, b) not in self._ignore and dist < best:
                best, where = dist, f"{self.m.geom(a).name or a}/{self.m.geom(b).name or b}"
        return best, where

    def margin(self, q: np.ndarray) -> float:
        return float(np.min(np.minimum(q - self.lo, self.hi - q)))

    # ---------------------------------------------------------------- checks
    def check_poses(self, poses, q_seed: np.ndarray, label: str = "") -> tuple[Verdict, list[np.ndarray]]:
        """IK through a sequence of poses (each seeded from the last), checking reach, joint margin, continuity and
        clearance. Returns the verdict and the joint solutions so far."""
        qs, q = [], np.asarray(q_seed, float)
        worst = {"pos_err": 0.0, "margin": np.inf, "jump": 0.0, "clearance": np.inf}
        for i, (p, R) in enumerate(poses):
            q_new, ep, er = self.ik(p, R, q)
            worst["pos_err"] = max(worst["pos_err"], ep)
            if ep > IK_TOL_POS or er > IK_TOL_ROT:
                return Verdict(False, "unreachable", f"{label}[{i}]", worst), qs
            v = self._check_q(q_new, worst, f"{label}[{i}]", q_prev=q if qs else None)
            if v is not None:
                return v, qs
            qs.append(q_new)
            q = q_new
        return Verdict(True, worst=worst), qs

    def check_joint_path(self, q0: np.ndarray, q1: np.ndarray, n: int = 12, label: str = "preroll") -> Verdict:
        """The straight joint-space move q0 -> q1 (the pre-roll), sampled n times: margin and clearance."""
        worst = {"margin": np.inf, "jump": 0.0, "clearance": np.inf}
        for i, a in enumerate(np.linspace(0.0, 1.0, n)):
            v = self._check_q(np.asarray(q0, float) + a * (np.asarray(q1, float) - np.asarray(q0, float)), worst,
                              f"{label}[{i}]")
            if v is not None:
                return v
        return Verdict(True, worst=worst)

    def _check_q(self, q, worst, where, q_prev=None) -> Verdict | None:
        mg = self.margin(q)
        worst["margin"] = min(worst["margin"], mg)
        if mg < JOINT_MARGIN:
            return Verdict(False, "joint_margin", where, worst)
        if q_prev is not None:
            jump = float(np.max(np.abs(q - q_prev)))
            worst["jump"] = max(worst["jump"], jump)
            if jump > MAX_JUMP:
                return Verdict(False, "discontinuous", where, worst)
        self.set_q(q)
        clr, pair = self.min_clearance()
        worst["clearance"] = min(worst["clearance"], clr)
        if clr < CLEARANCE:
            return Verdict(False, "collision", f"{where} {pair}", worst)
        return None

    # ---------------------------------------------------------------- caps
    def path_caps(self, geom, s_values, q_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(v_cap, a_cap) at each s: the path speed the joint velocity limits allow, and 60 % of the path
        acceleration the joint torque limits can drive with the tool (as ScriptedSwing._arm_caps on an arc)."""
        from tactile_sim.sim.world import full_inertia

        w, m, d = self.w, self.m, self.d
        spec = w.arm_spec
        mass, com = w.hammer_payload()
        I_tool = w.hammer_inertia_tcp()
        M = np.zeros((m.nv, m.nv))
        v_cap, a_cap = [], []
        q = np.asarray(q_seed, float)
        for s in s_values:
            p, R = geom.pose(s)
            q, _, _ = self.ik(p, R, q)
            mujoco.mj_crb(m, d)
            J = w.ctrl_jacobian(w.arm_dofs, d)
            dx, om, _ = geom.tangent(s)
            dq = np.linalg.pinv(J) @ np.concatenate([dx, om])
            v_cap.append(float(np.min(spec.velocity / np.maximum(np.abs(dq), 1e-9))))
            M_arm = full_inertia(m, d, M)[np.ix_(w.arm_dofs, w.arm_dofs)]
            M_eff = M_arm + w.payload_mass_matrix(mass, com, I_tool, data=d)
            a_cap.append(0.6 * float(np.min(spec.torque / np.maximum(np.abs(M_eff @ dq), 1e-9))))
        return np.asarray(v_cap), np.asarray(a_cap)
