"""Offline wrap-grasp synthesis for the WUJI Hand 2 on the hammer handle.

The hand is fixed in its hover orientation (palm down, fingers along the strike axis) with gravity on.
The hammer is placed at the nominal grasp point and welded to the hand while the fingers close under
full synergy torque; the weld is then released and the hammer seats in the wrap under gravity. The seated
finger angles and hammer pose (in the hand frame) are the grasp keyframe: the testbed starts every
episode from it and uses the seated hammer frame as its tool frame.

Results are cached per configuration in ~/.cache/tactile_sim/grasps/.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.config import SimConfig
from tactile_sim.control.hand import WrapSynergy
from tactile_sim.model.hands import wuji
from tactile_sim.model.tool_hammer import add_hammer, handle_segment_geoms
from tactile_sim.model.xmlutil import sub

VERSION = 2


@dataclass
class WrapGrasp:
    q: np.ndarray  # hand joint angles (vendor joint order)
    hammer_pos: np.ndarray  # hammer frame in the hand frame
    hammer_quat: np.ndarray
    contact_force: dict[str, float]  # normal force per hand body at the keyframe (N)
    seat_shift: float  # hammer motion from the nominal placement to the seated pose (m)
    seat_rot: float  # rad
    hold_drift: float  # hammer motion over the last 0.5 s of the static hold (m)
    load_ratio: dict[str, float] | None = None  # hard-stop torque / rated torque, per joint
    # per hand body in contact: force-weighted contact centroid, outward normal and the handle axis, in the
    # body frame
    contact_frames: dict[str, tuple[list[float], list[float], list[float]]] | None = None

    @property
    def max_load_ratio(self) -> float:
        return max(self.load_ratio.values()) if self.load_ratio else float("nan")

    def to_dict(self) -> dict:
        return {"q": self.q.tolist(), "hammer_pos": self.hammer_pos.tolist(), "hammer_quat": self.hammer_quat.tolist(),
                "contact_force": self.contact_force, "seat_shift": self.seat_shift, "seat_rot": self.seat_rot,
                "hold_drift": self.hold_drift, "load_ratio": self.load_ratio,
                "contact_frames": self.contact_frames}

    @classmethod
    def from_dict(cls, d: dict) -> WrapGrasp:
        return cls(np.array(d["q"]), np.array(d["hammer_pos"]), np.array(d["hammer_quat"]), d["contact_force"],
                   d["seat_shift"], d["seat_rot"], d["hold_drift"], d.get("load_ratio"),
                   d.get("contact_frames"))


def hover_hand_rotation(cfg: SimConfig | None = None) -> np.ndarray:
    """Hand orientation in the world with the tool in its nominal strike orientation (builder.tcp_rotation), so
    the grasp seats under gravity the way the task will load it."""
    from tactile_sim.model.builder import tcp_rotation

    return tcp_rotation(cfg) @ wuji.R_HAND_TCP.T


def _gravity_in_tool(cfg: SimConfig) -> list[float]:
    from tactile_sim.model.builder import tcp_rotation

    return np.round(tcp_rotation(cfg).T @ np.array([0.0, 0.0, -1.0]), 3).tolist()


def _cache_key(cfg: SimConfig, hand_path: Path) -> str:
    g, h = cfg.gripper, cfg.hammer
    blob = json.dumps({"v": VERSION, "tcp": g.wrap_tcp, "thumb": g.thumb_close, "pre": g.thumb_preshape,
                       "syn": g.finger_synergy,
                       "sol": g.hand_solref,
                       "mu": g.hand_friction, "tors": g.pad_torsion, "damp": g.hand_joint_damping,
                       "fric": g.hand_joint_friction, "dt": cfg.physics.timestep, "hammer": h.model,
                       "grip": h.grip_from_head, "mass": (h.head_mass, h.handle_mass), "g": _gravity_in_tool(cfg),
                       "hand": hashlib.sha256(hand_path.read_bytes()).hexdigest()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_path(cfg: SimConfig, hand_path: Path) -> Path:
    return Path.home() / ".cache" / "tactile_sim" / "grasps" / f"wuji2_{_cache_key(cfg, hand_path)}.json"


def build_scene(cfg: SimConfig, hand_path: Path) -> tuple[str, dict]:
    ph = cfg.physics
    root = ET.Element("mujoco", model="wuji_grasp")
    sub(root, "compiler", angle="radian")
    sub(root, "option", timestep=ph.timestep, integrator=ph.integrator, cone=ph.cone, impratio=ph.impratio,
        noslip_iterations=ph.noslip_iterations, iterations=ph.iterations, gravity=ph.gravity)
    sub(root, "asset")
    wb = sub(root, "worldbody")
    sub(root, "actuator")
    contact = sub(root, "contact")
    q_hand = np.zeros(4)
    mujoco.mju_mat2Quat(q_hand, hover_hand_rotation(cfg).reshape(-1))
    q_mount_inv = np.zeros(4)
    mujoco.mju_negQuat(q_mount_inv, np.array(wuji.mount_quat(cfg.gripper.mount_yaw)))
    q_base = np.zeros(4)
    mujoco.mju_mulQuat(q_base, q_hand, q_mount_inv)
    base = sub(wb, "body", name="flange", pos=(0, 0, 0.5), quat=q_base)
    info = wuji.add_wuji_hand(base, root, cfg.gripper, hand_path, (0, 0, 0))
    add_hammer(wb, cfg.hammer, root)
    wuji.contact_pairs(contact, info["col_geoms"], handle_segment_geoms(cfg.hammer), cfg.gripper)
    eq = sub(root, "equality")
    sub(eq, "weld", name=names.GRASP_WELD, body1=names.HAND_BODY, body2=names.HAMMER_BODY, active="false",
        solref=(0.005, 1))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode"), info


def contact_forces_by_body(m: mujoco.MjModel, d: mujoco.MjData, other_body: int) -> dict[str, float]:
    out: dict[str, float] = {}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if other_body not in (b1, b2):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        b = m.body(b2 if b1 == other_body else b1).name
        out[b] = out.get(b, 0.0) + float(f6[0])
    return out


def _rel(m, d, hand, hb):
    Rh = d.xmat[hand].reshape(3, 3)
    return Rh.T @ (d.xpos[hb] - d.xpos[hand]), Rh.T @ d.xmat[hb].reshape(3, 3)


def synthesize(cfg: SimConfig, hand_path: Path, close_time: float = 0.4, seat_time: float = 0.6,
               hold_time: float = 0.5) -> WrapGrasp:
    xml, info = build_scene(cfg, hand_path)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    joints = info["joints"]
    qadr = np.array([m.joint(j).qposadr[0] for j, _ in joints])
    dadr = np.array([m.joint(j).dofadr[0] for j, _ in joints])
    act = np.array([m.actuator(f"m_{j}").id for j, _ in joints])
    syn = WrapSynergy.for_wuji(joints, cfg.gripper.thumb_close, cfg.gripper.thumb_preshape,
                               cfg.gripper.finger_synergy)
    hand, hb = m.body(names.HAND_BODY).id, m.body(names.HAMMER_BODY).id
    tcp = m.site(names.TCP_SITE).id
    hq = m.joint("hammer_free").qposadr[0]
    d.qpos[qadr] = syn.q_hold
    mujoco.mj_kinematics(m, d)
    d.qpos[hq:hq + 3] = d.site_xpos[tcp]
    qq = np.zeros(4)
    mujoco.mju_mat2Quat(qq, d.site_xmat[tcp])
    d.qpos[hq + 3:hq + 7] = qq
    mujoco.mj_forward(m, d)
    p0, R0 = _rel(m, d, hand, hb)
    e = m.equality(names.GRASP_WELD).id
    eq = m.eq_data[e]
    eq[:] = 0.0
    eq[3:6] = p0
    mujoco.mju_mat2Quat(qq, R0.reshape(-1))
    eq[6:10] = qq
    eq[10] = 1.0
    d.eq_active[e] = 1

    def run(T):
        for _ in range(int(round(T / m.opt.timestep))):
            d.ctrl[act] = syn.torque(1.0, d.qpos[qadr], d.qvel[dadr])
            mujoco.mj_step(m, d)

    run(close_time)
    d.eq_active[e] = 0
    run(seat_time)
    pa, Ra = _rel(m, d, hand, hb)
    run(hold_time)
    p1, R1 = _rel(m, d, hand, hb)
    q1 = np.zeros(4)
    mujoco.mju_mat2Quat(q1, R1.reshape(-1))
    ang = float(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
    forces = {k: round(v, 2) for k, v in contact_forces_by_body(m, d, hb).items()}
    stop = joint_loads(m, d, dadr)
    ratio = {j: round(float(abs(x) / lim), 3) for (j, lim), x in zip(joints, stop, strict=True)}
    return WrapGrasp(d.qpos[qadr].copy(), p1, q1, forces, float(np.linalg.norm(p1 - p0)), ang,
                     float(np.linalg.norm(p1 - pa)), ratio, contact_frames(m, d, hb))


def contact_frames(m: mujoco.MjModel, d: mujoco.MjData, other_body: int) -> dict[str, tuple[list, list]]:
    """Per hand body touching `other_body`: force-weighted contact point, outward surface normal (from the
    hand toward the tool) and the tool's x axis (the handle), all in the hand body's frame."""
    ax = d.xmat[other_body].reshape(3, 3)[:, 0]
    acc: dict[int, list] = {}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if other_body not in (b1, b2):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        if f6[0] <= 0:
            continue
        hb = b2 if b1 == other_body else b1
        n = c.frame[:3].copy() * (1.0 if b1 == hb else -1.0)  # contact normal points from geom1 to geom2
        a = acc.setdefault(hb, [0.0, np.zeros(3), np.zeros(3)])
        a[0] += f6[0]
        a[1] += f6[0] * c.pos
        a[2] += f6[0] * n
    out = {}
    for b, (f, ps, ns) in acc.items():
        R, x = d.xmat[b].reshape(3, 3), d.xpos[b]
        n = ns / max(np.linalg.norm(ns), 1e-12)
        out[m.body(b).name] = (np.round(R.T @ (ps / f - x), 6).tolist(), np.round(R.T @ n, 6).tolist(),
                               np.round(R.T @ ax, 6).tolist())
    return out


def joint_loads(m: mujoco.MjModel, d: mujoco.MjData, dofs: np.ndarray) -> np.ndarray:
    """Torque the joints' mechanical hard stops carry in the current step (Nm), per dof in `dofs`.

    A limit row pushes a joint away from the stop it is at: +f at the lower stop, -f at the upper one.
    """
    out = np.zeros(m.nv)
    n = d.nefc
    for i in np.flatnonzero(d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT):
        j = d.efc_id[i]
        lo, hi = m.jnt_range[j]
        sign = 1.0 if d.qpos[m.jnt_qposadr[j]] < 0.5 * (lo + hi) else -1.0
        out[m.jnt_dofadr[j]] += sign * d.efc_force[i]
    return out[dofs]


def wrap_grasp(cfg: SimConfig, hand_path: Path) -> WrapGrasp:
    """Cached grasp keyframe for this configuration."""
    path = cache_path(cfg, hand_path)
    if path.exists():
        return WrapGrasp.from_dict(json.loads(path.read_text()))
    g = synthesize(cfg, hand_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(g.to_dict(), indent=1))
    return g
