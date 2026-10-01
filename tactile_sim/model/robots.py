"""Arm descriptions: which joints strike, where the wrist mounts, the arm's ratings and its control interface.

- FR3 (Franka): 1 kHz joint torque control through libfranka (L1 = Cartesian impedance in torque).
- Vega U and Vega-1P (Dexmate), right arm: joint position targets streamed at 100 Hz through dexcontrol,
  tracked by the drives' own PD servos (L1 = differential IK + gravity-droop compensation on the host; see
  tactile_sim.control.position_l1). Ratings from Dexmate's URDF (dexmate-ai/dexmate-urdf).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tactile_sim import names
from tactile_sim.config import ArmCfg
from tactile_sim.model.xmlutil import sub

VEGA_ARM = tuple(f"R_arm_j{i}" for i in range(1, 8))
VEGA_LEFT_ARM = tuple(f"L_arm_j{i}" for i in range(1, 8))
VEGA_TORSO = ("torso_j1", "torso_j2", "torso_j3")
VEGA_HEAD = ("head_j1", "head_j2", "head_j3")
VEGA_ROBOTS = ("vega_1u", "vega_1p")


@dataclass(frozen=True)
class ArmSpec:
    robot: str
    joints: tuple[str, ...]  # the striking arm, base to wrist
    motors: tuple[str, ...]
    flange_body: str
    attach_z: float  # wrist F/T body offset along the flange z
    torque: np.ndarray  # rated joint torque (Nm)
    velocity: np.ndarray  # joint velocity limit (rad/s)
    torque_rate: float | None  # Nm/s, None = not specified by the vendor
    payload: float  # kg beyond the flange
    interface: str  # "torque" or "position"


def arm_spec(cfg: ArmCfg) -> ArmSpec:
    if cfg.robot == "fr3":
        from tactile_sim.limits import FR3_PAYLOAD, FR3_TORQUE, FR3_TORQUE_RATE, FR3_VELOCITY

        return ArmSpec("fr3", tuple(names.ARM_JOINTS), tuple(names.ARM_MOTORS), "fr3_link7", 0.107, FR3_TORQUE,
                       FR3_VELOCITY, FR3_TORQUE_RATE, FR3_PAYLOAD, "torque")
    if cfg.robot in VEGA_ROBOTS:
        # URDF effort / velocity limits (the Vega U and 1P share the arm); payload: Dexmate quotes 10 lb (4.5 kg)
        # per arm at any pose (older material says 15 kg); the conservative figure is used
        return ArmSpec(cfg.robot, VEGA_ARM, tuple(f"m_{j}" for j in VEGA_ARM), "R_arm_l8", 0.0,
                       np.array([150.0, 150.0, 80.0, 80.0, 25.0, 25.0, 25.0]),
                       np.array([2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7]), None, 4.5, "position")
    raise ValueError(f"unknown robot {cfg.robot!r} (expected 'fr3', 'vega_1u' or 'vega_1p')")


def servo_gains(cfg: ArmCfg) -> tuple[np.ndarray, np.ndarray]:
    """Arm servo stiffness and damping after dexcontrol's P multiplier (which Dexmate limits to [0.1, 4])."""
    if not 0.1 <= cfg.servo_p_mult <= 4.0:
        raise ValueError(f"servo_p_mult {cfg.servo_p_mult} outside dexcontrol's allowed [0.1, 4]")
    kp = np.asarray(cfg.servo_kp, float) * cfg.servo_p_mult
    kd = np.asarray(cfg.servo_kd, float) * np.sqrt(cfg.servo_p_mult)  # keep the damping ratio
    return kp, kd


def vega_hold_pose(cfg: ArmCfg) -> dict[str, float]:
    """Held joints (everything but the striking arm): head, left arm, and the Vega-1P's torso."""
    pose = dict(zip(VEGA_TORSO, cfg.torso_pose, strict=True)) if cfg.robot == "vega_1p" else {}
    pose.update(zip(VEGA_HEAD, cfg.head_pose, strict=True))
    pose.update(zip(VEGA_LEFT_ARM, cfg.left_arm_pose, strict=True))
    return pose


def load_vega_tree(cfg: ArmCfg) -> tuple[ET.Element, str]:
    """Dexmate's Vega URDF compiled by MuJoCo into MJCF, base fixed to the floor, every joint on a torque-limited
    position servo. Visual geoms are Dexmate's collision meshes (OBJ, or GLB converted to OBJ); nothing
    collides (contacts come from explicit pairs).

    Vega U: a fixed pedestal with a lift (0-0.4 m) and a torso flip (0-1 rad) under the head and arms. Dexmate's
    Vega U profile drives only the upper body, so the lift and flip are set before a run (`vega_lift`,
    `vega_flip`) and become fixed joints. Vega-1P: the wheeled base is locked; its three torso joints hold a
    pose on servos."""
    import os
    import tempfile

    import mujoco

    from tactile_sim.assets.fetch_hands import hand_xml

    path = hand_xml(cfg.robot)
    if path is None:
        raise FileNotFoundError(f"{cfg.robot} not cached; run `python -m tactile_sim.assets.fetch_hands {cfg.robot}`")
    urdf = ET.parse(path).getroot()
    efforts = {}
    for link in urdf.findall("link"):
        for v in link.findall("visual"):
            link.remove(v)
        for mesh in link.iter("mesh"):
            f = mesh.get("filename")
            if f.lower().endswith(".glb"):
                from tactile_sim.assets.glb import glb_to_obj

                src = (Path(path).parent / f).resolve()
                dst = src.with_suffix(".obj")
                if not dst.exists():
                    glb_to_obj(src, dst)
                mesh.set("filename", str(dst))
    for j in urdf.findall("joint"):
        n = j.get("name")
        if "wheel" in n:
            j.set("type", "fixed")
        elif n in ("Lift", "torso_flip"):  # Vega U set-up axes: fixed at their configured positions
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz").split()]
            rpy = [float(v) for v in o.get("rpy").split()]
            ax = [float(v) for v in j.find("axis").get("xyz").split()]
            if n == "Lift":
                if not 0.0 <= cfg.vega_lift <= 0.4:
                    raise ValueError(f"vega_lift {cfg.vega_lift} outside the lift's 0-0.4 m")
                xyz = [x + cfg.vega_lift * a for x, a in zip(xyz, ax, strict=True)]  # axis is z in the joint frame
                # the joint frame is rotated by rpy about z only, so a z axis stays z
            else:
                if not 0.0 <= cfg.vega_flip <= 1.0:
                    raise ValueError(f"vega_flip {cfg.vega_flip} outside the torso flip's 0-1 rad")
                rpy = [r + cfg.vega_flip * a for r, a in zip(rpy, ax, strict=True)]  # axis is x, rpy[1:] are 0
            o.set("xyz", " ".join(f"{v:.9g}" for v in xyz))
            o.set("rpy", " ".join(f"{v:.9g}" for v in rpy))
            j.set("type", "fixed")
        lim = j.find("limit")
        if lim is not None and lim.get("effort"):
            efforts[j.get("name")] = float(lim.get("effort"))
    mj = ET.SubElement(urdf, "mujoco")
    ET.SubElement(mj, "compiler", meshdir=str(Path(path).parent), discardvisual="true", fusestatic="false",
                  strippath="false", balanceinertia="true")
    m = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding="unicode"))
    fd, tmp = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    try:
        mujoco.mj_saveLastXML(tmp, m)
        root = ET.parse(tmp).getroot()
    finally:
        os.unlink(tmp)
    root.set("model", cfg.robot)
    wb = root.find("worldbody")
    base = wb.find("body")
    base.set("pos", f"0 0 {_floor_offset(m):.6f}")
    for g in root.iter("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")
        g.set("group", "2")
        g.set("rgba", "0.86 0.87 0.88 1")
    arm_idx = {j: k for k, j in enumerate(VEGA_ARM)}
    left_idx = {j: k for k, j in enumerate(VEGA_LEFT_ARM)}
    kp_arm, kd_arm = servo_gains(cfg)
    act = sub(root, "actuator")
    for j in root.iter("joint"):
        n = j.get("name")
        k = arm_idx.get(n, left_idx.get(n))
        if k is not None:
            j.set("armature", f"{cfg.vega_armature[k]:.9g}")
            j.set("damping", f"{cfg.vega_damping:.9g}")
            kp, kv = kp_arm[k], kd_arm[k]
        else:  # torso and head: stiff, slow axes that only hold a pose here
            j.set("armature", "0.5")
            j.set("damping", "5")
            kp, kv = 5000.0, 200.0
        e = efforts[n]
        sub(act, "position", name=f"m_{n}", joint=n, kp=kp, kv=kv, forcerange=(-e, e), forcelimited="true")
    return root, cfg.robot


def _floor_offset(m) -> float:
    """Height to raise the base so its lowest mesh point touches the floor at the zero pose."""
    import mujoco

    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    lo = np.inf
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        lo = min(lo, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()))
    return -lo if np.isfinite(lo) else 0.0
