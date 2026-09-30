"""Wrist (F/T body + optional compliance) and a Franka-hand-class parallel gripper with compliant pads.

Frame conventions (hand frame = flange frame):
  z  approach axis, the TCP sits at z = tcp_offset
  y  finger closing axis; the left finger moves along +y as it opens
Each pad is its own body on the finger's inner face. In "explicit" mode it is mounted on a
normal spring (slide along the finger's local y) and two tangential springs, so the tool-in-grasp
resonance and Coulomb slip are both explicit. In "soft_contact" mode pads are rigid and the
compliance lives in a softer contact solref.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from tactile_sim import names
from tactile_sim.config import ArmCfg, GripperCfg
from tactile_sim.model.xmlutil import sub

FINGER_BASE_Z = 0.0584  # Franka hand finger mount height
HAND_OFFSET = 0.02  # flange to hand body (F/T sensor thickness)


def add_wrist_and_hand(link7: ET.Element, arm: ArmCfg, g: GripperCfg, attach_z: float = 0.107) -> None:
    ft = sub(link7, "body", name=names.FT_BODY, pos=(0, 0, attach_z))
    sub(ft, "inertial", pos=(0, 0, 0.01), mass=arm.ft_body_mass, diaginertia=(6e-5, 6e-5, 1e-4))
    sub(ft, "site", name=names.FT_SITE, pos=(0, 0, 0), size=0.005, rgba=(0.1, 0.6, 0.6, 1), group=4)
    sub(ft, "geom", type="cylinder", size=(0.035, 0.01), pos=(0, 0, 0.01), rgba=(0.2, 0.3, 0.3, 1),
        contype=0, conaffinity=0, mass=0, group=2)

    parent = ft
    if arm.flex_mode == "wrist":
        wf = sub(ft, "body", name=names.WRIST_FLEX_BODY, pos=(0, 0, HAND_OFFSET))
        sub(wf, "inertial", pos=(0, 0, 0), mass=0.02, diaginertia=(1e-6, 1e-6, 1e-6))
        for jn, ax in zip(names.WRIST_FLEX_JOINTS[:2], ((1, 0, 0), (0, 1, 0)), strict=True):
            sub(wf, "joint", name=jn, type="hinge", axis=ax, stiffness=arm.wrist_k_rot,
                damping=arm.wrist_d_rot, range=(-0.1, 0.1), limited="true", armature=1e-5)
        sub(wf, "joint", name=names.WRIST_FLEX_JOINTS[2], type="slide", axis=(0, 0, 1),
            stiffness=arm.wrist_k_lin, damping=arm.wrist_d_lin, range=(-0.005, 0.005), limited="true",
            armature=1e-4)
        parent = wf
    elif arm.flex_mode != "rigid":
        raise ValueError(f"unknown flex_mode {arm.flex_mode!r} (expected 'rigid' or 'wrist')")

    hand_z = 0.0 if arm.flex_mode == "wrist" else HAND_OFFSET
    hand = sub(parent, "body", name=names.HAND_BODY, pos=(0, 0, hand_z))
    sub(hand, "inertial", pos=(0, 0, 0.035), mass=g.hand_mass, diaginertia=(0.0027, 0.00052, 0.00265))
    sub(hand, "geom", type="box", size=(0.03, 0.1, 0.03), pos=(0, 0, 0.03), rgba=(0.9, 0.9, 0.9, 1),
        contype=0, conaffinity=0, mass=0, group=2)
    # TCP measured from the flange, whatever sits between the flange and the hand
    sub(hand, "site", name=names.TCP_SITE, pos=(0, 0, g.tcp_offset - HAND_OFFSET),
        size=0.004, rgba=(1, 0, 0, 1), group=4)

    fingers = zip(names.FINGER_BODIES, names.FINGER_JOINTS, names.PAD_BODIES, strict=True)
    for side, (fname, jname, pname) in enumerate(fingers):
        quat = (1, 0, 0, 0) if side == 0 else (0, 0, 0, 1)  # right finger is the left one rotated 180 deg about z
        fb = sub(hand, "body", name=fname, pos=(0, 0, FINGER_BASE_Z), quat=quat)
        sub(fb, "inertial", pos=(0, 0.01, 0.03), mass=g.finger_mass, diaginertia=(1e-5, 1e-5, 5e-6))
        sub(fb, "joint", name=jname, type="slide", axis=(0, 1, 0), range=(0, g.finger_range), limited="true",
            damping=g.finger_joint_damping, armature=0.01)
        sub(fb, "geom", type="box", size=(0.01, 0.006, 0.03), pos=(0, 0.006, 0.03), rgba=(0.2, 0.2, 0.2, 1),
            contype=0, conaffinity=0, mass=0, group=2)
        _add_pad(fb, pname, side, g, z_local=g.tcp_offset - HAND_OFFSET - FINGER_BASE_Z)


def _add_pad(finger: ET.Element, pname: str, side: int, g: GripperCfg, z_local: float) -> None:
    hx, hy, hz = g.pad_half
    pad = sub(finger, "body", name=pname, pos=(0, 0, z_local))
    sub(pad, "inertial", pos=(0, -hy, 0), mass=g.pad_mass, diaginertia=(1e-6, 1e-6, 1e-6))
    if g.pad_mode == "explicit":
        jn = names.PAD_JOINTS[pname]
        # normal: positive = compression into the finger
        sub(pad, "joint", name=jn[0], type="slide", axis=(0, 1, 0), stiffness=g.pad_k_n, damping=g.pad_d_n,
            range=(-0.0005, g.pad_travel_n), limited="true", armature=1e-4)
        sub(pad, "joint", name=jn[1], type="slide", axis=(1, 0, 0), stiffness=g.pad_k_t, damping=g.pad_d_t,
            range=(-g.pad_travel_t, g.pad_travel_t), limited="true", armature=1e-4)
        sub(pad, "joint", name=jn[2], type="slide", axis=(0, 0, 1), stiffness=g.pad_k_t, damping=g.pad_d_t,
            range=(-g.pad_travel_t, g.pad_travel_t), limited="true", armature=1e-4)
    elif g.pad_mode != "soft_contact":
        raise ValueError(f"unknown pad_mode {g.pad_mode!r}")
    sub(pad, "geom", name=names.PAD_GEOMS[side], type="box", size=(hx, hy, hz), pos=(0, -hy, 0),
        rgba=(0.15, 0.15, 0.6, 1), contype=0, conaffinity=0, mass=0)
    sub(pad, "site", name=names.PAD_IMU_SITES[side], pos=(0, -hy, 0), size=0.002, group=4)
    nr, nc = names.TAXEL_GRID
    for r in range(nr):  # rows along the pad's x (handle axis)
        for c in range(nc):  # columns along the pad's z
            x = -hx + (2 * r + 1) * hx / nr
            z = -hz + (2 * c + 1) * hz / nc
            sub(pad, "site", name=taxel_site(side, r, c), type="box", size=(hx / nr, hy + 0.003, hz / nc),
                pos=(x, -hy, z), rgba=(0.3, 0.8, 0.3, 0.2), group=5)


def taxel_site(side: int, r: int, c: int) -> str:
    return f"taxel_{'LR'[side]}_{r}{c}"


def add_grip_actuation(root: ET.Element, g: GripperCfg) -> None:
    tendon = root.find("tendon")
    if tendon is None:
        tendon = sub(root, "tendon")
    fx = sub(tendon, "fixed", name=names.GRIP_TENDON)
    for j in names.FINGER_JOINTS:
        sub(fx, "joint", joint=j, coef=1)
    eq = root.find("equality")
    if eq is None:
        eq = sub(root, "equality")
    sub(eq, "joint", name="finger_mirror", joint1=names.FINGER_JOINTS[0], joint2=names.FINGER_JOINTS[1],
        polycoef=(0, 1, 0, 0, 0), solref=(0.002, 1))
    act = root.find("actuator")
    # ctrl is the opening force per finger; the grip controller writes ctrl = -F_grip
    sub(act, "motor", name=names.GRIP_MOTOR, tendon=names.GRIP_TENDON,
        ctrlrange=(-g.grip_force_max, g.grip_force_max), ctrllimited="true")
