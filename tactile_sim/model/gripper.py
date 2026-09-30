"""Wrist (F/T body + optional compliance) and the Franka Hand with compliant fingertip pads.

The hand is Menagerie's Franka Hand (franka_emika_panda/hand.xml): its meshes, materials, inertials and
finger joints are imported as-is and attached to the FR3 flange the way Menagerie's panda.xml does
(rotated -45 deg about the flange z). Without the Menagerie cache a box stand-in with the same inertials,
joints and pad placement is built instead.

Only the contact surface is modified: the real hand's rubber fingertip pad (17 x 17 mm, Menagerie's
fingertip_pad_collision_1) becomes a separate pad body on a normal spring and two tangential springs, so
tool-in-grasp ringing and Coulomb slip are explicit. Its inner face sits exactly where the real pad's is.

Frame conventions (hand frame): z = approach axis, y = finger closing axis (left finger opens along +y).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from tactile_sim import names
from tactile_sim.assets import ensure_menagerie, hand_xml_path
from tactile_sim.config import ArmCfg, GripperCfg
from tactile_sim.model.xmlutil import sub

FINGER_BASE_Z = 0.0584  # Franka Hand: finger mount height above the hand base
PAD_Z = 0.0445  # real fingertip pad centre along the finger
PAD_FACE_Y = 0.0015  # real fingertip pad inner face, finger frame (fingers closed: 3 mm gap)
HAND_QUAT = (0.9238795, 0.0, 0.0, -0.3826834)  # Menagerie panda.xml hand mount (-45 deg about z)
HAND_OFFSET = 0.02  # F/T sensor body between the flange and the hand
HAND_INERTIAL = {"pos": (-0.01, 0.0, 0.03), "mass": 0.73, "diaginertia": (0.001, 0.0025, 0.0017)}
FINGER_INERTIAL = {"pos": (0.0, 0.0, 0.0), "mass": 0.015, "diaginertia": (2.375e-6, 2.375e-6, 7.5e-7)}


def finger_q_touch(g: GripperCfg, radius: float) -> float:
    """Finger opening at which the pad faces just touch a cylinder of `radius` centred between them."""
    return radius - PAD_FACE_Y


def _menagerie_hand() -> ET.Element | None:
    root = ensure_menagerie(download=False)
    if root is None:
        return None
    return ET.parse(hand_xml_path(root)).getroot()


def _import_hand_assets(scene_root: ET.Element, hand_root: ET.Element, hand_dir: Path) -> dict[str, str]:
    """Copy the hand's visual meshes (absolute paths) and materials (prefixed) into the scene."""
    asset = scene_root.find("asset")
    if asset is None:
        asset = sub(scene_root, "asset")
    rename = {}
    for mat in hand_root.find("asset").findall("material"):
        new = f"fh_{mat.get('name')}"
        rename[mat.get("name")] = new
        sub(asset, "material", name=new, rgba=mat.get("rgba"), specular="0.5", shininess="0.25")
    meshdir = hand_dir / hand_root.find("compiler").get("meshdir", ".")
    for mesh in hand_root.find("asset").findall("mesh"):
        f = mesh.get("file")
        if f.endswith(".stl"):
            continue  # collision mesh: all contacts come from explicit pairs
        sub(asset, "mesh", name=mesh.get("name") or Path(f).stem, file=str((meshdir / f).resolve()))
    return rename


def _visual_meshes(body_xml: ET.Element) -> list[tuple[str, str]]:
    """(mesh, material) of the visual geoms of a hand.xml body (non-recursive)."""
    return [(gm.get("mesh"), gm.get("material")) for gm in body_xml.findall("geom")
            if gm.get("class") == "visual" and gm.get("mesh")]


def add_wrist_and_hand(link7: ET.Element, arm: ArmCfg, g: GripperCfg, scene_root: ET.Element | None = None,
                       attach_z: float = 0.107) -> str:
    """Build F/T body -> (wrist compliance) -> Franka Hand under link 7. Returns the hand source."""
    ft = sub(link7, "body", name=names.FT_BODY, pos=(0, 0, attach_z))
    sub(ft, "inertial", pos=(0, 0, 0.01), mass=arm.ft_body_mass, diaginertia=(6e-5, 6e-5, 1e-4))
    sub(ft, "site", name=names.FT_SITE, pos=(0, 0, 0), size=0.005, rgba=(0.1, 0.6, 0.6, 1), group=4)
    sub(ft, "geom", type="cylinder", size=(0.04, 0.01), pos=(0, 0, 0.01), rgba=(0.25, 0.3, 0.3, 1),
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
    if g.hand_source not in ("auto", "franka", "box"):
        raise ValueError(f"unknown hand_source {g.hand_source!r}")

    hand_xml = _menagerie_hand() if (scene_root is not None and g.hand_source != "box") else None
    if g.hand_source == "franka" and hand_xml is None:
        raise FileNotFoundError("Menagerie Franka Hand not cached; run `python -m tactile_sim.assets.fetch_menagerie`")
    rename = {}
    if hand_xml is not None:
        rename = _import_hand_assets(scene_root, hand_xml, hand_xml_path().parent)

    hand_z = 0.0 if arm.flex_mode == "wrist" else HAND_OFFSET
    hand = sub(parent, "body", name=names.HAND_BODY, pos=(0, 0, hand_z), quat=HAND_QUAT)
    sub(hand, "inertial", **HAND_INERTIAL)
    if hand_xml is not None:
        hb = next(b for b in hand_xml.iter("body") if b.get("name") == "hand")
        for mesh, mat in _visual_meshes(hb):
            sub(hand, "geom", type="mesh", mesh=mesh, material=rename.get(mat, mat), contype=0, conaffinity=0,
                group=2, mass=0)
    else:
        sub(hand, "geom", type="box", size=(0.03, 0.1, 0.03), pos=(0, 0, 0.03), rgba=(0.9, 0.9, 0.9, 1),
            contype=0, conaffinity=0, mass=0, group=2)
    sub(hand, "site", name=names.TCP_SITE, pos=(0, 0, g.tcp_offset), size=0.004, rgba=(1, 0, 0, 1), group=4)

    fingers = zip(names.FINGER_BODIES, names.FINGER_JOINTS, names.PAD_BODIES, strict=True)
    for side, (fname, jname, pname) in enumerate(fingers):
        quat = (1, 0, 0, 0) if side == 0 else (0, 0, 0, 1)  # right finger = left rotated 180 deg about z
        fb = sub(hand, "body", name=fname, pos=(0, 0, FINGER_BASE_Z), quat=quat)
        sub(fb, "inertial", **FINGER_INERTIAL)
        sub(fb, "joint", name=jname, type="slide", axis=(0, 1, 0), range=(0, g.finger_range), limited="true",
            damping=g.finger_joint_damping, armature=g.finger_armature)
        if hand_xml is not None:
            fx = next(b for b in hand_xml.iter("body") if b.get("name") == "left_finger")
            for mesh, mat in _visual_meshes(fx):
                sub(fb, "geom", type="mesh", mesh=mesh, material=rename.get(mat, mat), contype=0, conaffinity=0,
                    group=2, mass=0)
        else:
            sub(fb, "geom", type="box", size=(0.0095, 0.006, 0.027), pos=(0, 0.0075, 0.027),
                rgba=(0.2, 0.2, 0.2, 1), contype=0, conaffinity=0, mass=0, group=2)
        _add_pad(fb, pname, side, g)
    return "franka" if hand_xml is not None else "box"


def _add_pad(finger: ET.Element, pname: str, side: int, g: GripperCfg) -> None:
    hx, hy, hz = g.pad_half
    # pad body origin = back of the compliant layer; its inner face lands on the real pad's inner face
    pad = sub(finger, "body", name=pname, pos=(0, PAD_FACE_Y + 2 * hy, PAD_Z))
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
        rgba=(0.12, 0.12, 0.14, 1), contype=0, conaffinity=0, mass=0)
    sub(pad, "site", name=names.PAD_IMU_SITES[side], pos=(0, -hy, 0), size=0.002, group=4)
    nr, nc = names.TAXEL_GRID
    for r in range(nr):  # rows along the pad's x (across the finger, = along the handle)
        for c in range(nc):  # columns along the pad's z (along the finger)
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
        polycoef=(0, 1, 0, 0, 0), solimp=(0.95, 0.99, 0.001), solref=(0.005, 1))
    act = root.find("actuator")
    # force-controlled grasp (the Franka Hand's "grasp" command): ctrl is the opening force per finger,
    # so the grip loop writes ctrl = -F_grip. Limit: 140 N peak (70 N continuous) per the Franka Hand spec.
    sub(act, "motor", name=names.GRIP_MOTOR, tendon=names.GRIP_TENDON,
        ctrlrange=(-g.grip_force_max, g.grip_force_max), ctrllimited="true")
