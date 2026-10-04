"""WUJI Hand 2 (Beta 2, right hand with WUJI's mount) on the FR3 flange, holding the hammer in a power wrap.

The vendor MJCF (wuji-technology/wuji-description, MIT) is imported as-is for kinematics, inertials,
joint ranges, armature, visual meshes and convex-hull collision meshes. Changes:

- Its PD position actuators are replaced by torque motors whose range is the vendor's per-joint
  `actuatorfrcrange` (MCP flexion 2.0 Nm, MCP abduction 0.2 Nm, PIP/DIP 0.3 Nm, thumb CMC 0.6 Nm, thumb
  MCP/IP 0.3 Nm). The hand's 1 kHz joint law (MIT mode: torque + PD) runs in tactile_sim.control.hand.
- All contacts come from explicit pairs: every hand link against short convex slices of the hammer
  handle (the handle is curved, one hull would bridge the curve), plus thumb-on-finger and
  finger-on-palm pairs so the wrap cannot pass through itself.
- Taxel patches: either the TaxelScan Rev3 skins (`patch_layout="taxelscan"`, tactile_sim.model.hands.taxel_layout:
  a 128-taxel palm sheet and 32-taxel patches on every finger's distal pad and middle segment, each taxel on the
  link's curved palmar surface), or flat 8 x 8 frames where the grasp loads the palm and thumb. The pressure model
  spreads each contact's force over the taxels near it.

Hand frame (= WUJI mount frame): fingers extend along -z, the palm faces +y, the thumb is on +x.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tactile_sim import names
from tactile_sim.config import GripperCfg
from tactile_sim.model.xmlutil import sub

PALM_BODY = "r_wrist"
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
# TCP (= hammer grasp frame) axes in the hand frame: handle along -x (head on the thumb side), striking
# face toward the fingertips (-z), hammer z along the palm normal
R_HAND_TCP = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
PALM_SURFACE_Y = 0.0173  # palm hull's +y face, hand frame
PALM_PATCH_NOMINAL = (-0.003, PALM_SURFACE_Y - 0.00025, -0.085 + 0.0285)  # palm body frame, under the handle


@dataclass(frozen=True)
class PatchSpec:
    """A taxel patch: a frame on a body. Rows along u, columns along v, the normal u x v points out of the
    skin toward the tool."""

    name: str
    body: str
    center: tuple[float, float, float]  # body frame
    u: tuple[float, float, float]
    v: tuple[float, float, float]
    half: tuple[float, float]  # half extent along u, v


PATCH_HALF = {"palm": (0.015, 0.015), "finger": (0.008, 0.008)}  # 30 x 30 mm palm, 16 x 16 mm phalanx
LAYOUTS = {
    # name -> (patch name, body selector); "thumb" = the thumb segment carrying the most load
    "palm_thumb": (("palm", PALM_BODY), ("thumb", "thumb")),
    "palm": (("palm", PALM_BODY),),
    "palm_thumb_index": (("palm", PALM_BODY), ("thumb", "thumb"), ("index", "index_finger")),
}


def patches_from_grasp(contact_frames: dict, contact_force: dict, layout: str) -> list[PatchSpec]:
    """Put each patch where the grasp loads that part of the hand: centred on the force-weighted contact
    point, facing the tool, rows along the handle."""
    out = []
    for pname, sel in LAYOUTS[layout]:
        if sel == PALM_BODY:
            cands = [PALM_BODY] if PALM_BODY in contact_frames else []
        else:
            cands = [b for b in contact_frames if sel in b]
        if not cands and sel == PALM_BODY:
            # the handle does not bear on the palm in this grasp: put the patch under the grasp point anyway
            c, n, ax, body = np.array(PALM_PATCH_NOMINAL), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), sel
        elif not cands:
            raise ValueError(f"grasp has no contact on {sel!r}; cannot place the {pname!r} patch")
        else:
            body = max(cands, key=lambda b: contact_force.get(b, 0.0))
            c, n, ax = (np.asarray(x, dtype=float) for x in contact_frames[body])
        u = ax - (ax @ n) * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        half = PATCH_HALF["palm" if body == PALM_BODY else "finger"]
        out.append(PatchSpec(pname, body, tuple(c), tuple(u), tuple(v), half))
    return out


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def mount_quat(yaw: float) -> tuple[float, float, float, float]:
    """Hand frame in the flange frame: fingers (-z) along the flange's +z, then `yaw` about the flange z."""
    flip = (0.0, 1.0, 0.0, 0.0)  # pi about x
    qy = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    return _quat_mul(qy, flip)


def tcp_quat() -> tuple[float, float, float, float]:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_HAND_TCP.reshape(-1))
    return tuple(float(x) for x in q)


def load_hand_xml(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def add_wuji_hand(parent: ET.Element, scene_root: ET.Element, g: GripperCfg, hand_xml_path: Path,
                  hand_pos: tuple[float, float, float], tcp_pos=None, tcp_q=None,
                  patches: list[PatchSpec] = (), body_name: str = names.HAND_BODY, with_tcp: bool = True,
                  hold_kp: float | None = None) -> dict:
    """Attach the hand under `parent` (the F/T body or wrist-flex body). Returns build info.

    The working hand gets torque motors (its joint law runs in tactile_sim.control.hand) and the tool frame.
    A second, idle hand (`with_tcp=False`, `hold_kp` set) gets position servos at `hold_kp` Nm/rad, still
    limited to each joint's rating, that hold whatever pose they are given."""
    hx = load_hand_xml(hand_xml_path)
    meshdir = (hand_xml_path.parent / hx.find("compiler").get("meshdir", ".")).resolve()
    default_arm = float(hx.find("default").find("joint").get("armature", "0.0002"))

    asset = scene_root.find("asset")
    if asset is None:
        asset = sub(scene_root, "asset")
    for mesh in hx.find("asset").findall("mesh"):
        sub(asset, "mesh", name=f"wuji_{mesh.get('name')}", file=str(meshdir / mesh.get("file")))

    src_root = hx.find("worldbody").find("body")  # r_mount
    hand = sub(parent, "body", name=body_name, pos=hand_pos, quat=mount_quat(g.mount_yaw))
    col_geoms: dict[str, list[str]] = {}
    joints: list[tuple[str, float]] = []

    def copy_body(src: ET.Element, dst: ET.Element, bname: str) -> None:
        for child in src:
            if child.tag == "inertial":
                dst.append(ET.fromstring(ET.tostring(child)))
            elif child.tag == "joint":
                j = ET.fromstring(ET.tostring(child))
                j.set("armature", j.get("armature", f"{default_arm:.9g}"))
                j.set("damping", f"{g.hand_joint_damping:.9g}")
                j.set("frictionloss", f"{g.hand_joint_friction:.9g}")
                j.set("solreflimit", "0.002 1")  # mechanical hard stops, not the 20 ms default
                lim = float(j.get("actuatorfrcrange").split()[1])
                joints.append((j.get("name"), lim))
                dst.append(j)
            elif child.tag == "geom":
                gm = ET.fromstring(ET.tostring(child))
                gm.set("mesh", f"wuji_{gm.get('mesh')}")
                gm.set("contype", "0")
                gm.set("conaffinity", "0")
                if gm.get("group") == "2":  # vendor collision hull
                    nm = f"wuji_col_{bname}"
                    gm.set("name", nm)
                    gm.set("group", "3")
                    gm.set("density", "0")
                    col_geoms.setdefault(bname, []).append(nm)
                else:
                    gm.set("group", "2")
                dst.append(gm)
            elif child.tag == "site":
                s = ET.fromstring(ET.tostring(child))
                s.set("group", "4")
                dst.append(s)
            elif child.tag == "body":
                nb = sub(dst, "body", name=child.get("name"), pos=child.get("pos", "0 0 0"),
                         quat=child.get("quat", "1 0 0 0"))
                copy_body(child, nb, child.get("name"))

    copy_body(src_root, hand, src_root.get("name"))
    act = scene_root.find("actuator")
    if not with_tcp:
        for jn, lim in joints:
            sub(act, "position", name=f"m_{jn}", joint=jn, kp=hold_kp, kv=0.05 * hold_kp,
                forcerange=(-lim, lim), forcelimited="true")
        return {"joints": joints, "col_geoms": col_geoms, "patches": []}
    # tool frame: the seated hammer pose from the grasp keyframe (nominal placement before synthesis)
    sub(hand, "site", name=names.TCP_SITE, pos=g.wrap_tcp if tcp_pos is None else tcp_pos,
        quat=tcp_quat() if tcp_q is None else tcp_q, size=0.004, rgba=(1, 0, 0, 1), group=4)

    # taxel patch frames; each doubles as the patch accelerometer's site
    import mujoco

    bodies = {b.get("name"): b for b in hand.iter("body")}
    for p in patches:
        R = np.column_stack([p.u, p.v, np.cross(p.u, p.v)])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.reshape(-1))
        sub(bodies[p.body], "site", name=f"patch_{p.name}", pos=p.center, quat=q,
            size=(p.half[0], p.half[1], 0.0005), type="box", rgba=(0.2, 0.5, 0.9, 0.6), group=4)
        for i, x in enumerate(getattr(p, "pos", ())):  # conforming skins: one marker per taxel
            sub(bodies[p.body], "site", name=f"taxel_{p.name}_{i}", pos=x, size=0.0006, rgba=(0.9, 0.6, 0.1, 1),
                group=5)

    for jn, lim in joints:
        sub(act, "motor", name=f"m_{jn}", joint=jn, ctrlrange=(-lim, lim), ctrllimited="true")
    return {"joints": joints, "col_geoms": col_geoms, "patches": list(patches)}


def contact_pairs(contact: ET.Element, col_geoms: dict[str, list[str]], handle_geoms: list[str],
                  g: GripperCfg) -> None:
    skin = dict(condim=4, friction=(g.hand_friction, g.hand_friction, g.pad_torsion, 0.0001, 0.0001),
                solref=g.hand_solref, solimp=g.pad_solimp)
    for bname, geoms in col_geoms.items():
        if bname.endswith("_mount"):
            continue
        for gm in geoms:
            for hg in handle_geoms:
                sub(contact, "pair", geom1=gm, geom2=hg, **skin)
    # the wrap must not pass through itself: thumb on the index/middle fingers, fingertips on the palm
    thumb = [gm for b, gs in col_geoms.items() if "thumb_middle" in b or "thumb_distal" in b for gm in gs]
    fingers = [gm for b, gs in col_geoms.items() if ("index" in b or "middle_finger" in b) and
               ("middle" in b or "distal" in b) for gm in gs]
    tips = [gm for b, gs in col_geoms.items() if b.endswith("_distal") and "thumb" not in b for gm in gs]
    palm = col_geoms.get(PALM_BODY, [])
    self_contact = dict(condim=3, friction=(0.8, 0.8, 0.005, 0.0001, 0.0001), solref=g.hand_solref)
    for a in thumb:
        for b in fingers:
            sub(contact, "pair", geom1=a, geom2=b, **self_contact)
    for a in tips:
        for b in palm:
            sub(contact, "pair", geom1=a, geom2=b, **self_contact)
