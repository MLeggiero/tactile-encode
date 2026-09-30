"""Hammer MJCF. The hammer's body frame is the grasp frame: origin on the handle axis at the grasp
centre, x along the handle (head at -grip_from_head), y along the strike axis (face at -y)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from tactile_sim import names
from tactile_sim.config import HammerCfg
from tactile_sim.model.xmlutil import sub


def face_offset(h: HammerCfg) -> tuple[float, float, float]:
    """Hammer face centre in the hammer (grasp) frame."""
    return (-h.grip_from_head, -h.head_half_len, 0.0)


def add_hammer(worldbody: ET.Element, h: HammerCfg) -> None:
    xh = -h.grip_from_head
    b = sub(worldbody, "body", name=names.HAMMER_BODY, pos=(0, 0, 1.0))
    sub(b, "freejoint", name="hammer_free")
    sub(b, "geom", name=names.HAMMER_HEAD_GEOM, type="cylinder", size=h.head_radius,
        fromto=(xh, -h.head_half_len, 0, xh, h.head_half_len, 0), mass=h.head_mass,
        rgba=(0.45, 0.45, 0.5, 1), contype=0, conaffinity=0)
    x_end = h.handle_len - h.grip_from_head
    sub(b, "geom", name=names.HAMMER_HANDLE_GEOM, type="capsule", size=h.handle_radius,
        fromto=(xh + h.head_radius, 0, 0, x_end, 0, 0), mass=h.handle_mass,
        rgba=(0.6, 0.4, 0.2, 1), contype=0, conaffinity=0)
    fx, fy, fz = face_offset(h)
    sub(b, "site", name=names.HAMMER_FACE_SITE, pos=(fx, fy, fz), size=0.003, group=4)
    sub(b, "site", name=names.HAMMER_REF_SITE, pos=(0, 0, 0), size=0.003, group=4)
    sub(b, "site", name=names.HAMMER_IMU_SITE, pos=(xh, 0, 0), size=0.003, group=4)


def add_grasp_weld(root: ET.Element) -> None:
    eq = root.find("equality")
    if eq is None:
        eq = sub(root, "equality")
    sub(eq, "weld", name=names.GRASP_WELD, body1=names.HAND_BODY, body2=names.HAMMER_BODY, active="false",
        solref=(0.005, 1))
