"""Held tools other than the hammer: a hand saw and an inline cordless driver.

Every held tool uses the hammer's names for the parts the rest of the testbed looks up (names.HAMMER_BODY for the
tool body with its free joint "hammer_free", names.HAMMER_HANDLE_GEOM for the gripped handle the pads touch,
names.HAMMER_REF_SITE at the grasp frame, names.HAMMER_FACE_SITE at the working point, names.HAMMER_IMU_SITE for the
tool accelerometer), so the arm, hand, grasp settle, sensors and slip metrics work unchanged. The tool body frame is
the grasp frame. All tool geoms except the handle are visual: the saw's and driver's interaction with the work is
modelled by their plants (tactile_sim.model.plants.saw / .drill) as applied wrenches.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from tactile_sim import names
from tactile_sim.config import DrillCfg, SawCfg
from tactile_sim.model.xmlutil import sub

STEEL = (0.70, 0.72, 0.75, 1.0)
GRIP = (0.20, 0.22, 0.24, 1.0)
WOOD = (0.74, 0.55, 0.34, 1.0)
BODY = (0.85, 0.45, 0.12, 1.0)

SAW_TEETH_A = "saw_teeth_a"
SAW_TEETH_B = "saw_teeth_b"
BIT_TIP_SITE = "bit_tip"


def _tool_body(worldbody: ET.Element) -> ET.Element:
    b = sub(worldbody, "body", name=names.HAMMER_BODY, pos=(0, 0, 1.0))
    sub(b, "freejoint", name="hammer_free")
    sub(b, "site", name=names.HAMMER_REF_SITE, pos=(0, 0, 0), size=0.003, group=4)
    return b


def add_saw(worldbody: ET.Element, s: SawCfg) -> str:
    b = _tool_body(worldbody)
    h = 0.5 * s.handle_len
    sub(b, "geom", name=names.HAMMER_HANDLE_GEOM, type="capsule", size=s.handle_radius,
        fromto=(-h, 0, 0, h, 0, 0), mass=s.handle_mass, rgba=WOOD, contype=0, conaffinity=0)
    x0, x1 = s.blade_x
    yt = s.blade_depth  # the teeth lie along +y (the fingers' closing axis), the blade in the tool x-y plane
    yc = yt - 0.5 * s.blade_height
    sub(b, "geom", name="saw_blade", type="box", size=(0.5 * (x1 - x0), 0.5 * s.blade_height, 0.5 * s.blade_thickness),
        pos=(0.5 * (x0 + x1), yc, 0), mass=s.blade_mass, rgba=STEEL, contype=0, conaffinity=0)
    # the back of the handle joining the blade's spine, behind and in front of the fingers (visual)
    web = 0.5 * (yt - s.blade_height)
    for xw in (-h - 0.012, h + 0.012):
        sub(b, "geom", type="box", size=(0.012, 0.5 * (yt - s.blade_height) + s.handle_radius, 0.006),
            pos=(xw, web, 0), mass=0, rgba=WOOD, contype=0, conaffinity=0)
    sub(b, "site", name=SAW_TEETH_A, pos=(x0, yt, 0), size=0.002, group=4)
    sub(b, "site", name=SAW_TEETH_B, pos=(x1, yt, 0), size=0.002, group=4)
    sub(b, "site", name=names.HAMMER_FACE_SITE, pos=(0.5 * (x0 + x1), yt, 0), size=0.003, group=4)
    sub(b, "site", name=names.HAMMER_IMU_SITE, pos=(0.5 * (x0 + x1), yc, 0), size=0.003, group=4)
    return "saw"


def add_driver(worldbody: ET.Element, dc: DrillCfg) -> str:
    b = _tool_body(worldbody)
    z0, z1 = dc.body_span  # the body runs along the bit (tool z, the hand's approach axis)
    sub(b, "geom", name=names.HAMMER_HANDLE_GEOM, type="capsule", size=dc.handle_radius,
        fromto=(0, 0, z0, 0, 0, z1), mass=dc.body_mass, rgba=BODY, contype=0, conaffinity=0)
    sub(b, "geom", type="cylinder", size=(0.5 * dc.handle_radius, 0.012), pos=(0, 0, z1 + dc.handle_radius),
        mass=0, rgba=GRIP, contype=0, conaffinity=0)
    tip = bit_tip_local(dc)[2]
    sub(b, "geom", name="driver_bit", type="capsule", size=0.003,
        fromto=(0, 0, z1 + dc.handle_radius + 0.012, 0, 0, tip - 0.003), mass=0, rgba=STEEL, contype=0, conaffinity=0)
    sub(b, "site", name=BIT_TIP_SITE, pos=(0, 0, tip), size=0.002, group=4)
    sub(b, "site", name=names.HAMMER_FACE_SITE, pos=(0, 0, tip), size=0.003, group=4)
    sub(b, "site", name=names.HAMMER_IMU_SITE, pos=(0, 0, 0.5 * (z0 + z1)), size=0.003, group=4)
    return "driver"


def saw_teeth_local(s: SawCfg) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (s.blade_x[0], s.blade_depth, 0.0), (s.blade_x[1], s.blade_depth, 0.0)


def bit_tip_local(dc: DrillCfg) -> tuple[float, float, float]:
    return (0.0, 0.0, dc.body_span[1] + dc.handle_radius + dc.bit_len)
