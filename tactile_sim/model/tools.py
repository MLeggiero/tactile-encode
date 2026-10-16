"""Held tools other than the hammer: a hand saw, an inline cordless driver and a pistol drill/driver (the Makita
DF333D, tactile_sim.model.df333d, or a primitive one).

Every held tool uses the hammer's names for the parts the rest of the testbed looks up (names.HAMMER_BODY for the
tool body with its free joint "hammer_free", names.HAMMER_HANDLE_GEOM for the gripped handle the pads touch,
names.HAMMER_REF_SITE at the grasp frame, names.HAMMER_FACE_SITE at the working point, names.HAMMER_IMU_SITE for the
tool accelerometer), so the arm, hand, grasp settle, sensors and slip metrics work unchanged. The tool body frame is
the grasp frame. All tool geoms except the handle are visual: the saw's and driver's interaction with the work is
modelled by their plants (tactile_sim.model.plants.saw / .drill) as applied wrenches.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

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


def add_pistol_driver(worldbody: ET.Element, dc: DrillCfg) -> str:
    """A pistol-grip drill/driver laid out like the hammer (tool_hammer): x down from the housing, y along the bit
    with the tip at -y (where the hammer's face points), z completing the frame; the tool frame is the grasp point
    on the grip's centreline. DrillCfg.pistol_body "df333d": the Makita DF333D, measured (model.df333d); "primitive":
    a capsule grip along x (the contact geom, names.HAMMER_HANDLE_GEOM) under a cylinder housing, chuck, bit and a
    battery box, mass and drawing only (the plant applies the bit's loads)."""
    b = _tool_body(worldbody)
    if dc.pistol_body == "df333d":
        from tactile_sim.model.df333d import add_df333d

        add_df333d(b, dc)
        _pistol_trigger(b, dc)
        return "pistol_driver"
    if dc.pistol_body != "primitive":
        raise ValueError(f"unknown pistol_body {dc.pistol_body!r} (expected 'df333d' or 'primitive')")
    xh = -dc.pistol_grip_from_head
    rh = dc.pistol_housing_radius
    x_end = xh + rh + dc.pistol_grip_len
    sub(b, "geom", name=names.HAMMER_HANDLE_GEOM, type="capsule", size=dc.pistol_grip_radius,
        fromto=(xh + rh, 0, 0, x_end, 0, 0), mass=dc.pistol_grip_mass, rgba=GRIP, contype=0, conaffinity=0)
    sub(b, "geom", name="driver_housing", type="cylinder", size=rh,
        fromto=(xh, dc.pistol_housing_back, 0, xh, -dc.pistol_housing_front, 0), mass=dc.pistol_housing_mass,
        rgba=BODY, contype=0, conaffinity=0)
    y_chuck = -dc.pistol_housing_front - dc.pistol_chuck_len
    sub(b, "geom", name="driver_chuck", type="cylinder", size=0.45 * rh,
        fromto=(xh, -dc.pistol_housing_front, 0, xh, y_chuck, 0), mass=0, rgba=GRIP, contype=0, conaffinity=0)
    tip = pistol_tip_local(dc)
    sub(b, "geom", name="driver_bit", type="capsule", size=0.003, fromto=(xh, y_chuck, 0, xh, tip[1] + 0.003, 0),
        mass=0, rgba=STEEL, contype=0, conaffinity=0)
    bx, by, bz = dc.pistol_battery_half
    sub(b, "geom", name="driver_battery", type="box", size=(bx, by, bz), pos=(x_end + bx, 0, 0),
        mass=dc.pistol_battery_mass, rgba=GRIP, contype=0, conaffinity=0)
    sub(b, "site", name=BIT_TIP_SITE, pos=tip, size=0.002, group=4)
    sub(b, "site", name=names.HAMMER_FACE_SITE, pos=tip, size=0.003, group=4)
    sub(b, "site", name=names.HAMMER_IMU_SITE, pos=(xh, 0, 0), size=0.003, group=4)
    _pistol_trigger(b, dc)
    return "pistol_driver"


def _pistol_trigger(b: ET.Element, dc: DrillCfg) -> None:
    if dc.trigger_mode == "finger":
        add_pistol_trigger(b, dc)
    elif dc.trigger_mode != "script":
        raise ValueError(f"unknown trigger_mode {dc.trigger_mode!r} (expected 'script' or 'finger')")


def pistol_trigger_frame(dc: DrillCfg) -> tuple[np.ndarray, np.ndarray]:
    """The trigger's centre at rest and the direction it is pulled (radially into the grip), in the tool frame."""
    if dc.pistol_body == "df333d":
        from tactile_sim.model.df333d import trigger_frame

        c, pull, _ = trigger_frame(dc)
        return c, pull
    a = dc.trigger_angle
    out = np.array([0.0, -np.cos(a), np.sin(a)])  # from the grip's axis, the front (-y) turned toward +z
    x = -dc.pistol_grip_from_head + dc.pistol_housing_radius + dc.trigger_x
    r = dc.pistol_grip_radius + dc.trigger_gap + dc.trigger_travel + dc.trigger_half[1]
    return np.array([x, 0.0, 0.0]) + r * out, -out


def add_pistol_trigger(tool_body: ET.Element, dc: DrillCfg) -> None:
    """A box on a slide joint, pulled into the grip against a preloaded spring (DrillCfg trigger_*): joint position 0
    at rest, trigger_travel fully pulled. It collides only through explicit pairs with the hand."""
    c, pull = pistol_trigger_frame(dc)
    y_b = -pull  # the box's y: radial, outward
    x_b, half = (1.0, 0.0, 0.0), dc.trigger_half  # the box's x: along the grip
    if dc.pistol_body == "df333d":
        from tactile_sim.model.df333d import body, trigger_frame

        x_b, half = tuple(trigger_frame(dc)[2]), tuple(1e-3 * v for v in body(dc).trigger_half)
    tb = sub(tool_body, "body", name=names.TRIGGER_BODY, pos=c, xyaxes=(*x_b, *y_b))
    f0, f1 = dc.trigger_force
    k = (f1 - f0) / dc.trigger_travel
    sub(tb, "joint", name=names.TRIGGER_JOINT, type="slide", axis=(0, -1, 0), limited="true",
        range=(0, dc.trigger_travel), stiffness=k, springref=-f0 / k, damping=dc.trigger_damping, armature=0.01,
        solreflimit=(0.002, 1.0))  # hard stops (MuJoCo's default limit is soft for a part this light)
    sub(tb, "geom", name=names.TRIGGER_GEOM, type="box", size=half, mass=0.008, rgba=(0.85, 0.2, 0.1, 1),
        contype=0, conaffinity=0)


def trigger_level(dc: DrillCfg, travel: float) -> float:
    """The motor's speed command (0..1) for a trigger pulled `travel` metres: nothing through the dead band."""
    x = travel / dc.trigger_travel
    return float(np.clip((x - dc.trigger_dead) / (1.0 - dc.trigger_dead), 0.0, 1.0))


def pistol_tip_local(dc: DrillCfg) -> tuple[float, float, float]:
    if dc.pistol_body == "df333d":
        from tactile_sim.model.df333d import tip_local

        return tip_local(dc)
    return (-dc.pistol_grip_from_head, -(dc.pistol_housing_front + dc.pistol_chuck_len + dc.bit_len), 0.0)


def pistol_bit_out(dc: DrillCfg) -> float:
    """The bit's length out of the chuck (m): the chuck is this far behind the tip."""
    if dc.form == "pistol" and dc.pistol_body == "df333d":
        from tactile_sim.model.df333d import bit_out

        return bit_out(dc)
    return dc.bit_len


def saw_teeth_local(s: SawCfg) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (s.blade_x[0], s.blade_depth, 0.0), (s.blade_x[1], s.blade_depth, 0.0)


def bit_tip_local(dc: DrillCfg) -> tuple[float, float, float]:
    return (0.0, 0.0, dc.body_span[1] + dc.handle_radius + dc.bit_len)
