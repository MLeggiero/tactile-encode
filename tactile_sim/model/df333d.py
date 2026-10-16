"""The Makita DF333D (12V max CXT drill/driver, 10 mm keyless chuck, slide-on BL1016 / BL1021B pack) as the pistol
driver's body (DrillCfg.pistol_body = "df333d"): its shape measured, its mass placed by part.

Sources:
- Makita's catalogue and manual: 179 x 66 x 212 mm (L x W x H) with a BL1016 / BL1021B pack, 1.1-1.2 kg, 0-450 /
  0-1,700 rpm, 14 / 30 Nm soft / hard joint (ISO 5393), clutch 0.5-3.5 Nm over 20 settings, 170 W.
- The BL1021B pack: 87 x 64 x 37 mm, 0.21 kg.
- The shape: silhouettes of Makita's own product renders (side, DF333D_C8C0; rear, DF333D_C7C0, transparent
  backgrounds), scaled to the catalogue's length and height and checked against the pack's 64 mm width
  (agreement within 2.5 %). Measured: the bit axis 29 mm below the top; the motor housing ~55 mm across, the
  gearbox 45, the clutch ring 42, the chuck 32-40; the grip 30-31 mm wide and 35-44 mm deep (40 over the hand),
  its centreline raked 10.7 deg back from the perpendicular to the bit; the trigger's face 13 mm proud of the grip,
  its centre 51 mm below the bit axis; the base and pack 151-212 mm below the top.
- The mass split (no published data): head 0.70 kg (motor and shell 0.34, gearbox and clutch 0.22, chuck 0.14),
  grip 0.12, base 0.12, pack 0.21; 1.15 kg in all.

Collision: the grip is two parallel capsules (a 30 x 40 mm stadium in section), the motor housing a capsule along
the bit; the gearbox, chuck, bit, base and pack are mass and drawing only. Capsules keep the GPU's taxel ray-cast
analytic. The tool frame is the grasp point (pistol_grip_from_head below the bit axis, pistol_grasp_offset off the
grip's centreline), x down (perpendicular to the bit), y back along the bit (the tip at -y, as on every pistol driver
here), so only the grip leans: its axis is (cos r, sin r, 0) for the rake r, and the hand's wrap turns by it
(wuji.hand_tcp_rotation).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from tactile_sim import names
from tactile_sim.model.xmlutil import sub

TEAL = (0.0, 0.53, 0.58, 1.0)
BLACK = (0.10, 0.11, 0.12, 1.0)
STEEL = (0.70, 0.72, 0.75, 1.0)
GRIP_FRONT_GEOM = "drill_grip_front"  # the grip's other capsule is names.HAMMER_HANDLE_GEOM
HOUSING_GEOM = "drill_housing"


@dataclass(frozen=True)
class DrillBody:
    """A pistol drill's shape in its side view, in mm: X forward along the bit from the tool's rear, Z down from its
    top. All parts are symmetric about the side view's plane."""
    axis_z: float = 29.0  # the bit axis
    grip_x0: float = 84.8  # the grip's centreline: X = grip_x0 - grip_slope * Z
    grip_slope: float = 0.1897  # tan(rake), 10.7 deg
    grip_width: float = 30.0  # across (the capsules' diameter)
    grip_depth: float = 40.0  # front to back (the capsules' span)
    grip_z: tuple[float, float] = (68.0, 140.0)  # the capsules' segment, top to bottom (their caps reach r beyond)
    housing_r: float = 27.5  # motor housing, a capsule on the bit axis
    housing_x: tuple[float, float] = (27.5, 112.0)
    gearbox_r: float = 22.0
    gearbox_x: tuple[float, float] = (112.0, 145.0)
    chuck_r: float = 18.0
    chuck_x: tuple[float, float] = (145.0, 172.0)
    nose_r: float = 7.0
    nose_x: tuple[float, float] = (172.0, 179.0)  # the chuck's jaws; the bit leaves at X 179
    bit_r: float = 3.2  # a 1/4 in hex power bit
    bit_out: float = 50.0  # bit length out of the chuck
    trigger_z: float = 80.0  # the trigger's centre
    trigger_half: tuple[float, float, float] = (9.0, 3.0, 6.0)  # along the grip, radial, across
    trigger_proud: float = 13.0  # its face ahead of the grip's front at rest
    base_x: tuple[float, float] = (38.0, 100.0)
    base_z: tuple[float, float] = (148.0, 175.0)
    base_width: float = 48.0
    pack_x: tuple[float, float] = (28.5, 115.5)  # BL1021B, 87 x 64 x 37 mm
    pack_z: tuple[float, float] = (175.0, 212.0)
    pack_width: float = 64.0
    imu_x: float = 70.0  # the tool IMU on the motor

    @property
    def rake(self) -> float:
        return float(np.arctan(self.grip_slope))


DF333D = DrillBody()


def body(dc) -> DrillBody:
    if dc.pistol_body != "df333d":
        raise ValueError(f"no measured body for pistol_body {dc.pistol_body!r}")
    return DF333D


def _grasp_z(dc) -> float:
    return DF333D.axis_z + 1e3 * dc.pistol_grip_from_head


def point(dc, X: float, Z: float) -> np.ndarray:
    """A side-view point (mm) in the tool frame (m): the grasp point is the origin, on the grip's centreline but for
    DrillCfg.pistol_grasp_offset (where the grip's centreline passes, across it, in the tool's y and z)."""
    b = body(dc)
    zg = _grasp_z(dc)
    xg = b.grip_x0 - b.grip_slope * zg
    oy, oz = dc.pistol_grasp_offset
    return np.array([(Z - zg) * 1e-3, -(X - xg) * 1e-3 + oy, oz])


def grip_axis(dc) -> np.ndarray:
    """Down the grip, in the tool frame."""
    r = body(dc).rake
    return np.array([np.cos(r), np.sin(r), 0.0])


def grip_front(dc) -> np.ndarray:
    """The grip's front (toward the bit's side), perpendicular to its axis."""
    r = body(dc).rake
    return np.array([np.sin(r), -np.cos(r), 0.0])


def tip_local(dc) -> tuple[float, float, float]:
    b = body(dc)
    return tuple(float(v) for v in point(dc, b.nose_x[1] + b.bit_out, b.axis_z))


def bit_out(dc) -> float:
    """The bit's length out of the chuck (m)."""
    return 1e-3 * body(dc).bit_out


def _on_grip(dc, Z: float) -> np.ndarray:
    """The grip's centreline at side-view height Z, in the tool frame."""
    b = body(dc)
    return point(dc, b.grip_x0 - b.grip_slope * Z, Z)


def trigger_frame(dc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The trigger's centre at rest, the direction it is pulled (into the grip), and the grip's axis (the box's x)."""
    b = body(dc)
    front = grip_front(dc)
    surface = 0.5e-3 * b.grip_depth  # the front capsule's surface, from the centreline
    out = surface + 1e-3 * b.trigger_proud - 1e-3 * b.trigger_half[1]
    # out along the raked grip's front normal, from the centreline point that leaves the centre at trigger_z
    c = _on_grip(dc, b.trigger_z - 1e3 * out * front[0]) + out * front
    return c, -front, grip_axis(dc)


def add_df333d(bd: ET.Element, dc) -> None:
    """The DF333D's geoms and sites in the tool body `bd` (tools._tool_body): contact capsules for the grip and the
    housing, the rest mass and drawing (the trigger part is added by tools.add_pistol_trigger)."""
    b = body(dc)
    front = grip_front(dc)
    off = 0.5e-3 * (b.grip_depth - b.grip_width)  # each capsule's axis off the centreline
    top, bot = _on_grip(dc, b.grip_z[0]), _on_grip(dc, b.grip_z[1])
    r = 0.5e-3 * b.grip_width
    for name, s in ((names.HAMMER_HANDLE_GEOM, -1.0), (GRIP_FRONT_GEOM, 1.0)):
        p0, p1 = top + s * off * front, bot + s * off * front
        sub(bd, "geom", name=name, type="capsule", size=r, fromto=(*p0, *p1), mass=0.5 * dc.pistol_grip_mass,
            rgba=BLACK, contype=0, conaffinity=0)

    def along(name, kind, radius, xs, mass, rgba):
        p0, p1 = point(dc, xs[0], b.axis_z), point(dc, xs[1], b.axis_z)
        sub(bd, "geom", name=name, type=kind, size=1e-3 * radius, fromto=(*p0, *p1), mass=mass, rgba=rgba,
            contype=0, conaffinity=0)

    along(HOUSING_GEOM, "capsule", b.housing_r, b.housing_x, dc.pistol_housing_mass, TEAL)
    along("driver_gearbox", "cylinder", b.gearbox_r, b.gearbox_x, dc.pistol_gearbox_mass, BLACK)
    along("driver_chuck", "cylinder", b.chuck_r, b.chuck_x, dc.pistol_chuck_mass, BLACK)
    along("driver_nose", "cylinder", b.nose_r, b.nose_x, 0, STEEL)
    tip = np.array(tip_local(dc))
    p0 = point(dc, b.nose_x[1], b.axis_z)
    sub(bd, "geom", name="driver_bit", type="capsule", size=1e-3 * b.bit_r, fromto=(*p0, *(tip + np.array(
        [0.0, 1e-3 * b.bit_r, 0.0]))), mass=0, rgba=STEEL, contype=0, conaffinity=0)

    def box(name, xs, zs, width, mass, rgba):
        c = point(dc, 0.5 * (xs[0] + xs[1]), 0.5 * (zs[0] + zs[1]))
        half = (0.5e-3 * (zs[1] - zs[0]), 0.5e-3 * (xs[1] - xs[0]), 0.5e-3 * width)  # tool x, y, z
        sub(bd, "geom", name=name, type="box", size=half, pos=tuple(c), mass=mass, rgba=rgba, contype=0,
            conaffinity=0)

    box("driver_base", b.base_x, b.base_z, b.base_width, dc.pistol_base_mass, TEAL)
    box("driver_battery", b.pack_x, b.pack_z, b.pack_width, dc.pistol_battery_mass, BLACK)
    from tactile_sim.model.tools import BIT_TIP_SITE

    sub(bd, "site", name=BIT_TIP_SITE, pos=tuple(tip), size=0.002, group=4)
    sub(bd, "site", name=names.HAMMER_FACE_SITE, pos=tuple(tip), size=0.003, group=4)
    sub(bd, "site", name=names.HAMMER_IMU_SITE, pos=tuple(point(dc, b.imu_x, b.axis_z)), size=0.003, group=4)


def contact_geoms() -> list[str]:
    """The DF333D's geoms a hand touches (the trigger gets its own pairs)."""
    return [names.HAMMER_HANDLE_GEOM, GRIP_FRONT_GEOM, HOUSING_GEOM]
