"""Hammer MJCF. The hammer's body frame is the grasp frame: origin on the handle centreline at the grasp
point, x along the handle away from the head, y along the head axis with the striking face at -y.

Default tool: the YCB 048_hammer scan (steel claw hammer, wooden handle, 665 g). Its full mesh is drawn;
contacts use convex hulls cut from the same scan (the gripped handle section for the pads, the striking
face for the nail, the whole head for glancing blows on the board); mass is split 0.45 kg steel head,
0.215 kg handle, each spread over its own hull. Without the YCB cache a primitive hammer with the same
masses is built (cylinder head, capsule handle).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from tactile_sim import names
from tactile_sim.config import HammerCfg
from tactile_sim.model.xmlutil import fmt, sub

STEEL = (0.62, 0.64, 0.67, 1.0)
WOOD = (0.74, 0.55, 0.34, 1.0)


@dataclass(frozen=True)
class HammerGeometry:
    source: str  # "ycb" or "primitive"
    face_local: tuple[float, float, float]  # striking-face centre in the hammer frame
    face_radius: float
    grip_half_width: float  # handle half-width along the finger closing axis at the grasp
    head_center: tuple[float, float, float]
    path: str | None = None


@lru_cache(maxsize=8)
def _geometry(model: str, grip_from_head: float, head_half_len: float, head_radius: float,
              handle_radius: float) -> HammerGeometry:
    if model not in ("auto", "ycb", "primitive"):
        raise ValueError(f"unknown hammer model {model!r}")
    if model != "primitive":
        from tactile_sim.assets.ycb import ensure_hammer, load_hulls

        res = ensure_hammer(grip_from_head, download=False)
        if res is not None:
            path, meta = res
            head = load_hulls(path)["head"]
            return HammerGeometry("ycb", tuple(meta["face_local"]), meta["face_radius"], meta["grip_half_width"],
                                  tuple(float(v) for v in head.mean(axis=0)), str(path))
        if model == "ycb":
            raise FileNotFoundError("YCB hammer not cached; run `python -m tactile_sim.assets.ycb`")
    return HammerGeometry("primitive", (-grip_from_head, -head_half_len, 0.0), head_radius, handle_radius,
                          (-grip_from_head, 0.0, 0.0))


def hammer_geometry(h: HammerCfg) -> HammerGeometry:
    return _geometry(h.model, h.grip_from_head, h.head_half_len, h.head_radius, h.handle_radius)


def face_offset(h: HammerCfg) -> tuple[float, float, float]:
    """Hammer face centre in the hammer (grasp) frame."""
    return hammer_geometry(h).face_local


def add_hammer(worldbody: ET.Element, h: HammerCfg, root: ET.Element | None = None) -> str:
    geo = hammer_geometry(h)
    b = sub(worldbody, "body", name=names.HAMMER_BODY, pos=(0, 0, 1.0))
    sub(b, "freejoint", name="hammer_free")
    if geo.source == "ycb":
        _add_ycb(root, b, h, geo)
    else:
        _add_primitive(b, h)
    sub(b, "site", name=names.HAMMER_FACE_SITE, pos=geo.face_local, size=0.003, group=4)
    sub(b, "site", name=names.HAMMER_REF_SITE, pos=(0, 0, 0), size=0.003, group=4)
    sub(b, "site", name=names.HAMMER_IMU_SITE, pos=geo.head_center, size=0.003, group=4)
    return geo.source


def _add_primitive(b: ET.Element, h: HammerCfg) -> None:
    xh = -h.grip_from_head
    sub(b, "geom", name=names.HAMMER_HEAD_GEOM, type="cylinder", size=h.head_radius,
        fromto=(xh, -h.head_half_len, 0, xh, h.head_half_len, 0), mass=h.head_mass,
        rgba=STEEL, contype=0, conaffinity=0)
    # the striking face: a thin disc flush with the head's -y end (contact only, massless)
    sub(b, "geom", name=names.HAMMER_FACE_GEOM, type="cylinder", size=h.head_radius,
        fromto=(xh, -h.head_half_len, 0, xh, -h.head_half_len + 0.004, 0), mass=0,
        rgba=STEEL, contype=0, conaffinity=0, group=3)
    x_end = h.handle_len - h.grip_from_head
    sub(b, "geom", name=names.HAMMER_HANDLE_GEOM, type="capsule", size=h.handle_radius,
        fromto=(xh + h.head_radius, 0, 0, x_end, 0, 0), mass=h.handle_mass,
        rgba=WOOD, contype=0, conaffinity=0)


def _add_ycb(root: ET.Element, b: ET.Element, h: HammerCfg, geo: HammerGeometry) -> None:
    from tactile_sim.assets.ycb import load_hulls

    path = Path(geo.path)
    hulls = load_hulls(path)
    asset = root.find("asset")
    if asset is None:
        asset = sub(root, "asset")
    sub(asset, "mesh", name="hammer_head_visual", file=str(path / "hammer_head_visual.stl"))
    sub(asset, "mesh", name="hammer_handle_visual", file=str(path / "hammer_handle_visual.stl"))
    for key in ("grip", "face", "head", "handle"):
        pts = hulls[key].astype(float)
        if len(pts) > 1500:  # the hull only needs its outline; thin the cloud deterministically
            pts = pts[np.linspace(0, len(pts) - 1, 1500).astype(int)]
        sub(asset, "mesh", name=f"hammer_{key}_hull", vertex=fmt(pts.reshape(-1)))
    vis = dict(contype=0, conaffinity=0, group=2, mass=0, density=0)
    sub(b, "geom", type="mesh", mesh="hammer_head_visual", rgba=STEEL, **vis)
    sub(b, "geom", type="mesh", mesh="hammer_handle_visual", rgba=WOOD, **vis)
    hull = dict(contype=0, conaffinity=0, group=3, rgba=(0.9, 0.3, 0.3, 0.3))
    # mass carriers: steel head and wooden handle, each uniform over its convex hull
    sub(b, "geom", name=names.HAMMER_HEAD_GEOM, type="mesh", mesh="hammer_head_hull", mass=h.head_mass, **hull)
    sub(b, "geom", name="hammer_handle_mass", type="mesh", mesh="hammer_handle_hull", mass=h.handle_mass, **hull)
    # contact-only hulls
    sub(b, "geom", name=names.HAMMER_FACE_GEOM, type="mesh", mesh="hammer_face_hull", mass=0, density=0, **hull)
    sub(b, "geom", name=names.HAMMER_HANDLE_GEOM, type="mesh", mesh="hammer_grip_hull", mass=0, density=0, **hull)


def add_grasp_weld(root: ET.Element) -> None:
    eq = root.find("equality")
    if eq is None:
        eq = sub(root, "equality")
    sub(eq, "weld", name=names.GRASP_WELD, body1=names.HAND_BODY, body2=names.HAMMER_BODY, active="false",
        solref=(0.005, 1))
