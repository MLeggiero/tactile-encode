"""Compose the full testbed scene: arm + wrist F/T + compliant-pad gripper + hammer + task plant."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from tactile_sim import names
from tactile_sim.config import SimConfig
from tactile_sim.model.arm import load_arm_tree
from tactile_sim.model.gripper import add_grip_actuation, add_wrist_and_hand
from tactile_sim.model.plants import make_plant
from tactile_sim.model.plants.base import Plant
from tactile_sim.model.sensors_mjcf import add_sensors
from tactile_sim.model.tool_hammer import add_grasp_weld, add_hammer, face_offset
from tactile_sim.model.xmlutil import find_body, sub


def tcp_rotation() -> np.ndarray:
    """Grasp/TCP orientation: approach (z) along world -z, fingers (y) along -y, handle (x) along +x.

    The hammer face points along the TCP's -y, i.e. world +y, which is the strike direction.
    """
    return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def strike_axis() -> np.ndarray:
    """Unit strike direction in the world (face normal)."""
    return -tcp_rotation()[:, 1]


@dataclass
class SceneSpec:
    xml: str
    arm_source: str
    plant: Plant
    hover_tcp: np.ndarray
    sensors: list[tuple[str, int]] = field(default_factory=list)


def hover_tcp_position(cfg: SimConfig) -> np.ndarray:
    return np.array(cfg.scene.hover_tcp, dtype=float)


def nail_head_target(cfg: SimConfig) -> np.ndarray:
    """World position of the nail head's struck surface: `hover_clearance` ahead of the face at hover."""
    face = hover_tcp_position(cfg) + tcp_rotation() @ np.array(face_offset(cfg.hammer))
    return face + cfg.scene.hover_clearance * strike_axis()


def build_scene(cfg: SimConfig) -> SceneSpec:
    root, source = load_arm_tree(cfg.arm)
    root.set("model", "tactile_testbed")
    ph = cfg.physics
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(1, opt)
    for k, v in dict(timestep=ph.timestep, integrator=ph.integrator, cone=ph.cone, impratio=ph.impratio,
                     iterations=ph.iterations, noslip_iterations=ph.noslip_iterations,
                     gravity=ph.gravity).items():
        opt.set(k, " ".join(f"{x:.9g}" for x in v) if isinstance(v, tuple) else str(v))
    sub(root, "size", memory="64M")
    vis = sub(root, "visual")
    sub(vis, "global", offwidth=640, offheight=480)

    wb = root.find("worldbody")
    sub(wb, "light", pos=(0.5, 0, 2.0), dir=(0, 0, -1), directional="true")
    sub(wb, "geom", name="floor", type="plane", size=(2, 2, 0.05), rgba=(0.3, 0.3, 0.32, 1), contype=0,
        conaffinity=0)
    sub(wb, "camera", name="scene_cam", pos=(1.3, -0.9, 0.9), xyaxes=(0.57, 0.82, 0, -0.35, 0.24, 0.9))

    add_wrist_and_hand(find_body(root, "fr3_link7"), cfg.arm, cfg.gripper)
    add_grip_actuation(root, cfg.gripper)
    add_hammer(wb, cfg.hammer)
    add_grasp_weld(root)

    contact = root.find("contact")
    if contact is None:
        contact = sub(root, "contact")
    g = cfg.gripper
    pad_solref = g.pad_solref if g.pad_mode == "explicit" else g.soft_pad_solref
    for pad in names.PAD_GEOMS:
        sub(contact, "pair", name=f"pair_{pad}_handle", geom1=pad, geom2=names.HAMMER_HANDLE_GEOM, condim=4,
            friction=(g.pad_friction, g.pad_friction, g.pad_torsion, 0.0001, 0.0001), solref=pad_solref,
            solimp=g.pad_solimp)

    plant = make_plant(cfg)
    plant.add_mjcf(root, wb, nail_head_target(cfg), strike_axis())
    sensors = add_sensors(root)
    ET.indent(root)
    return SceneSpec(xml=ET.tostring(root, encoding="unicode"), arm_source=source, plant=plant,
                     hover_tcp=hover_tcp_position(cfg), sensors=sensors)
