"""Compose the full testbed scene: arm + wrist F/T + compliant-pad gripper + hammer + task plant."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from tactile_sim import names
from tactile_sim.config import SimConfig
from tactile_sim.model.arm import load_arm_tree
from tactile_sim.model.gripper import add_grip_actuation, add_wrist, add_wrist_and_hand
from tactile_sim.model.plants import make_plant
from tactile_sim.model.plants.base import Plant
from tactile_sim.model.sensors_mjcf import add_sensors
from tactile_sim.model.tool_hammer import add_grasp_weld, add_hammer, face_offset, handle_segment_geoms
from tactile_sim.model.xmlutil import find_body, sub


def tcp_rotation(cfg: SimConfig | None = None) -> np.ndarray:
    """Grasp/TCP orientation: approach (z) along world -z, fingers (y) along -y, handle (x) along +x, then the
    whole task turned by `scene.strike_yaw` about the vertical.

    The hammer face points along the TCP's -y, i.e. world +y at zero yaw, which is the strike direction.
    """
    yaw = cfg.scene.strike_yaw if cfg is not None else 0.0
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def strike_axis(cfg: SimConfig | None = None) -> np.ndarray:
    """Unit strike direction in the world (face normal)."""
    return -tcp_rotation(cfg)[:, 1]


@dataclass
class SceneSpec:
    xml: str
    arm_source: str
    plant: Plant
    hover_tcp: np.ndarray
    sensors: list[tuple[str, int]] = field(default_factory=list)
    hand_source: str = "box"
    hammer_source: str = "primitive"
    hand_info: dict | None = None  # dexterous hands: joints, taxel patches, grasp keyframe


def strike_geometry(cfg: SimConfig):
    """The strike path: contact pose = hover_tcp + hover_clearance along the strike axis, nominal orientation."""
    from tactile_sim.behaviors.trajectories import StrikeGeometry

    R = tcp_rotation(cfg)
    contact = hover_tcp_position(cfg) + cfg.scene.hover_clearance * strike_axis(cfg)
    return StrikeGeometry(contact, R, np.array(face_offset(cfg.hammer)), cfg.scene.hover_clearance,
                          cfg.swing.arc_radius)


def hover_tcp_position(cfg: SimConfig) -> np.ndarray:
    return np.array(cfg.scene.hover_tcp, dtype=float)


def nail_head_target(cfg: SimConfig) -> np.ndarray:
    """World position of the nail head's struck surface: `hover_clearance` ahead of the face at hover."""
    face = hover_tcp_position(cfg) + tcp_rotation(cfg) @ np.array(face_offset(cfg.hammer))
    return face + cfg.scene.hover_clearance * strike_axis(cfg)


def build_scene(cfg: SimConfig) -> SceneSpec:
    if cfg.arm.robot.startswith("vega"):
        from tactile_sim.model.robots import load_vega_tree

        root, source = load_vega_tree(cfg.arm)
    else:
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
    size = root.find("size")
    if size is None:
        size = sub(root, "size")
    size.set("memory", "128M")
    vis = sub(root, "visual")
    sub(vis, "global", offwidth=640, offheight=480)

    wb = root.find("worldbody")
    sub(wb, "light", pos=(0.5, 0, 2.0), dir=(0, 0, -1), directional="true")
    sub(wb, "geom", name="floor", type="plane", size=(2, 2, 0.05), rgba=(0.3, 0.3, 0.32, 1), contype=0,
        conaffinity=0)
    sub(wb, "camera", name="scene_cam", pos=(1.3, -0.9, 0.9), xyaxes=(0.57, 0.82, 0, -0.35, 0.24, 0.9))

    g = cfg.gripper
    contact = root.find("contact")
    if contact is None:
        contact = sub(root, "contact")
    hand_info = None
    pad_sites = None
    if g.hand == "franka" and cfg.arm.robot != "fr3":
        raise ValueError("the Franka Hand is modelled on the FR3 only")
    if g.hand == "franka":
        hand_source = add_wrist_and_hand(find_body(root, "fr3_link7"), cfg.arm, g, root)
        add_grip_actuation(root, g)
        hammer_source = add_hammer(wb, cfg.hammer, root)
        pad_solref = g.pad_solref if g.pad_mode == "explicit" else g.soft_pad_solref
        for pad in names.PAD_GEOMS:
            sub(contact, "pair", name=f"pair_{pad}_handle", geom1=pad, geom2=names.HAMMER_HANDLE_GEOM, condim=4,
                friction=(g.pad_friction, g.pad_friction, g.pad_torsion, 0.0001, 0.0001), solref=pad_solref,
                solimp=g.pad_solimp)
    elif g.hand == "wuji2":
        hand_info = _add_wuji(root, cfg, contact)
        hand_source = "wuji2"
        hammer_source = hand_info["hammer_source"]
        pad_sites = {p.name: f"patch_{p.name}" for p in hand_info["patches"]}
    else:
        raise ValueError(f"unknown hand {g.hand!r} (expected 'franka' or 'wuji2')")
    add_grasp_weld(root)

    plant = make_plant(cfg)
    plant.add_mjcf(root, wb, nail_head_target(cfg), strike_axis(cfg))
    sensors = add_sensors(root, pad_sites)
    ET.indent(root)
    return SceneSpec(xml=ET.tostring(root, encoding="unicode"), arm_source=source, plant=plant,
                     hover_tcp=hover_tcp_position(cfg), sensors=sensors, hand_source=hand_source,
                     hammer_source=hammer_source, hand_info=hand_info)


def _add_wuji(root, cfg: SimConfig, contact) -> dict:
    from tactile_sim.assets.fetch_hands import hand_xml
    from tactile_sim.model.hands import wuji
    from tactile_sim.model.hands.wuji_grasp import wrap_grasp

    path = hand_xml("wuji2")
    if path is None:
        raise FileNotFoundError("WUJI Hand 2 not cached; run `python -m tactile_sim.assets.fetch_hands`")
    g = cfg.gripper
    grasp = wrap_grasp(cfg, path)
    patches = wuji.patches_from_grasp(grasp.contact_frames, grasp.contact_force, g.patch_layout)
    from tactile_sim.model.robots import arm_spec

    spec = arm_spec(cfg.arm)
    parent, hand_z = add_wrist(find_body(root, spec.flange_body), cfg.arm, attach_z=spec.attach_z)
    info = wuji.add_wuji_hand(parent, root, g, path, (0.0, 0.0, hand_z), tcp_pos=grasp.hammer_pos,
                              tcp_q=grasp.hammer_quat, patches=patches)
    wb = root.find("worldbody")
    info["hammer_source"] = add_hammer(wb, cfg.hammer, root)
    wuji.contact_pairs(contact, info["col_geoms"], handle_segment_geoms(cfg.hammer), g)
    info["grasp"] = grasp
    if cfg.arm.robot.startswith("vega") and cfg.arm.left_hand:
        left = hand_xml("wuji2_left")
        if left is None:
            raise FileNotFoundError("WUJI Hand 2 (left) not cached; run `python -m tactile_sim.assets.fetch_hands`")
        info["left"] = wuji.add_wuji_hand(find_body(root, "L_arm_l8"), root, g, left, (0.0, 0.0, 0.02),
                                          body_name="hand_left", with_tcp=False, hold_kp=0.6)
    return info
