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
    if cfg is not None and cfg.scene.strike_dir is not None:
        a = np.asarray(cfg.scene.strike_dir, float)
        a /= np.linalg.norm(a)
        h = np.asarray(cfg.scene.handle_dir, float)
        h = h - (h @ a) * a
        h /= np.linalg.norm(h)
        return np.column_stack([h, -a, np.cross(h, -a)])
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


def tool_target(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Where the plant meets the tool and the tool's working direction there: the nail head and the strike axis for
    the hammer, the board surface under the teeth and the cut direction for the saw, the screw head (or board
    surface) on the bit axis and the bit direction for the driver."""
    kind = cfg.plant.kind
    if kind == "nail":
        return nail_head_target(cfg), strike_axis(cfg)
    from tactile_sim.model.tools import bit_tip_local, saw_teeth_local

    R = tcp_rotation(cfg)
    p = hover_tcp_position(cfg)
    if kind == "saw":
        a, b = saw_teeth_local(cfg.saw)
        return p + R @ (0.5 * (np.array(a) + np.array(b))), R[:, 1]
    if kind == "drill":
        if cfg.drill.form == "pistol":
            from tactile_sim.model.tools import pistol_tip_local

            a = -R[:, 1]  # the pistol's bit points where the hammer's face does
            return p + R @ np.array(pistol_tip_local(cfg.drill)) + cfg.drill.clearance * a, a
        return p + R @ np.array(bit_tip_local(cfg.drill)) + cfg.drill.clearance * R[:, 2], R[:, 2]
    raise ValueError(f"unknown plant kind {kind!r}")


def add_tool(wb, cfg: SimConfig, root) -> str:
    """The held tool for this task (all share the hammer's part names; see tactile_sim.model.tools)."""
    kind = cfg.plant.kind
    if kind == "saw":
        from tactile_sim.model.tools import add_saw

        return add_saw(wb, cfg.saw)
    if kind == "drill":
        from tactile_sim.model.tools import add_driver, add_pistol_driver

        if cfg.drill.form == "pistol":
            return add_pistol_driver(wb, cfg.drill)
        if cfg.drill.form != "inline":
            raise ValueError(f"unknown drill form {cfg.drill.form!r} (expected 'inline' or 'pistol')")
        return add_driver(wb, cfg.drill)
    return add_hammer(wb, cfg.hammer, root)


def tool_contact_geoms(cfg: SimConfig) -> list[str]:
    """The held tool's geoms a wrap grasp touches: the hammer's handle slices, or the pistol driver's grip (its
    trigger, when it has one, gets its own pairs: trigger_contact_pairs)."""
    if cfg.plant.kind == "drill" and cfg.drill.form == "pistol":
        return [names.HAMMER_HANDLE_GEOM]
    if cfg.plant.kind != "nail":
        raise ValueError(f"the WUJI Hand 2 wraps the hammer or the pistol driver, not a {cfg.plant.kind!r} tool")
    return handle_segment_geoms(cfg.hammer)


def hand_friction_caps(cfg: SimConfig) -> dict[str, float] | None:
    """Hand bodies whose skin friction on the tool is capped (wuji.contact_pairs): the trigger finger on a pistol
    with a trigger part, at DrillCfg.trigger_finger_friction."""
    if cfg.plant.kind == "drill" and cfg.drill.form == "pistol" and cfg.drill.trigger_mode == "finger":
        return {"r_index_finger": cfg.drill.trigger_finger_friction}
    return None


def trigger_contact_pairs(contact: ET.Element, col_geoms: dict[str, list[str]], cfg: SimConfig) -> None:
    """Hand-trigger contact pairs when the pistol has a trigger part (DrillCfg.trigger_mode "finger"): the hand's skin
    on the trigger's smooth face, at DrillCfg.trigger_friction."""
    if not (cfg.plant.kind == "drill" and cfg.drill.form == "pistol" and cfg.drill.trigger_mode == "finger"):
        return
    from dataclasses import replace

    from tactile_sim.model.hands import wuji

    g = replace(cfg.gripper, hand_friction=cfg.drill.trigger_friction)
    wuji.contact_pairs(contact, col_geoms, [names.TRIGGER_GEOM], g, self_pairs=False)


def _scene_setup(root: ET.Element, cfg: SimConfig) -> tuple[ET.Element, ET.Element]:
    """Options, memory, light, floor and camera every scene shares; returns the worldbody and the contact section."""
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
    contact = root.find("contact")
    if contact is None:
        contact = sub(root, "contact")
    return wb, contact


def add_ego_camera(root: ET.Element, cfg: SimConfig) -> None:
    """The VLA's view: a camera on the head's left ZED imager (VlaCfg). The URDF gives that frame as a ROS optical
    frame (z forward, y down); a MuJoCo camera looks along its -z with y up, hence the half turn about x."""
    v = cfg.vla
    sub(find_body(root, names.EGO_CAMERA_BODY), "camera", name=names.EGO_CAMERA, quat=(0, 1, 0, 0), fovy=v.cam_fovy)
    vis = root.find("visual")
    vis = sub(root, "visual") if vis is None else vis
    glob = vis.find("global")
    glob = sub(vis, "global") if glob is None else glob
    for k, n in (("offwidth", v.cam_width), ("offheight", v.cam_height)):
        glob.set(k, str(max(int(glob.get(k, "0")), n)))


def build_scene(cfg: SimConfig) -> SceneSpec:
    if cfg.scene.reduced:
        return build_reduced_scene(cfg)
    if cfg.arm.robot.startswith("vega"):
        from tactile_sim.model.robots import load_vega_tree

        root, source = load_vega_tree(cfg.arm)
    else:
        root, source = load_arm_tree(cfg.arm)
    wb, contact = _scene_setup(root, cfg)
    if cfg.arm.robot.startswith("vega"):
        add_ego_camera(root, cfg)

    g = cfg.gripper
    hand_info = None
    pad_sites = None
    if g.hand == "franka" and cfg.arm.robot != "fr3":
        raise ValueError("the Franka Hand is modelled on the FR3 only")
    if g.hand == "franka":
        hand_source = add_wrist_and_hand(find_body(root, "fr3_link7"), cfg.arm, g, root)
        add_grip_actuation(root, g)
        hammer_source = add_tool(wb, cfg, root)
        pad_solref = g.pad_solref if g.pad_mode == "explicit" else g.soft_pad_solref
        for pad in names.PAD_GEOMS:
            sub(contact, "pair", name=f"pair_{pad}_handle", geom1=pad, geom2=names.HAMMER_HANDLE_GEOM, condim=4,
                friction=(g.pad_friction, g.pad_friction, g.pad_torsion, 0.0001, 0.0001), solref=pad_solref,
                solimp=g.pad_solimp)
    elif g.hand == "wuji2":
        if cfg.plant.kind != "nail" and not (cfg.plant.kind == "drill" and cfg.drill.form == "pistol"):
            raise ValueError("the WUJI Hand 2 holds the hammer or the pistol driver (drill.form = 'pistol'); the saw "
                             "and the inline driver are modelled with the Franka Hand only")
        hand_info = _add_wuji(root, cfg, contact)
        hand_source = "wuji2"
        hammer_source = hand_info["hammer_source"]
        pad_sites = {p.name: f"patch_{p.name}" for p in hand_info["patches"] if getattr(p, "accel", True)}
    else:
        raise ValueError(f"unknown hand {g.hand!r} (expected 'franka' or 'wuji2')")
    add_grasp_weld(root)

    plant = make_plant(cfg)
    plant.add_mjcf(root, wb, *tool_target(cfg))
    sensors = add_sensors(root, pad_sites)
    ET.indent(root)
    return SceneSpec(xml=ET.tostring(root, encoding="unicode"), arm_source=source, plant=plant,
                     hover_tcp=hover_tcp_position(cfg), sensors=sensors, hand_source=hand_source,
                     hammer_source=hammer_source, hand_info=hand_info)


REDUCED_ROBOTS = ("vega_1u",)


def build_reduced_scene(cfg: SimConfig, sensors: bool = True, markers: bool = True, visual: bool = False) -> SceneSpec:
    """The reduced training scene (SceneCfg.reduced): the Vega U on its fixed pedestal with only the striking (right)
    arm, the right WUJI Hand 2 on torque motors, the hammer, and the nail plant with its board and table.

    Compared with build_scene: no left arm, no left hand and no head (the lift and the torso flip are fixed joints,
    so the removed branches never move the right arm's base and its dynamics are unchanged); no visual-only meshes
    unless `visual` (every body keeps its explicit inertial, so masses and inertias are unchanged); the hammer body is
    the first in the worldbody, so its free joint leads qpos (mjlab takes an entity's leading free joint as its root).

    `sensors` adds the MuJoCo sensors the testbed reads (wrist F/T and accelerometers: sim-only, and costly on the
    GPU); `markers` adds a site per taxel (for the viewer). The GPU training env builds this scene without either
    (see tactile_sim.sim.training_parts).
    """
    if cfg.arm.robot not in REDUCED_ROBOTS or cfg.gripper.hand != "wuji2" or cfg.plant.kind != "nail":
        raise ValueError(f"the reduced training scene is the WUJI Hand 2 with the hammer on a {REDUCED_ROBOTS} "
                         f"(got robot {cfg.arm.robot!r}, hand {cfg.gripper.hand!r}, plant {cfg.plant.kind!r})")
    from tactile_sim.model.robots import VEGA_HEAD, VEGA_LEFT_ARM, load_vega_tree

    root, source = load_vega_tree(cfg.arm)
    removed = set(VEGA_HEAD) | set(VEGA_LEFT_ARM)
    # the head and the left arm: every body whose own joint is one of theirs, with its subtree
    _remove_children(root, "body", lambda b: any(j.get("name") in removed for j in b.findall("joint")))
    act = root.find("actuator")
    for a in [a for a in act if a.get("joint") in removed]:
        act.remove(a)
    wb, contact = _scene_setup(root, cfg)
    hand_info = _add_wuji(root, cfg, contact, left_hand=False)
    hammer = find_body(root, names.HAMMER_BODY)
    wb.remove(hammer)
    wb.insert(0, hammer)
    add_grasp_weld(root)
    plant = make_plant(cfg)
    plant.add_mjcf(root, wb, *tool_target(cfg))
    pad_sites = {p.name: f"patch_{p.name}" for p in hand_info["patches"] if getattr(p, "accel", True)}
    sensor_list = add_sensors(root, pad_sites) if sensors else []
    if not markers:
        _remove_children(root, "site", lambda s: s.get("name", "").startswith("taxel_"))
    if not visual:
        _strip_visual_meshes(root)
    ET.indent(root)
    return SceneSpec(xml=ET.tostring(root, encoding="unicode"), arm_source=source, plant=plant,
                     hover_tcp=hover_tcp_position(cfg), sensors=sensor_list, hand_source="wuji2",
                     hammer_source=hand_info["hammer_source"], hand_info=hand_info)


def _remove_children(root: ET.Element, tag: str, doomed) -> None:
    """Remove every `tag` child of a body (or of the worldbody) for which `doomed(child)` holds, subtree included."""
    hits = [(parent, child) for parent in root.iter() if parent.tag in ("body", "worldbody")
            for child in parent.findall(tag) if doomed(child)]
    for parent, child in hits:
        if any(c is child for c in parent):  # an ancestor may already have been removed with its subtree
            parent.remove(child)


def _strip_visual_meshes(root: ET.Element) -> None:
    """Drop the mesh geoms that only draw (visual group 2, no collisions) and the mesh assets nothing uses any more.
    A geom is dropped only if it cannot carry mass: its body has an explicit inertial, or the geom is massless."""
    inertial = {id(b) for b in root.iter("body") if b.find("inertial") is not None}

    def visual_only(parent_has_inertial: bool, g: ET.Element) -> bool:
        draws_only = g.get("type") == "mesh" and g.get("group") == "2" and g.get("contype") == "0" and \
            g.get("conaffinity") == "0"
        return draws_only and (parent_has_inertial or g.get("mass") == "0")

    hits = [(b, g) for b in root.iter("body") for g in b.findall("geom") if visual_only(id(b) in inertial, g)]
    for b, g in hits:
        b.remove(g)
    used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    asset = root.find("asset")
    if asset is not None:
        for mesh in [m for m in asset.findall("mesh") if m.get("name") not in used]:
            asset.remove(mesh)


def _add_wuji(root, cfg: SimConfig, contact, left_hand: bool = True) -> dict:
    from tactile_sim.assets.fetch_hands import hand_xml
    from tactile_sim.model.hands import wuji
    from tactile_sim.model.hands.wuji_grasp import wrap_grasp

    path = hand_xml("wuji2")
    if path is None:
        raise FileNotFoundError("WUJI Hand 2 not cached; run `python -m tactile_sim.assets.fetch_hands`")
    g = cfg.gripper
    grasp = wrap_grasp(cfg, path)
    if g.patch_layout == "taxelscan":
        from tactile_sim.model.hands.taxel_layout import conforming_layout

        s = cfg.sensors
        patches = list(conforming_layout(str(path), tuple(s.ts_palm_grid), tuple(s.ts_finger_grid)))
    else:
        patches = wuji.patches_from_grasp(grasp.contact_frames, grasp.contact_force, g.patch_layout)
    from tactile_sim.model.robots import arm_spec

    spec = arm_spec(cfg.arm)
    parent, hand_z = add_wrist(find_body(root, spec.flange_body), cfg.arm, attach_z=spec.attach_z)
    info = wuji.add_wuji_hand(parent, root, g, path, (0.0, 0.0, hand_z), tcp_pos=grasp.hammer_pos,
                              tcp_q=grasp.hammer_quat, patches=patches)
    wb = root.find("worldbody")
    info["hammer_source"] = add_tool(wb, cfg, root)
    wuji.contact_pairs(contact, info["col_geoms"], tool_contact_geoms(cfg), g, caps=hand_friction_caps(cfg))
    trigger_contact_pairs(contact, info["col_geoms"], cfg)
    info["grasp"] = grasp
    if left_hand and cfg.arm.robot.startswith("vega") and cfg.arm.left_hand:
        left = hand_xml("wuji2_left")
        if left is None:
            raise FileNotFoundError("WUJI Hand 2 (left) not cached; run `python -m tactile_sim.assets.fetch_hands`")
        info["left"] = wuji.add_wuji_hand(find_body(root, "L_arm_l8"), root, g, left, (0.0, 0.0, 0.02),
                                          body_name="hand_left", with_tcp=False, hold_kp=0.6)
    return info
