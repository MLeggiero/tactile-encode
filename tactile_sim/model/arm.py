"""Arm MJCF: load the Menagerie FR3 (or the fallback) and replace its actuators with torque motors."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from tactile_sim import names
from tactile_sim.assets import FALLBACK_ARM_XML, ensure_menagerie, fr3_xml_path
from tactile_sim.config import ArmCfg
from tactile_sim.model.xmlutil import disable_collisions, sub

TORQUE_LIMITS = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)


def resolve_arm_source(cfg: ArmCfg) -> tuple[Path, str]:
    if cfg.source == "fallback":
        return FALLBACK_ARM_XML, "fallback"
    root = ensure_menagerie(download=False)
    if root is not None:
        return fr3_xml_path(root), "fr3"
    if cfg.source == "fr3":
        raise FileNotFoundError("Menagerie FR3 not cached; run `python -m tactile_sim.assets.fetch_menagerie`")
    return FALLBACK_ARM_XML, "fallback"


def load_arm_tree(cfg: ArmCfg) -> tuple[ET.Element, str]:
    path, source = resolve_arm_source(cfg)
    root = ET.parse(path).getroot()
    comp = root.find("compiler")
    if comp is not None and comp.get("meshdir"):
        comp.set("meshdir", str((path.parent / comp.get("meshdir")).resolve()))
    for tag in ("actuator", "keyframe"):
        e = root.find(tag)
        if e is not None:
            root.remove(e)
    disable_collisions([root.find("worldbody")])
    act = sub(root, "actuator")
    for j, m, lim in zip(names.ARM_JOINTS, names.ARM_MOTORS, TORQUE_LIMITS, strict=True):
        sub(act, "motor", name=m, joint=j, ctrlrange=(-lim, lim), ctrllimited="true")
    return root, source
