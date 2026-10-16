"""The reduced training scene as one mjlab entity.

The whole reduced MJCF (tactile_sim.sim.training_parts: Vega U right arm, right WUJI Hand 2, hammer on its free joint,
nail on its slide joint, board and table) is one entity, so the hand/handle contact pairs and the grasp weld stay
inside one spec and the compiled model is the CPU model element for element (names gain the "robot/" prefix). The
hammer body leads the worldbody, so its free joint is the entity's root (mjlab allows one free joint per entity and
takes it as the root only if it comes first).

Actuators: the arm's <position> servos from the MJCF (XmlActuatorCfg; kp and kv from ArmCfg.servo_kp / servo_kd and
the P multiplier, force-limited at the URDF efforts) and the hand's <motor> actuators driven by the WUJI MIT-mode
drive (WujiMitActuatorCfg). Physics options come from `mujoco_cfg` (mjlab does not take an entity's <option>).
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sim import MujocoCfg

from tactile_rl.actuators.wuji_mit import WujiMitActuatorCfg
from tactile_sim.config import SimConfig
from tactile_sim.sim.training_parts import config_from_key, training_parts

ENTITY = "robot"
ARM_JOINTS = r"R_arm_j\d"
# the right WUJI Hand 2's 20 joints
HAND_JOINTS = (r"r_(thumb_(cmc_flex|cmc_abd|mcp|ip)"
               r"|(index_finger|middle_finger|ring_finger|pinky)_(mcp_flex|mcp_abd|pip|dip))")


def reduced_scene_spec(scene_key: str) -> mujoco.MjSpec:
    """The training MJCF as an MjSpec, without <option> (MujocoCfg sets the options) and the CPU arena size (MuJoCo
    Warp sizes its own buffers: SimulationCfg.nconmax / njmax)."""
    root = ET.fromstring(training_parts(scene_key).xml)
    for tag in ("option", "size"):
        for e in root.findall(tag):
            root.remove(e)
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def robot_entity_cfg(scene_key: str) -> EntityCfg:
    cfg = config_from_key(scene_key)
    return EntityCfg(
        spec_fn=functools.partial(reduced_scene_spec, scene_key),
        articulation=EntityArticulationInfoCfg(actuators=(
            XmlActuatorCfg(target_names_expr=(ARM_JOINTS,), command_field="position"),
            WujiMitActuatorCfg(target_names_expr=(HAND_JOINTS,), drive=cfg.wuji, physics_dt=cfg.physics.timestep,
                               scene_key=scene_key),
        )),
    )


def mujoco_cfg(cfg: SimConfig) -> MujocoCfg:
    """The CPU testbed's physics options (PhysicsCfg, and MuJoCo's defaults for the rest) as mjlab's MujocoCfg. noslip
    has no counterpart: MuJoCo Warp has no noslip solver."""
    ph = cfg.physics
    d = mujoco.MjSpec().option
    out = MujocoCfg(timestep=ph.timestep, integrator=ph.integrator, impratio=ph.impratio, cone=ph.cone,
                    solver="newton", iterations=ph.iterations, tolerance=float(d.tolerance),
                    ls_iterations=int(d.ls_iterations), ls_tolerance=float(d.ls_tolerance),
                    ccd_iterations=int(d.ccd_iterations), gravity=tuple(ph.gravity))
    if not getattr(ph, "multiccd", True):  # PhysicsCfg.multiccd, as World applies it
        out.disableflags = (*out.disableflags, "multiccd")
    return out
