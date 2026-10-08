"""What the GPU training env (tactile_rl, mjlab on MuJoCo Warp) takes from the CPU testbed: the reduced scene's MJCF
and the constants its reset and control need, all computed here by the reference implementation so the two
simulators start from the same numbers.

`export_training_parts(cfg)` builds the CPU World on the reduced scene (SceneCfg.reduced), resets it once and
returns a `TrainingParts`:

- `xml`: the reduced scene without the testbed's MuJoCo sensors and taxel marker sites (neither affects the
  dynamics; the sensors would cost the GPU an acceleration pass every step), with the model edits the World makes
  when it is built (the MIT drives widen the hand motors' range; the nail plant puts the face/nail pair in the
  direct solref form) applied, so the GPU model equals the CPU model as it runs;
- the reset recipe: the start state of World.reset (arm at the hover pose, hand at the grasp keyframe, hammer on the
  tool frame), the grasp weld's relative pose, the grip ramp and the settle durations;
- the hold: the arm's servo targets (hover pose plus gravity droop, as World._hold_arm writes them), the grip synergy
  and its constant hold scalar;
- the residual's per-joint constants (rated torque, effective inertia, as ResidualHandL2 computes them);
- the CPU reference after the settle (qpos, qvel, the hammer's pose in the hand).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.config import SimConfig, fast_config, hand_config
from tactile_sim.control.hand import WrapSynergy
from tactile_sim.model.builder import build_reduced_scene


def training_config(timestep: float = 0.00025, noslip_iterations: int = 0, **sections) -> SimConfig:
    """The CPU configuration of the GPU training scene: Vega U + WUJI Hand 2 (MIT-mode drives) + hammer + nail, the
    reduced scene, `timestep` physics. MuJoCo Warp has no noslip solver, hence noslip_iterations 0 by default (the
    8 kHz validator keeps noslip 5)."""
    cfg = hand_config("wuji2", fast_config(), robot="vega_1u")
    cfg = cfg.replace(physics={"timestep": timestep, "noslip_iterations": noslip_iterations},
                      gripper={"hand_control": "mit"}, scene={"reduced": True})
    return cfg.replace(**sections) if sections else cfg


@dataclass
class TrainingParts:
    cfg: SimConfig
    xml: str
    dt: float
    # joint and actuator names, in model order (hand joints in the vendor order WujiHandIO uses)
    hammer_joint: str
    nail_joint: str
    arm_joints: tuple[str, ...]
    hand_joints: tuple[str, ...]
    arm_actuators: tuple[str, ...]
    hand_actuators: tuple[str, ...]
    weld: str
    tcp_site: str
    hammer_site: str
    hand_tau_rated: np.ndarray  # (20,) Nm
    hand_ctrlrange: np.ndarray  # (20, 2) Nm, as WujiHandIO widens it for the MIT drives
    face_pair: str
    face_pair_solref: np.ndarray  # (2,), as NailPlant.bind sets it
    # reset: World.reset_pose, then World.settle
    qpos_start: np.ndarray  # (nq,)
    weld_eq_data: np.ndarray  # (11,) model eq_data of the grasp weld
    settle_weld: float
    settle_free: float
    grip_hold: float  # N of summed patch force; the synergy scalar is grip / f_full
    f_full: float
    # hold
    q_hover: np.ndarray  # (7,)
    arm_target: np.ndarray  # (7,) servo targets: hover pose + gravity droop
    synergy: WrapSynergy
    # residual action map constants (obs_spec.JointParams)
    i_eff: np.ndarray  # (20,) kg m^2
    # CPU reference after World.reset
    qpos_settled: np.ndarray
    qvel_settled: np.ndarray
    hammer_in_hand: tuple[np.ndarray, np.ndarray]  # position, rotation vector (World.hammer_in_hand)

    @property
    def s_hold(self) -> float:
        """The grip synergy's scalar while holding (World.reset ends at grip_hold)."""
        return self.grip_hold / self.f_full

    def s_settle(self, i: int, n_weld: int) -> float:
        """The synergy scalar at step i of the welded settle phase (World.settle: the grip ramps over its first
        half)."""
        return self.grip_hold * min(1.0, 2.0 * i / n_weld) / self.f_full


def _patch_xml(xml: str, hand_ctrlrange: dict[str, np.ndarray], pair_solref: dict[str, np.ndarray]) -> str:
    root = ET.fromstring(xml)
    for a in root.iter("motor"):
        r = hand_ctrlrange.get(a.get("name"))
        if r is not None:
            a.set("ctrlrange", " ".join(f"{x:.17g}" for x in r))
    for p in root.iter("pair"):
        s = pair_solref.get(p.get("name"))
        if s is not None:
            p.set("solref", " ".join(f"{x:.17g}" for x in s))
    return ET.tostring(root, encoding="unicode")


def export_training_parts(cfg: SimConfig | None = None) -> TrainingParts:
    """Build the CPU World on the reduced scene and export what the GPU env needs (see module doc)."""
    from tactile_sim.model.plants.nail import PAIR_FACE_NAIL
    from tactile_sim.sim.world import World, full_inertia

    cfg = cfg or training_config()
    if not cfg.scene.reduced or cfg.gripper.hand_control != "mit":
        raise ValueError("the training scene is the reduced scene (SceneCfg.reduced) with MIT-mode hand drives")
    w = World(cfg)
    m, d = w.model, w.data
    hand = w.hand
    hand_act = [m.actuator(int(a)).name for a in hand.act]
    ctrlrange = {n: m.actuator_ctrlrange[int(a)].copy() for n, a in zip(hand_act, hand.act, strict=True)}
    pid = m.pair(PAIR_FACE_NAIL).id
    face_solref = m.pair_solref[pid].copy()
    train_xml = build_reduced_scene(cfg, sensors=False, markers=False).xml
    train_xml = _patch_xml(train_xml, ctrlrange, {PAIR_FACE_NAIL: face_solref})

    w.reset_pose()
    qpos_start = d.qpos.copy()
    eq = m.eq_data[w.weld_id].copy()
    settle_weld, settle_free = 0.15, 0.25  # World.reset's defaults
    w.settle(settle_weld, settle_free)
    mujoco.mj_forward(m, d)
    M = full_inertia(m, d, np.zeros((m.nv, m.nv)))
    i_eff = np.diag(M)[hand.dofs].copy()
    return TrainingParts(
        cfg=cfg, xml=train_xml, dt=float(m.opt.timestep),
        hammer_joint="hammer_free", nail_joint=names.NAIL_JOINT,
        arm_joints=tuple(w.arm_spec.joints), hand_joints=tuple(hand.joint_names),
        arm_actuators=tuple(w.arm_spec.motors), hand_actuators=tuple(hand_act), weld=names.GRASP_WELD,
        tcp_site=names.TCP_SITE, hammer_site=names.HAMMER_REF_SITE,
        hand_tau_rated=hand.tau_max.copy(), hand_ctrlrange=np.array([ctrlrange[n] for n in hand_act]),
        face_pair=PAIR_FACE_NAIL, face_pair_solref=face_solref,
        qpos_start=qpos_start, weld_eq_data=eq, settle_weld=settle_weld, settle_free=settle_free,
        grip_hold=float(cfg.controller.grip_hold), f_full=float(hand.f_full),
        q_hover=w.q_hover.copy(), arm_target=w.q_hover + w._hover_droop, synergy=hand.synergy,
        i_eff=i_eff, qpos_settled=d.qpos.copy(), qvel_settled=d.qvel.copy(), hammer_in_hand=w.hammer_in_hand())


_CACHE: dict[str, TrainingParts] = {}


def _tuplify(x):
    if isinstance(x, list):
        return tuple(_tuplify(v) for v in x)
    if isinstance(x, dict):
        return {k: _tuplify(v) for k, v in x.items()}
    return x


def config_from_key(key: str) -> SimConfig:
    """The SimConfig a `config_key` came from (JSON lists back to tuples, nested ones included)."""
    return SimConfig.from_dict(_tuplify(json.loads(key)))


def config_key(cfg: SimConfig) -> str:
    """A plain-string handle on a configuration (term configs of the GPU env carry it instead of the SimConfig)."""
    return cfg.to_json()


def training_parts(cfg: SimConfig | str) -> TrainingParts:
    """export_training_parts, cached per configuration (the GPU env's scene, actions and events share one export).
    Takes the SimConfig or its config_key."""
    key = cfg if isinstance(cfg, str) else config_key(cfg)
    if key not in _CACHE:
        _CACHE[key] = export_training_parts(config_from_key(key) if isinstance(cfg, str) else cfg)
    return _CACHE[key]
