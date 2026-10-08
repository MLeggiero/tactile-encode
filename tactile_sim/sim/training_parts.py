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
- the CPU reference after the settle (qpos, qvel, the hammer's pose in the hand);
- for the GPU sensors and plants (`skin`, `nail`, the contact pairs and the arm servo gains): the TaxelScan skins
  (taxels in their bodies' frames, the geoms that load each patch, the fast-profile boards' scan offsets and poll
  slots), the nail plant's constants, and which contact pairs carry the hand's skin friction and contact softness.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.config import SimConfig, fast_config, hand_config, residual_config
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


def residual_training_config(rate: float = 1000.0, timestep: float = 0.00025, noslip_iterations: int = 0,
                             **sections) -> SimConfig:
    """The residual experiments' configuration (config.residual_config: Vega U + WUJI Hand 2 with MIT drives, the
    TaxelScan fast profile, the light hammer and its small nail, the residual at `rate` Hz) on the reduced scene at
    the training physics: `timestep` (0.25 ms) and no noslip, as MuJoCo Warp runs it. The hammer training tasks
    (tactile_rl) and the base library (tactile_sim.behaviors.base_library) are built from it."""
    cfg = residual_config(rate, base=fast_config())
    cfg = cfg.replace(physics={"timestep": timestep, "noslip_iterations": noslip_iterations}, scene={"reduced": True})
    return cfg.replace(**sections) if sections else cfg


@dataclass
class SkinPatch:
    """One TaxelScan skin (WujiHandIO's conforming patch) as the GPU load model and readout need it."""

    name: str
    body: str  # the patch body (taxel frames)
    geoms: tuple[str, ...]  # the hand's collision geoms that load it
    grid: tuple[int, int]
    taxel_pos: np.ndarray  # (n, 3) body frame
    taxel_normal: np.ndarray  # (n, 3) body frame, outward
    reach: float  # m: a contact farther than this from every taxel does not load the patch
    board: int
    offset: float  # s: scan start after the board's SYNC edge
    slot: float  # s: the board's poll slot after the next SYNC edge


@dataclass
class NailParts:
    """The nail plant's constants (model.plants.nail.NailPlant) for the GPU hooks."""

    joint: str
    face_pair: str
    face_site: str
    head_site: str
    face_geom: str
    nail_geom: str
    axis: np.ndarray  # (3,) unit strike direction
    k_face: float  # direct-form stiffness of the face/nail pair
    face_zeta: float  # the configured face damping ratio (HammerCfg.face_solref[1])
    dmax: float  # pair_solimp[1] of the face/nail pair (k_face = 1 / (dmax tc zeta)^2)


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
    # GPU sensors and plants
    skin: list[SkinPatch] = field(default_factory=list)  # TaxelScan patches, layout order (empty without skins)
    hammer_body: str = names.HAMMER_BODY
    tool_geoms: tuple[str, ...] = ()  # every geom of the hammer body
    handle_geom: str = names.HAMMER_HANDLE_GEOM
    nail: NailParts | None = None
    skin_pairs: tuple[int, ...] = ()  # hand/handle contact pairs: skin friction and hand_solref
    self_pairs: tuple[int, ...] = ()  # hand self-contact pairs (hand_solref)
    arm_kp: np.ndarray | None = None  # (7,) arm servo gains in the model (ArmCfg servo gains x P multiplier)
    arm_kd: np.ndarray | None = None

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
    skin_pairs, self_pairs = _hand_pairs(m, hand)
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
        i_eff=i_eff, qpos_settled=d.qpos.copy(), qvel_settled=d.qvel.copy(), hammer_in_hand=w.hammer_in_hand(),
        skin=_skin_patches(w), tool_geoms=tuple(m.geom(int(g)).name for g in sorted(hand.tool_geoms)),
        nail=_nail_parts(w), skin_pairs=skin_pairs, self_pairs=self_pairs,
        arm_kp=w.servo_kp.copy(), arm_kd=w.servo_kd.copy())


def _skin_patches(w) -> list[SkinPatch]:
    """The TaxelScan skins with their fast-profile boards (sensors.taxelscan.board_layout / poll_slots)."""
    hand, s = w.hand, w.cfg.sensors
    if not hand.patches or hand.patches[0].taxel_pos is None or s.ts_model != "rev3":
        return []
    from tactile_sim.sensors.taxelscan import board_layout, poll_slots, taxel_time

    m = w.model
    names_ = [p.name for p in hand.patches]
    grids = [tuple(hand.grid(k)) for k in range(len(names_))]
    boards = board_layout(s, names_, grids)
    slots = poll_slots(s, boards, grids)
    board_of, offset = {}, {}
    for b, ks in enumerate(boards):
        off = 0.0
        for k in ks:
            board_of[k], offset[k] = b, off
            off += grids[k][0] * grids[k][1] * taxel_time(s)
    return [SkinPatch(p.name, m.body(p.body).name, tuple(m.geom(int(g)).name for g in sorted(p.geoms)),
                      grids[k], np.asarray(p.taxel_pos, float).copy(), np.asarray(p.taxel_normal, float).copy(),
                      float(p.reach), board_of[k], offset[k], float(slots[board_of[k]]))
            for k, p in enumerate(hand.patches)]


def _nail_parts(w) -> NailParts | None:
    from tactile_sim.model.plants.nail import PAIR_FACE_NAIL, NailPlant

    p = w.plant
    if not isinstance(p, NailPlant):
        return None
    m = w.model
    return NailParts(joint=names.NAIL_JOINT, face_pair=PAIR_FACE_NAIL, face_site=names.HAMMER_FACE_SITE,
                     head_site=names.NAIL_HEAD_SITE, face_geom=names.HAMMER_FACE_GEOM, nail_geom=names.NAIL_HEAD_GEOM,
                     axis=np.asarray(p.axis, float).copy(), k_face=float(p.k_face),
                     face_zeta=float(w.cfg.hammer.face_solref[1]),
                     dmax=float(m.pair_solimp[m.pair(PAIR_FACE_NAIL).id, 1]))


def _hand_pairs(m: mujoco.MjModel, hand) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Contact pairs between a hand geom and the tool (skin: GripperCfg.hand_friction and hand_solref) and between
    two hand geoms (the wrap's self-contacts: hand_solref), as model.hands.wuji.contact_pairs makes them."""
    tool = set(int(g) for g in hand.tool_geoms)
    hand_body = m.body(names.HAND_BODY).id
    in_hand = set()
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        while b > 0 and b != hand_body:
            b = int(m.body_parentid[b])
        if b == hand_body:
            in_hand.add(g)
    skin, self_ = [], []
    for i in range(m.npair):
        a, b = int(m.pair_geom1[i]), int(m.pair_geom2[i])
        if (a in in_hand and b in tool) or (b in in_hand and a in tool):
            skin.append(i)
        elif a in in_hand and b in in_hand:
            self_.append(i)
    return tuple(skin), tuple(self_)


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
