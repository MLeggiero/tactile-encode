"""Configuration dataclasses for the testbed simulation.

All quantities are SI. Defaults follow the research report and the control flowchart:
0.125 ms physics (8 kHz) so every sensor rate divides the physics rate, a 0.45 kg hammer head,
compliant gripper pads, and a nail whose penetration resistance is Coulomb friction.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhysicsCfg:
    timestep: float = 0.000125
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    impratio: float = 100.0
    noslip_iterations: int = 5  # removes soft-friction creep so a static grasp really holds
    iterations: int = 50
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass
class ArmCfg:
    robot: str = "fr3"  # "fr3" (Franka FR3, 1 kHz torque control), "vega_1u" / "vega_1p" (Dexmate Vega, right arm)
    source: str = "auto"  # "fr3" (Menagerie, required), "fallback" (capsules), "auto" (fr3 if cached)
    flex_mode: str = "wrist"  # "rigid" or "wrist"
    wrist_k_rot: float = 1500.0  # Nm/rad, hinges about the flange x and y axes
    wrist_d_rot: float = 1.5  # Nms/rad
    wrist_k_lin: float = 2.0e5  # N/m along the flange z axis
    wrist_d_lin: float = 150.0  # Ns/m
    ft_body_mass: float = 0.12  # wrist F/T sensor (Bota SensONE class)
    torque_rate_limit: float = 1000.0  # Nm/s per joint, libfranka kMaxTorqueRate (0 = off)
    # --- Dexmate Vega U / Vega-1P (robot = "vega_1u" / "vega_1p"). Its public interface (dexcontrol) streams joint
    # position targets, with an optional velocity feedforward, at 100 Hz by default; stiffness is the factory PD
    # gains scaled by a per-joint P multiplier in [0.1, 4]. Dexmate does not publish the factory gains or the
    # drives' reflected inertia: the values below are assumptions (stiff harmonic-drive servos), to be swept.
    command_rate: float = 100.0  # Hz, position targets (ZOH between updates)
    servo_kp: tuple[float, ...] = (3000.0, 3000.0, 1500.0, 1500.0, 400.0, 400.0, 400.0)  # Nm/rad, factory (assumed)
    servo_kd: tuple[float, ...] = (60.0, 60.0, 30.0, 30.0, 4.0, 4.0, 4.0)  # Nms/rad (assumed)
    servo_p_mult: float = 1.0  # dexcontrol set_pid multiplier, [0.1, 4]
    vega_armature: tuple[float, ...] = (0.25, 0.25, 0.08, 0.08, 0.02, 0.02, 0.02)  # kg m^2 (assumed)
    vega_damping: float = 0.5  # Nms/rad viscous drive loss (assumed)
    torso_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Vega-1P torso joints
    vega_lift: float = 0.2  # m, Vega U lift (0-0.4); set before a run, not part of Dexmate's motion interface
    vega_flip: float = 0.0  # rad, Vega U torso flip (0-1), likewise
    head_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_arm_pose: tuple[float, ...] = (0.064, 0.3, 0.0, -1.556, 1.271, 0.0, 0.0)  # Dexmate "L_shape" pose
    left_hand: bool = True  # an idle WUJI Hand 2 on the left arm
    # joint-space hold used only while settling at reset
    hold_kp: tuple[float, ...] = (600, 600, 600, 600, 250, 150, 50)
    hold_kd: tuple[float, ...] = (50, 50, 50, 50, 20, 12, 5)
    # IK seed near the hover pose (see SceneCfg.hover_tcp)
    q_seed: tuple[float, ...] = (0.0, 0.01, 0.0, -2.36, 0.0, 2.37, 0.79)


@dataclass
class GripperCfg:
    hand: str = "franka"  # "franka" (two-finger Franka Hand) or "wuji2" (WUJI Hand 2, power wrap)
    hand_source: str = "auto"  # "franka" (Menagerie Franka Hand), "box" (same dims, no meshes), "auto"
    tcp_offset: float = 0.1034  # hand base to grasp centre (Franka Hand TCP)
    finger_range: float = 0.04  # opening per finger
    finger_armature: float = 0.1  # Menagerie hand.xml
    grip_force_max: float = 140.0  # N per finger: Franka Hand peak (70 N continuous)
    pad_mode: str = "explicit"  # "explicit" (spring-mounted pads) or "soft_contact"
    pad_half: tuple[float, float, float] = (0.0085, 0.002, 0.0085)  # Franka fingertip pad: 17 x 17 mm
    # hard-rubber fingertip insert (Franka Hand): stiff enough that a 3 g shake tilts the hammer ~2.5 deg,
    # soft enough that the tool still rings in the grasp after a blow
    pad_mass: float = 0.02
    pad_k_n: float = 2.0e5  # N/m normal
    pad_d_n: float = 60.0  # Ns/m normal
    pad_k_t: float = 1.0e5  # N/m tangential
    pad_d_t: float = 20.0
    pad_travel_n: float = 0.004
    pad_travel_t: float = 0.002
    pad_friction: float = 1.3
    pad_torsion: float = 0.02  # m; torsional friction of the soft pad contact patch
    pad_solref: tuple[float, float] = (0.002, 1.0)
    pad_solimp: tuple[float, float, float] = (0.9, 0.95, 0.001)
    # soft-contact-only pads (pad_mode="soft_contact")
    soft_pad_solref: tuple[float, float] = (0.004, 0.6)
    # Menagerie uses 1 Ns/m behind its position servo (kv 10); with a force-controlled drive in its place the
    # real hand's non-backdrivable spindle is represented by heavy joint damping instead
    finger_joint_damping: float = 300.0
    # --- dexterous hands (hand != "franka") ---
    hand_rate: float = 1000.0  # joint control rate (WUJI Hand 2: 1 kHz MIT mode)
    mount_yaw: float = 0.0  # rad, hand about the flange z after the mount flip
    wrap_tcp: tuple[float, float, float] = (-0.004, 0.0315, -0.086)  # handle grasp point, hand frame
    hand_joint_damping: float = 0.002  # Nms/rad, drive + gearbox viscous loss
    hand_joint_friction: float = 0.005  # Nm, drive friction
    hand_friction: float = 1.0  # skin on the handle
    hand_solref: tuple[float, float] = (0.002, 1.0)
    # thumb joints (CMC flex, CMC abd, MCP, IP): closing torque sign, 0 = held at thumb_preshape by PD
    thumb_close: tuple[int, int, int, int] = (1, -1, 1, 1)
    thumb_preshape: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # "backdrivable": the joints are plain torque motors (the vendor model). "self_locking": WUJI's drives do
    # not backdrive: a joint closes under its motor but external load cannot open it (nor move a shaping
    # joint); the gearbox then carries the load, reported as hand_stop_load.
    lock_mode: str = "backdrivable"
    # taxel patches: "taxelscan" (TaxelScan Rev3 skins conforming to the palm and to every finger's distal pad and
    # middle segment: 128 + 10 x 32 = 448 taxels), "palm_thumb" (2 flat 8 x 8 patches) or "palm"
    patch_layout: str = "palm_thumb"


@dataclass
class HammerCfg:
    model: str = "auto"  # "ycb" (YCB 048_hammer scan), "primitive" (cylinder + capsule), "auto"
    head_mass: float = 0.45
    head_radius: float = 0.0125
    head_half_len: float = 0.05  # along the strike axis
    handle_radius: float = 0.014
    handle_len: float = 0.28
    handle_mass: float = 0.215  # YCB 048_hammer: 665 g total
    grip_from_head: float = 0.19  # grasp centre to head axis: the flat, widest part of the YCB handle
    face_solref: tuple[float, float] = (0.001, 0.4)
    face_solimp: tuple[float, float, float] = (0.95, 0.99, 0.001)
    board_solref: tuple[float, float] = (0.002, 0.8)


@dataclass
class NailPlantCfg:
    kind: str = "nail"  # "nail" (hammer), "saw" (SawCfg) or "drill" (driver/drill, DrillCfg)
    board_half: tuple[float, float, float] = (0.08, 0.02, 0.12)  # (along handle, thickness, height)
    proud: float = 0.025  # nail head standing proud of the board surface at start
    drive_target: float = 0.020  # "done" depth
    head_radius: float = 0.0045
    head_half_h: float = 0.00075
    shank_radius: float = 0.0015
    nail_mass: float = 0.008
    resistance_0: float = 450.0  # N, Coulomb resistance at zero added depth
    resistance_per_m: float = 1.0e4  # N per metre of added depth
    damping: float = 20.0  # Ns/m, rate-dependent crushing
    limit_solref: tuple[float, float] = (0.0005, 1.0)
    friction_solref: tuple[float, float] = (0.0005, 1.0)
    # velocity-dependent restitution: dampratio = zeta0 + zeta1 * approach speed (clipped)
    vdr_enabled: bool = True
    vdr_zeta0: float = 0.3
    vdr_zeta1: float = 0.1
    vdr_zeta_max: float = 1.0


@dataclass
class SawCfg:
    """Hand saw and workpiece. The saw's grasp frame is the tool frame: x along the handle and blade (the stroke),
    y toward the teeth (the cut, along the fingers' closing axis), z across the blade (the hand's approach). The
    hand holds the handle palm-sideways, fingers above and below it, as a person does, so the stroke's pitching
    moments load the grip as a friction couple about the handle rather than as twist about the pad normal. The
    workpiece is a board lying on a table under the blade, crosscut through its thickness (a 2x4)."""
    handle_radius: float = 0.014
    handle_len: float = 0.11
    handle_mass: float = 0.15
    blade_x: tuple[float, float] = (-0.12, 0.16)  # blade extent along the handle axis (tool frame): under the grip
    blade_depth: float = 0.07  # handle axis to teeth (the blade starts below the lower finger)
    blade_height: float = 0.03
    blade_thickness: float = 0.0009
    blade_mass: float = 0.25
    kerf_width: float = 0.0016
    board_width: float = 0.089  # along the stroke
    board_thickness: float = 0.038  # along the cut
    board_length: float = 0.30  # across the blade
    clearance: float = 0.005  # teeth above the board at the hover pose
    cut_target: float = 0.020
    # cutting model: normal support k_n, tangential/normal force ratio, removal rate depth += F_n * |ds| / k_cut on the
    # cutting stroke (push for a western saw), kerf walls k_lat beyond the side clearance, binding friction mu_bind
    k_normal: float = 2.0e4
    d_normal: float = 60.0
    mu_cut: float = 1.1
    drag: float = 1.0  # N, tooth drag on either stroke while in contact
    k_cut: float = 1800.0  # N per (m of cut per m of stroke): ~1.9 mm per 0.14 m cutting stroke at 25 N
    cut_on: str = "push"  # "push" (western) or "pull" (Japanese)
    k_lateral: float = 3.0e4
    d_lateral: float = 40.0
    mu_bind: float = 0.6
    knot: tuple[float, float, float] = (0.0, 0.0, 1.0)  # (depth from, depth to, removal-resistance multiplier)
    # scripted stroke (stand-in for L2)
    stroke_amp: float = 0.07
    stroke_freq: float = 1.2
    push_force: float = 25.0
    approach_time: float = 0.6
    k_stroke: tuple[float, float, float] = (1500.0, 800.0, 150.0)  # (stroke, across, cut) N/m


@dataclass
class DrillCfg:
    """Inline cordless driver (or drill), held the way robots hold screwdrivers: the body between the fingers with
    the bit along the hand's approach axis (tool +z), so the push runs through the arm rather than the wrist. The
    target is a board lying on a table under the bit: a pre-started wood screw ("screw") or bare wood ("hole")."""
    mode: str = "screw"
    handle_radius: float = 0.018
    body_span: tuple[float, float] = (-0.10, 0.04)  # along the bit, tool frame
    body_mass: float = 0.8
    bit_len: float = 0.085  # chuck face (handle front) to bit tip
    clearance: float = 0.01  # bit tip short of the screw head / board at the hover pose
    board_thickness: float = 0.038
    board_half: tuple[float, float] = (0.12, 0.12)
    # motor and clutch: tau = stall * (trigger - w / w_free), clutch slips above clutch_torque with n_detents per rev
    stall_torque: float = 8.0
    free_speed: float = 50.0  # rad/s (~480 rpm: low gear, for driving screws)
    hole_free_speed: float = 140.0  # rad/s (~1340 rpm: high gear, for drilling)
    rotor_inertia: float = 2.0e-4
    trigger_slew: float = 6.0  # per second: the variable-speed trigger's soft start
    brake_torque: float = 0.4  # Nm: the electronic brake when the trigger is released
    spindle_drag: float = 1.0e-3
    clutch_torque: float = 1.8
    clutch_detents: int = 6
    # screw (#8 wood screw in pine, pre-started): torque grows with depth, rises steeply once the head seats
    screw_len: float = 0.022  # head proud of the surface at the start
    pitch: float = 0.0021
    screw_torque0: float = 0.25
    screw_torque_per_m: float = 40.0
    seat_stiffness: float = 3000.0  # Nm per m past flush
    engage_radius: float = 0.002  # bit tip within this of the screw axis to engage the recess
    cam_ratio: float = 0.03  # cam-out when torque > cam_ratio * axial push (Phillips)
    cam_max_angle: float = 0.26  # rad of bit misalignment at which cam-out needs no torque
    cam_time: float = 0.015
    cam_kick: float = 15.0  # N pushing the bit back out of the recess while it cams out
    strip_after: int = 6  # cam-outs before the recess strips
    # hole drilling: thrust supports the tip; feed per rev grows with thrust above f0; torque with the feed
    thrust_f0: float = 20.0
    feed_stiffness: float = 1.5e5  # N per (m/rev)
    drill_torque0: float = 0.15
    drill_torque_per_feed: float = 2500.0  # Nm per (m/rev)
    exit_len: float = 0.002  # support fades over the last exit_len; the bit grabs (catch) on the way out
    catch_gain: float = 2.5
    catch_feed: float = 0.0003  # m/rev: the flutes pull the bit through the last exit_len
    # contact between the bit and the screw head / hole bottom
    k_axial: float = 4.0e4
    d_axial: float = 80.0
    k_lateral: float = 2.0e4
    d_lateral: float = 30.0
    # scripted feed (stand-in for L2)
    push_force: float = 70.0
    approach_time: float = 0.5
    k_feed: tuple[float, float] = (150.0, 2500.0)  # (along the bit, across it) N/m
    trigger_contact: float = 15.0  # N of axial push before the trigger is pulled


@dataclass
class SceneCfg:
    # Grasp/TCP frame: hand points down (approach = -z), fingers close along world y, handle along
    # world x. The strike is horizontal along +y into a vertical board; the nail is placed so the
    # hammer face is `hover_clearance` short of the nail head at the hover pose.
    # Chosen for the FR3's limits: here the joint torque limits can push 123 N along the strike axis (the wrist
    # joints' 12 Nm bind first) and the joint velocity limits allow 2.6 m/s, with every joint >= 0.68 rad from
    # its range limits
    hover_tcp: tuple[float, float, float] = (0.50, 0.0, 0.20)
    hover_clearance: float = 0.06
    strike_yaw: float = 0.0  # rad: the task (strike axis, board, grasp frame) turned about the vertical
    # or, set explicitly: the strike direction (into the board) and the handle direction (from the head toward
    # the grip), e.g. strike_dir (0, 0, -1) for a nail driven down into a board lying on a table
    strike_dir: tuple[float, float, float] | None = None
    handle_dir: tuple[float, float, float] = (1.0, 0.0, 0.0)
    table: bool = True  # a table under the board when it lies flat (strike within 30 deg of vertical)
    table_half: tuple[float, float] = (0.35, 0.45)  # tabletop half extents (m)


@dataclass
class SensorsCfg:
    ft_rate: float = 4000.0
    ft_latency: float = 0.0005
    ft_bandwidth: float = 2000.0
    ft_noise_f: float = 0.1
    ft_noise_t: float = 0.005
    ft_range_f: float = 500.0
    ft_range_t: float = 20.0
    accel_rate: float = 8000.0
    accel_latency: float = 0.00025
    accel_bandwidth: float = 3500.0
    accel_noise: float = 0.05
    accel_range: float = 16 * 9.80665
    taxel_grid: tuple[int, int] = (8, 8)  # per pad (rows along the handle, columns along the finger)
    pressure_rate: float = 1000.0
    pressure_latency: float = 0.002
    pressure_bandwidth: float = 300.0
    pressure_noise: float = 0.1  # N per taxel
    pressure_range: float = 20.0  # N per taxel
    # --- TaxelScan Rev3 (patch_layout "taxelscan"): piezoresistive skins read by RP2350 boards. Each board
    # scans its taxels one at a time through its 12-bit SAR ADC (500 ksps rated), one full frame per 1 ms; a
    # taxel's value is the force at the instant the ADC sampled it, so a frame is skewed by the scan.
    ts_palm_grid: tuple[int, int] = (8, 16)  # rows along the fingers, columns across the palm
    ts_finger_grid: tuple[int, int] = (8, 4)  # rows along the segment, columns across it
    ts_rate: float = 1000.0  # frames per second, every board
    ts_boards: str = "per_patch"  # "per_patch" (11 boards) or "hand" (one board scans all 448 taxels)
    ts_adc_rate: float = 500e3  # RP2350 ADC conversions per second
    ts_adc_bits: int = 12
    ts_enob: float = 9.2  # RP2350 datasheet effective bits
    ts_oversample: int = 1  # conversions averaged per taxel
    ts_settle: float = 1e-6  # mux + RC settling before each taxel's first conversion (s)
    ts_f_half: float = 5.0  # N at which a taxel's divider reads half scale: counts ~ F / (F + f_half)
    ts_gain_mismatch: float = 0.05  # per-taxel sensitivity error left after calibration (1 sigma)
    ts_offset_lsb: float = 2.0  # per-taxel offset left after taring (1 sigma, LSB)
    ts_latency: float = 0.001  # frame done -> host (USB full-speed 1 ms polling)
    ts_bandwidth: float = 300.0  # elastomer skin (Hz)
    ts_spread: float = 0.0015  # sigma of a contact's spreading when the foundation model finds no tool (m)
    ts_skin_depth: float = 0.002  # compressible skin depth: taxels within this of the deepest press carry load (m)
    ts_shape_dt: float = 0.0005  # load shape refresh period (s); the total follows the contacts every step
    ts_shape_radius: float = 0.04  # taxels this close to a contact are tested against the tool (m)
    joint_rate: float = 1000.0
    joint_latency: float = 0.0
    joint_pos_quant: float = 2.0**-14
    joint_vel_noise: float = 0.002
    joint_tau_noise: float = 0.02
    tau_ext_rate: float = 1000.0
    cameras: bool = False
    camera_rate: float = 30.0


@dataclass
class ControllerCfg:
    rate: float = 1000.0
    grip_rate: float = 500.0
    k_trans: tuple[float, float, float] = (3000.0, 3000.0, 3000.0)
    k_rot: tuple[float, float, float] = (150.0, 150.0, 150.0)
    zeta: float = 1.0
    k_dot_max: float = 5.0e4  # N/m/s stiffness slew limit
    delta_max_pos: float = 0.05  # reference limiter
    delta_max_rot: float = 0.3
    f_ff_max: tuple[float, float] = (80.0, 8.0)
    k_null: float = 10.0
    d_null: float = 2.0
    osc_inertia: bool = True
    gate_duration: float = 0.050
    vel_guard_onset: float = 0.6  # post-impact / hold: joint damping above this fraction of the velocity limit
    stale_timeout: float = 0.020
    observer_gain: float = 400.0
    impact_force_thresh: float = 30.0
    impact_ft_thresh: float = 35.0  # N departure of the strike-axis wrist force from its 10 ms baseline
    impact_accel_thresh: float = 60.0  # m/s^2 pad-accelerometer deviation from its 5 ms baseline
    impact_refractory: float = 0.100
    grip_kp: float = 0.2
    grip_ki: float = 15.0
    grip_hold: float = 55.0  # N per pad; Franka Hand continuous rating is 70 N
    drop_force_frac: float = 0.2
    drop_force_abs: float = 1.5  # N; below the array's noise floor a fraction of a tiny setpoint is meaningless
    drop_accel: float = 50.0
    # a blow shakes the patches and can unload one of them for a few ms; a dexterous hand's drop check
    # ignores the accelerometer for this long after an impact flag (0 = never)
    drop_impact_holdoff: float = 0.0


@dataclass
class SwingCfg:
    rate: float = 200.0
    approach_time: float = 0.6
    first_tap_speed: float = 1.2  # m/s; a setting tap teaches the aim loop the swing's drift (0 = no tap)
    windup_height: float = 0.15
    windup_time: float = 0.45
    v_strike: float = 2.2  # m/s; drives 20 mm in 9 strikes at the default nail resistance
    overshoot: float = 0.03  # follow-through past the predicted contact when brake_decel = 0
    # Hardware-feasible striking (see tactile_sim.limits): the swing peaks just before contact and arrives
    # braking at brake_decel, so the torque reversal is under way when the blow lands; the strike speed is
    # capped at v_margin of the fastest speed the FR3's joint velocity limits allow at the strike pose.
    a_max: float = 25.0  # m/s^2 peak swing acceleration
    brake_decel: float = 15.0  # m/s^2 at contact (0 = the old accelerate-through-contact swing)
    v_margin: float = 0.8
    # 0 = straight-line strike with the tool's orientation held; > 0 = arc strike: the tool turns about a pivot this
    # far behind the face along the handle, so the elbow and shoulder whip the head (StrikeGeometry)
    arc_radius: float = 0.0
    strike_k: tuple[float, float, float] = (3000.0, 3000.0, 4000.0)  # (across, across, along) the strike axis
    recover_time: float = 0.4
    settle_time: float = 0.15
    grip_lead: float = 0.155  # ramp start before predicted contact (>= 150 ms before the actual one)
    grip_ramp_end: float = 0.050
    grip_peak_delay: float = 0.060
    grip_decay: float = 0.200
    grip_pre_gain: float = 12.0  # N per (kg m/s) of tool momentum
    grip_margin_step: float = 10.0
    strike_timeout: float = 0.100
    n_strikes: int = 10


@dataclass
class DRCfg:
    enabled: bool = False
    head_mass_scale: tuple[float, float] = (0.8, 1.2)
    resistance_0: tuple[float, float] = (200.0, 900.0)
    pad_friction: tuple[float, float] = (0.9, 1.5)
    pad_k_n_scale: tuple[float, float] = (0.6, 1.5)
    face_timeconst: tuple[float, float] = (0.001, 0.003)
    vdr_zeta0: tuple[float, float] = (0.2, 0.5)
    sensor_noise_scale: tuple[float, float] = (0.5, 2.0)
    observer_gain_scale: tuple[float, float] = (0.5, 1.5)


@dataclass
class LoggingCfg:
    truth_decim: int = 1
    compress: bool = True


@dataclass
class SimConfig:
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    arm: ArmCfg = field(default_factory=ArmCfg)
    gripper: GripperCfg = field(default_factory=GripperCfg)
    hammer: HammerCfg = field(default_factory=HammerCfg)
    plant: NailPlantCfg = field(default_factory=NailPlantCfg)
    saw: SawCfg = field(default_factory=SawCfg)
    drill: DrillCfg = field(default_factory=DrillCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)
    sensors: SensorsCfg = field(default_factory=SensorsCfg)
    controller: ControllerCfg = field(default_factory=ControllerCfg)
    swing: SwingCfg = field(default_factory=SwingCfg)
    dr: DRCfg = field(default_factory=DRCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimConfig:
        return _from_dict(cls, d)

    def replace(self, **sections: dict[str, Any]) -> SimConfig:
        """Return a copy with some fields overridden, e.g. cfg.replace(arm={"flex_mode": "rigid"})."""
        d = self.to_dict()
        for sec, vals in sections.items():
            if sec not in d:
                raise KeyError(sec)
            for k in vals:
                if k not in d[sec]:
                    raise KeyError(f"{sec}.{k}")
            d[sec].update(vals)
        return SimConfig.from_dict(d)


def _from_dict(tp, d):
    kwargs = {}
    for f in dataclasses.fields(tp):
        if f.name not in d:
            continue
        v = d[f.name]
        ftype = f.type if not isinstance(f.type, str) else None
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(default) and isinstance(v, dict):
            kwargs[f.name] = _from_dict(type(default), v)
        elif isinstance(v, list):
            kwargs[f.name] = tuple(v)
        else:
            kwargs[f.name] = v
        del ftype
    return tp(**kwargs)


WUJI2_OVERRIDES: dict[str, dict[str, Any]] = {
    "gripper": {"hand": "wuji2", "wrap_tcp": (0.004, 0.0315, -0.078), "thumb_close": (1, 0, 1, 1),
                "thumb_preshape": (0.0, -0.8, 0.0, 0.0), "mount_yaw": 3.141592653589793,
                "patch_layout": "taxelscan"},
    # the hand points along the strike axis. At this hover pose the FR3's joint velocity limits allow 2.6 m/s
    # along it (1.3 m/s at the Franka Hand's hover pose, where the strike is an elbow extension) and every
    # joint is >= 0.94 rad from its limits
    "arm": {"q_seed": (0.74, 0.14, -0.58, -1.8, 1.48, 1.49, -0.24)},
    "scene": {"hover_tcp": (0.52, 0.30, 0.50)},
    # grip force = summed normal force on the taxel patches (TaxelScan: palm + fingertips carry the load in this
    # wrap, ~89 N at the keyframe, close to the 95 N the palm + thumb patches saw, so the setpoints carry over)
    # two patches see only part of a wrap's load, and the share moves when the tool shifts a few degrees or the
    # swing loads the fingers: a drop is the patches going empty, not falling below a fraction of the setpoint
    "controller": {"grip_hold": 40.0, "drop_impact_holdoff": 0.05, "drop_force_frac": 0.0},
}


# Dexmate Vega: a forward strike with the right arm. Hover pose, strike direction and arc radius were searched for
# the fastest strike the arm's joint velocity and torque limits allow: ~1.0-1.4 m/s at best anywhere in reach
# (2.6 m/s for the FR3); the arc gives this pose twice the torque headroom of a straight strike. Both Vegas
# use the same arm, placed identically relative to the shoulders (arm_center).
_VEGA_STRIKE = {"swing": {"arc_radius": 0.8}}
VEGA_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    # Vega U: fixed pedestal, lift down (shoulders 1.24 m up), torso upright. The nail stands upright in a board
    # lying on a table and is driven straight down, the handle across the body; the hammer swings on an arc
    # (pivot 0.6 m behind the face), its head rising 0.20 m along it at the windup, so the tool turns 25 deg
    # through the swing. Searched over hover pose, handle direction and arc radius: the joint velocity limits
    # allow 1.5 m/s here (1.9 m/s on a flatter 0.8 m arc; 1.2 m/s for the best horizontal strike); the windup is
    # as long as the wrist roll's range allows.
    "vega_1u": {
        "arm": {"robot": "vega_1u", "flex_mode": "rigid", "vega_lift": 0.0, "vega_flip": 0.0,
                "q_seed": (0.63, -0.42, -2.55, -1.55, 1.42, -0.4, 0.94)},
        "scene": {"hover_tcp": (0.50, -0.10, 1.05), "strike_dir": (0.0, 0.0, -1.0), "handle_dir": (0.0, -1.0, 0.0)},
        "swing": {"arc_radius": 0.6, "windup_height": 0.20},
    },
    # Vega-1P: wheeled base locked, torso standing upright (shoulders 1.35 m up)
    "vega_1p": {
        "arm": {"robot": "vega_1p", "flex_mode": "rigid", "torso_pose": (0.6, 1.2, 0.6),
                "q_seed": (-1.32, -0.46, 0.18, -1.78, -0.81, 0.27, 0.72)},
        "scene": {"hover_tcp": (0.311, -0.30, 1.172), "strike_yaw": -1.5707963267948966},
        **_VEGA_STRIKE,
    },
}


def hand_config(hand: str, base: SimConfig | None = None, robot: str = "fr3",
                **sections: dict[str, Any]) -> SimConfig:
    """Config for a hand on an arm: hand "franka" (the default testbed) or "wuji2" (WUJI Hand 2 power wrap),
    robot "fr3", "vega_1u" (Dexmate Vega U) or "vega_1p" (Dexmate Vega-1P); on a Vega, WUJI hands on both arms,
    the right one striking."""
    cfg = base or SimConfig()
    if hand == "wuji2":
        cfg = cfg.replace(**WUJI2_OVERRIDES)
    elif hand != "franka":
        raise ValueError(f"unknown hand {hand!r}")
    if robot in VEGA_OVERRIDES:
        if hand != "wuji2":
            raise ValueError("the Vega is modelled with WUJI hands only")
        cfg = cfg.replace(**VEGA_OVERRIDES[robot])
    elif robot != "fr3":
        raise ValueError(f"unknown robot {robot!r}")
    return cfg.replace(**sections) if sections else cfg


# Saw and driver tasks on the FR3 with the Franka Hand; both work on a board lying on a table.
TASK_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    # saw: fingers close vertically (TCP y down), the hand approaches along +y, the stroke runs along +x
    "saw": {"plant": {"kind": "saw"}, "scene": {"hover_tcp": (0.48, 0.0, 0.30), "strike_dir": (0.0, 0.0, 1.0),
                                                  "handle_dir": (1.0, 0.0, 0.0)},
            "controller": {"gate_duration": 0.0, "grip_hold": 60.0}},
    # driver: the default hand-down grasp frame; the bit points down into a board on a table
    "drill": {"plant": {"kind": "drill"}, "scene": {"hover_tcp": (0.50, 0.0, 0.30)},
              "controller": {"gate_duration": 0.0, "grip_hold": 60.0}},
}


def task_config(task: str, base: SimConfig | None = None, **sections: dict[str, Any]) -> SimConfig:
    """Config for a tool task: "nail" (the hammer default), "saw" or "drill"."""
    cfg = base or SimConfig()
    if task != "nail":
        if task not in TASK_OVERRIDES:
            raise ValueError(f"unknown task {task!r} (expected 'nail', 'saw' or 'drill')")
        cfg = cfg.replace(**TASK_OVERRIDES[task])
    return cfg.replace(**sections) if sections else cfg


def fast_config(**sections: dict[str, Any]) -> SimConfig:
    """Config used by the test suite: 0.25 ms physics (4 kHz) and the pure-Python-friendly rates."""
    cfg = SimConfig().replace(
        physics={"timestep": 0.00025},
        sensors={"ft_rate": 2000.0, "accel_rate": 4000.0, "accel_bandwidth": 1500.0, "ft_bandwidth": 800.0},
    )
    return cfg.replace(**sections) if sections else cfg
