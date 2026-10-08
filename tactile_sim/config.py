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
    # MuJoCo's multi-contact convex collisions (on by default). MuJoCo Warp gives capsule/mesh and cylinder/cylinder
    # pairs (the light hammer's handle against the hand, its face against the nail) one contact whatever this says;
    # off on both sides, the CPU and the GPU find the same contacts (tactile_rl/README.md)
    multiccd: bool = True


@dataclass
class ArmCfg:
    robot: str = "fr3"  # "fr3" (Franka FR3, 1 kHz torque control), "vega_1u" / "vega_1p" (Dexmate Vega, right arm)
    source: str = "auto"  # "fr3" (Menagerie, required), "fallback" (capsules), "auto" (fr3 if cached)
    flex_mode: str = "wrist"  # "rigid" or "wrist"
    wrist_k_rot: float = 1500.0  # Nm/rad, hinges about the flange x and y axes
    wrist_d_rot: float = 1.5  # Nms/rad
    wrist_k_lin: float = 2.0e5  # N/m along the flange z axis
    wrist_d_lin: float = 150.0  # Ns/m
    wrist_range: float = 0.1  # rad, travel of each wrist hinge (a soft wrist needs more before it hits its stop)
    # what the arm controller servoes: "tcp", the hand's TCP, through the wrist compliance; "flange", the TCP's
    # zero-deflection pose carried by the F/T body, so the compliance is a passive element after the flange (a soft
    # wrist servoed through its own spring oscillates)
    wrist_control: str = "tcp"
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
    # idle left arm hanging at the side: shoulder pitched 90 deg down, 10 deg out from the hip, elbow bent 20 deg,
    # palm toward the thigh (49 mm clear of the body). Dexmate's raised "L_shape" pose is
    # (0.064, 0.3, 0.0, -1.556, 1.271, 0.0, 0.0).
    left_arm_pose: tuple[float, ...] = (1.5708, 0.1745, 0.0, -0.3491, 0.0, 0.0, 0.0)
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
    # first-order lag between the grip command and the grip the drives apply (s); 0 = instant, as the force-
    # controlled drive and the MIT-mode joint law are modelled (real grippers take 20-185 ms to change force)
    grip_lag: float = 0.0
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
    # finger closing torques: "rated" (every joint at its full rating: the fingers fold at the knuckle, the handle is
    # clamped against the palm by the proximal segments and fingertips, the middle segments bridge it) or "tendon"
    # (the human flexor-tendon ratio MCP : PIP : DIP = 1 : 0.73 : 0.25 with the fingertip joint following the
    # middle joint: a human-like wrap that loads the middle segments, but capped by the WUJI's 0.3 Nm PIP it is too
    # weak to hold the handle in the palm, and the blows drive far less); see tactile_sim.control.hand
    finger_synergy: str = "rated"
    # taxel patches: "taxelscan" (TaxelScan Rev3 skins conforming to the palm and to every finger's distal pad and
    # middle segment: 128 + 10 x 32 = 448 taxels), "palm_thumb" (2 flat 8 x 8 patches) or "palm"
    patch_layout: str = "palm_thumb"
    # joint drives: "synergy" (the grip scalar sets each closing joint's torque at 1 kHz, written straight to the
    # motors) or "mit" (WUJI Hand 2 MIT-mode drives, WujiDriveCfg / tactile_sim.sim.wuji_drive: per-joint
    # (q*, qd*, Kp, Kd, tau_ff) commands pass the SDK's 1 kHz cycle and a transport delay to a driver that runs the
    # servo law, clamps and derates the current and lags it through its current loop). In "mit" the grip synergy
    # still drives the hand, as MIT commands, unless an external hand command is set (WujiHandIO.set_hand_command)
    hand_control: str = "synergy"
    # tool masses the grasp keyframe is synthesized and cached for: "actual" (the configured masses, one grasp per
    # mass) or "nominal" (the tool model's nominal masses, so randomized masses share one grasp and the reset settle
    # seats the tool under its actual mass)
    grasp_mass: str = "actual"


@dataclass
class WujiDriveCfg:
    """WUJI Hand 2 joint drives in MIT mode (GripperCfg.hand_control = "mit") as the host sees them through the SDK
    realtime controller (PDO, 1 kHz): per joint the driver applies tau* = Kp (q* - q) + Kd (qd* - qd) + tau_ff,
    clamps the current, and its current loop and windings respond with the time constants below.

    WUJI publishes none of these internals: every value marked (assumed) is a placeholder until the hand is
    characterized, and is randomized in training. Per-group tuples follow wuji_drive.JOINT_GROUPS:
    (MCP flexion, MCP abduction, PIP/DIP, thumb CMC, thumb MCP/IP)."""

    sdk_rate: float = 1000.0  # Hz: the SDK latches the host's latest command and publishes the drives' state
    cmd_latency: float = 0.001  # s, host command -> driver (transport + pickup) (assumed)
    cmd_jitter: float = 0.00025  # s, uniform +- per command; commands are never reordered (assumed)
    fb_latency: float = 0.0005  # s, driver state -> host (assumed)
    # the realtime controller's first-order low-pass on the host command (None = off), on the fields named in
    # sdk_lpf_fields: any of "q", "qd", "kp", "kd", "tau"
    sdk_lpf_hz: float | None = None
    sdk_lpf_fields: tuple[str, ...] = ("q",)
    servo_rate: float | None = None  # Hz the driver evaluates the MIT law at; None = every physics step
    tau_i: float = 0.0002  # s, current-loop time constant (FOC) (assumed)
    kp_nom: tuple[float, ...] = (0.37, 0.43, 0.22, 0.55, 0.22)  # Nm/rad, nominal Kp: the vendor MJCF gains (assumed)
    zeta: float = 0.8  # damping ratio when Kd is derived from Kp: Kd = 2 zeta sqrt(Kp I)
    i_rated: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)  # A at each joint's rated torque (assumed)
    i_peak_ratio: float = 2.0  # the driver's peak current / rated (assumed)
    effort_limit_ratio: float = 1.0  # SDK effort_limit (the current ceiling) / rated current
    r_wind: float = 2.0  # ohm, winding resistance (assumed)
    r_th: float = 10.0  # K/W, winding to ambient (assumed)
    c_th: float = 5.0  # J/K, winding heat capacity (assumed)
    t_amb: float = 25.0  # degC
    t_warn: float = 70.0  # degC: the current ceiling derates linearly from here ...
    t_max: float = 90.0  # degC: ... to derate_floor here, where the driver faults (torque off until reset) (assumed)
    derate_floor: float = 0.5  # current ceiling fraction just below t_max (assumed)
    enc_bits: int = 14  # encoder resolution, bits per revolution (readback quantization) (assumed)
    i_noise: float = 0.01  # A rms on the current readback (assumed)
    i_quant: float = 0.001  # A, current readback resolution (assumed)


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
    # "inline": the driver the Franka Hand holds, the bit along the grasp's approach axis. "pistol": a pistol-grip
    # drill/driver held in a power wrap the way a hammer is: the grip is the handle, the motor housing the head, and
    # the bit points where the hammer's face does (tactile_sim.model.tools.add_pistol_driver), so the WUJI Hand 2's
    # wrap grasp, contacts and slip metrics carry over
    form: str = "inline"
    pistol_grip_radius: float = 0.016
    pistol_grip_len: float = 0.11  # along the grip, from the housing's underside to the battery
    pistol_grip_from_head: float = 0.055  # grasp centre to the housing (bit) axis, along the grip
    pistol_housing_radius: float = 0.028
    pistol_housing_back: float = 0.07  # housing behind the grip, away from the bit
    pistol_housing_front: float = 0.09  # housing ahead of the grip, to the chuck
    pistol_chuck_len: float = 0.045
    pistol_housing_mass: float = 0.55  # motor, gearbox, clutch, chuck: 1.12 kg with the grip and battery
    pistol_grip_mass: float = 0.12
    pistol_battery_mass: float = 0.45
    pistol_battery_half: tuple[float, float, float] = (0.035, 0.05, 0.03)  # box beyond the grip's end
    # how the WUJI wraps the pistol's grip: "side" (as the hammer's handle: the palm along the grip's side, the bit
    # toward the fingertips) or "pistol" (as a person holds a drill: the palm behind the grip, the bit pointing out
    # from it, the push into the palm, the index finger round the grip's front where the trigger is)
    pistol_grasp: str = "side"
    # the pistol's trigger. "script": the base sets the motor's trigger level itself (no trigger part). "finger": a
    # sprung slide trigger on the grip's front, where the WUJI wrap lays the index finger's proximal segment; the
    # index finger closes on it with its own level (trigger_rest / trigger_full below), and the motor runs on the
    # trigger's travel past a dead band. The trigger sits trigger_x below the grip's top, trigger_angle around
    # the grip from the front (-y) toward +z, and slides radially into the grip
    trigger_mode: str = "script"
    trigger_x: float = 0.002  # m
    trigger_angle: float = 0.32  # rad
    trigger_half: tuple[float, float, float] = (0.0045, 0.0035, 0.007)  # m: along the grip, radial, across
    trigger_friction: float = 0.05  # skin on the trigger's smooth face (more and the finger's edge contact sticks)
    trigger_travel: float = 0.006  # m
    trigger_gap: float = 0.001  # m between the fully pulled trigger and the grip
    trigger_force: tuple[float, float] = (8.0, 15.0)  # N of spring at rest and at full travel
    trigger_damping: float = 1.0  # Ns/m
    trigger_dead: float = 0.2  # fraction of the travel before the motor runs
    # the index finger's closing level on the trigger (its share of the grip synergy's torque, apart from the grip
    # level): resting on it, and at a full pull
    trigger_rest: float = 0.15
    trigger_full: float = 0.8
    trigger_pull_time: float = 0.15  # s for the base to go from resting to a full pull
    trigger_release_time: float = 0.25  # s the finger stays fully relaxed when the base lets go of a pull
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
    engage_radius: float = 0.003  # bit tip within this of the screw axis to engage the recess (a #2 Phillips
    # recess, ~6 mm across, steers a pressed bit that lands within about its radius into the cross)
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
    # position-interface arm (Vega U), no wrist F/T: the push comes from commanding the bit press_depth past where it
    # is, and the trigger is pulled trigger_delay after the press has ramped up over press_time
    press_depth: float = 0.006
    press_time: float = 0.2
    trigger_delay: float = 0.1
    trigger_level: float = 1.0  # how far the base pulls the variable-speed trigger (0..1)
    release_delay: float = 0.3  # s the trigger stays pulled after the tool reports done (the clutch ratchets)
    # aim the bit from where the tool settled in the hand (one look before the task), not from the grasp keyframe:
    # turn the hand so the bit lines up with the work and shift the path onto its axis (a degree of tool rotation in
    # the hand moves the tip ~4 mm, twice the screw recess's catch, and a tilted bit cams out of the recess sooner)
    aim_from_tool: bool = True
    # holes on a position arm: past the press the target advances at feed_rate (feed-rate control, as a CNC drills;
    # chasing the bit with a stiff servo turns the feed law's thrust-feed coupling into a runaway)
    feed_rate: float = 0.003  # m/s
    # bounds on the tip's forces: a bit knocked off the axis must not drive the contact springs to huge loads
    max_axial: float = 1500.0  # N
    max_lateral: float = 300.0  # N


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
    # the reduced training scene (model.builder.build_reduced_scene; Vega U + WUJI Hand 2 + hammer + nail only): the
    # striking arm, its hand, the tool and the plant, without the left arm, the left hand, the head or visual-only
    # meshes. The GPU training env (tactile_rl) is built from the same MJCF, so the CPU testbed can run its exact model
    reduced: bool = False


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
    # --- readout model. "rev2": a board per patch (or one per hand, ts_boards), each frame visible ts_latency after
    # its scan ends. "rev3": the 1 kHz fast profile (docs/taxelscan_fast_profile.md): patches grouped onto boards of
    # at most ts_max_per_board taxels (ts_board_map), every board scanning from the hardware SYNC edge; the master
    # polls the frames one period later, in board order, and the host gets mu-law codes plus patch features
    ts_model: str = "rev2"
    ts_board_map: tuple[tuple[str, ...], ...] = (
        ("palm",), ("thumb_middle", "thumb_distal", "index_middle", "index_distal"),
        ("middle_middle", "middle_distal", "ring_middle", "ring_distal"), ("pinky_middle", "pinky_distal"))
    ts_max_per_board: int = 128
    ts_taxel_time: float | None = None  # s per taxel; None = ts_settle + ts_oversample / ts_adc_rate
    ts_bus_baud: float = 12.5e6  # RS-485, 8N1
    ts_turnaround: float = 1e-5  # s per board poll: request plus bus turnaround
    ts_poll_lead: float = 5e-5  # s from the SYNC edge to the first poll
    ts_bridge_latency: tuple[float, float] = (1.25e-4, 1e-3)  # s, master -> host, uniform (USB full speed)
    ts_drop_prob: float = 0.0  # probability a frame never reaches the host
    ts_code_bits: int = 8  # taxel map codes: 8 (U8) or 6 (U6)
    ts_mu: float = 255.0
    ts_f_max: float = 30.0  # N, the top code
    ts_contact_mn: int = 50  # mN, a taxel at or above counts toward the patch's contact area
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
    # "closed": PI on the measured grip (pads / taxel patches); "open": the setpoint is the commanded effort. On a
    # wrap the patch sum includes the tool's weight and swing loads on the palm, so a closed loop squeezes less
    # when those press in
    grip_loop: str = "closed"
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
    # channels the L1 impact detector may use: "observer" (joint-torque momentum observer, FR3 only), "ft" (wrist
    # F/T), "accel" (patch accelerometers). The real Vega U + WUJI hand has none of them: () disables the flag
    impact_sources: tuple[str, ...] = ("observer", "ft", "accel")
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
    strike_k_rot: float = 250.0  # Nm/rad, rotational stiffness during the swing
    recover_time: float = 0.4
    settle_time: float = 0.15
    grip_lead: float = 0.155  # ramp start before predicted contact (>= 150 ms before the actual one)
    grip_ramp_end: float = 0.050
    grip_peak_delay: float = 0.060
    grip_decay: float = 0.200
    grip_pre_gain: float = 12.0  # N per (kg m/s) of tool momentum
    grip_margin_step: float = 10.0
    # "template": the human impact template above (ramp before the predicted contact, peak after the flag, decay);
    # "constant": hold force throughout, no anticipation
    grip_profile: str = "template"
    strike_timeout: float = 0.100
    # when the arm leaves the swing for its post-impact reference: "flag" (on the L1 impact flag; a clock timeout
    # declares a miss) or "time" (post_delay after the predicted contact, as a robot without an impact sensor must)
    post_switch: str = "flag"
    post_delay: float = 0.010
    n_strikes: int = 10


@dataclass
class ReactiveCfg:
    """Reactive grip: detect the tool slipping in the hand and answer with the hand's maximum grip.

    mode "off"; "oracle": simulation truth of the tool's motion in the hand, zero latency (an upper bound on any
    reactive loop); "taxel": the taxel arrays and pad accelerometers; "log": compute both detectors, never act
    (for calibration)."""

    mode: str = "off"
    rate: float = 1000.0
    # the grip a detection jumps to: "pre", the hand's pre-impact ceiling (the anticipatory ramp's cap, so reaction
    # and anticipation reach the same squeeze and differ in timing); "max", the hand's full grip command (on a wrap,
    # a step to full synergy torque reshapes the grasp)
    boost_to: str = "pre"
    hold_time: float = 0.200  # s at the boosted grip after the last detection
    release_time: float = 0.200  # s to ramp back to the scheduled grip
    # latched: after the hold the raised grip decays toward the schedule with time constant latch_tau instead of
    # releasing within release_time (the human answer to a slip: a raised safety margin that persists)
    latch: bool = False
    latch_tau: float = 2.0  # s
    confirm: int = 2  # consecutive ticks over threshold before acting (fast detectors)
    # fast detectors (impacts, jerks). oracle: rate of the tool's motion in the TCP frame
    omega_thresh: float = 1.0  # rad/s
    v_thresh: float = 0.02  # m/s
    # taxel: per patch, centre-of-pressure speed and relative load drop over `window`; pad accelerometer deviation
    window: float = 0.003  # s
    load_floor: float = 2.0  # N, a patch below this load is ignored
    cop_speed_thresh: float = 0.05  # m/s
    load_drop_thresh: float = 0.15  # fraction of the patch load lost within `window` (> 1 disables)
    accel_thresh: float = 60.0  # m/s^2 deviation from the 5 ms baseline
    # slow detectors (creep under periodic or steady load): the tool pose (oracle) or each patch's centre of pressure
    # (taxel), low-passed by two cascaded first-order filters of `drift_tau` so elastic flexing at the stroke rate is
    # filtered out, drifting from a reference that resets at every detection
    drift_tau: float = 0.5  # s
    drift_rot_thresh: float = 0.0175  # rad (1 deg)
    drift_trans_thresh: float = 0.001  # m
    cop_drift_thresh: float = 0.001  # m
    # taxel: rotation of a patch's pressure pattern (principal axis of its low-passed second moments), the cue a
    # normal-pressure array gives for a handle turning about the pad normal; only for elongated patterns
    pattern_rot_thresh: float = 0.035  # rad (2 deg)
    pattern_aniso: float = 1.5  # minimum ratio of the two largest second moments


@dataclass
class ResidualCfg:
    """The residual hand policy (L2) on the testbed (tactile_sim.control.residual_l2): it reads the robot's sensors,
    adds bounded offsets to the base hand command and sends the WUJI drives MIT commands. Needs
    GripperCfg.hand_control = "mit" and SensorsCfg.ts_model = "rev3"."""

    enabled: bool = False
    rate: float = 1000.0  # Hz the policy decides at (1000, 500, 250, 100); fast inputs are sampled at 1 kHz
    phase: float = 0.0  # s, offset of the policy ticks from the SDK cycle
    policy: str = "zero"  # "zero" (base only), or a policy object passed to ResidualHandL2
    task: str = "hammer"  # obs_spec.TASKS
    stale_taxel: float = 0.005  # s: the newest frame of any board older than this -> the residual decays
    stale_joint: float = 0.003  # s: hand joint readback older than this -> the residual decays
    decay_tau: float = 0.020  # s
    # drill trigger: residual on these joints is capped, and may only open while the base disables the trigger
    trigger_joints: tuple[str, ...] = ("r_index_finger_mcp_flex", "r_index_finger_pip", "r_index_finger_dip")
    trigger_dq_max: float = 0.05  # rad
    trigger_tau: tuple[float, float] = (-0.1, 0.05)  # Nm, along the closing direction
    # observation blocks (obs_spec.OBS_BLOCKS names) seen as zeros: the sensing ablation's students
    mask_blocks: tuple[str, ...] = ()


@dataclass
class ForecastCfg:
    """How far the load-regime forecast a base hands the residual is from the truth (behaviors.forecast)."""

    mask_prob: float = 0.25  # whole episodes without a forecast (the residual must work without one)
    dropout: float = 0.05  # updates lost within an episode
    bias_sigma: float = 0.015  # s, per-episode bias of the predicted event time
    jitter_sigma: float = 0.010  # s, per-update noise on the predicted event time
    class_confusion: float = 0.1  # up to this much probability moved off the true regime, per episode
    freq_err: float = 0.1  # log-normal sigma of the predicted dominant frequency
    amp_err: float = 0.25  # log-normal sigma of the predicted amplitude
    update_rate: float = 10.0  # Hz the forecast is refreshed at
    latency: float = 0.1  # s from the state a forecast reflects to when the residual sees it


@dataclass
class LoadGenCfg:
    """Synthetic loads at the tool's working point (tactile_sim.model.loadgen): the first training stage covers every
    load regime the residual will meet, before the nail and drill plants. Events are drawn per episode; forces are
    in the tool frame (x along the handle or bit, z along the strike or feed), torques about the tool's long axis."""

    rate: float = 2.0  # events per second (Poisson)
    p_kind: tuple[float, float, float, float, float] = (0.35, 0.25, 0.15, 0.15, 0.10)  # impulse periodic step bind push
    # impulse: half-sine force along the strike axis within cone_deg
    impulse_ms: tuple[float, float] = (0.5, 5.0)
    impulse_n: tuple[float, float] = (50.0, 700.0)
    cone_deg: float = 20.0
    # periodic: a force along the feed or a torque about the long axis, amplitude and phase drifting slowly
    periodic_hz: tuple[float, float] = (1.0, 100.0)
    periodic_n: tuple[float, float] = (5.0, 40.0)
    periodic_nm: tuple[float, float] = (0.2, 1.5)
    periodic_s: tuple[float, float] = (0.3, 2.0)
    drift: float = 0.3  # relative amplitude drift per second (random walk)
    # step: a steady push that vanishes (breakthrough) or appears
    step_n: tuple[float, float] = (20.0, 90.0)
    step_hold_s: tuple[float, float] = (0.2, 1.0)
    # bind: a torque spike about the long axis
    bind_ms: tuple[float, float] = (20.0, 80.0)
    bind_nm: tuple[float, float] = (0.5, 3.0)
    # push: a steady force along the feed
    push_n: tuple[float, float] = (10.0, 80.0)
    push_s: tuple[float, float] = (0.3, 1.5)


@dataclass
class ResidualDRCfg:
    """Domain randomization for the residual experiments, drawn once per seed and shared by every condition run on
    that seed (sim.randomize.residual_dr), and mirrored by the GPU training env. Ranges are (low, high), uniform
    unless noted."""

    head_mass: tuple[float, float] = (0.25, 0.70)  # kg
    hand_friction: tuple[float, float] = (0.6, 1.4)
    hand_solref_tc: tuple[float, float] = (0.0015, 0.004)  # s, hand-tool contact time constant
    nail_resistance_0: tuple[float, float] = (80.0, 300.0)  # N
    nail_resistance_per_m: tuple[float, float] = (2.5e3, 1.0e4)  # N/m
    face_timeconst: tuple[float, float] = (0.001, 0.003)  # s
    vdr_zeta0: tuple[float, float] = (0.2, 0.5)
    cmd_latency: tuple[float, float] = (0.0005, 0.003)  # s, WUJI host -> driver
    sdk_lpf_hz: tuple[float, float] = (50.0, 300.0)  # used with probability sdk_lpf_prob, else off
    sdk_lpf_prob: float = 0.5
    tau_i: tuple[float, float] = (0.0001, 0.0005)  # s
    kt_scale: tuple[float, float] = (0.85, 1.15)  # rated current scale (Kt error)
    effort_limit_ratio: tuple[float, float] = (0.9, 1.1)
    t_amb: tuple[float, float] = (20.0, 35.0)  # degC
    ts_gain_mismatch: tuple[float, float] = (0.05, 0.15)
    ts_offset_lsb: tuple[float, float] = (2.0, 6.0)
    ts_f_half: tuple[float, float] = (3.0, 8.0)  # N
    ts_bandwidth: tuple[float, float] = (150.0, 500.0)  # Hz
    ts_skin_depth: tuple[float, float] = (0.001, 0.003)  # m
    ts_drop_prob: tuple[float, float] = (0.0, 0.01)
    ts_bridge_hi: tuple[float, float] = (0.0005, 0.001)  # s, upper end of the host-bridge latency
    servo_p_mult: tuple[float, float] = (0.7, 1.4)  # Vega arm stiffness multiplier
    sensor_noise_scale: tuple[float, float] = (0.5, 2.0)  # log-uniform
    # the drill (plant kind "drill"), as multiples of its configured values unless noted
    drill_stall_scale: tuple[float, float] = (0.75, 1.25)
    drill_clutch_scale: tuple[float, float] = (0.7, 1.1)  # higher needs more push than the Vega's press gives
    drill_detents: tuple[int, int] = (4, 12)  # clutch detents per revolution, inclusive
    drill_cam_scale: tuple[float, float] = (0.8, 1.3)
    drill_exit_len: tuple[float, float] = (0.001, 0.004)  # m, the hole's breakthrough
    pistol_mass_scale: tuple[float, float] = (0.85, 1.3)  # the housing and the battery


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
class VlaCfg:
    """The slow policy's view and clock (tactile_sim.vla): the Vega's head ego camera and the rate the VLA records
    demos and runs at.

    The camera is the left imager of the head's ZED stereo pair (Dexmate's URDF: 50 mm baseline, as on a ZED X
    Mini), rectified, 2.2 mm lens: 105 x 78 deg at 16:10. The lens is an assumption until the unit is known."""
    fps: float = 30.0
    cam_width: int = 384
    cam_height: int = 240
    cam_fovy: float = 78.0  # deg, vertical
    # the head joints (head_j1, head_j2, head_j3) while the VLA works; None = aimed at the tool's working point
    # (vla.camera.aim_head), the way an operator would point the head before a task
    head_pose: tuple[float, float, float] | None = None
    # per-episode visual randomization (vla.camera.VisualDR); off = the nominal scene
    visual_dr: bool = True
    cam_pos_jitter: float = 0.01  # m, each axis
    cam_rot_jitter: float = 1.0  # deg, each axis
    light_jitter: float = 0.3  # fraction of the nominal intensity
    color_jitter: float = 0.15  # rgb, each channel, on the tool, the workpiece, the table and the floor


@dataclass
class LoggingCfg:
    truth_decim: int = 1
    compress: bool = True


@dataclass
class SimConfig:
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    arm: ArmCfg = field(default_factory=ArmCfg)
    gripper: GripperCfg = field(default_factory=GripperCfg)
    wuji: WujiDriveCfg = field(default_factory=WujiDriveCfg)
    hammer: HammerCfg = field(default_factory=HammerCfg)
    plant: NailPlantCfg = field(default_factory=NailPlantCfg)
    saw: SawCfg = field(default_factory=SawCfg)
    drill: DrillCfg = field(default_factory=DrillCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)
    sensors: SensorsCfg = field(default_factory=SensorsCfg)
    controller: ControllerCfg = field(default_factory=ControllerCfg)
    swing: SwingCfg = field(default_factory=SwingCfg)
    reactive: ReactiveCfg = field(default_factory=ReactiveCfg)
    residual: ResidualCfg = field(default_factory=ResidualCfg)
    forecast: ForecastCfg = field(default_factory=ForecastCfg)
    loadgen: LoadGenCfg = field(default_factory=LoadGenCfg)
    residual_dr: ResidualDRCfg = field(default_factory=ResidualDRCfg)
    dr: DRCfg = field(default_factory=DRCfg)
    vla: VlaCfg = field(default_factory=VlaCfg)
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


# A light claw hammer as primitives: an 8 oz (~230 g) steel head and a 100 g handle, 0.33 kg in all. The WUJI Hand 2
# at its rated torques cannot hold the 665 g YCB hammer through a blow (paper/sections/methods.tex), and primitive
# shapes keep the GPU taxel model analytic. Training randomizes the head mass over 0.25-0.7 kg. Its nail is a small
# one: the default nail (450 N before it moves, set for the 665 g hammer) stops the light hammer's ~370 N blows
# dead; at 150 N + 5 kN/m the Vega U's ~1.1 m/s blows (~190 N peak) drive it ~2 mm a strike, 20 mm in ~10.
LIGHT_HAMMER_OVERRIDES: dict[str, dict[str, Any]] = {
    "hammer": {"model": "primitive", "head_mass": 0.23, "handle_mass": 0.10, "head_radius": 0.011,
               "head_half_len": 0.04, "handle_radius": 0.013, "handle_len": 0.28, "grip_from_head": 0.17},
    "plant": {"resistance_0": 150.0, "resistance_per_m": 5000.0},
}


# The drill task on the Vega U + WUJI: a compact 12 V pistol driver (1.12 kg, 3 Nm stall in low gear) driving a small
# screw or drilling a pine board, fed straight down (no arc), with the arm pushing through its position servos
# (press 12 mm past contact: ~80 N) and drilling holes at a fixed feed rate.
VEGA_DRILL_OVERRIDES: dict[str, dict[str, Any]] = {
    "plant": {"kind": "drill"},
    "drill": {"form": "pistol", "mode": "screw", "stall_torque": 3.0, "clutch_torque": 1.5, "press_depth": 0.012,
              "drill_torque_per_feed": 1500.0, "trigger_mode": "finger"},
    "swing": {"arc_radius": 0.0},
}


def residual_config(rate: float = 1000.0, robot: str = "vega_1u", base: SimConfig | None = None,
                    task: str = "hammer", **sections: dict[str, Any]) -> SimConfig:
    """The residual experiments' configuration: WUJI Hand 2 with MIT-mode drives on the robot (default the Vega U),
    the TaxelScan fast profile at its target 7.8 us per taxel, the tool for `task` ("hammer": the light hammer and
    its nail; "drill": the pistol driver, VEGA_DRILL_OVERRIDES) with grasps cached at its nominal masses, and the
    residual policy at `rate` Hz. `base` defaults to the 8 kHz validator physics.

    The base holds the tool the way a VLA's hand targets would: a constant grip, open loop (no grip law on the
    pads, no slip feedback from the simulator's truth), so whatever reacts to touch is the residual."""
    if task not in ("hammer", "drill"):
        raise ValueError(f"unknown task {task!r} (expected 'hammer' or 'drill')")
    cfg = hand_config("wuji2", base or SimConfig(), robot=robot)
    cfg = cfg.replace(**(LIGHT_HAMMER_OVERRIDES if task == "hammer" else VEGA_DRILL_OVERRIDES)).replace(
        residual={"task": task}).replace(
        gripper={"hand_control": "mit", "grasp_mass": "nominal"},
        sensors={"ts_model": "rev3", "ts_taxel_time": 7.8e-6},
        residual={"enabled": True, "rate": rate},
        # the base has no impact flag: the arm switches to its post-impact reference on the clock
        controller={"impact_sources": (), "grip_loop": "open"},
        swing={"post_switch": "time", "grip_profile": "constant", "grip_margin_step": 0.0},
    )
    return cfg.replace(**sections) if sections else cfg
