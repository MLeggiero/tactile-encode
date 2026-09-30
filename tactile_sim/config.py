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
    source: str = "auto"  # "fr3" (Menagerie, required), "fallback" (capsules), "auto" (fr3 if cached)
    flex_mode: str = "wrist"  # "rigid" or "wrist"
    wrist_k_rot: float = 1500.0  # Nm/rad, hinges about the flange x and y axes
    wrist_d_rot: float = 1.5  # Nms/rad
    wrist_k_lin: float = 2.0e5  # N/m along the flange z axis
    wrist_d_lin: float = 150.0  # Ns/m
    ft_body_mass: float = 0.12  # wrist F/T sensor (Bota SensONE class)
    # joint-space hold used only while settling at reset
    hold_kp: tuple[float, ...] = (600, 600, 600, 600, 250, 150, 50)
    hold_kd: tuple[float, ...] = (50, 50, 50, 50, 20, 12, 5)
    # IK seed: Menagerie "home" keyframe
    q_seed: tuple[float, ...] = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)


@dataclass
class GripperCfg:
    hand_mass: float = 0.73
    finger_mass: float = 0.02
    tcp_offset: float = 0.1034  # flange to grasp centre along the flange z axis
    finger_range: float = 0.04  # opening per finger
    grip_force_max: float = 150.0  # N per pad
    pad_mode: str = "explicit"  # "explicit" (spring-mounted pads) or "soft_contact"
    pad_half: tuple[float, float, float] = (0.025, 0.002, 0.011)  # along handle, thickness, width
    pad_mass: float = 0.01
    pad_k_n: float = 5.0e4  # N/m normal
    pad_d_n: float = 15.0  # Ns/m normal
    pad_k_t: float = 3.0e4  # N/m tangential
    pad_d_t: float = 10.0
    pad_travel_n: float = 0.004
    pad_travel_t: float = 0.002
    pad_friction: float = 1.3
    pad_torsion: float = 0.005
    pad_solref: tuple[float, float] = (0.002, 1.0)
    pad_solimp: tuple[float, float, float] = (0.9, 0.95, 0.001)
    # soft-contact-only pads (pad_mode="soft_contact")
    soft_pad_solref: tuple[float, float] = (0.004, 0.6)
    finger_joint_damping: float = 20.0


@dataclass
class HammerCfg:
    head_mass: float = 0.45
    head_radius: float = 0.0125
    head_half_len: float = 0.05  # along the strike axis
    handle_radius: float = 0.014
    handle_len: float = 0.28
    handle_mass: float = 0.15
    grip_from_head: float = 0.16  # grasp centre to head axis, along the handle
    face_solref: tuple[float, float] = (0.001, 0.4)
    face_solimp: tuple[float, float, float] = (0.95, 0.99, 0.001)
    board_solref: tuple[float, float] = (0.002, 0.8)


@dataclass
class NailPlantCfg:
    kind: str = "nail"  # "nail" (Task A) or "drill" (Task B stub)
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
class SceneCfg:
    # Grasp/TCP frame: hand points down (approach = -z), fingers close along world y, handle along
    # world x. The strike is horizontal along +y into a vertical board; the nail is placed so the
    # hammer face is `hover_clearance` short of the nail head at the hover pose.
    hover_tcp: tuple[float, float, float] = (0.52, -0.05, 0.35)
    hover_clearance: float = 0.06


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
    pressure_rate: float = 500.0
    pressure_latency: float = 0.002
    pressure_bandwidth: float = 100.0
    pressure_noise: float = 0.3
    pressure_range: float = 60.0
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
    k_trans: tuple[float, float, float] = (1500.0, 1500.0, 1500.0)
    k_rot: tuple[float, float, float] = (60.0, 60.0, 60.0)
    zeta: float = 1.0
    k_dot_max: float = 5.0e4  # N/m/s stiffness slew limit
    delta_max_pos: float = 0.05  # reference limiter
    delta_max_rot: float = 0.3
    f_ff_max: tuple[float, float] = (80.0, 8.0)
    k_null: float = 10.0
    d_null: float = 2.0
    osc_inertia: bool = True
    gate_duration: float = 0.050
    stale_timeout: float = 0.020
    observer_gain: float = 400.0
    impact_force_thresh: float = 30.0
    impact_slope_thresh: float = 8.0e3  # N/s on the wrist F/T strike axis
    impact_accel_thresh: float = 60.0  # m/s^2 pad-accelerometer deviation from its 5 ms baseline
    impact_refractory: float = 0.100
    grip_kp: float = 0.8
    grip_ki: float = 40.0
    grip_hold: float = 40.0
    drop_force_frac: float = 0.2
    drop_accel: float = 50.0


@dataclass
class SwingCfg:
    rate: float = 200.0
    approach_time: float = 0.6
    windup_height: float = 0.15
    windup_time: float = 0.45
    v_strike: float = 2.0
    overshoot: float = 0.03
    strike_k: tuple[float, float, float] = (1200.0, 1200.0, 2500.0)
    recover_time: float = 0.4
    settle_time: float = 0.15
    grip_lead: float = 0.150
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


def fast_config(**sections: dict[str, Any]) -> SimConfig:
    """Config used by the test suite: 0.25 ms physics (4 kHz) and the pure-Python-friendly rates."""
    cfg = SimConfig().replace(
        physics={"timestep": 0.00025},
        sensors={"ft_rate": 2000.0, "accel_rate": 4000.0, "accel_bandwidth": 1500.0, "ft_bandwidth": 800.0},
    )
    return cfg.replace(**sections) if sections else cfg
