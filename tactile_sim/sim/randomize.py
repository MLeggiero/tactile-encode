"""Domain randomization over the parameters the report flags as sim-to-real gaps: tool mass, nail
resistance, pad friction and stiffness, impact contact stiffness and restitution, sensor noise, and
the momentum observer's model mismatch (gain)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tactile_sim.config import SimConfig


@dataclass
class DRSample:
    head_mass_scale: float
    resistance_0: float
    pad_friction: float
    pad_k_n_scale: float
    face_timeconst: float
    vdr_zeta0: float
    sensor_noise_scale: float
    observer_gain_scale: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def sample_dr(cfg: SimConfig, rng: np.random.Generator) -> DRSample:
    r = cfg.dr

    def u(lohi):
        return float(rng.uniform(*lohi))

    return DRSample(u(r.head_mass_scale), u(r.resistance_0), u(r.pad_friction), u(r.pad_k_n_scale),
                    u(r.face_timeconst), u(r.vdr_zeta0), float(np.exp(rng.uniform(*np.log(r.sensor_noise_scale)))),
                    u(r.observer_gain_scale))


def apply_dr(cfg: SimConfig, s: DRSample) -> SimConfig:
    """Model-level parameters go into the config (the scene is rebuilt from it)."""
    return cfg.replace(
        hammer={"head_mass": cfg.hammer.head_mass * s.head_mass_scale,
                "face_solref": (s.face_timeconst, cfg.hammer.face_solref[1])},
        plant={"resistance_0": s.resistance_0, "vdr_zeta0": s.vdr_zeta0},
        gripper={"pad_friction": s.pad_friction, "pad_k_n": cfg.gripper.pad_k_n * s.pad_k_n_scale},
    )


def apply_runtime_dr(testbed, s: DRSample) -> None:
    """Runtime parameters: sensor noise and the observer's gain."""
    testbed.sensors.set_noise_scale(s.sensor_noise_scale)
    observer = getattr(testbed.l1, "observer", None)
    if observer is not None:  # the position-interface L1 (Vega) has no joint-torque momentum observer
        observer.K = observer.K * s.observer_gain_scale


def residual_dr(cfg: SimConfig, rng: np.random.Generator) -> dict[str, dict]:
    """One draw of the residual experiments' randomization (ResidualDRCfg) as config overrides, for
    `cfg.replace(**overrides)`. The same seed gives the same draw, so conditions are compared pairwise."""
    r = cfg.residual_dr

    def u(lohi):
        return float(rng.uniform(*lohi))

    lpf = u(r.sdk_lpf_hz) if rng.uniform() < r.sdk_lpf_prob else None
    kt = u(r.kt_scale)
    out = {
        "hammer": {"head_mass": u(r.head_mass)},
        "gripper": {"hand_friction": u(r.hand_friction), "hand_solref": (u(r.hand_solref_tc), 1.0)},
        "plant": {"resistance_0": u(r.nail_resistance_0), "resistance_per_m": u(r.nail_resistance_per_m),
                  "vdr_zeta0": u(r.vdr_zeta0)},
        # the hammer's face time constant lives in HammerCfg
        "wuji": {"cmd_latency": u(r.cmd_latency), "sdk_lpf_hz": lpf, "tau_i": u(r.tau_i),
                 "i_rated": tuple(x * kt for x in cfg.wuji.i_rated), "effort_limit_ratio": u(r.effort_limit_ratio),
                 "t_amb": u(r.t_amb)},
        "sensors": {"ts_gain_mismatch": u(r.ts_gain_mismatch), "ts_offset_lsb": u(r.ts_offset_lsb),
                    "ts_f_half": u(r.ts_f_half), "ts_bandwidth": u(r.ts_bandwidth),
                    "ts_skin_depth": u(r.ts_skin_depth), "ts_drop_prob": u(r.ts_drop_prob),
                    "ts_bridge_latency": (cfg.sensors.ts_bridge_latency[0], u(r.ts_bridge_hi))},
        "arm": {"servo_p_mult": u(r.servo_p_mult)},
        "_runtime": {"sensor_noise_scale": float(np.exp(rng.uniform(*np.log(r.sensor_noise_scale)))),
                     "face_timeconst": u(r.face_timeconst)},
    }
    if cfg.plant.kind == "drill":  # drawn after everything else, so the hammer's draws for a seed are unchanged
        out["drill"] = _drill_dr(cfg, r, rng, u)
    return out


def _drill_dr(cfg: SimConfig, r, rng: np.random.Generator, u) -> dict:
    dc = cfg.drill
    mass = u(r.pistol_mass_scale)
    if dc.model == "v2":
        return {"stall_torque": dc.stall_torque * u(r.drill_stall_scale),
                "clutch_torque": dc.clutch_torque * u(r.drill_clutch_scale),
                "clutch_detents": int(rng.integers(r.drill_detents[0], r.drill_detents[1] + 1)),
                "cam_ratio": dc.cam_ratio * u(r.drill_cam_scale), "exit_len": u(r.drill_exit_len),
                "pistol_housing_mass": dc.pistol_housing_mass * mass,
                "pistol_battery_mass": dc.pistol_battery_mass * mass}

    def logu(lohi):
        return float(np.exp(rng.uniform(*np.log(lohi))))

    o = {"free_speed": u(r.drill_free_speed), "stall_torque": u(r.drill_stall),
         "spindle_inertia": logu(r.drill_spindle_inertia), "gear_ratio": u(r.drill_gear_ratio),
         "clutch_detents": int(rng.integers(r.drill_clutch_detents[0], r.drill_clutch_detents[1] + 1)),
         "clutch_ring_ratio": u(r.drill_ring_ratio), "clutch_duty": u(r.drill_clutch_duty),
         "clutch_overshoot": u(r.drill_clutch_overshoot), "strip_torque": u(r.drill_strip_torque),
         "screw_twist_off": u(r.drill_twist_off), "cut_energy": u(r.drill_cut_energy),
         "exit_len": u(r.drill_exit_len), "pistol_housing_mass": dc.pistol_housing_mass * mass,
         "pistol_battery_mass": dc.pistol_battery_mass * mass}
    k = u(r.drill_screw_scale)
    o.update(screw_torque0=dc.screw_torque0 * k, screw_torque_per_m=dc.screw_torque_per_m * k)
    # the clutch set the way a person sets it: enough to seat the drawn screw, not enough to ruin it
    flush = o["screw_torque0"] + o["screw_torque_per_m"] * dc.screw_len
    lo = max(r.drill_clutch[0], 1.1 * flush)
    hi = min(r.drill_clutch[1], 0.9 * min(o["strip_torque"], o["screw_twist_off"]) / (1.0 + o["clutch_overshoot"]))
    o["clutch_torque"] = float(rng.uniform(lo, hi)) if hi > lo else lo
    o.update(vib_unbalance=dc.vib_unbalance * logu(r.drill_vib_scale),
             vib_runout=dc.vib_runout * logu(r.drill_vib_scale), vib_noise=dc.vib_noise * logu(r.drill_vib_scale),
             noise_seed=int(rng.integers(2 ** 31)))
    return o


def apply_residual_dr(cfg: SimConfig, overrides: dict[str, dict]) -> SimConfig:
    """The config for one residual_dr draw (the scene is rebuilt from it)."""
    o = {k: v for k, v in overrides.items() if not k.startswith("_")}
    rt = overrides.get("_runtime", {})
    if "face_timeconst" in rt:
        o.setdefault("hammer", {})["face_solref"] = (rt["face_timeconst"], cfg.hammer.face_solref[1])
    return cfg.replace(**o)
