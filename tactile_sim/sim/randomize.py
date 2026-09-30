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
    testbed.l1.observer.K = testbed.l1.observer.K * s.observer_gain_scale
