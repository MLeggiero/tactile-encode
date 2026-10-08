"""The residual policy's interface: observation layout and action map, shared by the CPU testbed, the GPU training
env and the robot-side node (which import or generate from this file; `spec_hash` ties a policy to its version).

Action (60-D, per WUJI joint, in joint order): three blocks of 20, each squashed by tanh.

    q*  = q_base + dq_max[group] * tanh(a_q)
    Kp  = clip(Kp_base + kp_nom * (4^tanh(a_k) - 1), 0, kp_max)       additive, so a zero-stiffness base can stiffen
    Kd  = max(Kd_base + 2 zeta sqrt(I_eff) (sqrt(Kp) - sqrt(Kp_base)), 0)   damping follows the stiffness
    tau = tau_base + tau_frac * tau_rated * tanh(a_t)

so a = 0 reproduces the base command exactly. Slew limits are per second (the same at every policy rate) and act
on the residual's offsets, not on the base.

Observation: the blocks of OBS_BLOCKS, in order, each scaled to O(1). The student sees only what the robot has
(touch through the TaxelScan fast profile, the WUJI's joint angles and currents, the base command and its
lookahead, the forecast); `assert_deployable` rejects the testbed's sim-only sensors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from tactile_sim.sim.wuji_drive import JOINT_GROUPS, joint_group

OBS_SPEC_VERSION = 1
N_JOINTS = 20
N_ARM = 7
LOOKAHEAD = (0.010, 0.025, 0.050)  # s ahead of now at which the base command is sampled
TASKS = ("hammer", "drill", "loadgen")
FORECAST_DIM = 13  # see forecast_vector
TAXEL_PATCHES = 11
TAXELS = 448
BOARDS = 4

# name, dimension, fast (sampled at 1 kHz and stacked `stack` deep for a slower policy), what it is
OBS_BLOCKS: tuple[tuple[str, int, bool, str], ...] = (
    ("taxel_codes", TAXELS, True, "mu-law codes / (levels - 1), patches in layout order"),
    ("patch_features", 6 * TAXEL_PATCHES, True, "per patch: sum (10 N), CoP row and column (fractions of the grid; "
                                                "-1 unloaded), area (fraction), peak (10 N), rate (10 N per frame)"),
    ("board_status", 2 * BOARDS, True, "per board: age of its newest frame (5 ms units), frames received since "
                                       "the previous sample (0 = none: stale)"),
    ("hand_q", N_JOINTS, True, "encoder angle (rad)"),
    ("hand_current", N_JOINTS, True, "motor current / rated current"),
    ("last_action", 3 * N_JOINTS, False, "the residual's previous squashed action"),
    ("base_hand_now", N_JOINTS, False, "base hand target q* (rad)"),
    ("base_lookahead", (N_JOINTS + N_ARM) * len(LOOKAHEAD), False,
     "base hand and arm targets at the lookahead times, relative to now (rad)"),
    ("forecast", FORECAST_DIM, False, "load-regime forecast (forecast_vector); all zero when masked"),
    ("task", len(TASKS), False, "one-hot task"),
    ("trigger_enable", 1, False, "1 while the base allows the drill trigger"),
)


def stack_for(rate: float) -> int:
    """1 kHz samples a policy at `rate` sees per step: every sample since its previous step."""
    return max(1, int(round(1000.0 / rate)))


def obs_dim(stack: int = 1) -> int:
    return sum(d * (stack if fast else 1) for _, d, fast, _ in OBS_BLOCKS)


def obs_slices(stack: int = 1) -> dict[str, slice]:
    """Where each block sits; a fast block holds `stack` samples, newest first."""
    out, k = {}, 0
    for name, d, fast, _ in OBS_BLOCKS:
        n = d * (stack if fast else 1)
        out[name] = slice(k, k + n)
        k += n
    return out


REGIMES = ("impulse", "periodic", "steady", "step", "bind")


def forecast_vector(valid: bool, t_to_event: float = 0.0, sigma_t: float = 0.0, p_regime=None, f_dom: float = 0.0,
                    sigma_f: float = 0.0, amp: float = 0.0, sigma_amp: float = 0.0, age: float = 0.0) -> np.ndarray:
    """The 13-D forecast block: valid; time to the next event (s, clipped to +-1 s) and its sigma; regime
    probabilities (REGIMES); dominant frequency and its sigma (100 Hz units); amplitude and its sigma (100 N units);
    the forecast's age since it was issued (s, clipped to 1 s). All zero when masked (valid = False)."""
    if not valid:
        return np.zeros(FORECAST_DIM)
    p = np.zeros(len(REGIMES)) if p_regime is None else np.asarray(p_regime, dtype=float)
    return np.concatenate([[1.0, np.clip(t_to_event, -1.0, 1.0), sigma_t], p,
                           [f_dom / 100.0, sigma_f / 100.0, amp / 100.0, sigma_amp / 100.0, min(age, 1.0)]])


@dataclass(frozen=True)
class ActionSpec:
    """Bounds of the residual's action map (see module doc). Per-group tuples follow wuji_drive.JOINT_GROUPS."""

    dq_max: tuple[float, ...] = (0.20, 0.08, 0.20, 0.15, 0.20)  # rad
    kp_gain: float = 4.0  # the stiffness offset spans kp_nom * (1 / kp_gain - 1) .. kp_nom * (kp_gain - 1)
    kp_max_ratio: float = 8.0  # Kp never above kp_max_ratio * kp_nom
    tau_frac: float = 0.6  # feedforward offset, fraction of the rated torque
    zeta: float = 0.8
    # slew limits on the residual's offsets, per second
    dq_rate: float = 20.0  # rad/s
    kp_rate: float = 100.0  # kp_nom per second
    tau_rate: float = 100.0  # tau_rated per second

    @property
    def dim(self) -> int:
        return 3 * N_JOINTS


@dataclass
class JointParams:
    """Per-joint constants the action map needs (all (20,), joint order)."""

    group: np.ndarray  # index into JOINT_GROUPS
    tau_rated: np.ndarray
    kp_nom: np.ndarray
    i_eff: np.ndarray  # kg m^2, effective inertia at the joint (armature + links)

    @classmethod
    def for_joints(cls, names: list[str], tau_rated, kp_nom_groups, i_eff) -> JointParams:
        g = np.array([JOINT_GROUPS.index(joint_group(n)) for n in names])
        return cls(g, np.asarray(tau_rated, float), np.asarray(kp_nom_groups, float)[g], np.asarray(i_eff, float))


def squash(a: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(a, dtype=float))


def offsets(spec: ActionSpec, jp: JointParams, a_sq: np.ndarray) -> np.ndarray:
    """(3, 20) residual offsets [dq, dKp, dtau] for a squashed action (60,)."""
    a = np.asarray(a_sq, dtype=float).reshape(3, N_JOINTS)
    dq = np.asarray(spec.dq_max)[jp.group] * a[0]
    dkp = jp.kp_nom * (np.power(spec.kp_gain, a[1]) - 1.0)
    dtau = spec.tau_frac * jp.tau_rated * a[2]
    return np.stack([dq, dkp, dtau])


def slew(spec: ActionSpec, jp: JointParams, prev: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
    """Limit how fast the residual's offsets (3, 20) move, per second, over a policy period dt."""
    lim = np.stack([np.full(N_JOINTS, spec.dq_rate), spec.kp_rate * jp.kp_nom, spec.tau_rate * jp.tau_rated]) * dt
    return prev + np.clip(target - prev, -lim, lim)


def compose(spec: ActionSpec, jp: JointParams, base: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Final MIT command (5, 20) rows q, qd, kp, kd, tau from the base command (5, 20) and the offsets (3, 20)."""
    q, qd, kp_b, kd_b, tau_b = np.asarray(base, dtype=float)
    kp = np.clip(kp_b + off[1], 0.0, spec.kp_max_ratio * jp.kp_nom)
    kd = np.maximum(kd_b + 2.0 * spec.zeta * np.sqrt(jp.i_eff) * (np.sqrt(kp) - np.sqrt(np.maximum(kp_b, 0.0))), 0.0)
    return np.stack([q + off[0], qd, kp, kd, tau_b + off[2]])


def spec_hash(action: ActionSpec | None = None, stack: int = 1) -> str:
    """Version tag of the observation layout and action map: a policy runs only where its hash matches."""
    blob = json.dumps({"v": OBS_SPEC_VERSION, "blocks": [(n, d, f) for n, d, f, _ in OBS_BLOCKS], "stack": stack,
                       "lookahead": LOOKAHEAD, "tasks": TASKS, "regimes": REGIMES,
                       "action": asdict(action or ActionSpec())}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def assert_deployable(sensor_names: list[str]) -> None:
    """Raise if an observation would read a sensor the robot does not have."""
    from tactile_sim.sensors import is_sim_only

    bad = [n for n in sensor_names if is_sim_only(n)]
    if bad:
        raise ValueError(f"observation reads sim-only sensors: {bad}")
