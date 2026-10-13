"""The residual policy's interface: observation layout and action map, shared by the CPU testbed, the GPU training
env and the robot-side node (which import or generate from this file; `spec_hash` ties a policy to its version).

Action (40-D, per WUJI joint, in joint order): two blocks of 20, each squashed by tanh.

    q*  = q_base + dq_max[group] * tanh(a_q)
    tau = tau_base + tau_frac * tau_rated * tanh(a_t)
    Kp, Kd, qd* = the base's

so a = 0 reproduces the base command exactly. The drives' gains are configuration, not commands: WUJI Hand 2 writes
`mit_params (kp, kd)` once, before enable(), and its realtime JointCommand carries only position, velocity and torque
feedforward (OBS_SPEC_VERSION 1 also streamed a stiffness offset per joint, which the hardware cannot take at rate).
Under the grip synergy the closing joints run Kp = 0 (their grip is the feedforward), so there the residual acts
through tau; on the shaping joints (abduction, the thumb's base, a held fingertip joint) through the spring's
set point q* as well. Slew limits are per second (the same at every policy rate) and act on the residual's offsets,
not on the base.

Observation: the blocks of a layout (OBS_LAYOUTS; OBS_BLOCKS is the current one), in order, each scaled to O(1). The
student sees only what the robot has (touch through the TaxelScan fast profile, the WUJI's joint angles and
currents, the base command (its targets and its feedforward: the base's grip) and the arm's lookahead);
`assert_deployable` rejects the testbed's sim-only sensors. Version 2: the 40-D action (last_action), the base's
feedforward torque (base_hand_tau: under the synergy the grip lives there, not in q*), the lookahead of the arm only
(a scripted swing base has no hand plan ahead of time, and a VLA's hand targets mean something else), plus the patch
features, the forecast, the task and the trigger enable. Version 3: touch as the raw codes only (no patch
features) and no forecast; the task and the trigger enable stay. A policy's sidecar carries its layout's spec_hash;
the runtimes build each policy the layout it was trained on (layout_of).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from tactile_sim.sim.wuji_drive import JOINT_GROUPS, joint_group

OBS_SPEC_VERSION = 3  # the current layout; OBS_LAYOUTS keeps the earlier ones runnable
N_JOINTS = 20
N_ARM = 7
LOOKAHEAD = (0.010, 0.025, 0.050)  # s ahead of now at which the base command is sampled
TASKS = ("hammer", "drill", "loadgen")
FORECAST_DIM = 13  # see forecast_vector
TAXEL_PATCHES = 11
TAXELS = 448
BOARDS = 4
ACTION_DIM = 2 * N_JOINTS  # [dq, dtau] per joint

# every block a layout may use: name -> (dimension, fast (sampled at 1 kHz and stacked `stack` deep for a slower
# policy), what it is)
BLOCKS: dict[str, tuple[int, bool, str]] = {
    "taxel_codes": (TAXELS, True, "mu-law codes / (levels - 1), patches in layout order"),
    "patch_features": (6 * TAXEL_PATCHES, True, "per patch: sum (10 N), CoP row and column (fractions of the grid; "
                                                "-1 unloaded), area (fraction), peak (10 N), rate (10 N per frame)"),
    "board_status": (2 * BOARDS, True, "per board: age of its newest frame (5 ms units), frames received since "
                                       "the previous sample (0 = none: stale)"),
    "hand_q": (N_JOINTS, True, "encoder angle (rad)"),
    "hand_current": (N_JOINTS, True, "motor current / rated current"),
    "last_action": (ACTION_DIM, False, "the residual's previous squashed action"),
    "base_hand_now": (N_JOINTS, False, "base hand target q* (rad)"),
    "base_hand_tau": (N_JOINTS, False, "base hand feedforward torque / rated torque (the base's grip)"),
    "base_lookahead": (N_ARM * len(LOOKAHEAD), False,
                       "the arm's base targets at each lookahead time, relative to now (rad)"),
    "forecast": (FORECAST_DIM, False, "load-regime forecast (forecast_vector); all zero when masked"),
    "task": (len(TASKS), False, "one-hot task"),
    "trigger_enable": (1, False, "1 while the base allows the drill trigger"),
}
# each version's blocks, in order. 2: the 40-D action's first layout (touch through codes and patch features, the
# base, the forecast, the task, the trigger enable). 3 (2026-10-13): touch from the raw codes only (no patch features)
# and no forecast; the task one-hot stays (the tool's profile is the base's guidance, a VLA's instruction on the
# robot, not something to infer from touch) and so does the trigger enable (the base's own command state; the
# interlock itself is the safety layer's, HandSafety, outside the policy)
OBS_LAYOUTS: dict[int, tuple[str, ...]] = {
    2: ("taxel_codes", "patch_features", "board_status", "hand_q", "hand_current", "last_action", "base_hand_now",
        "base_hand_tau", "base_lookahead", "forecast", "task", "trigger_enable"),
    3: ("taxel_codes", "board_status", "hand_q", "hand_current", "last_action", "base_hand_now", "base_hand_tau",
        "base_lookahead", "task", "trigger_enable"),
}
# the 1 kHz sample the runtimes keep (every fast block any layout reads), in this order
FAST_SAMPLE = ("taxel_codes", "patch_features", "board_status", "hand_q", "hand_current")


def obs_blocks(version: int = OBS_SPEC_VERSION) -> tuple[tuple[str, int, bool, str], ...]:
    """A layout's blocks: (name, dimension, fast, what it is), in order."""
    if version not in OBS_LAYOUTS:
        raise ValueError(f"unknown observation layout {version} (known: {sorted(OBS_LAYOUTS)})")
    return tuple((n, *BLOCKS[n]) for n in OBS_LAYOUTS[version])


OBS_BLOCKS = obs_blocks()  # the current layout


def stack_for(rate: float) -> int:
    """1 kHz samples a policy at `rate` sees per step: every sample since its previous step."""
    return max(1, int(round(1000.0 / rate)))


def obs_dim(stack: int = 1, version: int = OBS_SPEC_VERSION) -> int:
    return sum(d * (stack if fast else 1) for _, d, fast, _ in obs_blocks(version))


def obs_slices(stack: int = 1, version: int = OBS_SPEC_VERSION) -> dict[str, slice]:
    """Where each block sits; a fast block holds `stack` samples, newest first."""
    out, k = {}, 0
    for name, d, fast, _ in obs_blocks(version):
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
    tau_frac: float = 0.6  # feedforward offset, fraction of the rated torque
    # slew limits on the residual's offsets, per second
    dq_rate: float = 20.0  # rad/s
    tau_rate: float = 100.0  # tau_rated per second

    @property
    def dim(self) -> int:
        return ACTION_DIM


@dataclass
class JointParams:
    """Per-joint constants the action map needs (all (20,), joint order)."""

    group: np.ndarray  # index into JOINT_GROUPS
    tau_rated: np.ndarray
    kp_nom: np.ndarray
    i_eff: np.ndarray  # kg m^2, effective inertia at the joint (armature + links; kept for the record)

    @classmethod
    def for_joints(cls, names: list[str], tau_rated, kp_nom_groups, i_eff) -> JointParams:
        g = np.array([JOINT_GROUPS.index(joint_group(n)) for n in names])
        return cls(g, np.asarray(tau_rated, float), np.asarray(kp_nom_groups, float)[g], np.asarray(i_eff, float))


def squash(a: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(a, dtype=float))


def offsets(spec: ActionSpec, jp: JointParams, a_sq: np.ndarray) -> np.ndarray:
    """(2, 20) residual offsets [dq, dtau] for a squashed action (40,)."""
    a = np.asarray(a_sq, dtype=float).reshape(2, N_JOINTS)
    dq = np.asarray(spec.dq_max)[jp.group] * a[0]
    dtau = spec.tau_frac * jp.tau_rated * a[1]
    return np.stack([dq, dtau])


def slew(spec: ActionSpec, jp: JointParams, prev: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
    """Limit how fast the residual's offsets (2, 20) move, per second, over a policy period dt."""
    lim = np.stack([np.full(N_JOINTS, spec.dq_rate), spec.tau_rate * jp.tau_rated]) * dt
    return prev + np.clip(target - prev, -lim, lim)


def compose(spec: ActionSpec, jp: JointParams, base: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Final MIT command (5, 20) rows q, qd, kp, kd, tau from the base command (5, 20) and the offsets (2, 20): the
    offsets move the set point and the feedforward; velocity target and gains are the base's, untouched."""
    q, qd, kp, kd, tau_b = np.asarray(base, dtype=float)
    return np.stack([q + off[0], qd, kp, kd, tau_b + off[1]])


def base_tau_obs(base: np.ndarray, tau_rated: np.ndarray) -> np.ndarray:
    """The base_hand_tau block: the base command's feedforward over each joint's rated torque."""
    return np.asarray(base, dtype=float)[4] / np.asarray(tau_rated, dtype=float)


def arm_lookahead(look: np.ndarray) -> np.ndarray:
    """The base_lookahead block from a base source's full lookahead (hand 20 then arm 7, per LOOKAHEAD time; what
    base sources return): the arm's part, per time."""
    look = np.asarray(look, dtype=float)
    if look.shape == (N_ARM * len(LOOKAHEAD),):
        return look
    return look.reshape(len(LOOKAHEAD), N_JOINTS + N_ARM)[:, N_JOINTS:].reshape(-1)


def spec_hash(action: ActionSpec | None = None, stack: int = 1, version: int = OBS_SPEC_VERSION) -> str:
    """Version tag of an observation layout and the action map: a policy runs only where its hash matches (a layout's
    hash never changes: version 2's is what its policies' sidecars carry)."""
    blob = json.dumps({"v": version, "blocks": [(n, d, f) for n, d, f, _ in obs_blocks(version)], "stack": stack,
                       "lookahead": LOOKAHEAD, "tasks": TASKS, "regimes": REGIMES,
                       "action": asdict(action or ActionSpec())}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def layout_of(hash_: str, action: ActionSpec | None = None, stack: int = 1) -> int | None:
    """The layout version whose spec_hash (at this action map and stack) is `hash_`, None if none is."""
    for v in OBS_LAYOUTS:
        if spec_hash(action, stack, v) == hash_:
            return v
    return None


def assert_deployable(sensor_names: list[str]) -> None:
    """Raise if an observation would read a sensor the robot does not have."""
    from tactile_sim.sensors import is_sim_only

    bad = [n for n in sensor_names if is_sim_only(n)]
    if bad:
        raise ValueError(f"observation reads sim-only sensors: {bad}")
