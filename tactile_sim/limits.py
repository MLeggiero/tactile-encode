"""Hardware limits of the testbed, enforced where the real system enforces them and monitored otherwise.

Enforced in the command path (the sim cannot exceed them, just as the hardware cannot):
- FR3 joint torque (87 / 12 Nm): actuator ctrlrange and the World's clip.
- FR3 torque rate (1000 Nm/s per joint, libfranka's `kMaxTorqueRate`): L1 rate-limits every command.
  libfranka rejects faster changes with a `controller_torque_discontinuity` reflex.
- Hand joint torque: Franka Hand 140 N peak per finger (70 N continuous hold); WUJI Hand 2 per-joint rated
  torque from the vendor model (MCP flexion 2.0 Nm, abduction 0.2 Nm, PIP/DIP 0.3 Nm, thumb CMC 0.6 Nm,
  thumb MCP/IP 0.3 Nm).

Monitored every physics step (the hardware would stop or be damaged; the sim records the worst ratio):
- FR3 joint velocity (2.62 rad/s joints 1-4, 5.26 / 4.18 / 5.26 rad/s joints 5-7): a velocity reflex on the
  real arm.
- FR3 joint position margin to the mechanical range.
- Payload beyond the flange (3 kg).
- Hard-stop load on the hand's joints: torque the mechanical stops carry, relative to the joint's rating.
  The stop rating is not published; by default a stop may carry the joint's rated torque.

Dexmate Vega-1P (right arm; ratings from Dexmate's URDF): joint torque 150 / 150 / 80 / 80 / 25 / 25 / 25 Nm,
enforced by the servos' force limits; joint velocity 2.4 rad/s (joints 1-2) and 2.7 rad/s (3-7), monitored;
payload 4.5 kg per arm (Dexmate's current figure), checked. Its interface takes joint position targets at
100 Hz (enforced by the scheduler) and a P-gain multiplier in [0.1, 4] (enforced in the config). Dexmate
publishes no torque-rate limit, so none is applied.

`LimitMonitor.summary()` returns the worst ratio for each limit (1.0 = at the limit) and `violations()` the
ones above 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FR3_TORQUE = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
FR3_TORQUE_RATE = 1000.0  # Nm/s, every joint
FR3_VELOCITY = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
FR3_PAYLOAD = 3.0  # kg
FRANKA_HAND_CONTINUOUS = 70.0  # N per finger
FRANKA_HAND_PEAK = 140.0


def rate_limit(tau: np.ndarray, tau_prev: np.ndarray, rate: float, dt: float) -> np.ndarray:
    """Clip a torque command so no joint changes faster than `rate` (Nm/s) over `dt`."""
    step = rate * dt
    return np.clip(tau, tau_prev - step, tau_prev + step)


@dataclass
class LimitMonitor:
    """Tracks the worst limit ratios over an episode (or a strike, after `reset`)."""

    world: object
    stop_factor: float = 1.0  # hard-stop load allowed, in multiples of the joint's rated torque
    worst: dict[str, float] = field(default_factory=dict)
    where: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        w = self.world
        m = w.model
        self._prev_tau = None
        self.spec = w.arm_spec
        self._jr = m.jnt_range[[m.joint(n).id for n in self.spec.joints]]
        hand = w.hand
        self._hand_dofs = getattr(hand, "dofs", None)
        self._hand_tau = getattr(hand, "tau_max", None)
        self._hand_names = getattr(hand, "joint_names", None)

    def reset(self) -> None:
        self.worst.clear()
        self.where.clear()
        self._prev_tau = None

    def _note(self, key: str, ratio: float, where: str = "") -> None:
        if ratio > self.worst.get(key, -np.inf):
            self.worst[key] = float(ratio)
            self.where[key] = where

    def step(self) -> None:
        w = self.world
        d = w.data
        sp = self.spec
        tau = w.arm_torque()
        k = int(np.argmax(np.abs(tau) / sp.torque))
        self._note("arm_torque", abs(tau[k]) / sp.torque[k], f"joint {k + 1}")
        if self._prev_tau is not None and sp.torque_rate:
            # commands change once per L1 period, so a step's change is the whole period's change
            r = np.abs(tau - self._prev_tau) * w.cfg.controller.rate / sp.torque_rate
            k = int(np.argmax(r))
            self._note("arm_torque_rate", r[k], f"joint {k + 1}")
        self._prev_tau = tau.copy()
        qd = d.qvel[w.arm_dofs]
        k = int(np.argmax(np.abs(qd) / sp.velocity))
        self._note("arm_velocity", abs(qd[k]) / sp.velocity[k], f"joint {k + 1}")
        q = d.qpos[w.arm_qadr]
        span = self._jr[:, 1] - self._jr[:, 0]
        # fraction of each joint's half range used: 1.0 = at a mechanical end
        used = np.abs(q - self._jr.mean(axis=1)) / (0.5 * span)
        k = int(np.argmax(used))
        self._note("arm_range", used[k], f"joint {k + 1}")
        if self._hand_dofs is not None:
            from tactile_sim.model.hands.wuji_grasp import joint_loads

            stop = np.abs(joint_loads(w.model, d, self._hand_dofs)) / (self.stop_factor * self._hand_tau)
            k = int(np.argmax(stop))
            self._note("hand_stop_load", stop[k], self._hand_names[k])
            act = np.abs(d.qfrc_actuator[self._hand_dofs]) / self._hand_tau
            k = int(np.argmax(act))
            self._note("hand_torque", act[k], self._hand_names[k])

    def summary(self) -> dict[str, float]:
        return dict(self.worst)

    def violations(self, tol: float = 1e-6) -> dict[str, tuple[float, str]]:
        return {k: (v, self.where.get(k, "")) for k, v in self.worst.items() if v > 1.0 + tol}


def payload_mass(world) -> float:
    """Mass carried beyond the arm's flange: F/T body, hand, and the tool."""
    from tactile_sim import names

    m = world.model
    ft = m.body(names.FT_BODY).id
    return float(m.body_subtreemass[ft] + m.body_mass[world.hammer_body])


def check_payload(world) -> float:
    """Payload as a fraction of the arm's rating."""
    return payload_mass(world) / world.arm_spec.payload


