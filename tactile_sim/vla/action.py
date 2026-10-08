"""The VLA's state and action on the Vega U + WUJI Hand 2, and the hand's per-task MIT table.

State (27): the striking (right) arm's 7 joint angles, then the hand's 20, as the host reads them.
Action (27): the arm's 7 joint position targets, before the host's gravity-droop compensation (PositionL1 adds it
on the way to the servos), then the hand's 20 MIT position targets q*. The VLA outputs no gains: the hand's Kp,
Kd and tau_ff come from a fixed per-task table (HandTable), as on the robot, and the residual (L2) adds to the
resulting command.

Demos come from bases that drive the hand another way (the grip synergy: torque on the closing joints, a PD on
the shaping ones). Their hand action is the table-equivalent target: the q* that, under the table, gives the
base's static torque at the angles the host read (HandTable.equivalent_targets). A base that holds its grip level
at the table's reference gives q* = q on the closing joints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tactile_sim.control.obs_spec import N_ARM, N_JOINTS
from tactile_sim.model.robots import VEGA_ARM
from tactile_sim.sim.wuji_drive import JOINT_GROUPS, joint_group

STATE_DIM = ACTION_DIM = N_ARM + N_JOINTS
ARM = slice(0, N_ARM)
HAND = slice(N_ARM, N_ARM + N_JOINTS)
ARM_NAMES = VEGA_ARM
TRIGGER_KP_SCALE = 4.0


@dataclass(frozen=True)
class HandTable:
    """Per-joint MIT gains and feedforward the VLA's hand targets run under (20 each)."""

    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray

    @classmethod
    def from_synergy(cls, synergy, s_ref: float, joint_names: list[str], kp_nom_groups,
                     trigger_kp_scale: float = TRIGGER_KP_SCALE) -> HandTable:
        """The synergy's MIT law at the grip level s_ref (a trigger finger at its resting level), with the vendor's
        nominal Kp on the closing joints (whose synergy Kp is 0), so a target steers them around the squeeze the
        feedforward holds. A trigger finger's Kp is trigger_kp_scale times stiffer, so a full pull stays within
        about a radian of where the finger is."""
        _, _, kp, kd, tau = synergy.mit_command(s_ref, np.zeros(len(joint_names)))
        kp_nom = np.asarray(kp_nom_groups, float)[[JOINT_GROUPS.index(joint_group(n)) for n in joint_names]]
        kp_nom[list(getattr(synergy, "trigger", ()))] *= trigger_kp_scale
        return cls(np.where(kp > 0, kp, kp_nom), np.asarray(kd, float), np.asarray(tau, float))

    @classmethod
    def for_hand(cls, hand, cfg) -> HandTable:
        """The table for a testbed's hand, built when the task starts: the synergy at the grip its scripted base
        holds, the WUJI's nominal Kp. The hammer's swing holds the configured grip (ControllerCfg.grip_hold, capped at
        the hand's hold limit); the drill feed sets none and keeps the grasp's settled grip (hand.s, full). A table
        at the base's own grip makes the demos' hand targets the joints' angles while the grip holds."""
        if cfg.plant.kind == "drill":
            s_ref = float(hand.s)
        else:
            s_ref = min(cfg.controller.grip_hold, hand.grip_hold_max) / hand.f_full
        return cls.from_synergy(hand.synergy, s_ref, list(hand.joint_names), cfg.wuji.kp_nom)

    def hold(self, q: np.ndarray | None) -> np.ndarray:
        """Base MIT command (5, n) that holds the hand where it is under this table, at the table's squeeze; with no
        reading of the angles yet, the squeeze alone (no stiffness)."""
        if q is None:
            return np.stack([np.zeros_like(self.kp), np.zeros_like(self.kp), np.zeros_like(self.kp), self.kd,
                             self.tau])
        return self.command(q)

    def command(self, q_star: np.ndarray) -> np.ndarray:
        """Base MIT command (5, n) rows q, qd, kp, kd, tau for the VLA's hand targets."""
        q_star = np.asarray(q_star, float).reshape(len(self.kp))
        return np.stack([q_star, np.zeros_like(q_star), self.kp, self.kd, self.tau])

    def equivalent_targets(self, base: np.ndarray, q: np.ndarray) -> np.ndarray:
        """q* that gives a base MIT command's (5, 20) static torque at the angles q under this table."""
        q_t, _, kp, _, tau = np.asarray(base, float)
        q = np.asarray(q, float)
        return q + (kp * (q_t - q) + tau - self.tau) / self.kp

    def to_dict(self) -> dict:
        return {"kp": self.kp.tolist(), "kd": self.kd.tolist(), "tau": self.tau.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> HandTable:
        return cls(np.asarray(d["kp"], float), np.asarray(d["kd"], float), np.asarray(d["tau"], float))
