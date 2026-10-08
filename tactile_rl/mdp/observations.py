"""Observation terms. Milestone 1 observes only the hand joints (angle, motor current / rated), true and undelayed;
the student's TaxelScan, readback timing and base lookahead (obs_spec.OBS_BLOCKS) come with milestone 2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tactile_rl.mdp.actions import hand_drive, joint_ids
from tactile_rl.scene.entities import ENTITY
from tactile_sim.sim.training_parts import training_parts

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class hand_joint_pos:  # noqa: N801 - mjlab term naming
    """Hand joint angles (rad), (N, 20) in the testbed's joint order."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        robot = env.scene[ENTITY]
        self.robot = robot
        self.ids = joint_ids(robot, training_parts(cfg.params["scene_key"]).hand_joints, env.device)

    def __call__(self, env: ManagerBasedRlEnv, scene_key: str) -> torch.Tensor:
        del scene_key
        return self.robot.data.joint_pos[:, self.ids]


def hand_current(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Hand motor currents as a fraction of each joint's rated current (obs_spec "hand_current"), (N, 20)."""
    drive = hand_drive(env.scene[ENTITY])
    return drive.i / drive.i_rated
