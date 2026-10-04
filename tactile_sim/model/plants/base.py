"""Plant protocol shared by the nail (hammer), saw and drill/screw plants."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

import mujoco
import numpy as np

from tactile_sim.model.xmlutil import sub


class Plant(ABC):
    """A task target. MJCF is emitted by `add_mjcf`; runtime hooks run once per physics step."""

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def add_mjcf(self, root: ET.Element, worldbody: ET.Element, target: np.ndarray, axis: np.ndarray) -> None:
        """Add bodies, joints and contact pairs. `target` is where the tool meets the plant; `axis` is
        the unit tool-motion direction at contact (the strike axis for the hammer)."""

    @abstractmethod
    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Resolve ids after the model is compiled."""

    def reset(self, rng: np.random.Generator | None = None) -> None:  # noqa: B027 - optional hook
        """Called after the scene is reset and settled."""

    def pre_step(self) -> None:  # noqa: B027 - optional hook
        """Called before every mj_step (depth-dependent resistance, restitution, motor models)."""

    @abstractmethod
    def truth(self) -> dict[str, float]:
        """Ground-truth task state (e.g. nail depth)."""

    @abstractmethod
    def done(self) -> bool:
        """Task complete."""


def add_table(worldbody: ET.Element, cfg, center: np.ndarray, top: float) -> None:
    """A table (visual only) under a workpiece lying flat: top surface at height `top`, centred under `center`."""
    tx, ty = cfg.scene.table_half
    table = sub(worldbody, "body", name="table", pos=(center[0], center[1], 0.0))
    sub(table, "geom", name="table_top", type="box", size=(tx, ty, 0.015), pos=(0, 0, top - 0.015),
        rgba=(0.55, 0.57, 0.6, 1), contype=0, conaffinity=0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            sub(table, "geom", type="cylinder", size=(0.02, 0.5 * (top - 0.03)),
                pos=(sx * (tx - 0.04), sy * (ty - 0.04), 0.5 * (top - 0.03)),
                rgba=(0.45, 0.47, 0.5, 1), contype=0, conaffinity=0)


def body_point_velocity(model: mujoco.MjModel, data: mujoco.MjData, body: int, point: np.ndarray,
                        buf: np.ndarray) -> np.ndarray:
    """World-frame linear velocity of a point fixed to `body`."""
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, buf, 0)
    return buf[3:] + np.cross(buf[:3], point - data.xpos[body])


def apply_point_forces(data: mujoco.MjData, body: int, forces: list[tuple[np.ndarray, np.ndarray]],
                       torque: np.ndarray | None = None) -> None:
    """Set the body's applied wrench (world frame, about its centre of mass) from point forces plus a free torque."""
    com = data.xipos[body]
    F = np.zeros(3)
    T = np.zeros(3) if torque is None else np.array(torque, dtype=float)
    for p, f in forces:
        F += f
        T += np.cross(p - com, f)
    data.xfrc_applied[body, :3] = F
    data.xfrc_applied[body, 3:] = T
