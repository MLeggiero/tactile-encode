"""Plant protocol shared by the nail (Task A) and the future drill/screw plant (Task B)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

import mujoco
import numpy as np


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
