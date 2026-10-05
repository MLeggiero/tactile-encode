"""Multi-rate scheduler: callbacks that run every N physics steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Task:
    name: str
    every: int
    fn: Callable[[float], None]
    offset: int = 0


class RateScheduler:
    def __init__(self, physics_dt: float):
        self.dt = physics_dt
        self.tasks: list[_Task] = []
        self.k = 0

    def steps_for(self, hz: float) -> int:
        n = 1.0 / (hz * self.dt)
        every = max(1, int(round(n)))
        if abs(n - every) > 1e-6 * n:
            raise ValueError(f"{hz} Hz is not an integer divisor of the physics rate {1 / self.dt:.0f} Hz")
        return every

    def register(self, name: str, hz: float, fn: Callable[[float], None], offset: int = 0) -> None:
        """`fn(t)` runs on physics steps k where (k - offset) % every == 0, before mj_step."""
        self.tasks.append(_Task(name, self.steps_for(hz), fn, offset))

    def rate(self, name: str) -> float:
        for t in self.tasks:
            if t.name == name:
                return 1.0 / (t.every * self.dt)
        raise KeyError(name)

    def tick(self, t: float) -> None:
        for task in self.tasks:
            if (self.k - task.offset) % task.every == 0:
                task.fn(t)
        self.k += 1

    def reset(self) -> None:
        self.k = 0
