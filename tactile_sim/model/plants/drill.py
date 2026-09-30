"""Task B slot: cordless driver -> screw. Not implemented yet.

The intended plant is phenomenological (see docs/control_flowchart.html): a DC motor with a
clutch, optional percussion, and an engagement state machine {free-spin, engaged, cam-out,
seated} producing reaction torque and vibration, calibrated from the screwdriving datasets in
research_notes/. It will implement the same Plant protocol as NailPlant.
"""

from __future__ import annotations

from tactile_sim.model.plants.base import Plant


class DrillPlant(Plant):
    def __init__(self, cfg):
        raise NotImplementedError("DrillPlant (Task B) is a placeholder; only plant.kind='nail' is implemented")

    def add_mjcf(self, root, worldbody, target, axis):  # pragma: no cover
        raise NotImplementedError

    def bind(self, model, data):  # pragma: no cover
        raise NotImplementedError

    def truth(self):  # pragma: no cover
        raise NotImplementedError

    def done(self):  # pragma: no cover
        raise NotImplementedError
