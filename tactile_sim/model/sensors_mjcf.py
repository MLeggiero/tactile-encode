"""MuJoCo <sensor> declarations. The rate/noise/latency models live in tactile_sim.sensors."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from tactile_sim import names
from tactile_sim.model.xmlutil import sub

# name -> dim, in declaration order
MJ_SENSORS: list[tuple[str, int]] = []


def add_sensors(root: ET.Element) -> list[tuple[str, int]]:
    s = sub(root, "sensor")
    out: list[tuple[str, int]] = []

    def add(tag, name, dim, **kw):
        sub(s, tag, name=name, **kw)
        out.append((name, dim))

    add("force", "ft_force", 3, site=names.FT_SITE)
    add("torque", "ft_torque", 3, site=names.FT_SITE)
    add("accelerometer", "pad_acc_L", 3, site=names.PAD_IMU_SITES[0])
    add("accelerometer", "pad_acc_R", 3, site=names.PAD_IMU_SITES[1])
    add("accelerometer", "hammer_acc", 3, site=names.HAMMER_IMU_SITE)
    # pad pressure arrays are binned from contacts in tactile_sim.sim.world.World.pad_taxels: MuJoCo's touch
    # sensor also counts contacts whose normal ray crosses a site, so neighbouring taxels share a contact
    return out
