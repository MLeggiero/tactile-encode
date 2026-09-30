"""MuJoCo <sensor> declarations. The rate/noise/latency models live in tactile_sim.sensors."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from tactile_sim import names
from tactile_sim.model.xmlutil import sub

# name -> dim, in declaration order
MJ_SENSORS: list[tuple[str, int]] = []


def add_sensors(root: ET.Element, pad_sites: dict[str, str] | None = None) -> list[tuple[str, int]]:
    """F/T at the wrist, one accelerometer per taxel patch (`pad_sites`: patch name -> site; default the
    Franka pads L/R) and one in the hammer head."""
    s = sub(root, "sensor")
    out: list[tuple[str, int]] = []

    def add(tag, name, dim, **kw):
        sub(s, tag, name=name, **kw)
        out.append((name, dim))

    add("force", "ft_force", 3, site=names.FT_SITE)
    add("torque", "ft_torque", 3, site=names.FT_SITE)
    if pad_sites is None:
        pad_sites = dict(zip("LR", names.PAD_IMU_SITES, strict=True))
    for patch, site in pad_sites.items():
        add("accelerometer", f"pad_acc_{patch}", 3, site=site)
    add("accelerometer", "hammer_acc", 3, site=names.HAMMER_IMU_SITE)
    # pad pressure arrays are binned from contacts in tactile_sim.sim.world.World.pad_taxels: MuJoCo's touch
    # sensor also counts contacts whose normal ray crosses a site, so neighbouring taxels share a contact
    return out
