import os

import pytest

from tactile_sim.assets import menagerie_available
from tactile_sim.config import fast_config

HAS_MENAGERIE = menagerie_available()
requires_menagerie = pytest.mark.skipif(not HAS_MENAGERIE, reason="Menagerie assets not cached")


def pytest_report_header(config):
    return f"menagerie assets: {'present' if HAS_MENAGERIE else 'absent (fallback arm)'}; " \
           f"TACTILE_SIM_ASSETS={os.environ.get('TACTILE_SIM_ASSETS', '')!r}"


@pytest.fixture
def cfg_fast():
    """0.25 ms physics, Menagerie FR3 if cached, else the fallback arm."""
    return fast_config()


@pytest.fixture(scope="module")
def settled_world():
    from tactile_sim.sim.world import World

    w = World(fast_config())
    w.reset()
    return w
