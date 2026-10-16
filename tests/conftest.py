import os

# offscreen rendering (the VLA camera tests) through EGL: MuJoCo picks its GL backend when it is first imported, and
# GLFW without a display aborts the whole process instead of raising
os.environ.setdefault("MUJOCO_GL", "egl")

import pytest  # noqa: E402

from tactile_sim.assets import menagerie_available  # noqa: E402
from tactile_sim.config import fast_config  # noqa: E402

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
