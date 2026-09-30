import os

import pytest

from tactile_sim.assets import menagerie_available

HAS_MENAGERIE = menagerie_available()
requires_menagerie = pytest.mark.skipif(not HAS_MENAGERIE, reason="Menagerie assets not cached")


def pytest_report_header(config):
    return f"menagerie assets: {'present' if HAS_MENAGERIE else 'absent (fallback arm)'}; " \
           f"TACTILE_SIM_ASSETS={os.environ.get('TACTILE_SIM_ASSETS', '')!r}"
