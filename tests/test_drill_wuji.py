"""The pistol driver in the WUJI Hand 2's wrap on the Vega U: the tool, the screw and hole tasks fed by a position
arm, the plant's load forecasts, and the drill as the residual's base."""

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.config import VEGA_DRILL_OVERRIDES, DrillCfg, fast_config, hand_config, residual_config
from tactile_sim.model.tools import pistol_tip_local

requires_wuji = pytest.mark.skipif(not hand_available("wuji2"), reason="WUJI Hand 2 not cached")


def drill_cfg(**drill):
    cfg = hand_config("wuji2", fast_config(), robot="vega_1u").replace(**VEGA_DRILL_OVERRIDES)
    return cfg.replace(drill=drill) if drill else cfg


def test_pistol_geometry():
    dc = DrillCfg(form="pistol")
    tip = pistol_tip_local(dc)
    assert tip[0] == -dc.pistol_grip_from_head and tip[1] < -0.2 and tip[2] == 0.0
    assert dc.pistol_housing_mass + dc.pistol_grip_mass + dc.pistol_battery_mass == pytest.approx(1.12)


def test_the_wuji_takes_the_pistol_but_not_the_inline_driver():
    from tactile_sim.model.builder import build_scene

    with pytest.raises(ValueError, match="pistol"):
        build_scene(drill_cfg(form="inline"))


@requires_wuji
@pytest.mark.parametrize("mode", ["screw", "hole"])
def test_the_vega_drives_a_screw_and_drills_through(mode):
    from tactile_sim.tool_task import ToolTask

    tt = ToolTask(drill_cfg(mode=mode), seed=0, t_max=16.0)
    w = tt.tb.world
    assert w.model.body_mass[w.hammer_body] == pytest.approx(1.12, abs=1e-6)
    res = tt.run()
    s = res.summary
    assert w.plant.done() and not s["lost"]
    assert s["peak_torque"] < 2.5 and s["max_slip_rot"] < 1.0
    if mode == "screw":
        assert s["t_done"] < 4.0 and s["cam_outs"] == 0
    else:
        assert res.truth["drill_depth"][-1] == pytest.approx(w.cfg.drill.board_thickness)


@requires_wuji
def test_the_plant_forecasts_seating_then_reports_the_clutch():
    from tactile_sim.tool_task import ToolTask

    tt = ToolTask(drill_cfg(mode="screw"), seed=0, t_max=6.0)
    tb, plant = tt.tb, tt.tb.world.plant
    seen = []
    tb.sched.register("probe", 100.0, lambda t: seen.append((t, plant.state, plant.event_truth(t))))
    tt.run()
    ahead = [(t, e) for t, st, e in seen if st == "engaged" and e.valid]
    assert ahead and all(e.regime == "periodic" and e.t_event >= t - 1e-9 for t, e in ahead)
    t_seat = next(t for t, st, _ in seen if st == "seated")
    early = [e.t_event for t, e in ahead if t < t_seat - 0.3]
    assert early and np.median(np.abs(np.array(early) - t_seat)) < 0.3  # seating forecast within 0.3 s
    under_way = [e for t, st, e in seen if st == "seated" and e.valid and e.f_dom > 0]
    assert len(under_way) >= 20  # the clutch ratchets while the trigger stays pulled after seating
    assert np.median([e.f_dom for e in under_way]) > 10.0  # at the detents' rate


@requires_wuji
def test_the_drill_drives_through_the_residual():
    from tactile_sim.behaviors.base_sources import DrillBase
    from tactile_sim.tool_task import ToolTask

    tt = ToolTask(residual_config(base=fast_config(), task="drill", forecast={"mask_prob": 0.0}), seed=0, t_max=6.0)
    tb = tt.tb
    assert tb.residual.cfg.task == "drill" and len(tb.residual.trig) == 0
    base = DrillBase(tb, tt.behavior, rng=np.random.default_rng(0))
    seen = []

    def recording(t):
        out = base(t)
        seen.append((tt.behavior.phase, out[1].copy(), out[2], out[3].copy()))
        return out

    tb.residual.base_fn = recording
    res = tt.run()
    assert tb.world.plant.done() and not res.summary["lost"] and tb.residual.n_nonfinite == 0
    assert any(trig for ph, _, trig, _ in seen if ph == "drive") and not any(trig for ph, _, trig, _ in seen
                                                                             if ph == "approach")
    approach = np.array([lk for ph, lk, _, _ in seen if ph == "approach"])
    assert np.abs(approach).max() > 1e-3  # the arm moves ahead of its targets
    fc = np.array([f for ph, _, _, f in seen if ph == "drive"])
    assert np.any(fc[:, 0] == 1.0)  # the seating is forecast while the screw goes in
