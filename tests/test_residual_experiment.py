"""The residual experiment harness: paired randomization, conditions, metrics and the report (fast, tiny)."""

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.experiments.residual_experiment import (
    e1_conditions,
    load_results,
    make_jobs,
    paired_bootstrap,
    per_seed,
    report,
    run_job,
)

requires_wuji = pytest.mark.skipif(not hand_available("wuji2"), reason="WUJI Hand 2 not cached")


def test_conditions_share_each_seeds_randomization():
    jobs = make_jobs("smoke")
    assert {j.condition.name for j in jobs} == {"base", "template_oracle", "reactive_taxel", "squeeze"}
    a = next(j for j in jobs if j.condition.name == "base" and j.seed == 1)
    b = next(j for j in jobs if j.condition.name == "squeeze" and j.seed == 1)
    (ca, da), (cb, db) = a.config(), b.config()
    assert da == db and ca.hammer.head_mass == cb.hammer.head_mass and ca.wuji.cmd_latency == cb.wuji.cmd_latency
    # the squeeze grips at the hand's full rating (a grip_hold above the hand's hold maximum was clipped back to the
    # base's own: a no-op)
    from tactile_sim.experiments.residual_experiment import SQUEEZE_LEVEL

    assert ca.residual.hold_level == 1.0 and cb.residual.hold_level == SQUEEZE_LEVEL
    assert SQUEEZE_LEVEL * ca.controller.grip_hold / 78.19 >= 1.0  # s capped at the full rating
    assert ca.hammer.model == "ycb" and ca.controller.grip_hold == cb.controller.grip_hold
    tiny = str(__import__("pathlib").Path(__file__).parent / "data" / "tiny_policy.onnx")
    last = e1_conditions({"s1": tiny})[-1]
    assert last.name == "residual:s1" and last.overrides["residual"]["rate"] == 1000.0  # the policy's own rate


def test_evaluation_seeds_never_meet_the_libraries():
    import inspect

    from tactile_sim.behaviors.base_library import LIBRARY_SEED0, export_library
    from tactile_sim.experiments.residual_experiment import STAGES

    assert inspect.signature(export_library).parameters["seed0"].default == LIBRARY_SEED0
    assert max(st["seeds"] for st in STAGES.values()) <= LIBRARY_SEED0
    assert max(j.seed for j in make_jobs("E1hold")) < LIBRARY_SEED0


def test_a_drop_is_neither_zero_slip_nor_skipped():
    """A strike a drop interrupted carries no slip (NaN, so the kept strikes carry the mean); an episode that dropped
    the tool before its first strike still counts as a drop."""
    def res(seed, strikes, dropped):
        return {"job": {"condition": "base", "seed": seed, "task": "hammer"}, "strikes": strikes, "dropped": dropped,
                "residual": {"i2t": 1.0, "i_peak_ratio": 1.0}}

    kept = {"slip_rot_deg": 2.0, "slip_trans_mm": 1.0, "advance_mm": 2.0}
    lost = {"slip_rot_deg": float("nan"), "slip_trans_mm": float("nan"), "advance_mm": 0.5}
    ps = per_seed([res(0, [kept, lost], True), res(1, [], True), res(2, [kept], False)])["base"]
    assert set(ps) == {0, 1, 2} and ps[1]["drops"] == 1.0 and np.isnan(ps[1]["slip_rot_deg"])
    assert ps[0]["slip_rot_deg"] == 2.0 and ps[0]["slip_trans_mm"] == 1.0 and ps[0]["drops"] == 1.0


def test_paired_bootstrap():
    d, lo, hi = paired_bootstrap([1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.6, 3.4])
    assert d == pytest.approx(-0.5, abs=1e-9) and lo <= d <= hi and hi < 0


@requires_wuji
def test_a_smoke_episode_writes_metrics_and_reports(tmp_path):
    for cond in ("base", "squeeze"):
        job = next(j for j in make_jobs("smoke") if j.condition.name == cond and j.seed == 0)
        r = run_job(job, str(tmp_path))
        assert r["status"] == "ok"
    res = load_results("smoke", tmp_path)
    assert len(res) == 2
    for r in res:
        assert len(r["strikes"]) == 3 and r["residual"]["ticks"] > 1000 and r["residual"]["i2t"] > 0
        assert all(np.isfinite(s["slip_rot_deg"]) for s in r["strikes"])
    ps = per_seed(res)
    assert set(ps) == {"base", "squeeze"} and ps["squeeze"][0]["i2t"] > ps["base"][0]["i2t"]  # squeezing costs
    text = report("smoke", tmp_path)
    assert "| squeeze | 1 |" in text


def test_later_stages_build_their_conditions():
    from tactile_sim.experiments.residual_experiment import SWEEPS, stage_conditions

    tiny = str(__import__("pathlib").Path(__file__).parent / "data" / "tiny_policy.onnx")
    e5 = stage_conditions("E5", {"t": tiny})
    assert len(e5) == 2 * sum(len(v[2]) for v in SWEEPS.values())  # base and the policy at every point
    lat = next(c for c in e5 if c.name == "residual:t@cmd_latency=0.003")
    assert lat.overrides["wuji"]["cmd_latency"] == 0.003 and lat.overrides["residual"]["rate"] == 1000.0
    job = next(j for j in make_jobs("E5", {"t": tiny}) if j.condition.name == "base@taxel_rate=500")
    cfg, _ = job.config()
    assert cfg.sensors.ts_rate == 500.0 and "base__taxel_rate-500" in str(job.path(__import__("pathlib").Path(".")))
    assert cfg.residual.stale_taxel == pytest.approx(0.006)  # two 2 ms frame periods plus latency
    e6 = stage_conditions("E6", {"t": tiny})
    heavy = next(c for c in e6 if c.name == "base@heavy_head")
    assert heavy.overrides["hammer"]["head_mass"] == 0.8
    assert [c.name for c in stage_conditions("E4", {"t": tiny})] == ["residual:t"]


def test_the_hold_stage_and_its_table_count_a_drop_as_a_lost_strike(tmp_path):
    """E1hold: the base at each hold level (the policies beside it); in the table a strike a drop interrupted (no
    window: the episode ended) is lost, the episode is not a success, and the kept strikes carry the slip."""
    import json

    from tactile_sim.experiments.residual_experiment import HOLDS, hold_pairs, hold_table

    jobs = [j for j in make_jobs("E1hold") if j.seed < 2]
    assert {j.condition.name for j in jobs} == {f"base@hold={h:g}" for h in HOLDS}
    cfg, _ = next(j for j in jobs if j.condition.name == "base@hold=0.3").config()
    assert cfg.residual.hold_level == 0.3
    nan = float("nan")
    for j in jobs:
        hold = j.config()[0].residual.hold_level
        dropped = hold == 0.3 and j.seed == 0
        strikes = [{"idx": k, "hit": True, "drop": False, "lost_w": False, "slip_trans_mm": hold, "slip_rot_deg": 2.0,
                    "path_mm": 5.0, "path_deg": 10.0, "i2t_w": 0.1, "advance_mm": 2.0} for k in range(3)]
        if dropped:  # interrupted in its first strike: no strike end, no window
            strikes = [{"idx": 0, "hit": False, "drop": True, "lost_w": False, "slip_trans_mm": 0.0,
                        "slip_rot_deg": 0.0, "path_mm": nan, "path_deg": nan, "i2t_w": nan, "advance_mm": 0.0}]
        p = j.path(tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"job": {"condition": j.condition.name, "seed": j.seed}, "strikes": strikes,
                                 "dropped": dropped}))
    text, rows = hold_table("E1hold", tmp_path)
    low = next(r for r in rows if r["hold"] == 0.3)
    assert low["strikes"] == 4 and low["lost"] == pytest.approx(0.25)
    assert low["episodes_dropped"] == 0.5 and low["lost_first"] == 0.5
    assert low["net_mm"] == pytest.approx(0.3) and low["advance_mm"] == pytest.approx(2.0)  # the kept strikes
    assert "| base | 1 | 2 / 6 | 0.0 % | 0 % | 0 % | 1.000 / 2.000 |" in text
    assert "| 1.00, 1.00, 1.00 | 0.30, 1.00, 1.00 |" in text  # the zero at 0.3: against itself, against the full hold
    pairs = hold_pairs("E1hold", tmp_path)  # per seed against the full hold: seed 0 at 0.3 kept no strike
    # lost per strike: none at 0.5; at 0.3 seed 0 lost its only strike (+1) and seed 1 none: +0.5
    assert "| base | 0.5 | 2 | +0 [+0, +0] | -0.5 [-0.5, -0.5] |" in pairs
    assert "| base | 0.3 | 2 | +0.5 [+0, +1] |" in pairs


def test_drill_jobs_have_their_own_conditions_and_place():
    from pathlib import Path

    from tactile_sim.experiments.residual_experiment import stage_conditions

    jobs = make_jobs("smoke", task="drill")
    assert {j.condition.name for j in jobs} == {"base", "reactive_taxel", "squeeze"}  # no impact-flag oracle
    j = jobs[0]
    assert j.path(Path("r")).parts[:3] == ("r", "drill", "smoke")
    cfg, dr = j.config()
    assert cfg.residual.task == "drill" and cfg.plant.kind == "drill" and "drill" in dr
    assert cfg.drill.trigger_mode == "finger"
    e6 = [c.name for c in stage_conditions("E6", task="drill")]
    assert "base@strong_clutch" in e6 and "base@heavy_head" not in e6


@requires_wuji
def test_a_drill_smoke_episode_writes_metrics_and_reports(tmp_path):
    job = next(j for j in make_jobs("smoke", task="drill") if j.condition.name == "base" and j.seed == 0)
    assert run_job(job, str(tmp_path))["status"] == "ok"
    res = load_results("smoke", tmp_path, task="drill")
    assert len(res) == 1
    r = res[0]
    assert r["job"]["task"] == "drill" and r["residual"]["ticks"] > 1000 and len(r["residual"]) > 0
    m = per_seed(res)["base"][0]
    assert set(m) >= {"success", "cam_outs", "slip_rot_deg", "t_seat", "i2t"}
    assert m["success"] in (0.0, 1.0) and np.isfinite(m["slip_rot_deg"])
    assert "| base | 1 |" in report("smoke", tmp_path, task="drill")


def test_rerun_check_compares_payloads_wall_time_aside():
    from tactile_sim.experiments.rerun_check import compare

    a = {"summary": {"slip": 1.5, "nan": float("nan"), "lost": False}, "strikes": [{"t": 0.1}, {"t": 0.2}],
         "wall_s": 12.0}
    n, diff, worst = compare(a, {**a, "wall_s": 99.0})
    assert n == 5 and diff == [] and worst == 0.0
    b = {**a, "summary": {**a["summary"], "slip": 1.75}, "strikes": [{"t": 0.1}]}
    n, diff, worst = compare(a, b)
    assert diff == ["/strikes[1]/t", "/summary/slip"] and worst == 0.25


def test_a_strike_window_reports_the_hands_mean_total_current():
    from tactile_sim.experiments.residual_experiment import Recorder

    rec = object.__new__(Recorder)  # the window needs only the per-millisecond logs
    rec.ms_t = [0.001 * k for k in range(1, 11)]
    rec.ms_pose = [np.zeros(6)] * 10
    rec.ms_path = [(0.0, 0.0)] * 10
    rec.ms_i2 = [0.0] * 10
    rec.ms_ia = [float(k) for k in range(1, 11)]  # A
    w = rec.window(0.0045, 0.0085)  # milliseconds 5-8
    assert w["iabs_w"] == pytest.approx(6.5)
    assert np.isnan(rec.window(0.0045, float("nan"))["iabs_w"])
