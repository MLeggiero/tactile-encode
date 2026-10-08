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
    assert ca.controller.grip_hold != cb.controller.grip_hold
    tiny = str(__import__("pathlib").Path(__file__).parent / "data" / "tiny_policy.onnx")
    last = e1_conditions({"s1": tiny})[-1]
    assert last.name == "residual:s1" and last.overrides["residual"]["rate"] == 1000.0  # the policy's own rate


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
