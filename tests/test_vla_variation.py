"""The planned demo generator (vla.variation): its sampling, the pre-check on a real Vega U + WUJI world, planned
demos end to end, their export, the coverage report and the varied-start evaluation's bookkeeping."""

import json

import numpy as np
import pytest

import tactile_sim.vla  # noqa: F401 - selects EGL before mujoco's first import, when this runs first
from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.behaviors.motion_plan import plan_from_dict, plan_to_dict
from tactile_sim.vla.variation import SPACES, candidate, task_shift, unit_point

requires_vega_wuji = pytest.mark.skipif(not (hand_available("wuji2") and hand_available("vega_1u")),
                                        reason="WUJI Hand 2 or the Vega U not cached")


def test_draws_are_deterministic_and_cover_the_space():
    assert np.array_equal(unit_point("hammer", 7, 5), unit_point("hammer", 7, 5))
    assert np.array_equal(unit_point("hammer", 7, 5, 3), unit_point("hammer", 7, 5, 3))
    assert not np.array_equal(unit_point("hammer", 7, 5), unit_point("hammer", 7, 6))
    assert not np.array_equal(unit_point("hammer", 7, 5, 1), unit_point("hammer", 7, 6, 1))  # retries: per episode
    pts = np.array([unit_point("drill", 3, i) for i in range(64)])
    assert pts.shape == (64, SPACES["drill"].dims)
    for axis in range(2, 5):  # the start position's three axes: every eighth of the range is visited
        assert len(set((pts[:, axis] * 8).astype(int))) == 8
    (xl, xh), (yl, yh) = SPACES["hammer"].task_shift
    for i in range(16):
        dx, dy = task_shift("hammer", 7, i)
        assert xl <= dx <= xh and yl <= dy <= yh and (dx, dy) == task_shift("hammer", 7, i)


def test_candidates_start_on_the_shell_or_from_an_anchor_posture():
    from tactile_sim.vla.variation import ANCHORS

    sp = SPACES["hammer"]
    p, R, q_hover = np.array([0.5, 0.0, 0.9]), np.eye(3), np.zeros(7)
    dists, anchors = [], 0
    for i in range(64):
        c = candidate("hammer", unit_point("hammer", 1, i), p, R, q_hover)
        if c["family"] == "anchor":
            anchors += 1
            assert c["anchor"] in ANCHORS
            assert np.all(np.abs(c["start_q"] - np.asarray(ANCHORS[c["anchor"]])) <= sp.anchor_jitter + 1e-12)
            continue
        off = c["start_pos"] - p
        r = float(np.linalg.norm(off))
        dists.append(r)
        assert sp.start_dist[0] - 1e-9 <= r <= sp.start_dist[1] + 1e-9
        if r > 1e-6:
            el = np.degrees(np.arcsin(off[2] / r))
            assert sp.start_elev[0] - 1e-6 <= el <= sp.start_elev[1] + 1e-6
        assert np.degrees(np.arccos(np.clip((np.trace(c["start_R"]) - 1) / 2, -1, 1))) <= np.sqrt(3) * sp.start_tilt
        assert sp.speed_frac[0] <= c["speed_frac"] <= sp.speed_frac[1]
        assert sp.junction[0] <= c["junction"] <= sp.junction[1]
        assert sp.apex_back[0] <= c["apex_offset"][0] <= sp.apex_back[1]
        assert sp.windup_time[0] <= c["windup_time"] <= sp.windup_time[1]
    assert abs(anchors / 64 - sp.anchor_share) <= 0.1  # the anchor share, within ten points
    assert min(dists) < 0.1 and max(dists) > 0.3  # the shell spans near and far starts
    # without the hover posture every start is a shell start (the pre-check's tests pose their own)
    assert all(candidate("hammer", unit_point("hammer", 1, i), p, R)["family"] == "shell" for i in range(16))


@pytest.fixture(scope="module")
def hammer_world():
    if not (hand_available("wuji2") and hand_available("vega_1u")):
        pytest.skip("WUJI Hand 2 or the Vega U not cached")
    from tactile_sim.vla.demos import EPISODES, demo_config

    cfg, _ = demo_config("hammer", 100_003, fast=True)
    tb, base, _ = EPISODES["hammer"](cfg, 100_003)
    return tb, base


def test_plans_pass_the_precheck_and_round_trip(hammer_world):
    from tactile_sim.vla.variation import MAX_ATTEMPTS, sample_plan

    tb, _ = hammer_world
    q_live = tb.world.data.qpos.copy()
    for i in range(4):
        plan, attempts = sample_plan("hammer", tb, 7, i)
        assert plan is not None and attempts[-1]["ok"] and len(attempts) <= MAX_ATTEMPTS
        assert plan_from_dict(json.loads(json.dumps(plan_to_dict(plan)))) == plan
    assert np.array_equal(tb.world.data.qpos, q_live)  # the pre-check works on a scratch copy


def test_the_precheck_rejects_unreachable_and_colliding_motions(hammer_world):
    from tactile_sim.behaviors.feasibility import ArmProbe
    from tactile_sim.vla.variation import precheck

    tb, _ = hammer_world
    w = tb.world
    probe = ArmProbe(w)
    q_hover = w.data.qpos[w.arm_qadr].copy()
    p, R = w.ctrl_pose()
    target = (np.asarray(w.hover_tcp, float), w.tcp_R_nominal)
    c = candidate("hammer", unit_point("hammer", 7, 0), p, R)
    far = dict(c, start_pos=p + np.array([0.0, 0.0, 0.6]))
    assert precheck(probe, "hammer", far, q_hover, target, tb.cfg)[0].reason == "unreachable"
    g = w.model.geom("table_top").id
    top = w.data.geom_xpos[g][2] + w.model.geom_size[g][2]
    low = dict(c, start_pos=np.array([p[0], p[1], top + 0.03]), start_R=R)  # the head (~4 cm below the grip) in it
    v = precheck(probe, "hammer", low, q_hover, target, tb.cfg)[0]
    assert not v.ok and v.reason in ("collision", "unreachable", "joint_margin")
    probe.set_q(q_hover)
    assert probe.min_clearance()[0] == np.inf  # the hover pose itself is clear


def test_path_caps_at_the_hover_pose_match_the_swings(hammer_world):
    from tactile_sim.behaviors.feasibility import ArmProbe

    tb, base = hammer_world
    w = tb.world
    v, a = ArmProbe(w).path_caps(w.geom, [0.0], w.data.qpos[w.arm_qadr].copy())
    assert v[0] == pytest.approx(base.swing.v_cap, rel=1e-3) and a[0] == pytest.approx(base.swing.a_cap, rel=1e-3)


@requires_vega_wuji
def test_a_planned_demo_starts_away_from_the_task_and_records_the_approach(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import tactile_sim.vla.demos as demos

    try:
        demos.ffmpeg_exe()
    except RuntimeError:
        pytest.skip("no ffmpeg")
    if not hasattr(np, "trapezoid"):
        monkeypatch.setattr(np, "trapezoid", np.trapz, raising=False)
    monkeypatch.setattr(demos, "MAX_STRIKES", 1)
    for index in range(3, 9):  # the first episode the pre-check accepts (a rejection is cheap: nothing is simulated)
        job = demos.DemoJob("hammer", index, 100 + index, str(tmp_path), fast=True, perturb=1.0, generator="planned",
                            seed0=100)
        try:
            r = demos.run_demo(job)
        except Exception as e:  # noqa: BLE001
            if "EGL" in str(e) or "GL" in type(e).__name__ or "gl" in str(e).lower():
                pytest.skip(f"offscreen rendering unavailable: {e}")
            raise
        if r["status"] == "ok":
            break
    assert r["status"] == "ok"
    d = demos.load_demo(job.stem.with_suffix(".npz"))
    m = d["meta"]
    assert m["generator"] == "planned" and m["status"] == "ok" and m["n_strikes"] == 1
    plan = plan_from_dict(m["plan"])
    assert np.linalg.norm(m["start_offset"]) > 0.01
    # the first recorded frame is at the start pose (after the unrecorded pre-roll), not lined up over the nail
    assert np.abs(d["state"][0, :7] - np.asarray(plan.start.q)).max() < 0.05
    assert d["t"][0] > 1.0  # the pre-roll took the first second
    pushes = m["pushes"]
    assert pushes and all(p["t0"] >= d["t"][0] for p in pushes)  # pushes only once the planned episode began
    assert m["depth"] > 1e-3  # the planned windup curve still drives the nail
    q = m["quality"]
    assert set(q["metrics"]) >= {"target_rate", "alternation", "max_strike_rot_deg", "off_centre", "limit_arm_velocity"}
    assert q["passed"] == (not q["reasons"])


def test_the_export_skips_rejected_records_and_names_generators(tmp_path):
    pytest.importorskip("pyarrow")
    from test_vla import fake_raw

    from tactile_sim.vla.lerobot_export import export

    raw = fake_raw(tmp_path)
    np.savez(raw / "episode_000009.npz", meta=json.dumps({"task": "hammer", "status": "rejected", "success": False,
                                                           "attempts": []}))
    s = export([raw], tmp_path / "ds", only_success=False)
    assert s["episodes"] == 3 and s["skipped"] == 1
    eps = [json.loads(x) for x in (tmp_path / "ds/meta/episodes.jsonl").read_text().splitlines()]
    assert {e["generator"] for e in eps} == {"scripted"} and all(e["quality_passed"] for e in eps)
    # a successful demo that fails the quality gates is skipped, unless everything is asked for
    f = raw / "episode_000000.npz"
    with np.load(f) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(str(d["meta"]))
    meta["quality"] = {"passed": False, "reasons": ["strike_turn: 6.0 deg"]}
    d["meta"] = np.array(json.dumps(meta))
    np.savez(f, **d)
    assert export([raw], tmp_path / "ds2")["episodes"] == 1  # of the two successes, one passes the gates
    s = export([raw], tmp_path / "ds3", only_success=False)
    eps = [json.loads(x) for x in (tmp_path / "ds3/meta/episodes.jsonl").read_text().splitlines()]
    assert s["episodes"] == 3 and [e["quality_passed"] for e in eps].count(False) == 1


def test_the_coverage_report_counts_outcomes_reasons_and_coverage():
    from tactile_sim.vla.coverage import report

    u_low, u_high = [0.1] * 20, [0.9] * 20
    metas = [
        {"task": "hammer", "success": True, "status": "ok", "start_offset": [0.05, 0.0, 0.0],
         "attempts": [{"ok": False, "reason": "collision", "where": "preroll[3] a/b"},
                      {"ok": True, "reason": "ok", "where": "", "u": u_low}],
         "plan": {"apex_offset": [0.01, 0.0, 0.0], "planner": "rrt"}, "quality": {"passed": True, "reasons": []}},
        {"task": "hammer", "success": True, "status": "ok", "start_offset": [0.05, 0.02, 0.0],
         "attempts": [{"ok": True, "reason": "ok", "where": "", "u": u_high}], "plan": {"apex_offset": [0.04, 0, 0]},
         "quality": {"passed": False, "reasons": ["strike_turn: 6.2 deg", "arm_velocity: 1.05x"]}},
        {"task": "hammer", "success": False, "status": "ok", "start_offset": [0.01, 0.0, 0.0],
         "attempts": [{"ok": True, "reason": "ok", "where": "", "u": u_high}], "plan": {"apex_offset": [0.04, 0, 0]}},
        {"task": "hammer", "success": False, "status": "rejected",
         "attempts": [{"ok": False, "reason": "unreachable", "where": "start"}] * 20},
    ]
    text = report(metas)
    assert "4 episodes: 1 exported, 1 failed the quality gates, 1 failed in simulation, 1 rejected" in text
    assert "1  strike_turn" in text and "1  arm_velocity" in text
    assert "20  unreachable (start)" in text and "1  collision (preroll)" in text
    assert "0-10 cm: 1/3 (33 %)" in text  # a demo failing the gates does not count as a success
    assert "rrt, shell: 1/1 (100 %)" in text and "curve, shell: 0/2 (0 %)" in text
    lines = text.splitlines()
    i = lines.index("  coverage after the pre-check (3 episodes; 5 bins per parameter, expected 0.6 each):")
    assert lines[i + 1].split(":")[1].split()[:5] == ["1", "0", "0", "0", "2"] and "UNDER-COVERED" in lines[i + 1]
    j = lines.index("  coverage exported (1 episodes; 5 bins per parameter, expected 0.2 each):")
    assert lines[j + 1].split()[1:6] == ["1", "0", "0", "0", "0"]


def test_varied_starts_run_and_report_apart_from_aligned_ones(tmp_path):
    from tactile_sim.experiments.vla_experiment import Condition, Job, start_root

    aligned = Job("hammer", Condition("vla"), 200_000, "groot://x:1")
    varied = Job("hammer", Condition("vla"), 200_000, "groot://x:1", starts="varied")
    assert aligned.path(tmp_path) == tmp_path / "vla" / "seed200000.json"
    assert varied.path(tmp_path) == tmp_path / "varied" / "vla" / "seed200000.json"
    assert start_root(tmp_path, "varied") == tmp_path / "varied"


def test_label_metrics_flag_steps_and_alternation():
    from tactile_sim.vla.quality import label_metrics

    v_lim = np.full(7, 2.4)
    t = np.linspace(0.0, 1.0, 31)
    smooth = np.outer(t**3 * (10 - 15 * t + 6 * t**2), np.full(7, 0.5))  # a 0.5 rad min-jerk move over 1 s
    m = label_metrics(smooth, 30.0, v_lim, 0.005)
    assert m["alternation"] == 0.0 and m["target_rate"] < 1.0
    jitter = smooth + np.outer((-1.0) ** np.arange(31), np.r_[0.03, np.zeros(6)])  # the closed-loop failure's pattern
    assert label_metrics(jitter, 30.0, v_lim, 0.005)["alternation"] > 0.5
    step = smooth.copy()
    step[15:, 0] += 0.2  # a 0.2 rad jump between frames: 6 rad/s
    assert label_metrics(step, 30.0, v_lim, 0.005)["target_rate"] > 1.0


def test_the_gates_check_arm_limits_slip_blows_and_labels_but_not_the_hands_stops():
    from tactile_sim.vla.quality import gate

    ok = {"limit_arm_velocity": 0.9, "limit_hand_stop_load": 4.0, "max_strike_slip_mm": 2.0, "max_strike_rot_deg": 2.0,
          "n_blows": 8, "off_centre": 1, "missed": 0, "target_rate": 0.8, "alternation": 0.01}
    assert gate("hammer", ok) == []  # the wrap's knuckles on their stops are normal
    bad = dict(ok, limit_arm_velocity=1.1, max_strike_rot_deg=7.0, off_centre=2, missed=1, alternation=0.2)
    keys = [r.split(":")[0] for r in gate("hammer", bad)]
    assert keys == ["arm_velocity", "strike_turn", "blows", "alternation"]
    drill = {"max_slip_mm": 1.0, "max_rot_deg": 6.5, "cam_outs": 1, "limit_arm_range": 1.2}  # 6.5 deg: normal roll
    assert [r.split(":")[0] for r in gate("drill", drill)] == ["arm_range", "cam_outs"]


def test_a_rejection_redraws_only_the_group_that_failed(hammer_world):
    from tactile_sim.vla.variation import APPROACH, START, SWING, redraw_groups, sample_plan

    assert redraw_groups("start") == [START] and redraw_groups("preroll[4] a/b") == [START]
    assert redraw_groups("approach[2]") == [START] and redraw_groups("approach") == [START]
    assert redraw_groups("align[3] driver_bit/board") == [SWING]  # the drill's angled entry shares the last group
    assert APPROACH.stop == SWING.start
    assert redraw_groups("stroke 0.81 of 0.93 m/s") == [SWING] and redraw_groups("windup") == [SWING]
    tb, _ = hammer_world
    for i in range(12):
        plan, attempts = sample_plan("hammer", tb, 7, i)
        if plan is None or len(attempts) < 2:
            continue
        touched = np.zeros(SPACES["hammer"].dims, bool)
        for a in attempts[:-1]:
            for g in redraw_groups(a["where"]):
                touched[g] = True
        u, sobol = np.asarray(attempts[-1]["u"]), unit_point("hammer", 7, i)
        assert np.array_equal(u[~touched], sobol[~touched])  # untouched coordinates keep their Sobol values
        assert not np.array_equal(u[touched], sobol[touched])
        return
    pytest.skip("no episode needed a retry")


def test_the_probe_sees_the_gap_to_a_primitive_not_only_overlap(hammer_world):
    """A mesh against a box (the hammer head over the board): MuJoCo's collision pass reports only overlap there, so
    the probe measures the gap itself; a pose 5 mm off the board fails CLEARANCE, as the pre-check means it to."""
    from tactile_sim.behaviors.feasibility import CLEARANCE, ArmProbe

    tb, _ = hammer_world
    w = tb.world
    probe = ArmProbe(w)
    q_hover = w.data.qpos[w.arm_qadr].copy()
    p, R = w.ctrl_pose()
    q0, *_ = probe.ik(p + np.array([-0.251, -0.145, -0.078]), R, q_hover)
    gaps = []
    for k in np.linspace(0.0, 1.0, 81):
        q = q0 + (q_hover - q0) * k
        d, pair = probe.clearance(q)
        if 0.0 < d < CLEARANCE:
            gaps.append((d, pair, probe.valid(q)))
    assert gaps, "no pose between 0 and CLEARANCE off an obstacle was seen"
    assert all(not ok for _, _, ok in gaps)  # near is not clear
    assert any("board" in pair for _, pair, _ in gaps)


def test_the_anchor_postures_are_valid_starts(hammer_world):
    from tactile_sim.behaviors.feasibility import ArmProbe
    from tactile_sim.vla.variation import ANCHORS

    tb, _ = hammer_world
    w = tb.world
    probe = ArmProbe(w)
    q_hover = w.data.qpos[w.arm_qadr].copy()
    for name, off in ANCHORS.items():
        assert probe.valid(q_hover + np.asarray(off)), name


def test_strikes_vary_around_the_episode_and_fall_back_when_infeasible(hammer_world, monkeypatch):
    import tactile_sim.vla.variation as var
    from tactile_sim.behaviors.feasibility import Verdict

    tb, _ = hammer_world
    plan, attempts = var.sample_plan("hammer", tb, 7, 1)
    assert plan is not None and len(plan.strikes) == var.N_STRIKE_DRAWS
    again, _ = var.sample_plan("hammer", tb, 7, 1)
    assert again.strikes == plan.strikes  # reproducible per episode
    apexes = np.array([s["apex_offset"] for s in plan.strikes])
    assert len({tuple(a) for a in apexes.round(6)}) > 1  # the strikes differ
    assert np.all(np.abs(apexes - np.asarray(plan.apex_offset)) <= var.STRIKE_JITTER["apex"] * np.sqrt(3) + 1e-9)
    assert all(abs(s["speed"] - 1.0) <= var.STRIKE_JITTER["speed"] + 1e-12 for s in plan.strikes)
    # with every jittered swing infeasible, the strikes keep the episode's checked values and the plan stands
    real = var.check_swing
    calls = {"n": 0}

    def fail_jitter(*a, **k):
        calls["n"] += 1
        return (Verdict(False, "slow_strike", "stroke"), {}) if calls["n"] > 1 else real(*a, **k)

    monkeypatch.setattr(var, "check_swing", fail_jitter)
    plan2, att2 = var.sample_plan("hammer", tb, 7, 1)
    assert plan2 is not None and att2[-1]["strike_fallbacks"] == var.N_STRIKE_DRAWS
    assert all(s["apex_offset"] == list(plan2.apex_offset) and s["speed"] == 1.0 for s in plan2.strikes)


@pytest.fixture(scope="module")
def drill_world():
    if not (hand_available("wuji2") and hand_available("vega_1u")):
        pytest.skip("WUJI Hand 2 or the Vega U not cached")
    from tactile_sim.vla.demos import EPISODES, demo_config

    cfg, _ = demo_config("drill", 100_003, fast=True)
    tb, base, _ = EPISODES["drill"](cfg, 100_003)
    return tb, base


def test_the_drill_comes_in_at_an_angle_and_its_near_pairs_only_must_not_overlap(drill_world):
    from tactile_sim.behaviors.feasibility import CLEARANCE, ArmProbe
    from tactile_sim.vla.variation import sample_plan

    tb, base = drill_world
    w = tb.world
    q_hover = w.data.qpos[w.arm_qadr].copy()
    probe = ArmProbe(w)
    d, _ = probe.clearance(q_hover)
    assert d < CLEARANCE and probe.valid(q_hover)  # the bit 1 cm short of the screw by design is still valid
    target = (base.feed.x0.copy(), base.feed.R.copy())
    for i in range(3):
        plan, attempts = sample_plan("drill", tb, 7, i, target=target)
        assert plan is not None, attempts
        assert 0.0 <= np.degrees(np.linalg.norm(plan.align_rotvec)) <= 15.0 + 1e-9
        assert 0.03 <= plan.align_back <= 0.06 and 0.0 <= plan.align_across <= 0.02
        assert plan.align_time > 0 and plan.approach_path is not None and len(plan.room) == 10


def test_pushes_in_a_planned_approach_start_only_where_the_plan_recorded_room():
    from types import SimpleNamespace

    from tactile_sim.vla.demos import DemoPerturbation

    feed = SimpleNamespace(phase="transit", t_phase=0.0, motion=SimpleNamespace(approach_time=2.0, room=(False,) * 10))
    base = SimpleNamespace(feed=feed, tb=None)
    pert = DemoPerturbation("drill", base, 1.0, np.random.default_rng(0))
    pert(0.0)
    assert pert.bumps == [] and pert.skipped == 1  # drawn, but no room anywhere
    feed.motion.room = (True,) * 10
    feed.t_phase = 5.0  # a new window
    pert(5.0)
    assert len(pert.bumps) == 1 and 5.0 <= pert.bumps[0][0] <= 7.0  # anywhere in the approach, where there is room


def test_the_clearance_gate_and_the_planned_push_default():
    from tactile_sim.vla import demos
    from tactile_sim.vla.quality import gate

    ok = {"limit_arm_velocity": 0.9, "max_strike_slip_mm": 2.0, "max_strike_rot_deg": 2.0, "n_blows": 8,
          "off_centre": 0, "missed": 0, "target_rate": 0.8, "alternation": 0.0, "free_clearance_ok": True,
          "free_clearance_mm": 7.0}
    assert gate("hammer", ok) == []
    bad = dict(ok, free_clearance_ok=False, free_clearance_mm=-2.0, free_clearance_pair="hammer_head/board_geom")
    assert gate("hammer", bad) == ["clearance: -2.0 mm hammer_head/board_geom"]
    assert demos.DemoJob("hammer", 0, 1, "out").perturb is None  # resolved per generator in run_demo
    assert demos.PLANNED_PERTURB > 0
