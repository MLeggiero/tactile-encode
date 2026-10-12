"""The residual hand policy's interface (obs_spec) and its loop on the testbed (control.residual_l2): layout,
action map, per-second slew, safety (non-finite actions, stale inputs, the drill-trigger interlock), and a zero
policy passing the base command through unchanged."""

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.config import fast_config, hand_config
from tactile_sim.control.obs_spec import (
    ACTION_DIM,
    N_JOINTS,
    ActionSpec,
    JointParams,
    arm_lookahead,
    assert_deployable,
    base_tau_obs,
    compose,
    forecast_vector,
    obs_dim,
    obs_slices,
    offsets,
    slew,
    spec_hash,
    stack_for,
)
from tactile_sim.sim.scheduler import RateScheduler
from tactile_sim.sim.wuji_drive import joint_group

requires_wuji = pytest.mark.skipif(not hand_available("wuji2"), reason="WUJI Hand 2 not cached")

FINGERS = ("index_finger", "middle_finger", "ring_finger", "pinky")
NAMES = ([f"r_thumb_{j}" for j in ("cmc_flex", "cmc_abd", "mcp", "ip")] +
         [f"r_{f}_{j}" for f in FINGERS for j in ("mcp_flex", "mcp_abd", "pip", "dip")])
RATED = {"mcp_flex": 2.0, "mcp_abd": 0.2, "pip_dip": 0.3, "thumb_cmc": 0.6, "thumb_mcp_ip": 0.3}
JP = JointParams.for_joints(NAMES, [RATED[joint_group(n)] for n in NAMES], (0.37, 0.43, 0.22, 0.55, 0.22),
                            np.full(20, 3e-4))
SPEC = ActionSpec()


def test_observation_layout():
    fast = 448 + 66 + 8 + 20 + 20
    # v2: the 40-D last action, the base's target and feedforward, the arm's lookahead (7 x 3)
    assert ACTION_DIM == 40 and obs_dim(1) == fast + 40 + 20 + 20 + 21 + 13 + 3 + 1 == 680
    assert obs_dim(4) == 680 + 3 * fast
    for stack in (1, 2, 4, 10):
        sl = list(obs_slices(stack).values())
        assert sl[0].start == 0 and sl[-1].stop == obs_dim(stack)
        assert all(a.stop == b.start for a, b in zip(sl, sl[1:], strict=False))
    assert [stack_for(r) for r in (1000, 500, 250, 100)] == [1, 2, 4, 10]


def test_zero_action_reproduces_the_base_exactly():
    rng = np.random.default_rng(0)
    # gains far above kp_nom included (the synergy's 2.0 Nm/rad on a held fingertip joint: OBS_SPEC_VERSION 1 clamped
    # the total stiffness to 8 kp_nom, 1.76 on a PIP/DIP, so a = 0 did not reproduce the real hammer's base)
    base = np.stack([rng.uniform(-1, 1, 20), rng.uniform(-1, 1, 20), rng.choice([0.0, 0.3, 2.0, 3.0], 20),
                     rng.uniform(0, 0.05, 20), rng.uniform(-0.3, 0.3, 20)])
    out = compose(SPEC, JP, base, offsets(SPEC, JP, np.zeros(ACTION_DIM)))
    assert np.array_equal(out, base)


def test_the_action_moves_the_set_point_and_the_feedforward_only():
    rng = np.random.default_rng(1)
    base = np.stack([rng.uniform(-1, 1, 20), rng.uniform(-1, 1, 20), rng.choice([0.0, 2.0], 20),
                     rng.uniform(0, 0.05, 20), rng.uniform(-0.3, 0.3, 20)])
    out = compose(SPEC, JP, base, offsets(SPEC, JP, np.ones(ACTION_DIM)))
    assert out[0] == pytest.approx(base[0] + np.asarray(SPEC.dq_max)[JP.group])
    assert out[4] == pytest.approx(base[4] + SPEC.tau_frac * JP.tau_rated)
    assert np.array_equal(out[1:4], base[1:4])  # qd*, Kp, Kd: the base's (configuration on WUJI Hand 2)
    lo = compose(SPEC, JP, base, offsets(SPEC, JP, -np.ones(ACTION_DIM)))
    assert lo[4] == pytest.approx(base[4] - SPEC.tau_frac * JP.tau_rated)


def test_the_base_blocks_of_the_observation():
    base = np.zeros((5, 20))
    base[4] = 0.5 * JP.tau_rated
    assert base_tau_obs(base, JP.tau_rated) == pytest.approx(np.full(20, 0.5))
    look = np.arange(27 * 3, dtype=float)  # a base source's lookahead: hand 20 then arm 7, per time
    arm = arm_lookahead(look)
    assert arm.shape == (21,) and np.array_equal(arm.reshape(3, 7), look.reshape(3, 27)[:, 20:])
    assert np.array_equal(arm_lookahead(arm), arm)


def test_slew_limit_is_per_second():
    target = offsets(SPEC, JP, np.ones(ACTION_DIM))
    a = np.zeros((2, N_JOINTS))
    for _ in range(4):
        a = slew(SPEC, JP, a, target, 0.001)
    b = slew(SPEC, JP, np.zeros((2, N_JOINTS)), target, 0.004)
    assert a == pytest.approx(b)  # four 1 ms steps move as far as one 4 ms step
    assert b[0] == pytest.approx(np.minimum(target[0], SPEC.dq_rate * 0.004))


def test_forecast_hash_and_deployability():
    assert np.all(forecast_vector(False) == 0) and forecast_vector(True, 0.1).shape == (13,)
    assert spec_hash(stack=1) != spec_hash(stack=4)
    assert spec_hash() != spec_hash(ActionSpec(tau_frac=0.5))
    assert_deployable(["pressure_palm", "hand_q", "hand_current"])
    for bad in ("ft", "pad_acc_palm", "hammer_acc", "tau_ext"):
        with pytest.raises(ValueError):
            assert_deployable(["hand_q", bad])


@pytest.mark.parametrize("dt", [0.00025, 0.000125])
def test_every_ablation_rate_divides_the_physics_rate(dt):
    s = RateScheduler(dt)
    assert [s.steps_for(r) for r in (1000, 500, 250, 100)] == [round(1 / (r * dt)) for r in (1000, 500, 250, 100)]


# ---------------------------------------------------------------- on the testbed


def res_cfg(rate=1000.0, **res):
    return hand_config("wuji2", fast_config(), gripper={"hand_control": "mit"},
                       sensors={"ts_model": "rev3", "ts_taxel_time": 7.8e-6},
                       residual={"enabled": True, "rate": rate, **res})


@pytest.fixture(scope="module")
def res_tb():
    from tactile_sim.sim.testbed import Testbed

    return Testbed(res_cfg())


class Const:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def reset(self):
        pass

    def __call__(self, obs):
        return self.a


@requires_wuji
def test_zero_policy_passes_the_base_through(res_tb):
    tb = res_tb
    r = tb.residual
    r.policy = Const(np.zeros(ACTION_DIM))
    tb.reset(0)
    p0, _ = tb.world.hammer_in_hand()
    tb.run_for(0.01)
    startup = r.n_stale  # no taxel frame or joint readback has arrived in the first milliseconds
    assert startup <= 3
    tb.run_for(0.29)
    p1, _ = tb.world.hammer_in_hand()
    assert r.n_ticks == pytest.approx(300, abs=1) and tb.sched.rate("residual") == 1000.0
    assert np.array_equal(r.last_cmd.array(), r.last_base)
    assert np.linalg.norm(p1 - p0) < 0.2e-3
    obs = r.last_obs
    assert obs.shape == (obs_dim(1),) and np.all(np.isfinite(obs))
    codes = obs[obs_slices(1)["taxel_codes"]]
    assert codes.min() >= 0.0 and codes.max() <= 1.0 and codes.max() > 0.0
    assert r.n_stale == startup and r.n_nonfinite == 0


@requires_wuji
def test_nonfinite_actions_and_stale_inputs_zero_the_residual(res_tb):
    tb = res_tb
    r = tb.residual
    tb.reset(0)
    r.policy = Const(np.full(ACTION_DIM, np.nan))
    tb.run_for(0.02)
    assert r.n_nonfinite >= 19 and np.all(r.off == 0.0)
    r.policy = Const(np.full(ACTION_DIM, 0.5))
    tb.run_for(0.05)
    built = np.abs(r.off).max()
    assert built > 0
    stale_taxel = r.cfg.stale_taxel
    r.cfg.stale_taxel = -1.0  # every frame now counts as stale
    try:
        tb.run_for(0.1)
    finally:
        r.cfg.stale_taxel = stale_taxel
    assert np.abs(r.off).max() < 0.05 * built and r.n_stale >= 99
    r.policy = Const(np.zeros(ACTION_DIM))


class Stateful:
    """A recurrent policy that keeps whatever it is fed in its state (an ONNX GRU's h does the same with a NaN)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 4))

    def __call__(self, obs):
        self.h = self.h + float(np.sum(obs)) * 0.0 + float(obs[0])
        return np.zeros(ACTION_DIM)


@requires_wuji
def test_nonfinite_inputs_never_reach_the_policy_state_nor_the_drives(res_tb):
    tb = res_tb
    r = tb.residual
    tb.reset(0)
    pol = Stateful()
    r.policy = pol
    base_fn = r.base_fn
    poisoned = {"n": 0}

    def nan_obs_base(t):
        base, look, trig, fc = base_fn(t)
        fc = np.array(fc, float)
        fc[0] = np.nan  # a non-finite input (a forecast field)
        poisoned["n"] += 1
        return base, look, trig, fc

    r.base_fn = nan_obs_base
    tb.run_for(0.02)
    assert r.n_obs_nonfinite >= 19 and np.all(np.isfinite(pol.h)) and r.n_state_reset == 0

    def nan_base(t):
        base, look, trig, fc = base_fn(t)
        base = np.array(base, float)
        base[4, 3] = np.nan
        return base, look, trig, fc

    good = r.good_base.copy()
    r.base_fn = nan_base
    try:
        tb.run_for(0.02)
    finally:
        r.base_fn = base_fn
    assert r.n_base_nonfinite >= 19 and np.all(np.isfinite(r.last_cmd.array()))
    assert np.array_equal(r.last_base, good)  # the last finite base holds
    pol.h[:] = np.nan  # a state that turned non-finite anyway restarts from zero
    tb.run_for(0.002)
    assert r.n_state_reset >= 1 and np.all(np.isfinite(pol.h))
    r.policy = Const(np.zeros(ACTION_DIM))


@requires_wuji
def test_trigger_joints_only_open_while_the_trigger_is_disabled(res_tb):
    from dataclasses import replace

    from tactile_sim.control.residual_l2 import ResidualHandL2

    r = ResidualHandL2(res_tb, cfg=replace(res_tb.cfg.residual, task="drill"))
    j = r.trig
    assert len(j) == 3
    off = np.zeros((2, N_JOINTS))
    off[0, j] = 0.2 * r.close_dir[j]  # close the finger hard
    off[1, j] = 1.0 * r.close_dir[j]
    shut = r._trigger_guard(off, trigger=False)
    assert np.all(r.close_dir[j] * shut[0, j] <= 0) and np.all(r.close_dir[j] * shut[1, j] <= 0)
    enabled = r._trigger_guard(off, trigger=True)
    assert np.all(np.abs(enabled[0, j]) <= r.cfg.trigger_dq_max + 1e-12)
    assert np.all(r.close_dir[j] * enabled[1, j] <= r.cfg.trigger_tau[1] + 1e-12)
    opening = -off
    assert np.allclose(r._trigger_guard(opening, trigger=False)[0, j], -r.cfg.trigger_dq_max * r.close_dir[j])
    others = np.setdiff1d(np.arange(20), j)
    assert np.array_equal(r._trigger_guard(off, False)[:, others], off[:, others])


@requires_wuji
def test_the_trigger_interlock_is_the_drill_tasks_only(res_tb):
    from tactile_sim.config import ResidualCfg
    from tactile_sim.control.residual_l2 import trigger_indices

    r = res_tb.residual  # task "hammer": no trigger, no interlock
    assert r.cfg.task == "hammer" and len(r.trig) == 0
    off = np.random.default_rng(0).uniform(-1.0, 1.0, (2, N_JOINTS))
    for trigger in (False, True):
        assert np.array_equal(r._trigger_guard(off, trigger), off)
    names = r.hand.joint_names
    assert [len(trigger_indices(ResidualCfg(task=t), names)) for t in ("hammer", "loadgen", "drill")] == [0, 0, 3]


@requires_wuji
def test_a_slower_policy_sees_every_millisecond():
    from tactile_sim.sim.testbed import Testbed

    tb = Testbed(res_cfg(rate=250.0))
    r = tb.residual
    assert r.stack == 4 and tb.sched.rate("residual") == 250.0 and tb.sched.rate("residual_sample") == 1000.0
    tb.reset(0)
    tb.run_for(0.1)
    obs = r.last_obs
    assert obs.shape == (obs_dim(4),) and r.n_ticks == pytest.approx(25, abs=1)
    cur = obs[obs_slices(4)["hand_current"]].reshape(4, 20)
    assert np.all(np.isfinite(cur)) and len({c.tobytes() for c in cur}) == 4  # four distinct 1 kHz samples


DATA = __import__("pathlib").Path(__file__).parent / "data"


def test_an_exported_policy_matches_its_torch_outputs():
    pytest.importorskip("onnxruntime")
    from tactile_sim.control.policy_io import OnnxPolicy

    pol = OnnxPolicy(DATA / "tiny_policy.onnx")
    ref = np.load(DATA / "tiny_policy_io.npz")
    assert pol.recurrent and pol.meta.rate == 1000.0 and pol.meta.spec_hash == spec_hash(ActionSpec(), 1)
    out = np.array([pol(o) for o in ref["obs"]])  # the state carries across ticks
    assert out == pytest.approx(ref["action"], abs=1e-5)
    pol.reset()
    assert pol(ref["obs"][0]) == pytest.approx(ref["action"][0], abs=1e-5)


@requires_wuji
def test_a_policy_for_another_interface_or_rate_is_refused(res_tb):
    pytest.importorskip("onnxruntime")
    from tactile_sim.control.policy_io import OnnxPolicy

    pol = OnnxPolicy(DATA / "tiny_policy.onnx")
    res_tb.residual.check_policy(pol)  # 1 kHz, stack 1: matches
    pol.meta.rate = 250.0
    with pytest.raises(ValueError, match="Hz"):
        res_tb.residual.check_policy(pol)
    pol.meta.rate, pol.meta.spec_hash = 1000.0, "0" * 16
    with pytest.raises(ValueError, match="interface"):
        res_tb.residual.check_policy(pol)


@requires_wuji
def test_masked_blocks_read_as_zero_and_must_match_the_policy(res_tb):
    pytest.importorskip("onnxruntime")
    from tactile_sim.control.policy_io import OnnxPolicy
    from tactile_sim.sim.testbed import Testbed

    tb = Testbed(res_cfg(mask_blocks=("taxel_codes", "patch_features", "board_status")))
    r = tb.residual
    tb.reset(0)
    tb.run_for(0.05)
    sl = obs_slices(1)
    assert np.all(r.last_obs[sl["taxel_codes"]] == 0) and np.all(r.last_obs[sl["patch_features"]] == 0)
    assert np.any(r.last_obs[sl["hand_current"]] != 0)  # current-only student: the motor currents stay
    pol = OnnxPolicy(DATA / "tiny_policy.onnx")  # trained with every block
    with pytest.raises(ValueError, match="blind"):
        r.check_policy(pol)
