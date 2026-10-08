"""The batched torch WUJI drive (tactile_rl.actuators.wuji_mit.TorchWujiDrive) against the numpy reference
(tactile_sim.sim.wuji_drive.WujiMitDrive) on identical command and joint-state streams: one torch env per reference
drive, each with its own parameters, the reference's jitter draws fed to the torch drive. |dtau| <= 1e-5 Nm at every
step, through transport delay and jitter, the SDK low-pass, the current clamp, the current loop, thermal derating
and the latched fault."""

import dataclasses

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.rl]  # torch, mjlab and CUDA (.venv-rl) or skipped
torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from tactile_rl.actuators.wuji_mit import TorchWujiDrive  # noqa: E402
from tactile_sim.config import WujiDriveCfg  # noqa: E402
from tactile_sim.control.interface import HandCommand  # noqa: E402
from tactile_sim.sim.wuji_drive import WujiMitDrive, joint_group  # noqa: E402

FINGERS = ("index_finger", "middle_finger", "ring_finger", "pinky")
NAMES = ([f"r_thumb_{j}" for j in ("cmc_flex", "cmc_abd", "mcp", "ip")] +
         [f"r_{f}_{j}" for f in FINGERS for j in ("mcp_flex", "mcp_abd", "pip", "dip")])
RATED = {"mcp_flex": 2.0, "mcp_abd": 0.2, "pip_dip": 0.3, "thumb_cmc": 0.6, "thumb_mcp_ip": 0.3}
TAU = np.array([RATED[joint_group(n)] for n in NAMES])
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# per-env overrides of WujiDriveCfg: delay and jitter, low-pass, current loop, clamp, fast heating to the fault
ENV_CFGS = [
    {},
    {"cmd_latency": 0.003, "cmd_jitter": 0.0009},
    {"cmd_latency": 0.0005, "cmd_jitter": 0.0, "sdk_lpf_hz": 50.0},
    {"sdk_lpf_hz": 300.0, "tau_i": 0.0005},
    {"cmd_jitter": 0.0, "tau_i": 0.0, "effort_limit_ratio": 1.5, "i_peak_ratio": 1.2},
    {"effort_limit_ratio": 2.0, "i_peak_ratio": 2.0, "c_th": 0.05, "r_th": 50.0, "t_warn": 40.0, "t_max": 60.0},
    {"cmd_latency": 0.0015, "cmd_jitter": 0.0004, "sdk_lpf_hz": 100.0, "t_amb": 33.0, "r_wind": 3.0},
    {"cmd_latency": 0.0, "cmd_jitter": 0.0, "effort_limit_ratio": 0.9},
]
PER_ENV = ("cmd_latency", "cmd_jitter", "r_wind", "r_th", "c_th", "t_amb", "t_warn", "t_max", "derate_floor", "tau_i")


def streams(n_env: int, steps: int, dt: float, seed: int = 0):
    """Host commands (changing every physics step, some envs silent at first) and smooth joint trajectories."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps) * dt
    f = rng.uniform(1.0, 30.0, (n_env, 1, 20))
    ph = rng.uniform(0, 2 * np.pi, (n_env, 1, 20))
    q = 0.6 * np.sin(2 * np.pi * f * t[None, :, None] + ph)  # (env, step, J)
    qd = 0.6 * 2 * np.pi * f * np.cos(2 * np.pi * f * t[None, :, None] + ph)
    cmd = np.empty((n_env, steps, 5, 20))
    cmd[:, :, 0] = 0.5 * np.sin(2 * np.pi * 3.0 * t[None, :, None] + ph) + rng.normal(0, 0.05, (n_env, steps, 20))
    cmd[:, :, 1] = rng.normal(0, 0.5, (n_env, steps, 20))
    cmd[:, :, 2] = np.abs(rng.normal(0.5, 0.5, (n_env, steps, 20)))
    cmd[:, :, 3] = np.abs(rng.normal(0.02, 0.01, (n_env, steps, 20)))
    cmd[:, :, 4] = rng.normal(0, 0.4, (n_env, steps, 20))
    cmd[5, :, 4] = 100.0  # the heating env asks for far more than the clamp gives
    cmd[4, :, 4] *= 10.0  # clamped some of the time
    first = rng.integers(0, 12, n_env)  # steps before each env's first host command
    first[0] = 0
    return q, qd, cmd, first


def run_parity(sdk_lpf_fields=("q",), servo_rate=None, steps=4000, dt=0.00025, seed=0):
    base = WujiDriveCfg(sdk_lpf_fields=sdk_lpf_fields, servo_rate=servo_rate)
    n_env = len(ENV_CFGS)
    cfgs = [dataclasses.replace(base, **c) for c in ENV_CFGS]
    refs = [WujiMitDrive(c, NAMES, TAU, dt, np.random.default_rng(100 + e)) for e, c in enumerate(cfgs)]
    mirrors = [np.random.default_rng(100 + e) for e in range(n_env)]  # replay of each reference's jitter draws

    drv = TorchWujiDrive(base, NAMES, TAU, dt, n_env, DEV)
    drv.set_params(**{k: torch.tensor([getattr(c, k) for c in cfgs], dtype=torch.float64) for k in PER_ENV},
                   sdk_lpf_hz=torch.tensor([c.sdk_lpf_hz or 0.0 for c in cfgs], dtype=torch.float64),
                   effort_limit_ratio=torch.tensor([[c.effort_limit_ratio] for c in cfgs]),
                   i_peak_ratio=torch.tensor([[c.i_peak_ratio] for c in cfgs]))
    drv.reset()

    def jitter(push: torch.Tensor) -> torch.Tensor:
        out = np.zeros(n_env)
        for e in np.flatnonzero(push.cpu().numpy()):
            j = cfgs[e].cmd_jitter
            if j > 0:
                out[e] = mirrors[e].uniform(-j, j)
        return torch.as_tensor(out, device=DEV)

    drv.jitter_fn = jitter
    q, qd, cmd, first = streams(n_env, steps, dt, seed)
    err = np.zeros(steps)
    tq = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    tqd = torch.as_tensor(qd, dtype=torch.float32, device=DEV)
    tcmd = torch.as_tensor(cmd, dtype=torch.float32, device=DEV)
    for k in range(steps):
        t = k * dt
        sending = k >= first
        ref_tau = np.zeros((n_env, 20))
        for e in range(n_env):
            if sending[e]:
                refs[e].set_host_command(HandCommand.from_array(t, cmd[e, k]))
            ref_tau[e] = refs[e].update(t, q[e, k], qd[e, k])
        ids = torch.as_tensor(np.flatnonzero(sending), device=DEV)
        if len(ids):
            drv.set_host(tcmd[ids, k], ids)
        tau = drv.step(tq[:, k], tqd[:, k]).cpu().numpy()
        err[k] = np.abs(tau - ref_tau).max()
    return err, refs, drv


@pytest.mark.parametrize("fields", [("q",), ("q", "qd", "kp", "kd", "tau")])
def test_torch_drive_matches_the_reference(fields):
    err, refs, drv = run_parity(sdk_lpf_fields=fields)
    assert err.max() <= 1e-5, f"max |dtau| {err.max():.3g} Nm at step {int(err.argmax())}"
    # the streams exercised what they should: the heating env faulted, the clamp engaged, commands were delayed
    assert refs[5].fault.all() and bool(drv.fault[5].all())
    assert not refs[0].fault.any() and not bool(drv.fault[0].any())
    T = drv.T.cpu().numpy()
    assert np.abs(T - np.array([r.T for r in refs])).max() < 1e-6
    assert np.array_equal(drv.n_sdk.cpu().numpy(), [r.n_sdk for r in refs])


def test_torch_drive_matches_the_reference_at_a_servo_rate():
    err, _, _ = run_parity(servo_rate=1000.0, steps=1200)
    assert err.max() <= 1e-5


def test_no_torque_before_the_first_command_and_state_round_trip():
    cfg = WujiDriveCfg()
    drv = TorchWujiDrive(cfg, NAMES, TAU, 0.00025, 3, DEV)
    z = torch.zeros(3, 20, device=DEV)
    for _ in range(8):
        assert torch.all(drv.step(z, z) == 0)
    cmd = torch.zeros(3, 5, 20, device=DEV)
    cmd[:, 4] = 0.1
    drv.set_host(cmd)
    for _ in range(3):
        drv.step(z, z)
    state = drv.get_state()
    a = torch.stack([drv.step(z, z) for _ in range(12)])
    drv.set_state(state)
    b = torch.stack([drv.step(z, z) for _ in range(12)])
    assert torch.equal(a, b)
    # a time shift rebases every clock: the same future, on the new clock
    drv.set_state(state)
    drv.shift_time(-0.5)
    c = torch.stack([drv.step(z, z) for _ in range(12)])
    assert torch.equal(a, c)


def test_line_capacity_is_checked():
    with pytest.raises(ValueError):
        TorchWujiDrive(WujiDriveCfg(cmd_latency=0.008), NAMES, TAU, 0.00025, 2, DEV)
