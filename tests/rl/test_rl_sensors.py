"""The GPU sensors (tactile_rl.sensors) against the testbed's on identical input streams and identical random draws:
the TaxelScan codec and patch features bit-exact on the frozen golden frames, the WUJI readback (hand_q /
hand_current) sample for sample, and the rev3 readout (skin low-pass, scan, ADC, calibration, codes, features, frame
timing, drops) frame for frame against TaxelScanRev3Sensor, every patch, several parameter sets."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.rl]  # torch, mjlab and CUDA (.venv-rl) or skipped
torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from tactile_rl.sensors.readback import TorchReadback  # noqa: E402
from tactile_rl.sensors.taxel_codec import TorchCodec, patch_features  # noqa: E402
from tactile_rl.sensors.taxelscan import TorchTaxelScan  # noqa: E402
from tactile_sim.config import SensorsCfg  # noqa: E402
from tactile_sim.sensors.base import RateLimitedSensor, SensorSpec  # noqa: E402
from tactile_sim.sensors.taxel_codec import CODEC_VERSION, CodecSpec, decode_frame, fast_profile_boards  # noqa: E402
from tactile_sim.sensors.taxelscan import TaxelScanRev3Sensor, board_layout, poll_slots, taxel_time  # noqa: E402
from tactile_sim.sim.training_parts import SkinPatch  # noqa: E402

DEV = "cuda"
GOLDEN = Path(__file__).parents[1] / "data" / f"taxel_codec_v{CODEC_VERSION}.json"
NAMES = ["palm"] + [f"{f}_{seg}" for f in ("thumb", "index", "middle", "ring", "pinky") for seg in ("middle", "distal")]
GRIDS = [(8, 16)] + [(8, 4)] * 10


class RecordingRng:
    """A numpy Generator that logs its draws by kind, so the GPU can be fed the same ones."""

    def __init__(self, seed: int):
        self.g = np.random.default_rng(seed)
        self.log: dict[str, list] = {"normal": [], "uniform2": [], "uniform0": []}

    def normal(self, loc=0.0, scale=1.0, size=None):
        x = self.g.normal(loc, scale, size)
        self.log["normal"].append(np.array(x, dtype=float))
        return x

    def uniform(self, low=None, high=None, size=None):
        if low is None:
            x = self.g.uniform()
            self.log["uniform0"].append(float(x))
        else:
            x = self.g.uniform(low, high, size)
            self.log["uniform2"].append(float(x))
        return x


@pytest.mark.parametrize("bits", [8, 6])
def test_codec_and_features_are_bit_exact_on_the_golden_frames(bits):
    spec = CodecSpec(bits=bits)
    codec = TorchCodec(spec, DEV)
    frozen = json.loads(GOLDEN.read_text())["u8" if bits == 8 else "u6"]
    boards = {b.addr: b for b in fast_profile_boards()}
    prev = {}
    for g in frozen:
        b = boards[g["board"]]
        ref = decode_frame(spec, b, bytes.fromhex(g["frame_hex"]))
        f = torch.as_tensor(g["f_mn"], dtype=torch.long, device=DEV)
        assert np.array_equal(codec.encode(f).cpu().numpy(), ref["codes"]), (g["board"], g["case"])
        p = prev.get(b.addr, np.zeros(len(b.patches), dtype=np.int64))
        k, feats = 0, []
        for i, (_, (rows, cols)) in enumerate(b.patches):
            n = rows * cols
            feats.append(patch_features(f[k:k + n], rows, cols, torch.tensor(int(p[i]), device=DEV),
                                        spec.contact_mn).cpu().numpy())
            k += n
        feats = np.array(feats)
        assert np.array_equal(feats, ref["features"]), (g["board"], g["case"])
        prev[b.addr] = feats[:, 0]


def test_codec_matches_the_reference_over_the_whole_range():
    from tactile_rl.sensors.taxel_codec import to_mn as to_mn_t
    from tactile_sim.sensors.taxel_codec import to_mn

    spec = CodecSpec()
    codec = TorchCodec(spec, DEV)
    f = np.concatenate([spec.thresholds_mn.astype(np.int64) + d for d in (-1, 0, 1)] + [np.arange(0, 70000, 7)])
    assert np.array_equal(codec.encode(torch.as_tensor(f, device=DEV)).cpu().numpy(), spec.encode(f))
    x = np.random.default_rng(0).uniform(-1.0, 80.0, 100_000)
    x[:50] = np.arange(50) * 0.0005  # exact half-mN values: round half to even, as numpy
    assert np.array_equal(to_mn_t(torch.as_tensor(x, device=DEV)).cpu().numpy(), to_mn(x))
    codes = np.arange(spec.levels)
    assert np.array_equal(codec.decode(torch.as_tensor(codes, device=DEV)).cpu().numpy(), spec.decode(codes))


# ------------------------------------------------------------------------------------------------ readback


READBACK_ENVS = [  # (rate, latency, noise, quant)
    (1000.0, 0.0005, 0.0, 2 * math.pi / 2**14),
    (1000.0, 0.0017, 0.01, 0.001),
    (500.0, 0.0, 0.03, 0.001),
    (1000.0, 0.003, 0.02, 0.0),
]


@pytest.mark.parametrize("dt", [0.00025, 0.000125])
def test_readback_matches_the_rate_limited_sensor(dt):
    n_env, dim, steps = len(READBACK_ENVS), 20, 400
    rng = np.random.default_rng(1)
    raw = np.cumsum(rng.normal(0, 0.01, (steps, n_env, dim)), axis=0)
    cpu, rngs = [], []
    for e, (rate, lat, noise, q) in enumerate(READBACK_ENVS):
        r = RecordingRng(10 + e)
        feed = {"v": np.zeros(dim)}
        cpu.append(RateLimitedSensor(SensorSpec("x", rate, dim, latency_s=lat, noise_std=noise, quant_step=q or None,
                                                decimation="zoh"), dt, (lambda f=feed: f["v"]), r))
        cpu[-1].noise_scale = 1.0 + 0.5 * e
        rngs.append((r, feed))
    gpu = TorchReadback(n_env, dim, dt, device=DEV)
    gpu.set_params(every=torch.tensor([round(1 / (r * dt)) for r, *_ in READBACK_ENVS]),
                   latency=torch.tensor([x[1] for x in READBACK_ENVS], dtype=torch.float64),
                   noise_std=torch.tensor([[x[2]] * dim for x in READBACK_ENVS], dtype=torch.float64),
                   noise_scale=torch.tensor([1.0 + 0.5 * e for e in range(n_env)], dtype=torch.float64),
                   quant=torch.tensor([x[3] for x in READBACK_ENVS], dtype=torch.float64))
    gpu.reset()
    ref = []
    t = 0.0
    for k in range(steps):
        t += dt
        row = []
        for e, s in enumerate(cpu):
            rngs[e][1]["v"] = raw[k, e]
            s.step(t)
            smp = s.latest()
            row.append((smp.value.copy(), smp.t_sample, smp.t_avail, smp.seq))
        ref.append((t, row))
    # the noise each emission drew, by sequence number (the first normal draw is the sensor's zero bias)
    draws = [np.array(r.log["normal"][1:]) if len(r.log["normal"]) > 1 else np.zeros((1, dim)) for r, _ in rngs]
    width = max(len(d) for d in draws)
    table = np.zeros((n_env, width + 1, dim))
    for e, d in enumerate(draws):
        table[e, :len(d)] = d
    tab = torch.as_tensor(table, device=DEV)
    gpu.noise_fn = lambda seq, mask: tab[torch.arange(n_env, device=DEV), seq.clamp(0, width)]
    for k, (t, row) in enumerate(ref):
        gpu.step(torch.full((n_env,), t, dtype=torch.float64, device=DEV),
                 torch.as_tensor(raw[k], dtype=torch.float64, device=DEV))
        for e, (v, ts, ta, seq) in enumerate(row):
            assert int(gpu.seq[e]) == seq, (k, e)
            if seq >= 0:
                assert float(gpu.t_sample[e]) == pytest.approx(ts, abs=1e-12)
                assert float(gpu.t_avail[e]) == pytest.approx(ta, abs=1e-12)
                assert gpu.value[e].cpu().numpy() == pytest.approx(v, abs=1e-12), (k, e)


# ------------------------------------------------------------------------------------------------ TaxelScan rev3


def skin_layout(s: SensorsCfg) -> list[SkinPatch]:
    """The default 448-taxel hand's patches with their boards, scan offsets and poll slots (taxelscan.make_taxelscan);
    the readout does not need the taxels' positions."""
    boards = board_layout(s, NAMES, GRIDS)
    slots = poll_slots(s, boards, GRIDS)
    out = {}
    for b, ks in enumerate(boards):
        off = 0.0
        for k in ks:
            n = GRIDS[k][0] * GRIDS[k][1]
            out[k] = SkinPatch(NAMES[k], "body", (), GRIDS[k], np.zeros((n, 3)), np.zeros((n, 3)), 0.005, b, off,
                               slots[b])
            off += n * taxel_time(s)
    return [out[k] for k in range(len(NAMES))]


SCAN_ENVS = [  # SensorsCfg overrides per env
    {},
    {"ts_bandwidth": 150.0, "ts_f_half": 3.0, "ts_gain_mismatch": 0.15, "ts_offset_lsb": 6.0},
    {"ts_bandwidth": 500.0, "ts_f_half": 8.0, "ts_bridge_latency": (0.000125, 0.0005), "ts_drop_prob": 0.2},
    {"ts_drop_prob": 0.05, "ts_bridge_latency": (0.0003, 0.0003), "ts_enob": 7.0},
]


def forces(steps: int, n_env: int, n_tax: int, dt: float, seed: int = 0) -> np.ndarray:
    """Smooth loads with contacts that come and go: (steps, env, taxel) N."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, steps + 1)[:, None, None] * dt
    f0 = rng.uniform(0.0, 25.0, (1, n_env, n_tax)) * (rng.uniform(size=(1, n_env, n_tax)) < 0.4)
    w = rng.uniform(20.0, 400.0, (1, n_env, n_tax))
    ph = rng.uniform(0, 2 * np.pi, (1, n_env, n_tax))
    return np.maximum(f0 * (0.6 + 0.6 * np.sin(w * t + ph)), 0.0)


@pytest.mark.parametrize("dt", [0.00025, 0.000125])
def test_taxelscan_matches_the_rev3_sensor_frame_for_frame(dt):
    base = SensorsCfg(ts_model="rev3", ts_taxel_time=7.8e-6)
    n_env, steps = len(SCAN_ENVS), int(round(0.03 / dt))
    skin = skin_layout(base)
    n_tax = sum(g[0] * g[1] for g in GRIDS)
    load = forces(steps, n_env, n_tax, dt)
    starts = np.concatenate([[0], np.cumsum([g[0] * g[1] for g in GRIDS])[:-1]])
    cpu = []  # per env: list of (sensor, rng, feed)
    for e, over in enumerate(SCAN_ENVS):
        s = base.__class__(**{**base.__dict__, **over})
        row = []
        for k, p in enumerate(skin):
            r = RecordingRng(100 * e + k)
            feed = {"f": np.zeros(GRIDS[k][0] * GRIDS[k][1])}
            sen = TaxelScanRev3Sensor(f"pressure_{p.name}", p.grid, dt, (lambda f=feed: f["f"]), r, s, offset=p.offset,
                                      slot=p.slot, board=p.board)
            sen.noise_scale = 1.0 + 0.25 * e
            row.append((sen, r, feed))
        cpu.append(row)
    gpu = TorchTaxelScan(skin, base, dt, n_env, DEV)
    for e, over in enumerate(SCAN_ENVS):
        s = base.__class__(**{**base.__dict__, **over})
        ids = torch.tensor([e], device=DEV)
        gpu.set_params(ids, f_half=s.ts_f_half, enob=s.ts_enob, bandwidth=s.ts_bandwidth,
                       bridge_latency=s.ts_bridge_latency, drop_prob=s.ts_drop_prob, noise_scale=1.0 + 0.25 * e)
        gpu.gain[e] = torch.as_tensor(np.concatenate([row[0].gain for row in cpu[e]]), device=DEV)
        gpu.offs[e] = torch.as_tensor(np.concatenate([row[0].offs for row in cpu[e]]), device=DEV)
    gpu.reset()
    # the testbed, step by step; then its draws, by patch and frame, for the GPU
    ref = []
    t = 0.0
    for k in range(steps):
        t += dt
        out = []
        for e in range(n_env):
            for j, (sen, _, feed) in enumerate(cpu[e]):
                feed["f"] = load[k, e, starts[j]:starts[j] + len(feed["f"])]
                sen.step(t)
                smp = sen.latest()
                out.append((smp.seq, smp.t_sample, smp.t_avail, sen.codes.copy(), sen.features.copy(), sen.sync_id))
        ref.append((t, out))
    kmax = 64
    bridge = np.zeros((n_env, len(skin), kmax))
    drop = np.ones((n_env, len(skin), kmax))
    noise = np.zeros((n_env, len(skin), kmax, 128))
    for e in range(n_env):
        for j, (sen, r, _) in enumerate(cpu[e]):
            b = r.log["uniform2"]
            bridge[e, j, :len(b)] = b
            if not b:  # lo == hi: no draw
                bridge[e, j] = sen.s.ts_bridge_latency[0]
            u0 = r.log["uniform0"]
            drop[e, j, :len(u0)] = u0
            nz = r.log["normal"][3:]  # after the gain, offset and (zero) bias draws
            for f, z in enumerate(nz):
                noise[e, j, f, :len(z)] = z
    tb, td, tn = (torch.as_tensor(x, device=DEV) for x in (bridge, drop, noise))
    env = torch.arange(n_env, device=DEV)[:, None]
    pat = torch.arange(len(skin), device=DEV)[None]
    gpu.bridge_fn = lambda mask, k: tb[env, pat, k.clamp(0, kmax - 1)]
    gpu.drop_fn = lambda mask, k: td[env, pat, k.clamp(0, kmax - 1)]
    idx_in = torch.as_tensor(np.concatenate([np.arange(g[0] * g[1]) for g in GRIDS]), device=DEV)

    def noise_fn(mask, k):
        kk = k.clamp(0, kmax - 1)[:, gpu.patch]  # (N, T)
        return tn[env, gpu.patch[None], kk, idx_in[None]]

    gpu.noise_fn = noise_fn
    frames = 0
    for k, (t, out) in enumerate(ref):
        gpu.step(torch.full((n_env,), t, dtype=torch.float64, device=DEV),
                 torch.as_tensor(load[k], dtype=torch.float64, device=DEV))
        seq, ts, ta = gpu.seq.cpu().numpy(), gpu.t_sample.cpu().numpy(), gpu.t_avail.cpu().numpy()
        codes, feats, sync = gpu.codes.cpu().numpy(), gpu.features.cpu().numpy(), gpu.sync_id.cpu().numpy()
        for e in range(n_env):
            for j in range(len(skin)):
                c_seq, c_ts, c_ta, c_codes, c_feat, c_sync = out[e * len(skin) + j]
                assert seq[e, j] == c_seq, (k, e, NAMES[j])
                if c_seq < 0:
                    continue
                frames += 1
                assert ts[e, j] == pytest.approx(c_ts, abs=1e-12) and ta[e, j] == pytest.approx(c_ta, abs=1e-12)
                assert sync[e, j] == c_sync
                assert np.array_equal(codes[e, starts[j]:starts[j] + len(c_codes)], c_codes), (k, e, NAMES[j])
                assert np.array_equal(feats[e, j], c_feat), (k, e, NAMES[j])
    assert frames > 1000
    assert int(gpu.n_dropped[2].sum()) > 0 and int(gpu.overflow.sum()) == 0


# ------------------------------------------------------------------------------------------------ forecast


class DrawLog:
    """A Generator stand-in for ForecastModel that logs each standard draw in order (uniform: u, normal: z)."""

    def __init__(self, seed: int):
        self.g = np.random.default_rng(seed)
        self.log: list[tuple[str, float]] = []

    def uniform(self, low=0.0, high=1.0):
        u = float(self.g.random())
        self.log.append(("u", u))
        return low + (high - low) * u

    def normal(self, loc=0.0, scale=1.0):
        z = float(self.g.standard_normal())
        self.log.append(("n", z))
        return loc + scale * z


FORECAST_ENVS = [
    {"mask_prob": 0.0},
    {"mask_prob": 0.0, "dropout": 0.3, "bias_sigma": 0.02, "jitter_sigma": 0.01, "class_confusion": 0.2},
    {"mask_prob": 0.0, "horizon_err": 0.2, "class_confusion": 0.6, "jitter_sigma": 0.0},
    {"mask_prob": 0.0, "freq_err": 0.2, "amp_err": 0.4, "latency": 0.05, "update_rate": 5.0},
    {"mask_prob": 1.0},
    {"mask_prob": 0.0, "latency": 0.0, "update_rate": 10.0, "dropout": 0.0},
]


@pytest.mark.parametrize("tick", [0.001, 0.004])
def test_forecast_matches_the_forecast_model(tick):
    from tactile_rl.mdp.forecast import TorchForecastModel
    from tactile_sim.behaviors.forecast import ForecastModel, ForecastTruth
    from tactile_sim.config import ForecastCfg
    from tactile_sim.control.obs_spec import REGIMES

    n = len(FORECAST_ENVS)
    logs = [DrawLog(e) for e in range(n)]
    cpu = [ForecastModel(ForecastCfg(**over), logs[e]) for e, over in enumerate(FORECAST_ENVS)]
    steps = int(round(1.2 / tick))

    def truth(e, t):  # events come and go; regimes, frequencies and amplitudes change
        k = int(t * 3.0 + e) % 4
        if k == 3:
            return ForecastTruth(False)
        return ForecastTruth(True, t_event=0.4 * np.floor(t / 0.4) + 0.3 + 0.01 * e, regime=REGIMES[(k + e) % 5],
                             f_dom=10.0 * (k + 1), amp=50.0 + 20.0 * k)

    ref = []
    for i in range(steps):
        t = i * tick
        ref.append([cpu[e](t, truth(e, t)).copy() for e in range(n)])
    # the same draws, in the same order, for the torch model
    queues = [list(lg.log) for lg in logs]
    gpu = TorchForecastModel(n, None, DEV)
    for k in ("mask_prob", "dropout", "bias_sigma", "jitter_sigma", "horizon_err", "class_confusion", "freq_err",
              "amp_err", "update_rate", "latency"):
        gpu.set_params(**{k: torch.tensor([getattr(c.cfg, k) for c in cpu], dtype=torch.float64)})
    kinds = {"mask": "u", "bias": "n", "confusion": "u", "wrong": "u", "dropout": "u", "freq": "n", "amp": "n",
             "jitter": "n"}

    def draw_fn(name, mask):
        out = torch.zeros(n, dtype=torch.float64)
        for e in torch.nonzero(mask.cpu()).flatten().tolist():
            kind, v = queues[e].pop(0)
            assert kind == kinds[name], (e, name)
            out[e] = v
        return out.to(DEV)

    gpu.draw_fn = draw_fn
    gpu.reset()
    for i in range(steps):
        t = i * tick
        tr = [truth(e, t) for e in range(n)]
        out = gpu(torch.full((n,), t, dtype=torch.float64, device=DEV),
                  torch.tensor([x.valid for x in tr], device=DEV),
                  torch.tensor([x.t_event for x in tr], dtype=torch.float64, device=DEV),
                  torch.tensor([REGIMES.index(x.regime) for x in tr], device=DEV),
                  torch.tensor([x.f_dom for x in tr], dtype=torch.float64, device=DEV),
                  torch.tensor([x.amp for x in tr], dtype=torch.float64, device=DEV)).cpu().numpy()
        for e in range(n):
            assert out[e] == pytest.approx(ref[i][e], abs=1e-12), (i, e)
    assert all(not q for q in queues)  # every draw the testbed made was used, in order
    shown = np.array([[r[e][0] for e in range(n)] for r in ref])
    assert shown[:, 0].any() and not shown[:, 4].any()  # forecasts were seen; the masked env saw none


# ------------------------------------------------------------------------------------------------ student obs


@pytest.mark.parametrize("stack, mask", [(1, ()), (4, ()), (10, ("hand_current", "forecast")),
                                         (2, ("taxel_codes", "patch_features", "board_status"))])
def test_student_observation_matches_residual_l2(stack, mask):
    import types
    from collections import deque

    from tactile_rl.mdp.obs_builder import StudentObs
    from tactile_sim.control.obs_spec import ACTION_DIM, BOARDS, TASKS, obs_slices
    from tactile_sim.control.residual_l2 import ResidualHandL2
    from tactile_sim.sensors.base import SensorSample

    n_env, steps, rng = 3, 40, np.random.default_rng(stack)
    board_first = [0, 1, 5, 9]
    codec = CodecSpec()
    i_rated = rng.uniform(0.8, 1.2, (n_env, 20))
    tau_rated = rng.uniform(0.2, 2.0, 20)
    fakes = []
    for e in range(n_env):
        patches = [types.SimpleNamespace(codes=np.zeros(g[0] * g[1], np.uint8), features=np.zeros(6, np.int64),
                                         grid=g, codec=codec, sample=SensorSample(0.0, 0.0, np.zeros(1), -1))
                   for g in GRIDS]
        for p in patches:
            p.latest = (lambda p=p: p.sample)
        q = types.SimpleNamespace(sample=SensorSample(0.0, 0.0, np.zeros(20), -1))
        cur = types.SimpleNamespace(sample=SensorSample(0.0, 0.0, np.zeros(20), -1))
        q.latest, cur.latest = (lambda s=q: s.sample), (lambda s=cur: s.sample)
        fakes.append(types.SimpleNamespace(
            patch_sensors=patches, board_sensors=[patches[b] for b in board_first], q_sensor=q, i_sensor=cur,
            i_rated=i_rated[e], last_seq=[-1] * BOARDS, stack=stack, fast_dim=562,
            ring=deque([np.zeros(562)] * stack, maxlen=stack), last_a=np.zeros(ACTION_DIM),
            jp=types.SimpleNamespace(tau_rated=tau_rated),
            task_vec=np.eye(len(TASKS))[0], masked=[obs_slices(stack)[b] for b in mask]))
        fakes[-1].fast_sample = (lambda t, f=fakes[-1]: ResidualHandL2.fast_sample(f, t))
    gpu = StudentObs(n_env, stack, GRIDS, codec.levels, mask, "hammer", DEV, tau_rated=tau_rated)
    starts = np.concatenate([[0], np.cumsum([g[0] * g[1] for g in GRIDS])[:-1]])
    seq = np.full((n_env, len(GRIDS)), -1)
    for k in range(steps):
        t = k * 1e-3
        codes = rng.integers(0, 256, (n_env, 448)).astype(np.uint8)
        feats = rng.integers(-5, 3000, (n_env, len(GRIDS), 6))
        feats[:, :, 0] *= rng.uniform(size=(n_env, len(GRIDS))) < 0.7  # unloaded patches
        seq = seq + rng.integers(0, 2, seq.shape) * (k > 2)
        avail = t - rng.uniform(0.0, 0.006, seq.shape)
        qv, cv = rng.normal(0, 1, (n_env, 20)), rng.normal(0, 1, (n_env, 20))
        last_a, base = rng.uniform(-1, 1, (n_env, ACTION_DIM)), rng.normal(0, 1, (n_env, 5, 20))
        look, fc = rng.normal(0, 0.1, (n_env, 81)), rng.normal(0, 1, (n_env, 13))
        look_arm = look.reshape(n_env, 3, 27)[:, :, 20:].reshape(n_env, 21)  # the GPU base's lookahead: the arm's
        for e, f in enumerate(fakes):
            for j, p in enumerate(f.patch_sensors):
                p.codes = codes[e, starts[j]:starts[j] + len(p.codes)]
                p.features = feats[e, j]
                p.sample = SensorSample(avail[e, j], avail[e, j], np.zeros(1), int(seq[e, j]))
            f.q_sensor.sample = SensorSample(t, t, qv[e], k)
            f.i_sensor.sample = SensorSample(t, t, cv[e], k)
            ResidualHandL2.sample(f, t)
            f.last_a = last_a[e]
        tt = torch.full((n_env,), t, dtype=torch.float64, device=DEV)
        bs = torch.as_tensor(seq[:, board_first], device=DEV)
        ba = torch.as_tensor(avail[:, board_first], device=DEV)
        row = gpu.sample(tt, torch.as_tensor(codes, device=DEV), torch.as_tensor(feats, device=DEV), bs, ba,
                         torch.as_tensor(qv, device=DEV), torch.as_tensor(cv / i_rated, device=DEV))
        gpu.push(row)
        out = gpu.observe(*(torch.as_tensor(x, device=DEV) for x in (last_a, base, look_arm, fc)),
                          torch.zeros(n_env, device=DEV)).cpu().numpy()
        for e, f in enumerate(fakes):
            ref = ResidualHandL2.observe(f, t, base[e], look[e], False, fc[e])
            assert out[e] == pytest.approx(ref, abs=2e-6, rel=1e-6), (k, e)


# ------------------------------------------------------------------------------------------------ skin load rays


def test_hull_rays_match_mujocos_ray_mesh():
    """The YCB hammer's handle slices: each taxel's ray (3 mm inside the skin, along its normal) against the hulls its
    patch pairs with, in the Warp kernel, against mj_rayMesh on the CPU model at the settled hold and with the hammer
    shifted and turned about it (some rays then start inside a hull)."""
    import mujoco

    from tactile_rl.sensors.taxel_load import BACK, TaxelLoadModel
    from tactile_sim.sim.training_parts import residual_training_config, training_parts

    parts = training_parts(residual_training_config())
    m = mujoco.MjModel.from_xml_string(parts.xml)
    d = mujoco.MjData(m)
    hj = m.joint(parts.hammer_joint)
    a = int(hj.qposadr[0])
    rng = np.random.default_rng(0)
    states = [parts.qpos_settled.copy()]
    for _ in range(3):
        q = parts.qpos_settled.copy()
        q[a:a + 3] += rng.normal(0, 0.004, 3)
        dq = rng.normal(0, 0.1, 3)
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, dq / np.linalg.norm(dq), np.linalg.norm(dq))
        mujoco.mju_mulQuat(q[a + 3:a + 7], quat, q[a + 3:a + 7].copy())
        states.append(q)
    n = len(states)
    lm = TaxelLoadModel(parts, m, n, DEV)
    assert lm.mode == "hull" and lm.G >= 10
    ray_ids = lm.ray_geoms.cpu().numpy()
    xp, xm, gp, gm = [], [], [], []
    for q in states:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        xp.append(d.xpos[lm.body.cpu().numpy()].copy())
        xm.append(d.xmat[lm.body.cpu().numpy()].reshape(-1, 3, 3).copy())
        gp.append(d.geom_xpos[ray_ids].copy())
        gm.append(d.geom_xmat[ray_ids].reshape(-1, 3, 3).copy())

    def t(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEV)

    P, K = lm.P, lm.K
    _, _, _, _, _, org, ray = lm._contacts(torch.zeros(n, P, K, device=DEV), torch.zeros(n, P, K, 3, device=DEV),
                                           torch.zeros(n, P, K, dtype=torch.bool, device=DEV), t(xp), t(xm))
    need = torch.ones(n, lm.T, dtype=torch.bool, device=DEV)
    gap = lm._hull_gaps(org, ray, need, t(gp), t(gm)).cpu().numpy()
    org, ray = org.cpu().numpy().astype(float), ray.cpu().numpy().astype(float)
    patch, pt = lm.patch.cpu().numpy(), lm.patch_tool.cpu().numpy()
    hits = 0
    for i, q in enumerate(states):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        for j in range(lm.T):
            best = np.inf
            for gi in np.flatnonzero(pt[patch[j]]):
                h = mujoco.mj_rayMesh(m, d, int(ray_ids[gi]), org[i, j], ray[i, j] / np.linalg.norm(ray[i, j]))
                if h >= 0:
                    best = min(best, h - BACK)
            if np.isfinite(best) or np.isfinite(gap[i, j]):
                assert np.isfinite(best) and np.isfinite(gap[i, j]), (i, j, best, gap[i, j])
                assert gap[i, j] == pytest.approx(best, abs=2e-5), (i, j)
                hits += 1
    assert hits > 50  # the palm and the pads see the handle
