"""TaxelScan Rev3 readout: a piezoresistive taxel skin scanned by an RP2350's on-chip ADC, one frame per 1 ms.

Per board (one per patch by default, or one for the whole hand), every frame:

    force per taxel (physics rate) -> elastomer skin low-pass (first order, ts_bandwidth)
        -> sequential scan: taxel i is sampled at t_frame + offset + i * t_taxel,
           t_taxel = settle + oversample / adc_rate (linear interpolation between physics steps)
        -> divider: x = gain_i * F / (F + f_half)  (a force-sensitive resistor against a fixed resistor)
        -> 12-bit SAR ADC: counts = round(x * 4095 + offset_i + noise), noise from the ENOB, /sqrt(oversample)
        -> on-board calibration back to newtons with the nominal curve: F = f_half * x / (1 - x)
        -> frame sent when the scan ends; visible to the host ts_latency later (USB full-speed polling)

A sample's `t_sample` is the frame's start; each taxel was in fact sampled up to one scan length later. The
per-taxel gain and offset errors are what calibration and taring leave behind; they are drawn once per sensor,
like a physical skin's, and kept across resets.
"""

from __future__ import annotations

import numpy as np

from tactile_sim.sensors.base import RateLimitedSensor, SensorSample, SensorSpec


def taxel_time(s) -> float:
    """Time one taxel takes on a TaxelScan board (s): ts_taxel_time if set (a measured fast-profile figure), else
    mux settling plus its conversions."""
    if getattr(s, "ts_taxel_time", None) is not None:
        return s.ts_taxel_time
    return s.ts_settle + s.ts_oversample / s.ts_adc_rate


class TaxelScanSensor(RateLimitedSensor):
    def __init__(self, name: str, n: int, physics_dt: float, read_fn, rng, s, offset: float = 0.0):
        spec = SensorSpec(name, s.ts_rate, n, latency_s=s.ts_latency, saturation=(0.0, np.inf), decimation="zoh")
        self.s = s
        self.t_tax = taxel_time(s)
        self.offset = offset
        if offset + n * self.t_tax > 1.0 / s.ts_rate + 1e-12:
            raise ValueError(f"{name}: scanning {n} taxels takes {1e3 * (offset + n * self.t_tax):.3f} ms, longer "
                             f"than the {1e3 / s.ts_rate:.3f} ms frame")
        self.a = 1.0 - np.exp(-2 * np.pi * s.ts_bandwidth * physics_dt)
        self.full = 2**s.ts_adc_bits - 1
        rng = rng or np.random.default_rng()
        self.gain = 1.0 + s.ts_gain_mismatch * rng.normal(size=n)
        self.offs = s.ts_offset_lsb * rng.normal(size=n)
        self.noise_lsb = 2.0 ** (s.ts_adc_bits - s.ts_enob) / np.sqrt(12.0) / np.sqrt(s.ts_oversample)
        # an unloaded taxel reads below this (N): three sigma of residual offset plus ADC noise near zero force
        self.floor = s.ts_f_half * 3.0 * np.hypot(s.ts_offset_lsb, self.noise_lsb) / self.full
        super().__init__(spec, physics_dt, read_fn, rng)

    def reset(self, t0: float = 0.0) -> None:
        super().reset(t0)
        self.xf = None
        self.prev = None  # (t, filtered force) at the previous physics step
        self.frame_t = t0 + self.offset
        self.idx = 0
        self.buf = np.zeros(self.spec.dim)
        self.frames: list[tuple[float, float, np.ndarray]] = []  # (t_frame, t_done, force)
        self.counts = np.zeros(self.spec.dim, dtype=int)

    def step(self, t: float) -> SensorSample | None:
        raw = np.asarray(self.read_fn(), dtype=float).reshape(self.spec.dim)
        self.xf = raw.copy() if self.xf is None else self.xf + self.a * (raw - self.xf)
        tp, xp = self.prev if self.prev is not None else (t - self.dt, self.xf)
        n = self.spec.dim
        while True:
            # taxels whose sampling instant falls in (tp, t]
            hi = min(n, int(np.floor((t - self.frame_t) / self.t_tax + 1e-9)) + 1)
            if hi > self.idx:
                i = np.arange(self.idx, hi)
                al = np.clip((self.frame_t + i * self.t_tax - tp) / (t - tp), 0.0, 1.0)
                self.buf[i] = xp[i] + al * (self.xf[i] - xp[i])
                self.idx = hi
            if self.idx < n:
                break
            t0 = self.frame_t - self.offset
            self.frames.append((t0, self._avail_time(t0, self.frame_t + n * self.t_tax), self.buf.copy()))
            self.frame_t += self.period
            self.idx = 0
        self.prev = (t, self.xf)
        emitted = None
        while self.frames and self.frames[0][1] <= t + 0.5 * self.dt:
            t0, t_avail, f = self.frames.pop(0)
            s = self._emit_frame(t0, t_avail, f)
            emitted = s if s is not None else emitted
        return emitted

    def _avail_time(self, t0: float, t_done: float) -> float:
        """When the host has the frame whose scan started at t0 (SYNC) and ended at t_done."""
        return t_done + self.spec.latency_s

    def adc(self, f: np.ndarray) -> np.ndarray:
        """Counts the board reads for taxel forces `f` (N)."""
        f = np.maximum(f, 0.0)
        x = self.gain * f / (f + self.s.ts_f_half)
        c = x * self.full + self.offs + self.rng.normal(size=f.shape) * self.noise_lsb * self.noise_scale
        return np.clip(np.round(c), 0, self.full).astype(int)

    def calibrate(self, counts: np.ndarray) -> np.ndarray:
        """Board firmware: counts -> N with the nominal divider curve."""
        x = np.minimum(counts / self.full, 0.999)
        return self.s.ts_f_half * x / (1.0 - x)

    def _emit_frame(self, t0: float, t_avail: float, f: np.ndarray) -> SensorSample | None:
        self.counts = self.adc(f)
        v = self.calibrate(self.counts)
        s = SensorSample(t0, t_avail, v, self.seq)
        self.seq += 1
        self._latest = s
        if self.keep_history:
            self.t_sample.append(s.t_sample)
            self.t_avail.append(s.t_avail)
            self.values.append(v)
        return s


class TaxelScanRev3Sensor(TaxelScanSensor):
    """One patch on a fast-profile board (ts_model "rev3", docs/taxelscan_fast_profile.md).

    The patch is scanned `offset` after its board's SYNC edge, behind the board's earlier patches. Its frame is
    polled in the next period, in its board's slot (`slot`: s after that SYNC edge), and reaches the host a uniform
    ts_bridge_latency later, in order; with probability ts_drop_prob it never does. The board firmware converts its
    counts to millinewtons and sends mu-law codes plus the patch's six features (sensors.taxel_codec). The value the
    host sees is the codes decoded back to newtons; `features`, `codes` and `sync_id` hold the rest of the frame."""

    def __init__(self, name: str, grid: tuple[int, int], physics_dt: float, read_fn, rng, s, offset: float,
                 slot: float, board: int):
        from tactile_sim.sensors.taxel_codec import CodecSpec

        self.grid = (int(grid[0]), int(grid[1]))
        self.slot = slot
        self.board = board
        self.codec = CodecSpec(bits=s.ts_code_bits, mu=s.ts_mu, f_max=s.ts_f_max, contact_mn=s.ts_contact_mn)
        super().__init__(name, self.grid[0] * self.grid[1], physics_dt, read_fn, rng, s, offset=offset)

    def reset(self, t0: float = 0.0) -> None:
        super().reset(t0)
        self.frame_k = 0  # SYNC pulses since reset
        self.prev_sum = 0  # the board's force-sum feature of its previous frame (for the rate feature)
        self.last_avail = -np.inf
        self.features = np.zeros(6, dtype=np.int64)
        self.codes = np.zeros(self.spec.dim, dtype=np.uint8)
        self.sync_id = -1
        self.feature_history: list[np.ndarray] = []
        self.code_history: list[np.ndarray] = []

    def _avail_time(self, t0: float, t_done: float) -> float:
        lo, hi = self.s.ts_bridge_latency
        bridge = self.rng.uniform(lo, hi) if hi > lo else lo
        t = max(t0 + self.period + self.slot + bridge, self.last_avail)
        self.last_avail = t
        return t

    def _emit_frame(self, t0: float, t_avail: float, f: np.ndarray) -> SensorSample | None:
        from tactile_sim.sensors.taxel_codec import patch_features, to_mn

        k = self.frame_k
        self.frame_k += 1
        self.counts = self.adc(f)
        f_mn = to_mn(self.calibrate(self.counts))
        codes = self.codec.encode(f_mn)
        feats = patch_features(f_mn, self.grid, self.prev_sum, self.codec.contact_mn)
        self.prev_sum = int(feats[0])  # the board computes the rate whether or not the host gets the frame
        if self.s.ts_drop_prob > 0 and self.rng.uniform() < self.s.ts_drop_prob:
            return None
        v = self.codec.decode(codes)
        smp = SensorSample(t0, t_avail, v, self.seq)
        self.seq += 1
        self._latest = smp
        self.codes, self.features, self.sync_id = codes, feats, k & 0xFFFF
        if self.keep_history:
            self.t_sample.append(smp.t_sample)
            self.t_avail.append(smp.t_avail)
            self.values.append(v)
            self.feature_history.append(feats)
            self.code_history.append(codes)
        return smp


def board_layout(s, patch_names: list[str], grids: list[tuple[int, int]]) -> list[list[int]]:
    """Patch indices per board for the fast profile: every patch on exactly one board, at most ts_max_per_board
    taxels a board."""
    index = {n: k for k, n in enumerate(patch_names)}
    boards, seen = [], set()
    for b, names in enumerate(s.ts_board_map):
        ks = []
        for n in names:
            if n not in index:
                raise ValueError(f"ts_board_map board {b}: unknown patch {n!r} (patches: {patch_names})")
            if n in seen:
                raise ValueError(f"ts_board_map: patch {n!r} is on two boards")
            seen.add(n)
            ks.append(index[n])
        n_tax = sum(grids[k][0] * grids[k][1] for k in ks)
        if n_tax > s.ts_max_per_board:
            raise ValueError(f"ts_board_map board {b} has {n_tax} taxels, more than ts_max_per_board "
                             f"{s.ts_max_per_board}")
        boards.append(ks)
    missing = set(patch_names) - seen
    if missing:
        raise ValueError(f"ts_board_map leaves patches unread: {sorted(missing)}")
    return boards


def poll_slots(s, boards: list[list[int]], grids: list[tuple[int, int]]) -> list[float]:
    """When each board's frame has crossed the bus, after the SYNC edge of the period it is polled in: the polls
    run in board order from ts_poll_lead, each costing a turnaround plus its frame's bytes at 10 bits per byte."""
    from tactile_sim.sensors.taxel_codec import FEATURE_BYTES, HEADER_BYTES

    t, out = s.ts_poll_lead, []
    for ks in boards:
        n = sum(grids[k][0] * grids[k][1] for k in ks)
        map_bytes = n if s.ts_code_bits == 8 else (6 * n + 7) // 8
        nbytes = HEADER_BYTES + map_bytes + FEATURE_BYTES * len(ks) + 2
        t += s.ts_turnaround + nbytes * 10.0 / s.ts_bus_baud
        out.append(t)
    if out and out[-1] > 1.0 / s.ts_rate:
        raise ValueError(f"the polls take {1e3 * out[-1]:.3f} ms, longer than the {1e3 / s.ts_rate:.3f} ms period")
    return out


def make_taxelscan(world, cfg, rng) -> list[TaxelScanSensor]:
    """One sensor per conforming patch, in patch order. ts_model "rev2": `ts_boards="hand"` makes one board scan
    the patches back to back, so each patch's scan starts where the previous one's ended. ts_model "rev3": the
    fast-profile boards of ts_board_map."""
    s = cfg.sensors
    hand = world.hand
    names = [p.name for p in hand.patches]
    grids = [tuple(hand.grid(k)) for k in range(len(names))]
    if s.ts_model == "rev3":
        boards = board_layout(s, names, grids)
        slots = poll_slots(s, boards, grids)
        sensors: dict[int, TaxelScanSensor] = {}
        for b, ks in enumerate(boards):
            off = 0.0
            for k in ks:
                sensors[k] = TaxelScanRev3Sensor(hand.pressure_names[k], grids[k], world.dt,
                                                 (lambda k=k: world.pad_taxels(k)), rng, s, offset=off,
                                                 slot=slots[b], board=b)
                off += grids[k][0] * grids[k][1] * taxel_time(s)
        return [sensors[k] for k in range(len(names))]
    if s.ts_model != "rev2":
        raise ValueError(f"unknown ts_model {s.ts_model!r} (expected 'rev2' or 'rev3')")
    out, off = [], 0.0
    for k, name in enumerate(hand.pressure_names):
        n = int(np.prod(hand.grid(k)))
        out.append(TaxelScanSensor(name, n, world.dt, (lambda k=k: world.pad_taxels(k)), rng, s,
                                   offset=off if s.ts_boards == "hand" else 0.0))
        off += n * taxel_time(s)
    if s.ts_boards not in ("hand", "per_patch"):
        raise ValueError(f"unknown ts_boards {s.ts_boards!r}")
    return out
