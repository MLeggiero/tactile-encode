"""The TaxelScan fast profile (rev3) batched over environments: the mirror of tactile_sim.sensors.taxelscan
.TaxelScanRev3Sensor, one per patch, for all patches and environments at once.

Per patch, every physics step (after MuJoCo stepped, at time t; the previous step at tp):

    taxel load (N) -> skin: first-order low-pass xf += a (raw - xf), a = 1 - exp(-2 pi f_skin dt), started on the
        first value after a reset
    -> scan: taxel i of the patch is sampled at SYNC_k + offset + i t_tax (SYNC_k = k / ts_rate since the reset), by
       linear interpolation between the step before and the step after that instant
    -> when the patch's last taxel is sampled the frame is complete: divider x = gain f / (f + f_half), 12-bit ADC
       counts = clip(round(x full + offset + ENOB noise), 0, full), the firmware's calibration back to newtons with
       the nominal curve, integer mN, mu-law codes and the six patch features (sensors.taxel_codec, integer exact;
       the rate feature from the previous frame's sum, kept even when a frame is dropped)
    -> availability: max(SYNC_k + period + the board's poll slot + bridge latency ~ U(lo, hi), the previous frame's),
       dropped with probability drop_prob; the host's latest frame is the newest one available.

The testbed converts a frame when it reaches the host and draws the bridge latency when it completes; here the whole
frame is converted on completion (identical given the same draws, as the parity test feeds them). `bridge_fn`,
`noise_fn` and `drop_fn` replace the draws: f(mask (N, P) bool, frame (N, P) long) -> (N, P) seconds, (N, n_taxels)
standard normals, (N, P) uniforms.

Per-environment parameters (set_params): per-taxel gain and offset errors (drawn like a physical skin's), the
divider's f_half (the board calibrates with that nominal value; `f_half_mismatch` optionally spreads each taxel's own
divider about it, 0 by default as on the testbed), ENOB noise and its scale, the skin bandwidth,
the bridge latency range and the drop probability. Times are float64, the scan and the ADC float64 (as the testbed),
the codes uint8 and the features int64.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from tactile_rl.sensors.taxel_codec import TorchCodec, patch_features, to_mn
from tactile_sim.sensors.taxel_codec import CodecSpec
from tactile_sim.sensors.taxelscan import taxel_time

DrawFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class TorchTaxelScan:
    def __init__(self, skin, sensors_cfg, physics_dt: float, num_envs: int, device: str | torch.device = "cpu",
                 depth: int = 4, generator: torch.Generator | None = None):
        """`skin`: the patches (tactile_sim.sim.training_parts.SkinPatch, layout order); `sensors_cfg`: SensorsCfg."""
        s = sensors_cfg
        self.cfg = s
        self.dev = torch.device(device)
        self.dt = float(physics_dt)
        self.n_env = N = int(num_envs)
        self.D = int(depth)
        self.generator = generator
        self.period = 1.0 / s.ts_rate
        self.t_tax = taxel_time(s)
        self.full = 2**s.ts_adc_bits - 1
        self.codec = TorchCodec(CodecSpec(bits=s.ts_code_bits, mu=s.ts_mu, f_max=s.ts_f_max,
                                          contact_mn=s.ts_contact_mn), self.dev)
        self.P = P = len(skin)
        self.n_of = [p.grid[0] * p.grid[1] for p in skin]
        self.start = np.concatenate([[0], np.cumsum(self.n_of)[:-1]]).astype(int)
        self.T = T = int(sum(self.n_of))
        i_in = np.concatenate([np.arange(n) for n in self.n_of])
        patch = np.repeat(np.arange(P), self.n_of)
        off = np.array([p.offset for p in skin])
        for k, p in enumerate(skin):
            if p.offset + self.n_of[k] * self.t_tax > self.period + 1e-12:
                raise ValueError(f"{p.name}: the scan does not fit the {1e3 * self.period:.3f} ms frame")
        f64 = dict(dtype=torch.float64, device=self.dev)
        self.patch = torch.as_tensor(patch, dtype=torch.long, device=self.dev)  # (T,)
        self.s_off = torch.as_tensor(off[patch] + i_in * self.t_tax, **f64)  # (T,) sample offset after SYNC
        last = self.start + np.array(self.n_of) - 1
        self.last = torch.as_tensor(last, dtype=torch.long, device=self.dev)  # (P,) each patch's last taxel
        self.slot = torch.as_tensor([p.slot for p in skin], **f64)  # (P,)
        self.board = np.array([p.board for p in skin])
        self.board_first = [int(np.flatnonzero(self.board == b)[0]) for b in sorted(set(self.board.tolist()))]
        self._board_first = torch.as_tensor(self.board_first, dtype=torch.long, device=self.dev)
        self.grids = [tuple(p.grid) for p in skin]
        # feature groups: patches with the same grid share one batched computation
        self.groups = []
        for g in sorted(set(self.grids)):
            ks = [k for k in range(P) if self.grids[k] == g]
            idx = np.stack([np.arange(self.start[k], self.start[k] + self.n_of[k]) for k in ks])
            self.groups.append((g, torch.as_tensor(ks, device=self.dev),
                                torch.as_tensor(idx, dtype=torch.long, device=self.dev)))
        self.tol_k = 1e-9 * self.t_tax / self.period
        # per-env parameters
        self.gain = torch.ones(N, T, **f64)
        self.offs = torch.zeros(N, T, **f64)
        self.f_half = torch.full((N,), float(s.ts_f_half), **f64)
        self.f_half_taxel = torch.full((N, T), float(s.ts_f_half), **f64)  # each taxel's divider
        self.noise_lsb = torch.full((N,), 2.0 ** (s.ts_adc_bits - s.ts_enob) / math.sqrt(12.0) /
                                    math.sqrt(s.ts_oversample), **f64)
        self.noise_scale = torch.ones(N, **f64)
        self.a_skin = torch.full((N,), 1.0 - math.exp(-2 * math.pi * s.ts_bandwidth * self.dt), **f64)
        self.bridge_lo = torch.full((N,), float(s.ts_bridge_latency[0]), **f64)
        self.bridge_hi = torch.full((N,), float(s.ts_bridge_latency[1]), **f64)
        self.drop_prob = torch.full((N,), float(s.ts_drop_prob), **f64)
        self.alive = torch.ones(N, T, **f64)  # dead taxels read no load
        self.phase = torch.zeros(N, **f64)  # s: the boards' SYNC edges relative to the env's clock
        self.bridge_fn: DrawFn | None = None
        self.noise_fn: DrawFn | None = None
        self.drop_fn: DrawFn | None = None
        self._alloc()
        self.set_params(gain_mismatch=s.ts_gain_mismatch, offset_lsb=s.ts_offset_lsb)
        self.reset()

    # ------------------------------------------------------------------ state
    def _alloc(self) -> None:
        N, P, T, D, dev = self.n_env, self.P, self.T, self.D, self.dev
        f64, i64 = dict(dtype=torch.float64, device=dev), dict(dtype=torch.long, device=dev)
        self.xf = torch.zeros(N, T, **f64)
        self.t_prev = torch.zeros(N, **f64)
        self.fresh = torch.ones(N, dtype=torch.bool, device=dev)
        self.buf = torch.zeros(N, 2, T, **f64)
        self.done = torch.zeros(N, P, **i64)  # frames completed
        self.emitted = torch.zeros(N, P, **i64)  # frames that reached the host or were dropped on the way
        self.last_avail = torch.zeros(N, P, **f64)
        self.prev_sum = torch.zeros(N, P, **i64)
        self.ring_codes = torch.zeros(N, D, T, dtype=torch.uint8, device=dev)
        self.ring_feat = torch.zeros(N, D, P, 6, **i64)
        self.ring_avail = torch.zeros(N, D, P, **f64)
        self.ring_drop = torch.zeros(N, D, P, dtype=torch.bool, device=dev)
        self.codes = torch.zeros(N, T, dtype=torch.uint8, device=dev)  # the host's latest frame per patch
        self.features = torch.zeros(N, P, 6, **i64)
        self.t_avail = torch.zeros(N, P, **f64)
        self.t_sample = torch.zeros(N, P, **f64)
        self.seq = torch.full((N, P), -1, **i64)
        self.sync_id = torch.full((N, P), -1, **i64)
        self.n_dropped = torch.zeros(N, P, **i64)
        self.overflow = torch.zeros(N, P, **i64)
        self.frame_force = torch.zeros(N, T, **f64)  # the force each taxel's last completed frame scanned (truth)

    def reset(self, env_ids=None) -> None:
        """The boards restart with the env's clock at 0 (frame 0 starts at t = 0); the skin calibration is kept."""
        ids = slice(None) if env_ids is None else env_ids
        self.xf[ids] = 0.0
        self.t_prev[ids] = 0.0
        self.fresh[ids] = True
        self.buf[ids] = 0.0
        for k in ("done", "emitted", "prev_sum", "n_dropped", "overflow"):
            getattr(self, k)[ids] = 0
        self.last_avail[ids] = -math.inf
        self.codes[ids] = 0
        self.features[ids] = 0
        self.t_avail[ids] = 0.0
        self.t_sample[ids] = 0.0
        self.seq[ids] = -1
        self.sync_id[ids] = -1
        self.frame_force[ids] = 0.0

    def set_params(self, env_ids=None, gain_mismatch=None, offset_lsb=None, f_half=None, enob=None,
                   oversample=None, bandwidth=None, bridge_latency=None, drop_prob=None, noise_scale=None,
                   f_half_mismatch=None, dead_frac=None, phase=None) -> None:
        """Per-environment readout parameters (scalars or (n,) per env). gain_mismatch / offset_lsb draw new per-taxel
        errors (gain 1 + mismatch N(0, 1), offset offset_lsb N(0, 1) counts), like a new physical skin; dead_frac
        draws which taxels are dead (each with that probability); phase (s) shifts the boards' SYNC edges (frame k
        starts at phase + k / ts_rate; the residual loop's phase to SYNC)."""
        ids = slice(None) if env_ids is None else env_ids
        n = self.n_env if env_ids is None else len(env_ids)
        dev = self.dev

        def col(v):
            t = torch.as_tensor(v, dtype=torch.float64, device=dev)
            return t.expand(n).clone() if t.ndim == 0 else t

        if gain_mismatch is not None:
            z = torch.randn(n, self.T, dtype=torch.float64, device=dev, generator=self.generator)
            self.gain[ids] = 1.0 + col(gain_mismatch)[:, None] * z
        if offset_lsb is not None:
            z = torch.randn(n, self.T, dtype=torch.float64, device=dev, generator=self.generator)
            self.offs[ids] = col(offset_lsb)[:, None] * z
        if f_half is not None or f_half_mismatch is not None:
            if f_half is not None:
                self.f_half[ids] = col(f_half)
            spread = torch.zeros(n, self.T, dtype=torch.float64, device=dev) if f_half_mismatch is None else \
                col(f_half_mismatch)[:, None] * torch.randn(n, self.T, dtype=torch.float64, device=dev,
                                                            generator=self.generator)
            self.f_half_taxel[ids] = self.f_half[ids][:, None] * (1.0 + spread)
        if enob is not None or oversample is not None:
            e = col(self.cfg.ts_enob if enob is None else enob)
            o = col(self.cfg.ts_oversample if oversample is None else oversample)
            self.noise_lsb[ids] = 2.0 ** (self.cfg.ts_adc_bits - e) / math.sqrt(12.0) / torch.sqrt(o)
        if bandwidth is not None:
            self.a_skin[ids] = 1.0 - torch.exp(-2 * math.pi * col(bandwidth) * self.dt)
        if bridge_latency is not None:
            lo, hi = bridge_latency
            self.bridge_lo[ids], self.bridge_hi[ids] = col(lo), col(hi)
        if drop_prob is not None:
            self.drop_prob[ids] = col(drop_prob)
        if noise_scale is not None:
            self.noise_scale[ids] = col(noise_scale)
        if dead_frac is not None:
            u = torch.rand(n, self.T, dtype=torch.float64, device=dev, generator=self.generator)
            self.alive[ids] = (u >= col(dead_frac)[:, None]).double()
        if phase is not None:
            self.phase[ids] = col(phase)

    # ------------------------------------------------------------------ one physics step
    def step(self, t: torch.Tensor, raw: torch.Tensor) -> None:
        """`t` (N,) float64 the time now (after the step); `raw` (N, n_taxels) the taxel loads (N)."""
        T_ = self.period
        raw = raw.double()
        fresh = self.fresh
        xf_prev = torch.where(fresh[:, None], raw, self.xf)
        xf = torch.where(fresh[:, None], raw, self.xf + self.a_skin[:, None] * (raw - self.xf))
        tp = torch.where(fresh, t - self.dt, self.t_prev)
        # which taxels the scan reached in (tp, t]: the newest frame each taxel was sampled for, now and before
        s_off = self.s_off[None] + self.phase[:, None]
        k_now = torch.floor((t[:, None] - s_off) / T_ + self.tol_k).long()
        k_old = torch.floor((tp[:, None] - s_off) / T_ + self.tol_k).long()
        k_old = torch.where(fresh[:, None], torch.full_like(k_old, -1), k_old)
        new = (k_now > k_old) & (k_now >= 0)
        instant = k_now.double() * T_ + s_off
        al = ((instant - tp[:, None]) / (t - tp)[:, None]).clamp(0.0, 1.0)
        val = xf_prev + al * (xf - xf_prev)
        par = (k_now % 2).clamp(min=0)
        cur = self.buf.gather(1, par[:, None, :]).squeeze(1)
        self.buf.scatter_(1, par[:, None, :], torch.where(new, val, cur)[:, None, :])
        self.xf, self.t_prev, self.fresh = xf, t.clone(), torch.zeros_like(fresh)
        # frames completed this step: each patch's last taxel was sampled
        complete = new[:, self.last]  # (N, P)
        k_frame = k_now[:, self.last]  # (N, P) the completing frame's index
        self._complete(complete, k_frame)
        self._emit(t)

    def _complete(self, complete: torch.Tensor, k_frame: torch.Tensor) -> None:
        N, P, dev = self.n_env, self.P, self.dev
        par = (k_frame % 2).clamp(min=0)[:, self.patch]  # (N, T) the completing frame's buffer, per taxel
        f = self.buf.gather(1, par[:, None, :]).squeeze(1)  # (N, T)
        # divider, ADC, the firmware's calibration (sensors.taxelscan.TaxelScanSensor.adc / calibrate)
        f = f.clamp(min=0.0)
        f_read = f * self.alive  # a dead taxel reads no load
        x = self.gain * f_read / (f_read + self.f_half_taxel)
        noise = self.noise_fn(complete, k_frame).double() if self.noise_fn is not None else \
            torch.randn(N, self.T, dtype=torch.float64, device=dev, generator=self.generator)
        c = x * self.full + self.offs + noise * self.noise_lsb[:, None] * self.noise_scale[:, None]  # numpy's order
        counts = torch.clamp(torch.round(c), 0.0, float(self.full))
        xc = torch.clamp(counts / self.full, max=0.999)
        f_mn = to_mn(self.f_half[:, None] * xc / (1.0 - xc))
        codes = self.codec.encode(f_mn)
        feats = torch.zeros(N, P, 6, dtype=torch.long, device=dev)
        for (rows, cols), ks, idx in self.groups:
            feats[:, ks] = patch_features(f_mn[:, idx], rows, cols, self.prev_sum[:, ks], self.codec.contact_mn)
        self.prev_sum = torch.where(complete, feats[..., 0], self.prev_sum)
        # availability, kept in order, and drops
        if self.bridge_fn is not None:
            bridge = self.bridge_fn(complete, k_frame).double()
        else:
            u = torch.rand(N, P, dtype=torch.float64, device=dev, generator=self.generator)
            bridge = self.bridge_lo[:, None] + u * (self.bridge_hi - self.bridge_lo)[:, None]
        t0 = self.phase[:, None] + k_frame.double() * self.period
        avail = torch.maximum(t0 + self.period + self.slot[None] + bridge, self.last_avail)
        self.last_avail = torch.where(complete, avail, self.last_avail)
        u_drop = self.drop_fn(complete, k_frame).double() if self.drop_fn is not None else \
            torch.rand(N, P, dtype=torch.float64, device=dev, generator=self.generator)
        drop = (self.drop_prob[:, None] > 0) & (u_drop < self.drop_prob[:, None])
        # into the frame ring at slot k % D
        slot = (k_frame % self.D).clamp(min=0)  # (N, P)
        self.overflow += (complete & (self.done - self.emitted >= self.D)).long()
        sl_t = slot[:, self.patch]  # (N, T)
        cm_t = complete[:, self.patch]
        old = self.ring_codes.gather(1, sl_t[:, None, :]).squeeze(1)
        self.ring_codes.scatter_(1, sl_t[:, None, :], torch.where(cm_t, codes, old)[:, None, :])
        n_ar = torch.arange(N, device=dev)[:, None]
        p_ar = torch.arange(P, device=dev)[None]
        self.ring_feat[n_ar, slot, p_ar] = torch.where(complete[..., None], feats, self.ring_feat[n_ar, slot, p_ar])
        self.ring_avail[n_ar, slot, p_ar] = torch.where(complete, avail, self.ring_avail[n_ar, slot, p_ar])
        self.ring_drop[n_ar, slot, p_ar] = torch.where(complete, drop, self.ring_drop[n_ar, slot, p_ar])
        self.done += complete.long()
        self.frame_force = torch.where(cm_t, f, self.frame_force)

    def _emit(self, t: torch.Tensor) -> None:
        """Frames whose availability has come, in order; the newest one not dropped becomes the host's latest."""
        N, P, D, dev = self.n_env, self.P, self.D, self.dev
        n_ar = torch.arange(N, device=dev)[:, None]
        p_ar = torch.arange(P, device=dev)[None]
        lim = (t + 0.5 * self.dt)[:, None]
        ok = torch.ones(N, P, dtype=torch.bool, device=dev)
        popped = torch.zeros(N, P, dtype=torch.long, device=dev)
        kept = torch.zeros(N, P, dtype=torch.long, device=dev)
        newest = torch.full((N, P), -1, dtype=torch.long, device=dev)  # frame index of the newest kept frame
        for d in range(D):
            e = self.emitted + d
            sl = e % D
            ok = ok & (e < self.done) & (self.ring_avail[n_ar, sl, p_ar] <= lim)
            keep = ok & ~self.ring_drop[n_ar, sl, p_ar]
            popped += ok.long()
            kept += keep.long()
            newest = torch.where(keep, e, newest)
        got = newest >= 0
        sl = newest.clamp(min=0) % D
        self.features = torch.where(got[..., None], self.ring_feat[n_ar, sl, p_ar], self.features)
        self.t_avail = torch.where(got, self.ring_avail[n_ar, sl, p_ar], self.t_avail)
        self.t_sample = torch.where(got, self.phase[:, None] + newest.double() * self.period, self.t_sample)
        self.sync_id = torch.where(got, newest & 0xFFFF, self.sync_id)
        sl_t = sl[:, self.patch]
        codes = self.ring_codes.gather(1, sl_t[:, None, :]).squeeze(1)
        self.codes = torch.where(got[:, self.patch], codes, self.codes)
        self.seq = self.seq + kept
        self.n_dropped += popped - kept
        self.emitted += popped

    # ------------------------------------------------------------------ views
    def decoded(self) -> torch.Tensor:
        """The host's latest taxel values (N, float64): the codes decoded back to newtons (the testbed's value)."""
        return self.codec.decode(self.codes)

    def board_status(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(seq, t_avail) of each board's newest frame (its first patch's, as ResidualHandL2 reads them), (N, B)."""
        b = self._board_first
        return self.seq[:, b], self.t_avail[:, b]
