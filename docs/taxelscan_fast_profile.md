# TaxelScan fast profile: the 1 kHz frame contract

The L2 residual policy reads the hand's 448 taxels every millisecond. The simulation, the board firmware and the
robot-side reader must agree on what a frame contains, byte for byte. This document is that contract, version 1.

The reference implementation is `tactile_sim/sensors/taxel_codec.py`. The frozen frames are in
`tests/data/taxel_codec_v1.json`. Firmware and host code should be checked against those frames, not against
this prose.

## Boards and patch order

The skin is the simulation's TaxelScan layout. There are 11 patches, and each board holds at most 128 taxels, so
it can scan once per millisecond. Taxels within a patch are row-major: rows run along the link (proximal to
distal) and columns run across it. A board's patches are concatenated in the order listed.

| Board (addr) | Patches, in frame order | Taxels | U8 frame | U6 frame |
|---|---|---|---|---|
| 0 | palm (8×16) | 128 | 150 B | 118 B |
| 1 | thumb_middle, thumb_distal, index_middle, index_distal (8×4 each) | 128 | 186 B | 154 B |
| 2 | middle_middle, middle_distal, ring_middle, ring_distal | 128 | 186 B | 154 B |
| 3 | pinky_middle, pinky_distal | 64 | 98 B | 82 B |
| **All** | 11 patches | **448** | **620 B/ms** | **508 B/ms** |

## Timing

| Item | Requirement |
|---|---|
| Frame start | Hardware SYNC edge every 1.000 ms, on every board |
| Scan | All of a board's taxels within the period |
| Poll | The master polls every board in fixed slots inside each period, reporting the previous period's scan |
| Host latency | SYNC edge → frame at the host: ≤ 3 ms at p99 |
| Bus | 12.5 Mbaud (8N1): U8 frames take about 0.50 ms of each period, U6 about 0.41 ms, plus polls and turnarounds |
| USB | The master's link to the host must carry about 0.6 MB/s with ≤ 1 ms added. Arduino CDC (1.28 ms per 1.1 kB frame on rev1) is too slow |

**Scan time is the open risk.** `firmware/rev3/PLAN.md` gives about 10.2 µs per taxel: a 6 µs settle plus two
bank conversions per channel pair. At that rate 128 taxels take about 1.30 ms. Options to fit 1 ms:

- **Overlap the mux settling with the bank-B conversion.** About 15.6 µs per pair, so 128 taxels take about
  1.0 ms with no margin.
- **Dark sweep every N frames** instead of every frame.
- **At most 96 taxels per board,** with the palm split into two 8×8 halves.

The simulation takes the board map, the per-taxel time and the poll schedule as parameters, so whatever the
firmware measures can be modelled exactly. That includes a frame rate below 1 kHz.

## Frame layout (little endian)

| Offset | Type | Field |
|---|---|---|
| 0 | u16 | `sync_id`: SYNC pulse count, low 16 bits |
| 2 | u8 | `seq`: the board's frame count |
| 3 | u8 | `flags`: bit 0 NOSYNC, 1 OVERRUN, 2 TARE_SUSPECT, 3 SATURATED; bits 4–5 map format (0 U8, 1 U6) |
| 4 | u8 | `addr` |
| 5 | u8 | `n_patches` |
| 6 | u16 | `n_taxels` |
| 8 | bytes | Taxel map: U8, one code per taxel; or U6, four 6-bit codes in three bytes (code k in bits [6k, 6k+6) of a little-endian bit stream) |
| … | 12 B × n_patches | Patch features, see below |
| end − 2 | u16 | CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection) over everything before it |

## Encoding

- **Unit.** Taxel force is held in integer millinewtons after the board's calibration, clipped to [0, 65535].
- **Codes.** A taxel's code is the number of entries of `TS_THRESH_MN` at or below its force.
  - The thresholds are μ-law, with μ = 255 and full scale 30 N. They sit midway, in the companded domain,
    between the codes' representative forces 30 N·((1+μ)^(k/(L−1)) − 1)/μ.
  - For U8 a step is about 2.6 mN near zero and about 0.45 N at 20 N.
  - U6 uses the same formula with L = 64.
- **Features per patch.** They are computed from the millinewton values, not from the codes. Use 64-bit
  accumulators; every division rounds half away from zero.

  | Field | Type | Definition |
  |---|---|---|
  | `sum_dN` | u16 | ΣF / 100 (0.1 N) |
  | `cop_r_q8`, `cop_c_q8` | i16 | 256·Σ(F·row) / ΣF and 256·Σ(F·col) / ΣF (Q8.8 taxel-grid units); −1, −1 when ΣF = 0 |
  | `area` | u16 | Taxels with F ≥ 50 mN |
  | `peak_cN` | u16 | max F / 10 (10 mN) |
  | `rate_dN` | i16 | `sum_dN` minus the previous frame's `sum_dN` |

## Conformance

The C header holds the threshold table, a `ts_encode()`, the flags and the frame sizes. Regenerate it with:

```
python -c "from tactile_sim.sensors.taxel_codec import CodecSpec, export_c_header; export_c_header(CodecSpec(bits=8), 'taxel_codec.h')"
```

- **Golden frames.** `tests/data/taxel_codec_v1.json` holds frames for U8 and U6. Each record gives the board,
  the case name, the input forces in mN and the expected frame as hex. A firmware build passes when it emits
  those exact bytes for those inputs.
- **Tests.** `tests/test_taxel_codec.py` checks the Python encoder against the frozen frames. It also compiles
  the generated header with `cc` and checks that `ts_encode()` matches Python on every threshold boundary.
- **Changes.** Any change to the format bumps `CODEC_VERSION` and regenerates the frames.

## To measure on the boards

Each of these sets a simulation parameter.

| Measurement | Simulation parameter |
|---|---|
| Per-taxel scan time with the fast profile | Scan duration and skew |
| Poll slot times | Per-board availability |
| SYNC → host latency histogram | Latency and jitter |
| Frame drop and CRC error rates | Drop probability |
| Indenter response (f_half, gain spread, bandwidth, hysteresis) | Load and readout model |
