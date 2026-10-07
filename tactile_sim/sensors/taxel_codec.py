"""TaxelScan fast-profile frame codec: the contract between the simulation, the board firmware and the host.

At 1 kHz each board sends one frame per SYNC pulse:

    header (8 B) | taxel map (U8 mu-law codes, or U6: four 6-bit codes in three bytes) |
    per-patch features (6 x 16-bit each) | CRC-16/CCITT-FALSE (2 B)

Everything is integer arithmetic on taxel forces in millinewtons (the firmware's calibrated unit), so numpy, torch
and C produce identical bytes:

- A taxel's code is the number of thresholds at or below its force (`CodecSpec.thresholds_mn`). The thresholds sit
  midway, in the companded domain, between the codes' representative forces f_max ((1 + mu)^(k / (L - 1)) - 1) / mu,
  so steps are fine at light touch and coarse near full scale.
- Patch features are computed from the millinewton values, not from the codes, so slip cues survive a coarse map:
  force sum, centre of pressure (Q8.8 taxel-grid units, rows then columns), contact area (taxels at or above
  contact_mn), peak taxel, and the change of the force sum since the previous frame.

Header (little endian): sync_id u16 (SYNC pulse count, low 16 bits), seq u8 (board frame count), flags u8
(FLAG_* bits, map format in bits 4-5), addr u8, n_patches u8, n_taxels u16. Features per patch, in this order:
sum_dN u16 (0.1 N), cop_r_q8 i16, cop_c_q8 i16 (-1, -1 when the patch is unloaded), area u16, peak_cN u16 (10 mN),
rate_dN i16 (0.1 N per frame).

`export_c_header` writes the same constants and table for the firmware and the robot-side reader;
`golden_vectors` makes the frames both must reproduce byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

CODEC_VERSION = 1
HEADER_BYTES = 8
FEATURE_FIELDS = ("sum_dN", "cop_r_q8", "cop_c_q8", "area", "peak_cN", "rate_dN")
FEATURE_SIGNED = (False, True, True, False, False, True)
FEATURE_BYTES = 2 * len(FEATURE_FIELDS)
MAP_FORMATS = {"u8": 0, "u6": 1}

FLAG_NOSYNC = 0x01  # free-running: no SYNC edge seen for this frame
FLAG_OVERRUN = 0x02  # the scan or the poll missed its slot
FLAG_TARE_SUSPECT = 0x04  # baseline tracking was frozen on a suspected contact
FLAG_SATURATED = 0x08  # some taxel at or above full scale
FLAG_FORMAT_SHIFT = 4


@dataclass(frozen=True)
class CodecSpec:
    bits: int = 8  # 8 (U8) or 6 (U6)
    mu: float = 255.0
    f_max: float = 30.0  # N: full scale, the top code at and above it
    contact_mn: int = 50  # a taxel at or above this counts toward the patch's contact area

    def __post_init__(self):
        if self.bits not in (6, 8):
            raise ValueError(f"bits must be 6 or 8, not {self.bits}")

    @property
    def levels(self) -> int:
        return 2**self.bits

    @property
    def map_format(self) -> str:
        return "u8" if self.bits == 8 else "u6"

    @cached_property
    def thresholds_mn(self) -> np.ndarray:
        """levels - 1 strictly increasing thresholds (uint32, mN): code k covers [t_(k-1), t_k)."""
        top = self.levels - 1
        y = (np.arange(1, self.levels) - 0.5) / top
        t = np.round(1000.0 * self.f_max * (np.power(1.0 + self.mu, y) - 1.0) / self.mu).astype(np.int64)
        t = np.maximum(t, np.arange(1, self.levels))  # strictly increasing even where mN rounding would tie
        for k in range(1, len(t)):
            t[k] = max(t[k], t[k - 1] + 1)
        return t.astype(np.uint32)

    def code_force(self, codes: np.ndarray) -> np.ndarray:
        """Representative force (N) of each code."""
        y = np.asarray(codes, dtype=float) / (self.levels - 1)
        return self.f_max * (np.power(1.0 + self.mu, y) - 1.0) / self.mu

    def encode(self, f_mn: np.ndarray) -> np.ndarray:
        """Codes (uint8) for taxel forces in integer mN."""
        f = np.asarray(f_mn, dtype=np.int64)
        return np.searchsorted(self.thresholds_mn.astype(np.int64), f, side="right").astype(np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return self.code_force(codes)


def to_mn(force_n: np.ndarray) -> np.ndarray:
    """Taxel forces (N) as the firmware holds them: integer mN, rounded, clipped to [0, 65535]."""
    return np.clip(np.round(np.asarray(force_n, dtype=float) * 1000.0), 0, 65535).astype(np.int64)


def _div_round(num: np.ndarray | int, den: int) -> int:
    """Integer division rounding half away from zero (num may be negative, den > 0)."""
    num = int(num)
    q = (abs(num) * 2 + den) // (2 * den)
    return q if num >= 0 else -q


def patch_features(f_mn: np.ndarray, grid: tuple[int, int], prev_sum_dN: int = 0,
                   contact_mn: int = 50) -> np.ndarray:
    """The six features of one patch (int64 array, field order FEATURE_FIELDS) from its taxels in mN, row-major
    over `grid` = (rows, columns)."""
    f = np.asarray(f_mn, dtype=np.int64).reshape(grid)
    total = int(f.sum())
    sum_dN = min(_div_round(total, 100), 65535)
    if total > 0:
        r = np.arange(grid[0], dtype=np.int64)[:, None]
        c = np.arange(grid[1], dtype=np.int64)[None, :]
        cop_r = _div_round(256 * int((f * r).sum()), total)
        cop_c = _div_round(256 * int((f * c).sum()), total)
    else:
        cop_r = cop_c = -1
    area = int((f >= contact_mn).sum())
    peak = min(_div_round(int(f.max()), 10), 65535)
    rate = int(np.clip(sum_dN - prev_sum_dN, -32768, 32767))
    return np.array([sum_dN, cop_r, cop_c, area, peak, rate], dtype=np.int64)


def feature_values(feat: np.ndarray) -> dict[str, np.ndarray]:
    """Features in physical units: force sum (N), centre of pressure (taxel-grid rows, columns; NaN unloaded),
    area (taxels), peak (N), rate (N per frame). `feat` is (..., 6)."""
    feat = np.asarray(feat, dtype=float)
    loaded = feat[..., 0] > 0
    return {"sum": feat[..., 0] * 0.1, "cop_r": np.where(loaded, feat[..., 1] / 256.0, np.nan),
            "cop_c": np.where(loaded, feat[..., 2] / 256.0, np.nan), "area": feat[..., 3],
            "peak": feat[..., 4] * 0.01, "rate": feat[..., 5] * 0.1}


def pack_u6(codes: np.ndarray) -> bytes:
    """Four 6-bit codes in three bytes: code k in bits [6k, 6k + 6) of a little-endian bit stream."""
    c = np.asarray(codes, dtype=np.uint32)
    if np.any(c > 63):
        raise ValueError("U6 codes must be < 64")
    n = len(c)
    pad = (-n) % 4
    c = np.concatenate([c, np.zeros(pad, dtype=np.uint32)]).reshape(-1, 4)
    word = c[:, 0] | (c[:, 1] << 6) | (c[:, 2] << 12) | (c[:, 3] << 18)
    out = np.stack([word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF], axis=1).astype(np.uint8).reshape(-1)
    return out[: (6 * n + 7) // 8].tobytes()


def unpack_u6(data: bytes, n: int) -> np.ndarray:
    b = np.frombuffer(bytes(data) + bytes((-len(data)) % 3), dtype=np.uint8).astype(np.uint32).reshape(-1, 3)
    word = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    c = np.stack([(word >> s) & 0x3F for s in (0, 6, 12, 18)], axis=1).reshape(-1)
    return c[:n].astype(np.uint8)


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class BoardLayout:
    """The patches one board scans, in scan and frame order: (name, (rows, columns)) each."""

    addr: int
    patches: tuple[tuple[str, tuple[int, int]], ...]

    @property
    def n_taxels(self) -> int:
        return sum(r * c for _, (r, c) in self.patches)


def encode_frame(spec: CodecSpec, board: BoardLayout, f_mn: np.ndarray, sync_id: int, seq: int,
                 prev_sums_dN: np.ndarray | None = None, flags: int = 0) -> tuple[bytes, np.ndarray]:
    """One board frame from its taxels (mN, the board's patches concatenated row-major). Returns the bytes and the
    (n_patches, 6) features (whose sum_dN the next frame's rate needs)."""
    f = np.asarray(f_mn, dtype=np.int64)
    if f.shape != (board.n_taxels,):
        raise ValueError(f"board {board.addr} has {board.n_taxels} taxels, got {f.shape}")
    codes = spec.encode(f)
    if np.any(f >= spec.thresholds_mn[-1]):
        flags |= FLAG_SATURATED
    flags |= MAP_FORMATS[spec.map_format] << FLAG_FORMAT_SHIFT
    prev = np.zeros(len(board.patches), dtype=np.int64) if prev_sums_dN is None else prev_sums_dN
    feats, k = [], 0
    for i, (_, grid) in enumerate(board.patches):
        n = grid[0] * grid[1]
        feats.append(patch_features(f[k:k + n], grid, int(prev[i]), spec.contact_mn))
        k += n
    feats = np.array(feats, dtype=np.int64)
    head = np.array([sync_id & 0xFFFF], dtype="<u2").tobytes() + bytes(
        [seq & 0xFF, flags & 0xFF, board.addr & 0xFF, len(board.patches)]) + np.array(
        [board.n_taxels], dtype="<u2").tobytes()
    body = codes.astype(np.uint8).tobytes() if spec.bits == 8 else pack_u6(codes)
    fbytes = b"".join(np.array([v], dtype="<i2" if s else "<u2").tobytes()
                      for row in feats for v, s in zip(row, FEATURE_SIGNED, strict=True))
    payload = head + body + fbytes
    return payload + np.array([crc16_ccitt(payload)], dtype="<u2").tobytes(), feats


def frame_bytes(spec: CodecSpec, board: BoardLayout) -> int:
    n = board.n_taxels
    return HEADER_BYTES + (n if spec.bits == 8 else (6 * n + 7) // 8) + FEATURE_BYTES * len(board.patches) + 2


def decode_frame(spec: CodecSpec, board: BoardLayout, data: bytes) -> dict:
    """Parse and check one frame; raises ValueError on a bad CRC or a layout mismatch."""
    if crc16_ccitt(data[:-2]) != int(np.frombuffer(data[-2:], dtype="<u2")[0]):
        raise ValueError("CRC mismatch")
    sync_id = int(np.frombuffer(data[0:2], dtype="<u2")[0])
    seq, flags, addr, n_p = data[2], data[3], data[4], data[5]
    n = int(np.frombuffer(data[6:8], dtype="<u2")[0])
    if addr != board.addr or n_p != len(board.patches) or n != board.n_taxels:
        raise ValueError("frame does not match the board layout")
    fmt = (flags >> FLAG_FORMAT_SHIFT) & 0x3
    if fmt != MAP_FORMATS[spec.map_format]:
        raise ValueError("map format does not match the codec")
    k = HEADER_BYTES
    if spec.bits == 8:
        codes = np.frombuffer(data[k:k + n], dtype=np.uint8).copy()
        k += n
    else:
        nb = (6 * n + 7) // 8
        codes = unpack_u6(data[k:k + nb], n)
        k += nb
    feats = np.zeros((n_p, len(FEATURE_FIELDS)), dtype=np.int64)
    for i in range(n_p):
        for j, signed in enumerate(FEATURE_SIGNED):
            feats[i, j] = int(np.frombuffer(data[k:k + 2], dtype="<i2" if signed else "<u2")[0])
            k += 2
    return {"sync_id": sync_id, "seq": seq, "flags": flags & 0x0F, "addr": addr, "codes": codes, "features": feats}


# ------------------------------------------------------------------ fast-profile boards and golden vectors

def fast_profile_boards(palm_grid=(8, 16), finger_grid=(8, 4)) -> tuple[BoardLayout, ...]:
    """The default 448-taxel hand on four boards of at most 128 taxels, patches in TaxelScan layout order:
    [palm], [thumb m/d, index m/d], [middle m/d, ring m/d], [pinky m/d]."""
    def seg(f):
        return ((f"{f}_middle", finger_grid), (f"{f}_distal", finger_grid))

    return (BoardLayout(0, (("palm", palm_grid),)),
            BoardLayout(1, seg("thumb") + seg("index")),
            BoardLayout(2, seg("middle") + seg("ring")),
            BoardLayout(3, seg("pinky")))


def golden_vectors(spec: CodecSpec, seed: int = 20261007) -> list[dict]:
    """Deterministic frames for each fast-profile board: empty, a single loaded taxel, a ramp across full scale, a
    saturated patch, and random contact blobs, chained so the rate feature sees a previous frame."""
    rng = np.random.default_rng(seed)
    out = []
    for board in fast_profile_boards():
        n = board.n_taxels
        cases = {"empty": np.zeros(n), "single": np.eye(1, n, n // 3).ravel() * 1.234,
                 "ramp": np.linspace(0.0, 1.2 * spec.f_max, n), "saturated": np.full(n, 2.0 * spec.f_max)}
        for k in range(3):
            blob = np.zeros(n)
            c = rng.integers(0, n)
            blob[max(0, c - 6):c + 6] = rng.uniform(0.0, 0.6 * spec.f_max, len(blob[max(0, c - 6):c + 6]))
            cases[f"blob{k}"] = blob
        prev = None
        for seq, (name, f) in enumerate(cases.items()):
            f_mn = to_mn(f)
            data, feats = encode_frame(spec, board, f_mn, sync_id=0xFFF0 + seq, seq=seq, prev_sums_dN=prev)
            out.append({"board": board.addr, "case": name, "f_mn": f_mn, "bytes": np.frombuffer(data, np.uint8),
                        "features": feats})
            prev = feats[:, 0]
    return out


def export_c_header(spec: CodecSpec, path=None) -> str:
    """C header with the codec constants, the threshold table and the frame layout (generated; do not edit)."""
    t = spec.thresholds_mn
    rows = ",\n".join("    " + ", ".join(f"{int(v)}u" for v in t[i:i + 12]) for i in range(0, len(t), 12))
    boards = fast_profile_boards()
    lines = [
        "/* TaxelScan fast-profile codec (generated by tactile_sim.sensors.taxel_codec; do not edit). */",
        "#pragma once", "#include <stdint.h>", "",
        f"#define TS_CODEC_VERSION {CODEC_VERSION}",
        f"#define TS_CODE_BITS {spec.bits}",
        f"#define TS_CODE_LEVELS {spec.levels}",
        f"/* mu-law, mu = {spec.mu:g}, full scale {spec.f_max:g} N */",
        f"#define TS_F_MAX_MN {int(round(1000 * spec.f_max))}u",
        f"#define TS_CONTACT_MN {spec.contact_mn}u",
        f"#define TS_HEADER_BYTES {HEADER_BYTES}",
        f"#define TS_FEATURE_BYTES {FEATURE_BYTES}",
        "#define TS_FLAG_NOSYNC 0x01u", "#define TS_FLAG_OVERRUN 0x02u", "#define TS_FLAG_TARE_SUSPECT 0x04u",
        "#define TS_FLAG_SATURATED 0x08u", f"#define TS_FLAG_FORMAT_SHIFT {FLAG_FORMAT_SHIFT}",
        f"#define TS_MAP_FORMAT {MAP_FORMATS[spec.map_format]}u /* 0 = U8, 1 = U6 */", "",
        "/* features per patch, little endian: sum_dN u16 (0.1 N), cop_r_q8 i16, cop_c_q8 i16 (Q8.8 taxel rows,",
        "   columns; -1 when unloaded), area u16 (taxels >= TS_CONTACT_MN), peak_cN u16 (10 mN), rate_dN i16",
        "   (0.1 N per frame). Accumulate sums in 64 bits; divisions round half away from zero. */",
        "",
        f"static const uint32_t TS_THRESH_MN[{len(t)}] = {{", rows, "};", "",
        "/* code = number of thresholds <= f (mN) */",
        "static inline uint8_t ts_encode(uint32_t f_mn) {",
        "    uint32_t lo = 0, hi = TS_CODE_LEVELS - 1;",
        "    while (lo < hi) {",
        "        uint32_t mid = (lo + hi) / 2;",
        "        if (TS_THRESH_MN[mid] <= f_mn) lo = mid + 1; else hi = mid;",
        "    }",
        "    return (uint8_t)lo;", "}", "",
        "/* fast-profile boards: taxels and frame bytes */",
    ]
    for b in boards:
        names = ", ".join(n for n, _ in b.patches)
        lines.append(f"#define TS_BOARD{b.addr}_TAXELS {b.n_taxels}u  /* {names} */")
        lines.append(f"#define TS_BOARD{b.addr}_FRAME_BYTES {frame_bytes(spec, b)}u")
    text = "\n".join(lines) + "\n"
    if path is not None:
        from pathlib import Path

        Path(path).write_text(text)
    return text
