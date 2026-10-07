"""TaxelScan fast-profile codec: mu-law threshold table, U6 packing, CRC, patch features, frame round trips, the
frozen golden frames, and the generated C header encoding the same codes."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tactile_sim.sensors.taxel_codec import (
    CODEC_VERSION,
    FLAG_SATURATED,
    CodecSpec,
    crc16_ccitt,
    decode_frame,
    encode_frame,
    export_c_header,
    fast_profile_boards,
    feature_values,
    frame_bytes,
    golden_vectors,
    pack_u6,
    patch_features,
    to_mn,
    unpack_u6,
)

GOLDEN = Path(__file__).parent / "data" / f"taxel_codec_v{CODEC_VERSION}.json"
SPECS = {"u8": CodecSpec(bits=8), "u6": CodecSpec(bits=6)}


@pytest.mark.parametrize("fmt", ["u8", "u6"])
def test_threshold_table_is_monotone_and_covers_full_scale(fmt):
    s = SPECS[fmt]
    t = s.thresholds_mn.astype(np.int64)
    assert len(t) == s.levels - 1 and np.all(np.diff(t) > 0)
    assert t[0] > 0 and t[-1] < 1000 * s.f_max
    f = np.arange(0, 1000 * s.f_max + 2000, 7)
    codes = s.encode(f)
    assert codes[0] == 0 and codes[-1] == s.levels - 1 and np.all(np.diff(codes.astype(int)) >= 0)
    # code k covers [t_(k-1), t_k)
    k = np.arange(1, s.levels)
    assert np.all(s.encode(t) == k) and np.all(s.encode(t - 1) == k - 1)


def test_u8_steps_are_fine_at_light_touch_and_coarse_at_full_scale():
    s = SPECS["u8"]
    t = s.thresholds_mn.astype(float) / 1000.0
    step = np.diff(t)
    assert step[0] < 0.005  # a few mN per code near zero
    k20 = int(np.searchsorted(t, 20.0))
    assert 0.3 < step[k20] < 0.6  # ~0.45 N per code at 20 N
    # round-trip error: half a companded step (relatively larger at light touch, where mu-law is near linear),
    # plus the 1 mN integer resolution
    f = np.linspace(0.2, s.f_max, 500)
    err = np.abs(s.decode(s.encode(to_mn(f))) - f) / f
    half = (1 + s.mu) ** (0.5 / (s.levels - 1)) - 1
    assert np.all(err <= 1.2 * (half * (1 + s.f_max / (s.mu * f)) + 0.001 / f))
    assert err[f > 5.0].max() < 0.013


def test_u6_packing_round_trips():
    rng = np.random.default_rng(0)
    for n in (1, 3, 4, 5, 32, 64, 127, 128):
        c = rng.integers(0, 64, n).astype(np.uint8)
        b = pack_u6(c)
        assert len(b) == (6 * n + 7) // 8
        assert np.array_equal(unpack_u6(b, n), c)
    with pytest.raises(ValueError):
        pack_u6(np.array([64]))


def test_crc16_ccitt_false_check_value():
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_patch_features():
    grid = (8, 4)
    f = np.zeros(32, dtype=np.int64)
    feat = patch_features(f, grid, prev_sum_dN=7)
    assert list(feat) == [0, -1, -1, 0, 0, -7]
    f[2 * 4 + 3] = 1234  # row 2, column 3
    feat = patch_features(f, grid)
    assert list(feat) == [12, 2 * 256, 3 * 256, 1, 123, 12]
    f[5 * 4 + 1] = 1234  # a second, equal taxel at row 5, column 1: CoP at the midpoint
    sum_dN, cop_r, cop_c, area, peak, rate = patch_features(f, grid, prev_sum_dN=12)
    assert (cop_r, cop_c, area, rate) == (int(3.5 * 256), 2 * 256, 2, 25 - 12)
    v = feature_values(np.array([sum_dN, cop_r, cop_c, area, peak, rate]))
    assert v["sum"] == pytest.approx(2.5) and v["cop_r"] == pytest.approx(3.5) and v["peak"] == pytest.approx(1.23)
    assert patch_features(np.full(32, 10), grid, contact_mn=50)[3] == 0  # below the contact threshold


@pytest.mark.parametrize("fmt", ["u8", "u6"])
def test_frames_round_trip_and_fit_the_bus(fmt):
    s = SPECS[fmt]
    rng = np.random.default_rng(1)
    total = 0
    for b in fast_profile_boards():
        f = to_mn(rng.uniform(0, 1.3 * s.f_max, b.n_taxels))
        data, feats = encode_frame(s, b, f, sync_id=70000, seq=300)
        assert len(data) == frame_bytes(s, b)
        out = decode_frame(s, b, data)
        assert out["sync_id"] == 70000 & 0xFFFF and out["seq"] == 300 & 0xFF and out["addr"] == b.addr
        assert np.array_equal(out["codes"], s.encode(f)) and np.array_equal(out["features"], feats)
        assert out["flags"] & FLAG_SATURATED
        bad = bytearray(data)
        bad[9] ^= 0x01  # one flipped bit in the map
        with pytest.raises(ValueError):
            decode_frame(s, b, bytes(bad))
        total += len(data)
    # one 1 ms frame from every board, at 12.5 Mbaud (0.8 us per byte): well inside the period
    assert sum(b.n_taxels for b in fast_profile_boards()) == 448
    assert total * 0.8e-6 < (0.55e-3 if fmt == "u8" else 0.45e-3)


def _golden_json(spec: CodecSpec) -> list[dict]:
    return [{"board": int(g["board"]), "case": g["case"], "f_mn": [int(x) for x in g["f_mn"]],
             "frame_hex": bytes(g["bytes"]).hex()} for g in golden_vectors(spec)]


def test_golden_frames_are_frozen():
    """The bytes the firmware and the host reader must reproduce. Regenerate only with a CODEC_VERSION bump:
    python -c "from tests.test_taxel_codec import write_golden; write_golden()" """
    assert GOLDEN.exists(), f"missing {GOLDEN}; run write_golden()"
    frozen = json.loads(GOLDEN.read_text())
    for fmt, spec in SPECS.items():
        assert frozen[fmt] == _golden_json(spec), fmt


def write_golden() -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for fmt, spec in SPECS.items():
        recs = ",\n".join(json.dumps(r, separators=(",", ":")) for r in _golden_json(spec))
        parts.append(f'"{fmt}": [\n{recs}\n]')
    GOLDEN.write_text("{\n" + ",\n".join(parts) + "\n}\n")


def test_c_header_carries_the_threshold_table():
    s = SPECS["u8"]
    text = export_c_header(s)
    body = text.split("TS_THRESH_MN[255] = {")[1].split("};")[0]
    table = [int(x.strip().rstrip("u")) for x in body.replace("\n", " ").split(",") if x.strip()]
    assert table == [int(v) for v in s.thresholds_mn]
    for b in fast_profile_boards():
        assert f"#define TS_BOARD{b.addr}_FRAME_BYTES {frame_bytes(s, b)}u" in text


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_encoder_matches_python(tmp_path):
    s = SPECS["u8"]
    (tmp_path / "taxel_codec.h").write_text(export_c_header(s))
    f = np.unique(np.concatenate([s.thresholds_mn.astype(np.int64) + d for d in (-1, 0, 1)] +
                                 [np.arange(0, 40000, 13)]))
    f = f[f >= 0]
    (tmp_path / "in.txt").write_text("\n".join(str(int(x)) for x in f))
    (tmp_path / "t.c").write_text(
        '#include <stdio.h>\n#include "taxel_codec.h"\n'
        'int main(void){unsigned long v; while (scanf("%lu", &v) == 1) printf("%u\\n", ts_encode((uint32_t)v));'
        'return 0;}\n')
    subprocess.run(["cc", "-O2", "-o", str(tmp_path / "t"), str(tmp_path / "t.c")], check=True)
    out = subprocess.run([str(tmp_path / "t")], input=(tmp_path / "in.txt").read_text(), capture_output=True,
                         text=True, check=True).stdout.split()
    assert [int(x) for x in out] == [int(c) for c in s.encode(f)]
