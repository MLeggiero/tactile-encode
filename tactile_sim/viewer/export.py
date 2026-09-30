"""Export Task A episodes for the browser replay viewer.

    python -m tactile_sim.viewer.export --n 8 --out runs/replay.html
    python -m tactile_sim.viewer.export --n 6 --preset low-torsion --preset weak-grip --out runs/compare.html

Poses of every visual body are recorded at 2 kHz; the export keeps 200 Hz frames plus every frame within
[-20 ms, +40 ms] of each hammer-nail contact, so the 5 ms blows can be replayed in slow motion. Traces are
max-pooled where short pulses matter (contact force, wrist F/T) so peaks survive downsampling.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.assets import FALLBACK_ARM_XML
from tactile_sim.config import SimConfig, fast_config
from tactile_sim.episode import Episode, EpisodeResult

TEMPLATE = Path(__file__).with_name("template.html")

PRESETS = {
    "default": ("Default testbed", {}),
    "low-torsion": ("Slippery pads (torsional friction 0.005 m)", {"gripper": {"pad_torsion": 0.005}}),
    "weak-grip": ("Weak hold force (15 N)", {"controller": {"grip_hold": 15.0}}),
    "rigid-wrist": ("Rigid wrist (no wrist compliance)", {"arm": {"flex_mode": "rigid"}}),
    "fast-swing": ("Faster swing (2.6 m/s)", {"swing": {"v_strike": 2.6}}),
}


def cluster_decimate(verts: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering simplification: snap vertices to a `cell`-sized grid, merge each cell to its mean,
    drop collapsed and duplicate triangles. Crude but dependency-free; fine for a replay view."""
    key = np.floor(verts / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    out_v = np.zeros((n, 3))
    np.add.at(out_v, inv, verts)
    out_v /= np.bincount(inv, minlength=n)[:, None]
    f = inv[faces]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    f = np.unique(f, axis=0) if len(f) else f
    return out_v, f


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def export_mesh(m: mujoco.MjModel, mesh_id: int, cell: float) -> dict:
    va, vn = int(m.mesh_vertadr[mesh_id]), int(m.mesh_vertnum[mesh_id])
    fa, fn = int(m.mesh_faceadr[mesh_id]), int(m.mesh_facenum[mesh_id])
    v = m.mesh_vert[va:va + vn].astype(float)
    f = m.mesh_face[fa:fa + fn].astype(np.int64)
    if len(f) > 200:
        v, f = cluster_decimate(v, f, cell)
    lo, hi = v.min(axis=0), v.max(axis=0)
    scale = np.maximum(hi - lo, 1e-9) / 65535.0
    q = np.round((v - lo) / scale).astype(np.uint16)
    idx_dtype = np.uint16 if len(v) < 65536 else np.uint32
    return {"min": np.round(lo, 6).tolist(), "scale": scale.tolist(), "v": _b64(q),
            "f": _b64(f.astype(idx_dtype)), "idx32": idx_dtype is np.uint32, "nf": int(len(f))}


def _rgba(m: mujoco.MjModel, g: int) -> list[float]:
    mat = int(m.geom_matid[g])
    rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
    return np.round(rgba, 3).tolist()


def _geom_record(m: mujoco.MjModel, g: int, body_name: str) -> dict | None:
    typ = int(m.geom_type[g])
    kinds = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
             mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
             mujoco.mjtGeom.mjGEOM_MESH: "mesh"}
    if typ not in kinds or int(m.geom_group[g]) > 2:
        return None  # collision-only and helper geoms are not drawn
    rec = {"body": body_name, "type": kinds[typ], "size": np.round(m.geom_size[g], 5).tolist(),
           "pos": np.round(m.geom_pos[g], 5).tolist(), "quat": np.round(m.geom_quat[g], 6).tolist(),
           "rgba": _rgba(m, g), "name": m.geom(g).name}
    if typ == mujoco.mjtGeom.mjGEOM_MESH:
        rec["mesh"] = m.mesh(int(m.geom_dataid[g])).name
    return rec


def scene_geometry(model: mujoco.MjModel, cell: float = 0.002) -> tuple[list[str], list[dict], dict]:
    """Drawable geoms per body, with the real FR3 / Franka Hand meshes (simplified) when the scene uses them.
    A fallback-arm scene is drawn with the fallback's capsules."""
    geoms: list[dict] = []
    meshes: dict[str, dict] = {}
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b == 0:
            continue  # floor and other world geoms: the viewer draws its own ground
        rec = _geom_record(model, g, model.body(b).name)
        if rec is None:
            continue
        if rec["type"] == "mesh" and rec["mesh"] not in meshes:
            meshes[rec["mesh"]] = export_mesh(model, int(model.geom_dataid[g]), cell)
        geoms.append(rec)
    if not any(r["body"].startswith("fr3_link") for r in geoms):
        fb = mujoco.MjModel.from_xml_path(str(FALLBACK_ARM_XML))
        for g in range(fb.ngeom):
            rec = _geom_record(fb, g, fb.body(int(fb.geom_bodyid[g])).name)
            if rec is not None:
                geoms.append(rec)
    bodies = sorted({r["body"] for r in geoms}, key=lambda n: model.body(n).id)
    return bodies, geoms, meshes


def _pool(t: np.ndarray, y: np.ndarray, dt: float, mode: str = "maxabs") -> tuple[np.ndarray, np.ndarray]:
    if len(t) == 0:
        return t, y
    edges = np.arange(t[0], t[-1] + dt, dt)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 1)
    out_t, out_y = [], []
    for k in np.unique(idx):
        sel = idx == k
        ys = y[sel]
        out_t.append(float(t[sel][0]))
        out_y.append(float(ys[np.argmax(np.abs(ys))]) if mode == "maxabs" else float(ys.mean()))
    return np.array(out_t), np.array(out_y)


def _r(a, nd=4) -> list:
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def export_episode(ep: Episode, res: EpisodeResult, label: str, key: str) -> dict:
    tb, w = res.testbed, res.testbed.world
    m = w.model
    bodies, _, _ = scene_geometry(m)
    bid = [m.body(n).id for n in bodies]
    ft = np.asarray(ep.frames_t)
    contacts = [r.t_contact_truth for r in res.strikes if np.isfinite(r.t_contact_truth)]
    keep = np.zeros(len(ft), dtype=bool)
    step = max(1, int(round(0.005 / (ft[1] - ft[0])))) if len(ft) > 1 else 1
    keep[::step] = True
    for tc in contacts:
        keep |= (ft >= tc - 0.02) & (ft <= tc + 0.04)
    pos = np.asarray(ep.frames_pos)[keep][:, bid, :]
    quat = np.asarray(ep.frames_quat)[keep][:, bid, :]

    T = res.truth
    t = T["t"]
    series = {}
    tf, ff = _pool(t, T["contact_force"], 0.0005)
    series["contact_force"] = {"t": _r(tf), "y": _r(ff, 1)}
    td, dd = _pool(t, T["nail_depth"] * 1e3, 0.005, "mean")
    series["nail_depth"] = {"t": _r(td), "y": _r(dd, 3)}
    tg, gg = _pool(t, T["pad_forces"].mean(axis=1), 0.002, "mean")
    series["grip_truth"] = {"t": _r(tg), "y": _r(gg, 2)}
    gl = tb.grip.log
    series["grip_setpoint"] = {"t": _r(gl.t[::2]), "y": _r(gl.setpoint[::2], 2)}
    ang = np.degrees(np.linalg.norm(T["hammer_in_hand_rotvec"] - T["hammer_in_hand_rotvec"][0], axis=1))
    ts, ss = _pool(t, ang, 0.005, "mean")
    series["tool_tilt"] = {"t": _r(ts), "y": _r(ss, 3)}
    fth = tb.sensors["ft"].history()
    if len(fth["t_sample"]):
        Rft = w.data.site_xmat[w.site[names.FT_SITE]].reshape(3, 3)
        f_ax = fth["value"][:, :3] @ Rft.T @ tb.l1.axis
        tq, fq = _pool(fth["t_sample"], f_ax - np.median(f_ax[: max(1, len(f_ax) // 20)]), 0.001)
        series["ft_axis"] = {"t": _r(tq), "y": _r(fq, 2)}
    acc = tb.sensors["pad_acc_L"].history()
    if len(acc["t_sample"]):
        a = np.linalg.norm(acc["value"], axis=1) - 9.81
        ta, aa = _pool(acc["t_sample"], a, 0.001)
        series["pad_accel"] = {"t": _r(ta), "y": _r(aa, 1)}

    strikes = []
    for r in res.strikes:
        strikes.append({
            "idx": r.idx + 1, "hit": bool(r.hit), "t_swing": round(r.t_swing_start, 4),
            "t_contact": round(r.t_contact_truth, 5) if np.isfinite(r.t_contact_truth) else None,
            "t_pred": round(r.t_c_pred, 5),
            "t_flag": round(r.t_flag, 5) if np.isfinite(r.t_flag) else None, "flag_source": r.flag_source,
            "v": round(r.v_strike_actual, 3), "v_cmd": round(r.v_cmd, 2), "peak": round(r.peak_force_truth, 1),
            "width_ms": round(r.pulse_width * 1e3, 2), "depth_mm": round(r.depth_inc * 1e3, 3),
            "depth_after_mm": round(r.depth_after * 1e3, 3),
            "slip_mm": round(r.slip_trans * 1e3, 3), "slip_deg": round(float(np.degrees(r.slip_rot)), 3),
            "latency_ms": round(r.flag_latency * 1e3, 3) if np.isfinite(r.flag_latency) else None,
            "grip_contact": round(r.grip_at_contact, 1) if np.isfinite(r.grip_at_contact) else None,
            "grip_peak_rel_ms": round(r.t_grip_peak_rel * 1e3, 1) if np.isfinite(r.t_grip_peak_rel) else None,
            "ring_ratio": round(r.ringing_energy / r.pre_energy, 1)
            if np.isfinite(r.ringing_energy) and r.pre_energy > 0 else None,
        })
    summ = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in res.summary.items()}
    return {
        "tactile": export_tactile(tb, ep.cfg, contacts),
        "key": key, "label": label, "arm": w.arm_source, "hand": w.hand_source, "dt": w.dt,
        "config": {"v_strike": ep.cfg.swing.v_strike, "grip_hold": ep.cfg.controller.grip_hold,
                   "pad_torsion": ep.cfg.gripper.pad_torsion, "flex_mode": ep.cfg.arm.flex_mode,
                   "resistance_0": ep.cfg.plant.resistance_0, "drive_target_mm": ep.cfg.plant.drive_target * 1e3},
        "bodies": bodies,
        "frames": {"t": _r(ft[keep], 5), "pos": _r(pos.reshape(len(pos), -1), 5),
                   "quat": _r(quat.reshape(len(quat), -1), 5)},
        "series": series, "strikes": strikes, "summary": summ,
        "nail_head": _r(w.data.site_xpos[w.site[names.NAIL_HEAD_SITE]], 4),
        "strike_axis": _r(tb.l1.axis, 3),
    }


def export_tactile(tb, cfg, contacts: list[float], base_dt: float = 0.005, window=(-0.02, 0.08)) -> dict | None:
    """Both pads' taxel frames as the sensor model reported them, quantised to one byte per taxel over the
    sensor's range: base_dt spacing through the episode plus every sample around each contact."""
    if "pressure_L" not in tb.sensors:
        return None
    hl, hr = tb.sensors["pressure_L"].history(), tb.sensors["pressure_R"].history()
    n = min(len(hl["t_sample"]), len(hr["t_sample"]))
    if n == 0:
        return None
    t = hl["t_sample"][:n]
    step = max(1, int(round(base_dt / np.median(np.diff(t))))) if n > 1 else 1
    keep = np.zeros(n, dtype=bool)
    keep[::step] = True
    for tc in contacts:
        keep |= (t >= tc + window[0]) & (t <= tc + window[1])
    rng = float(cfg.sensors.pressure_range)
    vals = np.concatenate([hl["value"][:n][keep], hr["value"][:n][keep]], axis=1)
    q = np.clip(np.round(vals / rng * 255.0), 0, 255).astype(np.uint8)
    nr, nc = cfg.sensors.taxel_grid
    return {"t": _r(t[keep], 5), "data": _b64(q), "rows": nr, "cols": nc, "range": rng,
            "pad_mm": [round(2e3 * cfg.gripper.pad_half[0], 1), round(2e3 * cfg.gripper.pad_half[2], 1)],
            "rate": cfg.sensors.pressure_rate, "noise": cfg.sensors.pressure_noise}


def run_and_export(presets: list[str], n: int, seed: int = 0, fast: bool = False) -> dict:
    episodes, geoms, meshes = [], None, None
    for key in presets:
        label, over = PRESETS[key]
        base = fast_config() if fast else SimConfig()
        cfg = base.replace(**over) if over else base
        ep = Episode(cfg, seed=seed, frame_hz=2000.0)
        res = ep.run(n)
        if geoms is None:
            _, geoms, meshes = scene_geometry(res.testbed.world.model)
        episodes.append(export_episode(ep, res, label, key))
        s = res.summary
        print(f"{key}: {len(res.strikes)} strikes, depth {1e3 * s.get('total_depth', 0):.1f} mm, "
              f"hit rate {s.get('hit_rate', 0):.2f}, max slip {np.degrees(s.get('max_slip_rot', 0)):.2f} deg")
    return {"geoms": geoms, "meshes": meshes, "episodes": episodes}


def build_html(data: dict, out: Path) -> Path:
    html = TEMPLATE.read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__REPLAY_DATA__*/null", payload))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8, help="strikes per episode")
    ap.add_argument("--preset", action="append", choices=sorted(PRESETS), help="repeatable; default: default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="4 kHz physics instead of 8 kHz")
    ap.add_argument("--out", type=Path, default=Path("runs/replay.html"))
    ap.add_argument("--json", type=Path, default=None, help="also write the raw data as JSON")
    args = ap.parse_args(argv)
    data = run_and_export(args.preset or ["default"], args.n, args.seed, args.fast)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, separators=(",", ":")))
    out = build_html(data, args.out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
