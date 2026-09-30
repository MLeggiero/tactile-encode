"""Export Task A episodes for the browser replay viewer.

    python -m tactile_sim.viewer.export --n 8 --out runs/replay.html
    python -m tactile_sim.viewer.export --n 6 --preset low-torsion --preset weak-grip --out runs/compare.html

Poses of every visual body are recorded at 2 kHz; the export keeps 200 Hz frames plus every frame within
[-20 ms, +40 ms] of each hammer-nail contact, so the 5 ms blows can be replayed in slow motion. Traces are
max-pooled where short pulses matter (contact force, wrist F/T) so peaks survive downsampling.
"""

from __future__ import annotations

import argparse
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


def _geom_record(m: mujoco.MjModel, g: int, body_name: str) -> dict | None:
    typ = int(m.geom_type[g])
    kinds = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
             mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere"}
    if typ not in kinds:
        return None
    return {"body": body_name, "type": kinds[typ], "size": np.round(m.geom_size[g], 5).tolist(),
            "pos": np.round(m.geom_pos[g], 5).tolist(), "quat": np.round(m.geom_quat[g], 6).tolist(),
            "rgba": np.round(m.geom_rgba[g], 3).tolist(), "name": m.geom(g).name}


def scene_geometry(model: mujoco.MjModel) -> tuple[list[str], list[dict]]:
    """Drawable primitives per body. Mesh geoms (Menagerie FR3) are replaced by the fallback arm's capsules,
    which share body names, kinematics and inertias."""
    geoms: list[dict] = []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b == 0:
            continue  # floor and other world geoms: the viewer draws its own ground
        rec = _geom_record(model, g, model.body(b).name)
        if rec is not None:
            geoms.append(rec)
    have_arm = any(r["body"].startswith("fr3_link") for r in geoms)
    if not have_arm:
        fb = mujoco.MjModel.from_xml_path(str(FALLBACK_ARM_XML))
        for g in range(fb.ngeom):
            rec = _geom_record(fb, g, fb.body(int(fb.geom_bodyid[g])).name)
            if rec is not None:
                geoms.append(rec)
    bodies = sorted({r["body"] for r in geoms}, key=lambda n: model.body(n).id)
    return bodies, geoms


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
    bodies, _ = scene_geometry(m)
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
            "v": round(r.v_strike_actual, 3), "peak": round(r.peak_force_truth, 1),
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
        "key": key, "label": label, "arm": w.arm_source, "dt": w.dt,
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


def run_and_export(presets: list[str], n: int, seed: int = 0, fast: bool = False) -> dict:
    episodes, geoms = [], None
    for key in presets:
        label, over = PRESETS[key]
        base = fast_config() if fast else SimConfig()
        cfg = base.replace(**over) if over else base
        ep = Episode(cfg, seed=seed, frame_hz=2000.0)
        res = ep.run(n)
        if geoms is None:
            _, geoms = scene_geometry(res.testbed.world.model)
        episodes.append(export_episode(ep, res, label, key))
        s = res.summary
        print(f"{key}: {len(res.strikes)} strikes, depth {1e3 * s.get('total_depth', 0):.1f} mm, "
              f"hit rate {s.get('hit_rate', 0):.2f}, max slip {np.degrees(s.get('max_slip_rot', 0)):.2f} deg")
    return {"geoms": geoms, "episodes": episodes}


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
