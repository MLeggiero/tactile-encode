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
from tactile_sim.config import SimConfig, fast_config, hand_config
from tactile_sim.episode import Episode, EpisodeResult

TEMPLATE = Path(__file__).with_name("template.html")
POS_UNIT = 1e-4  # m; int16 covers +-3.27 m

PRESETS = {
    "default": ("Franka Hand (default testbed)", {}),
    "wuji2": ("WUJI Hand 2 power wrap", {"hand": "wuji2"}),
    "wuji2-selflock": ("WUJI Hand 2, self-locking drives", {"hand": "wuji2", "gripper": {"lock_mode": "self_locking"}}),
    "vega-wuji2": ("Vega U, WUJI Hand 2 on each arm", {"hand": "wuji2", "robot": "vega_1u"}),
    "vega-wuji2-selflock": ("Vega U, WUJI hands, self-locking drives",
                            {"hand": "wuji2", "robot": "vega_1u", "gripper": {"lock_mode": "self_locking"}}),
    "vega1p-wuji2": ("Vega-1P, WUJI Hand 2 on each arm", {"hand": "wuji2", "robot": "vega_1p"}),
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
    """Drawable geoms per body, with the real FR3 / hand meshes (simplified) when the scene uses them, plus the
    taxel patches of a dexterous hand. A fallback-arm scene is drawn with the fallback's capsules."""
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
    skins = {model.site(sid).name.rsplit("_", 1)[0][len("taxel_"):] for sid in range(model.nsite)
             if model.site(sid).name.startswith("taxel_")}
    for sid in range(model.nsite):
        name = model.site(sid).name
        if name.startswith("taxel_"):  # a conforming skin's taxel, a small square on the surface
            geoms.append({"body": model.body(int(model.site_bodyid[sid])).name, "type": "sphere",
                          "size": [0.0007, 0.0007, 0.0007], "pos": np.round(model.site_pos[sid], 5).tolist(),
                          "quat": [1.0, 0.0, 0.0, 0.0], "rgba": [0.16, 0.47, 0.84, 0.95], "name": name})
        if name.startswith("patch_") and name[len("patch_"):] not in skins:  # flat patches, drawn as thin plates
            sz = model.site_size[sid]
            geoms.append({"body": model.body(int(model.site_bodyid[sid])).name, "type": "box",
                          "size": np.round([sz[0], sz[1], 0.0006], 5).tolist(),
                          "pos": np.round(model.site_pos[sid], 5).tolist(),
                          "quat": np.round(model.site_quat[sid], 6).tolist(), "rgba": [0.16, 0.47, 0.84, 0.85],
                          "name": name})
    if not any(r["body"].startswith(("fr3_link", "R_arm_l")) for r in geoms):
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


def export_episode(ep: Episode, res: EpisodeResult, label: str, key: str, base_dt: float = 0.005) -> dict:
    tb, w = res.testbed, res.testbed.world
    m = w.model
    bodies, _, _ = scene_geometry(m)
    bid = [m.body(n).id for n in bodies]
    ft = np.asarray(ep.frames_t)
    contacts = [r.t_contact_truth for r in res.strikes if np.isfinite(r.t_contact_truth)]
    keep = np.zeros(len(ft), dtype=bool)
    step = max(1, int(round(base_dt / (ft[1] - ft[0])))) if len(ft) > 1 else 1
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
    grip = T["pad_forces"].mean(axis=1) if w.hand.kind == "franka" else T["pad_forces"].sum(axis=1)
    tg, gg = _pool(t, grip, 0.002, "mean")
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
    acc = tb.sensors[w.hand.accel_names[0]].history()
    if len(acc["t_sample"]):
        a = np.linalg.norm(acc["value"], axis=1) - 9.81
        ta, aa = _pool(acc["t_sample"], a, 0.001)
        series["pad_accel"] = {"t": _r(ta), "y": _r(aa, 1)}

    strikes = []
    for r in res.strikes:
        # the hammer face's path from the start of the swing to just past contact, every 2 ms
        t_end = r.t_contact_truth if np.isfinite(r.t_contact_truth) else r.t_c_pred
        sel = np.flatnonzero((t >= r.t_swing_start) & (t <= t_end + 0.01))[::max(1, int(round(0.002 / w.dt)))]
        strikes.append({
            "path": _r(T["face_pos"][sel], 4),
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
        "limits": {k[6:]: (round(v, 3) if isinstance(v, float) else v) for k, v in res.summary.items()
                   if k.startswith("limit_")},
        "config": {"v_strike": ep.cfg.swing.v_strike, "grip_hold": ep.cfg.controller.grip_hold,
                   "pad_torsion": ep.cfg.gripper.pad_torsion, "flex_mode": ep.cfg.arm.flex_mode,
                   "resistance_0": ep.cfg.plant.resistance_0, "drive_target_mm": ep.cfg.plant.drive_target * 1e3},
        "bodies": bodies,
        # poses as int16: positions in units of POS_UNIT (0.1 mm), quaternions scaled by 32767
        "frames": {"t": _r(ft[keep], 5), "n": int(keep.sum()), "pos_unit": POS_UNIT,
                   "pos": _b64(np.round(pos.reshape(len(pos), -1) / POS_UNIT).astype(np.int16)),
                   "quat": _b64(np.round(quat.reshape(len(quat), -1) * 32767).astype(np.int16))},
        "series": series, "strikes": strikes, "summary": summ,
        "nail_head": _r(w.data.site_xpos[w.site[names.NAIL_HEAD_SITE]], 4),
        "strike_axis": _r(tb.l1.axis, 3),
        "handle_axis": _r(w.tcp_R_nominal[:, 0], 3),
    }


def export_tactile(tb, cfg, contacts: list[float], base_dt: float = 0.005, window=(-0.02, 0.08)) -> dict | None:
    """Every taxel patch's frames as the sensor model reported them, quantised to one byte per taxel over the
    sensor's range: base_dt spacing through the episode plus every sample around each contact."""
    hand = tb.world.hand
    names = hand.pressure_names
    if names[0] not in tb.sensors:
        return None
    hs = [tb.sensors[n].history() for n in names]
    n = min(len(h["t_sample"]) for h in hs)
    if n == 0:
        return None
    t = hs[0]["t_sample"][:n]
    step = max(1, int(round(base_dt / np.median(np.diff(t))))) if n > 1 else 1
    keep = np.zeros(n, dtype=bool)
    keep[::step] = True
    for tc in contacts:
        keep |= (t >= tc + window[0]) & (t <= tc + window[1])
    rng = float(cfg.sensors.pressure_range)
    if any(p.taxel_pos is not None for p in hand.patches):
        rng = 10.0  # skins: a load spreads over many small taxels, so a finer colour scale
    vals = np.concatenate([h["value"][:n][keep] for h in hs], axis=1)
    q = np.clip(np.round(vals / rng * 255.0), 0, 255).astype(np.uint8)
    nr, nc = cfg.sensors.taxel_grid
    patches, off = [], 0
    skin = any(p.taxel_pos is not None for p in hand.patches)
    view = None
    if skin:
        from tactile_sim.assets.fetch_hands import hand_xml
        from tactile_sim.model.hands.taxel_layout import palm_view

        view = palm_view(str(hand_xml("wuji2")), tuple(tb.world.spec.hand_info["patches"]))
    for k, p in enumerate(hand.patches):
        pr, pc = hand.grid(k)
        if hand.kind == "franka":
            # viewed from the handle: hammer head to the left, fingertip up; the right finger mirrored
            label, v = ("Left pad", "L") if p.name == "L" else ("Right pad", "R")
        else:
            label = {"palm": "Palm", "thumb": "Thumb"}.get(p.name, p.name.replace("_", " ").capitalize())
            v = "uv"
        rec = {"name": p.name, "label": label, "view": v, "rows": pr, "cols": pc, "off": off,
               "mm": [round(2e3 * p.half[0], 1), round(2e3 * p.half[1], 1)], "col": k}
        if view is not None:
            rec["xy"] = _r(view[k]["xy"].reshape(-1), 2)
            rec["cell"] = list(view[k]["cell"])
            rec["angle"] = view[k]["angle"]
        patches.append(rec)
        off += pr * pc
    if skin:
        s = cfg.sensors
        note = ("TaxelScan Rev3 skins on the open hand seen from the palm (thumb to the right), each patch unrolled "
                f"flat at its true taxel pitch where it sits: a {s.ts_palm_grid[0]} x "
                f"{s.ts_palm_grid[1]} palm sheet and {s.ts_finger_grid[0]} x {s.ts_finger_grid[1]} patches on each "
                "finger's middle segment and fingertip, rows running along the link toward its tip. Values are the "
                "RP2350 boards' calibrated output (12-bit SAR, sequential scan).")
        rate, noise = s.ts_rate, "ADC"
    else:
        note = ("Viewed from the handle: hammer head to the left, fingertip up." if hand.kind == "franka" else
                "Rows run along the handle (hammer head to the left), columns across it; each patch sits where "
                "the wrap loads that part of the hand.")
        rate, noise = cfg.sensors.pressure_rate, cfg.sensors.pressure_noise
    return {"t": _r(t[keep], 5), "data": _b64(q), "rows": nr, "cols": nc, "range": rng, "patches": patches,
            "rate": rate, "noise": noise, "note": note, "layout": "hand" if skin else "pads"}


def run_and_export(presets: list[str], n: int, seed: int = 0, fast: bool = False) -> dict:
    """Episodes may use different hands: each scene's geoms are tagged with a scene index and the viewer
    shows the ones of the selected episode's scene."""
    episodes, geoms, meshes, scenes = [], [], {}, {}
    for key in presets:
        label, over = PRESETS[key]
        over = dict(over)
        base = fast_config() if fast else SimConfig()
        cfg = hand_config(over.pop("hand", "franka"), base, robot=over.pop("robot", "fr3"), **over)
        ep = Episode(cfg, seed=seed, frame_hz=2000.0)
        res = ep.run(n)
        scene_key = (cfg.arm.robot, cfg.gripper.hand, res.testbed.world.hammer_source)
        if scene_key not in scenes:
            scenes[scene_key] = len(scenes)
            _, g, ms = scene_geometry(res.testbed.world.model)
            for rec in g:
                rec["scene"] = scenes[scene_key]
            geoms += g
            meshes.update(ms)
        exp = export_episode(ep, res, label, key)
        exp["scene"] = scenes[scene_key]
        episodes.append(exp)
        s = res.summary
        print(f"{key}: {len(res.strikes)} strikes, depth {1e3 * s.get('total_depth', 0):.1f} mm, "
              f"hit rate {s.get('hit_rate', 0):.2f}, max slip {np.degrees(s.get('max_slip_rot', 0)):.2f} deg")
    return {"geoms": geoms, "meshes": meshes, "episodes": episodes}


def export_replays(specs: list[str], fast: bool = False, seed: int = 0, robot: str = "fr3") -> dict:
    """Recorded motions replayed on a testbed robot (tactile_sim.replay), one episode each.
    A spec is "adroit:<demo>" or "dextoolbench:<category/object/task>[@speed]"; demos the robot cannot reach are
    skipped."""
    from tactile_sim.replay.runner import ReplayEpisode, robot_config
    from tactile_sim.replay.sources import load

    episodes, geoms, meshes = [], [], {}
    for spec in specs:
        src, _, arg = spec.partition(":")
        speed = 1.0
        if "@" in arg:
            arg, sp = arg.split("@")
            speed = float(sp)
        kw = {"adroit": {"demo": int(arg or 0)}, "dextoolbench": {"task": arg}, "grab": {"seq": arg}}[src]
        motion = load(src, **kw)
        cfg = robot_config(robot, None, fast)
        try:
            ep = ReplayEpisode(cfg, motion=motion, seed=seed, frame_hz=2000.0, speed=speed)
        except ValueError as e:
            print(f"{spec}: skipped ({e})")
            continue
        res = ep.run()
        res.summary.update(n_strikes=len(res.strikes), hit_rate=1.0 if res.strikes else 0.0,
                           total_depth=res.summary["nail_depth"],
                           mean_depth_inc=res.summary["nail_depth"] / max(len(res.strikes), 1))
        if not geoms:
            _, geoms, meshes = scene_geometry(res.testbed.world.model)
            for rec in geoms:
                rec["scene"] = 0
        who = {"adroit": f"Adroit human demonstration {arg} (VR + data glove)",
               "dextoolbench": f"DexToolBench {arg} (tracked from human video)",
               "grab": f"GRAB {arg} (motion capture)"}[src]
        rig = "the Vega U + WUJI Hand 2" if robot.startswith("vega") else "the FR3 + Franka Hand"
        label = f"{who}, replayed on {rig} at {res.summary['time_scale']:.1f}x the recorded time"
        exp = export_episode(ep, res, label, spec, base_dt=0.010)  # slow replays: frames every 10 ms between blows
        exp["scene"] = 0
        episodes.append(exp)
        print(f"{spec}: {len(res.strikes)} blows ({res.summary['recorded_contacts']} recorded), "
              f"peak {res.summary['peak_force']:.0f} N, slowed {res.summary['time_scale']:.1f}x")
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
    ap.add_argument("--replay", action="append", default=None,
                    help='replay a recorded motion instead of the presets: "adroit:<demo>" or '
                         '"dextoolbench:<category/object/task>[@speed]" (repeatable)')
    ap.add_argument("--replay-robot", choices=["fr3", "vega_1u"], default="fr3")
    args = ap.parse_args(argv)
    if args.replay:
        data = export_replays(args.replay, args.fast, args.seed, args.replay_robot)
    else:
        data = run_and_export(args.preset or ["default"], args.n, args.seed, args.fast)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, separators=(",", ":")))
    out = build_html(data, args.out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
