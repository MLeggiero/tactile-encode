"""Search the WUJI Hand 2's wrap on a hammer: where the handle sits in the hand and at what angle across the palm, how
the thumb closes, where along the handle the hand holds it.

Per candidate the keyframe is synthesized as wuji_grasp.synthesize does it (the hammer welded at the placement, the
fingers closed at full synergy, the weld released, the hammer seated under gravity, held), then the hand holds at the
hammer table's grip level and the candidate is measured in the hand frame (fingers along -z, the palm facing +y, the
thumb on +x):

- the normal force on the handle per hand body; the palm's share and over how many of the handle's slices it bears
  (the handle is a chain of 15 mm convex slices, so this is how long a stretch of handle lies on the palm);
- the thumb: its force on the handle, the direction it pushes (a thumb on top of the handle pushes it toward the
  palm, -y), how far its tip sits above the index and middle fingers' middle segments, and its force on them;
- the joints at their mechanical stops (wuji_grasp.joint_loads): a knuckle on its stop puts its torque into the
  stop, not into the handle;
- optionally (breakaway=True) the torque the grasp resists before the hammer turns in the hand: an external torque
  on the hammer, ramped about each of its own axes both ways (x along the handle, y along the head, z across both),
  until it has turned STOP_DEG; the torques at 2 and 5 degrees.

The taxels' view of a grasp (how many of each patch's taxels it loads) needs the testbed's skin model; the scripts that
chose REAL_HAMMER_OVERRIDES (config) took the most promising candidates of this search into the testbed for that.

    python -m tactile_sim.model.hands.wrap_search --out runs/wrap_search --workers 16 --breakaway \\
        --grid '{"wrap_tcp_z": [-0.066, -0.072, -0.078], "wrap_yaw_deg": [0, 15, 30]}'
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np

from tactile_sim import names

RAMP = 4.0  # Nm/s
T_MAX = 6.0  # s of ramp: 24 Nm
STOP_DEG = 10.0
S_HOLD = 0.7  # the hammer table's grip level

BASE = {"hammer": "ycb", "grip_from_head": 0.22, "wrap_tcp_x": -0.004, "wrap_tcp_y": 0.0315, "wrap_tcp_z": -0.099,
        "wrap_yaw_deg": 0.0, "thumb_close": [1, -1, 1, 1], "thumb_preshape": [0.0, -0.8, 0.0, 0.0],
        "finger_synergy": "rated", "s_hold": S_HOLD}
PALM = "r_wrist"


def candidate_config(p: dict):
    """The residual experiments' hammer configuration with a candidate's wrap."""
    from tactile_sim.config import residual_config

    cfg = residual_config(hammer=p["hammer"])
    h = {**dataclasses.asdict(cfg.hammer), "grip_from_head": p["grip_from_head"]}
    return cfg.replace(hammer=h, gripper={
        "wrap_tcp": (p["wrap_tcp_x"], p["wrap_tcp_y"], p["wrap_tcp_z"]),
        "wrap_yaw": float(np.radians(p["wrap_yaw_deg"])),
        "thumb_close": tuple(p["thumb_close"]), "thumb_preshape": tuple(p["thumb_preshape"]),
        "finger_synergy": p["finger_synergy"]})


def _rel_angle(R0: np.ndarray, R1: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1))))


def hand_frame_contacts(m, d, hb: int, hand: int) -> list[dict]:
    """Contacts on the hammer: hand body, normal force, point and normal (from the hand into the hammer) in the hand
    frame, and the hammer geom (its handle slice)."""
    Rh, xh = d.xmat[hand].reshape(3, 3), d.xpos[hand]
    f6 = np.zeros(6)
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if hb not in (b1, b2) or b1 == b2:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        if f6[0] <= 0:
            continue
        other, g_tool = (b2, c.geom1) if b1 == hb else (b1, c.geom2)
        n = c.frame[:3] * (1.0 if b1 == other else -1.0)
        out.append({"body": m.body(other).name, "N": float(f6[0]), "p": Rh.T @ (c.pos - xh), "n": Rh.T @ n,
                    "tool_geom": int(g_tool)})
    return out


def thumb_on_fingers(m, d) -> float:
    """Normal force between the thumb and the index or middle finger (the wrap's self-contact pairs)."""
    f6 = np.zeros(6)
    tot = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = m.body(int(m.geom_bodyid[c.geom1])).name, m.body(int(m.geom_bodyid[c.geom2])).name
        if ("thumb" in a and ("index" in b or "middle_finger" in b)) or \
                ("thumb" in b and ("index" in a or "middle_finger" in a)):
            mujoco.mj_contactForce(m, d, i, f6)
            tot += max(float(f6[0]), 0.0)
    return tot


def grasp_metrics(m, d, hb: int, hand: int) -> dict:
    con = hand_frame_contacts(m, d, hb, hand)
    by: dict[str, float] = {}
    for c in con:
        by[c["body"]] = by.get(c["body"], 0.0) + c["N"]
    thumb = [c for c in con if "thumb" in c["body"]]
    t_n = sum(c["N"] for c in thumb)
    push = sum(c["N"] * c["n"] for c in thumb) / t_n if t_n > 0 else np.zeros(3)
    Rh, xh = d.xmat[hand].reshape(3, 3), d.xpos[hand]

    def at(body):
        return Rh.T @ (d.xpos[m.body(body).id] - xh)

    tip = at("r_thumb_tip_sensor_frame")
    mids = [at("r_index_finger_middle"), at("r_middle_finger_middle")]
    return {"contacts": {k: round(v, 2) for k, v in by.items()}, "total_N": round(sum(by.values()), 1),
            "palm_N": round(by.get(PALM, 0.0), 1),
            "palm_slices": len({c["tool_geom"] for c in con if c["body"] == PALM}),
            "thumb_N": round(t_n, 1), "thumb_push": np.round(push, 3).tolist(),
            "thumb_on_fingers_N": round(thumb_on_fingers(m, d), 1),
            "thumb_tip_above_fingers_mm": round(1e3 * float(tip[1] - max(p[1] for p in mids)), 1),
            "thumb_tip": np.round(tip, 4).tolist(), "finger_mids": [np.round(p, 4).tolist() for p in mids]}


def evaluate(p: dict, breakaway: bool = False) -> dict:
    """One candidate's keyframe and hold, and (breakaway) the torques it resists."""
    from tactile_sim.assets.fetch_hands import hand_xml
    from tactile_sim.control.hand import WrapSynergy
    from tactile_sim.model.hands.wuji_grasp import _rel, build_scene, grasp_config, joint_loads

    p = {**BASE, **p}
    t_wall = time.time()
    gc = grasp_config(candidate_config(p))
    g = gc.gripper
    xml, info = build_scene(gc, hand_xml("wuji2"))
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    joints = info["joints"]
    jn = [j for j, _ in joints]
    qadr = np.array([m.joint(j).qposadr[0] for j in jn])
    dadr = np.array([m.joint(j).dofadr[0] for j in jn])
    jid = np.array([m.joint(j).id for j in jn])
    act = np.array([m.actuator(f"m_{j}").id for j in jn])
    syn = WrapSynergy.for_wuji(joints, g.thumb_close, g.thumb_preshape, g.finger_synergy, g.finger_dip)
    hand, hb = m.body(names.HAND_BODY).id, m.body(names.HAMMER_BODY).id
    tcp = m.site(names.TCP_SITE).id
    hq = m.joint("hammer_free").qposadr[0]
    # the synthesis, as wuji_grasp.synthesize: placed and welded, closed at full synergy, released, seated, held
    d.qpos[qadr] = syn.q_hold
    mujoco.mj_kinematics(m, d)
    d.qpos[hq:hq + 3] = d.site_xpos[tcp]
    qq = np.zeros(4)
    mujoco.mju_mat2Quat(qq, d.site_xmat[tcp])
    d.qpos[hq + 3:hq + 7] = qq
    mujoco.mj_forward(m, d)
    p0, R0 = _rel(m, d, hand, hb)
    e = m.equality(names.GRASP_WELD).id
    m.eq_data[e][:] = 0.0
    m.eq_data[e][3:6] = p0
    mujoco.mju_mat2Quat(qq, R0.reshape(-1))
    m.eq_data[e][6:10] = qq
    m.eq_data[e][10] = 1.0
    d.eq_active[e] = 1
    level = [1.0]

    def run(T, dd, each=None):
        for _ in range(int(round(T / m.opt.timestep))):
            dd.ctrl[act] = syn.torque(level[0], dd.qpos[qadr], dd.qvel[dadr])
            if each is not None and each(dd):
                return
            mujoco.mj_step(m, dd)

    run(0.4, d)
    d.eq_active[e] = 0
    run(1.1, d)
    p1, R1 = _rel(m, d, hand, hb)
    keyframe = {"seat_shift_mm": round(1e3 * float(np.linalg.norm(p1 - p0)), 2),
                "seat_rot_deg": round(_rel_angle(R0, R1), 2), "q": d.qpos[qadr].round(3).tolist(),
                **grasp_metrics(m, d, hb, hand)}
    # the hold at the table's level
    level[0] = p["s_hold"]
    run(0.4, d)
    q = d.qpos[qadr]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    hold = {"q": dict(zip(jn, q.round(3).tolist(), strict=True)),
            "margin_to_stop": dict(zip(jn, np.minimum(q - lo, hi - q).round(3).tolist(), strict=True)),
            "stop_tau": dict(zip(jn, joint_loads(m, d, dadr).round(3).tolist(), strict=True)),
            **grasp_metrics(m, d, hb, hand)}
    out = {"params": p, "keyframe": keyframe, "hold": hold, "wall_s": None}
    if breakaway:
        # the torque on the hammer, ramped about its own axes, when it has turned 2 / 5 degrees in the hand
        base = copy.deepcopy(d)
        brk = {}
        for ax, sg in itertools.product(range(3), (1, -1)):
            dd = copy.deepcopy(base)
            _, Ra = _rel(m, dd, hand, hb)
            t0, u, first = dd.time, np.eye(3)[ax] * sg, {}

            def each(x, t0=t0, u=u, first=first, Ra=Ra):
                tau = RAMP * (x.time - t0)
                x.xfrc_applied[hb, 3:] = tau * (x.xmat[hb].reshape(3, 3) @ u)
                if int(round((x.time - t0) / m.opt.timestep)) % 8:
                    return False
                ang = _rel_angle(Ra, _rel(m, x, hand, hb)[1])
                for a in (2.0, 5.0):
                    if ang > a and a not in first:
                        first[a] = round(tau, 3)
                return ang > STOP_DEG

            run(T_MAX, dd, each)
            brk[f"{'xyz'[ax]}{'+' if sg > 0 else '-'}"] = {"2deg": first.get(2.0), "5deg": first.get(5.0)}
        cap = RAMP * T_MAX
        v5 = [cap if b["5deg"] is None else b["5deg"] for b in brk.values()]
        out.update(breakaway_Nm=brk, min5_Nm=min(v5), gmean5_Nm=float(np.exp(np.mean(np.log(v5)))))
    out["wall_s"] = round(time.time() - t_wall, 1)
    return out


def _name(p: dict) -> str:
    return "_".join(f"{k}={'/'.join(str(x) for x in v) if isinstance(v, list) else v}" for k, v in sorted(p.items()))


def _run(p: dict, out: str, breakaway: bool) -> dict:
    r = evaluate(p, breakaway)
    Path(out).write_text(json.dumps(r, indent=1))
    return r


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", default="{}", help="JSON: parameter -> list of values (the rest from BASE)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--breakaway", action="store_true", help="also ramp a torque about each axis (slower)")
    a = ap.parse_args(argv)
    grid = json.loads(a.grid)
    keys = sorted(grid)
    cands = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))] or [{}]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_run, c, str(out / f"{_name(c) or 'base'}.json"), a.breakaway): c for c in cands}
        for f in as_completed(futs):
            h = f.result()["hold"]
            print(f"{_name(futs[f]) or 'base'}: handle {h['total_N']:.0f} N, palm {h['palm_N']:.0f} N on "
                  f"{h['palm_slices']} slices, thumb {h['thumb_N']:.0f} N pushing {h['thumb_push']}, tip "
                  f"{h['thumb_tip_above_fingers_mm']:+.0f} mm over the fingers", flush=True)


if __name__ == "__main__":
    main()
