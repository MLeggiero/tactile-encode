"""Contact calibration without the arm: launch the free hammer at the nail and measure the pulse.

    python -m tactile_sim.calibrate pulse        # sweep hammer-face solref timeconst x dampratio
    python -m tactile_sim.calibrate resistance   # sweep nail resistance x strike speed
    python -m tactile_sim.calibrate depth        # advance per strike as the nail goes in
"""

from __future__ import annotations

import argparse
import itertools

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.config import SimConfig
from tactile_sim.model.tool_hammer import face_offset
from tactile_sim.sim.truth import pair_force, pulse_stats
from tactile_sim.sim.world import World


def free_strike(cfg: SimConfig, v: float = 2.5, gap: float = 0.002, duration: float = 0.03,
                depth0: float = 0.0, world: World | None = None) -> dict[str, float]:
    """Hammer released from the gripper, moving at `v` along the strike axis, face `gap` short of the nail.

    Gravity is switched off for the flight so only the contact acts. Returns the pulse statistics,
    nail advance, the face's velocity along the strike axis afterwards, and the restitution
    e = -v_after / v measured at the face.
    """
    w = world or World(cfg)
    m, d = w.model, w.data
    mujoco.mj_resetData(m, d)
    q, _ = w.ik(w.hover_tcp)
    d.qpos[w.arm_qadr] = q
    d.qpos[w.finger_qadr] = cfg.gripper.finger_range
    w.plant.reset()
    d.qpos[w.plant.qadr] = depth0
    mujoco.mj_kinematics(m, d)
    R = w.tcp_R_nominal
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    head = d.site_xpos[w.site[names.NAIL_HEAD_SITE]].copy()
    face_local = np.array(face_offset(cfg.hammer))
    s = w.plant.axis
    d.qpos[w.hammer_qadr:w.hammer_qadr + 3] = head - s * gap - R @ face_local
    d.qpos[w.hammer_qadr + 3:w.hammer_qadr + 7] = quat
    d.qvel[w.hammer_dofadr:w.hammer_dofadr + 3] = v * s
    gravity = m.opt.gravity.copy()
    m.opt.gravity[:] = 0.0
    g1, g2 = m.geom(names.HAMMER_FACE_GEOM).id, m.geom(names.NAIL_HEAD_GEOM).id
    face = w.site[names.HAMMER_FACE_SITE]
    n = int(round(duration / w.dt))
    ts, fs, vf, pen = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    try:
        for i in range(n):
            w.set_grip_force(-cfg.gripper.grip_force_max)  # keep the fingers open
            w.set_arm_torque(w.hold_torque(q))
            w.step()
            ts[i] = d.time
            fs[i] = pair_force(m, d, g1, g2)[0]
            vf[i] = w.site_velocity(face)[:3] @ s
            pen[i] = -w.plant.gap()
    finally:
        m.opt.gravity[:] = gravity
    st = pulse_stats(ts, fs)
    st["advance"] = float(d.qpos[w.plant.qadr] - depth0)
    st["v_after"] = float(vf[-1])
    st["e"] = float(-vf[-1] / v)
    st["max_penetration"] = float(pen.max())  # face past the nail head's struck surface (m)
    st["t"], st["f"], st["v_face"] = ts, fs, vf
    return st


def _row(st: dict) -> str:
    return (f"peak {st['peak']:7.1f} N  width {st['width'] * 1e3:5.2f} ms  impulse {st['impulse']:.3f} Ns  "
            f"advance {st['advance'] * 1e3:5.2f} mm  e {st['e']:5.2f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["pulse", "resistance", "depth"])
    ap.add_argument("--v", type=float, default=2.5)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--arm", default="auto")
    args = ap.parse_args(argv)
    base = SimConfig().replace(arm={"source": args.arm})
    if args.dt:
        base = base.replace(physics={"timestep": args.dt})
    if args.what == "pulse":
        for tc, z in itertools.product((0.0005, 0.001, 0.0015, 0.002, 0.003), (0.2, 0.4, 0.7)):
            cfg = base.replace(hammer={"face_solref": (tc, z)}, plant={"vdr_enabled": False})
            print(f"timeconst {tc * 1e3:4.1f} ms dampratio {z:.1f}: " + _row(free_strike(cfg, args.v)))
        cfg = base.replace(plant={"proud": 0.0015 + 0.001})  # nail already at its stop: pure bounce
        for z in (0.2, 0.5, 1.0):
            st = free_strike(base.replace(plant={"vdr_zeta0": z, "vdr_zeta1": 0.0}), args.v,
                             depth0=base.plant.proud - 0.001)
            print(f"nail at stop, zeta {z:.1f}: " + _row(st))
    elif args.what == "resistance":
        for r0, v in itertools.product((200, 450, 800, 1500), (1.5, 2.5, 4.0)):
            st = free_strike(base.replace(plant={"resistance_0": r0}), v)
            print(f"R0 {r0:5.0f} N v {v:.1f} m/s: " + _row(st))
    else:
        for depth in np.arange(0.0, 0.021, 0.004):
            st = free_strike(base, args.v, depth0=float(depth))
            print(f"depth {depth * 1e3:4.1f} mm: " + _row(st))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
