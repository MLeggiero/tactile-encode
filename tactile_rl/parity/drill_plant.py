"""The GPU drill plant (tactile_rl.plants.drill, 12 V model) against the CPU's, and the virtual dyno on the batched
drivetrain.

- **Plant laws on recorded inputs.** A CPU drill episode (Vega U + WUJI holding the pistol driver, the trigger
  pulled by the index finger, the trigger finger's friction capped; zero residual; DrillBase feeding) records, before
  every physics step, what DrillPlant.pre_step reads (the tool's rotation, the bit tip's position and velocity, the
  tool's angular velocity, the trigger's travel, the time, its generator's draws) and what it wrote (the tool's
  applied wrench, the screw's or the hole's state, depth, spindle speed, clutch clicks). TorchDrillLaws replays the
  inputs and the draws; the table gives the largest differences over the episode.
- **The dyno** (tactile_sim.model.plants.drill_dyno) on TorchDrivetrain, many parameter sets at once (the nominal
  drill and the randomization's draws), against the CPU dyno per set: no-load speed, ISO 5393 soft and hard joint
  torque, the clutch's mean and peak torque, slip speed and click rate, the brake's stop time, the housing's
  momentum balance.

    python -m tactile_rl.parity.drill_plant [--modes screw,hole] [--seeds 0,1] [--fast] [--dyno-draws 8]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

import numpy as np
import torch

from tactile_rl.plants.drill import TorchDrillLaws
from tactile_rl.plants.drivetrain import PARAMS, TorchDrivetrain


class _RngTap:
    """The CPU plant's generator, recording the draws its pre_step makes (unit-scaled)."""

    def __init__(self, rng):
        self.rng = rng
        self.normals: list[float] = []
        self.uniforms: list[float] = []

    def standard_normal(self, *a, **k):
        x = self.rng.standard_normal(*a, **k)
        self.normals.append(float(x))
        return x

    def uniform(self, low=0.0, high=1.0, *a, **k):
        x = self.rng.uniform(low, high, *a, **k)
        self.uniforms.append(float((x - low) / (high - low)) if high > low else 0.0)
        return x


def record_cpu(mode: str = "screw", seed: int = 0, fast: bool = True, t_max: float = 8.0) -> dict:
    """One zero-residual drill episode on the CPU with the plant's inputs and outputs recorded per physics step."""
    from tactile_sim.behaviors.base_sources import DrillBase
    from tactile_sim.config import fast_config, residual_config
    from tactile_sim.model.plants.base import body_point_velocity
    from tactile_sim.model.plants.drill import STATES
    from tactile_sim.sim.randomize import apply_residual_dr, residual_dr
    from tactile_sim.tool_task import ToolTask

    cfg = residual_config(base=fast_config() if fast else None, task="drill")
    cfg = cfg.replace(drill={"mode": mode})
    dr = residual_dr(cfg, np.random.default_rng(seed))
    cfg = apply_residual_dr(cfg, dr)
    tt = ToolTask(cfg, seed=seed, t_max=t_max)
    tt.tb.residual.base_fn = DrillBase(tt.tb, tt.behavior, rng=np.random.default_rng(10_000 + seed))
    tt.tb.sensors.set_noise_scale(dr["_runtime"]["sensor_noise_scale"])
    plant = tt.tb.world.plant
    tap = _RngTap(plant.rng)
    plant.rng = tap
    recs: list[dict] = []
    orig = plant.pre_step
    buf = np.zeros(6)

    def tapped() -> None:
        d, m = plant.d, plant.m
        tip = d.site_xpos[plant.tip_site].copy()
        v = body_point_velocity(m, d, plant.tool, tip, buf).copy()
        r = {"t": float(d.time), "R": d.xmat[plant.tool].reshape(3, 3).copy(), "tip": tip, "v": v,
             "omega": buf[:3].copy(),
             "travel": float(d.qpos[plant.trigger_qadr]) if plant.trigger_qadr >= 0 else float("nan"),
             "level": float(plant.trigger)}
        n0, u0 = len(tap.normals), len(tap.uniforms)
        orig()
        r.update(normal=tap.normals[n0:], uniform=tap.uniforms[u0:], xfrc=d.xfrc_applied[plant.tool].copy(),
                 com=d.xipos[plant.tool].copy(), state=STATES.index(plant.state), depth=float(plant.depth),
                 w=float(plant.drive.w), clicks=int(plant.drive.clicks), tau_out=float(plant.tau_out),
                 slipping=bool(plant.drive.slipping))
        recs.append(r)

    plant.pre_step = tapped
    res = tt.run()
    return {"cfg": cfg, "dr": dr, "recs": recs, "summary": res.summary, "a": plant.a.copy(),
            "head0": plant.head0.copy(), "dt": float(plant.m.opt.timestep)}


def replay(rec: dict) -> dict[str, float]:
    """TorchDrillLaws on the recorded inputs (float64, CPU): the largest differences against the CPU plant."""
    cfg = rec["cfg"]
    laws = TorchDrillLaws(cfg.drill, 1, rec["a"], rec["head0"])
    laws.reset()
    f64 = torch.float64
    err = {"force": 0.0, "torque": 0.0, "depth": 0.0, "w": 0.0, "tau_out": 0.0}
    n_state = n_clicks = n_slip = 0
    for r in rec["recs"]:
        draws = {}
        if r["normal"]:
            draws["normal"] = torch.tensor([r["normal"][0]], dtype=f64)
        if r["uniform"]:
            draws["uniform"] = torch.tensor([r["uniform"][0]], dtype=f64)
        kw = ({"trigger_travel": torch.tensor([r["travel"]], dtype=f64)} if np.isfinite(r["travel"])
              else {"trigger": torch.tensor([r["level"]], dtype=f64)})
        parts = laws.step(torch.tensor([r["t"]], dtype=f64), rec["dt"], torch.as_tensor(r["R"])[None],
                          torch.as_tensor(r["tip"])[None], torch.as_tensor(r["v"])[None],
                          torch.as_tensor(r["omega"])[None], draws=draws, **kw)
        wr = TorchDrillLaws.wrench_about(torch.as_tensor(r["com"])[None], parts)[0].numpy()
        err["force"] = max(err["force"], float(np.abs(wr[:3] - r["xfrc"][:3]).max()))
        err["torque"] = max(err["torque"], float(np.abs(wr[3:] - r["xfrc"][3:]).max()))
        err["depth"] = max(err["depth"], abs(float(laws.depth[0]) - r["depth"]))
        err["w"] = max(err["w"], abs(float(laws.drive.w[0]) - r["w"]))
        err["tau_out"] = max(err["tau_out"], abs(float(laws.tau_out[0]) - r["tau_out"]))
        n_state += int(int(laws.state[0]) != r["state"])
        n_clicks += int(int(laws.drive.clicks[0]) != r["clicks"])
        n_slip += int(bool(laws.drive.slipping[0]) != r["slipping"])
    s = rec["summary"]
    return {"steps": len(rec["recs"]), **{f"max_abs_{k}": v for k, v in err.items()}, "state_mismatches": n_state,
            "click_mismatches": n_clicks, "slip_mismatches": n_slip, "final_state": rec["recs"][-1]["state"],
            "done": bool(s.get("done")), "t_done": float(s.get("t_done", float("nan"))),
            "peak_torque": float(s.get("peak_torque", float("nan"))), "clicks": rec["recs"][-1]["clicks"]}


# ---------------------------------------------------------------------------------------------- the dyno, batched
DT = 1e-4


def _drive(params: list, device="cpu") -> TorchDrivetrain:
    dr = TorchDrivetrain(params[0], len(params), device)
    dr.set_params(None, **{k: [float(getattr(p, k)) for p in params] for k in PARAMS if k != "clutch_lock"})
    dr.clutch_lock[:] = torch.as_tensor([bool(p.clutch_lock) for p in params])
    return dr


def no_load_speed(params: list, t: float = 1.0, dt: float = DT) -> np.ndarray:
    dr = _drive(params)
    one = torch.ones(len(params), dtype=torch.float64)
    for _ in range(int(t / dt)):
        dr.step(one, 0 * one, dt)
    return dr.w.numpy()


def joint_peak(params: list, test_torque: np.ndarray, angle_deg: float, dt: float = DT,
               t_max: float = 20.0) -> np.ndarray:
    dr = _drive(params)
    n = len(params)
    T = torch.as_tensor(test_torque, dtype=torch.float64)
    k = 0.9 * T / math.radians(angle_deg)
    th0 = 0.1 * T / k
    w0 = dr.free_speed.clone()
    dr.w[:] = w0
    dr.u[:] = 1.0
    one = torch.ones(n, dtype=torch.float64)
    peak = torch.zeros(n, dtype=torch.float64)
    going = torch.ones(n, dtype=torch.bool)
    for _ in range(int(t_max / dt)):
        dr.step(one, k * (th0 + dr.out_angle), dt)
        peak = torch.where(going, torch.maximum(peak, dr.tau_chuck), peak)
        going &= ~((dr.w <= 1e-3 * w0) | (dr.slipping & (dr.clicks > 3)))
        if not bool(going.any()):
            break
    return peak.numpy()


def iso5393(params: list, angle_deg: float, iters: int = 6) -> np.ndarray:
    T = np.array([p.stall_torque for p in params])
    for _ in range(iters):
        T = joint_peak(params, T, angle_deg)
    return T


def clutch_test(params: list, t: float = 1.0, dt: float = DT) -> dict[str, np.ndarray]:
    dr = _drive(params)
    n = len(params)
    one = torch.ones(n, dtype=torch.float64)
    steps = int(t / dt)
    tau, w = [], []
    clicks0 = None
    for i in range(steps):
        dr.step(one, 1e3 * one, dt)
        if i >= steps // 2:
            clicks0 = dr.clicks.clone() if clicks0 is None else clicks0
            tau.append(dr.tau_chuck.clone())
            w.append(dr.w.clone())
    tau, w = torch.stack(tau), torch.stack(w)
    t_half = (steps - steps // 2) * dt
    return {"mean_torque": tau.mean(0).numpy(), "peak_torque": tau.max(0).values.numpy(),
            "slip_speed": w.mean(0).numpy(), "click_rate": ((dr.clicks - clicks0).double() / t_half).numpy()}


def brake_stop_time(params: list, dt: float = DT) -> np.ndarray:
    dr = _drive(params)
    n = len(params)
    dr.w[:] = dr.free_speed
    zero = torch.zeros(n, dtype=torch.float64)
    t_stop = torch.full((n,), math.inf, dtype=torch.float64)
    for i in range(int(5.0 / dt)):
        dr.step(zero, zero, dt)
        t_stop = torch.where(torch.isinf(t_stop) & (dr.w < 0.01 * dr.free_speed), torch.full_like(t_stop, (i + 1) * dt),
                             t_stop)
        if not bool(torch.isinf(t_stop).any()):
            break
    return t_stop.numpy()


def dyno_table(params: list) -> dict[str, np.ndarray]:
    """The batched dyno's numbers per parameter set."""
    lock = [replace(p, clutch_lock=True) for p in params]
    ct = clutch_test(params)
    return {"no_load_speed": no_load_speed(params), "soft": iso5393(lock, 650.0), "hard": iso5393(lock, 27.0),
            "clutch_mean": ct["mean_torque"], "clutch_peak": ct["peak_torque"], "slip_speed": ct["slip_speed"],
            "click_rate": ct["click_rate"], "brake_stop": brake_stop_time(params)}


def cpu_dyno(p) -> dict[str, float]:
    from tactile_sim.model.plants import drill_dyno as dy

    ct = dy.clutch_test(p)
    return {"no_load_speed": dy.no_load_speed(p), "soft": dy.soft_torque(p), "hard": dy.hard_torque(p),
            "clutch_mean": ct["mean_torque"], "clutch_peak": ct["peak_torque"], "slip_speed": ct["slip_speed"],
            "click_rate": ct["click_rate"], "brake_stop": dy.brake_stop_time(p)}


def dyno_params(draws: int = 8, seed: int = 0) -> list:
    """The nominal 12 V drill and `draws` of the residual randomization's drill draws."""
    from tactile_sim.config import residual_config
    from tactile_sim.model.plants.drivetrain import DrivetrainParams
    from tactile_sim.sim.randomize import apply_residual_dr, residual_dr

    cfg = residual_config(task="drill")
    out = [DrivetrainParams.from_cfg(cfg.drill)]
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        c = apply_residual_dr(cfg, residual_dr(cfg, rng))
        out.append(DrivetrainParams.from_cfg(c.drill))
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - report script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default="screw,hole")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--fast", action="store_true", help="4 kHz physics (else the 8 kHz validator's)")
    ap.add_argument("--t-max", type=float, default=8.0)
    ap.add_argument("--dyno-draws", type=int, default=8)
    a = ap.parse_args(argv)
    print("| episode | steps | force | torque | depth | spindle speed | chuck torque | state / click / slip "
          "mismatches | final state | seated or through at | peak torque | clicks |")
    print("|---" * 12 + "|")
    from tactile_sim.model.plants.drill import STATES

    for mode in a.modes.split(","):
        for seed in (int(s) for s in a.seeds.split(",")):
            r = replay(record_cpu(mode, seed, a.fast, a.t_max))
            print(f"| {mode} seed {seed} | {r['steps']} | {r['max_abs_force']:.1e} N | {r['max_abs_torque']:.1e} Nm | "
                  f"{r['max_abs_depth']:.1e} m | {r['max_abs_w']:.1e} rad/s | {r['max_abs_tau_out']:.1e} Nm | "
                  f"{r['state_mismatches']} / {r['click_mismatches']} / {r['slip_mismatches']} | "
                  f"{STATES[r['final_state']]} | {r['t_done']:.2f} s | {r['peak_torque']:.2f} Nm | {r['clicks']} |",
                  flush=True)
    params = dyno_params(a.dyno_draws)
    gpu = dyno_table(params)
    keys = list(gpu)
    print("\n| set | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    worst = {k: 0.0 for k in keys}
    for i, p in enumerate(params):
        c = cpu_dyno(p)
        cells = []
        for k in keys:
            g = float(gpu[k][i])
            rel = abs(g - c[k]) / max(abs(c[k]), 1e-9)
            worst[k] = max(worst[k], rel)
            cells.append(f"{g:.4g} ({c[k]:.4g})")
        print(f"| {'nominal' if i == 0 else f'draw {i}'} | " + " | ".join(cells) + " |", flush=True)
    print("\nlargest relative difference, batched against the CPU dyno: "
          + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
