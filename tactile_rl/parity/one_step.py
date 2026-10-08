"""P1 of the CPU/GPU parity ladder: one physics step from identical states in CPU MuJoCo and MuJoCo Warp (both without
noslip), on the same reduced-scene model.

States are sampled along CPU runs of the reduced scene (World, training configuration): the welded settle, the free
hold, and a perturbed hold (random wrenches on the hammer, random residual MIT commands on the hand, random arm
targets). They are rounded to float32 (the GPU state's precision) and both sides start from the rounded state; each
state gets its own GPU world, and one forward pass and one step run in all of them at once.

Compared, per state:

- the smooth dynamics (bias, passive and actuator forces, unconstrained acceleration): relative error;
- qacc: `||dqacc||_M / ||qacc_smooth||_M`, the acceleration error in the kinetic-energy norm relative to the
  unconstrained acceleration the constraints balance. (A plain ||dqacc|| / ||qacc|| is meaningless in a static grasp,
  where qacc is ~0 and a finger joint's 1e-4 kg m^2 turns any force noise into rad/s^2; it is reported too);
- contacts: the contact count per geom pair, and the normal force per geom pair (summed over its contacts),
  |df| / max(f_cpu, f_gpu) over pairs carrying at least `f_min`, plus the force on pairs only one side found;
- after the step: qpos and qvel.

The same metrics between the CPU and itself, with the state nudged by one float32 ulp (`noise_floor`), measure how
ill-conditioned the grasp is: GPU-vs-CPU differences at that level are float32 arithmetic, not model differences.

    python -m tactile_rl.parity.one_step      # prints the report (multi-contact on, as the testbed runs, and off)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.config import SimConfig
from tactile_sim.control.interface import HandCommand
from tactile_sim.control.obs_spec import ActionSpec, JointParams, compose, offsets

STATE_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "eq_active", "xfrc_applied")


@dataclass
class States:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    qacc_warmstart: np.ndarray
    eq_active: np.ndarray
    xfrc_applied: np.ndarray
    labels: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return self.qpos.shape[0]

    def subset(self, idx) -> States:
        return States(*(getattr(self, k)[idx] for k in STATE_FIELDS), labels=[self.labels[i] for i in idx])

    def float32(self) -> States:
        return replace(self, **{k: getattr(self, k).astype(np.float32).astype(np.float64) for k in STATE_FIELDS
                                if k != "eq_active"})


def sample_cpu_states(cfg: SimConfig, n_weld: int = 40, n_hold: int = 60, n_perturbed: int = 100,
                      seed: int = 0) -> tuple[States, mujoco.MjModel]:
    """States along CPU runs of the reduced scene (see module doc); also returns the CPU model."""
    from tactile_sim.sim.world import World, full_inertia

    rng = np.random.default_rng(seed)
    w = World(cfg)
    m, d = w.model, w.data
    rec: dict[str, list] = {k: [] for k in STATE_FIELDS}
    labels: list[str] = []
    arm_off = np.zeros(len(w.arm_act))

    def run(n_steps: int, n_samples: int, label: str, before=None) -> None:
        every = max(1, n_steps // max(n_samples, 1))
        for i in range(n_steps):
            if before is not None:
                before(i)
            w.hand.tick(w.t)
            w.set_arm_position(w.q_hover + arm_off, droop=w._hover_droop)  # World._hold_arm, plus any offset
            w.step()
            if (i + 1) % every == 0 and labels.count(label) < n_samples:
                for k in rec:
                    rec[k].append(getattr(d, k).copy())
                labels.append(label)

    # the welded settle (World.settle's first phase), then the free hold
    w.reset_pose()
    hold = cfg.controller.grip_hold
    n_w = int(round(0.15 / w.dt))
    run(n_w, n_weld, "weld", before=lambda i: w.set_grip_force(hold * min(1.0, 2.0 * i / n_w)))
    d.eq_active[w.weld_id] = 0
    w.set_grip_force(hold)
    run(int(round(0.25 / w.dt)), 0, "seat")
    run(int(round(0.3 / w.dt)), n_hold, "hold")
    # perturbed: wrenches on the hammer, residual-style MIT commands on the hand, arm target offsets
    hand = w.hand
    spec = ActionSpec()
    mujoco.mj_forward(m, d)
    i_eff = np.diag(full_inertia(m, d, np.zeros((m.nv, m.nv))))[hand.dofs]
    jp = JointParams.for_joints(hand.joint_names, hand.tau_max, cfg.wuji.kp_nom, i_eff)
    hb = m.body(names.HAMMER_BODY).id

    def perturb(i: int) -> None:
        if i % int(round(0.02 / w.dt)) == 0:
            d.xfrc_applied[hb, :3] = rng.uniform(-15.0, 15.0, 3)
            d.xfrc_applied[hb, 3:] = rng.uniform(-0.5, 0.5, 3)
            a = rng.uniform(-0.6, 0.6, 3 * len(hand.joint_names))
            base = np.stack(hand.synergy.mit_command(hand.s, d.qpos[hand.qadr]))
            hand.set_hand_command(HandCommand.from_array(w.t, compose(spec, jp, base, offsets(spec, jp, a))))
            arm_off[:] = rng.uniform(-0.02, 0.02, len(arm_off))

    try:
        run(int(round(0.4 / w.dt)), n_perturbed, "perturbed", before=perturb)
    finally:
        hand.set_hand_command(None)
        d.xfrc_applied[:] = 0.0
    return States(**{k: np.array(v) for k, v in rec.items()}, labels=labels), m


def nudge(s: States, rng: np.random.Generator, ulps: float = 1.0) -> States:
    """The states moved by about `ulps` float32 ulp in position and velocity (the CPU's own noise floor)."""
    eps = ulps * np.finfo(np.float32).eps
    q = s.qpos * (1.0 + rng.uniform(-eps, eps, s.qpos.shape))
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)  # the hammer's quaternion (its free joint leads)
    v = s.qvel * (1.0 + rng.uniform(-eps, eps, s.qvel.shape))
    return replace(s, qpos=q, qvel=v)


@dataclass
class StepResult:
    qacc: np.ndarray  # (n, nv)
    smooth: dict[str, np.ndarray]  # qfrc_bias, qfrc_passive, actuator_force, qacc_smooth: (n, ...)
    pair_force: list[dict[tuple[int, int], float]]  # per state: (geom1, geom2) -> summed normal force
    pair_count: list[dict[tuple[int, int], int]]
    qpos1: np.ndarray  # after one step
    qvel1: np.ndarray
    nefc: np.ndarray
    M: np.ndarray | None = None  # (n, nv, nv) CPU joint-space inertia


SMOOTH = ("qfrc_bias", "qfrc_passive", "actuator_force", "qacc_smooth")


def _pair(g1: int, g2: int) -> tuple[int, int]:
    return (g1, g2) if g1 <= g2 else (g2, g1)


def cpu_one_step(m: mujoco.MjModel, s: States) -> StepResult:
    from tactile_sim.sim.world import full_inertia

    qacc, sm, pf, pc, q1, v1, nefc, Ms = [], {k: [] for k in SMOOTH}, [], [], [], [], [], []
    d = mujoco.MjData(m)
    for i in range(len(s)):
        mujoco.mj_resetData(m, d)
        for k in STATE_FIELDS:
            getattr(d, k)[:] = getattr(s, k)[i]
        mujoco.mj_forward(m, d)
        qacc.append(d.qacc.copy())
        for k in SMOOTH:
            sm[k].append(getattr(d, k).copy())
        Ms.append(full_inertia(m, d, np.zeros((m.nv, m.nv))).copy())
        forces: dict[tuple[int, int], float] = {}
        counts: dict[tuple[int, int], int] = {}
        for c in d.contact[:d.ncon]:
            if c.efc_address < 0:
                continue
            key = _pair(int(c.geom1), int(c.geom2))
            forces[key] = forces.get(key, 0.0) + float(d.efc_force[c.efc_address])
            counts[key] = counts.get(key, 0) + 1
        pf.append(forces)
        pc.append(counts)
        nefc.append(d.nefc)
        mujoco.mj_step(m, d)
        q1.append(d.qpos.copy())
        v1.append(d.qvel.copy())
    return StepResult(np.array(qacc), {k: np.array(v) for k, v in sm.items()}, pf, pc, np.array(q1), np.array(v1),
                      np.array(nefc), np.array(Ms))


def gpu_one_step(env, s: States) -> StepResult:
    """Load the states into the env's worlds (one each), run one forward pass and one step."""
    import torch

    sim = env.sim
    d = sim.data
    n = len(s)
    if env.num_envs != n:
        raise ValueError(f"the env has {env.num_envs} worlds for {n} states")
    dev = env.device

    def load():
        for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "xfrc_applied"):
            getattr(d, k)[:] = torch.as_tensor(getattr(s, k), dtype=torch.float32, device=dev)
        d.eq_active[:] = torch.as_tensor(s.eq_active, dtype=torch.bool, device=dev)
        d.time[:] = 0.0

    load()
    sim.forward()
    torch.cuda.synchronize()
    qacc = d.qacc.cpu().numpy().astype(float)
    smooth = {k: getattr(d, k).cpu().numpy().astype(float) for k in SMOOTH}
    nacon = int(d.nacon.cpu().numpy()[0])
    geom = d.contact.geom.cpu().numpy()[:nacon]
    world = d.contact.worldid.cpu().numpy()[:nacon]
    adr = d.contact.efc_address.cpu().numpy()[:nacon, 0]
    frc = d.efc.force.cpu().numpy()
    pf: list[dict[tuple[int, int], float]] = [{} for _ in range(n)]
    pc: list[dict[tuple[int, int], int]] = [{} for _ in range(n)]
    for (g1, g2), wid, a in zip(geom, world, adr, strict=True):
        if a < 0:
            continue
        key = _pair(int(g1), int(g2))
        pf[wid][key] = pf[wid].get(key, 0.0) + float(frc[wid, a])
        pc[wid][key] = pc[wid].get(key, 0) + 1
    nefc = d.nefc.cpu().numpy().copy()
    if int(d.overflow.cpu().numpy().max()) != 0:
        raise RuntimeError("MuJoCo Warp buffer overflow during the parity step: raise nconmax / njmax")
    load()
    sim.step()
    torch.cuda.synchronize()
    return StepResult(qacc, smooth, pf, pc, d.qpos.cpu().numpy().astype(float), d.qvel.cpu().numpy().astype(float),
                      nefc)


def _pct(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(x)), float(np.percentile(x, 99)), float(x.max())


def compare(ref: StepResult, other: StepResult, f_min: float = 0.1) -> dict:
    """Metrics of `other` against `ref` (the CPU; it carries the inertia matrices)."""
    n = ref.qacc.shape[0]
    dq = other.qacc - ref.qacc
    eM = np.array([np.sqrt(dq[i] @ ref.M[i] @ dq[i]) /
                   np.sqrt(ref.smooth["qacc_smooth"][i] @ ref.M[i] @ ref.smooth["qacc_smooth"][i]) for i in range(n)])
    e_raw = np.linalg.norm(dq, axis=1) / np.maximum(np.linalg.norm(ref.qacc, axis=1), 1e-12)
    smooth = {k: np.linalg.norm(other.smooth[k] - ref.smooth[k], axis=1) /
              np.maximum(np.linalg.norm(ref.smooth[k], axis=1), 1e-12) for k in SMOOTH}
    rel, rel_matched, only, total = [], [], 0.0, 0.0
    same = np.array([ref.pair_count[i] == other.pair_count[i] for i in range(n)])
    for i in range(n):
        for k in set(ref.pair_force[i]) | set(other.pair_force[i]):
            fc, fg = ref.pair_force[i].get(k), other.pair_force[i].get(k)
            total += max(fc or 0.0, fg or 0.0)
            if fc is None or fg is None:
                only += fc or fg or 0.0
                rel.append(1.0)
                continue
            if max(fc, fg) >= f_min:
                r = abs(fg - fc) / max(fc, fg)
                rel.append(r)
                if same[i]:
                    rel_matched.append(r)
    out = {"states": n, "same_contacts": int(same.sum())}
    for name, x in (("qacc_M", eM), ("qacc_M_same", eM[same]), ("qacc_raw", e_raw), ("force", np.array(rel)),
                    ("force_same", np.array(rel_matched))):
        out[f"{name}_median"], out[f"{name}_p99"], out[f"{name}_max"] = _pct(x)
    for k, x in smooth.items():
        out[f"{k}_p99"] = _pct(x)[1]
    out["force_unmatched_frac"] = only / max(total, 1e-12)
    tally: dict[tuple[int, int, int, int], int] = {}  # (geom1, geom2, ref count, other count) -> states
    for i in range(n):
        for k in set(ref.pair_count[i]) | set(other.pair_count[i]):
            a, b = ref.pair_count[i].get(k, 0), other.pair_count[i].get(k, 0)
            if a != b:
                tally[(*k, a, b)] = tally.get((*k, a, b), 0) + 1
    out["count_mismatches"] = dict(sorted(tally.items(), key=lambda kv: -kv[1]))
    out["dqpos_max"] = float(np.abs(other.qpos1 - ref.qpos1).max())
    out["dqvel_max"] = float(np.abs(other.qvel1 - ref.qvel1).max())
    out["nefc_max"] = int(max(ref.nefc.max(), other.nefc.max()))
    return out


def run_p1(n_weld: int = 40, n_hold: int = 60, n_perturbed: int = 100, seed: int = 0, multiccd: bool = True,
           device: str = "cuda:0") -> dict:
    """Sample, step on both sides, compare. Returns {"gpu": metrics, "floor": CPU-vs-nudged-CPU metrics,
    "by_label": {label: metrics}}. `multiccd` False disables MuJoCo's multi-contact convex collisions on both sides."""
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.tasks.hold import hold_env_cfg
    from tactile_sim.sim.training_parts import config_from_key

    n = n_weld + n_hold + n_perturbed
    cfg = hold_env_cfg(num_envs=n, nconmax=128, njmax=512)
    if not multiccd:
        cfg.sim.mujoco.disableflags = (*cfg.sim.mujoco.disableflags, "multiccd")
    sim_cfg = config_from_key(cfg.actions["residual"].scene_key)
    states, m = sample_cpu_states(sim_cfg, n_weld, n_hold, n_perturbed, seed)
    states = states.float32()
    if not multiccd:
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        cpu = cpu_one_step(m, states)
        gpu = gpu_one_step(env, states)
    finally:
        env.close()
    nudged = cpu_one_step(m, nudge(states, np.random.default_rng(seed + 1)))
    by_label = {}
    for lab in dict.fromkeys(states.labels):
        idx = [i for i, x in enumerate(states.labels) if x == lab]

        def sub(r: StepResult, idx=idx) -> StepResult:
            return StepResult(r.qacc[idx], {k: v[idx] for k, v in r.smooth.items()}, [r.pair_force[i] for i in idx],
                              [r.pair_count[i] for i in idx], r.qpos1[idx], r.qvel1[idx], r.nefc[idx],
                              None if r.M is None else r.M[idx])

        by_label[lab] = compare(sub(cpu), sub(gpu))
    return {"gpu": compare(cpu, gpu), "floor": compare(cpu, nudged), "by_label": by_label}


def format_report(r: dict, title: str, model: mujoco.MjModel | None = None) -> str:
    g, f = r["gpu"], r["floor"]
    lines = [f"{title}: {g['states']} states, {g['same_contacts']} with the same contacts per pair on both sides "
             f"(CPU vs nudged CPU: {f['same_contacts']})"]
    for key, label in (("qacc_M", "qacc, M-norm rel"), ("qacc_M_same", "  same contacts"),
                       ("qacc_raw", "qacc, plain rel"), ("force", "pair normal force rel"),
                       ("force_same", "  same contacts")):
        lines.append(f"  {label:24s} GPU median {g[key + '_median']:.2e} p99 {g[key + '_p99']:.2e} max "
                     f"{g[key + '_max']:.2e} | floor median {f[key + '_median']:.2e} p99 {f[key + '_p99']:.2e}")
    lines.append(f"  force on pairs one side only: GPU {g['force_unmatched_frac']:.2e}, floor "
                 f"{f['force_unmatched_frac']:.2e}")
    lines.append("  smooth dynamics p99 rel: " + ", ".join(f"{k} {g[k + '_p99']:.1e}" for k in SMOOTH))
    lines.append(f"  after one step: max |dqpos| {g['dqpos_max']:.2e}, max |dqvel| {g['dqvel_max']:.2e} "
                 f"(floor {f['dqpos_max']:.2e}, {f['dqvel_max']:.2e}); nefc max {g['nefc_max']}")
    for lab, x in r["by_label"].items():
        lines.append(f"  [{lab}] same contacts {x['same_contacts']}/{x['states']}; qacc M median "
                     f"{x['qacc_M_median']:.2e} p99 {x['qacc_M_p99']:.2e}; force median {x['force_median']:.2e} p99 "
                     f"{x['force_p99']:.2e}")

    def gname(i: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) if model is not None else str(i)

    worst = list(g["count_mismatches"].items())[:5]
    if worst:
        lines.append("  contact-count differences (geom pair: CPU -> GPU contacts, states): " + "; ".join(
            f"{gname(a)}/{gname(b)}: {ca}->{cb}, {k}" for (a, b, ca, cb), k in worst))
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - report script
    from tactile_sim.sim.training_parts import training_config
    from tactile_sim.sim.world import World

    model = World(training_config()).model
    for multiccd in (True, False):
        r = run_p1(multiccd=multiccd)
        print(format_report(r, f"P1 one-step parity, multi-contact {'on (as the testbed)' if multiccd else 'off'}",
                            model))


if __name__ == "__main__":  # pragma: no cover
    main()
