"""The GPU taxel load model (tactile_rl.sensors.taxel_load) against the testbed's elastic foundation
(HandIO.skin_taxels / _foundation) on recorded hold and strike states of the hammer scene.

States come from a CPU hammering episode (one_step.sample_strike_states: the hold before the first windup, the 2 ms
before each blow and the face/nail impacts). Per state, the CPU's per-taxel loads with a freshly computed shape are the
reference. Two GPU numbers:

- `model`: the GPU model fed the CPU's own contacts, patch poses and tool poses: the load model alone (float32
  arithmetic; analytic capsule rays against MuJoCo's mju_rayGeom, or the hull planes' rays against mj_rayMesh);
- `end_to_end`: the GPU model on MuJoCo Warp's contacts after a forward pass from the same state: what the GPU env's
  skin sees, the simulators' contact differences included (see P1).

Per state and patch with load on either side (> 0.5 N): the patch-sum error |sum_gpu - sum_cpu| / max(sums), the shape
error sum |f_gpu - f_cpu| / max(sums) (L1, of the taxel distribution) and the centre-of-pressure distance (taxel
pitches); distributions over the states, by label.

    python -m tactile_rl.parity.taxel_load
"""

from __future__ import annotations

import json

import mujoco
import numpy as np


def cpu_reference(world, states) -> dict:
    """The CPU's taxel loads (fresh foundation shape), contacts and poses at each state."""
    m, d = world.model, world.data
    hand = world.hand
    patches = hand.patches
    P = len(patches)
    K = 16
    tool = set(int(g) for g in hand.tool_geoms)
    ray = sorted(set().union(*hand._pair_tool.values()))  # the ray geoms, in TaxelLoadModel's order
    geom_patch = {}
    for k, p in enumerate(patches):
        for g in p.geoms:
            geom_patch[int(g)] = k
    n = len(states)
    T = sum(len(p.taxel_pos) for p in patches)
    G = len(ray)
    out = {"loads": np.zeros((n, T)), "f": np.zeros((n, P, K)), "pos": np.zeros((n, P, K, 3)),
           "occ": np.zeros((n, P, K), bool), "tg": -np.ones((n, P, K), int), "xpos": np.zeros((n, P, 3)),
           "xmat": np.zeros((n, P, 3, 3)), "cpos": np.zeros((n, G, 3)), "cmat": np.zeros((n, G, 3, 3)),
           "overflow": 0}
    saved = {k: getattr(m, k).copy() for k in states.model}
    f6 = np.zeros(6)
    from tactile_rl.parity.one_step import STATE_FIELDS

    for i in range(n):
        mujoco.mj_resetData(m, d)
        for k in STATE_FIELDS:
            getattr(d, k)[:] = getattr(states, k)[i]
        for k, v in states.model.items():
            getattr(m, k)[:] = v[i]
        mujoco.mj_forward(m, d)
        hand._skin_key, hand._shape = None, {}
        loads = hand.skin_taxels()
        out["loads"][i] = np.concatenate([x if x is not None else np.zeros(len(p.taxel_pos))
                                          for x, p in zip(loads, patches, strict=True)])
        slot = np.zeros(P, int)
        for ci in range(d.ncon):
            c = d.contact[ci]
            if c.efc_address < 0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in tool) == (g2 in tool):
                continue
            k = geom_patch.get(g2 if g1 in tool else g1)
            if k is None:
                continue
            mujoco.mj_contactForce(m, d, ci, f6)
            if f6[0] <= 0:
                continue
            if slot[k] >= K:
                out["overflow"] += 1
                continue
            out["f"][i, k, slot[k]] = f6[0]
            out["pos"][i, k, slot[k]] = c.pos
            out["occ"][i, k, slot[k]] = True
            tg = g1 if g1 in tool else g2
            out["tg"][i, k, slot[k]] = ray.index(tg) if tg in ray else -1
            slot[k] += 1
        for k, p in enumerate(patches):
            out["xpos"][i, k] = d.xpos[p.body]
            out["xmat"][i, k] = d.xmat[p.body].reshape(3, 3)
        out["cpos"][i] = d.geom_xpos[ray]
        out["cmat"][i] = d.geom_xmat[ray].reshape(G, 3, 3)
    for k, v in saved.items():
        getattr(m, k)[:] = v
    return out


def gpu_model_on_cpu_contacts(parts, mj_model, ref: dict, device: str) -> np.ndarray:
    import torch

    from tactile_rl.sensors.taxel_load import TaxelLoadModel

    n = ref["f"].shape[0]
    lm = TaxelLoadModel(parts, mj_model, n, device, prefix="", max_per_patch=ref["f"].shape[2])

    def t(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=device)

    out = lm.loads(t(ref["f"]), t(ref["pos"]), t(ref["occ"], torch.bool), t(ref["xpos"]), t(ref["xmat"]),
                   t(ref["cpos"]), t(ref["cmat"]), torch.zeros(n, dtype=torch.float64, device=device),
                   t(ref["tg"], torch.long))
    return out.cpu().numpy().astype(float)


def gpu_end_to_end(states, device: str, multiccd: bool = True) -> tuple[np.ndarray, dict]:
    """Load the states into GPU worlds (the hammer scene), forward, and run the GPU load model on MuJoCo Warp's
    contacts."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.parity.one_step import STATE_FIELDS
    from tactile_rl.scene.entities import ENTITY
    from tactile_rl.sensors.taxel_load import TaxelLoadModel
    from tactile_rl.tasks.hammer import hammer_scene_env_cfg, hammer_scene_key
    from tactile_sim.sim.training_parts import training_parts

    s = states.float32()
    n = len(s)
    cfg = hammer_scene_env_cfg(n)
    if not multiccd:
        cfg.sim.mujoco.disableflags = (*cfg.sim.mujoco.disableflags, "multiccd")
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        sim, d = env.sim, env.sim.data
        for k in STATE_FIELDS:
            if k == "eq_active":
                d.eq_active[:] = torch.as_tensor(s.eq_active, dtype=torch.bool, device=device)
            else:
                getattr(d, k)[:] = torch.as_tensor(getattr(s, k), dtype=torch.float32, device=device)
        for k, v in s.model.items():
            getattr(sim.model, k)[:] = torch.as_tensor(v, dtype=torch.float32, device=device)
        sim.forward()
        lm = TaxelLoadModel(training_parts(hammer_scene_key()), sim.mj_model, n, device, prefix=f"{ENTITY}/")
        out = lm.step(d, torch.zeros(n, dtype=torch.float64, device=device))
        torch.cuda.synchronize()
        info = {"overflow": int(lm.overflow), "warp_overflow": int(d.overflow.max())}
        return out.cpu().numpy().astype(float), info
    finally:
        env.close()


def compare(ref: np.ndarray, other: np.ndarray, parts, labels: list[str], f_min: float = 0.5) -> dict:
    """Per (state, patch) with load: patch-sum error, shape error (L1 of the distribution) and CoP distance."""
    skin = parts.skin
    starts = np.concatenate([[0], np.cumsum([len(p.taxel_pos) for p in skin])])
    rows = {"sum": [], "shape": [], "cop": [], "label": [], "patch": []}
    for k, p in enumerate(skin):
        sl = slice(starts[k], starts[k + 1])
        a, b = ref[:, sl], other[:, sl]
        sa, sb = a.sum(1), b.sum(1)
        on = np.maximum(sa, sb) > f_min
        rr, cc = np.divmod(np.arange(p.grid[0] * p.grid[1]), p.grid[1])
        for i in np.flatnonzero(on):
            m_ = max(sa[i], sb[i])
            rows["sum"].append(abs(sb[i] - sa[i]) / m_)
            rows["shape"].append(np.abs(b[i] - a[i]).sum() / m_)
            if sa[i] > 0 and sb[i] > 0:
                ca = np.array([(a[i] * rr).sum(), (a[i] * cc).sum()]) / sa[i]
                cb = np.array([(b[i] * rr).sum(), (b[i] * cc).sum()]) / sb[i]
                rows["cop"].append(float(np.linalg.norm(cb - ca)))
            else:
                rows["cop"].append(np.nan)
            rows["label"].append(labels[i])
            rows["patch"].append(p.name)
    out = {}
    lab = np.array(rows["label"])
    for name in sorted(set(labels)) + ["all"]:
        sel = np.ones(len(lab), bool) if name == "all" else lab == name
        if not sel.any():
            continue
        r = {"patches": int(sel.sum())}
        for key in ("sum", "shape", "cop"):
            x = np.asarray(rows[key], float)[sel]
            x = x[np.isfinite(x)]
            if len(x):
                r[f"{key}_median"], r[f"{key}_p90"], r[f"{key}_p99"] = (float(np.median(x)),
                                                                          float(np.percentile(x, 90)),
                                                                          float(np.percentile(x, 99)))
        r["sum_under_5pct"] = float(np.mean(np.asarray(rows["sum"], float)[sel] < 0.05))
        out[name] = r
    return out


def run(n_hold: int = 60, n_strike: int = 240, seed: int = 0, device: str = "cuda:0") -> dict:
    """The comparison with multi-contact on (the testbed's default) and off on both sides: MuJoCo Warp gives a
    capsule/mesh pair (the handle against a hand link) at most one contact, CPU MuJoCo several."""
    from tactile_rl.parity.one_step import sample_strike_states
    from tactile_sim.sim.training_parts import residual_training_config, training_parts

    cfg = residual_training_config()
    states, world = sample_strike_states(cfg, n_hold, n_strike, seed=seed)
    states = states.float32()
    parts = training_parts(cfg)
    labels = list(states.labels)
    out = {"states": len(states), "labels": {k: labels.count(k) for k in set(labels)}}
    m = world.model
    for multiccd in (True, False):
        tag = "multiccd_on" if multiccd else "multiccd_off"
        if not multiccd:
            m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
        ref = cpu_reference(world, states)
        model_only = gpu_model_on_cpu_contacts(parts, m, ref, device)
        e2e, info = gpu_end_to_end(states, device, multiccd)
        out[tag] = {"model": compare(ref["loads"], model_only, parts, labels),
                    "end_to_end": compare(ref["loads"], e2e, parts, labels),
                    "info": {**info, "cpu_overflow": ref["overflow"]}}
    m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    return out


def main() -> None:  # pragma: no cover - report script
    print(json.dumps(run(), indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
