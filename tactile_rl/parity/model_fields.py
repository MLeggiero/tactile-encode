"""P0 of the CPU/GPU parity ladder: the model mjlab compiles (and the MuJoCo Warp model built from it) against the CPU
testbed's model of the same reduced scene, field by field: options, bodies (mass, inertia), joints and dofs (ranges,
armature, damping, friction loss, limit softness), geoms, meshes, contact pairs, actuators and equalities.

Elements are matched by index and must carry the same names (the GPU's with the entity prefix). Sites and sensors
are not compared: the training model drops the testbed's sensors and taxel marker sites on purpose (neither enters
the dynamics). Usage:

    python -m tactile_rl.parity.model_fields   # against the training (noslip 0) and validator (noslip 5) models
"""

from __future__ import annotations

import mujoco
import numpy as np

PREFIX = "robot/"

OPT_FIELDS = ("timestep", "integrator", "cone", "impratio", "iterations", "tolerance", "ls_iterations", "ls_tolerance",
              "ccd_iterations", "ccd_tolerance", "noslip_iterations", "noslip_tolerance", "gravity", "jacobian",
              "solver", "disableflags", "enableflags", "o_margin", "o_solref", "o_solimp", "o_friction")
SIZES = ("nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nmesh", "npair", "nexclude", "neq", "ntendon")
# (element kind, name accessor, fields)
ARRAYS = {
    "body": ("nbody", mujoco.mjtObj.mjOBJ_BODY, ("body_parentid", "body_mass", "body_inertia", "body_ipos",
                                                 "body_iquat", "body_pos", "body_quat", "body_gravcomp",
                                                 "body_jntnum", "body_dofnum", "body_geomnum")),
    "joint": ("njnt", mujoco.mjtObj.mjOBJ_JOINT, ("jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr", "jnt_range",
                                                  "jnt_limited", "jnt_solref", "jnt_solimp", "jnt_margin",
                                                  "jnt_stiffness", "jnt_pos", "jnt_axis", "jnt_actfrclimited",
                                                  "jnt_actfrcrange")),
    "dof": ("nv", None, ("dof_armature", "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp",
                         "dof_jntid", "dof_bodyid", "dof_parentid")),
    "geom": ("ngeom", mujoco.mjtObj.mjOBJ_GEOM, ("geom_type", "geom_bodyid", "geom_dataid", "geom_size", "geom_pos",
                                                 "geom_quat", "geom_contype", "geom_conaffinity", "geom_condim",
                                                 "geom_priority", "geom_friction", "geom_solref", "geom_solimp",
                                                 "geom_solmix", "geom_margin", "geom_gap", "geom_rbound")),
    "mesh": ("nmesh", mujoco.mjtObj.mjOBJ_MESH, ("mesh_vertnum", "mesh_facenum", "mesh_pos", "mesh_quat")),
    "pair": ("npair", mujoco.mjtObj.mjOBJ_PAIR, ("pair_dim", "pair_geom1", "pair_geom2", "pair_friction",
                                                 "pair_solref", "pair_solreffriction", "pair_solimp", "pair_margin",
                                                 "pair_gap")),
    "actuator": ("nu", mujoco.mjtObj.mjOBJ_ACTUATOR, ("actuator_trntype", "actuator_trnid", "actuator_dyntype",
                                                      "actuator_gaintype", "actuator_biastype", "actuator_gainprm",
                                                      "actuator_biasprm", "actuator_dynprm", "actuator_ctrlrange",
                                                      "actuator_forcerange", "actuator_ctrllimited",
                                                      "actuator_forcelimited", "actuator_gear",
                                                      "actuator_actlimited")),
    "equality": ("neq", mujoco.mjtObj.mjOBJ_EQUALITY, ("eq_type", "eq_obj1id", "eq_obj2id", "eq_active0",
                                                       "eq_solref", "eq_solimp")),
}
# MuJoCo Warp model arrays checked against the CPU model (float32 storage: compared to 1e-6 relative)
WARP_FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "jnt_range",
               "jnt_solref", "jnt_solimp", "dof_armature", "dof_damping", "dof_frictionloss", "geom_size",
               "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "pair_dim", "pair_friction",
               "pair_solref", "pair_solreffriction", "pair_solimp", "pair_margin", "pair_gap", "actuator_gainprm",
               "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "actuator_gear", "eq_solref",
               "eq_solimp", "qpos0")


def _names(m: mujoco.MjModel, obj, n: int) -> list[str]:
    return [mujoco.mj_id2name(m, obj, i) or "" for i in range(n)]


def compare_models(cpu: mujoco.MjModel, gpu: mujoco.MjModel, rtol: float = 0.0, atol: float = 0.0) -> list[str]:
    """Differences between the CPU model and the mjlab-compiled model (empty: they agree)."""
    out: list[str] = []
    for f in OPT_FIELDS:
        a, b = np.asarray(getattr(cpu.opt, f)), np.asarray(getattr(gpu.opt, f))
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            out.append(f"opt.{f}: cpu {a.tolist()} gpu {b.tolist()}")
    for s in SIZES:
        if getattr(cpu, s) != getattr(gpu, s):
            out.append(f"{s}: cpu {getattr(cpu, s)} gpu {getattr(gpu, s)}")
    if out and any(not o.startswith("opt.") for o in out):
        return out  # sizes differ: element-wise comparison is meaningless
    for kind, (count, obj, flds) in ARRAYS.items():
        n = getattr(cpu, count)
        if obj is not None:
            cn, gn = _names(cpu, obj, n), _names(gpu, obj, n)
            first = 1 if kind == "body" else 0  # the world body keeps its name
            bad = [i for i in range(first, n) if (cn[i] or gn[i]) and PREFIX + cn[i] != gn[i]]
            if bad:
                out.append(f"{kind} names differ at {bad[:5]}: {[(cn[i], gn[i]) for i in bad[:3]]}")
        for f in flds:
            a, b = np.asarray(getattr(cpu, f)), np.asarray(getattr(gpu, f))
            if a.shape != b.shape:
                out.append(f"{f}: shape cpu {a.shape} gpu {b.shape}")
            elif not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                i = np.unravel_index(int(np.argmax(np.abs(a - b))), a.shape)
                out.append(f"{f}: max |diff| {np.max(np.abs(a - b)):.3g} at {i}: cpu {a[i]} gpu {b[i]}")
    for f in ("mesh_vert", "mesh_face", "qpos0", "qpos_spring"):
        a, b = np.asarray(getattr(cpu, f)), np.asarray(getattr(gpu, f))
        if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol):
            out.append(f"{f}: differs")
    return out


def warp_array(wp_model, field: str) -> np.ndarray:
    a = getattr(wp_model, field)
    x = a.numpy()
    return x[0] if x.ndim >= 1 and x.shape[0] == 1 and field != "qpos0" else x


def compare_warp(cpu: mujoco.MjModel, wp_model, rtol: float = 1e-6, atol: float = 1e-7) -> list[str]:
    """Differences between the CPU model and the MuJoCo Warp model's arrays (float32)."""
    out = []
    for f in WARP_FIELDS:
        a = np.asarray(getattr(cpu, f), dtype=float)
        try:
            b = np.asarray(warp_array(wp_model, f), dtype=float).reshape(a.shape)
        except (AttributeError, ValueError) as e:
            out.append(f"warp {f}: {e}")
            continue
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            out.append(f"warp {f}: max |diff| {np.max(np.abs(a - b)):.3g}")
    opt = {f: float(np.asarray(getattr(cpu.opt, f)).ravel()[0]) for f in
           ("timestep", "tolerance", "ls_tolerance", "integrator", "cone", "solver", "iterations", "ls_iterations",
            "ccd_iterations", "disableflags", "enableflags")}
    opt["impratio_invsqrt"] = 1.0 / np.sqrt(cpu.opt.impratio)
    for f, a in opt.items():
        v = getattr(wp_model.opt, f)
        b = float(np.asarray(v.numpy() if hasattr(v, "numpy") else v).ravel()[0])
        if not np.isclose(a, b, rtol=rtol):
            out.append(f"warp opt.{f}: cpu {a} gpu {b}")
    g = np.asarray(wp_model.opt.gravity.numpy()).reshape(-1, 3)[0]
    if not np.allclose(g, cpu.opt.gravity, rtol=rtol):
        out.append(f"warp opt.gravity: cpu {cpu.opt.gravity} gpu {g}")
    return out


def main() -> None:  # pragma: no cover - report script
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.tasks.hold import hold_env_cfg
    from tactile_sim.sim.training_parts import config_from_key, training_config
    from tactile_sim.sim.world import World

    cfg = hold_env_cfg(num_envs=2)
    key = cfg.actions["residual"].scene_key
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    for label, scfg in (("training (noslip 0)", config_from_key(key)),
                        ("validator (noslip 5)", training_config(noslip_iterations=5))):
        w = World(scfg)
        diffs = compare_models(w.model, env.sim.mj_model) + compare_warp(w.model, env.sim.wp_model)
        print(f"P0 vs the {label} CPU model: {'identical' if not diffs else ''}")
        for d in diffs:
            print("  ", d)


if __name__ == "__main__":  # pragma: no cover
    main()
