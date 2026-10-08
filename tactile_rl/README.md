# tactile_rl: GPU training env for the tactile residual (milestone 1)

GPU-parallel training of the WUJI Hand 2 residual (L2) on MuJoCo Warp through mjlab 1.6.0, mirroring the CPU testbed
(`tactile_sim`), which stays the reference and the validator. Plan: `~/.claude/plans/i-want-to-build-tidy-axolotl.md`
(sections "Training recipe", "Implementation §2", "Verification").

Milestone 1 covers the scene, the WUJI MIT-mode drive, the residual's action map, resets from settled states, a
minimal task, the parity ladder's P0/P1 plus drive parity, and the throughput benchmark.

## What is here

| Path | What |
|---|---|
| `tactile_sim/model/builder.py` `build_reduced_scene` | The reduced training scene (`SceneCfg.reduced`, default off): Vega U fixed pedestal, right arm (7 position servos), right WUJI Hand 2 (20 torque motors), hammer, nail plant with board and table. No left arm, left hand, head or visual-only meshes; the hammer leads the worldbody, so its free joint leads qpos. The CPU `World`/`Testbed` run it unchanged. |
| `tactile_sim/sim/training_parts.py` | `export_training_parts(cfg)`: builds the CPU `World` on the reduced scene, resets it once and exports everything the GPU env needs. That covers the MJCF (without the testbed's sensors and taxel marker sites, with the model edits the World makes at construction), the reset recipe (start state, weld pose, grip ramp, settle times), the arm servo targets (hover pose + gravity droop), the synergy and its hold scalar, the residual constants, and the CPU reference after the settle. `training_config(timestep)` is the training configuration: noslip 0, MIT drives, reduced scene. |
| `scene/entities.py` | The whole reduced MJCF as one mjlab entity, so the hand/handle pairs and the grasp weld stay in one spec. The arm uses its MJCF `<position>` servos (`XmlActuatorCfg`); the hand uses `WujiMitActuatorCfg` on its MJCF `<motor>`s. `mujoco_cfg` turns `PhysicsCfg` into mjlab's `MujocoCfg`. |
| `actuators/wuji_mit.py` | `TorchWujiDrive`: `tactile_sim.sim.wuji_drive.WujiMitDrive` batched over envs (see below); `WujiMitActuator(Cfg)`: the drive as an mjlab actuator. |
| `mdp/action_map.py` | `obs_spec.offsets/slew/compose`, `ResidualHandL2._trigger_guard` and `WrapSynergy.mit_command` in torch. |
| `mdp/actions.py` | `ResidualHandAction` (60-D): tanh, then offsets, per-second slew over the policy period, the trigger interlock, and composition with the base (the grip synergy at the constant hold scalar World.reset ends with). It sends the drives the host command. `SubstepHooks` (0-D): every physics substep before MuJoCo steps, it writes the arm servo targets and advances the drives one step. |
| `mdp/events.py` | Resets from a bank of settled states: at startup every world runs `World.reset_pose` + `World.settle`. That is the arm at the hover pose; the hand closing on the welded hammer while the grip ramps over the first half of 0.15 s; the weld released; 0.25 s free; the drive clocks rebased to 0. The bank keeps qpos, qvel, the solver warm start and the drives' full state; a reset copies a random entry. The settle fails loudly on a MuJoCo Warp buffer overflow. |
| `mdp/observations.py`, `rewards.py`, `terminations.py`, `kinematics.py` | Placeholders for milestone 1: hand joint angles and currents plus the last action (100-D). The reward is the hammer's drift in the hand plus I² and residual penalties. Terminations: dropped tool (> 2 cm), thermal fault, non-finite state, time out. |
| `tasks/` | `Tactile-Hold-VegaWuji-{1000,500,250,100}Hz` in the mjlab registry: physics 0.25 ms, decimation 4/8/16/40, PPO settings per rate from the plan. |
| `parity/model_fields.py`, `parity/one_step.py` | P0 and P1 (below), as functions and as report scripts. |
| `bench/throughput.py` | The throughput benchmark (below). |
| `train.py` | mjlab's training CLI with these tasks registered. |
| `tests/rl/` | GPU tests (marker `gpu`, skipped without torch + mjlab + CUDA). The CPU test of the reduced scene is `tests/test_reduced_scene.py`. |

## How to run

On the desktop, the training venv is `~/tactile-encode/.venv-rl` (torch 2.14 cu130, MuJoCo / MuJoCo Warp 3.11, mjlab
1.6.0). The validator venv is `~/tactile-encode/.venv` (MuJoCo 3.14). From the repo root:

```bash
# CPU suite (validator venv): the GPU tests skip themselves there
~/tactile-encode/.venv/bin/python -m pytest -q -m "not network"
~/tactile-encode/.venv/bin/python -m ruff check .

# GPU tests (training venv), 1-2 min; -s prints the P1 report
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m pytest -q -m gpu tests/rl -s

# reports
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.parity.model_fields   # P0
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.parity.one_step       # P1
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.bench.throughput      # throughput, ~8 min

# training-style rollouts (tensorboard logs under logs/rsl_rl/)
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.train Tactile-Hold-VegaWuji-1000Hz \
    --env.scene.num-envs 4096 --agent.max-iterations 100
```

An env build takes 7–22 s at 1024–4096 worlds: the CPU export (~5 s), `put_model` and the startup settle. The first
run on a machine also compiles the Warp kernels (about a minute, cached in `~/.cache/warp`).

## Results (desktop, RTX 5070 Ti, 2026-10-07)

### WUJI drive: torch vs numpy reference

`tests/rl/test_rl_drive_parity.py` runs one torch env per reference `WujiMitDrive`, eight envs with their own
parameters, for 4000 steps (1 s at 0.25 ms). The host commands change every step; the jitter draws are replayed from
each reference's RNG. The streams cover:

- transport delay of 0–3 ms with up to 0.9 ms jitter;
- the SDK low-pass (50, 100, 300 Hz) on `q`, or on all five fields;
- the current clamp (effort limit 0.9–2 times rated, with the peak current below it), and current loops of 0 to
  0.5 ms;
- a winding driven through derating into the latched fault.

**Max |Δτ| = 6.6e-7 Nm** (target 1e-5), and 4.5e-7 Nm with a 1 kHz servo rate. Fault state, winding temperatures
(< 1e-6 K) and SDK cycle counts match exactly.

### P0: model fields

The model mjlab compiles from the entity equals the CPU testbed's reduced model (training configuration, MuJoCo 3.11)
**exactly**, on every field compared:

- options and sizes;
- bodies (mass, inertia, frames);
- joints and dofs (ranges, limit softness, armature, damping, friction loss);
- geoms and meshes;
- all 358 contact pairs (dim, friction, solref, solreffriction, solimp, margin);
- the 27 actuators (gains, biases, ranges);
- equalities.

The MuJoCo Warp model's arrays match to float32. The only differences:

| Field | CPU | GPU | Note |
|---|---|---|---|
| `opt.noslip_iterations` | 5 (validator) | 0 | MuJoCo Warp has no noslip; the training configuration uses 0 on both sides |
| Newton tolerance | 1e-8 | 1e-6 | `put_model` raises it for float32; no effect on the one-step results (tested down to 1e-10) |
| sensors, taxel marker sites | present | dropped | no effect on dynamics; the sensors would cost an acceleration pass per step |
| MuJoCo version | 3.14 (validator) | 3.11 | mesh frames differ across versions (see open issues) |

The GPU settle reproduces `World.reset`: the hammer's pose in the hand lands within 0.1 mm / 0.1° of the CPU's
(typically 0.02 mm / 0.03°), and the arm joints within 1e-4 rad.

### P1: one physics step

200 states sampled along CPU runs: 40 in the welded settle, 60 in the hold, 100 in a perturbed hold. The
perturbations are ±15 N / ±0.5 Nm on the hammer, residual MIT commands at 0.6 of full scale, and ±0.02 rad arm targets.
States are rounded to float32 and loaded one per GPU world, then compared with `mj_forward`/`mj_step`.

qacc is compared as `||Δqacc||_M / ||qacc_smooth||_M`: the acceleration error in the kinetic-energy norm, relative to
the unconstrained acceleration the constraints balance. A plain relative error is meaningless in a static grasp, where
qacc is ~0 and a 1e-4 kg m² finger joint turns any force noise into rad/s²; it reaches 1e3 here, and is 6.6 at p99 even
between the CPU and itself. The "floor" column is the CPU against itself with the state nudged by one float32 ulp.

| | GPU, multi-contact on (testbed default) | GPU, multi-contact off on both | floor (1 ulp) |
|---|---|---|---|
| states with the same contacts per pair | 135 / 200 | 188 / 200 | 198 / 200 |
| smooth dynamics, p99 rel (bias / passive / actuator / qacc_smooth) | 3.7e-7 / 3.6e-8 / 1.8e-5 / 9.3e-6 | same | — |
| qacc, M-norm rel, all states: median / p99 | 3.7e-3 / 0.27 | 6.9e-4 / 0.27 | 9.1e-5 / 8.1e-4 |
| qacc, states with the same contacts: median / p99 | **6.8e-4** / 0.091 | **6.8e-4** / 0.080 | 9.1e-5 / 6.8e-4 |
| pair normal force rel, all pairs ≥ 0.1 N: median / p99 | 3.1e-3 / 0.83 | 2.0e-3 / 0.83 | 4.1e-4 / 9.4e-3 |
| pair normal force, same contacts: median / p99 | 1.6e-3 / 0.13 | 1.8e-3 / 0.099 | 4.0e-4 / 6.6e-3 |
| force on pairs one side only | 1.3 % | 1.4 % | 0.05 % |

Against the plan's P1 targets (median ≤ 1e-3, p99 ≤ 1e-2):

- **Met:** the qacc median where both sides find the same contacts (6.8e-4).
- **Not met:** the force median (1.6–2.0e-3) and every p99.

What the gap is:

- **Float32 arithmetic in a stiff, ill-conditioned grasp.** The smooth dynamics agree to ≤ 2e-5, and the contact
  geometry where both sides find a contact to ~1e-7 m. Nudging the CPU state by 10 float32 ulps gives qacc 8.7e-4 and
  force 4.8e-3 at the median (multi-contact off), the same scale as the GPU.
- **Contact detection.** Most mismatched states are the palm/handle face contact (`hammer_seg06`/`wuji_col_r_wrist`):
  - 1 contact on the CPU and 2 on the GPU in 51 states;
  - 1 on the CPU and none on the GPU in 9 states;
  - CPU MuJoCo's own count never flips under nudges.

  Second points carry, e.g., 44 N instead of 25 N on the palm. The few remaining mismatches are contacts at ~zero
  penetration.

`tests/rl/test_rl_parity.py` asserts the achievable levels (and catches regressions of the pipeline). Run
`python -m tactile_rl.parity.one_step` for the full report.

### Throughput

`python -m tactile_rl.bench.throughput` (each configuration in its own process). The task is a zero-residual hold at
a 1 kHz policy with the full hand/handle contact set (358 pairs), MIT drives in torch every substep, and default
buffers `nconmax` 52 and `njmax` 88. Hammer drift is measured 1 s after a reset.

| worlds | dt (ms) | physics steps/s | world-steps/s | simulated s / wall s | env GPU memory (GB) | GPU drift 1 s median / max (mm) | CPU drift 1 s, noslip 0 / 5 (mm) |
|---|---|---|---|---|---|---|---|
| 1024 | 0.25 | 239 | 2.45e5 | **61** | 1.37 | 0.138 / 0.188 | 0.0114 / 0.0017 |
| 2048 | 0.25 | 163 | 3.33e5 | **83** | 2.66 | 0.140 / 0.188 | 0.0114 / 0.0017 |
| 4096 | 0.25 | 87 | 3.56e5 | **89** | 5.18 | 0.140 / 0.190 | 0.0114 / 0.0017 |
| 1024 | 0.5 | 257 | 2.63e5 | **131** | 1.37 | 0.087 / 0.100 | 0.0091 / 0.0007 |
| 2048 | 0.5 | 172 | 3.53e5 | **176** | 2.66 | 0.086 / 0.100 | 0.0091 / 0.0007 |
| 4096 | 0.5 | 89 | 3.65e5 | **183** | 5.18 | 0.086 / 0.103 | 0.0091 / 0.0007 |

- **Speed against the plan.** The plan's gate is 10 simulated s per wall s (goal 25), so every configuration passes
  it several times over without fusing the hooks. The CPU testbed runs this hold at 1.5 simulated s per wall s per core
  at 0.25 ms.
- **Where the time goes at 4096 worlds, 0.25 ms.** An env step takes 43.7 ms:
  - physics: 4 × 8.2 ms;
  - mjlab's per-step `forward()`: 8.2 ms;
  - the drive hook: 4 × 0.49 ms (launch-bound, independent of the world count, ~4.5 %);
  - action processing, rewards and terminations: ~1.2 ms.
- **Buffers.** MuJoCo Warp pools contacts over worlds, and the broadphase candidates must fit that pool too. The
  welded settle is the peak:

  | | broadphase candidates / world | contacts / world | constraint rows (fullest world) |
  |---|---|---|---|
  | 0.25 ms | 45 (hold 42) | 10 | 77 (hold 67) |
  | 0.5 ms | 50 (hold 47) | 10 | 72 (hold 59) |

  **The minimum that works is `nconmax` 50, `njmax` 77**, verified without overflow at both steps. The defaults 52/88
  leave a little headroom; strikes and other plants will need more. The settle raises an error on any overflow; a
  silent one corrupts the bank.
- **Memory** is ~1.3 GB per 1024 worlds.
- **Drift.** The GPU hold drifts ~12x more than the CPU without noslip (see open issues).

## How the env hooks into mjlab

- **One entity.** mjlab allows one free joint per entity and takes it as the root only if it comes first, so the
  reduced scene puts the hammer body first in the worldbody. The compiled model is the CPU model element for element,
  with names prefixed `robot/`.
- **Options** come from `MujocoCfg`; mjlab does not take an entity's `<option>`. MuJoCo Warp has no noslip and raises
  the Newton tolerance to at least 1e-6 (float32). The entity spec drops `<option>` and `<size>`.
- **Pre-substep hook.** Every physics substep, `ManagerBasedRlEnv.step` runs each action term's `apply_actions`, then
  `scene.write_data_to_sim`, `sim.step` and `scene.update`. A 0-D action term (`SubstepHooks`) therefore runs once per
  substep before MuJoCo steps, and needs no override of `step()`. The drive is advanced there, not in the actuator's
  `compute`: mjlab also calls `write_data_to_sim` outside the physics loop (after every reset, for all envs), so
  `compute` only reads the torque.
- **Command order.** As on the testbed, a policy step sets the host command before the substep in which the SDK
  latches it. With phase 0, the SDK cycle and the policy tick fall on the same substep.
- **Resets.** mjlab resets the sim (`reset_data`) and the entity (which zeroes the actuator targets and resets the
  drive), then runs the reset events. The bank event writes qpos, qvel, the warm start, time 0, the drives' state and
  the arm targets.
- **Per-step `forward()`.** mjlab runs one `forward()` per env step, as costly as a physics step. At 1 kHz with 4
  substeps this adds ~20 % (8.2 ms of 43.7 ms at 4096 worlds). Removing it needs a `step()` override, which the plan
  allows at a pinned commit.

## Known differences and open issues

- **The GPU hold creeps ~10x more than the CPU.** Over 1 s at 0.25 ms the hammer drifts 0.14 mm in the hand on the GPU,
  against 0.011 mm on the CPU without noslip and 0.002 mm with the validator's noslip 5. The cause is not world
  coordinates in float32: moving the scene to the origin changes nothing. The palm multi-contact (below) is part of
  it: 0.092 mm with multi-contact off. In the static hold the GPU's velocities jitter 5–30x more than the CPU's (RMS
  hand joints 3.6e-3 vs 7.0e-4 rad/s; hammer 1.8e-4 vs 5.9e-6 m/s, 2.9e-3 vs 1.3e-4 rad/s). That is float32 noise in a
  stiff, near-singular grasp (impratio 100, 2 ms contacts, 1e-4 kg m² finger joints) acting as micro-vibration on the
  frictional contacts; a larger step creeps less (0.086 mm at 0.5 ms). The plan's contact recalibration (static-hold
  creep over 5 s, impulse slip, ripple) is the remedy. It is not milestone 1.
- **Multi-contact on the palm.** CPU MuJoCo and MuJoCo Warp both default to multi-contact convex collisions. On the
  palm/handle face contact, Warp emits a second point where CPU MuJoCo emits one in about a third of the hold states.
  The CPU's own contact set does not flip under float32-ulp nudges, so this is an algorithm difference. It raises the
  palm's share of the grip (e.g. 25 N to 44 N in one state). MuJoCo Warp also warns that cylinder/mesh pairs (the nail
  head against the hammer face) get at most one contact. That matters at impact; P1 needs strike states in
  milestone 2.
- **MuJoCo version drift.** The validator (3.14) and the training venv (3.11) compile the same MJCF with different mesh
  frames: geom quaternions, mesh sizes and bounding radii differ, while masses, inertias (to 1e-9), joints, pairs and
  actuators match. The settled grasp after `World.reset` differs by up to 2.2e-3 in qpos. A test across the two venvs
  (the plan's version-drift guard) is still missing.
- **Mirrors to revisit.** (1) The drill-trigger interlock applies on the hammer task too, since `ResidualHandL2` applies
  it whatever the task; with the synergy base the index finger's residual may only open. (2) The CPU L2's base reads
  the joint angles through the readback sensor (fb latency, encoder quantization); the GPU base reads true angles. This
  only matters for the coupled DIP target of the "tendon" synergy, and for Δq on joints with Kp > 0. (3) Stale-input
  decay is not mirrored: there are no GPU sensors yet.
- **Placeholders.** The observation (hand q, current, last action), the reward (drift, I², residual size) and an MLP
  actor in the PPO config are placeholders until milestone 2. The plan's teacher is a GRU with a zero-initialized
  output layer.

## Missing for milestone 2

- **Plant hooks** in `SubstepHooks`, from the CPU `NailPlant.pre_step`: the nail's `dof_frictionloss` growing with
  depth, and the face/nail pair's damping from the approach speed (`pair_solref`, an expanded model field). Then the
  arm's 100 Hz joint stream (ZOH) for strikes.
- **Batched TaxelScan.** Contacts from `ContactSensorCfg` (patch bodies against the tool), the elastic-foundation load
  model with analytic rays, the rev3 readout and codec (bit-exact with `sensors/taxel_codec.py`), and frame timing.
- **Readback sensors** with SDK-rate sampling, fb latency, encoder quantization and current noise (`hand_q`,
  `hand_current`), and the stale-input decay.
- **Load generator and drill**: the plants, the pistol driver, the trigger joint and the base sources.
- **Observations and rewards from the plan**: the `obs_spec` student layout, a 1 kHz frame stack for slower rates, the
  teacher/critic privileged terms, and substep-integrated rewards (an accumulator post-substep hook, e.g. a custom
  `Sensor` whose `update` runs every substep).
- **Domain randomization events** on the drive (`TorchWujiDrive.set_params` takes per-env tensors), contacts
  (`pair_solref`/`solimp`), masses, arm gains. A bank refresh follows parameter changes.
- **Contact recalibration** (above), a fused-graph version of the hooks if they cost more than 15 % (now ~5 % at 4096
  worlds), and the cross-venv MuJoCo version-drift test.
