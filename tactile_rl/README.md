# tactile_rl: GPU training env for the tactile residual (milestones 1–2)

GPU-parallel training of the WUJI Hand 2 residual (L2) on MuJoCo Warp through mjlab 1.6.0, mirroring the CPU testbed
(`tactile_sim`), which stays the reference and the validator. Plan: `~/.claude/plans/i-want-to-build-tidy-axolotl.md`
(sections "Training recipe", "Implementation §2", "Verification").

- **Milestone 1** built the reduced Vega U + WUJI scene, the WUJI MIT-mode drive in torch, the residual's action map,
  resets from settled states, the hold tasks, P0/P1 parity and the throughput benchmark (results kept below).
- **Milestone 2** adds the hammer task on the residual experiments' configuration: the robot's sensors on the GPU (WUJI
  readback, TaxelScan rev3 with its load model, codec and frame timing), the nail plant, the base library of scripted
  swings recorded on the CPU and played on the GPU, the student / teacher / critic observations, the stale-input decay,
  and parity evidence for each against the testbed.

## What is here

| Path | What |
|---|---|
| `tactile_sim/sim/training_parts.py` | `export_training_parts(cfg)`: the CPU export the GPU env is built from (M1), now also the TaxelScan skins (taxels in their bodies' frames, the geoms loading each patch, board scan offsets and poll slots), the nail plant's constants, the skin/self contact pairs and the arm servo gains. `residual_training_config(rate)`: `residual_config` (light hammer, small nail, MIT drives, rev3 at 7.8 µs/taxel, the open-loop constant-grip base) on the reduced scene at 0.25 ms without noslip. `training_config`: M1's hold scene. |
| `tactile_sim/behaviors/base_library.py` | The base library. `export_library` runs SwingBase episodes (zero residual, one `residual_dr` draw per episode) and records every millisecond the arm's 100 Hz servo stream (target, velocity feedforward, droop, as PositionL1 writes them), the synergy scalar, the lookahead, the forecast truth and the swing phase; each swing is an entry with its start state (qpos, qvel, warm start, the drives' full state), the episode's randomization and model constants, and the CPU's outcome. `replay_entry` plays an entry open loop on a CPU World the way the GPU does. |
| `sensors/readback.py` | `TorchReadback`: `hand_q` / `hand_current` as the testbed's `RateLimitedSensor` (sample-and-hold at the SDK rate, fb latency, noise, quantization), per-env parameters. |
| `sensors/taxel_codec.py` | The fast-profile codec in torch: mN, mu-law codes from the shared threshold table, integer patch features. |
| `sensors/taxelscan.py` | `TorchTaxelScan`: `TaxelScanRev3Sensor` for all patches and envs: skin low-pass, sequential scan from SYNC, divider / 12-bit ADC / ENOB noise with per-env per-taxel gain and offset (and an optional per-taxel f_half spread), calibration, codes, features, frame availability (poll slots, bridge latency, in order), drops. |
| `sensors/taxel_load.py` | `TaxelLoadModel`: the testbed's elastic-foundation skin model (`HandIO.skin_taxels`) on MuJoCo Warp's contacts, with MuJoCo's `ray_capsule` in torch for the light hammer's capsule handle; its stateless core runs through `torch.compile`. |
| `sensors/hub.py` | `TactileHub`: a scene sensor whose `update` mjlab calls after every physics substep: the post-substep hook. |
| `plants/nail.py` | `TorchNailPlant`: `NailPlant.pre_step` per world and substep (friction loss from depth, velocity-dependent restitution through the pair's direct-form solref, frozen in contact), plus the strike truth. |
| `mdp/runtime.py` | `TactileRuntime`, one per env: the substep clock, the sensors, the plant, the student's 1 kHz ring, the truth; the post-substep pipeline and the stale check. |
| `mdp/commands.py` | `BaseTrajectoryCommand`: the library on the GPU, per-env entry, start state and randomization, perturbations (time warp, amplitude, start delay), the 100 Hz arm stream, the synergy scalar, the lookahead, the forecast. |
| `mdp/forecast.py` | `TorchForecastModel`: `behaviors.forecast.ForecastModel` batched, every error parameter per env. |
| `mdp/obs_builder.py` | `StudentObs`: `ResidualHandL2.fast_sample` / `observe` batched (obs_spec layout, `stack` deep, masks). |
| `mdp/actions.py` | `ResidualHandAction`: tanh, offsets, the drill-only trigger interlock, the stale-input decay, slew, composition with the base (the synergy at the library's scalar from the read-back angles). `SubstepHooks`: arm targets, plant pre-step, drives. |
| `mdp/observations.py` | `student_obs` and the privileged terms (`tool_in_hand`, `patch_contact`, `event_truth`, `nail_state`, `drive_internals`, `randomization`); M1's placeholder terms for the hold tasks. |
| `mdp/events.py`, `rewards.py`, `terminations.py` | M1's bank; `tactile_model_fields` (declares the per-world fields); placeholder hammer rewards (nail advance, slip path, I², residual size); `base_done`, `tool_left_hand`. |
| `tasks/hammer.py` | `Tactile-Hammer-VegaWuji-{1000,500,250,100}Hz`; `hammer_scene_env_cfg` (the scene without base or sensors, for the parity scripts). `tasks/hold.py`: M1's `Tactile-Hold-VegaWuji-*`. |
| `parity/` | `model_fields` (P0), `one_step` (P1, now also on strike states), `taxel_load` (the load model against the CPU's foundation), `strikes` (library entries on the GPU against the CPU). |
| `tests/rl/` | `test_rl_sensors.py` (codec, readback, rev3, forecast, student observation against the CPU), `test_rl_hammer.py` (the hammer task), M1's tests. CPU: `tests/test_base_library.py`, `tests/test_residual_l2.py`. |

## How to run

On the desktop, the training venv is `~/tactile-encode/.venv-rl` (torch 2.14 cu130, MuJoCo / MuJoCo Warp 3.11, mjlab
1.6.0). The validator venv is `~/tactile-encode/.venv` (MuJoCo 3.14). From the repo root:

```bash
# CPU suite (validator venv): the GPU tests skip themselves there
~/tactile-encode/.venv/bin/python -m pytest -q -m "not network"
~/tactile-encode/.venv/bin/python -m ruff check .

# GPU tests (training venv), ~10 min (the hammer tests record a small library on the CPU first); -s prints reports
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m pytest -q tests/rl

# the base library: record it in the training venv (MuJoCo 3.11, as the GPU), ~4 swings per episode until the nail
# is in; 48 episodes give ~200 swings (~5 min on 14 workers). Regenerate it whenever residual_config's base changes.
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_sim.behaviors.base_library export \
    --episodes 48 --strikes 10 --workers 14 --out runs/base_library/hammer_v1.npz

# reports
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.parity.model_fields        # P0
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.parity.one_step --scene strikes   # P1 (hold|both)
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.parity.taxel_load          # skin loads
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.parity.strikes --lib runs/base_library/hammer_v1.npz
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.bench.throughput           # M1's hold benchmark
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.bench.throughput --task hammer --seconds 0.25 \
    --rate 1000 250 --library runs/base_library/hammer_v1.npz                                # the hammer task

# training-style rollouts (tensorboard logs under logs/rsl_rl/); the hammer tasks read the library named by
# $TACTILE_BASE_LIBRARY (default runs/base_library/hammer_v1.npz)
TACTILE_BASE_LIBRARY=runs/base_library/hammer_v1.npz PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python \
    -m tactile_rl.train Tactile-Hammer-VegaWuji-1000Hz --env.scene.num-envs 4096 --agent.max-iterations 100
```

A hammer env builds in ~11 s (the CPU export, `put_model`, loading the library, compiling the load model's core).
The base library warns when it was recorded under another base than `residual_config`'s (`base_signature`: grip
loop, grip profile, margin step, hold, post switch); regenerate it then.

## How the hammer env works

- **Scene.** `residual_training_config(rate)` on the reduced scene: Vega U right arm on position servos, WUJI Hand 2 on
  the torch MIT drives, the light hammer (primitive: cylinder head, capsule handle), the small nail on its slide joint.
  0.25 ms physics; decimation 4 / 8 / 16 / 40 for 1000 / 500 / 250 / 100 Hz.
- **Clock and hooks.** Each env counts substeps since its reset (time k·dt, float64). Before a substep
  (`SubstepHooks.apply_actions`): the arm's servo targets, the nail plant's pre-step, one drive step. After it (the
  `TactileHub` sensor, which mjlab updates right after `sim.step`): the taxel loads from that step's contacts, the scan,
  the readback, the plant's measurements, and on every millisecond the student's 1 kHz sample. Everything a post-step
  hook reads is the step's own pre-integration state, as the testbed's sensors read MjData after `mj_step`; mjlab's
  per-env-step `forward()` comes after. Resets happen between env steps and every decimation is a multiple of four,
  so every env's millisecond boundaries fall on the same substeps.
- **Episodes.** At reset an env takes a library entry (random, or `assign="sequential"`) and starts in the CPU's state
  at that entry's windup (qpos with the nail's depth, qvel, warm start, the WUJI drives' full state including commands
  in flight), with the entry episode's randomization: the hammer's inertial with MuJoCo's derived constants from the
  CPU model, skin friction and contact softness on the hand pairs, arm servo gains, nail resistance and restitution,
  drive latency / low-pass / current loop / Kt / effort limit / ambient temperature, the TaxelScan's gain and offset
  spread, f_half, bandwidth, skin depth, drops and bridge latency, sensor noise. The episode ends with the entry
  (`base_done`, a truncation), on a dropped tool, a thermal fault or a non-finite state.
- **Base.** Library time u = w·max(τ − d, 0) with time warp w ∈ [0.8, 1.2], start delay d ∈ [0, 0.1] s and amplitude
  a ∈ [0.9, 1.1] (`perturb`; all off replays the entry exactly). Arm: 100 Hz ticks in env time, phased onto the CPU's
  ticks when unperturbed; each tick interpolates the CPU's targets at u and sends q0 + a(q − q0) + droop + (kd/kp)·a·w·qd.
  Hand: the grip synergy (`WrapSynergy.mit_command`) at the library's scalar s(u) from the angles the host last read.
  Lookahead: refreshed on the arm's ticks, the library's displacements at the warped horizons, scaled by a. Forecast:
  the library's truth (event time mapped to env time, amplitude ×a·w) through the torch ForecastModel.
- **Sensors.** Readback and TaxelScan exactly as on the testbed (see parity below), every parameter per env. The skin
  load: per patch the summed normal force of its hand/handle contacts within reach of a taxel, shaped by the
  elastic foundation (rays along the taxel normals against the capsule; a Gaussian spread when no ray finds it),
  refreshed every 0.5 ms or when a patch starts carrying load.
- **Residual action** (`ResidualHandL2.tick`): non-finite → 0, tanh, offsets, trigger interlock (drill only), stale
  inputs (a board's newest frame older than 5 ms or none, the readback older than 3 ms or none) decay the offsets with
  τ = 20 ms, slew limit, composition with the base; the squashed action is the next observation's `last_action`.
- **Observations.** `student`: obs_spec's layout exactly, `stack_for(rate)` 1 kHz samples per fast block, newest first,
  masked blocks zero (`mask_blocks`). `teacher` and `critic`: the student plus 140 privileged entries: the tool in the
  hand (pose change since the start, relative twist), per patch the true normal force and the slip velocity at its
  contacts, the base's true plan and the true contact (onset, time since, peak), the nail (depth, resistance, gap,
  approach speed, force), the drives (currents, winding temperatures, fault, latency, low-pass and current-loop
  coefficients), the randomization (25 values including the perturbations). The PPO config trains on `teacher`.

## Results (desktop, RTX 5070 Ti, 2026-10-08)

### Sensors and observation builder against the testbed (`tests/rl/test_rl_sensors.py`)

| What | Against | Result |
|---|---|---|
| codec: codes and the six patch features | the frozen golden frames (`tests/data/taxel_codec_v1.json`), U8 and U6, every board and case | **bit-exact** |
| codec over the whole range | `CodecSpec.encode` at every threshold ±1 mN and every 7 mN to 70 N; `to_mn` incl. half-mN ties | **bit-exact** |
| readback (`hand_q`, `hand_current`) | `RateLimitedSensor` with the same noise draws, 4 parameter sets (1 kHz / 500 Hz, 0–3 ms latency, noise, quantization), 0.25 and 0.125 ms physics | value, t_sample, t_avail, seq **identical** at every step |
| TaxelScan rev3, 11 patches × 4 parameter sets (bandwidth, f_half, gain/offset spread, ENOB, bridge range, drops 5–20 %) | `TaxelScanRev3Sensor` on identical force streams with the same gain/offset, bridge, ADC-noise and drop draws, 0.25 and 0.125 ms | seq, t_sample, t_avail, sync_id, codes and features **identical** for every frame (> 1000 frames), drops too |
| ForecastModel (5 parameter sets incl. masked, dropout, latency 0–100 ms, 5–10 Hz, 1 kHz and 250 Hz ticks) | `ForecastModel` with the same draws | 13-D vector to 1e-12 at every tick |
| student observation (stack 1, 2, 4, 10, with masks) | `ResidualHandL2.sample` / `observe` on identical sensor streams | equal to float32 (2e-6) |

### Skin loads: the GPU load model against the testbed's foundation (`python -m tactile_rl.parity.taxel_load`)

300 states of a CPU hammering episode (hammer scene, no randomization): 60 hold, 16 in the 2 ms before a blow, 224
during and up to 20 ms after the face/nail contacts. Per state and patch with load (> 0.5 N on either side, 970
patch observations): the patch-sum error |Σ_gpu − Σ_cpu| / max, the shape error Σ|f_gpu − f_cpu| / max (L1 of the
taxel distribution), the centre-of-pressure distance (taxel pitches).

| | patch sum: median / p90 / p99 | shape (L1): median / p90 | CoP: median / p90 | patches < 5 % |
|---|---|---|---|---|
| GPU model on the CPU's contacts and poses | **3.1e-8** / 7.6e-8 / 1.3e-7 | **1.3e-5** / 2.7e-5 | 6.3e-6 / 1.7e-5 | 100 % |
| end to end, MuJoCo Warp's contacts, multi-contact on (testbed default) | 1.2e-1 / 2.8e-1 / 1.0 | same | 1.1e-5 / 3.2e-5 | 25 % |
| end to end, multi-contact off on both sides | **1.6e-3** / 5.5e-2 / 1.0 | same | 1.1e-5 / 3.2e-5 | **89 %** (hold 100 %, strikes 85 %) |

- **The load model is the testbed's**, to float32: given the same contacts the GPU distributes the same force over the
  same taxels (the rays are MuJoCo's `ray_capsule` term for term).
- **End to end the gap is the simulators' contacts.** MuJoCo Warp gives a capsule/mesh pair (the handle against a hand
  hull) and a cylinder/cylinder pair (the face against the nail head) **one contact**, multi-contact or not; CPU MuJoCo
  with multi-contact gives up to 3 (handle on a fingertip) and 5 (face on nail). The patch loads then split
  differently (12 % median). With multi-contact off on the CPU too, the patch sums agree to 0.16 % (hold 0.05 %, p99
  0.19 %); the strike tail (p99 1.0) is states where one side has a contact the other lacks.
  `PhysicsCfg.multiccd = False` turns it off on both sides (World and `mujoco_cfg`); the default is unchanged.

### Strikes on the GPU against the CPU (`python -m tactile_rl.parity.strikes --validator --perturbed 4`)

The library: 203 swings from 48 episodes (one `residual_dr` draw each; the open-loop constant-grip base of
`residual_config`; ~5 swings per episode until the nail is in). Every entry plays once on the GPU, unperturbed, from the
CPU's state at its windup and with its episode's randomization, zero residual; the nail plant runs as hooks. Compared,
per entry, with the CPU run that recorded it (the testbed closed loop at the training physics: 0.25 ms, no noslip), the
CPU replaying it open loop the same way, and the CPU replaying it at the validator's physics (8 kHz, noslip 5).

| per strike (means over hits) | CPU recorded | CPU replay | **GPU** | CPU validator replay |
|---|---|---|---|---|
| hit rate | 203 / 203 | 203 / 203 | **203 / 203** | 203 / 203 |
| nail advance (mm) | 5.05 | 5.05 | **5.09** | 1.16 |
| peak face/nail force, 2nd-largest sample (N) | 260.1 | 260.1 | **259.6** | 354 |
| pulse width (ms) | 13.8 | 13.9 | 12.7 | 10.3 |
| face speed at contact (m/s) | 1.252 | 1.252 | 1.258 | 1.253 |
| slip in the hand (mm / mrad) | 1.26 / 31.7 | 1.24 / 32.0 | 1.21 / 33.7 | 2.00 / 63.5 |

| per entry, against the CPU replay | GPU | CPU recorded |
|---|---|---|
| hit agreement | **100 %** | 100 % |
| advance error: median / p90 | **1.4 % / 4.4 %** | 0 / 0 (identical) |
| peak force (2nd largest) error: median / p90 | **0.9 % / 6.1 %** | 0 / 0 |
| pulse width error: median / p90 | 9 % / 70 % | 0 / 0 |
| contact time: median / p90 of the absolute difference | −0.25 ms / 2.8 ms | 0 |
| face speed at contact: median / p90 of the absolute difference | 0.7 / 23 mm/s | 0 |

- **The GPU strikes as the CPU does** at the training physics: same hits, nail advance within 1.4 % and blow within
  0.9 % at the median. Pulse width is the noisy metric (a yielding nail chatters; `pulse_stats` bridges gaps up to
  1 ms).
- **The open-loop replay is exact on the CPU**: with the constant grip the base's hand command does not change, so the
  replay reproduces the recorded run bit for bit (its blows, advance and slip identical). The GPU's difference is
  the simulators'.
- **Perturbed bases still hit:** 812 worlds (each entry 4 times; time warp 0.8–1.2, amplitude 0.9–1.1, start delay
  0–0.1 s): **hit rate 100 %**, advance 5.19 mm on average; 4.07 / 5.13 / 6.46 mm for an effective speed scale (warp
  × amplitude) of 0.7–0.9 / 0.9–1.1 / 1.1–1.35.
- **The validator's physics drives the nail 4x less** on the same arm streams, and its blows are spikier (9 one-sample
  spikes in 203, mean raw peak 993 N). This is noslip, not the step: replaying 11 entries, the advance is 4.74 mm at
  0.25 and at 0.125 ms without noslip, 0.79 and 0.74 mm with noslip 5. Noslip also acts on the nail's friction-loss
  constraint (its Coulomb resistance): without it the soft constraint lets the nail slip further at impact and creep
  ~0.1 mm in each windup. MuJoCo Warp has no noslip, so the GPU follows the no-noslip CPU (see open issues).

### P1 with strike states (`python -m tactile_rl.parity.one_step --scene strikes`)

300 states of the hammer scene as above, rounded to float32, one GPU world each (with the plant's per-state friction
loss and face/nail solref written per world). Metrics as in milestone 1's P1 (below).

| | multi-contact on (testbed default) | multi-contact off on both | floor (CPU vs CPU nudged 1 ulp) |
|---|---|---|---|
| states with the same contacts per pair | 4 / 300 | 248 / 300 | 295 / 300 |
| qacc, M-norm rel, all states: median / p99 | 4.4e-2 / 0.75 | 1.6e-3 / 0.67 | 9.8e-5 / 0.16 |
| qacc, same contacts: median / p99 | 1.9e-4 / 4.4e-4 (4 states) | **1.1e-3** / 9.9e-2 | 9.6e-5 / 7.4e-4 |
| pair normal force rel: median / p99 | 1.1e-1 / 1.0 | **1.6e-3** / 1.0 | 1.5e-4 / 9.3e-2 |
| hold states (60): qacc median / p99; force median / p99 | 4.3e-2 / 4.6e-2; 0.12 / 0.29 | **1.7e-4 / 3.8e-4; 5.4e-4 / 3.8e-3** | — |
| strike states (224): qacc median / p99; force median | 4.8e-2 / 0.82; 0.11 | 2.4e-3 / 0.73; 3.5e-3 | — |
| face/nail force, strike states: both sides in contact | 71 of 108 states, rel median 0.11 | 71 of 108, rel median **3.0e-3**, p90 2.1e-2 | 106 of 109 |
| face/nail contact on one side only | CPU only: 37 states | CPU only: 37 states (1 → 0 contacts) | 3 |

- With multi-contact off on both sides, **the hold states meet the plan's P1 targets** (median ≤ 1e-3, p99 ≤ 1e-2) for
  qacc and contact forces. With it on, almost no state has the same contacts (above).
- At impact, MuJoCo Warp misses the face/nail contact in 37 of 224 strike states where CPU MuJoCo (single contact) finds
  one, never the reverse; where both find it, the force agrees to 3e-3 at the median. The blows themselves still
  agree (the strike table above).

### Throughput (hammer task, zero residual, `nconmax` 40, `njmax` 128)

`python -m tactile_rl.bench.throughput --task hammer --seconds 0.25 --rate 1000 250 --library <lib>` (perturbed library
swings, every hook running; each configuration in its own process):

| worlds | policy | ms per env step | simulated s / wall s | env GPU memory (GB) |
|---|---|---|---|---|
| 1024 | 1 kHz | 33.8 | 30 | 0.88 |
| 2048 | 1 kHz | 44.9 | 46 | 1.62 |
| 4096 | 1 kHz | 67.0 | **61** | 3.06 |
| 1024 | 250 Hz | 110.6 | 37 | 0.91 |
| 2048 | 250 Hz | 151.2 | 54 | 1.72 |
| 4096 | 250 Hz | 223.5 | **73** | 3.33 |

The plan's gate is 10 (goal 25): passed everywhere. At 4096 worlds and 1 kHz one substep costs: physics 8.0 ms; taxel
loads 1.5 ms (7.0 ms before compiling the core); TaxelScan 3.7 ms (frame completion 2.0, emission 1.0, sampling 0.6);
readback 1.0 ms; nail 0.9 ms; drives 0.9 ms; arm stream 0.5 ms; the 1 kHz sample 0.4 ms (every fourth substep); per env
step the base command 2.0 ms and the observations. Below 2048 worlds the step is bound by launching the hooks' small
kernels. The hooks cost more than the plan's 15 % threshold: fusing them (CUDA graphs, Warp kernels, or compiling the
scan's integer parts, which stay exact) is the next step for speed. Peaks over the library swings: 25 broadphase
candidates and 7.5 contacts per world, 71 constraint rows (the defaults leave room for the load generator's contacts).

## Milestone 1 results (hold scene, kept)

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

### P1: one physics step (hold scene)

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

The gap was float32 arithmetic in a stiff, ill-conditioned grasp (nudging the CPU state by 10 float32 ulps gives the
same scale) and contact detection on the YCB handle's palm hull (1 contact on the CPU, 2 on the GPU in a third of the
states). `tests/rl/test_rl_parity.py` asserts the achievable levels.

### Throughput (hold scene)

`python -m tactile_rl.bench.throughput`: a zero-residual hold at 1 kHz with the YCB hammer's full hand/handle contact
set (358 pairs), MIT drives in torch every substep, `nconmax` 52 and `njmax` 88.

| worlds | dt (ms) | physics steps/s | world-steps/s | simulated s / wall s | env GPU memory (GB) | GPU drift 1 s median / max (mm) | CPU drift 1 s, noslip 0 / 5 (mm) |
|---|---|---|---|---|---|---|---|
| 1024 | 0.25 | 239 | 2.45e5 | **61** | 1.37 | 0.138 / 0.188 | 0.0114 / 0.0017 |
| 2048 | 0.25 | 163 | 3.33e5 | **83** | 2.66 | 0.140 / 0.188 | 0.0114 / 0.0017 |
| 4096 | 0.25 | 87 | 3.56e5 | **89** | 5.18 | 0.140 / 0.190 | 0.0114 / 0.0017 |
| 1024 | 0.5 | 257 | 2.63e5 | **131** | 1.37 | 0.087 / 0.100 | 0.0091 / 0.0007 |
| 2048 | 0.5 | 172 | 3.53e5 | **176** | 2.66 | 0.086 / 0.100 | 0.0091 / 0.0007 |
| 4096 | 0.5 | 89 | 3.65e5 | **183** | 5.18 | 0.086 / 0.103 | 0.0091 / 0.0007 |

At 4096 worlds and 0.25 ms an env step took 43.7 ms: physics 4 × 8.2 ms, mjlab's per-step `forward()` 8.2 ms, the drive
hook 4 × 0.49 ms, the rest ~1.2 ms. The minimum buffers for the hold were `nconmax` 50, `njmax` 77.

## How the env hooks into mjlab

- **One entity.** mjlab allows one free joint per entity and takes it as the root only if it comes first, so the
  reduced scene puts the hammer body first in the worldbody. The compiled model is the CPU model element for element,
  with names prefixed `robot/`.
- **Options** come from `MujocoCfg` (`mujoco_cfg`), including `PhysicsCfg.multiccd`; mjlab does not take an entity's
  `<option>`. MuJoCo Warp has no noslip and raises the Newton tolerance to at least 1e-6 (float32).
- **Pre-substep hook.** Every physics substep, `ManagerBasedRlEnv.step` runs each action term's `apply_actions`, then
  `scene.write_data_to_sim`, `sim.step` and `scene.update`. A 0-D action term (`SubstepHooks`) therefore runs once per
  substep before MuJoCo steps. The drive is advanced there, not in the actuator's `compute`: mjlab also calls
  `write_data_to_sim` outside the physics loop.
- **Post-substep hook.** `scene.update` updates every scene sensor after `sim.step`: the `TactileHub` sensor calls the
  runtime's post-substep pipeline. mjlab's single `forward()` per env step comes after the loop and the resets.
- **Per-world model fields.** The `tactile_model_fields` startup event declares the fields the hooks and the library's
  randomization write (`requires_model_fields`), so mjlab expands them before the terms are built; the values come
  from the CPU model (derived constants included), so nothing is recomputed.
- **Resets.** mjlab resets the sim and the entity (the drives), then the scene's sensors (the runtime: clock, sensors,
  plant, student ring with the reset sample), the reset events, and last the command, which writes the entry's state
  and randomization. The hold tasks still reset from M1's settled-state bank.
- **No host synchronization** in the hooks: the contacts are sorted into per-(world, patch) slots on the device, every
  branch is a mask. mjlab synchronizes twice per env step (resets and command resampling).

## Known differences and open issues

- **The nail without noslip (the main training/validator difference).** The nail's resistance is Coulomb friction on
  its slide joint (`frictionloss`, a soft constraint). The validator's noslip 5 post-processes that constraint too:
  on the same arm streams it drives the nail 0.75 mm per blow, the training physics (no noslip, as MuJoCo Warp must
  run) 4.74 mm, at either step size; without noslip the nail also creeps ~0.1 mm while the hammer winds up. The GPU
  follows the no-noslip CPU closely (above), so the gap is CPU-side and plant-side: recalibrate the nail for the
  no-noslip physics (resistance, `friction_solref`; the plan's contact recalibration lists the nail's solref), or run
  the validator without noslip for the nail. Until then, validator and training disagree on how far a blow drives.
- **Contacts per pair (the main CPU/GPU difference on the hammer scene).** MuJoCo Warp generates one contact for
  capsule/mesh, cylinder/cylinder and cylinder/box pairs whatever `multiccd` says; CPU MuJoCo's multi-contact gives
  several. The light hammer is primitives, so every hand/handle contact and the face/nail contact differ: patch loads
  12 % apart at the median, few states with the same contact set. With multi-contact off on the CPU too the hold meets
  P1's targets and the skin loads agree to 0.16 % at the median. Choosing `PhysicsCfg(multiccd=False)` for the
  residual experiments (validator included) is a decision for the experiment; it changes the validator's grasp
  contacts, not the GPU's.
- **The face/nail contact at impact.** Single contact on both sides, MuJoCo Warp still misses the cylinder/cylinder
  contact in a sixth of the strike states where the CPU finds one. Blows agree to a few percent in advance and peak
  force all the same (above).
- **The library must match the base.** The library records the base the CPU ran (`residual_config`'s grip loop and
  profile); `BaseLibrary` warns on a mismatch. The current library was recorded with the open-loop constant grip
  (feat/residual-l2 b46b62a); it lives on the desktop (`~/scratch/tactile-gpu-runs/m2/lib/hammer_v1.npz`; `*.npz`
  files are not committed).
- **Peak force is spiky.** A blow's largest single-step force can be a one-sample solver spike (1 in 203 blows at the
  training physics, 9 in 203 at the validator's, up to ~10x the pulse). `peak2` (the second-largest sample) is the
  robust metric. With the old closed grip loop the spikes depended on the drives' command jitter; CPU replays draw
  the jitter from the entry's index, so they do not depend on what a World played before.
- **Hooks are launch-bound** (above): ~50 % of the step at 4096 worlds. Fusing them is the next speed step; the scan's
  float64 ADC chain should stay unfused (or be checked) to keep the frame-for-frame parity.
- **The GPU hold creeps ~10x more than the CPU** (M1): 0.14 mm in 1 s at 0.25 ms against 0.011 mm (CPU, no noslip). The
  contact recalibration is the remedy.
- **MuJoCo version drift** (M1): the validator (3.14) and the training venv (3.11) compile mesh frames differently;
  record the library in the training venv. The cross-venv test is still missing.
- **Mirrors.** The drill-trigger interlock now applies on task "drill" only, on both sides; the GPU base reads the
  read-back angles as the CPU's does; the stale-input decay is mirrored.

## What remains

- **The GPU load generator** (`model/loadgen.py`) and **the drill** (plant, pistol driver, trigger, DrillBase): the
  interlock is ready for task "drill". The drill to port is the 12 V model: `plants/drivetrain.py` (a small ODE,
  written to vectorize) plus the tip contact and screw/hole laws of `plants/drill.py`; parity against the CPU
  plant, and the dyno's ratings (`model/plants/drill_dyno.py`), on the batched version.
- **Rewards** from the plan (substep-integrated slip path, nail advance, rebound energy, I²t, thermal margin,
  smoothness): the hammer's are placeholders.
- **Randomization events** independent of the library (the library entries carry one `residual_dr` draw each, applied at
  reset), and the forecast's error model per env.
- **Curriculum** C0–C4, **contact recalibration** (static-hold creep, impulse slip, ripple, and the nail's
  resistance without noslip), **teacher PPO** (GRU 256, zero-initialized output) and **DAgger** students, ONNX
  export.
- **Speed:** CUDA graphs or Warp kernels for the hooks; dropping mjlab's per-step `forward()`.
