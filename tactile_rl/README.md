# tactile_rl: GPU training env for the tactile residual (milestones 1–4)

GPU-parallel training of the WUJI Hand 2 residual (L2) on MuJoCo Warp through mjlab 1.6.0, mirroring the CPU testbed
(`tactile_sim`), which stays the reference and the validator. Plan: `~/.claude/plans/i-want-to-build-tidy-axolotl.md`
(sections "Training recipe", "Implementation §2", "Verification").

- **Milestone 1** built the reduced Vega U + WUJI scene, the WUJI MIT-mode drive in torch, the residual's action map,
  resets from settled states, the hold tasks, P0/P1 parity and the throughput benchmark (results kept below).
- **Milestone 2** adds the hammer task on the residual experiments' configuration: the robot's sensors on the GPU (WUJI
  readback, TaxelScan rev3 with its load model, codec and frame timing), the nail plant, the base library of scripted
  swings recorded on the CPU and played on the GPU, the student / teacher / critic observations, the stale-input decay,
  and parity evidence for each against the testbed.
- **Milestone 3** takes in the hard nail and the single-contact validator (feat/residual-l2 51e7ca1), re-records the
  library and runs P3; then the plan's rewards integrated over the physics substeps, the load-generator task
  (`Tactile-LoadGen-VegaWuji-*`), the reset randomization events, the curriculum C0–C2, and the GRU teacher with PPO and
  its first 1 kHz run.
- **Milestone 4** rescales the load generator to this hand's capacity, gives the hammer's base a per-episode hold level,
  adds C3 (70 % hammer swings, 30 % load generator) as `Tactile-Residual-VegaWuji-*`, trains the 1 kHz and 250 Hz
  teachers through C0–C3 on one simulated-time budget, evaluates them against the zero residual at matched hold levels,
  and exports the teacher actor to ONNX.

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
| `plants/loadgen.py` | `TorchLoadGen`: `model/loadgen.py` batched (M3): per-env schedules of up to 16 events (impulse, periodic force or torque with drifting amplitude, breakthrough or appearing step, bind, push) drawn as `sample_schedule` draws them, the summed wrench (`LoadEvent.value`), `forecast_truth`, each event's window; `LoadApplicator` puts the wrench on the tool through `xfrc_applied` (force at the hammer face, torque about the handle). |
| `mdp/runtime.py` | `TactileRuntime`, one per env: the substep clock, the sensors, the plant, the student's 1 kHz ring, the truth; the post-substep pipeline and the stale check. M3: pre-substep hooks (the load generator) and the reward integrands (`acc`, zeroed every policy step). |
| `mdp/loadgen_command.py` | `LoadGenCommand` (M3): the load generator's base: the hold (arm at hover with each env's droop, the synergy at the hold scalar from the read-back angles), the schedule's loads every substep, the forecast of its next event, the per-event slip scoring for the curriculum and the metrics. |
| `mdp/randomization.py` | The reset randomization (M3): `pair_solref_solimp`, `randomize_wuji_drive`, `randomize_taxelscan`, `randomize_plant`, `randomize_forecast` over ResidualDRCfg plus the plan's DR-table entries (`DRExtraCfg`), per env, scaled by the curriculum; the hammer's inertial for any head mass from a table compiled on the CPU; `TactileDR.param_vector` (33 values with the base's hold level, teacher / critic). |
| `mdp/curriculum.py` | `TactileCurriculum` (M3): stages C0 (hold, DR ×0.1, clean forecast), C1 (loads ×0.25, DR ×0.3, clean forecast), C2 (full band, full DR, noisy forecast masked in 25 % of episodes), C3 (M4: 70 % hammer episodes); advances when, over the episodes that began in the stage, drops < 1 % and 80 % of events stay within the slip target, or after `stage_cap_s` simulated seconds in it (M4). |
| `mdp/mixed_command.py` | `MixedBaseCommand` (M4): the hammer's `BaseTrajectoryCommand` and the `LoadGenCommand` side by side; per env and episode one of them (`hammer_frac`), the task block telling the student which. |
| `tasks/residual.py` | `Tactile-Residual-VegaWuji-{rate}Hz` (M4): the curriculum C0–C3 in one task (the hammer's scene, rewards and terminations, the load generator's events and DR), per-kind metrics, mjlab's NaN guard on the observations. |
| `export/onnx_export.py` | The teacher actor as an ONNX graph for the CPU runner (`obs, h_in → action, h_out`, normalizer folded in, opset 17) with its sidecar, and the torch / onnxruntime parity check on recorded inputs (M4). |
| `bench/` | `throughput` (M1–M2), `teacher_eval` (M3), and M4's `loadgen_calibration` (single events by kind × amplitude × hold level; the band check), `hammer_hold` (every library swing at a hold level), `residual_eval` (policies at matched hold levels on the hammer and the load generator), `curves` (learning curves against simulated seconds). |
| `algos/` | `ZeroInitRNNModel` (rsl_rl's RNNModel with a zero output layer) and `ScheduledPPO` (the entropy schedule), M3. |
| `mdp/commands.py` | `BaseTrajectoryCommand`: the library on the GPU, per-env entry, start state and randomization, perturbations (time warp, amplitude, start delay), the 100 Hz arm stream, the synergy scalar, the lookahead, the forecast. |
| `mdp/forecast.py` | `TorchForecastModel`: `behaviors.forecast.ForecastModel` batched, every error parameter per env. |
| `mdp/obs_builder.py` | `StudentObs`: `ResidualHandL2.fast_sample` / `observe` batched (obs_spec layout, `stack` deep, masks). |
| `mdp/actions.py` | `ResidualHandAction`: tanh, offsets, the drill-only trigger interlock, the stale-input decay, slew, composition with the base (the synergy at the library's scalar from the read-back angles). `SubstepHooks`: arm targets, plant pre-step, drives. |
| `mdp/observations.py` | `student_obs` and the privileged terms (`tool_in_hand`, `patch_contact`, `event_truth`, `nail_state`, `load_truth`, `drive_internals`, `randomization`); M1's placeholder terms for the hold tasks. |
| `mdp/rewards.py` | The plan's rewards, substep-integrated (M3): I²t, torque roughness about its 50 Hz low-pass, torque above rated, the hand's joint-limit loads, thermal margin, residual size; slip path in the hand (mm, degrees), drift from the reference pose, nail advance, rebound energy; the drop's and the thermal fault's penalties. |
| `mdp/events.py`, `terminations.py` | M1's bank and `reset_from_state_bank`; `tactile_model_fields` (declares the per-world fields); `base_done`, `tool_left_hand` (2 cm or 30° from the reference pose). |
| `tasks/` | `hammer.py`: `Tactile-Hammer-VegaWuji-{1000,500,250,100}Hz` and `hammer_scene_env_cfg` (the scene without base or sensors, for the parity scripts); `loadgen.py`: `Tactile-LoadGen-VegaWuji-*` (M3); `common.py`: the groups, rewards, terminations and DR events the tasks share; `rl_cfg.py`: the PPO settings per rate and `teacher_runner_cfg`; `hold.py`: M1's `Tactile-Hold-VegaWuji-*`. |
| `parity/` | `model_fields` (P0), `one_step` (P1, also on strike states), `taxel_load` (the load model against the CPU's foundation), `strikes` (library entries on the GPU against the CPU), `p3` (M3: the plan's P3 over four conditions). |
| `tests/rl/` | `test_rl_sensors.py` (codec, readback, rev3, forecast, student observation against the CPU), `test_rl_hammer.py` (the hammer task; M4: the hold level), `test_rl_loadgen.py` (M3: the load generator against the CPU's, the task, integrated rewards across rates, randomization, the curriculum, the teacher; M4: the C3 mix, stage caps, each stage on its own episodes), `test_rl_export.py` (M4: the ONNX export's parity and sidecar), M1's tests. CPU: `tests/test_base_library.py`, `tests/test_residual_l2.py`. |

## How to run

On the desktop, the training venv is `~/tactile-encode/.venv-rl` (torch 2.14 cu130, MuJoCo / MuJoCo Warp 3.11, mjlab
1.6.0). The validator venv is `~/tactile-encode/.venv` (MuJoCo 3.14). From the repo root:

```bash
# CPU suite (validator venv): the GPU tests skip themselves there
~/tactile-encode/.venv/bin/python -m pytest -q -m "not network"
~/tactile-encode/.venv/bin/python -m ruff check .

# GPU tests (training venv), ~10 min (the hammer tests record a small library on the CPU first); -s prints reports
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m pytest -q tests/rl

# the base library (M3: the hard nail, multiccd off, ~8 swings per episode until the nail is in; 100 episodes give
# 776 swings in ~13 min on 10 workers). Regenerate it whenever residual_config's base or scene changes.
~/tactile-encode/.venv/bin/python -m tactile_sim.behaviors.base_library export \
    --episodes 100 --strikes 10 --workers 10 --out runs/base_library/hammer_v2.npz

# P3: the GPU conditions in the training venv, the CPU ones in the validator's, then merged
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.parity.p3 --lib runs/base_library/hammer_v2.npz \
    --skip cpu_8k_noslip5,cpu_4k --out p3_gpu.json
PYTHONPATH=. ~/tactile-encode/.venv/bin/python -m tactile_rl.parity.p3 --lib runs/base_library/hammer_v2.npz \
    --skip gpu_4k,gpu_2k --out p3_cpu.json
PYTHONPATH=. ~/tactile-encode/.venv/bin/python -m tactile_rl.parity.p3 --lib - --combine p3_cpu.json p3_gpu.json

# the load-generator teacher at 1 kHz (tensorboard under --log-root; ~1.8 h for 600 iterations), then against the
# zero residual on the same episodes
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.train Tactile-LoadGen-VegaWuji-1000Hz \
    --env.scene.num-envs 4096 --agent.max-iterations 600 --agent.algorithm.entropy-decay-iterations 600 \
    --agent.algorithm.desired-kl 0.02 --log-root runs/rsl_rl
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.bench.teacher_eval <run>/model_599.pt \
    --modes zero,teacher,zero@hold,teacher@hold
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.bench.throughput --task loadgen --stage 2 \
    --rate 1000 250 --envs 1024 4096 --seconds 0.2

# M4: the load generator's calibration (single events by kind, amplitude and hold level; then the band check), the
# hammer base at reduced hold levels
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.bench.loadgen_calibration --envs 4096 --out calib.json
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.bench.loadgen_calibration --envs 4096 --band
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.bench.hammer_hold --lib runs/base_library/hammer_v2.npz

# M4: the teachers through C0-C3 (1000 iterations of 4096 envs x 0.128 s = 524 288 simulated s at either rate; ~3 h
# at 1 kHz), then the evaluation at matched hold levels, the ONNX export with its parity check, the curves
export TACTILE_BASE_LIBRARY=runs/base_library/hammer_v2.npz
for RATE in 1000 250; do
  PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.train Tactile-Residual-VegaWuji-${RATE}Hz \
      --env.scene.num-envs 4096 --agent.max-iterations 1000 --agent.algorithm.entropy-decay-iterations 1000 \
      --agent.algorithm.desired-kl 0.02 --agent.run-name teacher_$RATE --log-root runs/rsl_rl
done
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.bench.residual_eval --library $TACTILE_BASE_LIBRARY \
    --policies zero@1000,t1k=<run_1k>/model_999.pt@1000,zero@250,t250=<run_250>/model_999.pt@250 --out eval.json
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.export.onnx_export <run_1k>/model_999.pt \
    teacher_1k.onnx --task mixed --rate 1000 --record 64x160 --library $TACTILE_BASE_LIBRARY
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python -m tactile_rl.bench.curves curves <run_1k>:1kHz <run_250>:250Hz

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
- **Rewards (M3)**, integrated over the substeps (`TactileRuntime.acc`; the reward manager multiplies by the policy
  step, so each term returns its integral over the step divided by the step): +1 per mm of nail advance; −1 per mm
  and −1 per degree of slip path (the tool's pose in the hand on the 1 kHz clock, low-passed at 30 Hz so the grasp's
  ringing is not slip); −1 per (mm + degree)·s of drift from the entry's start pose; −10 per J of rebound energy
  (the tool's kinetic energy relative to the hand, mean over [5, 60] ms after a blow's onset); the effort terms: −1 per
  s of mean (i / i_rated)², −10 per s of torque roughness about its 50 Hz low-pass and of torque above rated (per rated
  torque, squared), −0.1 per s of joint-limit load (limit constraint force per rated torque, squared), −10 per s of
  thermal margin, −0.1 per s of residual size; −50 for a lost grasp, −20 for a thermal fault. The same at every rate
  (`test_rewards_are_integrated_over_substeps`).

## How the load-generator env works (M3)

- **Scene and episodes.** The hammer task's scene (`residual_training_config`), the nail idle. Episodes of 3 s start
  from the bank of settled grasps (M1's `reset_from_state_bank`), then the reset randomization draws per env. The
  drift reference is the tool's pose in the hand at 0.2 s, once the reset has settled; the episode ends on a time-out,
  a lost grasp (2 cm or 30° from it, −50), a thermal fault (−20) or a non-finite state.
- **Base.** The hold: the arm on its servos at the hover pose with each env's droop for its randomized gains, the hand
  at a fraction of the grip synergy's hold scalar (the base's hold level, drawn per episode: C0 1.0, C1 0.5–1.0, C2
  0.3–0.8) from the read-back angles; the lookahead is zero.
- **Loads.** A schedule per env and episode, drawn as `tactile_sim.model.loadgen.sample_schedule` draws it (Poisson,
  2 events per second, from 0.2 s): half-sine impulses of 0.5–5 ms and 14–200 N within 20° of the strike axis; a
  periodic force along it (1.25–10 N) or torque about the handle (0.05–0.4 Nm) at 1–100 Hz with a drifting amplitude;
  steps of 2.7–12 N (a breakthrough or an appearing push); binds of 0.1–0.6 Nm for 20–80 ms; pushes of 1.5–12 N (M4's
  band, this hand's capacity; M3 ran 50–700 N, 5–40 N, 0.2–1.5 Nm, 20–90 N, 0.5–3 Nm, 10–80 N). Before
  every substep their sum goes on the tool through `xfrc_applied`: the force at the hammer face (tool frame: x along
  the handle, z from the face into the head, so an impulse pushes back like a blow), the torque about the handle.
  The batched schedules match the CPU sampler's statistics and give the CPU's wrench and forecast truth to 1e-9 on the
  same events (`test_rl_loadgen.py`).
- **Forecast.** The schedule's next event (or the one under way) through the torch ForecastModel; the error model per
  env from `randomize_forecast`.
- **Events scored.** An event's net slip is the tool's pose in the hand 50 ms after it against its pose when it began;
  within 0.7 mm and 1.4° it meets the target (the plan's G3 slip numbers). Logged per episode: events, the fraction
  within the target, net slip and slip path per event, the hold level (`Metrics/base/*`); the curriculum's window
  (recent episodes): drop rate, events within the target.
- **Curriculum.** C0 the hold (DR ×0.1, clean forecast) → C1 loads ×0.25, base at 0.5–1.0 of the hold, DR ×0.3, clean
  forecast → C2 the full band, base at 0.3–0.8, full DR, noisy forecast masked in 25 % of episodes; a stage advances
  when drops < 1 % and 80 % of events meet the target.
- **Observations.** `teacher` = the student's 740 + 149 privileged (the tool in the hand, patch contacts, the event
  truth, the wrench on the tool, the drives, the 33 randomization values); the registered task returns `teacher` only
  (the critic reads it too), the test configurations every group.

## How C3 and Tactile-Residual work (M4)

- **One task for the whole curriculum.** `Tactile-Residual-VegaWuji-{rate}Hz` runs C0–C2 as the load generator's task
  does and C3 on top: the hammer's scene (the nail live), rewards (nail advance and rebound are zero on a
  load-generator env) and terminations (an entry that played out ends its episode), the load generator's reset bank,
  randomization and per-event scoring. `MixedBaseCommand` drives both bases on every env and hands each env's terms
  the one it plays: at reset an env plays a library swing with probability `hammer_frac` (C3: 0.7) and the load
  generator otherwise; a hammer env gets no loads, a load-generator env leaves the nail idle; the student's task block
  (obs_spec `task`) says which. A hammer env's randomization is its entry's (written into the DR state, so the
  teacher's 33 randomization values show it) and its base grips at the episode's hold level.
- **Hold level on the hammer.** `BaseTrajectoryCommandCfg.hold_level`: the library's synergy scalar s(u) times a level
  drawn per episode (C0 1.0, C1 0.5–1.0, C2 and C3 0.3–0.8), as on the load generator.
- **Stages.** Judged on the episodes that began in the stage (each env remembers its episode's stage: the episodes under
  way at a change ran the old stage's settings) and capped at 120 000 simulated seconds per gated stage, so a budget
  reaches C3 (`stage_cap_s`; `Curriculum/stage/capped` counts the capped advances). C3 is the last stage here.
- **Teacher.** The M3 teacher (GRU 256, (256, 128) head with a zero output layer, MLP critic on the same group, robust
  normalizer) at desired KL 0.02; discount horizon 0.5 s (the hammer's) through all stages; 894 teacher inputs at
  1 kHz (the student's 740 + 154 privileged, both plants' truth), 2580 at 250 Hz (the student's 2426 with every fast
  block 4 samples deep).
- **Metrics.** Per kind: the load generator's per-event values (`Metrics/base/*`, zero on hammer envs), the hammer's
  net slip per strike that kept the grasp (`Episode_Metrics/hammer_net_*` over `hammer_strike`), lost grasps, slip
  path and I²t for all envs and for the hammer's (`Episode_Metrics/hammer_*`); `bench/curves.py` separates the kinds
  with the logged hammer share.
- **NaN guard.** The observations run mjlab's `nan_policy="warn"` (a non-finite value is printed with its envs and
  zeroed), so one env cannot end a long run; the tests use "error".

## Milestone 4 results (desktop, RTX 5070 Ti, 2026-10-08/09)

### The load generator at this hand's capacity (`python -m tactile_rl.bench.loadgen_calibration`)

One event per episode at 0.4 s, of one kind at a fixed amplitude (durations, frequencies and directions drawn as
`LoadGenCfg` draws them), the base at a hold level, zero residual, the full randomization, 128 episodes per cell. Per
cell: lost grasps (2 cm or 30° from the reference by the end of the event's window) / net slip per event (the tool's
pose in the hand 50 ms after the event against its start, kept grasps; mm / °). A force at the hammer face acts 17 cm
from the grasp, so steady and periodic forces are what the grasp tolerates least, and short impulses the most.

| impulse (N) | hold 0.3 | hold 0.5 | hold 0.8 | hold 1.0 |
|---|---|---|---|---|
| 25 | 0 % / 0.09 / 0.22 | 0 % / 0.03 / 0.08 | 0 % / 0.01 / 0.04 | 0 % / 0.01 / 0.04 |
| 50 | 0 % / 0.28 / 0.79 | 0 % / 0.13 / 0.35 | 0 % / 0.09 / 0.23 | 0 % / 0.10 / 0.24 |
| 75 | 0 % / 0.61 / 1.82 | 0 % / 0.29 / 0.92 | 0 % / 0.20 / 0.61 | 0 % / 0.18 / 0.43 |
| 100 | 0 % / 0.89 / 3.05 | 0 % / 0.51 / 1.55 | 0 % / 0.29 / 0.86 | 0 % / 0.29 / 0.82 |
| 150 | 0 % / 1.65 / 5.55 | 0 % / 1.14 / 3.12 | 0 % / 0.80 / 1.93 | 0 % / 0.67 / 1.81 |
| 200 | 1 % / 2.23 / 8.27 | 0 % / 1.82 / 4.64 | 0 % / 1.35 / 3.38 | 0 % / 1.28 / 3.06 |
| 300 | 10 % / 3.06 / 11.33 | 1 % / 2.85 / 8.21 | 0 % / 2.55 / 6.18 | 0 % / 2.37 / 5.93 |
| 450 | 27 % / 3.51 / 12.76 | 2 % / 3.96 / 12.01 | 4 % / 3.53 / 8.57 | 2 % / 3.44 / 9.26 |

| periodic_force (N) | hold 0.3 | hold 0.5 | hold 0.8 | hold 1.0 |
|---|---|---|---|---|
| 2.5 | 1 % / 1.66 / 3.92 | 0 % / 0.15 / 0.52 | 0 % / 0.08 / 0.14 | 0 % / 0.08 / 0.18 |
| 5 | 18 % / 3.87 / 8.49 | 0 % / 1.33 / 3.36 | 0 % / 0.19 / 0.46 | 0 % / 0.12 / 0.29 |
| 10 | 36 % / 4.23 / 9.44 | 22 % / 3.27 / 8.82 | 1 % / 1.03 / 3.16 | 0 % / 0.46 / 1.38 |
| 15 | 50 % / 4.69 / 9.96 | 45 % / 2.91 / 8.80 | 11 % / 2.08 / 7.21 | 4 % / 2.24 / 6.78 |
| 20 | 63 % / 5.21 / 10.75 | 50 % / 3.91 / 10.54 | 23 % / 2.29 / 8.70 | 20 % / 3.73 / 10.40 |
| 30 | 66 % / 5.90 / 11.43 | 54 % / 4.77 / 12.09 | 48 % / 3.71 / 11.65 | 43 % / 2.61 / 8.81 |
| 40 | 74 % / 6.38 / 12.14 | 57 % / 4.72 / 10.50 | 49 % / 3.16 / 11.35 | 48 % / 2.65 / 11.36 |
| 60 | 87 % / 7.74 / 14.11 | 69 % / 4.91 / 9.68 | 61 % / 3.62 / 11.95 | 62 % / 3.20 / 11.51 |

| periodic_torque (Nm) | hold 0.3 | hold 0.5 | hold 0.8 | hold 1.0 |
|---|---|---|---|---|
| 0.05 | 0 % / 0.37 / 0.98 | 0 % / 0.08 / 0.19 | 0 % / 0.09 / 0.16 | 0 % / 0.09 / 0.20 |
| 0.1 | 0 % / 0.77 / 1.99 | 0 % / 0.08 / 0.22 | 0 % / 0.09 / 0.16 | 0 % / 0.09 / 0.18 |
| 0.15 | 3 % / 1.48 / 4.57 | 0 % / 0.18 / 0.45 | 0 % / 0.08 / 0.16 | 0 % / 0.09 / 0.20 |
| 0.2 | 11 % / 2.26 / 7.53 | 0 % / 0.30 / 1.37 | 0 % / 0.11 / 0.24 | 0 % / 0.12 / 0.28 |
| 0.3 | 48 % / 3.67 / 12.52 | 4 % / 0.57 / 3.73 | 0 % / 0.30 / 0.71 | 0 % / 0.24 / 0.61 |
| 0.5 | 65 % / 5.39 / 17.13 | 57 % / 0.99 / 7.53 | 5 % / 0.49 / 2.54 | 2 % / 0.45 / 1.70 |
| 0.75 | 83 % / 6.57 / 16.67 | 89 % / 2.32 / 13.97 | 34 % / 0.54 / 6.65 | 20 % / 0.54 / 4.08 |
| 1 | 89 % / 7.21 / 17.97 | 88 % / 3.55 / 15.58 | 73 % / 0.57 / 9.56 | 54 % / 0.58 / 6.33 |

| step (N) | hold 0.3 | hold 0.5 | hold 0.8 | hold 1.0 |
|---|---|---|---|---|
| 5 | 0 % / 0.85 / 1.76 | 0 % / 0.06 / 0.14 | 0 % / 0.06 / 0.16 | 0 % / 0.07 / 0.16 |
| 10 | 2 % / 2.46 / 8.03 | 0 % / 1.99 / 3.22 | 0 % / 0.83 / 1.27 | 0 % / 0.34 / 0.55 |
| 20 | 30 % / 4.85 / 11.41 | 0 % / 3.32 / 5.25 | 0 % / 2.57 / 4.03 | 0 % / 2.49 / 3.81 |
| 30 | 100 % / - | 2 % / 7.25 / 13.80 | 0 % / 3.69 / 6.18 | 0 % / 3.20 / 5.19 |
| 45 | 100 % / - | 100 % / - | 18 % / 7.78 / 13.57 | 0 % / 5.40 / 9.79 |
| 60 | 100 % / - | 100 % / - | 98 % / 8.99 / 17.82 | 66 % / 7.86 / 12.52 |
| 90 | 100 % / - | 100 % / - | 100 % / - | 100 % / - |
| 120 | 100 % / - | 100 % / - | 100 % / - | 100 % / - |

| bind (Nm) | hold 0.3 | hold 0.5 | hold 0.8 | hold 1.0 |
|---|---|---|---|---|
| 0.1 | 0 % / 0.12 / 0.33 | 0 % / 0.01 / 0.02 | 0 % / 0.01 / 0.01 | 0 % / 0.01 / 0.01 |
| 0.2 | 0 % / 0.32 / 1.82 | 0 % / 0.05 / 0.20 | 0 % / 0.01 / 0.04 | 0 % / 0.01 / 0.02 |
| 0.3 | 2 % / 1.11 / 10.00 | 0 % / 0.24 / 1.55 | 0 % / 0.06 / 0.24 | 0 % / 0.04 / 0.13 |
| 0.5 | 59 % / 1.34 / 15.45 | 7 % / 0.54 / 7.03 | 0 % / 0.32 / 1.79 | 0 % / 0.22 / 1.13 |
| 0.75 | 88 % / 0.72 / 11.06 | 52 % / 0.62 / 15.16 | 2 % / 0.55 / 8.53 | 0 % / 0.54 / 4.07 |
| 1 | 94 % / 0.53 / 7.10 | 84 % / 0.65 / 18.49 | 30 % / 0.53 / 13.63 | 12 % / 0.58 / 8.74 |
| 1.5 | 98 % / 0.23 / 8.27 | 99 % / 0.62 / 20.67 | 86 % / 0.53 / 20.71 | 65 % / 0.50 / 18.71 |
| 2 | 99 % / 0.24 / 24.66 | 100 % / - | 97 % / 0.52 / 27.54 | 91 % / 0.46 / 23.37 |

| push (N) | hold 0.3 | hold 0.5 | hold 0.8 | hold 1.0 |
|---|---|---|---|---|
| 5 | 0 % / 0.87 / 1.79 | 0 % / 0.10 / 0.18 | 0 % / 0.10 / 0.21 | 0 % / 0.12 / 0.22 |
| 10 | 3 % / 2.47 / 7.95 | 0 % / 2.01 / 3.27 | 0 % / 0.89 / 1.33 | 0 % / 0.40 / 0.61 |
| 20 | 28 % / 4.83 / 11.51 | 0 % / 3.33 / 5.28 | 0 % / 2.59 / 4.04 | 0 % / 2.52 / 3.83 |
| 30 | 100 % / - | 2 % / 7.29 / 13.71 | 0 % / 3.69 / 6.19 | 0 % / 3.23 / 5.21 |
| 45 | 100 % / - | 100 % / - | 18 % / 7.80 / 13.58 | 0 % / 5.40 / 9.85 |
| 60 | 100 % / - | 100 % / - | 97 % / 9.06 / 17.83 | 66 % / 7.89 / 12.50 |
| 80 | 100 % / - | 100 % / - | 100 % / - | 100 % / - |
| 100 | 100 % / - | 100 % / - | 100 % / - | 100 % / - |

The old band (impulses 50–700 N, periodic 5–40 N and 0.2–1.5 Nm, steps 20–90 N, binds 0.5–3 Nm, pushes 10–80 N)
reached the rows where the full hold loses a fifth to all of its grasps. The new band scales each kind's range by one
factor so that its top sits at the full hold's no-loss edge: impulses 14–200 N (×0.29; the light hammer's own blows
peak ~185 N), periodic 1.25–10 N and 0.05–0.4 Nm (×0.25–0.27), steps 2.7–12 N (×0.13), binds 0.1–0.6 Nm (×0.2),
pushes 1.5–12 N (×0.15); rates, kinds' shares and durations unchanged. Shared with the CPU (`tactile_sim/config.py`).

**Band check** (`--band`): the whole band as the generator draws it (Poisson, 2 events/s from 0.2 s), 1024 episodes of
3 s per hold level, zero residual, the full randomization:

| hold level | lost grasps | events per kept episode | events within 0.7 mm / 1.4° | net slip per event (mm / °) | slip path per event (mm / °) | I²t per episode |
|---|---|---|---|---|---|---|
| 0.3 | 27.3 % | 2.57 | 38.6 % | 1.27 / 4.50 | 2.23 / 6.70 | 0.28 |
| 0.5 | 4.2 % | 2.57 | 61.8 % | 0.62 / 2.10 | 1.21 / 3.42 | 0.68 |
| 0.8 | 0.1 % | 2.58 | 81.8 % | 0.31 / 0.80 | 0.59 / 1.42 | 1.50 |
| 1.0 | **0.0 %** | 2.63 | 86.7 % | 0.26 / 0.61 | 0.45 / 1.02 | 2.20 |

The full hold keeps every grasp (the target was ≥ 95 %) and 87 % of its events within the slip target; at 0.3 of the
hold 27 % of the episodes lose the grasp. C2's gate (80 % within, < 1 % lost) is now passable by a residual that
restores the hold's grip over a base at 0.3–0.8 of it.

### The hammer's base at reduced hold levels (`python -m tactile_rl.bench.hammer_hold`)

Every library swing (776) replayed on the GPU, unperturbed, with its entry's randomization, zero residual, the hand's
synergy scalar times the hold level. Per strike: hits (the face met the nail), lost grasps (2 cm or 30° from the
entry's start at any point), the tool's net slip in the hand from the swing's start to the entry's end (kept
strikes), the nail's advance:

| hold level | hits | lost grasps | net slip per strike (mm / °) | p90 (mm / °) | advance per kept hit (mm) |
|---|---|---|---|---|---|
| 0.3 | 37.8 % | **54.8 %** | 10.0 / 13.3 | 23.2 / 23.1 | 1.59 |
| 0.5 | 87.4 % | 7.6 % | 4.38 / 8.03 | 15.9 / 17.2 | 2.42 |
| 0.8 | 100 % | 0 % | 1.23 / 1.84 | 1.67 / 3.41 | 2.71 |
| 1.0 | 100 % | 0 % | 1.11 / 1.75 | 1.61 / 3.00 | 2.69 |

The base holds down to 0.8 of its grip (+11 % translation, +5 % rotation slip); below it degrades fast: at 0.5 one
strike in thirteen loses the hammer and the kept ones slip four times as much, at 0.3 more than half lose it (the
load generator's hold loses 27 % of its grasps at 0.3 over 3 s of the band).

### Two fixes from the first C3 run

- **Each stage judged on its own episodes.** The first 1 kHz run went C1 → C2 → C3 in 11 iterations: the episodes
  under way at a stage change ran the old stage's settings but counted for the new stage's gate, so C2 passed on
  episodes that began in C1. Each env now remembers the stage its episode began in.
- **The hammer's "no event" is 0, not NaN.** The base library stores the forecast's event time as NaN where no event
  is planned (after the contact: the last 36 % of every entry); the hammer's truth carried it and the
  privileged `event_truth` term computed (t_event − t)·valid = NaN there. rsl_rl's NaN check stopped the run 6
  iterations into C3. The truth is now 0 where invalid (as the load generator's), and the observations carry mjlab's
  NaN guard (`nan_policy="warn"`; none fired in the runs below).

### The teachers through C0–C3 (1 kHz and 250 Hz, 524 288 simulated s each)

`python -m tactile_rl.train Tactile-Residual-VegaWuji-{1000,250}Hz --env.scene.num-envs 4096 --agent.max-iterations
1000 --agent.algorithm.entropy-decay-iterations 1000 --agent.algorithm.desired-kl 0.02`: the M3 teacher (GRU 256,
robust normalizer, desired KL 0.02, entropy 1e-3 → 0 over the run), one budget in simulated time for both rates:
1000 iterations of 4096 envs × 0.128 s = **524 288 simulated s** (146 simulated hours). At 1 kHz an iteration is 128
steps (524 k samples), at 250 Hz 32 steps (131 k samples): the same simulated experience and the same number of
updates, a quarter of the samples. One seed each.

| | 1 kHz | 250 Hz |
|---|---|---|
| C0 → C1 (passed) | iteration 11 (6 k s) | iteration 12 (7 k s) |
| C1 → C2 (passed: 99.8 % of events within the target, no drops) | iteration 47 (25 k s) | iteration 50 (27 k s) |
| C2 → C3 | **capped** at iteration 276 (145 k s): 72 % within, 0.9 % drops | **capped** at iteration 279 (147 k s): 67 % within, 1.7 % drops |
| C3 | 379 k s (723 iterations) | 377 k s (720 iterations) |
| wall time; s per iteration (C0 / C3) | 3 h 34 min; 9.8 / 12.3 | 2 h 35 min; 7.4 / 9.6 |
| GPU memory | 10.1 GB | 7.8 GB |

Neither teacher passed C2's 80 % gate within its cap (the 1 kHz teacher met the drop gate, < 1 %, by then); C3 keeps C2's
load-generator episodes as its 30 %, and at the end both teachers meet the gate's numbers on C2's settings (the
evaluation below: 74–84 % within the target over holds 0.3–0.8, ≤ 0.4 % lost). No NaN warning fired in either run.

**Learning curves** (training metrics with the exploration noise on: σ grew from 0.30 to 0.48 at 1 kHz and from 0.15
to 0.21 at 250 Hz; window means against simulated seconds, per kind from the logged hammer share;
`python -m tactile_rl.bench.curves`; PNG and CSVs: `~/scratch/tactile-gpu-runs/m4/train/curves_m4{.png,_1kHz.csv,
_250Hz.csv}` on the desktop):

| simulated s | return 1k / 250 | load gen. net slip per event mm (1k / 250) | load gen. slip path per episode mm | load gen. lost grasps | load gen. I²t per episode | hammer net slip per strike mm | hammer slip path per episode mm | hammer lost grasps | hammer I²t per episode |
|---|---|---|---|---|---|---|---|---|---|
| 0–25 k (C0–C1) | -3.5 / -3.3 | 0.01 / 0.01 | 0.35 / 0.33 | 0.0 % / 0.0 % | 1.43 / 1.46 | - / - | - / - | - / - | - / - |
| 25–85 k (C1–C2) | -27.8 / -29.6 | 0.53 / 0.54 | 4.60 / 4.67 | 2.9 % / 3.6 % | 0.98 / 0.96 | - / - | - / - | - / - | - / - |
| 85–145 k (C2) | -26.4 / -30.6 | 0.57 / 0.61 | 4.64 / 5.12 | 1.5 % / 2.3 % | 0.95 / 0.86 | - / - | - / - | - / - | - / - |
| 145–200 k (C3) | -35.3 / -41.3 | 0.47 / 0.56 | 3.77 / 4.66 | 0.6 % / 1.7 % | 1.00 / 0.88 | 1.85 / 2.01 | 10.0 / 11.3 | 0.4 % / 0.6 % | 0.37 / 0.31 |
| 200–300 k (C3) | -28.5 / -31.1 | 0.44 / 0.50 | 3.37 / 3.88 | 0.2 % / 0.6 % | 1.04 / 0.90 | 1.41 / 1.40 | 7.2 / 7.6 | 0.0 % / 0.0 % | 0.41 / 0.36 |
| 300–400 k (C3) | -24.5 / -25.0 | 0.40 / 0.44 | 2.97 / 3.27 | 0.1 % / 0.2 % | 1.08 / 0.93 | 1.39 / 1.25 | 6.1 / 6.1 | 0.0 % / 0.0 % | 0.46 / 0.40 |
| 400–524 k (C3) | -22.6 / -22.7 | 0.37 / 0.41 | 2.60 / 2.93 | 0.1 % / 0.1 % | 1.14 / 0.95 | 1.42 / 1.24 | 5.7 / 5.5 | 0.0 % / 0.0 % | 0.49 / 0.42 |

- C2 (to 145 k s): the load generator's lost grasps fall from 5 % to < 1 % and its net slip per event from 0.70 to
  0.55 mm; C3's first hammer episodes lose 8–10 % of the swings for a few iterations, then none.
- The 250 Hz teacher learns slower per simulated second (its return trails until ~300 k s; at 145–200 k s its hammer
  slip path is 11.3 against 10.0 mm and the load generator loses 1.7 % against 0.6 %) and ends level; both still
  improve at the end of the budget (not converged).
- Both teachers grip harder as they learn: the hammer's I²t per episode climbs (0.33 → 0.50 at 1 kHz, 0.23 → 0.43 at
  250 Hz) while its slip path falls by 60 %.

### The teachers against the zero residual at matched hold levels (`python -m tactile_rl.bench.residual_eval`)

Each policy at its own rate under C3's settings (the full randomization, the noisy forecast masked in 25 % of
episodes, the band at full amplitude), the base fixed at one hold level per run, 1024 episodes per cell from the same
seed, one episode per env without resets: the hammer plays every library swing in order (perturbed: time warp, amplitude,
start delay), the load generator 3 s of the band. The teachers play their mean action (`model_999.pt`); `grip` is the
strongest constant residual toward closing (every joint's offset, stiffness and feedforward at their limits: M3's
authority bound). Lost grasps over all episodes; slip and current over the kept ones; peak current after the first
0.1 s (the reset state's opening transient excluded), time at the limit = share of joint-time at |i| ≥ 0.99 rated.

**Hammer** (per strike: one library swing per episode)

| policy | hold | lost grasps | net slip per strike (mm / deg) | slip path per strike (mm / deg) | I²t per episode | RMS current / rated | peak current / rated | time at the limit | hits | advance (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| grip@1000 | 0.3 | 0.0 % | 1.122 / 2.265 | 5.13 / 11.58 | 1.230 | 0.895 | 1.00 | 33.67 % | 100.0 % | 2.73 |
| t1k@1000 | 0.3 | 0.0 % | 1.501 / 2.661 | 5.64 / 14.19 | 0.368 | 0.490 | 1.00 | 2.58 % | 100.0 % | 2.65 |
| t250@250 | 0.3 | 0.0 % | 1.223 / 2.671 | 5.65 / 14.51 | 0.301 | 0.443 | 1.00 | 2.43 % | 100.0 % | 2.62 |
| zero@1000 | 0.3 | 53.9 % | 6.436 / 12.830 | 27.21 / 58.29 | 0.087 | 0.236 | 0.88 | 0.08 % | 40.0 % | 1.45 |
| zero@250 | 0.3 | 53.9 % | 6.486 / 12.681 | 27.54 / 58.30 | 0.087 | 0.235 | 0.88 | 0.07 % | 39.6 % | 1.37 |
| grip@1000 | 0.5 | 0.0 % | 1.125 / 2.281 | 5.12 / 11.59 | 1.279 | 0.913 | 1.00 | 44.73 % | 100.0 % | 2.73 |
| t1k@1000 | 0.5 | 0.0 % | 1.445 / 2.565 | 4.97 / 12.11 | 0.472 | 0.554 | 1.00 | 2.63 % | 100.0 % | 2.66 |
| t250@250 | 0.5 | 0.0 % | 1.268 / 2.567 | 5.00 / 12.21 | 0.398 | 0.509 | 1.00 | 2.51 % | 100.0 % | 2.62 |
| zero@1000 | 0.5 | 9.6 % | 3.975 / 10.047 | 19.18 / 45.88 | 0.219 | 0.377 | 1.00 | 1.85 % | 85.9 % | 2.10 |
| zero@250 | 0.5 | 9.1 % | 4.036 / 10.165 | 19.18 / 45.94 | 0.219 | 0.377 | 1.00 | 1.86 % | 86.1 % | 2.11 |
| grip@1000 | 0.8 | 0.0 % | 1.123 / 2.278 | 5.11 / 11.58 | 1.283 | 0.914 | 1.00 | 44.73 % | 100.0 % | 2.73 |
| t1k@1000 | 0.8 | 0.0 % | 1.394 / 2.465 | 4.58 / 10.73 | 0.632 | 0.642 | 1.00 | 5.78 % | 100.0 % | 2.68 |
| t250@250 | 0.8 | 0.0 % | 1.345 / 2.496 | 4.70 / 10.93 | 0.556 | 0.602 | 1.00 | 2.83 % | 100.0 % | 2.64 |
| zero@1000 | 0.8 | 0.1 % | 1.445 / 3.553 | 8.51 / 21.59 | 0.458 | 0.547 | 1.00 | 2.84 % | 99.9 % | 2.68 |
| zero@250 | 0.8 | 0.0 % | 1.450 / 3.561 | 8.52 / 21.60 | 0.459 | 0.547 | 1.00 | 2.83 % | 100.0 % | 2.68 |
| grip@1000 | 1 | 0.0 % | 1.123 / 2.276 | 5.11 / 11.58 | 1.284 | 0.914 | 1.00 | 44.73 % | 100.0 % | 2.73 |
| t1k@1000 | 1 | 0.0 % | 1.345 / 2.371 | 4.55 / 10.45 | 0.741 | 0.695 | 1.00 | 8.77 % | 100.0 % | 2.68 |
| t250@250 | 1 | 0.0 % | 1.316 / 2.438 | 4.70 / 10.79 | 0.687 | 0.669 | 1.00 | 4.58 % | 100.0 % | 2.66 |
| zero@1000 | 1 | 0.0 % | 1.197 / 2.482 | 6.63 / 16.00 | 0.663 | 0.658 | 1.00 | 2.84 % | 100.0 % | 2.67 |
| zero@250 | 1 | 0.0 % | 1.196 / 2.463 | 6.63 / 15.99 | 0.664 | 0.658 | 1.00 | 2.84 % | 100.0 % | 2.67 |

**Load generator** (per event; 3 s episodes of the band)

| policy | hold | lost grasps | net slip per event (mm / deg) | slip path per event (mm / deg) | I²t per episode | RMS current / rated | peak current / rated | time at the limit | within 0.7 mm / 1.4 deg |
|---|---|---|---|---|---|---|---|---|---|
| grip@1000 | 0.3 | 0.0 % | 0.271 / 0.647 | 0.46 / 1.08 | 2.398 | 0.895 | 1.00 | 43.67 % | 85.3 % |
| t1k@1000 | 0.3 | 0.3 % | 0.413 / 1.002 | 0.70 / 1.64 | 0.839 | 0.530 | 1.00 | 2.70 % | 77.7 % |
| t250@250 | 0.3 | 0.4 % | 0.457 / 1.221 | 0.86 / 2.15 | 0.604 | 0.449 | 1.00 | 2.68 % | 74.0 % |
| zero@1000 | 0.3 | 22.2 % | 1.293 / 4.624 | 2.27 / 6.90 | 0.282 | 0.307 | 0.96 | 0.38 % | 38.4 % |
| zero@250 | 0.3 | 22.1 % | 1.279 / 4.564 | 2.26 / 6.88 | 0.282 | 0.306 | 0.95 | 0.38 % | 39.0 % |
| grip@1000 | 0.5 | 0.0 % | 0.272 / 0.650 | 0.46 / 1.08 | 2.430 | 0.900 | 1.00 | 43.67 % | 85.2 % |
| t1k@1000 | 0.5 | 0.1 % | 0.340 / 0.799 | 0.55 / 1.25 | 1.091 | 0.604 | 1.00 | 2.94 % | 82.1 % |
| t250@250 | 0.5 | 0.0 % | 0.379 / 0.984 | 0.69 / 1.69 | 0.832 | 0.527 | 1.00 | 2.72 % | 77.8 % |
| zero@1000 | 0.5 | 4.4 % | 0.633 / 2.056 | 1.27 / 3.47 | 0.684 | 0.478 | 1.00 | 2.66 % | 60.8 % |
| zero@250 | 0.5 | 4.3 % | 0.615 / 2.019 | 1.26 / 3.47 | 0.684 | 0.478 | 1.00 | 2.66 % | 61.8 % |
| grip@1000 | 0.8 | 0.0 % | 0.273 / 0.651 | 0.46 / 1.08 | 2.432 | 0.901 | 1.00 | 43.67 % | 85.2 % |
| t1k@1000 | 0.8 | 0.0 % | 0.305 / 0.709 | 0.47 / 1.08 | 1.514 | 0.711 | 1.00 | 7.35 % | 83.8 % |
| t250@250 | 0.8 | 0.0 % | 0.319 / 0.802 | 0.57 / 1.37 | 1.342 | 0.669 | 1.00 | 3.06 % | 81.8 % |
| zero@1000 | 0.8 | 0.1 % | 0.352 / 0.895 | 0.63 / 1.51 | 1.498 | 0.708 | 1.00 | 2.73 % | 80.1 % |
| zero@250 | 0.8 | 0.1 % | 0.341 / 0.864 | 0.63 / 1.51 | 1.499 | 0.708 | 1.00 | 2.73 % | 80.9 % |
| grip@1000 | 1 | 0.0 % | 0.273 / 0.652 | 0.46 / 1.08 | 2.433 | 0.901 | 1.00 | 43.67 % | 85.2 % |
| t1k@1000 | 1 | 0.0 % | 0.285 / 0.675 | 0.44 / 1.02 | 1.835 | 0.783 | 1.00 | 12.50 % | 84.8 % |
| t250@250 | 1 | 0.0 % | 0.278 / 0.675 | 0.49 / 1.13 | 1.856 | 0.787 | 1.00 | 6.48 % | 84.9 % |
| zero@1000 | 1 | 0.0 % | 0.273 / 0.657 | 0.47 / 1.10 | 2.205 | 0.859 | 1.00 | 2.75 % | 85.4 % |
| zero@250 | 1 | 0.0 % | 0.263 / 0.627 | 0.47 / 1.10 | 2.206 | 0.859 | 1.00 | 2.75 % | 85.9 % |

**Against the zero residual at the full hold** (same rate): net slip and current, ratio in brackets

| task | policy | hold | net slip mm (ratio) | net slip deg (ratio) | lost grasps | I²t (ratio) | RMS current (ratio) |
|---|---|---|---|---|---|---|---|
| hammer | grip@1000 | 0.3 | 1.122 (0.94) | 2.265 (0.91) | 0.0 % | 1.230 (1.85) | 0.895 (1.36) |
| hammer | grip@1000 | 0.5 | 1.125 (0.94) | 2.281 (0.92) | 0.0 % | 1.279 (1.93) | 0.913 (1.39) |
| hammer | grip@1000 | 0.8 | 1.123 (0.94) | 2.278 (0.92) | 0.0 % | 1.283 (1.93) | 0.914 (1.39) |
| hammer | grip@1000 | 1 | 1.123 (0.94) | 2.276 (0.92) | 0.0 % | 1.284 (1.94) | 0.914 (1.39) |
| hammer | t1k@1000 | 0.3 | 1.501 (1.25) | 2.661 (1.07) | 0.0 % | 0.368 (0.55) | 0.490 (0.74) |
| hammer | t1k@1000 | 0.5 | 1.445 (1.21) | 2.565 (1.03) | 0.0 % | 0.472 (0.71) | 0.554 (0.84) |
| hammer | t1k@1000 | 0.8 | 1.394 (1.16) | 2.465 (0.99) | 0.0 % | 0.632 (0.95) | 0.642 (0.98) |
| hammer | t1k@1000 | 1 | 1.345 (1.12) | 2.371 (0.96) | 0.0 % | 0.741 (1.12) | 0.695 (1.06) |
| hammer | t250@250 | 0.3 | 1.223 (1.02) | 2.671 (1.08) | 0.0 % | 0.301 (0.45) | 0.443 (0.67) |
| hammer | t250@250 | 0.5 | 1.268 (1.06) | 2.567 (1.04) | 0.0 % | 0.398 (0.60) | 0.509 (0.77) |
| hammer | t250@250 | 0.8 | 1.345 (1.12) | 2.496 (1.01) | 0.0 % | 0.556 (0.84) | 0.602 (0.91) |
| hammer | t250@250 | 1 | 1.316 (1.10) | 2.438 (0.99) | 0.0 % | 0.687 (1.03) | 0.669 (1.02) |
| hammer | zero@1000 | 0.3 | 6.436 (5.38) | 12.830 (5.17) | 53.9 % | 0.087 (0.13) | 0.236 (0.36) |
| hammer | zero@1000 | 0.5 | 3.975 (3.32) | 10.047 (4.05) | 9.6 % | 0.219 (0.33) | 0.377 (0.57) |
| hammer | zero@1000 | 0.8 | 1.445 (1.21) | 3.553 (1.43) | 0.1 % | 0.458 (0.69) | 0.547 (0.83) |
| hammer | zero@250 | 0.3 | 6.486 (5.42) | 12.681 (5.15) | 53.9 % | 0.087 (0.13) | 0.235 (0.36) |
| hammer | zero@250 | 0.5 | 4.036 (3.38) | 10.165 (4.13) | 9.1 % | 0.219 (0.33) | 0.377 (0.57) |
| hammer | zero@250 | 0.8 | 1.450 (1.21) | 3.561 (1.45) | 0.0 % | 0.459 (0.69) | 0.547 (0.83) |
| loadgen | grip@1000 | 0.3 | 0.271 (0.99) | 0.647 (0.99) | 0.0 % | 2.398 (1.09) | 0.895 (1.04) |
| loadgen | grip@1000 | 0.5 | 0.272 (1.00) | 0.650 (0.99) | 0.0 % | 2.430 (1.10) | 0.900 (1.05) |
| loadgen | grip@1000 | 0.8 | 0.273 (1.00) | 0.651 (0.99) | 0.0 % | 2.432 (1.10) | 0.901 (1.05) |
| loadgen | grip@1000 | 1 | 0.273 (1.00) | 0.652 (0.99) | 0.0 % | 2.433 (1.10) | 0.901 (1.05) |
| loadgen | t1k@1000 | 0.3 | 0.413 (1.51) | 1.002 (1.53) | 0.3 % | 0.839 (0.38) | 0.530 (0.62) |
| loadgen | t1k@1000 | 0.5 | 0.340 (1.24) | 0.799 (1.22) | 0.1 % | 1.091 (0.49) | 0.604 (0.70) |
| loadgen | t1k@1000 | 0.8 | 0.305 (1.12) | 0.709 (1.08) | 0.0 % | 1.514 (0.69) | 0.711 (0.83) |
| loadgen | t1k@1000 | 1 | 0.285 (1.04) | 0.675 (1.03) | 0.0 % | 1.835 (0.83) | 0.783 (0.91) |
| loadgen | t250@250 | 0.3 | 0.457 (1.74) | 1.221 (1.95) | 0.4 % | 0.604 (0.27) | 0.449 (0.52) |
| loadgen | t250@250 | 0.5 | 0.379 (1.44) | 0.984 (1.57) | 0.0 % | 0.832 (0.38) | 0.527 (0.61) |
| loadgen | t250@250 | 0.8 | 0.319 (1.21) | 0.802 (1.28) | 0.0 % | 1.342 (0.61) | 0.669 (0.78) |
| loadgen | t250@250 | 1 | 0.278 (1.06) | 0.675 (1.08) | 0.0 % | 1.856 (0.84) | 0.787 (0.92) |
| loadgen | zero@1000 | 0.3 | 1.293 (4.73) | 4.624 (7.04) | 22.2 % | 0.282 (0.13) | 0.307 (0.36) |
| loadgen | zero@1000 | 0.5 | 0.633 (2.32) | 2.056 (3.13) | 4.4 % | 0.684 (0.31) | 0.478 (0.56) |
| loadgen | zero@1000 | 0.8 | 0.352 (1.29) | 0.895 (1.36) | 0.1 % | 1.498 (0.68) | 0.708 (0.82) |
| loadgen | zero@250 | 0.3 | 1.279 (4.86) | 4.564 (7.28) | 22.1 % | 0.282 (0.13) | 0.306 (0.36) |
| loadgen | zero@250 | 0.5 | 0.615 (2.34) | 2.019 (3.22) | 4.3 % | 0.684 (0.31) | 0.478 (0.56) |
| loadgen | zero@250 | 0.8 | 0.341 (1.30) | 0.864 (1.38) | 0.1 % | 1.499 (0.68) | 0.708 (0.82) |

**What the teachers do.**

- **No lost grasps at any hold level.** The zero base loses 54 % of the swings at 0.3 of its grip and 9–10 % at 0.5
  (22 % and 4 % of the load generator's episodes); both teachers lose none of the swings and ≤ 0.4 % of the load
  generator's episodes, hit the nail every time and advance it as far as the full hold does (2.62–2.68 against
  2.67 mm per strike): no loss of progress.
- **The full hold's slip from a weak base.** Against the zero base at the same hold level the teachers cut the
  hammer's net slip per strike by 77–81 % at 0.3 and 64–75 % at 0.5 (rotation 0.3: 12.8 → 2.7°), the load generator's
  per event by 64–78 % and 38–61 %; they end at 1.0–1.25× the full hold's translation and 1.0–1.1× its rotation on the
  hammer, 1.2–1.5× (1 kHz) and 1.4–2.0× (250 Hz) on the load generator at 0.3–0.5.
- **At a fraction of the current** (against the zero residual at the full hold): the hammer at 0.5 of the hold, the
  full hold's rotation slip and +6–21 % translation at 60–71 % of its I²t (77–84 % of its RMS current), at 0.3 at
  45–55 % (67–74 %); the load generator at 0.5, +22–57 % slip at 38–49 % of the I²t, at 0.8, +8–28 % at 61–69 %.
- **Over the full hold the slip does not fall** (the authority limit of M3): the hammer's net slip is +10–12 % in
  translation and −1 to −4 % in rotation, the load generator's +3–8 %. What the teachers add there is damping: the
  hammer's slip path −29 to −35 % (6.6 → 4.6 mm, 16 → 10.5°) at +3–12 % I²t, and on the load generator 16–17 % less I²t
  for the same slip.
- **Peak current is the SDK's limit for every policy** at a hold of 0.5 and above (and for the teachers at every
  hold): one joint reaches the rated current at some point in nearly every episode, so the peak does not separate
  the policies; the time at the limit does: 2–3 % for the zero base, up to 9–12 % for the 1 kHz teacher over the full
  hold (it pushes into the limit where it has no authority), 4.6–6.5 % at 250 Hz.
- **The authority bound** (`grip`: every channel at its limit toward closing) is the same at every hold level: the
  drives sit at the current limit (34–45 % of joint-time) whatever the base asks. It keeps every grasp and lands at
  the full hold's slip on the load generator (1.00 / 0.99×) and 6–8 % below it on the hammer, at 1.1× (load generator)
  and 1.9× (hammer) the full hold's I²t. Beating the hold's slip needs current above rated (M3). Within rated, the
  teachers stop short of the bound where the I²t penalty says so: at 0.3–0.5 of the hold they remove 69–98 % of the
  slip the bound removes against the zero base, at 45–71 % (hammer) and 27–49 % (load generator) of the full hold's
  I²t where the bound spends 110–190 %.

**1 kHz against 250 Hz on the same simulated time (E3, first look).** On the hammer the 250 Hz teacher is as good
(translation 1.22–1.35 against 1.35–1.50 mm, rotation equal, slip path equal) on 7–18 % less I²t. On the load
generator the 1 kHz teacher holds better at low hold levels (77.7 against 74.0 % of events within the target at 0.3,
82.1 against 77.8 % at 0.5; net rotation 1.00 against 1.22° and 0.80 against 0.98°) and spends 31–39 % more I²t; over
the full hold they are equal. The rates also differ in samples (the 1 kHz run had four times as many) and exploration
noise (σ 0.48 against 0.21 at the end); one seed each, and the zero residual's own run-to-run spread is ±1–3 % (the
GPU solver is not deterministic): the load generator's difference is above that, the hammer's are not decisive.

### The teacher actor in ONNX (`python -m tactile_rl.export.onnx_export`)

`teacher_1k.onnx` (4.0 MB) and `teacher_250.onnx` (9.2 MB) with their sidecars, on the desktop in
`~/scratch/tactile-gpu-runs/m4/export/`. The graph is the CPU runner's convention: `obs` [B, 894 | 2580] and `h_in`
[B, 256] → `action` [B, 60] (tanh of the policy's mean, as the env squashes it) and `h_out` [B, 256]; the robust
normalizer folded in (mean, scale floored at 0.1, ±5 clip), the GRU as ONNX's GRU op, the (256, 128) ELU head; opset
17, dynamic batch. The sidecar is `PolicyMeta`'s (obs-spec hash `c6e8591bad20469f` at stack 1, `578fc8d167368135` at
stack 4; rate, stack, hidden size) plus the group and its widths (the student's 740 or 2426 and the 154 privileged),
the normalizer's floor and clip, the codec (U8 μ-law, 30 N full scale, `taxel_codec_v1`), the opset, the commit, the
checkpoint and the parity result. `tactile_sim.control.policy_io.OnnxPolicy` loads both (one thread, a Python loop:
~0.1 ms per step; the plan's E7 benchmark is for the students).

Parity against the torch actor on 10 240 recorded inputs (64 envs × 160 steps of the teacher playing C3, episodes
resetting along the way):

| teacher | action, each runtime its own state | state, each its own | action, per step (graph fed torch's state) | state, per step |
|---|---|---|---|---|
| 1 kHz | 2.2e-6 | 1.25e-5 | 7.5e-7 | 8.2e-6 |
| 250 Hz | 5.6e-7 | 6.0e-6 | 3.9e-7 | 3.1e-6 |

Per step the graph agrees to 1e-5 in both outputs (the plan's criterion). Running free, float32 differences between
the two GRU kernels accumulate in the state (1 kHz: 1.25e-5 after 160 steps, 3e-5 after ~1 s on tiled inputs) while the
action stays within 4.5e-6: deployment runs the ONNX graph alone, so only the per-step error matters there. A
teacher has no auxiliary heads, so the plan's third output (`aux`) comes with the students.

## Milestone 3 results (desktop, RTX 5070 Ti, 2026-10-08)

### P3 on the hard nail (`python -m tactile_rl.parity.p3`)

The library re-recorded on feat/residual-l2 51e7ca1 (the nail's friction constraint hard, `multiccd` off,
`residual_dr`'s nail at 50–150 N and 2.5–7.5 kN/m): 776 swings from 100 episodes (~8 per episode until the nail is in).
Every swing plays once per condition from the CPU's state at its windup, unperturbed, with its episode's
randomization, zero residual. Mean per strike [90 % bootstrap interval]; against the validator: the relative
difference of the means and the two-sample KS p; "ok" when within 10 % (slip 15 %), the intervals overlap and p > 0.01.

| per strike | CPU 8 kHz, noslip 5 (validator) | CPU 4 kHz | GPU 4 kHz | GPU 2 kHz |
|---|---|---|---|---|
| hits | 776 / 776 | 776 / 776 | 776 / 776 | 776 / 776 |
| nail advance (mm) | 2.54 [2.48, 2.60] | 2.63, ok (3.7 %, p 0.18) | 2.69, **6.1 %**, intervals apart (p 0.054) | 3.22, **27 %** (p 2e-25) |
| peak force, 2nd sample (N) | 185 [182, 187] | 185, ok (0.2 %) | 185, ok (0.4 %) | 185, ok (0.3 %) |
| pulse width (ms) | 6.52 [6.12, 6.94] | 6.24, ok (4.3 %, p 0.012) | 6.36, 2.4 % (p 1e-4) | 4.77, 27 % |
| slip, translation (mm) | 1.08 [1.05, 1.10] | 1.07, ok (0.4 %) | 1.11, ok (2.7 %) | 1.06, ok (1.4 %) |
| slip, rotation (deg) | 1.59 [1.53, 1.65] | 1.55, ok (2.5 %) | 1.75, **10.0 %** (p 3e-3) | 1.74, 9.2 % (p 3e-3) |

- **The M2 gap is closed.** With the hard nail the training physics (0.25 ms, no noslip) passes every test against the
  validator: noslip no longer changes how far a blow drives (2.63 against 2.54 mm; it was 5.0 against 1.1 mm).
  Replaying the CPU training physics under MuJoCo 3.11 (the GPU's version) gives the same table to 0.4 %.
- **MuJoCo Warp at 0.25 ms against the CPU at the same physics** passes on advance (2.3 %), peak force (0.2 %), pulse
  width (6 %) and slip translation (3.3 %) and fails only on **slip rotation: +13 %** (1.75 against 1.55°, paired
  +0.2° per strike, KS p 1e-3): the GPU grasp gives a little more in rotation. Against the validator, the GPU's
  advance (+6 %: 3.7 % from the step and noslip, 2.3 % from the GPU) and rotation (+10 %) are inside the plan's
  tolerances, but with 776 strikes the intervals no longer overlap; on 200-strike subsets (the plan's minimum) the
  GPU passes advance, peak and translation in 50 of 50 and rotation in 43 of 50.
- **0.5 ms physics is not usable for the hammer**: the nail advances 27 % further and the pulses are 27 % shorter.

### Throughput with rewards, randomization and the curriculum (load generator, C2, zero residual)

`python -m tactile_rl.bench.throughput --task loadgen --stage 2 --rate 1000 --envs 1024 4096 --seconds 0.2`:

| worlds | policy | ms per env step | simulated s / wall s | physics share of the step | env GPU memory (GB) |
|---|---|---|---|---|---|
| 1024 | 1 kHz | 28.3 | 36 | 34 % | 0.73 |
| 4096 | 1 kHz | 53.5 | **76** | 42 % | 2.67 |

Peaks: 24 broadphase candidates and 7.3 contacts per world, 61 constraint rows. In training (4096 envs, 128-step
rollouts, the GRU teacher's inference and update) an iteration takes 9–11 s: 7.7–9.7 s collecting, 1.2 s learning,
48–58 sim-s per wall-s.

### The first load-generator teacher (1 kHz)

`python -m tactile_rl.train Tactile-LoadGen-VegaWuji-1000Hz --env.scene.num-envs 4096 --agent.max-iterations 600
--agent.algorithm.entropy-decay-iterations 600 --agent.algorithm.desired-kl 0.02`: the plan's teacher (GRU 256 with a
(256, 128) head and a zero output layer, MLP critic (512, 256, 128) on the same privileged group, γ = e^(−1 ms / 0.3 s)
= 0.9967, λ = e^(−1 ms / 0.15 s) = 0.9934, 128-step rollouts, 5 epochs × 8 mini-batches, entropy 1e-3 → 0), from C0.
It took four runs to get a teacher that learns; each fixed one thing:

| run | what changed | what happened (C1, the loads at a quarter amplitude) |
|---|---|---|
| v1 | the plan's settings | the learning rate fell to its 1e-5 floor when the loads began and stayed: rsl_rl's normalizer scaled the features that never moved in C0 (the wrench, untouched taxels) ~100-fold |
| v2 | `RobustNormalization` (scale floor 0.1, ±5 clip) | the rate stayed at 1e-5–1e-4 under the loads with the plan's desired KL of 0.005; events within the target flat at 57 % over 125 iterations |
| v3 | desired KL 0.02 | the rate settled near 3e-4; still flat at 57 % over 240 iterations |
| v4 | the base grips at 0.5–1.0 of the hold (the plan's base hold level), events scored on net slip | learns (below) |

**Why v3 could not learn: at the settled hold the residual has no authority over slip.** Constant actions on the
load generator (C1 amplitudes, 256 envs, 3 s, no randomization; "within target" here is v3's criterion, the slip
path within 0.7 mm / 1.4°):

| base grip | residual | events within target (path) | net slip per event (mm / °) | slip path per event (mm / °) | lost grasps |
|---|---|---|---|---|---|
| the hold | zero | 55 % | 0.69 / 1.28 | 1.51 / 2.89 | 0 % |
| the hold | grip (τ_ff +0.6 rated toward closing) and stiffen (Kp ×4) | 57 % | 0.68 / 1.25 | 1.44 / 2.76 | 0 % |
| half the hold | zero | 36 % | 1.15 / 2.98 | 2.57 / 5.85 | 6 % |
| half the hold | grip and stiffen | 57 % | 0.67 / 1.24 | 1.44 / 2.76 | 0 % |
| the hold, full band (C2 amplitudes) | zero | 8 % | 2.90 / 7.48 | 4.87 / 13.2 | 57 % |
| the hold, full band | grip and stiffen | 10 % | 2.66 / 7.85 | 4.36 / 12.2 | 54 % |

At the hold the closing joints already draw near the drives' current ceiling (the SDK effort limit is the rated
current: mean |i| 0.81 rated, 0.84 with the residual pushing every closing joint to the limit), so gripping harder
buys < 5 % (the feedforward alone 2 %, the stiffness alone 4 %, a position offset toward closing nothing); below the
hold the residual restores the hold's slip. The slip path is about twice the net slip (the
grasp's elastic excursion). The full band (700 N impulses, 3 Nm binds) loses the grasp in more than half the episodes
whatever the residual does.

**v4 learns** (training metrics with the exploration noise on, means over 50-iteration windows; 0.52 M env steps per
iteration; the first loaded episodes end at iteration ~36):

| iterations | env steps (M) | episode return | events within target | net slip per event (mm / °) | slip path per event (mm) | lost grasps | I²t per episode | action σ |
|---|---|---|---|---|---|---|---|---|
| 50–99 | 26–52 | −30.2 | 62.2 % | 0.745 / 1.68 | 1.69 | 0.61 % | 1.42 | 0.33 |
| 150–199 | 79–104 | −27.7 | 62.3 % | 0.747 / 1.60 | 1.58 | 0.24 % | 1.44 | 0.37 |
| 300–349 | 157–183 | −25.2 | 63.8 % | 0.756 / 1.49 | 1.45 | 0.07 % | 1.45 | 0.40 |
| 550–599 | 288–314 | −24.0 | 65.1 % | 0.750 / 1.43 | 1.35 | 0.00 % | 1.47 | 0.41 |

The teacher stops losing the grasp, cuts the rotation slip and the slip path by 15–20 % and draws a little more
current (it grips harder when the base grips weakly); the net translation per event does not move yet, and the
stage's gate (80 % of events within 0.7 mm / 1.4°) is far: no C2 in this run. It is learning, not converged.

**The trained teacher against the zero residual** on the same 1024 episodes (the checkpoint at iteration 599, its
mean action, no exploration noise; `python -m tactile_rl.bench.teacher_eval <checkpoint> --modes
zero,teacher,zero@hold,teacher@hold`):

| | events within 0.7 mm / 1.4° | net slip per event (mm / °) | slip path per episode (mm / °) | lost grasps | I²t per episode |
|---|---|---|---|---|---|
| C1 (its stage: base at 0.5–1.0 of the hold), zero residual | 61.2 % | 0.714 / 1.78 | 6.10 / 13.9 | 1.8 % | 1.37 |
| C1, the teacher | **68.5 %** | **0.651 / 1.29** | **3.61 / 7.32** | **0 %** | 1.43 |
| C1 with the base at the full hold, zero residual | 70.2 % | 0.567 / 1.12 | 4.25 / 8.45 | 0 % | 2.25 |
| C1 with the base at the full hold, the teacher | 72.1 % | 0.572 / 1.09 | 3.35 / 6.46 | 0 % | 2.09 |
| C2 (never trained there), zero residual | 14.9 % | 3.00 / 8.80 | 27.2 / 90.1 | 80.8 % | 0.88 |
| C2, the teacher | 17.7 % | 2.66 / 7.92 | 22.5 / 78.5 | 73.0 % | 1.00 |

Over a weak base grip the teacher keeps every grasp and buys back most of what the grip lacks (net rotation per
event −27 %, slip path −40–47 %) for 4 % more I²t: it ends near the full hold's slip at 63 % of the full hold's
current. Over the full hold it changes the net slip by nothing (0.572 against 0.567 mm, 1.09 against 1.12°), damps the
slip path by ~22 % and saves 7 % I²t: the authority limit again. The full band (C2) is mostly lost grasps whatever the
policy, as the authority probe predicted.

Curves (reward, net slip, events within the target, lost grasps, I²t against env steps):
`~/scratch/tactile-gpu-runs/m3/train/curves_v4.png` on the desktop (`curves_v4.csv` beside it); the run's tensorboard
logs and checkpoints are in `~/scratch/tactile-gpu-runs/m3/train/tactile_loadgen_teacher_1000hz/`.

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
  `PhysicsCfg.multiccd = False` turns it off on both sides (World and `mujoco_cfg`); since 51e7ca1 `residual_config`
  sets it, so the validator and the training scene make single contacts.

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

- **Closed in M3: the nail without noslip and the contacts per pair.** The nail's friction constraint is hard
  (`NailPlantCfg.friction_solimp`, 51e7ca1), so noslip no longer changes it, and `residual_config` turns multi-contact
  off, so the validator makes one contact per convex pair as MuJoCo Warp does (P3 above; M2's analysis: the validator's
  grasp slips and drops the same with or without the extra contacts).
- **Rotation slip on the GPU** (P3): 13 % more than the CPU at the same physics; with the M1 hold creep (below) the
  GPU grasp yields a little more than the validator's. The plan's contact recalibration (impulse slip, ripple, hold
  creep) is the remedy; until then a teacher trained on the GPU meets a slightly firmer grasp in the validator.
- **The hold creeps on the GPU**: under the load generator's hold (no loads) the tool's slip path is ~0.16 mm and 0.3°
  per second, low-frequency (a 30 Hz low-pass barely changes it): it counts against the slip rewards and fills ~20 % of
  an event's 0.7 mm / 1.4° budget.
- **The base's wrap rests some joints on their limits** (the joint-limit load penalty is −0.025 per second for the
  zero residual), hence its light weight (−0.1).
- **The residual's authority** (M3, confirmed in M4): at the settled hold the WUJI's closing joints run near the
  current ceiling (effort limit = rated current), so no residual cuts the hold's slip by more than a few percent (the
  strongest constant one: −6–8 % on the hammer, 0 on the load generator, at 1.1–1.9× the I²t). Both bases therefore
  grip at a hold level below the hold (C1 0.5–1.0, C2 and C3 0.3–0.8; M4 gave the hammer's base the same), and the
  residual's measurable value is the hold's stability from a weak or variable base at a fraction of the current.
- **The load generator's band** is this hand's since M4 (`LoadGenCfg`, shared with the CPU; M3's 700 N impulses and
  3 Nm binds lost more than half the grasps whatever the residual did).
- **C2's 80 % gate was not reached in 120 000 simulated s** at either rate (72 % at the cap, the drop gate met); the
  cap moved both runs to C3, where the load generator's episodes continue and the teachers end at 74–84 % within the
  target on C2's settings. A longer C2 or a 75 % gate would let the gate decide.
- **The plan's desired KL (0.005)** holds the adaptive learning rate at 1e-5–1e-4 once the loads begin (60-D actions
  at σ 0.3); the teachers run at 0.02 (`teacher_runner_cfg(desired_kl=...)`; Tactile-Residual registers 0.02).
- **Peak current does not separate policies**: the SDK caps the current at rated and every policy touches the cap at
  some joint in nearly every episode; time at the limit and I²t are the measures.
- **The GPU is not deterministic** (MuJoCo Warp's solver): the same evaluation repeats to ±1–3 % (the zero residual's
  lost grasps 8.9 and 9.6 % at the hammer's 0.5 hold in two runs); one training seed per rate so far (the plan: ≥ 3).
- **The library's forecast stream is NaN where no event is planned** (after each contact): the GPU truth zeroes it
  (M4); any new consumer of `fc_t_event` must mask it.
- **The face/nail contact at impact.** Single contact on both sides, MuJoCo Warp still misses the cylinder/cylinder
  contact in a sixth of the strike states where the CPU finds one. Blows agree to a few percent in advance and peak
  force all the same (above).
- **The library must match the base and the scene.** The library records the base the CPU ran (`residual_config`'s
  grip loop and profile; `BaseLibrary` warns on a mismatch) and its start states come from the scene's plant. The
  current library (`~/scratch/tactile-gpu-runs/m3/lib/hammer_v2.npz` on the desktop; `*.npz` files are not
  committed) has the hard nail and single contacts; M2's (`m2/lib/hammer_v1.npz`) is stale.
- **Peak force is spiky.** A blow's largest single-step force can be a one-sample solver spike (1 in 203 blows at the
  training physics, 9 in 203 at the validator's, up to ~10x the pulse). `peak2` (the second-largest sample) is the
  robust metric. With the old closed grip loop the spikes depended on the drives' command jitter; CPU replays draw
  the jitter from the entry's index, so they do not depend on what a World played before.
- **Hooks are launch-bound** (above): ~50 % of the hammer's step at 4096 worlds, 58 % of the load generator's with the
  rewards and randomization. Above the plan's 15 %, but the step meets the throughput goal (76 sim-s per wall-s); fusing
  them is the next speed step, and the scan's float64 ADC chain should stay unfused (or be checked) to keep the
  frame-for-frame parity.
- **The GPU hold creeps ~10x more than the CPU** (M1): 0.14 mm in 1 s at 0.25 ms against 0.011 mm (CPU, no noslip). The
  contact recalibration is the remedy.
- **MuJoCo version drift** (M1): the validator (3.14) and the training venv (3.11) compile mesh frames differently;
  record the library in the training venv. The cross-venv test is still missing.
- **Mirrors.** The drill-trigger interlock now applies on task "drill" only, on both sides; the GPU base reads the
  read-back angles as the CPU's does; the stale-input decay is mirrored.

## What remains

- **The drill** on the GPU (plant, pistol driver, trigger, the drill base; its rewards): the interlock is ready for
  task "drill"; `TorchSynergy` needs WrapSynergy's `levels()` / `s_trigger` split first. The plant to port is the
  12 V model (feat/residual-l2 2249c57, 59a5b1a): `plants/drivetrain.py` (a small ODE, written to vectorize) plus
  the tip contact and screw/hole laws of `plants/drill.py`; parity against the CPU plant, and the dyno's ratings
  (`model/plants/drill_dyno.py`), on the batched version. After M4.
- **Curriculum C4**: latency extremes and VLA chunks (C3, the hammer mixed 70/30 with the load generator, is in M4).
- **Teachers**: still improving at the end of M4's budget; ≥ 3 seeds per rate for E3, and 1 kHz against 250 Hz at
  equal samples as well as equal simulated time; **contact recalibration** (static-hold creep, impulse slip, ripple,
  the GPU's 13 % extra rotation slip); **DAgger** students with the auxiliary labels (`aux_labels`), their ONNX export
  with the `aux` output, and the E7 latency benchmark.
- **Speed:** CUDA graphs or Warp kernels for the hooks; dropping mjlab's per-step `forward()`.
