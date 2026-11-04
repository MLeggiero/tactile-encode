# The VLA in simulation

The slow layer (L3) of the stack, trained and run entirely in the CPU testbed: a VLA fine-tuned on rendered
demos of the Vega U + WUJI Hand 2, then run in closed loop under the residual (L2). Which VLA: a bake-off between
π0.5 and GR00T N1.7 (`docs/vla_survey.md`).

## Pieces (`tactile_sim/vla/`)

| Module | What it does |
|---|---|
| `camera.py` | The head ego camera: the left imager of the Vega head's ZED pair (`ego_cam` on body `zed_left_camera`), 384 × 240 at 16:10, 78° vertical (a ZED X Mini with the 2.2 mm lens, 105° × 78° rectified; the lens is an assumption). `aim_head` points the head at the task (the tool's working point raised 10 cm), `vla_config` holds that pose. `EgoCamera` renders offscreen (EGL) and draws visual randomization per episode: camera mount ±1 cm / ±1°, lights ±30 % and ±20° tilt, tool / workpiece / table / floor colors ±0.15 |
| `action.py` | State and action, 27-D each: right arm 7, WUJI 20. Arm actions are joint targets before the host's droop compensation; hand actions are MIT position targets q* under a fixed per-task table (Kp, Kd, τ_ff: the grip synergy at the hold level, the vendor Kp on the closing joints). A scripted demo's hand action is the table-equivalent target of its command |
| `demos.py` | Scripted demos: the residual experiment's testbed (MIT drives, TaxelScan fast profile, zero residual) with its physics randomization, the task shifted ±3 cm on the table, and visual randomization. Hammer: the scripted swing (strike speed ×0.85–1.15) until the nail is home; drill: the scripted feed driving a screw, the trigger pulled by the index finger (its pull shows in the index finger's targets). 30 Hz frames to h264, vectors to `.npz`; only completed tasks count as successes |
| `lerobot_export.py` | Demos → a GR00T-flavored LeRobot v2.1 dataset laid out like `~/groot_n1_7_redcup` (G1_INSPIRE): parquet + mp4 + `info.json`, `episodes.jsonl`, `tasks.jsonl`, `stats.json`, `episodes_stats.jsonl`, `modality.json`, and `vega_wuji.json` (what the vectors mean, the hand table, the camera) |
| `client.py` | Closed loop: `VlaBase` requests a chunk every `exec_steps` frames, applies it `latency` later in simulated time, streams the arm through `PositionL1.set_joint_stream`, and is the residual's base (the chunk's hand targets under the table, lookahead from the chunk, no forecast). `GrootPolicy` speaks GR00T's server protocol; `ReplayPolicy` replays a demo. `run_vla_episode` runs one episode and returns strikes, depth, drops and the chunk log |
| `finetune/gr00t_vega_wuji_config.py` | GR00T NEW_EMBODIMENT modality config (arm RELATIVE, hand ABSOLUTE, horizon 16 = 0.53 s) |
| `finetune/gr00t_finetune.py` | GR00T fine-tune with a `16gb` profile (diffusion transformer and VL self-attention frozen in bf16; the new embodiment's projectors trained) and the `full` recipe (96 GB) |
| `finetune/openpi_vega_wuji.py` | π0.5 transforms and training configs (full, LoRA, and a tiny-model debug config), `widen_norm_stats`, to copy into an openpi checkout |

The experiment's scripted base holds the tool with a constant, open-loop grip (`residual_config`): no grip law
on the pads and no slip feedback from the simulator's truth, which a VLA without touch could not reproduce either.
Everything that reacts to touch is the residual's.

## Planned motion variation (2026-10-31)

The scripted demos all start lined up over the task and swing the same way. So a model can score on them without
looking at where the nail is. The planned generator, `vla.demos`' default, varies the motion itself (plan:
`docs/plans/2026-10-29-001-feat-vla-demo-motion-variation-plan.md`); `--generator scripted` records the one scripted
motion, as datasets v1-v5 were.

**What is varied** (`vla/variation.py`, `HammerSpace` and `DrillSpace`, Sobol point *i* of the dataset's `--seed0` for
episode *i*; widened 2026-11-03, plan: `docs/plans/2026-11-03-001-feat-far-starts-joint-planner-strike-variation-plan.md`):
- The start: a point 0–40 cm from the hover or pre-contact pose, at any azimuth, from 15° below its height to 70°
  above, tilted up to ±20° per axis. About a fifth of the episodes instead start from one of the arm's anchor postures
  (tucked, lowered, raised), jittered.
  - The arm reaches it in an unrecorded pre-roll (`behaviors/motion_plan.py`), and recording begins with the planned
    approach.
- The approach: its speed (45–100 % of the fastest the joint limits allow) and, for the simple curve, its sideways bulge.
  - It is executed as a joint stream along the joint path the pre-check validated, so the motion run is the motion
    checked.
- The task's position on the table: ±6 cm, or ±5 cm for the drill (the scripted demos use ±3 cm).
- The hammer's windup curve (`CompositeStrikeGeometry`):
  - an apex up to 5–6 cm off the nominal one, back, along the handle or across;
  - where the curve joins the terminal stroke (0–6 cm before the hover pose);
  - the windup time (0.35–0.6 s) and the settle time (0.1–0.3 s).
  - Then every strike varies around those: apex ±2 cm, windup time ±15 %, strike speed ±5 %. A strike whose variation
    fails the pre-check after three draws keeps the episode's values.

  The blow still meets the nail square, along the strike axis.
- The drill's entry: the transit ends turned up to 15°, 3–6 cm back along the bit and up to 2 cm off its axis. The feed
  then lines the bit up with the screw (its `align` phase) before its own approach.
- Not varied, on purpose: the contact pose, idle pauses (a hold looks the same in every frame, so a policy cannot tell
  when it ends), path wobble (it would be in the labels). Recovery comes from DART pushes, on by default for planned
  demos (`--perturb`, default 0.3; 0 for scripted).

**The pre-check** (`behaviors/feasibility.ArmProbe`) runs before anything is simulated, on a scratch copy of the
episode's world:
- **What it checks:** the start's IK, the approach, the pre-roll, the hammer's whole stroke up to 3 cm before contact
  and the drill's line-up, for reach, joint margin, IK continuity, and 1 cm of clearance to the table, board, torso and
  other arm.
- **How:** the robot's geoms have collisions disabled in the scene, so physics would not catch an arm through the
  table. The probe runs MuJoCo's collision pipeline on a copy in which only arm/tool-vs-obstacle pairs collide. For a
  mesh against a primitive (an arm link or the hammer head against the table or board), that pass reports only
  overlap, whatever the margin (MuJoCo 3.11), so those gaps come from `mj_geomDistance`. Mesh-against-mesh pairs keep
  the collision pass: `mj_geomDistance` reports zero for some of them tens of cm apart on this scene. Pairs nearer than
  1 cm at the start by design (the drill's bit 1 cm short of the screw) must only not overlap.
- **Planning:** the approach is tried first as the simple curve (`TransitPath`), solved into joint waypoints that end
  at the hover posture's own IK solution. When that fails, a straight joint-space move is tried, then RRT-Connect in
  joint space (`behaviors/joint_planner.py`) around the obstacles. The path is shortcut, splined and re-checked, then
  timed so no joint exceeds 70 % of its velocity limit. The pre-roll is straight in joint space when that is clear,
  planned otherwise. Plans record which produced the approach (`planner`).
- **Retries:** a rejected candidate is redrawn, up to 20 times. An episode with no feasible draw is written as a
  metadata-only record (`status: "rejected"`), which the export skips.
- **The strike's speed caps** are evaluated along the whole planned stroke, not only at the hover pose.
- **Room for pushes:** the plan records which tenths of the approach keep 5 cm between the arm and tool and every
  obstacle. A DART push in the approach starts at a random time, and only where it has that room.

The scripted generator is unchanged, bit for bit: `tests/test_motion_plan.py` checks the default bases against golden
traces (`tests/data/golden_bases.json`).

```bash
python -m tactile_sim.vla.demos --task hammer --episodes 500 --workers 8 --out runs/vla_demos/hammer_planned
python -m tactile_sim.vla.demos --task drill --episodes 500 --workers 8 --out runs/vla_demos/drill_planned
python -m tactile_sim.vla.coverage runs/vla_demos/hammer_planned runs/vla_demos/drill_planned   # outcomes, rejections, success by region
python -m tactile_sim.vla.lerobot_export runs/vla_demos/hammer_planned runs/vla_demos/drill_planned --out runs/lerobot/vega_wuji_tools_v6
# the closed loop from misaligned starts (sampled on EVAL_SEED0, away from the demos'), reported under <out>/varied
python -m tactile_sim.experiments.vla_experiment run --task hammer --server groot://localhost:5555 --starts varied --out runs/e2/groot_hammer
python -m tactile_sim.experiments.vla_experiment report --task hammer --starts varied --out runs/e2/groot_hammer
```

Exported datasets tag each episode's generator in `episodes.jsonl`, so scripted and planned demos can be exported
together or kept apart.

**Timing.** The pre-check also times the pre-roll, the approach and the windup so that no joint's target moves faster
than 70 % of its velocity limit. A windup too short for its path is stretched to fit (up to 2.5×), and the plan records
the durations it got.

**Quality gates** (`vla/quality.py`, both generators). A demo that completes its task is exported only if it also
passes:
- **The arm's limits:** joint velocity, range and torque within their ratings over the episode. The WUJI's hard-stop
  loads are recorded but not gated: the wrap's knuckles rest on their stops in every normal blow.
- **The tool in the hand:** per hammer strike, no more than 7 mm or 5° (scripted and planned demos stay under 3°).
  Over a drill episode, no more than 5 mm or 10°, and no cam-outs. The drill's clutch rolls it 4.7–6.5° in the fingers
  in good demos.
- **The blows:** no more than a quarter land off the nail head or miss.
- **Clearance in free motion:** over the approach and windups (the drill's transit and line-up), sampled at 10 Hz,
  the arm and tool keep 5 mm from the table, board and robot body. Nothing else checks what was executed, since the
  scene simulates no robot collisions.
- **The labels:** the arm's recorded targets never move faster than a joint's limit between frames, and show no
  sustained frame-to-frame oscillation. This is three or more consecutive sign flips of the second difference, the
  signature of the closed-loop failure above. Real demos score 0.

Each demo's metrics, pass or fail, and reasons are in its metadata (`quality`). `lerobot_export --all` includes failing
demos, flagged `quality_passed: false` in `episodes.jsonl`.

**Coverage** (`vla/coverage.py`). The report also gives each sampled parameter's coverage against the intended
uniform range, in 5 bins, with a bin under half its share flagged. It does this twice: after the pre-check (the plans
simulated) and among the exported demos, so a gap can be traced to infeasibility or to the simulation and the gates.
It also gives success by start distance (to 40 cm and beyond) and by what produced the approach (curve, joint move or
planner) and the start family. A rejected candidate redraws only the parameter group that failed: the start (for the
start, the pre-roll or an approach no path was found for), or the swing (hammer) or entry (drill). The episode's other
coordinates keep their Sobol values.

## Commands (desktop)

```bash
# demos (CPU; ~3 min per episode per worker at the 8 kHz validator physics)
python -m tactile_sim.vla.demos --task hammer --generator scripted --episodes 500 --workers 8 --out runs/vla_demos/hammer
python -m tactile_sim.vla.lerobot_export runs/vla_demos/hammer --out runs/lerobot/vega_wuji_hammer

# GR00T N1.7 (Isaac-GR00T's conda env; torchcodec needs FFmpeg 4-7 and deepspeed an nvcc, both kept in
# separate conda envs so the gr00t env is untouched)
cd ~/Isaac-GR00T
export LD_LIBRARY_PATH=~/scratch/envs/ffmpeg7/lib CUDA_HOME=~/scratch/envs/cuda128
C=~/tactile-encode/tactile_sim/vla/finetune
~/miniconda3/envs/gr00t/bin/python gr00t/data/stats.py --dataset-path <ds> --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path $C/gr00t_vega_wuji_config.py
~/miniconda3/envs/gr00t/bin/python $C/gr00t_finetune.py --dataset-path <ds> --output-dir <out> --profile 16gb
~/miniconda3/envs/gr00t/bin/python gr00t/eval/run_gr00t_server.py --model-path <out>/checkpoint-N \
    --embodiment-tag NEW_EMBODIMENT --modality-config-path $C/gr00t_vega_wuji_config.py --port 5555
```

On the 96 GB GPU: the same with `--profile full`, and π0.5 per the header of `finetune/openpi_vega_wuji.py`.

## Status (2026-10-08)

- **Hammer dataset v1**: 500 scripted episodes, 195 drove the nail home (the rest met hard nails or light heads
  under the physics randomization and ran out of strikes; 13 dropped the hammer) → `runs/lerobot/vega_wuji_hammer`
  on the desktop, 195 episodes, 90,183 frames (50 min at 30 Hz), 113 MB. GR00T's loader and stats tool read it;
  openpi's LeRobot reads it and computes norm stats.
- **GR00T N1.7, `16gb` profile** (diffusion transformer frozen): 327 M trainable parameters, 12.7–13.2 GB. Steady
  state on g1's RTX 5080: 1.84 optimizer steps/s at micro-batch 4 × 8 accumulation (~59 samples/s), so 5k steps at
  batch 32 take ~45 min (a 20-step probe is dominated by start-up and suggests far less). It does not fit next to
  RL training or the demo renderers on one 16 GB card.
- **First closed loop (E2-lite, 2026-10-08)**: the `16gb` model (combined dataset v1, 5k steps, loss 0.053) served on
  g1, the VLA alone, 20 seeds a task under the hard nail (51e7ca1).
  - Hammer: 0/20 done; it drops the hammer in every episode, mostly within 1-2 s and before any strike.
  - Drill: 0/20 seated; it holds the drill but the bit never meets the screw, and the drill turns 9-28° in the hand.
  - The cause: the model's chunks alternate frame to frame by about 0.03 rad (e.g. a wrist target 1.270, 1.298,
    1.265, 1.271, 1.285, 1.265 ...). The demos are smooth, at similar speeds. The arm follows the alternation as
    acceleration spikes of hundreds of rad/s², and the hammer turns 24° in the hand within 0.7 s.
  - Not the cause: the demos' exact actions replayed through the same client under the same physics hold the
    hammer and drive the nail home (3 demos, latency 0 and 0.1 s). The hand table is also not the cause: its Kp
    on the closing joints is 0.22-0.55 Nm/rad, so the model's 0.05-0.1 rad hand errors change the squeeze
    (MCP 1.4 Nm, PIP/DIP 0.21 Nm) by under 10 %.
  - Client-side remedies cut the drops but complete nothing: a 0.1 s chunk fade-in (`--blend`) gives 85 % drops;
    adding one pass of a [1, 2, 1] filter along each chunk (`--smooth`) gives 70 %. A better-trained model (the full
    recipe) is the fix to test.
- **30 Hz actions soften a blow**: a replayed swing meets the nail on time but at 0.7-0.9 m/s where the scripted
  one did at 1.16 (30 Hz targets cannot carry the strike profile's braking in its last ~30 ms), driving it 40-55 %
  as far. A VLA's strikes have the same limit.
- **Dataset v2** uses the hard nail, multiccd off, and the hammer's first windup at once (0.15 s instead of a 0.7 s
  idle hold, which looks the same in every frame).
  - Hammer: 488/500 succeed (v1: 39 %). Drill (the v2 plant): 303/500.
  - The drill's failures had three causes:
    - The trigger jams at hand friction of about 1.25 or more: the index fingertip locks on the grip and the motor never runs (119 episodes).
    - Clutch draws below the screw's run-down torque never seat.
    - The feed retracts while the motor still runs: every cam-out in the successful demos happened then.
  - 791 episodes, 214,409 frames.
- **Dataset v3 (`vega_wuji_tools_v3`)**: v2's hammer demos plus 500 drill demos on Drill v3 (2249c57).
  - Drill v3 is a 12 V low-gear drill/driver with a Torx drive, the trigger-finger friction cap (59a5b1a) and a release phase.
  - The new code reproduces a v2 hammer demo bit for bit, so the hammer demos carry over unchanged.
  - Drill: 500/500 seat, with 0 cam-outs, in 2.4-2.7 s.
  - 988 episodes, 231,782 frames. The drill is 23 % of the frames.
- **Hand tables per task** (384dfd1): each task's table is built at the grip its scripted base holds.
  - Hammer: the configured hold (1.4 Nm at the MCPs). Drill: the settled full grip (2.0 Nm).
  - Both tasks' demo hand targets are now the joint angles. Dataset v2's drill targets sat ~1.6 rad past them.
  - A model trained on v2 runs its hands under v2's tables with `vla_experiment --hand-tables <v2 dataset>`.
- **pi0.5 data on Skynet**:
  - openpi's norm-stats pass and its training loader both decode one video frame per sample. From the shared /coc storage that ran at 11-22 s a batch.
  - `openpi_norm_stats.py` computes the stats from the parquet files in seconds; it matches openpi within 0.4 % of range on v1.
  - The training job copies the dataset to the node's own disk and uses 12 loader workers (5a72408).
- **Compute (2026-10-08)**: g1 (RTX 5080, 20 cores) runs GR00T fine-tunes/serving and the CPU testbed (env at
  `g1:~/tactile-encode/.venv`, MuJoCo 3.14.0); Skynet (A40s, overcap) runs the full recipes via
  `vla/finetune/skynet_*.sbatch` (`~/scratch/DexVLA/tactile`).
- **π0.5 on the 5070 Ti**: openpi's JAX 0.5.3 trains on Blackwell (sm_120) with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`;
  `pi05_vega_wuji_debug` (tiny model, dataset v1, the real transforms and norm stats) runs and checkpoints. The real
  model needs the 96 GB GPU.
- **Desktop environment**: openpi in `~/scratch/openpi/.venv` (uv from `~/scratch/envs/uv`), with the Vega+WUJI
  policy module and config entry added and dataset v1 linked at `~/.cache/huggingface/lerobot/local/vega_wuji_hammer`.
  torchcodec in both the GR00T and openpi envs needs `LD_LIBRARY_PATH=~/scratch/envs/ffmpeg7/lib`.

## The real hammer and its wrap (2026-10-09)

GR00T's closed-loop hammer drops (dataset v3 model: 95-100 % of episodes) were traced partly to the grasp itself.

- **The light hammer's wrap** held the handle with the four fingertips against the index finger's knuckle link.
  Measured in the testbed 0.4 s into a VLA episode, at the hammer table's hold (s 0.7):
  - palm and thumb carried 0 N;
  - all four knuckles (MCP flexion) sat on their 90° stops, which took 1.29-1.57 of the 1.4 Nm each was commanded;
  - so the squeeze came from the 0.3 Nm middle and tip joints alone: 5-12 N per fingertip;
  - an external torque of 0.3 Nm about the handle started turning the hammer, 0.7-0.9 Nm turned it 5°.
- **The real hammer**: the YCB 048 claw hammer, the repo's scan of a real 16 oz hammer (665 g), held 22 cm from the
  head.
- **The wrap** (`config.REAL_HAMMER_OVERRIDES`, chosen with the user from the 3D viewer below) is a power grip:
  - the handle lies on the palm;
  - the four fingers curl over it with their fingertip joints held at 0.6 rad (`GripperCfg.finger_dip`, new), so
    the middle joints close on to ~80° and the fingertips' pads, not their ends, lie on the handle;
  - the thumb lies on top of the handle over the fingers: its base joints are held (thumb_close (0, 0, 1, 1),
    preshape flex 0.5, abduction -0.1) and its outer joints close;
  - the fingers collide with each other (`GripperCfg.finger_self_contact`, new, off by default). The hand had no
    finger-to-finger contacts: in the first approved version of this grip the index finger closed 11 mm into the
    middle one. With them on, the grip was searched again (1,080 candidates).
- **What the taxels feel** at the hold, testbed seed 200002:
  - the palm on 17 of its 128 taxels (53 N), the thumb's pad on 5 (5 N), the fingertip pads on 10-14 each;
  - 69 taxels loaded in all (the light hammer's wrap: 20); ~115 N on the handle.
- **Under blows** (scripted swings, 8 seeds, 6 blows): no drops, contact at 0.95-1.10 m/s, the nail 11-13 mm deep.
  - The hammer turns 7-10° into the grip over the six blows, the later ones adding about 1° each.
  - The wraps without the fingertip pads crept about 2.5° per blow: 13-18° after six.
- **Constraints of this hand.**
  - The knuckles rest on their 90° stops. WUJI publishes the MCP range as -60° to 90° (Hand 2.2 Beta manual,
    matching the model files), and with the knuckle axes 25 mm behind the palm no handle on the palm is wrapped
    short of them. The middle joints (PIP) reach 120°.
  - The palm is rigid and the handle round, so the palm feels a band (14-22 taxels in every candidate), not an area.
  - All four fingertip pads on the handle and the thumb pressing hard on top did not happen together in any
    candidate. With the index finger wrapped under it, the thumb's 0.3 Nm outer joints only rest on top. The other
    side of that trade ("thumb presses": the thumb's pad at 27 N, the index fingertip off the handle) kept creeping
    under blows.
- **Wraps tried on the way.** The viewer below shows these.
  - Across the finger bases: the knuckles squeezing the handle against the side of the thumb, 350 N. Strong, but
    the palm and the thumb's pad felt nothing, and the hammer crept ~4° per blow without springing back.
  - The handle on the palm with the fingertips' ends on it (fingertip joints closing to their stops): the thumb's
    pad and palm loaded, the fingertip pads not.
- **The search** (`python -m tactile_sim.model.hands.wrap_search`):
  - candidates are synthesized as the grasp cache does it, scored on the palm's load, the thumb's push direction and
    the height of its tip over the fingers;
  - the best are rerun in the testbed for the taxels' view (`hand.skin_taxels()`, loaded taxels per patch and
    their spread) and then under scripted blows.
- **Using it.**
  - `residual_config(hammer_kind="light" | "ycb")`: the default is "light", the GPU env's and the residual
    experiments' hammer.
  - `vla.demos` and `vla_experiment` default to `--hammer ycb`; models trained on datasets v1-v3 need
    `--hammer light`.
  - The LeRobot sidecar records each task's tool and wrap (`vega_wuji.json` "tools").
  - The nail: a blow puts ~0.32 J into it, so 20 mm takes ~11 blows at the preset's 100 N + 7 kN/m. The demos
    randomize it over 60-140 N and 4-10 kN/m (6-15 blows).
  - The swing's windup is 0.16 m for this hammer (the Vega's 0.20 m otherwise). This grip seats the face higher above
    the nail, and from 0.20 m it started 0.25-0.27 m away. There the strike profile (its peak at most 1.25x the
    contact speed) has no solution at any speed, so the scripted swing fell back to 0.2 m/s taps: 5 of 16 full
    drives fell short and 1 dropped the hammer. Within ~0.26 m it plans 0.86-0.94 m/s.
- **The arm's hold** (chosen with the user, 2026-10-09). The Vega U's default hover arched the arm over the table, the
  elbow above the shoulder.
  - The search varied the hammer's heading about the vertical strike axis (15° steps), the task's place (0.35-0.55 m
    ahead, 0-0.30 m right) and height, and the arm's redundancy (IK from 24 seeds, the best solution kept per elbow
    band: below, level with or above the shoulder).
  - Each pose was scored on its holding torque against the joints' ratings, the joints' margins at the hover and
    windup poses, and the strike speed the joint velocity limits allow along the arc.
  - Holding torque hardly separates the poses (the most loaded joint at 13-18 % of its rating), and the cheapest
    hold folds the elbow high. The blow is what loads this arm: 60-72 % of a joint's rating in every candidate tried.
  - The hold now: the task 0.55 m ahead and centred, the grasp 0.90 m up, the handle 75° right of ahead. The arm
    reaches forward with the elbow tucked in, 11 cm below the shoulder. Holding takes at most 17 % of a joint's
    rating and a blow 70 %, at 0.99 m/s, 9 blows on the test seed.
  - The alternatives shown were the arch (15 %, 67 %, 0.97 m/s), the head pointing forward (16 %, 72 %, 0.80 m/s,
    12 blows) and the forearm level across the body (16 %, 72 %, 0.91-0.94 m/s).
- **Viewer**: the wraps in 3D, with per-taxel loads, contact forces, twist and blow results:
  https://claude.ai/artifact/AdVgftZiaKXreB6dmNVYZo
