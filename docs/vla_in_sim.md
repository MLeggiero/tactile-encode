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

## Commands (desktop)

```bash
# demos (CPU; ~3 min per episode per worker at the 8 kHz validator physics)
python -m tactile_sim.vla.demos --task hammer --episodes 500 --workers 8 --out runs/vla_demos/hammer
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
  - the four fingers curl over it with their fingertip joints held at 0.7 rad (`GripperCfg.finger_dip`, new), so
    the middle joints close on to ~80° and the fingertips' pads, not their ends, lie on the handle;
  - the thumb lies on top of the handle over the fingers: its base joints are held (thumb_close (0, 0, 1, 1),
    preshape flex 0.5, abduction -0.2) and its outer joints close.
- **What the taxels feel** at the hold, testbed seed 200002:
  - the palm on 15 of its 128 taxels (47 N), the thumb's pad on 4 (7 N), the fingertip pads on 14-19 each;
  - 82 taxels loaded in all (the light hammer's wrap: 20);
  - ~110 N on the handle; 0.7-0.9 Nm about it, 1.4-2.0 sideways, 2.1-2.7 head down or up, before it turns 5°.
- **Under blows** (scripted swings, 8 seeds, 6 blows): no drops, contact at 1.03-1.21 m/s.
  - The hammer settles 6-8° into the grip over the first two blows and then stays put.
  - The other wraps tried kept creeping about 2.5° per blow: 13-18° after six.
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
  - `residual_config(hammer="light" | "ycb")`: the default is "light", the GPU env's and the residual experiments'
    hammer.
  - `vla.demos` and `vla_experiment` default to `--hammer ycb`; models trained on datasets v1-v3 need
    `--hammer light`.
  - The LeRobot sidecar records each task's tool and wrap (`vega_wuji.json` "tools").
  - The nail: a blow puts ~0.32 J into it, so 20 mm takes ~11 blows at the preset's 100 N + 7 kN/m. The demos
    randomize it over 60-140 N and 4-10 kN/m (6-15 blows).
- **Viewer**: the wraps in 3D, with per-taxel loads, contact forces, twist and blow results:
  https://claude.ai/artifact/AdVgftZiaKXreB6dmNVYZo
