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
- **Compute (2026-10-08)**: g1 (RTX 5080, 20 cores) runs GR00T fine-tunes/serving and the CPU testbed (env at
  `g1:~/tactile-encode/.venv`, MuJoCo 3.14.0); Skynet (A40s, overcap) runs the full recipes via
  `vla/finetune/skynet_*.sbatch` (`~/scratch/DexVLA/tactile`).
- **π0.5 on the 5070 Ti**: openpi's JAX 0.5.3 trains on Blackwell (sm_120) with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`;
  `pi05_vega_wuji_debug` (tiny model, dataset v1, the real transforms and norm stats) runs and checkpoints. The real
  model needs the 96 GB GPU.
- **Desktop environment**: openpi in `~/scratch/openpi/.venv` (uv from `~/scratch/envs/uv`), with the Vega+WUJI
  policy module and config entry added and dataset v1 linked at `~/.cache/huggingface/lerobot/local/vega_wuji_hammer`.
  torchcodec in both the GR00T and openpi envs needs `LD_LIBRARY_PATH=~/scratch/envs/ffmpeg7/lib`.
