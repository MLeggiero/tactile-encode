# Which open VLA has had the most real-world success? (survey, 2026-10-07)

Base-model choice for the Vega U + WUJI Hand 2 tool-use project (the slow layer, L3). A research agent compiled
this from the sources listed at the end; figures are as those sources report them and were not re-measured
here. Statements marked **(judgment)** are inferences, not facts from a source.

## Bottom line

- **The most successful open VLA on real robots is π0.5** (Physical Intelligence; openpi and LeRobot). In every
  multi-model real-robot comparison not run by Physical Intelligence, it is first or tied first among
  fine-tunable VLAs:
  - RoboArena: highest standard VLA, Elo 1608 over 745 A/B evaluations [1].
  - RoboChallenge Table30: best third-party-trained model, 42.7 % [2].
  - RoboDojo real-world track: first on all three robots [6].
  - GM-100: best baseline over 22,500 trials [10].
  - VLA-REPLICA: first [8]. FluxVLA: about twice GR00T N1.5's success [9]. PhAIL: tied with GR00T N1.6 [7].
  - Models that beat it did so only as developer-submitted entries, in their authors' own runs, or as 14B-class
    world-action models that cannot run on a 16 GB GPU.
- **GR00T trails π0.5 on gripper and bimanual robots, but wins both dexterous-hand head-to-heads found:**
  - Dexora (36-DoF hands): GR00T N1 beats π0 [15].
  - T-Rex, on this robot family (Dexmate Vega-1 with 22-DoF Sharpa hands): EgoScale, built on GR00T N1.7, scores
    35 % against π0.5's 17 % [14].
- **"Qwen VLAs have been really successful" is only partly true.** Most 2026 open VLAs are Qwen-based (GR00T N1.7,
  Spirit-v1.5, Xiaomi-Robotics-0, DM0, LingBot, WALL-OSS, starVLA, T-Rex, MolmoAct2), and Qwen-based models
  submitted by their developers top RoboChallenge [2]. When a third party fine-tuned them on new robots in
  RoboDojo, several collapsed: Spirit-v1.5 0.6 %, DM0 0.0 %, StarVLA-α 1.7 %, Xiaomi-R0 3.9 %, against π0.5 at
  12.8 % [6]. **(judgment)** The backbone family does not predict real-world success.

**Decision for this project:** a matched bake-off in simulation between
1. **π0.5**, which has the best real-world record. WUJI (the hand maker) runs it on two 7-DoF arms with two
   20-DoF WUJI hands and reports 100 % on five real dexterous tasks after DAgger and residual RL [16, 17].
2. **GR00T N1.7**, which has a Qwen3-VL backbone, the strongest dexterous-hand evidence, and native support for up
   to 132 action dimensions. It is also what Dexmate's own stack uses [20], and the cheapest of the two to
   fine-tune; it is already set up on the desktop.

Both are trained on the same dataset (`tactile_sim.vla`; see `docs/vla_in_sim.md`). Pick GR00T if it beats π0.5 by
about 10 points or more on hammer and drill in closed loop.

## Real-robot evaluations

| Evaluation (who ran it) | Setup | Results for open models |
|---|---|---|
| **RoboArena**: distributed double-blind A/B on DROID Franka, leaderboard as of 2026-10-07 [1] | ≥100 A/B evaluations per policy for an official rank | dreaming_zebra 1735±43 (n=190, anonymized; probably DreamZero, **judgment**) > **π0.5-DROID 1608±31 (n=745)** > π0-FAST 1582 > PaliGemma-recipe policies 1516–1546 > π0 1461. No GR00T or SmolVLA entries |
| **RoboChallenge Table30 v1**: Dexmal + Hugging Face; UR5, Franka, ALOHA, ARX-5; 30 tasks × 10 trials; ≤1k demos per task [2, 4, 5] | Inference on the submitter's hardware | Developer-submitted: Era0 (closed) 64.3 %, DM0 62.0, GigaBrain-0.1 51.7, Spirit-v1.5 51.0. Third-party-trained: **π0.5 42.7**, WALL-OSS 35.3, π0 28.3, X-VLA 21.3, RDT-1B 15.0, **GR00T 14.3**, CogACT 11.7, OpenVLA-OFT 5.0 |
| **RoboDojo-RealEval**: HKU/UNC/THU/Berkeley et al.; ARX X5, Piper, Piper X; 18 tasks × 10; ~100 demos per task [6] | Success | **π0.5 12.8** > InternVLA-A1 7.2 > G0 4.4 > Xiaomi-R0 3.9 > X-VLA 3.3 > **GR00T-N1.7 1.7** = π0 = StarVLA-α > Spirit-v1.5 0.6 > DM0 0.0 |
| **GM-100**: LingBot authors; 4 platforms; 22,500 trials; 130 demos per task, one pipeline for all [10] | Average success | LingBot 16.9 > **π0.5 13.0** > **GR00T N1.6 7.3** > WALL-OSS 3.9 |
| GM-100 generalist, 9 tasks; LingBot-2.0 authors [11] | AgileX / R1 Pro | LingBot-2.0 34.4 / 15.6 > **π0.5 32.2 / 8.9** > **GR00T N1.7 17.8 / 5.6** |
| **PhAIL**: Positronic Robotics; Franka FR3; 449 demos; each repository's default recipe [7] | Throughput, teleop = 100 | **π0.5 13.8 ≈ GR00T N1.6 13.3** (p = 0.54) > ACT 10.5 > SmolVLA 6.4 |
| **VLA-REPLICA**: UT Dallas; SO-101; 500 demos; reproduced at a second site [8] | In / out of distribution | **π0.5 0.54 / 0.35** > π0 0.34 / 0.30 > SmolVLA 0.26 / 0.30 > X-VLA 0.14 / 0.075 |
| **FluxVLA**: LimX and universities [9] | ALOHA, 5 tasks | **π0.5 70/110** vs GR00T N1.5 36/110 |
| MolmoAct2 (Ai2, a competitor) [13] | Bimanual YAM, 8 tasks × 50 | MolmoAct2 50.1 %, ~15 points above OpenVLA-OFT; on DROID 87.1 % vs π0.5 45.2 % |
| **Dexora**: 36 DoF (2×6 arm + 2×12 XHAND); 100 demos per task [15] | Basic / dexterous tasks | GR00T N1 82.1 / 51.7 vs π0 50.4 / 26.7 |
| **T-Rex**: **Dexmate Vega-1 + 2 × 22-DoF Sharpa hands**, 62-D actions; 12 contact-rich tasks × 16 [14] | Average success | T-Rex 65 > EgoScale (GR00T-N1.7) 35 > **π0.5 17** > Tactile-VLA 15 > π0.5 with tactile input 6 |
| EgoSteer: PKU; dexterous hands [18] | — | Being-H0.5 39 % vs π0.5 22 % |
| **WUJI post-training**: 2 × 7-DoF arms + **2 × 20-DoF WUJI hands**, 30 Hz, 200 demos per task [16] | π0.5 only | SFT, DAgger, then residual RL in the model's latent action space: 100 % on 5 tasks |

## Candidates

| Model | Weights / license | Backbone · size | Action head | Fit to this project's 27-D | Fine-tune memory | Real-world summary |
|---|---|---|---|---|---|---|
| **π0.5** | Open; code Apache-2.0, weights under Gemma terms [21] | PaliGemma (Gemma-2B + SigLIP) + 300M expert, ~3.3B, not Qwen | Flow matching, 10 steps | Trained at 32-D: 27-D fits without new projections | LoRA > 22.5 GB, full > 70 GB [21] | Best overall; weak in T-Rex's dexterous tasks |
| **GR00T N1.7** | Open; code Apache-2.0, weights NVIDIA Open Model License [26, 27] | Cosmos-Reason2-2B (**Qwen3-VL**), 3B | Flow-matching DiT, 4 steps | 132-D native, per-embodiment projectors | Default (projector + DiT) ~35 GB [26] | Weaker on grippers, strongest on dexterous hands |
| SmolVLA | Apache-2.0 | 450M | Flow matching | Padded | Fits consumer GPUs | Last in PhAIL [7] |
| LingBot-VLA 2.0 | Code Apache-2.0 | Qwen3-VL-4B, 6B | MoE action expert | Only 12 hand joints | 8-GPU scripts | Best GM-100, but self-reported [10, 11] |
| T-Rex | MIT | Qwen3-VL-2B + tactile expert | Flow matching | 62-D on Vega + Sharpa | 24 × H100 for SFT | 65 % (authors' evaluation) [14] |
| Avoid | OpenVLA-OFT (7B, 5 % on RoboChallenge); InternVLA-A1 and Galaxea G0/G0.5 (non-commercial); Spirit, DM0, Xiaomi-R0, StarVLA-α (collapsed under third-party fine-tuning [6]); world-action models (seconds of latency) | | | | | |

## Evidence for contact-rich work

- No public evaluation of any VLA on hammering or power-drill tasks exists.
- Flow-matching VLAs undertrain the low-noise regime that governs contact precision; a logit-normal timestep
  schedule in post-training raised π0.5 from 39 % to 56 % on five contact tasks [55]. GR00T N1.7 also samples
  Beta(1.5, 1.0) timesteps [26], so the same fix applies to both **(judgment)**.
- π0/π0.5 hit protective stops above 75 N in wiping tasks [56].
- **Tactile input inside the VLA hurt**: on Vega, feeding tactile into π0.5 dropped success from 17 % to 6 % [14].
  This supports keeping touch in the separate residual (the project's decision).
- Residuals on top of VLAs: WUJI's latent-space residual (TD3) on a frozen π0.5 went 70 → 100 % [16]; Microsoft's
  object-centric residual trained in MuJoCo lifted GR00T-N1.5 from 42 to 76 % [58]; Opt2VLA ran GR00T N1.7 at
  5 Hz above a 200 Hz RL controller [57].
- Sim-only training: OpenVLA-OFT trained only in sim transferred at 0–26 % per task; camera pose and table height
  randomization mattered most [59]. Sim + real data beat real-only in RoboCasa365 (79.8 vs 61.8 %) [60].

## Practical notes

- **π0.5 / openpi**: base checkpoint `gs://openpi-assets/checkpoints/pi05_base`; one ego camera as `base_0_rgb`
  with the wrist slots masked; compute norm stats per config; full fine-tune on one GPU with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`; JAX multi-GPU collectives hang on RTX PRO 6000 Blackwell, so use one GPU
  [65]; serve with `scripts/serve_policy.py` (websocket, port 8000).
- **GR00T N1.7**: `nvidia/GR00T-N1.7-3B` (the gated `nvidia/Cosmos-Reason2-2B` backbone is accessible from the
  desktop's Hugging Face login); NEW_EMBODIMENT with a Python modality config; arms RELATIVE, hands ABSOLUTE;
  regenerate stats with `gr00t/data/stats.py` when the horizon changes; lower `state_dropout_prob` for tasks that
  depend on proprioception; serve with `gr00t/eval/run_gr00t_server.py` (ZeroMQ, port 5555).
- On the desktop, GR00T's video decoder (torchcodec 0.4) needs FFmpeg 4–7 shared libraries, which the `gr00t`
  conda env does not have (it has FFmpeg 2.8), and its deepspeed needs an nvcc the desktop lacks; separate envs
  provide both (`docs/vla_in_sim.md`).

## Sources

1. RoboArena leaderboard https://robo-arena.github.io/
2. RoboChallenge Table30 v1 https://robochallenge.ai/api/v1/leaderboard/leaderboard_all.json
3. RoboChallenge Table30 v2 https://robochallenge.ai/api/v2/leaderboard/leaderboard_table30_v2.json
4. RoboChallenge paper https://arxiv.org/abs/2510.17950
5. RoboChallenge 2025 annual report https://robochallenge.ai/
6. RoboDojo https://arxiv.org/abs/2607.04434
7. PhAIL https://arxiv.org/abs/2605.29710
8. VLA-REPLICA https://arxiv.org/abs/2605.20774
9. FluxVLA Engine https://arxiv.org/abs/2609.17210
10. LingBot-VLA (GM-100) https://arxiv.org/abs/2601.18692
11. LingBot-VLA 2.0 https://arxiv.org/abs/2607.06403
12. NVIDIA world-action models blog https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/
13. MolmoAct2 https://arxiv.org/abs/2605.02881
14. T-Rex https://arxiv.org/abs/2606.17055 ; https://tactile-rex.github.io/
15. Dexora https://arxiv.org/abs/2605.18722
16. WUJI VLA post-training https://arxiv.org/abs/2609.19666
17. wuji-openpi https://github.com/wuji-technology/wuji-openpi
18. EgoSteer https://arxiv.org/abs/2607.09701
19. GR00T N1.7 blog https://huggingface.co/blog/nvidia/gr00t-n1-7
20. NVIDIA blog (Dexmate quote) https://blogs.nvidia.com/blog/2-million-robotics-developers/
21. openpi https://github.com/Physical-Intelligence/openpi
26. Isaac-GR00T https://github.com/NVIDIA/Isaac-GR00T
27. GR00T-N1.7-3B model card https://huggingface.co/nvidia/GR00T-N1.7-3B
55. FACT https://arxiv.org/abs/2608.01402
56. CR-VLA-Force https://arxiv.org/abs/2609.05832
57. Opt2VLA https://arxiv.org/abs/2609.23968
58. Object-centric residual RL https://arxiv.org/abs/2606.18953
59. Sim-to-real study https://arxiv.org/abs/2603.22876
60. RoboCasa365 https://arxiv.org/abs/2603.04356
65. JAX issue #41183 https://github.com/jax-ml/jax/issues/41183
