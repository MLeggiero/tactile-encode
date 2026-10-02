# Force/Tactile-Aware VLAs and Hierarchical Fast-Slow Robot Policies (2024-2026): Novelty Check for a Kilohertz Tactile Grip Loop for Impulsive Tool Use

Method note: all 23 arXiv IDs supplied were checked in one batch query against the official arXiv API (export.arxiv.org/api/query, run 2026-10-02). Titles/authors/abstract content below come from that query unless another source is cited. Rates and sensor details come from the papers' HTML versions, read with a summarizing fetch tool. Treat those as "read from paper, not hand-checked line by line".

## Q1. Which systems already combine a slow policy with a fast tactile/force layer? Do the cited arXiv IDs exist and match?

### Takeaway
All 23 supplied arXiv IDs exist, and every one matches the topic it was cited for. None is a fabricated ID. Several fast-slow tactile/force hierarchies already exist (RDP, FAVLA, T-Rex, AT-VLA, PhaForce, DuoCore-FS, Helix 02). In all of them the "fast" learned layer runs at about 7-300 Hz on downsampled tactile/force data and outputs pose deltas or joint targets. None outputs a per-phase stiffness plus grip-force schedule from a VLA, and none runs a learned layer on kHz tactile/vibration windows. Helix 02 comes closest on rate: S1 runs at 200 Hz with fingertip tactile input, and S0 at 1 kHz. Its 1 kHz layer is whole-body control from proprioception, though, not tactile. Two caveats on attribution: FACT is "2608.01402" (its title does not contain "FACT", which appears only in the abstract), and "FACTR 2" is really about NEXT/FIRST.

### Cited Findings

**ID verification table (all EXIST; content matches unless noted)**

| arXiv ID | Verified title (first authors) | Date | Match notes |
|---|---|---|---|
| 2503.02881 | Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation (Han Xue, Jieji Ren, Wendi Chen, Gu Zhang, Yuan Fang...) | 2025-03-04 | Matches RDP + TactAR |
| 2409.11047 | TacDiffusion: Force-domain Diffusion Policy for Precise Tactile Manipulation (Yansong Wu, Zongxie Chen, Fan Wu...) | 2024-09-17 | Matches; 95.7% zero-shot transfer, +9.15% from dynamic-system filter |
| 2410.09309 | Adaptive Compliance Policy: Learning Approximate Compliance for Diffusion Guided Control (Yifan Hou, Zeyi Liu, Cheng Chi, Eric Cousineau, Naveen Kuppuswamy...) | 2024-10-12 | Matches; ">50% improvement" |
| 2411.15753 | FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation (Zihao He, Hongjie Fang, Jingjing Chen, Hao-Shu Fang, Cewu Lu) | 2024-11-24 | Matches |
| 2410.07554 | ForceMimic: Force-Centric Imitation Learning with Force-Motion Capture System (Wenhai Liu, Junbo Wang, ... Cewu Lu) | 2024-10-10 | Matches; zucchini peeling, +54.5% relative success |
| 2608.01402 | Demystifying When and Why VLAs Fail in Contact-Rich Tasks and How to Fix Them (Carlota Parés-Morlans, Nils Kuhn, Isabel Liu, Alberta Longhini, Jeannette Bohg) | 2026-08-02 | "FACT" is the method name in the abstract, not the title. 66% vs 41% average success over 5 tasks, ~2,500 real rollouts |
| 2606.17055 | T-Rex: Tactile-Reactive Dexterous Manipulation (Dantong Niu, Zhuoyang Liu, Zekai Wang, Boning Shao, Zhao-Heng Yin...) | 2026-06-15 | Matches; 100 h dataset, variable-rate MoT, >30% over best baseline, 12 tasks |
| 2609.24621 | Learning tactile perception from high-bandwidth single-point sensing (Joseph Rigal, Emmanuel Virot, Caroline Pascal) | 2026-09-21 | "SpectRobot" is the framework name in the abstract, not the title |
| 2512.20188 | Asynchronous Fast-Slow Vision-Language-Action Policies for Whole-Body Robotic Manipulation (Teqiang Zou, Hongliang Zeng...) | 2025-12-23 | "DuoCore-FS" is in the abstract. Astribot; 3B VLM, 30 Hz chunks. It is not a tactile paper |
| 2509.07962 | TA-VLA: Elucidating the Design Space of Torque-aware Vision-Language-Action Models (Zongzheng Zhang, Haobo Xu...) | 2025-09-09 | Matches; decoder-side torque adapters, torque as auxiliary prediction |
| 2603.12665 | TacVLA: Contact-Aware Tactile Fusion for Robust Vision-Language-Action Manipulation (Kaidi Zhang, Heng Zhang, Zhengtong Xu...) | 2026-03-13 (v4) | Matches; contact-aware gating; +20% disassembly, +60% in-box picking, 2.1x under occlusion |
| AT-VLA (no ID supplied) | AT-VLA: Adaptive Tactile Injection for Enhanced Feedback Reaction in VLA Models (Xiaoqi Li, Muhe Cai, ... Hao Dong) = **arXiv 2605.07308** | 2026-05-08 | Found by search |
| 2602.23648 | FAVLA: A Force-Adaptive Fast-Slow VLA model for Contact-Rich Robotic Manipulation (Yao Li, Peiyuan Tang...) | 2026-02-27 | Matches |
| 2609.31225 | Imp-ACT: Adaptive Impedance Control and Action Chunking with Transformers... (Luca Zanetti, Doganay Sirintuna, Idil Ozdamar, Pietro Balatti, Heng Zhang...) | 2026-09-25 | Matches; wiping + plug insertion |
| 2606.12406 | FACTR 2: Learning External Force Sensing for Commodity Robot Arms Improves Policy Learning (Steven Oh, Jason Jingzhou Liu, Tony Tao...) | 2026-06-10 | Matches. Contributions are NEXT (sensorless external-torque estimation) and FIRST (re-sampling contact segments); >17% task progress over prior force-aware policies |
| 2601.09988 | In-the-Wild Compliant Manipulation with UMI-FT (Hojung Choi, Yifan Hou, Chuer Pan...) | 2026-01-15 | Matches |
| 2512.08920 | OSMO: Open-Source Tactile Glove for Human-to-Robot Skill Transfer (Jessica Yin, Haozhi Qi, Youngsun Wi, Sayantan Kundu, Mike Lambeta...) | 2025-12-09 | Matches; 12 three-axis sensors; wiping 72% |
| 2606.09243 | EgoTactile: Learning Grasp Pressure for Everyday Objects from Egocentric Video (Yuan Zeng...) | 2026-06-08 | Exists. A pressure-estimation benchmark (EgoPressureFormer/EgoPressureDiff), not a control policy. Cite only for grasp-pressure priors |
| 2606.11396 | PLUME: Probabilistic Latent Unified World Modeling and Parameter Estimation for Multi-Finger Manipulation (Abhinav Kumar, Soshi Iba, Rana Soltani Zarrin, Dmitry Berenson) | 2026-06-09 | Exists. A belief over physical parameters (friction, etc.), screwdriver turning. Relevant to L0 parameter estimation, not to tactile control |
| 2512.05964 | Training-Time Action Conditioning for Efficient Real-Time Chunking (Kevin Black, Allen Z. Ren, Michael Equi, Sergey Levine) | 2025-12-05 | Matches; π0.6, box building + espresso |
| 2410.21845 | Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning (Jianlan Luo, Charles Xu, Jeffrey Wu, Sergey Levine) | 2024-10-29 | Matches HIL-SERL; 1-2.5 h training, 2x success, 1.8x faster |
| 2602.18397 | How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf (Wenqi Jiang, Jason Clemons, Karu Sankaralingam, Christos Kozyrakis) | 2026-02-20 | Matches; 15 takeaways |
| 2604.02523 | Tune to Learn: How Controller Gains Shape Robot Policy Learning (Antonia Bronars, Younghyo Park, Pulkit Agrawal) | 2026-04-02 | Matches |
| 2602.14174 | Direction Matters: Learning Force Direction Enables Sim-to-Real Contact-Rich Manipulation (Yifei Yang, Anzhe Chen...) | 2026-02-15 | Matches; predicts force direction plus contact state and configures an admittance controller; tasks: microwave, peg-in-hole, whiteboard wiping, door |

Source for the table: [arXiv API batch query](https://export.arxiv.org/api/query?id_list=2503.02881,2409.11047,2410.09309,2411.15753,2410.07554,2608.01402,2606.17055,2609.24621,2512.20188,2509.07962,2603.12665,2602.23648,2609.31225,2606.12406,2601.09988,2512.08920,2606.09243,2606.11396,2512.05964,2410.21845,2602.18397,2604.02523,2602.14174); AT-VLA from [arXiv 2605.07308](https://arxiv.org/abs/2605.07308).

**Closest fast-slow systems: rates, sensors, action spaces, tasks**

- **Figure Helix 02:** S0 "executes at 1 kHz, handling balance, contact, and coordination". S0 inputs are "full-body joint state and base motion" and its outputs are joint-level actuator commands at 1 kHz. S1 runs at 200 Hz, taking head cameras, palm cameras, fingertip tactile sensors and proprioception and outputting full-body joint targets. S2 is a slow semantic reasoner emitting latents, with no rate given on the page. The fingertip tactile sensors detect "forces as small as three grams". The page does not mention stiffness or force outputs, impacts or tool use. — [Figure, Introducing Helix 02](https://www.figure.ai/news/helix-02)
- **Reactive Diffusion Policy (RDP):** a slow latent diffusion policy at 1-2 Hz and a fast asymmetric tokenizer at about 24 Hz. The fast rate is bounded by the GelSight Mini (25 FPS) and MCTac (30 FPS). Joint torques stream at 120 Hz but are downsampled to 24 FPS. Actions are relative end-effector trajectories, interpolated and sent to a Flexiv arm at >>500 Hz. Tasks are peeling, wiping and bimanual lifting. — [RDP HTML](https://arxiv.org/html/2503.02881v3)
- **FAVLA:** a slow VLM at a fixed low rate plus a fast action expert run 1x to N_max times per VLM cycle, scheduled by predicted force variation. F/T is collected at 200 Hz and downsampled to 30 Hz for training. Actions are 32-step chunks of delta end-effector pose plus gripper width, with no stiffness and no grip force. Tasks are USB insertion, gear assembly, box flipping and board wiping. Results: 80.8% average success (+13.8% over ForceVLA); peak forces 7.7 N (gear) and 9.9 N (box) vs 12.0 N and 12.2 N for vision-only. The abstract explicitly motivates the design with "delayed responses to impacts, stick-slip, and force spikes". — [FAVLA HTML](https://arxiv.org/html/2602.23648v1); [abstract](https://arxiv.org/abs/2602.23648)
- **T-Rex:** a variable-rate Mixture-of-Transformers with a temporal tactile VQ-VAE (16-frame windows). Per-finger tactile sensors give a 6-axis wrench plus deformation depth maps. The tactile expert is re-triggered at intra-chunk offsets {0,4,8,12} of a 30 Hz, 16-step chunk. A low-level control thread runs at 300 Hz. Actions are 62-D: end-effector deltas for two 7-DoF arms plus 22-DoF finger joint positions. Tasks: 12 tasks (egg transfer, paste application, page flipping, mahjong sorting...), with no impact or striking tasks. — [T-Rex HTML](https://arxiv.org/html/2606.17055v2)
- **AT-VLA:** the GO-1 VLA (InternVL-2B + DiT) with a tactile gate. When the gate is active, the tactile stream runs at 3:1 against the slow stream, at about 0.04 s per inference (~25 Hz). The sensor is a Xense gripper with 6-D force. Actions are 14-DoF end-effector poses. Tasks: unzip bag, stamp, wipe vase, unscrew lid. Success is 0.50 vs 0.22 for vanilla GO-1. Naive tactile fusion cost 9%. — [AT-VLA HTML](https://arxiv.org/html/2605.07308v1)
- **DuoCore-FS:** an asynchronous fast-slow VLA built on a 3B VLM with a latent buffer, producing 30 Hz whole-body action chunks. It uses no tactile or force input. — [arXiv 2512.20188](https://arxiv.org/abs/2512.20188)
- **PhaForce (additional find, arXiv 2603.08342):** "low-rate chunk-level planning and high-rate residual correction", with a contact-aware phase predictor (CAP) estimating contact probability and phase belief. 86% average success (+40 pp). Rates are not given in the abstract. — [arXiv 2603.08342](https://arxiv.org/abs/2603.08342)
- **TA-VLA / TacVLA / FACT:** single-rate VLAs that add torque, tactile or force tokens (decoder adapters, contact gating, failure-mode fixes). None is a fast inner loop. — [TA-VLA](https://arxiv.org/abs/2509.07962); [TacVLA](https://arxiv.org/abs/2603.12665); [FACT](https://arxiv.org/abs/2608.01402)
- **Training-time RTC:** about smooth asynchronous chunking for a VLA (π0.6). It has no tactile layer. — [arXiv 2512.05964](https://arxiv.org/abs/2512.05964)

### Inferences
- The broad claim "no system fuses a slow VLA with a fast tactile layer" is false. RDP, FAVLA, T-Rex, AT-VLA, PhaForce and Helix 02 all do this in some form, and the paper must cite them.
- The narrower claim can still be defended. Every learned fast layer found runs at ≤300 Hz (Helix S1 at 200 Hz, T-Rex at 300 Hz). Most downsample tactile or force data to 24-30 Hz. All of them output kinematic targets, not stiffness plus grip-force setpoints. None targets impulsive tool strikes. Defensible novelty: a learned layer at 100-200 Hz that consumes raw kHz tactile/vibration windows and outputs impedance parameters plus grip force, for millisecond impacts.
- Helix 02's S1 (200 Hz, tactile in) is the same rate as the proposed L2. The difference has to be argued on action space (stiffness and grip force vs joint targets), on input bandwidth (kHz windows), and on the impact task. Helix 02's internals are not public, so the paper should say "to our knowledge, based on public descriptions".

### Gaps
- The Helix 02 page gives no rate for S2. The original Helix (Feb 2025) S2 rate of about 7-9 Hz is from memory and was not re-verified here.
- The fetch tool did not confirm the sampling rate of T-Rex's raw tactile sensor.
- PhaForce rates and its action space (whether it outputs stiffness) were not verified beyond the abstract.

## Q2. Impulsive vs quasi-static contact; stiffness/grip-force outputs; tactile at ≥500 Hz

### Takeaway
Almost every listed system is evaluated on quasi-static contact: wiping, insertion, peeling, flipping, assembly, delicate grasping. None is evaluated on millisecond impacts such as hammering. Stiffness outputs exist (ACP, UMI-FT, Imp-ACT, Direction Matters via admittance), and so do grip-force outputs (UMI-FT, FARM). None of these comes from a VLA, and none comes from a policy that reads kHz tactile data. Raw sensing at ≥500 Hz appears only at the sensor or controller level (ACP's ATI at 7 kHz, SpectRobot at up to 200 kS/s); the learned policy runs at camera rate.

### Cited Findings
- **ACP:** outputs a 19-D vector per arm: a 9-D reference pose, a 9-D virtual target pose, and a scalar stiffness for the low-stiffness direction (k_low along force feedback, k_high elsewhere). There is no explicit grip force. The UR5e compliance controller runs at 500 Hz and the ATI F/T streams "at up to 7000Hz". Tasks: item flipping (96%) and vase wiping (93.75%). — [ACP HTML](https://arxiv.org/html/2410.09309v2)
- **UMI-FT:** six-axis F/T on each finger. The policy "predicts position targets, grasp force, and stiffness for execution on standard compliance controllers". Tasks: whiteboard wiping, skewering zucchini, lightbulb insertion. — [arXiv 2601.09988](https://arxiv.org/abs/2601.09988)
- **Imp-ACT:** ACT predicts end-effector pose, gripper action and motion-direction stiffness. Results: 29x / 180x less contact-force vibration in wiping, and 43% lower orthogonal forces in plug insertion. — [arXiv 2609.31225](https://arxiv.org/abs/2609.31225)
- **Direction Matters:** predicts pose, contact state and desired force direction to configure an admittance controller. Force magnitude is a hand-tuned scalar per contact state. — [arXiv 2602.14174](https://arxiv.org/abs/2602.14174)
- **ForceMimic (HybridIL):** predicts wrench-position parameters for a hybrid force-position primitive. Task: vegetable peeling. — [arXiv 2410.07554](https://arxiv.org/abs/2410.07554)
- **TacDiffusion:** generates a 6-D wrench for high-precision insertion. Uses a dynamic-system filter to bridge the frequency gap between the diffusion policy and the real-time control loop. — [arXiv 2409.11047](https://arxiv.org/abs/2409.11047)
- **FARM (arXiv 2510.13324):** GelSight Mini on a UMI gripper; the policy "jointly predict[s] robot pose, grip width, and grip force". — [search snippet: arXiv 2510.13324](https://arxiv.org/pdf/2510.13324) (not fetched in full)
- **FoAR:** fuses "high-frequency force/torque sensing" with vision via a future-contact predictor and uses simple position control. — [arXiv 2411.15753](https://arxiv.org/abs/2411.15753)
- **Impact-adjacent work found:**
  - FAVLA names impacts and force spikes as motivation, but its tasks are quasi-static and it trains on 30 Hz F/T. — [FAVLA](https://arxiv.org/abs/2602.23648)
  - HIL-SERL covers "dynamic manipulation" through real-world RL with vision and proprioception, not tactile. — [arXiv 2410.21845](https://arxiv.org/abs/2410.21845)
  - Tool-as-Interface learns tool-use policies from human video with vision only. A search snippet lists nail hammering among its tasks, but the abstract I fetched does not name it. — [arXiv 2504.04612](https://arxiv.org/abs/2504.04612)
  - An optimal-control paper treats hammering as an impact task whose performance index is hammer-head speed at contact. — [arXiv 2402.05502](https://arxiv.org/pdf/2402.05502)
- **Supporting context:** Tune to Learn argues that controller gains should be chosen for learnability, because effective stiffness emerges from the policy and controller together. — [arXiv 2604.02523](https://arxiv.org/abs/2604.02523). VLA-Perf analyzes the inference latency limits of VLAs, including dual-system pipelines. — [arXiv 2602.18397](https://arxiv.org/abs/2602.18397)

### Inferences
- The "learned per-phase stiffness + grip force" part is incremental over ACP, UMI-FT and Imp-ACT. The new parts are: a VLA choosing the schedule per phase, a reactive network using kHz tactile and vibration windows, and the impulsive tool-strike task.
- If Tool-as-Interface really includes nail hammering, it is the closest task-level prior. It is vision-only with no grip or stiffness control, so it actually strengthens the tactile motivation, but it must be cited. Verify its task list in the full paper.

### Gaps
- I did not verify whether ForceMimic, FoAR or TacDiffusion run any learned component at ≥500 Hz. The abstracts suggest they do not.
- I found no hammering, nailing or striking result in any VLA or tactile-policy paper beyond the unconfirmed Tool-as-Interface mention.

## Q3. Learned impact-phase or contact-phase estimation at kHz inside a policy

### Takeaway
Several learned contact-phase estimators feed policies: FoAR's future-contact predictor, PhaForce's CAP, TacVLA and AT-VLA contact gates, Direction Matters' contact state, VibeAct's onset and slip detection. Each runs at the policy rate (≈10-30 Hz), except VibeAct, whose rate was not confirmed. I found no learned impact-phase classifier running at kHz inside a manipulation policy. Classical kHz impact detection exists in legged locomotion.

### Cited Findings
- FoAR: "future contact predictor" adjusts the F/T weighting between non-contact and contact phases. — [arXiv 2411.15753](https://arxiv.org/abs/2411.15753)
- PhaForce: a "contact-aware phase predictor (CAP) that estimates contact probability and phase belief". — [arXiv 2603.08342](https://arxiv.org/abs/2603.08342)
- AT-VLA: the tactile gate "helps in discriminating different contact stages". — [AT-VLA HTML](https://arxiv.org/html/2605.07308v1)
- FACTR 2's FIRST segments demonstrations into free-space, pre-contact and contact phases (at training time only). — [arXiv 2606.12406](https://arxiv.org/abs/2606.12406)
- VibeAct (Mao, Yoo, Oh, Francis, Ichnowski; arXiv 2606.27344, June 2026): piezo microphones in a dexterous hand. A learned estimator maps waveforms to per-fingertip contact onset, slip presence and slip magnitude, feeding sim-trained RL policies. Sim success went from 21.6% to 50.5%. Tasks are not impact or tool use, and rates were not given in the abstract. — [arXiv 2606.27344](https://arxiv.org/abs/2606.27344)
- Classical example: probabilistic contact estimation and impact detection for quadrupeds, with sensor inputs at 1 kHz. — [Camurri et al. 2017 RAL](https://www.robots.ox.ac.uk/~mfallon/publications/2017RAL_camurri.pdf) (outside 2024-2026; background only)

### Inferences
- VibeAct is the closest prior to the proposed L0/L2 pattern of compact contact/slip features learned from high-bandwidth vibration. The paper should cite it and set itself apart on impacts, tool-target state (nail depth) and the impedance/grip action space.

### Gaps
- VibeAct's estimator rate and audio sample rate were not available in the abstract. Check the full paper before claiming kHz novelty against it.

## Q4. Vibration/audio-based policies for tool use

### Takeaway
High-bandwidth vibration and audio policies exist: ManiWAV, SonicSense, SpectRobot, VibeAct, Audio-VLA, PolyUMI. All of them use vibration as a perception input at camera rate or for object identification. None closes a fast grip or impedance loop on it during impacts.

### Cited Findings
- ManiWAV: an "ear-in-hand" piezo contact microphone on a UMI-style gripper; a diffusion policy from audio-visual human demos; noise augmentation bridges the domain gap. Motor noise limits it in low-interaction tasks. — [ManiWAV arXiv 2406.19464](https://arxiv.org/html/2406.19464v1); [project](https://mani-wav.github.io/)
- SonicSense: a four-finger hand with fingertip contact microphones. Used for perception (material, shape, container contents, re-ID over 83 objects) via tapping, grasping and shaking, not for control. — [arXiv 2406.17932](https://arxiv.org/abs/2406.17932)
- SpectRobot: seven sensor types (MEMS/IEPE accelerometers, load cell, strain gauge, PZT...), sampled at up to 200 kS/s (≈100 kHz bandwidth). Data become 224×224 spectrograms every 33 ms, fed to ACT, with motor commands every ~33 ms. Task: shake-and-sort occluded boxes (vision 23%, tactile 77-82%). A 2.9 s history scored 86% vs 31% for 0.36 s at 10 kHz. — [SpectRobot HTML](https://arxiv.org/html/2609.24621v2)
- Audio-VLA adds contact-audio perception to a VLA. — [arXiv 2511.09958](https://arxiv.org/pdf/2511.09958) (search result only, not fetched)
- PolyUMI: visual-tactile-audio data collection. — [arXiv 2609.29760](https://arxiv.org/html/2609.29760) (search result only, not fetched)
- OSMO: a tactile glove (12 three-axis sensors) for human-to-robot transfer; wiping at 72%. — [arXiv 2512.08920](https://arxiv.org/abs/2512.08920)

### Inferences
- SpectRobot is the closest representation prior to the planned "8 kHz accelerometer spectrogram frames". It already turns ≤200 kS/s vibration into spectrogram images for an ACT policy, but at 30 Hz, for classification-style tasks, and with no impedance or grip outputs. The planned paper should cite it and set itself apart on loop rate (100-200 Hz), action space and impulsive tool use.

### Gaps
- Audio-VLA and PolyUMI were not fetched, so their rates and tasks are unverified.
- I could not confirm whether any audio/vibration work does hammering or striking with closed-loop control.
