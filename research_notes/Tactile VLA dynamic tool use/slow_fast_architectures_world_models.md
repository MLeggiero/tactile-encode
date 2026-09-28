# Slow-Fast (System-2 / System-1) Robot Learning Architectures and Learned World Models for Contact-Rich Manipulation

Scope: dual-system VLAs (commercial + academic), real-time VLA inference, slow/fast interface design for contact, RL/residual layers on top of VLAs, learned world models (incl. tactile and hierarchical/multi-rate), learned-model MPC on hardware, and design-pattern synthesis. Emphasis on Hz, latency, interface type, sensing modalities, force/tactile in the fast loop, and impacts/tool use. Coverage: primarily 2023 to Sep 2026, with foundational papers where needed. "Not disclosed" is marked explicitly. Marketing claims vs documented results are distinguished where relevant.

---

## KQ1. Dual-system (System-2 / System-1) VLA architectures: what runs at what rate, what the interface is, and what the fast layer takes as input

### Takeaway
Every shipped dual-system VLA uses the same pattern: a VLM-scale "System 2" at 1–10 Hz emits a latent (or a language sub-goal) that a small "System 1" policy consumes at 10–200 Hz to produce joint/end-effector targets, which a classical or learned low-level controller tracks at 500 Hz–1 kHz. Only Figure's Helix 02 documents tactile sensing entering the 200 Hz fast loop (with a 1 kHz learned whole-body controller underneath); no commercial dual-system VLA discloses force/torque targets or impedance parameters as the interface, and none documents impacts, vibration, or powered-tool use.

### Cited Findings

**Commercial systems**

*Figure Helix (Feb 2025)*
- S2 is a "7B-parameter open-source, open-weight VLM" running at 7–9 Hz; S1 is an "80M parameter cross-attention encoder-decoder transformer" running at 200 Hz. — [Figure, Helix](https://www.figure.ai/news/helix)
- S1 inputs: monocular robot images through a fully convolutional multi-scale vision backbone, robot proprioception ("wrist pose and finger positions"), and the S2 latent vector "projected into S1's token space". — [Figure, Helix](https://www.figure.ai/news/helix)
- S1 outputs: a 35-DoF action space at 200 Hz comprising "desired wrist poses, finger flexion and abduction control, and torso and head orientation targets", plus a synthetic "percentage task completion" action used for behavior sequencing. — [Figure, Helix](https://www.figure.ai/news/helix)
- Training: end-to-end supervised with a "standard regression loss" on ~500 hours of teleoperated demonstrations, auto-labeled by a VLM; a "temporal offset between S1 and S2 inputs" is deliberately introduced during training to match the deployment latency gap between the two systems; S2 and S1 run asynchronously on two separate embedded GPUs. — [Figure, Helix](https://www.figure.ai/news/helix)
- The Helix (v1) post makes no mention of force or tactile sensing, nor of explicit lookahead mechanisms or limitations. — [Figure, Helix](https://www.figure.ai/news/helix)

*Figure Helix 02 (Feb 2026) — three-tier S2 / S1 / S0*
- S0 is a 10M-parameter learned whole-body controller taking "full-body joint state and base motion as input" and outputting "joint-level actuator commands at 1 kHz"; it was "trained entirely in simulation across more than 200,000 parallel environments with extensive domain randomization", on >1,000 hours of retargeted human motion data, and Figure says it replaced "109,504 lines of hand-engineered C++". — [Figure, Helix 02](https://www.figure.ai/news/helix-02)
- S1 is a transformer conditioned on S2 latents whose inputs are head cameras, palm cameras, fingertip tactile sensors and full-body proprioception; it produces "full-body joint targets that S0 tracks at kHz rates" at 200 Hz across legs, torso, head, arms, wrists and individual fingers. — [Figure, Helix 02](https://www.figure.ai/news/helix-02)
- S2 does scene understanding/language and "produces latent goals for S1"; it now sequences multi-step behaviors such as "Walk to the dishwasher and open it". Its rate is not restated in the Helix 02 post. — [Figure, Helix 02](https://www.figure.ai/news/helix-02)
- Tactile: "Tactile sensors embedded in each fingertip detect forces as small as three grams"; Figure describes tactile-regulated grip force for bottle-cap unscrewing and "force-controlled actuation with tactile feedback" for a syringe plunger (5 ml precision). The post frames "using the entire body as a tool" (closing drawers with the hip, lifting a dishwasher door with the foot), not hand-tool use. — [Figure, Helix 02](https://www.figure.ai/news/helix-02)
- Secondary reporting summarizes the same split: S2 handles language/scene, S1 produces full-body movements at 200 Hz from head cameras, palm cameras, fingertip sensors and proprioception, S0 executes at 1 kHz "handling balance, contact forces, and coordination"; the demo was a 61-step dishwasher unload/reload without resets. — [eWeek](https://www.eweek.com/news/figure-helix-02-humanoid-robot-autonomy/); [Thomasnet](https://www.thomasnet.com/insights/figure-ai-unveils-helix-02/)
- Helix 2.5 (Sep 17, 2026) discloses no new rates or architecture; it reports zero-shot whole-body autonomy across 30 homes with 56% success for the Index-pretrained model vs 9% from scratch, and half the task-specific data of Helix 02. — [Figure, Helix 2.5](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization)

*NVIDIA GR00T N1 / N1.5 / N1.6*
- GR00T N1 (Mar 2025): System 2 is the Eagle-2 VLM (1.34B) running at 10 Hz on an L40 GPU; System 1 is a diffusion transformer that "generates closed-loop motor actions at 120 Hz"; action chunk H = 16; sampling 16 actions takes "63.9 ms on an L40 GPU using bf16" with K = 4 forward-Euler denoising steps under a flow-matching loss. — [GR00T N1 paper](https://arxiv.org/html/2503.14734v1)
- N1 System 1 inputs: noised action embeddings, proprioception encoded through embodiment-specific MLPs, and vision-language embeddings, fused with alternating cross- and self-attention; action spaces are embodiment-specific (end-effector pose with axis-angle rotation, joint positions, gripper state) with min-max normalization. Sensing is RGB (224×224 → 64 tokens/frame) + proprioception + language; no force or tactile input. Stated limitation: focus on "short-horizon tabletop manipulation". — [GR00T N1 paper](https://arxiv.org/html/2503.14734v1)
- N1.5: the MLP connector between VL features and the DiT was modified, and it was "trained jointly with flow matching and world-modeling objectives"; the model card does not state rates or action horizons. — [GR00T-N1.5-3B model card](https://huggingface.co/nvidia/GR00T-N1.5-3B)
- N1.6: a ~2B VLM backbone (Cosmos-Reason-2B variant via the `EagleBackbone` class) plus a 32-layer DiT (up from 16 layers in N1.5); action horizon 16 steps; 4 denoising steps; inference throughput 27.3 Hz on an RTX 5090; ~3B parameters, ~12 GB bf16; up to 10 embodiments via category-specific projectors; inputs are images, text and proprioception, with no tactile/force. — [Isaac GR00T architecture docs](https://nvidia-isaac-gr00t.mintlify.app/architecture)
- The N1.6 model card describes a "flow matching transformer" action decoder with AdaLN conditioning and "joint training with flow matching and world-modeling objectives", but gives no Hz, chunk length, latency, or force/tactile information. — [GR00T-N1.6-3B model card](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- Aggregator claims that N1.6 System 1 runs at "30 Hz" or "50+ Hz" are inconsistent and not from NVIDIA primary sources. — [FinancialContent (aggregator)](https://markets.financialcontent.com/stocks/article/tokenring-2026-1-19-nvidia-unveils-isaac-gr00t-n16-the-foundation-for-a-global-humanoid-robot-fleet)

*Physical Intelligence π0 / π0.5 / π*0.6*
- π0 attaches a flow-matching "action expert" to a PaliGemma VLM and emits 50-step action chunks at up to 50 Hz, sent directly to the robot's low-level controller. — [π0 paper (PI)](https://www.pi.website/download/pi0.pdf); [alphaXiv summary](https://www.alphaxiv.org/abs/2410.24164)
- π*0.6 (Nov 2025): a separate 860M-parameter flow-matching action expert; actions are produced as chunks at 50 Hz consisting of "joint angles and gripper commands"; observations are "joint and gripper positions, as well as images from three cameras" (base + two wrist); no force or tactile sensing. — [π*0.6 paper](https://arxiv.org/html/2511.14759)
- Recap (RL with Experience & Corrections via Advantage-conditioned Policies): a multi-task distributional value function over B = 201 bins; a binarized advantage indicator (positive if advantage exceeds a task-dependent threshold set at the 30th percentile of predicted values); expert teleoperator interventions are forced to "I_t = True" (treated as high-advantage). Tasks: laundry folding (11 item types), double-shot espresso (~200 s), factory box assembly (~600 s). Stated limitations: relies on human labeling, interventions and resets; exploration is "relatively naïve"/greedy; uses "iterated 'offline' updates" rather than fully online RL. — [π*0.6 paper](https://arxiv.org/html/2511.14759)
- PI's blog claims Recap-trained π*0.6 roughly doubles throughput and cuts failures by 2×+, and ran espresso making for a full day, folded 50 novel laundry items in a new home, and assembled/labeled 59 factory boxes (company-reported). — [PI blog, π*0.6](https://www.pi.website/blog/pistar06)
- Hi Robot (Feb 2025): a hierarchical scheme in which a high-level VLM reasons over complex prompts and situated user feedback and emits the next step in language, which a low-level policy executes; evaluated on single-arm, dual-arm and mobile dual-arm platforms (table cleaning, sandwich making, grocery shopping). Rates for each level are not stated in the abstract. — [Hi Robot (arXiv)](https://arxiv.org/abs/2502.19417)

*Google DeepMind Gemini Robotics / 1.5 / On-Device*
- Gemini Robotics (Mar 2025) uses "a VLA backbone hosted in the cloud" and "a local action decoder running on the robot's onboard computer"; backbone query-to-response latency is under 160 ms, end-to-end latency ~250 ms from raw observations to action, with an effective control frequency of 50 Hz achieved via action chunks. Inputs are cameras, proprioception and language; no force/tactile is specified. The report notes the generalist struggles with long-horizon precision (origami, cards) without specialization. — [Gemini Robotics report](https://arxiv.org/html/2503.20020v1)
- Gemini Robotics 1.5 (Sep/Oct 2025) pairs a VLA with an orchestrating VLM (GR-ER 1.5) and adds "Embodied Thinking" (language thinking traces before acting) and a Motion Transfer recipe across ALOHA, bi-arm Franka and Apollo; the report does not state Hz, chunk length or end-to-end latency, and acknowledges that dexterity is comparable to the previous generation. — [Gemini Robotics 1.5 report](https://arxiv.org/html/2510.03342)
- Gemini Robotics On-Device runs locally with "low-latency inference" (no numbers given), supports ALOHA, Franka FR3 and Apptronik's Apollo, adapts with 50–100 demonstrations, and underperforms the cloud model on out-of-distribution and multi-step tasks. — [DeepMind blog](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)

*Generalist AI GEN-0 (Nov 2025)*
- "Harmonic Reasoning" is described as "a 'harmonic' interplay between asynchronous, continuous-time streams of sensing and acting tokens" that lets the model think and act at once without a separate System1/System2 split or inference-time guidance; models are 7B–10B+; 270,000+ hours of manipulation data growing 10,000 h/week; a power-law scaling fit L(D) = (D_c/D)^α is reported. No control frequency, latency or tactile/force information is given. — [Generalist, GEN-0](https://generalistai.com/blog/nov-04-2025-GEN-0)

*Skild AI (Skild Brain)*
- Skild describes a two-level structure: a low-frequency high-level policy producing manipulation/navigation commands that a high-frequency low-level policy converts into "precise joint angles and motor torques"; trained from large-scale simulation, internet video, and targeted real data. No Hz values and no tactile/force inputs are stated. — [Skild AI blog](https://www.skild.ai/blogs/building-the-general-purpose-robotic-brain)
- Third-party claims (500B-parameter MoE, tactile and force-torque inputs, "2.8 billion real robot trajectories") come from an aggregator and are not corroborated by Skild's primary page. — [dev.to (aggregator)](https://dev.to/xberry-tech/skild-brain-13500-humanoids-and-a-nasdaq-ticker-57g5)

*1X (Redwood)*
- Redwood is a "160M parameter transformer" fusing language embeddings, vision tokens and proprioception embeddings, decoded via a diffusion policy; it runs "around 5hz" fully on NEO's onboard GPU; it "predicts not only the arm and hand commands, but also walking, manipulation, and pelvis pose commands simultaneously"; proprioception includes "joint positions and joint applied forces", used for force-aware behaviors like bracing. The page does not describe the underlying whole-body controller or its rate. — [1X, Redwood](https://www.1x.tech/discover/redwood-ai)

*Dyna Robotics (DYNA-1, Dyna-2)*
- DYNA-1 (Apr 2025) is marketed as a commercial-ready foundation model emphasizing quality, speed and robustness, using a "special type of reinforcement learning" and a "scalable foundation reward model"; no architecture rates are disclosed. — [SiliconANGLE](https://siliconangle.com/2025/04/29/dyna-robotics-debuts-dyna-1-foundation-model-powering-robots/); [Dyna research page](https://www.dyna.co/research/dyna-1)
- Dyna-2 is "a single generative model that can denoise future video and future actions jointly or separately, built on a video-diffusion backbone" (mixture-of-transformers; action stream shallower and joined early "for latency efficiency"; flow matching), pretrained on >1M hours of egocentric human video; no control frequency, latency, or force/tactile are stated. — [Dyna-2](https://www.dyna.co/dyna-2)

*Tesla Optimus*
- Only aggregator/fan sources describe Optimus as an FSD-derived end-to-end network with "force-torque and tactile feedback layered in at the hands"; no primary technical disclosure of layer rates or interfaces was found. — [optimusk.blog (aggregator)](https://optimusk.blog/blog/tesla-optimus-agi-neural-nets/); [tooldirectory.ai (aggregator)](https://tooldirectory.ai/tools/tesla-optimus)

**Academic dual-system VLAs**

- HiRT (Tsinghua / Shanghai Qizhi, 2024): the VLM (InstructBLIP) runs asynchronously at ~1–4 Hz and writes latent embeddings to a buffer; a lightweight vision-based policy consumes the latest cached latent and acts at 9.8 Hz (≈2× a vanilla VLA); Metaworld 76.4% vs 73.8%; on dynamic real tasks success rose from 48% to 75%; authors note insufficient data for high-speed dynamic tasks. — [HiRT](https://arxiv.org/html/2410.05273)
- A 2025 survey characterizes the "Fast-Slow Dual-System paradigm" (HiRT, RoboDual, OpenHelix) as a large VLM slow system at 1–5 Hz and a lightweight fast policy at 20–50 Hz; RoboDual uses OpenVLA as the slow system and a DiT as the fast system, with slow-system latents refined through a Perceiver Resampler. — [Efficient VLA survey](https://arxiv.org/pdf/2510.17111)
- DuoCore-FS (Astribot, Dec 2025): slow system 1–3 Hz; fast system 25–30 Hz (measured 32.3 Hz); interface is a "bridge buffer" of latent semantic representations ("text embeddings, reasoning features, and learnable fusion queries"); the fast system takes three camera views, proprioception, the latest latents and raw instruction embeddings; trained end-to-end with "cross-timescale co-training" in which the fast system receives temporally shifted observations to simulate asynchronous deployment (λ_slow = 1, λ_fast = 10); Astribot S1 (25-DoF); 1,780 demos (10.22 h); 90% in-distribution vs 85% π0, 50% OOD vs 10%. No force/tactile ("future work"). — [DuoCore-FS](https://arxiv.org/html/2512.20188v1)
- UniFS (JD.com et al., 2026): instead of two networks, VLM layers are stratified into update frequencies (timescales 1, 2, 4, 8, 16 steps); the slowest pathway refreshes once per 16 steps, the fastest every step; ~2.1× speedup (17.8 ms vs 36.5 ms); 98.3% on LIBERO; no tactile/force. — [UniFS](https://arxiv.org/html/2606.22794)
- DexVLA (2025): VLM + a 1B-parameter diffusion action expert with a three-stage embodied curriculum; rates not stated in the abstract. — [DexVLA](https://arxiv.org/abs/2502.05855)
- NebulaVLA (2026) is a further "dual-frequency" VLA with a guide action (details not extracted). — [NebulaVLA](https://arxiv.org/pdf/2608.16503)

**Rate summary (documented values only)**

| System | Slow layer | Fast layer | Lowest layer | Interface slow→fast | Force/tactile in fast loop |
|---|---|---|---|---|---|
| Figure Helix (2025) | 7B VLM, 7–9 Hz | 80M policy, 200 Hz | not disclosed | latent vector into S1 tokens | none stated |
| Figure Helix 02 (2026) | VLM, rate not restated | 200 Hz joint targets | S0 10M net, 1 kHz | latent goals; S1→S0 joint targets | fingertip tactile into S1 (3 g sensitivity) |
| GR00T N1 | 1.34B VLM, 10 Hz | DiT, 120 Hz, H=16 | robot controller (not specified) | VL embeddings via cross-attn | none |
| GR00T N1.6 | 2B VLM | DiT 27.3 Hz on RTX 5090, H=16 | not specified | VL features via cross-attn | none |
| π0 / π*0.6 | VLM backbone | 50 Hz, 50-step chunks | robot controller | joint-angle chunks | none |
| Gemini Robotics | cloud backbone <160 ms | local decoder, 50 Hz effective, ~250 ms E2E | not specified | action chunks | none stated |
| 1X Redwood | — | ~5 Hz single model | not disclosed | — | joint applied forces as input |
| HiRT | 1–4 Hz | 9.8 Hz | — | latent buffer | none |
| DuoCore-FS | 1–3 Hz | 25–30 Hz | — | latent bridge buffer | none |

### Inferences
- The industry-standard interface is a learned latent (Helix, HiRT, RoboDual, DuoCore-FS) or an action chunk (π0, Gemini, GR00T); no commercial dual-system VLA exposes force targets, stiffness, or impedance parameters between layers.
- Helix 02 is the only shipped stack that (a) puts tactile into a 200 Hz learned loop and (b) inserts a learned 1 kHz whole-body controller (S0) trained purely in simulation; this is the closest public analogue to the proposed "fast loop regulating grip during contact", but Figure documents only quasi-static grip-force regulation (caps, syringes), not impacts.
- Latency-aware training is a recurring trick: Helix's temporal offset between S1/S2 inputs, DuoCore-FS's temporally shifted observations, and PI's training-time action-prefix conditioning (see KQ2) all teach the fast layer to expect stale slow-layer context.

### Gaps
- Helix 02's S2 rate, S1 network size, the exact tactile encoding, and whether S1 outputs any force/stiffness quantity are not disclosed.
- GR00T N1.6's deployed System 1 closed-loop rate is not stated by NVIDIA; the 30 Hz / 50+ Hz figures come from aggregators.
- Hi Robot's level rates, Gemini Robotics 1.5/On-Device latencies, GEN-0's control rate, Skild Brain's rates, Dyna-1/Dyna-2 rates, and any Tesla, Apptronik or Agility architecture details are not publicly documented (search budget was exhausted before Apptronik/Agility could be checked further).

---

## KQ2. Making VLAs real-time: action chunking, real-time chunking, parallel decoding (OFT), asynchronous inference, and measured observation-to-action latency

### Takeaway
Documented VLA control loops top out at 50 Hz effective (via chunks), with per-chunk inference latencies of ~60–100 ms on datacenter GPUs and ~50 ms+ on Jetson-class edge hardware; real-time chunking (inference-time inpainting or training-time prefix conditioning) is now the standard way to hide 100–200 ms delays, and the observation-to-action latency of these systems is in the 100–250 ms range — an order of magnitude too slow to react to an impact.

### Cited Findings
- Real-Time Chunking (PI/Berkeley, Jun 2025): real-world control at 50 Hz (Δt = 20 ms) with H = 50 action chunks for π0.5 and execution horizon s = 25; π0.5 vanilla inference latency 76 ms on GPU, 97 ms with RTC's guidance backprop, plus 10–20 ms LAN latency; baseline delay ≈ 6 steps (~120 ms) with injected +100 ms (d≈11) and +200 ms (d≈16) tested; RTC freezes the first d actions of the previous chunk and soft-masks the remainder with exponentially decaying weights; on 6 real bimanual tasks (incl. match lighting) RTC kept throughput and success under injected delay where temporal ensembling "failed catastrophically". Limitations: only diffusion/flow policies; extra compute. — [RTC paper](https://arxiv.org/html/2506.07339); [RTC abstract](https://arxiv.org/abs/2506.07339)
- Training-time RTC (Black, Ren, Equi, Levine, Dec 2025): simulate inference delay during training and condition on the action prefix, "eliminating any inference-time overhead"; matches inference-time RTC on box building and espresso with π0.6 and surpasses it in simulation as delay grows; "a few additional lines of code". — [Training-time RTC](https://arxiv.org/abs/2512.05964)
- OpenVLA-OFT (2025): parallel decoding + action chunking + continuous (L1) actions; on an A100 with chunk 8, OpenVLA 4.2 Hz / 0.240 s vs OFT 109.7 Hz / 0.073 s (26×); a diffusion head had ~3× higher latency (50 denoising steps); on real ALOHA the policy ran at 25 Hz (reduced from 50) with 25-step chunks, and OFT+ measured 77.9 Hz throughput / 0.321 s latency with 3 cameras + 14-D state. Limitations: L1 regression may not capture multimodal actions; language following was weak on ALOHA. — [OpenVLA-OFT paper](https://arxiv.org/html/2502.19645v1); [project page](https://openvla-oft.github.io/)
- VLA-Perf (NVIDIA Research, 2026) measured π0 latency/throughput: Jetson Thor 52.57 ms (19.0 Hz), RTX 4090 31.06 ms (32.2 Hz), A100 16.20 ms (61.7 Hz), H100 6.15 ms (162.5 Hz), B100 3.18 ms (314.4 Hz); action prediction is memory-bandwidth-bound; Jetson Thor's 19 Hz is below typical 24–60 Hz camera rates; latency scales roughly linearly with model size; diffusion action heads with chunking beat vanilla autoregressive decoding by "one to two orders of magnitude". — [VLA-Perf](https://arxiv.org/html/2602.18397v1)
- LeRobot asynchronous inference (SmolVLA stack): a PolicyServer computes the next chunk while a RobotClient drains an action queue; key knobs are `actions_per_chunk` (default 50, typical 10–50) and `chunk_size_threshold` (docs default 0.7; text recommends 0.5–0.6; 0 collapses to synchronous), with overlapping chunks aggregated by a weighted average; π0 needs ~14 GB at inference vs ~2 GB for SmolVLA; "reduce your fps if you consistently run out of actions in queue". — [LeRobot async docs](https://huggingface.co/docs/lerobot/async); [SmolVLA](https://arxiv.org/abs/2506.01844)
- Gemini Robotics hides a <160 ms cloud round-trip behind 50 Hz chunked execution with ~250 ms end-to-end latency. — [Gemini Robotics report](https://arxiv.org/html/2503.20020v1)
- GR00T N1: 63.9 ms to sample a 16-action chunk on an L40 (4 Euler steps); N1.6: 27.3 Hz inference on RTX 5090 with 16-step horizon. — [GR00T N1](https://arxiv.org/html/2503.14734v1); [GR00T docs](https://nvidia-isaac-gr00t.mintlify.app/architecture)
- World-action models are slower still: NVIDIA reports "between 590ms and 800ms per action chunk, compared to roughly 190ms for Pi-0.5" for two common WAM inference modes. — [NVIDIA WAM blog](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)
- Practitioner deployment study on a real UR5e (Luqia, 2026): OpenVLA ran at ~3 Hz on a remote A100 via Gradio; 256-bin discretization caused 30–40% systematic underestimation of motion amplitude; OFT's chunking gave speedups but "did not eliminate accumulated error during closed-loop control"; recommendation: closed-loop evaluation and "hybrid approaches combining learned policies with classical feedback control". — [UR5 VLA study](https://arxiv.org/html/2606.30456)
- Diffusion policies for force can be run much faster when small: TacDiffusion's wrench-output diffusion models ran at 51.2–503.8 Hz depending on size (141.8 Hz for the chosen model) with a second-order dynamic-system filter interpolating to a 1 kHz force-impedance controller (+9.15% success). — [TacDiffusion](https://arxiv.org/html/2409.11047v1)

### Inferences
- "50 Hz" for π0/Gemini/RTC is the execution rate of pre-computed chunks, not a 50 Hz sensing-to-action loop; the true reaction latency to a new observation is ≥ one inference (60–100 ms) plus queue position, i.e., 100–250 ms.
- Impacts (sub-10 ms rise times) and tool oscillations (tens to hundreds of Hz) therefore cannot be handled inside any VLA inference loop; they must be absorbed by a controller below the VLA that runs at ≥500 Hz and gets local sensing directly, which is exactly the role of Helix's S0, SERL's 1 kHz impedance controller, and TacDiffusion's 1 kHz force-impedance loop.
- Edge deployment (Jetson Thor at 19 Hz for π0) makes chunking + async mandatory on-robot, reinforcing the need for a separate fast loop.

### Gaps
- No source measured end-to-end observation-to-effector latency (including camera capture, transport, and controller tracking) for any commercial dual-system VLA; only inference-side numbers exist.
- Speculative/early-exit decoding for VLAs was not covered (search budget exhausted); PD-VLA is listed only as a related parallel-decoding work. — [PD-VLA](https://www.alphaxiv.org/abs/2503.02310)

---

## KQ3. Interface design between slow and fast layers for contact: impedance/stiffness, force targets, waypoints-with-compliance, latent goals, and tactile in the fast loop

### Takeaway
The academic literature has converged on giving the low-rate learned layer a richer-than-position interface for contact tasks — a virtual/equilibrium pose plus stiffness (ACP, Imp-ACT, VIDP, Impedance Cloning, VICES/Bogdanovic), a wrench (TacDiffusion, HIL-SERL dynamic tasks), or motion goals plus explicit contact-force references (Opt2VLA) — while the only slow/fast learned split that closes the fast loop on tactile is Reactive Diffusion Policy (1–2 Hz latent chunks refined at 24 Hz by tactile) and, in a world-model setting, OmniVTA (60 Hz reflexive tactile controller). Controller gains themselves are an inductive bias: compliant, overdamped low-level gains and 10–20 Hz policies avoid the high-frequency oscillations that stiff gains produce.

### Cited Findings

**Slow/fast learned policies with tactile/force in the fast loop**
- Reactive Diffusion Policy (Mar 2025): a slow Latent Diffusion Policy at 1–2 Hz predicts action chunks in latent space (~0.67 s of actions at 12 FPS); a fast "asymmetric tokenizer" decoder at 24 Hz takes "the latent action chunk and high-frequency tactile representation" at each step and outputs relative end-effector trajectories; tactile input at 24 FPS is limited by the GelSight Mini, though "the system theoretically supports >300 Hz"; sensors: GelSight Mini (25 FPS), MCTac (30 FPS), and Flexiv Rizon 4 joint torque (120 Hz, downsampled to 24 FPS); robots: two Flexiv Rizon 4 arms; limitations: two-finger grippers only, fast policy cannot process high-rate images, single-task. — [RDP paper](https://arxiv.org/html/2503.02881)
- The RDP project page describes the fast decoder as acting like a "learnable impedance controller" with <1 ms inference, reporting 0.95 (peeling), 0.87 (wiping), 0.70 (bimanual lifting). — [RDP project page](https://reactive-diffusion-policy.github.io/)
- FACT (Stanford, Bohg lab, 2026) diagnoses why π0/π0.5-style flow policies fail under contact: "delta collapse" (actions shrink to near-zero in contact) while the Beta noise schedule gives only 8.9% of gradient signal to τ<0.2 where fine corrections live; and force signals are sparse, temporally structured, and need contact-state-dependent sensitivity. Fixes: a logit-normal schedule (6× more gradient in the contact-correction regime), AdaRMSNorm modulation by current force, a 30-step (~2 s) force history as tokens, and gradient gating in non-contact phases. Control stack: policy 15 Hz, joint-torque control 1 kHz, F/T at 400 Hz windowed into 27 samples per policy step. Average success 39.0% (π0.5) → 66.0% (FACT) across plug/USB/key insertion, button push, board erasing, vs ForceVLA 40.5% and TA-VLA 37.5% (~2,500 rollouts). — [FACT](https://arxiv.org/html/2608.01402)

**Force-aware VLAs (force as input, single-rate)**
- ForceVLA (2025): π0 base with an FVLMoE module (4 experts, top-1 routing) fusing 6-axis F/T tokens with VL embeddings; action = TCP pose + gripper width chunks; 60.5% average (up to 80%), +23.2% over π0; limitation: relies on estimated external wrench and F/T-equipped robots; rates not specified. — [ForceVLA](https://arxiv.org/html/2505.22159v1)
- TA-VLA (CoRL 2025): joint torque used as input and as an auxiliary prediction target; decoder-side torque adapters beat encoder-side; auxiliary torque prediction strengthens a "physically grounded internal representation of interaction dynamics". — [TA-VLA](https://arxiv.org/abs/2509.07962)
- FD-VLA (ICRA 2026): a force-distillation module predicts a force token from vision and state so no F/T sensor is needed at deployment; the authors claim the distilled token "outperforms direct sensor force measurements". — [FD-VLA](https://arxiv.org/abs/2602.02142)
- CR-VLA-Force (RA-L, 2026): a multimodal MoE encodes force sequences and a "VLA-guided adaptive compliance controller" executes; rates not given. — [CR-VLA-Force](https://arxiv.org/abs/2609.05832)
- Opt2VLA (2026): a single VLA "jointly predicts both geometric motion goals and continuous contact-force references" that task-specific RL whole-body controllers track on a humanoid; training data come from whole-body trajectory optimization with explicit force references and torque supervision; "explicit force conditioning enables more accurate and consistent force regulation than motion-only control". — [Opt2VLA](https://arxiv.org/abs/2609.23968)
- 1X Redwood takes "joint applied forces" as proprioceptive input for bracing-type behaviors. — [1X Redwood](https://www.1x.tech/discover/redwood-ai)

**Compliance/impedance as the policy output**
- Adaptive Compliance Policy (Stanford/TRI, 2024): per arm a 19-D output = 9-D reference pose + 9-D virtual target pose + scalar stiffness; the direction between virtual and reference targets encodes where to be compliant (low stiffness along the sensed force direction, justified by a constraint-violation theorem), and stiffness decreases piecewise-linearly with force magnitude; stack: 60 Hz camera, 500 Hz pose command/feedback, 7 kHz ATI Mini-45 F/T, admittance controller; flipping 96% vs 23% (compliant) / 14% (stiff); wiping 93.75% vs 43.75%; limitation: needs kinesthetic teaching. — [ACP](https://arxiv.org/html/2410.09309)
- Imp-ACT (IIT/CMU, 2026): 11-D action = Cartesian equilibrium position (3) + orientation (6) + gripper + a motion-direction stiffness parameter; policy at 50 Hz, Franka FCI at 1 kHz, teleop at 72 Hz, cameras 30 fps, ATI Mini-45; wiping 100% with ~29× less vibration than a compliant baseline and ~180× less than a stiff one; plug insertion 76% vs 72%/60% with 43% lower transverse forces; future work: rotational stiffness and a passivity filter "to regulate energy injection associated with stiffness variations". — [Imp-ACT](https://arxiv.org/html/2609.31225)
- VIDP (2026): diffusion policy jointly predicts pose and stiffness profiles inferred from kinematic demonstrations alone (TP-DAMM), no force sensors; outperforms fixed-impedance baselines while reducing interaction forces. — [VIDP](https://arxiv.org/abs/2608.06210)
- Impedance Cloning (2026): learns stiffness + equilibrium point from bilateral teleoperation via a particle filter without F/T sensors; maintains 4–5 N in wiping where trajectory baselines lose contact; 84/100 pick-and-place from one demo on CRANE-X7. — [Impedance Cloning](https://arxiv.org/abs/2609.30842)
- TacDiffusion (TUM, 2024): the policy outputs a 6-D wrench from an 18-D input (external wrench, internal wrench, EE velocity, current + previous) at 51–504 Hz, filtered into a 1 kHz force-impedance controller on a Franka; 95.7% zero-shot on novel sub-millimeter insertions. — [TacDiffusion](https://arxiv.org/html/2409.11047v1)
- Foundational: VICES (Martín-Martín et al., IROS 2019) proposed variable impedance in end-effector space as the RL action space, improving sample efficiency, energy and safety on path following, door opening and wiping with sim-to-real; Bogdanovic, Khadiv, Righetti (2019/2020) had RL output impedance plus desired joint positions for contact-sensitive tasks with a regularizer for interpretability. — [VICES](https://arxiv.org/abs/1906.08880); [Bogdanovic et al.](https://arxiv.org/abs/1907.07500)

**Controller gains as the interface**
- "Tune to Learn" (MIT Improbable AI, 2026): behavior cloning works best with "compliant and overdamped gain regimes" because the controller attenuates action-prediction error (position-error variance scales with stiffness/damping); RL finds solutions across gains spanning two orders of magnitude; stiff, overdamped gains hurt sim-to-real via "high-frequency oscillations", which lower policy frequency (10–20 Hz vs 100 Hz) mitigates. — [Tune to Learn](https://arxiv.org/html/2604.02523)
- HIL-SERL uses 6-D Cartesian twist targets for a downstream impedance controller for continuous tasks and, for dynamic tasks, has the policy "directly command feedforward wrenches in the end-effector frame"; the impedance controller applies "reference limiting in the real-time layer to ensure safety". — [HIL-SERL](https://arxiv.org/html/2410.21845)

### Inferences
- For a hammering/drilling fast loop, the literature's best-supported interface from the slow layer is (target pose or trajectory latent) + (stiffness/compliance direction and magnitude) + (a force/wrench reference), tracked by a ≥500 Hz impedance/admittance or torque controller; ACP, Imp-ACT and Opt2VLA each realize a subset of this.
- Tactile has only been closed at 24–60 Hz in learned fast loops (RDP, OmniVTA), bounded by optical tactile sensor frame rates; sub-100 Hz tactile cannot see impact transients, but F/T (400 Hz–7 kHz) and joint torque (120 Hz–1 kHz) can, which is why the FACT and TacDiffusion stacks route F/T into the fast side.
- Imp-ACT's vibration reductions (29×/180×) and its call for a passivity filter indicate that variable stiffness at 50 Hz can itself inject energy; a high-energy tool loop needs passivity-aware stiffness scheduling.

### Gaps
- No slow/fast learned system was found that uses tactile or force in the fast loop for impacts, vibration, or powered tools; all contact-rich examples are quasi-static (insertion, wiping, peeling, flipping).
- HACTS was not located in this session; Opt2VLA's controller rates and Impedance Cloning's controller rates are not in the abstracts.

---

## KQ4. RL fine-tuning, residual policies, and learned compliance on top of VLAs / foundation policies; real-world RL controller stacks; impacts and tools

### Takeaway
Real-world RL on manipulators is consistently built as a 10 Hz learned policy over a 1 kHz impedance controller with wrist F/T in the observation (SERL/HIL-SERL), and RL-on-VLA methods (ConRFT, RLDG, iRe-VLA, HiL-ResRL, Recap) reuse either that stack or offline/advantage-conditioned updates; the only dynamic-contact RL results found (HIL-SERL's pan flipping and Jenga whipping) command feedforward wrenches, and no RL-fine-tuned VLA has been demonstrated on hammering or powered tools.

### Cited Findings
- SERL (2024): policy at 10 Hz; impedance controller at 1 kHz; action = 6-D end-effector delta pose; the controller clips the reference so |e| ≤ Δ to bound interaction forces ("clipping the reference in this way is simple but very effective"); observations include "end-effector pose, twist, force, and torque"; PCB insertion learned in 20 min (100%), cable routing 31 min, object relocation 105 min, from 20 demos each; RL beat BC by up to 10× with 5× fewer demos. — [SERL](https://arxiv.org/html/2401.16013)
- HIL-SERL (Luo et al., 2024): 1–6 h of training with human interventions to near-100% success on RAM insertion, timing-belt assembly, object flipping in a pan, Jenga whipping and dual-arm handover; 1.8× faster cycle time than BC; RAM insertion 1.5 h to 100% vs 29% BC; dynamic tasks use feedforward wrench commands. — [HIL-SERL](https://arxiv.org/html/2410.21845); [project page](https://hil-serl.github.io/)
- A 2026 paper building on this stack restates the standard hierarchy: "A high-level RL controller sends control targets at 10 Hz for the low-level impedance controller to track at 1 kHz", with F = k_p·e + k_d·ė + F_ff + F_cor. — [Agent-guided robotic RL](https://arxiv.org/html/2602.11978v2)
- RLDG (Xu, Li, Luo, Levine, 2024): uses RL policies to generate training data for fine-tuning generalist policies, giving up to 40% higher success on connector insertion/assembly than human demos. — [RLDG](https://arxiv.org/abs/2412.09858)
- ConRFT (RSS 2025): consistency-policy reinforced fine-tuning of a VLA with alternating RL/SFT; 96.3% average across eight real tasks within 45–90 min of online fine-tuning, +144% success and 1.9× shorter episodes vs supervised fine-tuning; motivated by "contact-rich environments". — [ConRFT](https://arxiv.org/abs/2502.05450); [RSS listing](https://roboticsconference.org/program/papers/19/)
- iRe-VLA (ICRA 2025): iterates RL and supervised stages to avoid instability and compute burden; two simulated benchmarks and a real manipulation suite. — [iRe-VLA](https://arxiv.org/abs/2501.16664)
- VLA-RL (2025): trajectory-level RL for autoregressive OpenVLA-7B with a robotic process reward model; +4.5% over the strongest fine-tuned baseline on 40 LIBERO tasks, matching π0-FAST; no real-robot experiments in the abstract. — [VLA-RL](https://arxiv.org/abs/2505.18719)
- HiL-ResRL (2026): a residual RL policy trained on top of a frozen VLA with human-in-the-loop guidance, >95% success after 1.5 h of real-world online RL; model-agnostic. — [HiL-ResRL](https://arxiv.org/abs/2606.22860)
- Recap (π*0.6) applies advantage-conditioned offline RL plus corrections to a 50 Hz flow-matching VLA without force sensing (see KQ1). — [π*0.6](https://arxiv.org/html/2511.14759)
- Residual RL (Johannink et al., 2018): the final policy is "a superposition" of a conventional controller and an RL residual that handles "contacts and friction, which are difficult to capture with first-order physical modeling", demonstrated on a real block assembly with contacts and unstable objects. — [Residual RL](https://arxiv.org/abs/1812.03201)
- Residual MPC (MIT, 2025): RL residual torques added to MPC torques (τ = τ_MPC + λa) at 100 Hz (OSQP on CPU) on the MIT Humanoid; +78% trackable forward velocity; 3–4 h training vs 30 min for end-to-end RL; notable sim-to-real gap. — [Residual MPC](https://arxiv.org/html/2510.12717v1)
- Impact tasks: Ti, Gao, Zhao, Calinon (2024) formulate hammering a nail (pilot-hole assisted) as an optimal control problem (iLQR + ADMM) maximizing directional velocity manipulability on a real 7-axis robot; the abstract does not describe explicit impact-dynamics modeling. — [Tool affordance for impact tasks](https://arxiv.org/abs/2402.05502)
- Related but not detailed here: a residual RL method for robotic assembly using visual and force information, and Tool-as-Interface (learning policies from human tool usage). — [Residual RL assembly](https://www.sciencedirect.com/science/article/abs/pii/S0278612523002352); [Tool-as-Interface](https://arxiv.org/html/2504.04612v1)

### Inferences
- The SERL family shows that a 10 Hz learned layer suffices for insertion-class contact when a 1 kHz impedance controller with reference limiting absorbs the dynamics; the "fast loop" in these systems is not learned at all.
- HIL-SERL's dynamic tasks demonstrate that switching the interface from pose targets to feedforward wrenches is what enabled impulsive behaviors (whipping a Jenga block), which supports a wrench/force interface for hammering.
- Residual RL over a model-based controller (Johannink; Residual MPC) is the most direct template for "learned corrections during impacts" but has not been demonstrated on manipulator tool impacts.

### Gaps
- No RL-fine-tuned VLA or residual-RL result involving hammering, drilling, or other high-energy tool impacts was found.
- Silver et al. 2018 "Residual Policy Learning" was not fetched/verified in this session and is therefore not cited.
- HIL-SERL's exact policy/controller rates were not confirmed from its own HTML (only via SERL and a citing paper); whether F/T enters the HIL-SERL policy observation is stated only for SERL.

---

## KQ5. Learned world models for manipulation (2023–2026): time steps, contact/tactile treatment, hierarchical and multi-rate models, and whether any can represent impacts

### Takeaway
Manipulation world models run at 4–16 fps (V-JEPA 2-AC 4 fps; VT-WM and FeelWorld 6 fps; ViTacWorld 15 Hz; Cosmos-Predict2 10–16 fps; Genie 3 24 fps) and plan with CEM at seconds-to-minutes per action, so none can represent sub-100 ms impact events; the 2026 wave of visuo-tactile world models (VT-WM, FeelWorld, OmniVTA, TacForeSight, ViTacWorld, Dream-Tac) adds contact geometry/deformation and slip prediction at 6–60 Hz but explicitly does not predict forces, and hierarchical/multi-timescale world models (THICK, Director, MTS-WM) abstract upward in time rather than downward toward impact dynamics.

### Cited Findings

**Latent / video world models used for control**
- V-JEPA 2-AC (Meta, 2025): action-conditioned predictor at 4 fps, actions are 7-D end-effector deltas (position, Euler orientation, gripper); CEM planning with 800 samples × 10 iterations takes "16 seconds per action" vs 4 minutes for a Cosmos baseline; Franka Panda success: reach 100%, grasp cup 65% / box 25%, pick-and-place cup 80% / box 65%; limitations: manual camera positioning, error accumulation over long horizons, visual goals only. Post-trained on <62 h of DROID video. — [V-JEPA 2 paper](https://arxiv.org/html/2506.09985); [alphaXiv summary](https://www.alphaxiv.org/abs/2506.09985)
- DINO-WM (NYU/Meta, 2024): ViT transition model over DINOv2 patch latents; CEM/MPC planning takes ~53 s for 100 samples × 10 optimization steps (0.014 s per batch inference); tasks include Push-T, rope and granular manipulation; limitation: "current planning occurs in action space rather than hierarchical structures for fine-grained control". — [DINO-WM](https://arxiv.org/html/2411.04983)
- TD-MPC2 (ICLR 2024): decoder-free latent world model with MPPI local trajectory optimization; 104 tasks; a 317M-parameter multi-task agent; no real-robot or contact-rich results are stated in the abstract. — [TD-MPC2](https://arxiv.org/abs/2310.16828)
- DreamerV3 (2023): a single configuration over 150+ tasks with an RSSM world model and imagination-based actor-critic; no real-robot contact results in the abstract. — [DreamerV3](https://arxiv.org/abs/2301.04104)
- Cosmos-Predict2-2B-Video2World: 480P–720P at 10–16 fps, 5-s clips (80 frames at 16 fps); base model has no action conditioning (a separate 2B sample variant does); 720P generation takes 111–229 s on an H100 and 1,281–2,567 s on an L40S; NVIDIA acknowledges limits in "accurately representing physical laws". — [Cosmos-Predict2 model card](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)
- NVIDIA's WAM framing: policies built on pretrained video backbones cost 590–800 ms per action chunk vs ~190 ms for π0.5, with video-generation cost, memory, and unresolved language grounding as limitations. — [NVIDIA WAM blog](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)
- Genie 3 (DeepMind, Aug 2025): 24 fps at 720p, real-time interaction, consistency for "several minutes" with ~1 minute visual memory; "the range of actions agents can perform directly is currently constrained". — [Genie 3 blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)
- 1X World Model: an action-conditioned video generator used as "a virtual simulator" for policy evaluation on EVE; documented failure modes include objects losing shape or disappearing and "inconsistent adherence to gravity"; no contact/force or fps is stated. — [1X World Model](https://www.1x.tech/discover/1x-world-model)
- Dyna-2: video-diffusion world-action model with a shallower action stream for latency; 1M-hour human-video scaling law; no force/tactile; no latency numbers. — [Dyna-2](https://www.dyna.co/dyna-2)
- GR00T N1.5/N1.6 are trained "jointly with flow matching and world-modeling objectives" (no details on the world-model time step). — [GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B)

**Visuo-tactile world models (2026)**
- VT-WM (UW / FAIR, Feb 2026): 12-layer factorized spatio-temporal transformer; vision via the Cosmos tokenizer at 6 fps (9 frames = 1.5 s, 320×192) and Digit 360 tactile via Sparsh-X (2 frames per sensor over 0.16 s); 30 Hz actions chunked in groups of 5; CEM planning; +33% object permanence, +29% "causal compliance" in rollouts, up to +35% zero-shot real success; captures contact "implicitly through tactile images" and does not model force; CEM is expensive and executed open-loop. — [VT-WM](https://arxiv.org/html/2602.06001v1)
- FeelWorld (CASIA / Imprintx / BAAI, 2026): V-JEPA 2 visual backbone + frozen FG-CLTP tactile encoder over DM 3-D tactile point clouds; hierarchical heads predict contact probability, a 3-D tactile latent, and slip; runs at 6 fps downsampled from 30 Hz with a 9-frame (1.5 s) context; contact-gated CEM planning; contact F1 98.1%, slip F1 83.4%; zero-shot planning 82.5% chips, 87.5% fruit, 75% USB; it "encodes force-related contact information" rather than raw force and predicts slip "rather than high-frequency force events"; authors state CEM "is not suitable for high-frequency real-time control". — [FeelWorld](https://arxiv.org/html/2607.24267)
- OmniVTA (NUS TARS et al., 2026): TactileVAE + two-stream visuo-tactile diffusion-transformer world model + adaptive fusion policy + a "Reflexive Latent Tactile Controller" that outputs refined actions at 60 Hz from predicted-vs-observed tactile features; sensors at 25–60 Hz (Xense 60 Hz, Daimon 60 Hz, Tac3D 30 Hz, GelSight Mini 25 Hz); models 3-D displacement fields, not forces, with "a dynamic weight map based on local temporal differences" to emphasize high-frequency contact dynamics; 21,879 trajectories / 86 tasks; success 80% wiping, 55% peeling, 85% cutting, 60% assembly, 90% grasping. — [OmniVTA](https://arxiv.org/html/2603.19201v1)
- TacForeSight (NUS TARS, 2026): an 11.8M-parameter force-conditioned tactile world model (TacForceWM) plus a 68.9M predictive policy; Xense 35×20 3-D displacement maps at 30 Hz and wrist F/T at 120 Hz; force conditions the tactile latent dynamics via adaptive layer norm but "the system does not predict future force values"; real-time at 20 Hz on an RTX 4090D; 79.0% nominal / 86.7% under perturbation across wiping, card swiping, tube/bulb/wire insertion, beating RDP and others. — [TacForeSight](https://arxiv.org/html/2606.11184)
- ViTacWorld (ShanghaiTech, 2026): action-conditioned DiT extending a pretrained robot video world model; 13-frame windows from 15 Hz trajectories; used mainly to generate "dream data" (policy success 42.5% → 67.5% → 80% with two augmentation rounds); "no explicit force or impact modeling". — [ViTacWorld](https://arxiv.org/html/2607.22530v1)
- Dream-Tac (PKU/HKUST, 2026): video-DiT world-action model with contact-gated visuo-tactile fusion; Xense Photon sensors; observations at 30 Hz, H = 20 action chunks; 83.3% average on six tasks vs Cosmos Policy 51.7% and ForceVLA 50.8%; force "not explicitly modeled". — [Dream-Tac](https://arxiv.org/html/2606.08737v1)
- DexTacWAM injects per-fingertip tactile latents into a video diffusion world model for dexterous hands (details not extracted). — [DexTacWAM](https://www.researchgate.net/publication/414564190_DexTacWAM_A_Visuo-Tactile_World-Action_Model_for_Dexterous_Manipulation)

**Hierarchical / multi-timescale world models**
- THICK (ICLR 2024): a two-level world model where the lower level "selectively updates parts of its latent state sparsely in time, forming invariant contexts, while the higher level is trained exclusively to predict situations involving these sparse context state changes"; environments are MiniHack, VisualPinpad and Multiworld door-hook; used via THICK Dreamer and THICK PlaNet. — [THICK repo](https://github.com/CognitiveModeling/THICK); [OpenReview](https://openreview.net/forum?id=5qappsbO73r)
- Director (2022): a manager selects latent goals in the world model's representation and a worker learns to reach them; evaluated on egocentric quadruped mazes, visual control, Atari and DMLab. — [Director](https://arxiv.org/abs/2206.04114)
- Multi Time Scale World Models (KIT/Bosch/SAP, 2023): hierarchical linear-Gaussian state-space models with a fast SSM at Δt and a slow SSM at H·Δt (H ∈ {3, 10, 30, 75} ablated), the slow latent parameterizing the fast dynamics; evaluated on a Panda with varying payloads, a hydraulic excavator, D4RL and a wheeled robot; limitations include linear task-conditional dynamics and 2-level hierarchies only. — [MTS-WM](https://arxiv.org/html/2310.18534)

### Inferences
- The fastest learned world-model loop found for manipulation is OmniVTA's 60 Hz reflexive tactile controller (bounded by tactile sensor rate); world-model-in-the-loop planning (CEM) is 16 s–4 min per action, ruling out MPC-over-world-model for anything dynamic.
- All tactile world models represent contact as deformation/geometry/slip, not force; only TacForeSight conditions on F/T (120 Hz) and none predicts force, so none can forecast impact loads or tool vibration.
- Multi-timescale world models exist (MTS-WM with explicit fast/slow SSMs; THICK; Director) but their "fast" level is the base simulation rate (tens of Hz) and the hierarchy points toward longer horizons; nothing in this literature adds a kHz-scale contact-dynamics level below the vision-rate model.
- The practical role of world models for a hammering/drilling system today is offline: data augmentation (ViTacWorld), policy evaluation (1X, ViTacWorld), or pretraining a policy backbone (Dyna-2, GR00T N1.6), not online impact prediction.

### Gaps
- No world model (video, latent, or tactile) with a time step below ~16 ms or any demonstrated representation of impacts, vibration, or powered-tool dynamics was found.
- Genie 3 and Cosmos are not action-conditioned on robot joint/force commands at the level needed for manipulation control; IRASim, RoboDreamer, UniPi/UniSim, Vid2World and Hieros were not covered (search budget exhausted).
- NVIDIA "TWM" and an MIT tactile-dynamics-for-insertion work were not located.

---

## KQ6. Model-predictive control with learned or simulated models at high rate on real hardware for contact

### Takeaway
On hardware, MPC over neural or physics models reaches 50–100 Hz only for non-contact or locomotion problems (Neural-MPC on quadrotors at 50 Hz; Residual MPC on a humanoid at 100 Hz), while sampling-based MPC for contact-rich manipulation with a GPU physics model replans at only 8–10 Hz and is fundamentally limited by the small simulation timesteps contact requires; learned contact models that capture impacts (ContactNets) exist but have not been shown inside a real-time manipulation MPC loop.

### Cited Findings
- Real-time Neural-MPC (Salzmann et al., RA-L 2023): integrates neural dynamics "over 4000 times larger" than prior neural MPC into a "50Hz real-time window on an embedded platform" on an agile quadrotor, cutting tracking error up to 82%; no contact tasks. — [Neural-MPC](https://arxiv.org/abs/2203.07747)
- Residual MPC (MIT, 2025): MPC at 100 Hz on CPU with an RL torque residual on the MIT Humanoid (see KQ4). — [Residual MPC](https://arxiv.org/html/2510.12717v1)
- Structured NN MPC (Neural Computation 2022): a neural dynamics model with explicit inertia-matrix representation and a two-stage procedure for contact-rich dynamics from limited samples, applied to a trackball manipulation task with a physical 3-DoF finger robot, outperforming a fully connected-network MPC (control rate not retrievable; page returned 403). — [SNN-MPC (MIT Press)](https://direct.mit.edu/neco/article/34/2/360/108537/Implicit-Contact-Dynamics-Modeling-With-Explicit)
- Massively parallel sampling-based MPC on a Franka Research 3 (TU Darmstadt, 2026): 1,024 sampled control sequences over a 0.375 s horizon in MuJoCo MJX, replanning at 8–10 Hz with action selection at 50 Hz and a 1 kHz MoveIt2 servo layer; "small timesteps are required to avoid deep interpenetrations", which conflicts with real-time replanning; collision-margin (contact-initiation) parameters yield usable online adaptation signals whereas global physics parameters do not. — [Sampling-based MPC deployment](https://arxiv.org/html/2606.20712v1)
- MuJoCo MPC (DeepMind, 2022): an open-source real-time predictive-control framework with Predictive Sampling; the abstract frames it as a simulation framework without hardware rates. — [MuJoCo MPC](https://arxiv.org/abs/2212.00541)
- MuJoCo MPC on a physical biomimetic tendon-driven hand (2024): first sampling-based MPC in-hand manipulation (ball rolling, flipping, catching) on such hardware, with a VLM adapting the objective; replanning rate not stated. — [Tendon-hand MPC](https://arxiv.org/abs/2411.06183)
- Hierarchical contact-implicit MPC for in-hand manipulation (2025): high-level contact-implicit MPC with a low-level tracker using hand force-motion models and tactile feedback; five real-world tasks under disturbances; analytic rather than learned models; rates not in abstract. — [In-hand motion-contact planning](https://arxiv.org/abs/2505.04978)
- ContactNets (Pfrommer, Halm, Posa, CoRL 2020): learns inter-body signed distance and contact-frame Jacobians to capture "impact, non-penetration, and stiction" from 60 s of real data; the abstract says the representation is "compatible with many simulation, control, and planning environments" but does not report closed-loop control. — [ContactNets](https://arxiv.org/abs/2009.11193)

### Inferences
- Contact MPC on hardware is throttled by simulation timestep, not just compute: the Darmstadt result (8–10 Hz with 1,024 GPU rollouts) shows that even a physics model, let alone a learned video model, cannot replan at impact timescales.
- The viable high-rate learned component below 100 Hz→1 kHz is therefore a reactive policy or residual (Residual MPC, S0, TacDiffusion), not a planner; learned contact models like ContactNets are better suited to offline model identification of impact behavior than to online MPC.

### Gaps
- No example of MPC over a learned dynamics model at ≥100 Hz on a manipulator performing impacts or tool use was found; a "Learning Legged MPC with Smooth Neural Surrogates" paper appeared in results but was not reviewed. — [Legged MPC surrogates](https://arxiv.org/pdf/2601.12169)

---

## KQ7. Design-pattern synthesis: what practitioners recommend for the slow/fast split in contact-rich tasks (and what it implies for hammering/drilling)

### Takeaway
Across commercial stacks and the contact-rich learning literature the consistent recipe is: a 1–10 Hz VLM emits latent/sub-goal context; a 10–50 Hz learned policy converts it into compliant motion targets (pose + stiffness and/or wrench) using F/T at 100 Hz–7 kHz and tactile at 25–60 Hz; a 500 Hz–1 kHz impedance/torque controller (hand-designed or, in Helix 02, learned in simulation) absorbs the contact dynamics; and training deliberately simulates the latency between layers. No published system handles impacts or powered tools, so the 1 kHz layer must be designed from control principles (reference limiting, passivity) rather than learned from demonstrations.

### Cited Findings
- Three-tier split with a learned kHz layer: Helix 02's S2 (latent goals) → S1 (200 Hz joint targets from cameras + fingertip tactile + proprioception) → S0 (1 kHz joint commands, sim-trained in 200k parallel environments). — [Figure, Helix 02](https://www.figure.ai/news/helix-02)
- Latency-aware training of the fast layer: Helix's "temporal offset between S1 and S2 inputs"; DuoCore-FS's temporally shifted observations in cross-timescale co-training; PI's training-time action-prefix conditioning. — [Figure, Helix](https://www.figure.ai/news/helix); [DuoCore-FS](https://arxiv.org/html/2512.20188v1); [Training-time RTC](https://arxiv.org/abs/2512.05964)
- Real-world RL practitioners run a 10 Hz policy over a 1 kHz impedance controller with reference limiting and wrist F/T in the observation; dynamic tasks switch the interface to feedforward wrenches. — [SERL](https://arxiv.org/html/2401.16013); [HIL-SERL](https://arxiv.org/html/2410.21845)
- Force needs a contact-state-aware fast path inside the policy (gradient gating, force history tokens, timestep modulation) — merely appending F/T to a 15 Hz flow policy is insufficient (ForceVLA 40.5% vs FACT 66.0%). — [FACT](https://arxiv.org/html/2608.01402)
- Choose compliant, overdamped low-level gains and 10–20 Hz policy rates for imitation-learned policies to avoid high-frequency oscillation on real hardware; gains are an inductive bias, not just a task parameter. — [Tune to Learn](https://arxiv.org/html/2604.02523)
- Emit compliance explicitly: ACP's reference + virtual-target + stiffness at 500 Hz command rate with 7 kHz F/T; Imp-ACT's equilibrium pose + directional stiffness at 50 Hz over a 1 kHz FCI loop, with a passivity filter recommended for energy injection. — [ACP](https://arxiv.org/html/2410.09309); [Imp-ACT](https://arxiv.org/html/2609.31225)
- Emit force references alongside motion goals and let RL whole-body controllers track them (humanoid). — [Opt2VLA](https://arxiv.org/abs/2609.23968)
- Put tactile in a separate fast decoder (24 Hz) that refines 1–2 Hz latent chunks, and treat that decoder as a learnable impedance-like controller. — [RDP](https://arxiv.org/html/2503.02881); [RDP project page](https://reactive-diffusion-policy.github.io/)
- If a policy must output forces at low rate, bridge to the 1 kHz controller with a dynamic-system (second-order) filter rather than zero-order hold. — [TacDiffusion](https://arxiv.org/html/2409.11047v1)
- Deployment practitioners recommend hybrid learned + classical feedback control and closed-loop (not offline) evaluation because VLA chunks drift in closed loop. — [UR5 VLA study](https://arxiv.org/html/2606.30456)
- Edge inference budgets (π0 at 19 Hz on Jetson Thor) mean the VLA cannot be the reactive loop on-robot. — [VLA-Perf](https://arxiv.org/html/2602.18397v1)
- Impact-specific planning today is model-based: hammering formulated with iLQR/ADMM to maximize directional velocity manipulability. — [Tool affordance for impact tasks](https://arxiv.org/abs/2402.05502)

### Inferences
- For the proposed system (vision + tactile + joint state + joint effort → hammer/drill), the evidence supports: (1) VLA at 1–10 Hz producing a latent or sub-goal plus a short target trajectory; (2) a 50–200 Hz learned mid-layer (Helix S1 / RDP-style decoder) that ingests tactile (≤60 Hz), joint torque/F/T (≥120 Hz, windowed like FACT's 27-sample F/T windows) and outputs equilibrium pose + stiffness + wrench feedforward; (3) a ≥1 kHz impedance/torque controller with reference limiting and passivity safeguards, which is where impact regulation, grip-force clamping and oscillation damping actually happen.
- The "world model" layer, given current capabilities, should be used for offline pretraining, data augmentation and outcome evaluation, or at most for slow (≤6 fps) contact/slip anticipation as in FeelWorld/TacForeSight; it should not be on the critical path for impact response.
- Learning the kHz layer is feasible only via sim-to-real RL with massive parallelism (Helix S0 precedent); learning it from demonstrations is unsupported by any result found.

### Gaps
- No workshop reports, talks or blog posts specifically recommending slow/fast splits for impact/tool tasks were retrievable (search budget exhausted before workshop-page queries); the synthesis above is inferred from the primary papers cited.
- No published quantitative data on grip-force regulation during hammer impacts or drill engagement transitions by any learned system was found.
