# Grip-force anticipation, slip detection under impulsive loads, and trial-to-trial grip adaptation: prior work vs. the planned hammering paper

Research date: 2026-10-02. 17 tool calls; the abstracts and HTML pages I could reach are listed. PubMed, Springer, and Europe PMC were blocked by cookie walls, paywalls, or HTTP 503, so the numbers for several neuroscience papers come from secondary citations. Each such case is flagged.

## Q1. Human motor-control timing for grip during impacts and hammering (verifying the >=150 ms lead and the 50-70 ms post-peak claims)

### Takeaway
The "grip peaks ~60-65 ms after impact/peak load" claim is well supported. White et al. 2011 (Neuroscience) is the main source, and White et al. 2018 summarises the literature as "~60 ms after peak load force". The ">=150 ms lead" is plausible but should be cited as "positive grip-force rate for ~200 ms before impact". That figure appears in Turrell et al. 1999 (secondary citation) and White et al. 2018 (high-stiffness condition). It does not support a fixed 150 ms onset. White 2011 is a tapping task with a hand-held object, not real hammering. Peak timing also depends on condition: a soft surface gives peak at impact (~4 ms), and a stiff surface gives a peak ~40 ms after peak force.

### Cited Findings
- White O, Thonnard J-L, Wing AM, Bracewell RM, Diedrichsen J, Lefèvre P (2011), "Grip force regulates hand impedance to optimize object stability in high impact loads," *Neuroscience* 189:269-276, doi:10.1016/j.neuroscience.2011.04.055. Participants did a **targeted tapping task with a hand-held object**, not hammering. "In collisions, grip force peaked approximately 65 ms after the impact." The authors argue the CNS regulates stiffness and damping through grip force. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306452211004957); [PubMed 21640167](https://pubmed.ncbi.nlm.nih.gov/21640167/) (abstract text came via search snippet; PubMed was cookie-walled)
- White O, Karniel A, Papaxanthis C, Barbiero M, Nisky I (2018), "Switching in feedforward control of grip force during tool-mediated interaction with elastic force fields," *Front. Neurorobot.* 12:31:
  - Literature summary: "A common observation was the occurrence of a maximum of grip force approximately 60 ms after peak load force that signed the impact." This cites Johansson & Westling 1988, Bleyenheuft et al. 2009, and White et al. 2011, 2012.
  - In their own data, at high stiffness "peak grip force was clearly delayed by 40 ms (SD = 6 ms) after the peak of the elastic force field."
  - At low stiffness, peaks "were synchronized with the impact, both in real and catch trials (mean = 4 ms, SD = 6 ms)."
  - "positive grip force rates for 200 ms before impact."
  - Peak delay was similar in real and catch trials, which suggests the peak is pre-programmed.
  - [Frontiers](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2018.00031/full); [PMC5999723](https://pmc.ncbi.nlm.nih.gov/articles/PMC5999723)
- Turrell YN, Li FX, Wing AM (1999), "Grip force dynamics in the approach to a collision," *Exp Brain Res* 128:86-91, PMID 10473745. Secondary summaries (search snippets) describe the profile this way: grip first peaks with the inertial load, dips, then rises again in anticipation of contact, with positive grip-force rates for ~200 ms before impact. — [PubMed 10473745](https://pubmed.ncbi.nlm.nih.gov/10473745/). I could not read the primary abstract (cookie wall). Treat the 200 ms figure as Turrell-attributed only through secondary sources, or cite White 2018 for it.
- Delevoye-Turrell YN, Li FX, Wing AM (2003), "Efficiency of grip force adjustments for impulsive loading during imposed and actively produced collisions," *Q J Exp Psychol A* 56A:1113-1128. It is **not** in J Neurophysiol, as one might assume. — [search result listing](https://pubmed.ncbi.nlm.nih.gov/10473745/) (citation found via search; I did not obtain the numbers)
- "Grip force preparation for collisions," *Exp Brain Res* 2019, doi:10.1007/s00221-019-05606-y. This is directly relevant, but the page was paywalled and I could not extract its timing numbers. — [Springer](https://link.springer.com/article/10.1007/s00221-019-05606-y)
- Danion and colleagues: "Grip force anticipation of nonlinear, underactuated load force," *J Neurophysiol* (2020/21), doi:10.1152/jn.00616.2020. It describes grip control as a feedforward baseline plus anticipatory modulation from an internal model that is updated across repeated interactions. This is relevant background for trial-to-trial updating. — [J Neurophysiol](https://journals.physiology.org/doi/full/10.1152/jn.00616.2020)
- Bleyenheuft & Thonnard (2010), on predictive vs. reactive grip control in hemiplegic children. This is a related collision/impact grip paradigm from the same group. — [Neurorehabil Neural Repair](https://doi.org/10.1177/1545968309353327)

### Inferences
- The ">=150 ms lead" is conservative relative to the ~200 ms anticipatory window and is defensible. Frame it as "grip-force rate turns positive ~200 ms before contact (Turrell 1999; White 2018)", not as a measured 150 ms onset.
- "Peak 50-70 ms after peak load" fits the ~60-65 ms figure (White 2011; White 2018 review). The paper should note that the delay depends on contact stiffness: ~40 ms at high stiffness and ~0-4 ms at low stiffness (White 2018). A hammer on a rigid target is the high-stiffness or true-impact case.
- None of the cited human studies is real hammering at strike speeds. All use tapping or collisions with instrumented manipulanda. If the planned paper says "human hammering grip", it should say "impact/tapping paradigms".
- The paper's own premise (a ~4 ms blow vs. 50-185 ms actuation) mirrors the human argument that feedback is too slow for impacts (White 2011). This is a good framing.

### Gaps
- I did not obtain primary grip/load ratios (GF/LF) for impact paradigms. White 2011 and Delevoye-Turrell 2003 report safety-margin data, but I could not read it (paywalls). Check the PDFs directly.
- Not verified from a primary source: Johansson & Westling 1988 (*Exp Brain Res* 71:72-86, "Programmed and triggered actions to rapid load changes during precision grip"). It is cited for the ~60 ms post-impact peak by White 2018. From my background knowledge: for self-triggered loading the grip rises in anticipation, and for unexpected loading a triggered response comes at ~60-80 ms latency. Confirm before citing those latencies.
- Not checked: Flanagan & Wing (1993, 1997) on grip-load coupling in cyclic movements.
- I could not extract the numbers from Exp Brain Res 2019, "Grip force preparation for collisions".

## Q2. Robots that scale grip ahead of a predicted impulsive load (anticipatory/feedforward grip)

### Takeaway
I found **no robotics paper that schedules a grip-force ramp ahead of a predicted impact, sized from tool mass, strike speed and friction**. Closely related work does exist:
- grasp *selection* that minimises predicted wrench during hammering (iTuP/SDG-Net, 2025/26)
- RL residual joint corrections to prevent in-hand slip during hammering with a dexterous hand (Grasp to Act, 2026)
- internal-model *feedforward wrench* compensation for arm force control, not grip (IMPACT, 2026)

The anticipatory-grip contribution looks novel, but reviewers will point to these three.

### Cited Findings
- **Physics-Conditioned Grasping for Stable Tool Use** (Trupin, Wang, Qureshi, Purdue; arXiv:2505.01399, May 2025, revised Mar 2026). Inverse Tool-use Planning (iTuP) selects grasps "by minimizing predicted interaction wrench along a task-conditioned trajectory" using torque, slip and alignment penalties. SDG-Net scores grasps in real time. Results across hammering, sweeping, knocking and reaching: induced torque reduced up to 17.6%, and real-world success improved 17.5%. — [arXiv 2505.01399](https://arxiv.org/abs/2505.01399)
- **Grasp to Act: Dexterous Grasping for Tool Use in Dynamic Settings** (arXiv:2602.20466, 2026). Combines physics-based grasp optimisation with RL grasp adaptation on a 16-DoF hand. "an adaptive controller that residually issues joint corrections to prevent in-hand slip while tracking the object trajectory." It handles "impacts, torques, and continuous resistance" across hammering, sawing, cutting, stirring and scooping. RL adaptation helps most for repeated-impact tasks like hammering. The abstract does not say whether it uses tactile sensing or an explicit grip-force schedule. — [arXiv 2602.20466](https://arxiv.org/abs/2602.20466)
- **IMPACT: Learning Internal-Model Predictive Control for Forceful Robotic Manipulation** (arXiv:2606.10818, 2026). Learns an internal model that predicts interaction forces from state and action history, then applies the predicted wrench as feedforward. It concerns arm/end-effector force, not grip force. — [arXiv 2606.10818](https://arxiv.org/html/2606.10818)
- **An Optimal Control Formulation of Tool Affordance Applied to Impact Tasks** (arXiv:2402.05502). Treats hammering as an impact-aware task whose performance index is hammer-head speed at contact. It is about arm and tool trajectory, not grip. — [arXiv 2402.05502](https://arxiv.org/pdf/2402.05502)
- **Tracing Energy Flow: Learning Tactile-based Grasping Force Control to Reduce Slippage in Dynamic Object Interaction** (arXiv:2512.21043). Model-based learning of grasp-force control from tactile "energy" inconsistency. From the abstract it is reactive/model-predictive, not anticipatory of impulsive loads. — [arXiv 2512.21043](https://arxiv.org/pdf/2512.21043)

### Inferences
- The novelty claim is defensible as "the first robot grip controller that schedules a human-like anticipatory grip ramp (timing and magnitude) ahead of a predicted impulsive tool load". The paper must cite and contrast:
  - iTuP: grasp placement, not force timing
  - Grasp to Act: learned residual joint corrections, with no explicit pre-impact ramp reported
  - IMPACT: feedforward wrench, not grip
- "Tracing Energy Flow" shows the field is pushing reactive tactile grip control into dynamic interactions. Its reaction speed will not match a ~4 ms blow, which strengthens the planned paper's argument.

### Gaps
- My targeted searches did not cover older robotic catching/throwing work with pre-shaped grip force (e.g., high-speed catching from Ishikawa lab) or prosthetics/exoskeleton work on "anticipatory grip". The latter may contain human-inspired feedforward grip and should be checked.
- I could not confirm whether Grasp to Act uses tactile feedback or logs grip force per strike (I read only the abstract).

## Q3. Tactile slip detection/prediction 2022-2026: any evaluation under impact or vibration?

### Takeaway
State-of-the-art slip detectors report latencies of ~20-25 ms and FPR <~2%. Their negative classes cover gripper actuation, arm motion, grasp changes and environment sliding. **None that I found is evaluated with tool-impact transients (hammer strikes) as a negative class.** SlipSense explicitly says gripper actuation and grasp variation "generate strong vibration responses", which is the same confound the planned paper targets. Review literature notes that frequency-domain slip detectors misfire on "actuator motion and impacts" but gives no impact-specific FPR. This is a genuine gap.

### Cited Findings
- **SlipSense** (arXiv:2609.15910, submitted 2026-09-14; Jian, Senthil Kumar, Li, Chen, Dai, Sengul, Grimaldi, Lu, Nabi, Yu). It **exists**.
  - Sensor (TacV5): "32×32 piezoresistive array ... 240 Hz" (942 active taxels, 0.45×0.45 mm elements) plus a "3-axis MEMS accelerometer ... 8 kHz".
  - Results: "96.7% Macro F1 with a false-positive rate below 1.6%, detecting 76% of slip events within 23.1 ms." It transfers between a UMI parallel-jaw gripper and a Tesollo DG-5F hand.
  - Negative states include no-contact, no-slip, crosshead motion without contact, holding an unlinked object, and free grasping/pick-and-place. They note "grasp variation ... and gripper actuation ... also generate strong vibration responses".
  - No impact/tool-strike negatives were reported, and no per-class FPR.
  - Conflict: a PDF-extraction pass gave "16 taxels, 500 Hz pressure, 200 Hz accelerometer". The HTML full text quotes 32×32 @240 Hz and 8 kHz. I trust the HTML quotes.
  - [arXiv abs](https://arxiv.org/abs/2609.15910); [HTML](https://arxiv.org/html/2609.15910)
- **Evetac** (Funk, Helmut, Chalvatzaki, Calandra, Peters; arXiv:2312.01236; accepted IEEE T-RO). An event-camera optical tactile sensor processed at 1000 Hz, sensing vibrations up to 498 Hz, with learned slip detection/prediction and closed-loop grasp control. **Verified.** — [arXiv 2312.01236](https://arxiv.org/abs/2312.01236)
- **PapillArray incipient slip** (Wang, Martinez Ulloa, Burke, Cordova Bulens, Redmond; arXiv:2307.04011, July 2023). Learning-based incipient slip detection: 95.6% success in the lab and 96.8% in real gripping. **Verified.** — [arXiv 2307.04011](https://arxiv.org/abs/2307.04011)
- **Reactive Slip Control in Multifingered Grasping** (Ayral, Aloui, Grossard; arXiv:2602.16127, Feb 2026, revised Mar 2026). This is the "PzE hand" citation. It uses piezoelectric sensors for fast slip detection plus piezoresistive arrays for contact location, combining learned slip detection with model-based internal-force control. Slip is detected "approximately 20 ms after onset" and the full response takes ~30 ms, "matching human reflex speeds"; no friction model is needed. **Verified.** No impact evaluation is mentioned. — [arXiv 2602.16127](https://arxiv.org/abs/2602.16127)
- FORTE (arXiv:2506.18960) detects slip on fin-ray fingers within 100 ms. — [arXiv 2506.18960](https://arxiv.org/pdf/2506.18960)
- Action-conditioned tactile slip prediction (arXiv:2205.09430). — [arXiv 2205.09430](https://arxiv.org/html/2205.09430v2)
- Neuromorphic event-based slip detection and suppression (arXiv:2004.07386). — [arXiv 2004.07386](https://arxiv.org/pdf/2004.07386)
- Several 2025 Frontiers papers (search-snippet level):
  - Frequency-domain slip detection "may produce incorrect detections when the measurement is affected by non-slip vibrations, such as actuator motion and impacts."
  - Environment sliding is a known confound for vibration-based detectors.
  - Accelerometer plus gyroscope fusion is proposed to reduce false positives.
  - [Frontiers Robot AI 2025](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1698591/full); [Frontiers Neurorobot 2025, "Universal slip detection"](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1478758/full); [PMC12756126, inertial slip detection for compliant hands](https://pmc.ncbi.nlm.nih.gov/articles/PMC12756126/). I did not establish which statement comes from which paper; check before quoting.

### Inferences
- Reporting slip-detector FPR under hammer impacts would be a clear, novel evaluation, especially with a per-event breakdown (impact-only vs. impact plus slip). So would showing that patch-only drop detection false-triggers in a wrap grasp.
- SlipSense reaches 8 kHz accelerometer plus pressure-array fusion with <1.6% FPR. The planned sensor stack (8×8 @1 kHz plus 8 kHz accelerometer) is therefore close to SlipSense's. The novelty must come from the impact regime and the across-strike use, not from the sensing modality.
- Ayral 2026 (~30 ms reflex) and SlipSense (23 ms) set the "fast reflex" bar. Both are still about 5x slower than a ~4 ms blow, which supports the "adjust the next strike" design.

### Gaps
- I found no paper reporting slip-detector false positives specifically under tool impacts or hammering. This rests on a limited search and should be stated as "to our knowledge".
- I did not check BioTac vibration slip work (Su et al. 2015; Veiga et al.) or GelSight slip (Dong et al.) for impact tests.

## Q4. Trial-to-trial / iterative adaptation of grip margin; regrasp between repetitions

### Takeaway
Within-grasp adaptation is common: online friction estimation and adaptive grasp control (2026), experience-based grasp adaptation (EPFL), and online RL across repeated forces (Grasp to Act). I found **no explicit iterative-learning-control or Bayesian across-trial update of a grip safety margin driven by per-strike tactile/vibration evidence**. Human literature describes exactly this internal-model updating (Danion group). The across-strike margin update is therefore a plausible novelty, but it is less clear-cut than the anticipatory ramp.

### Cited Findings
- "Synchronized Online Friction Estimation and Adaptive Grasp Control for Robust Gentle Grasp" (arXiv:2602.02026). Real-time friction estimation plus closed-loop force modulation within a grasp. — [arXiv 2602.02026](https://arxiv.org/pdf/2602.02026)
- "Learning of Grasp Adaptation through Experience and Tactile Sensing" (EPFL, Li/Billard group). Object-level impedance control plus a grasp-stability estimator that triggers adaptation when instability is predicted. — [EPFL Infoscience](https://infoscience.epfl.ch/bitstreams/9d2ba8b3-cd31-4678-9fc8-7a34e7c7ff98/download)
- Grasp to Act: online RL adaptation under repeated task forces reduces cumulative pose error and slip rotation, with the most benefit in repeated-impact hammering. This is the closest "across-repetition" precedent. — [arXiv 2602.20466](https://arxiv.org/html/2602.20466)
- Learning Gentle Grasping from Human-Free Force Control Demonstration (arXiv:2409.10371). — [arXiv 2409.10371](https://arxiv.org/pdf/2409.10371)
- Human: the internal model behind feedforward grip is "updated during repeated interactions". — [J Neurophysiol, Danion et al.](https://journals.physiology.org/doi/full/10.1152/jn.00616.2020)

### Inferences
- Grasp to Act is the main competitor on "adapting over repeated strikes". The planned paper should contrast:
  - it uses an explicit, interpretable per-strike grip-margin update from tactile/vibration evidence (e.g., ILC or Bayesian friction posterior), where Grasp to Act uses a black-box RL residual on joint positions;
  - it adds explicit regrasp decisions between strikes.

### Gaps
- Not searched: ILC applied to grip force, and Bayesian friction-coefficient posteriors updated across trials. Absence is suggestive but not established.
- I found no paper doing regrasp planning *between* repetitions of an impact task.

## Q5. High-rate tactile hardware and fast-force-loop grippers

### Takeaway
Evetac (1 kHz) and SlipSense TacV5 (240 Hz array plus 8 kHz MEMS accelerometer) document high-rate tactile sensing, and ManiWAV documents contact microphones. The Franka Hand is **not** a fast force-loop gripper. Its grasp command is a high-level force-setpoint call outside the 1 kHz FCI arm loop, and a user report shows the force is poorly tracked. This supports the paper's 50-185 ms actuation-latency premise.

### Cited Findings
- Evetac: 1000 Hz processing, vibrations up to 498 Hz. — [arXiv 2312.01236](https://arxiv.org/abs/2312.01236)
- SlipSense TacV5: 32×32 piezoresistive array @240 Hz plus 3-axis MEMS accelerometer @8 kHz. — [arXiv HTML 2609.15910](https://arxiv.org/html/2609.15910)
- ManiWAV (Liu, Chi, Cousineau, Kuppuswamy, Burchfiel, Song; CoRL 2024; arXiv:2406.19464). An "ear-in-hand" contact-microphone gripper for in-the-wild audio-visual demonstrations, covering 4 contact-rich tasks. **Verified.** — [arXiv 2406.19464](https://arxiv.org/abs/2406.19464)
- Ayral et al. 2026 combine piezoelectric (fast) and piezoresistive (spatial) sensing, with a ~30 ms slip-to-response loop. — [arXiv 2602.16127](https://arxiv.org/abs/2602.16127)
- Franka:
  - libfranka gives 1 kHz real-time control of the **arm** through FCI. — [libfranka GitHub](https://github.com/frankarobotics/libfranka)
  - The gripper exposes `grasp(width, speed, force, epsilon_inner, epsilon_outer)` as a separate command. — [libfranka Gripper API](https://frankarobotics.github.io/libfranka/0.15.0/classfranka_1_1Gripper.html)
  - A user requesting 25 N saw an external sensor read ~35 N peak and then ~5 N. The issue was closed "not planned" with no technical answer. — [libfranka issue #136](https://github.com/frankarobotics/libfranka/issues/136)

### Inferences
- The latency premise is sound, and SlipSense/Ayral reflex times show even fast reflexes are much longer than a 4 ms blow. Make it concrete by measuring the gripper's force step response on the paper's own hardware.

### Gaps
- I did not verify current specs for Xela uSkin, PapillArray (Contactile), or Robotiq/Schunk/Weiss fast-force grippers (e.g., Weiss WSG force-control rates). Check vendor datasheets.
- I found no published measurement of Franka Hand force-command latency. The 50-185 ms figure should come from the authors' own measurements.

## Citation verification summary
| Cited ID | Status | Actual content |
|---|---|---|
| Evetac arXiv:2312.01236 | Verified | Funk et al., event-based optical tactile sensor, 1 kHz, IEEE T-RO |
| PapillArray arXiv:2307.04011 | Verified | Wang et al. 2023, incipient slip detection, 95.6/96.8% |
| SlipSense arXiv:2609.15910 | Verified (exists, Sep 14 2026) | Jian et al., TacV5, 96.7% F1, FPR <1.6%, 23.1 ms |
| ManiWAV arXiv:2406.19464 | Verified | Liu et al., CoRL 2024, audio-visual robot learning |
| PzE hand arXiv:2602.16127 | Verified | Ayral, Aloui, Grossard, "Reactive Slip Control in Multifingered Grasping", piezoelectric plus piezoresistive, ~20 ms detection / ~30 ms response |
| White et al. 2011 Neuroscience | Verified (189:269-276) | Tapping with hand-held object; grip peak ~65 ms after impact |
| Delevoye-Turrell 2003 | Verified; venue is Q J Exp Psychol A 56A:1113-1128, not J Neurophysiol | |
