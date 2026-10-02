# Force-truth datasets/benchmarks for tool use, and the publication landscape for a tactile hammering-grip paper

Scope note: research done 2026-10-02 with about 22 search/fetch calls. Items marked UNVERIFIED come only from search snippets or memory and were not confirmed against a primary page.

## Q1. Existing datasets with force/tactile streams for manipulation or tool use (2022-2026): which have impact, slip, depth or outcome labels, or kHz rates?

### Takeaway
I found no public dataset of robot hammer strikes with per-strike impact force, nail-depth increment, in-grasp tool slip and kHz F/T, accelerometer or pressure-array streams. Current force datasets cover two other areas. Contact-rich assembly and teaching datasets (REASSEMBLE, RH20T, UMI-FT, ForceMimic) are quasi-static and have no impact labels. Impact datasets (TU/e I.AM.) cover box tossing and impacts on a plane, with no tools and no grasp slip. The closest work on robot hammering with labels predates 2022 (USC contact-localization dataset, around 2016) or uses mocap slip labels without force truth (Grasp to Act, 2026).

### Cited Findings
- **REASSEMBLE (RSS 2025, TU Wien):** built on NIST Assembly Task Board 1. It has 4 actions (pick, insert, remove, place) on 17 objects, with 4,551 demonstrations (4,035 successful) over 781 minutes. Modalities are event cameras, a wrist 6-axis F/T (AIDIN AFT200-D80-C), microphones and multi-view RGB. Its labels support temporal action segmentation, policy learning, anomaly detection and task inversion. It has no impact or slip labels. — [arXiv 2502.05086](https://arxiv.org/abs/2502.05086); [TU Wien news](https://www.tuwien.at/en/etit/ict/asl/news/reassemble-at-rss-2025); [project page](https://tuwien-asl.github.io/REASSEMBLE_page/)
- **RH20T:** 4 robots (Franka Panda, UR5, KUKA iiwa, Flexiv Rizon), 147 tasks, about 110K episodes, with visual, force, audio and action data. I could not verify the F/T sampling rate. — [ResearchGate entry](https://www.researchgate.net/publication/382981695_RH20T_A_Comprehensive_Robotic_Dataset_for_Learning_Diverse_Skills_in_One-Shot)
- **UMI-FT (arXiv 2601.09988, 2026):** a handheld UMI variant with a CoinFT 6-axis capacitive F/T sensor (about $10 BOM) on each finger, plus RGB, depth and pose. Tasks are whiteboard wiping, skewering zucchini and lightbulb insertion. The policy predicts position targets, grasp force and stiffness. There are no impact tasks. — [arXiv 2601.09988](https://arxiv.org/pdf/2601.09988)
- **TacUMI (arXiv 2601.14550):** a multimodal UMI for contact-rich tasks. I only saw the title, so details are UNVERIFIED. — [arXiv 2601.14550](https://arxiv.org/pdf/2601.14550)
- **ForceMimic / ForceCapture (ICRA 2025):** a handheld, robot-free rig that records interaction wrench using a ratchet lock and gravity compensation. The task is vegetable peeling, where it reports a +54.5% relative success gain over vision-only imitation learning. The dataset is on GitHub and Google Drive. — [arXiv 2410.07554](https://arxiv.org/abs/2410.07554); [GitHub](https://github.com/ForceMimic/)
- **Impact-Aware Robotics Database (TU/e, EU H2020 I.AM., grant 871899):** UR10 tossing and dropping boxes onto a conveyor, Franka Panda impacts on an "ImpactPlane", dual-arm grabbing, release experiments and a drone impact. Sensors include motion capture, joint encoders, cameras and pressure sensors. Sampling rates and recording counts are not listed on the summary page. It has no tools and no in-hand slip. — [Datasets page](https://impact-aware-robotics-database.tue.nl/datasets); [4TU I.AM. box archive](https://data.4tu.nl/articles/dataset/Impact_Aware_Manipulation_I_AM_archive_containing_box_impact_recordings/17122553/1); [4TU collection](https://data.4tu.nl/collections/c46f0d20-c62d-407b-8355-6737243d11c9/21)
- **van Steen et al. (submitted to T-RO; arXiv 2411.06319):** quantifies the sim-to-real gap for impact maps (velocity jumps), with a 3.1% average error in post-impact velocity between simulation and experiment, using reference-spreading control. The abstract does not name the simulator or say whether data was released. This is the closest prior work on the "sim faithful at impact" claim. — [arXiv 2411.06319](https://arxiv.org/abs/2411.06319)
- **Hammering with tactile contact localization (Molchanov, Kroemer et al., IROS 2016, pre-2022):** a tactile dataset with accurate labels of contact location during hammering-like tasks, at bicl.robotics.usc.edu. It is the only labeled robot-hammering tactile dataset I found. — [PDF](https://publications.ri.cmu.edu/storage/publications/2019/03/Kroemer_Molchanov_IROS_2016.pdf)
- **Measuring impact force in robot hammering is hard.** A search snippet (source paper not confirmed, probably the [optimal-control tool affordance paper, arXiv 2402.05502](https://arxiv.org/pdf/2402.05502)) says the soft gripper–tool connection keeps impact force from reaching the robot's sensors. The authors therefore modeled maximum impact force from nail insertion depth. This directly motivates a force-truth rig (UNVERIFIED attribution). — [arXiv 2402.05502](https://arxiv.org/pdf/2402.05502)
- A 2025 survey on forceful robot foundation models covers force datasets and is a useful related-work anchor. I did not read it in full. — [arXiv 2504.11827](https://arxiv.org/pdf/2504.11827)
- **Other tactile datasets found in passing:** TacCompress, a multi-point tactile compression benchmark on a dexterous hand ([arXiv 2505.16289](https://arxiv.org/pdf/2505.16289)), and HABIT, a 2026 manipulation dataset ([site](https://habit-dataset.github.io/)). Neither covers impact.

### Inferences
- The proposed dataset's unique combination is per-strike impact-force truth from a load cell under the nail or anvil, a depth increment, in-grasp slip and kHz multimodal streams. The novelty claim looks defensible but should be phrased as "to our knowledge". The datasets to position against are REASSEMBLE (multimodal, contact-rich, F/T), RH20T (scale, F/T), I.AM./TU/e (impacts) and UMI-FT/ForceMimic (grasp force).
- Most cited force datasets log F/T at teleop or policy rates. The sampling rates were not verified here and should be checked before claiming "first kHz".

### Gaps
- Not checked: ObjectFolder, Touch and Go, TacQuad, OSMO, EgoTactile, Open X-Embodiment force subsets, FurnitureBench, Factory/AutoMate, and NIST board datasets beyond REASSEMBLE. From memory, ObjectFolder and Touch and Go are visuo-tactile object or material datasets with no dynamics or impact, and DROID has no force. All of this is UNVERIFIED.
- I did not find a human hammering biomechanics dataset with grip-force traces. Search time ran out, and this needs a dedicated search (motor-control literature, for example grip-force/load-force coupling during hammering).
- Not verified: RH20T's F/T rate and whether I.AM. pressure sensors are arrays or point sensors.

## Q2. Benchmarks that score hammering, and their metrics

### Takeaway
Existing hammering benchmarks score binary or geometric task success (nail depth past a threshold, or tool-trajectory tracking). None scores impact force, grip force, slip or energy transfer, so a force- and slip-based metric would be new.

### Cited Findings
- **Adroit Hammer (Gymnasium-Robotics):** success means the whole nail is inside the board. The shaped reward has palm–hammer distance, hammer-head–nail distance, nail-to-board distance ×10, +2 for a lift above 0.04 m, and +25 or +75 when the nail is within 0.02 or 0.01 m of the board. — [Gymnasium docs](https://robotics.farama.org/envs/adroit_hand/adroit_hammer/)
- **Meta-World hammer:** success means the nail moves more than 0.09 m into the block (reported in search results; Meta-World+ standardizes the benchmark). — [Meta-World arXiv](https://arxiv.org/pdf/1910.10897); [Meta-World+ OpenReview](https://openreview.net/pdf?id=eYZ9ebLIXo)
- **DexToolBench (SimToolReal, arXiv 2602.16863, 2026):** 24 real tasks with digital-twin simulation, using 12 objects in 6 categories (hammer, marker, eraser, brush, spatula, screwdriver). The hammer tasks are grasp, 90° in-hand rotation and a swing down or to the side. The metric is "Task Progress", the percentage of demonstrated goal poses tracked from a human video. Nail driving and force are not scored. — [arXiv 2602.16863](https://arxiv.org/html/2602.16863v2); [GitHub](https://github.com/hgupt3/simtoolreal)
- **New MuJoCo dexterous benchmarks:** DexJoCo, task-oriented dexterous manipulation on MuJoCo ([arXiv 2605.16257](https://arxiv.org/html/2605.16257v1)), and DexVerse, multi-task and multi-embodiment ([alphaXiv 2607.08751](https://www.alphaxiv.org/overview/2607.08751)). I have not checked whether either includes hammering. These are the closest "MuJoCo testbed" competitors to contribution (2).

### Inferences
- ManiSkill and RLBench are, from memory, mostly success-rate based (UNVERIFIED for hammer-specific tasks). The testbed can position itself as the first to report physically grounded per-strike metrics (impact impulse, depth per strike, slip per strike, grip-force margin) at hardware-faithful rates.

### Gaps
- Not verified: ManiSkill or RLBench hammer tasks, and the "Beijing humanoid games" (World Humanoid Robot Games 2025) events and scoring.

## Q3. Recent comparable papers (2023-2026), their contributions and evaluation protocol

### Takeaway
The closest recent works use 10 real trials per task, simulation with 3 seeds, several learned or analytical baselines and one key ablation. Slip is measured with mocap, not tactile sensing, and impact force is not measured. Simulator and library papers (TacSL in T-RO) are accepted with GPU speed-ups and sim-to-real policy transfer as the evidence.

### Cited Findings
- **Grasp to Act (G2A), Gupta, Mirzaee, Yuan (UIUC), arXiv 2602.20466, Feb 2026:** the most direct competitor.
  - Setup: 16-DoF LEAP hand on a UR5. Five dynamic tool tasks, including hammering: a 2 cm nail into drywall with 4 strikes of a 520 g claw hammer.
  - Sensing: RGB-D and OptiTrack for ground-truth in-hand slip. There is no tactile or F/T sensing.
  - Metrics: grasp success, translational slip (cm), rotational slip (deg), task completion (%).
  - Trials and statistics: 10 real trials per task, 3 simulation seeds in Isaac Lab, real results shown as means and bar charts.
  - Baselines: analytical grasp optimization and RL variants (contact rewards, pre-grasp, eigengrasp). Ablation: without online adaptation.
  - Finding: the gains are largest in repeated-impact tasks such as hammering.
  - [arXiv 2602.20466](https://arxiv.org/html/2602.20466v1)
- **Physics-Constrained / Physics-Conditioned Grasp Planning for Dynamic Tool Use (iTuP / SDG-Net), arXiv 2505.01399:** grasp selection that minimizes predicted interaction wrench, with torque, slip and alignment penalties. Tasks are hammering, sweeping, knocking and reaching in simulation and on hardware. It reduces induced torque by up to 17.6% and improves real success by 17.5% over a compositional baseline. — [arXiv 2505.01399](https://arxiv.org/html/2505.01399v2)
- **Tracing Energy Flow: tactile grasp-force control to reduce slippage in dynamic object interaction, arXiv 2512.21043:** directly relevant to contribution (3). I could not parse the PDF, so details are unverified. — [arXiv 2512.21043](https://arxiv.org/pdf/2512.21043)
- **Reactive slip control in multifingered grasping (hybrid tactile sensing and internal-force optimization), arXiv 2602.16127.** — [arXiv](https://arxiv.org/pdf/2602.16127)
- **FORTE (arXiv 2506.18960):** tactile force and slip sensing on compliant fingers. — [arXiv](https://arxiv.org/pdf/2506.18960)
- **Universal slip detection on a robotic hand (Frontiers in Neurorobotics 2025):** a PID with dead zone for grasp-force control. On slip it raises grasp force from 100 to 700 mN and limits slip to less than 1 cm. — [Frontiers](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1478758/full)
- **Tool-as-Interface (arXiv 2504.04612):** learns policies from human tool use, including dynamic tool tasks. — [arXiv](https://arxiv.org/html/2504.04612v2)
- **TacSL (T-RO 2025, NVIDIA):** GPU visuotactile simulation in Isaac Gym, more than 200× faster than the prior state of the art. It provides normal and shear force fields, a learning toolkit and the AACD RL algorithm, and is validated by sim-to-real policy transfer. — [arXiv 2408.06506](https://arxiv.org/html/2408.06506v1); [ADS T-RO record](https://ui.adsabs.harvard.edu/abs/2025ITRob..41.2645A/abstract)
- **Other tactile sim-to-real work** (the kind of evidence reviewers expect for sensor-model realism): TacEx, GelSight in Isaac Sim ([arXiv 2411.04776](https://arxiv.org/pdf/2411.04776)); Tacmap ([arXiv 2602.21625](https://arxiv.org/html/2602.21625)); TactSpace ([arXiv 2606.18959](https://arxiv.org/html/2606.18959)); SimShear (CoRL 2025, [OpenReview](https://openreview.net/forum?id=MJmzIunyBy)); and "Closing the Reality Gap: zero-shot sim-to-real for dexterous force-based grasping" ([arXiv 2601.02778](https://arxiv.org/pdf/2601.02778)).
- **IJRR data papers:** short (about 5-6 pages) and must show the data's use with an existing public algorithm. They "should not present a new algorithm in addition to the dataset". — [IJRR submission guidelines](https://journals.sagepub.com/author-instructions/ijr); [Sage data paper guidance](https://uk.sagepub.com/sites/default/files/data_papers_submission_final.pdf)

### Inferences
- The bar to clear against G2A is at least 10 real trials per condition, reported slip, and the same task framing (a nail driven N cm in K strikes). Force-truth impact plus tactile slip would be a clear step beyond G2A's mocap-only measurement.
- The four planned contributions pull in different directions. The IJRR data-paper format excludes a new controller. A single CoRL or RSS paper cannot hold a dataset, a simulator, a control stack and findings at full depth. A likely split: (a) a dataset plus rig paper (RA-L, or an IJRR data paper using an existing baseline controller), and (b) a testbed plus control-stack plus findings paper (CoRL, RSS, ICRA, or T-RO for the long version).

### Gaps
- Not checked: NeurIPS Datasets & Benchmarks robotics papers, Science Robotics hammering or impact papers, and RA-L dataset-paper conventions.
- Most cited arXiv papers have unconfirmed venue status.

## Q4. Typical reviewer criticisms (sim-only, systems without learning, tactile)

### Takeaway
I could not get specific OpenReview review texts. Confirmed policy facts: CoRL 2026 publishes de-anonymized reviews of accepted papers. RSS 2027 explicitly accepts rigorously negative results and stresses hypothesis and methodology. The criticism list below is an inference from community norms and the evaluation protocols of comparable papers, not from quoted reviews.

### Cited Findings
- CoRL 2026: OpenReview, double-blind, at least 2 reviewers, 1-page rebuttal, 8-page main text (9 at camera-ready). "The accepted papers and reviews will be publicly accessible and de-anonymized after decisions are announced." — [CoRL 2026 author instructions](https://www.corl.org/contributions/instruction-for-authors)
- RSS 2027: "Submissions will not be rejected for yielding negative experimental results. Rigorously falsifying a hypothesis is considered a valuable scientific contribution." Stage 1 is judged on hypothesis and methodology. — [RSS 2027 CfP](https://roboticsconference.org/information/cfp/)
- A curated list of CoRL 2025 papers (200+) can be mined for reviews. — [GitHub corl-2025-papers](https://github.com/smallfryy/corl-2025-papers)
- "High-fidelity, fast tactile simulation is a long-standing obstacle." The sim-to-real realism of tactile sensors is the field's acknowledged weak point. — [SimShear, OpenReview](https://openreview.net/forum?id=MJmzIunyBy); [Tactile Robotics: An Outlook, arXiv 2508.11261](https://arxiv.org/pdf/2508.11261)

### Inferences (not sourced to actual reviews)
- **Sim-only:** reviewers will ask "does the 8 kHz hardware-limit fidelity matter?" The answer needs real-versus-sim validation of impact transients (peak force, impulse, rebound) using the force-truth rig, in the style of van Steen et al.'s 3.1% post-impact-velocity check.
- **Tactile sensor models:** reviewers will ask for calibration against real pressure arrays (noise, latency, saturation, spatial resolution) and an ablation over sensor-model fidelity.
- **Systems without learning:** reviewers will ask for baselines such as a fixed high grip force, reactive slip control only (no anticipation), a learned policy (for example RL or diffusion) and G2A-style grasp optimization. They will also ask for ablations of the anticipatory term, sensing rate (8 kHz versus 1 kHz versus 100 Hz) and each sensing modality.
- **Statistics:** expect at least 10 real trials per condition (the G2A precedent). Better practice is 20 or more, with confidence intervals and per-strike distributions, since one trial contains many strikes. Simulation should report 3 or more seeds and many strikes.
- **Dexterous wrap grasps under impact:** findings need several hands or objects (hammer masses, handle diameters) to count as general.

### Gaps
- I did not retrieve or quote specific OpenReview reviews of related CoRL papers (G2A, SimShear, TacSL-like work). This is the next step: open the forum pages of accepted CoRL 2024-2025 tactile and tool-use papers.

## Q5. Venue deadlines and cycles for late 2026-2027

### Takeaway
The ICRA 2027 direct deadline (Sept 16, 2026) and the Humanoids 2026 deadline (July 24, 2026) have passed. The next realistic targets are the RSS 2027 extended abstract (Dec 4, 2026), RA-L with the ICRA 2027 option (transfers by Dec 31, 2026), IROS 2027 (about March 2027) and CoRL 2027 (about late May or June 2027). The IROS and CoRL dates are estimates.

### Cited Findings
- **ICRA 2027:** Seoul, May 24-28, 2027. The paper deadline is Sept 16, 2026, 23:59 PST, and notification is Jan 31, 2027. "The deadline for all transfers of journal papers to ICRA 2027 is 31 December 2026." An ICRA tweet gave Sept 15, which conflicts with the official page's Sept 16; I treat the official page as authoritative. — [ICRA 2027 CfP](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/); [ICRA on X](https://x.com/ieee_ras_icra/status/2089365401201201516)
- **RSS 2027:** Athens, July 6-11, 2027. It uses a new two-stage process:
  - Dec 4, 2026: 6-page extended abstract (5 pages of content plus 1 page of references)
  - Feb 5, 2027: Stage-1 decision
  - Feb 12, 2027: rebuttal
  - Feb 26, 2027: Stage-2 decision, an invitation to carry out an "Extension Charter"
  - Apr 16, 2027: final paper (8 pages excluding references)
  - Apr 30, 2027: acceptance
  - May 14, 2027: camera-ready
  - [RSS CfP](https://roboticsconference.org/information/cfp/)
- **CoRL 2026:** Austin, Texas. The paper deadline was "Thursday (5/28)". An aggregator listed "Nov 10, 2026" as the CoRL 2026 deadline; this is probably wrong (likely the conference date) and should be treated as unreliable. — [CoRL instructions](https://www.corl.org/contributions/instruction-for-authors); [aggregator](https://www.underleaf.ai/conference-deadlines)
- **Humanoids 2026:** Santa Clara, Dec 6-9, 2026. The deadline was July 24, 2026 (updates allowed until July 27), with an 8-page limit including references. Notification is Oct 6, 2026. — [Humanoids 2026 CfP](https://2026.ieee-humanoids.org/call-for-papers/)
- **IROS 2027:** a call-for-papers listing exists on the IEEE-RAS site, but I did not retrieve its date. — [IEEE-RAS IROS 2027](https://www.ieee-ras.org/event/call-for-papers-paper-submission-deadline-iros-2027-ieee-rsj-international-conference-on-intelligent-robots-and-systems-iros-27403-0/)

### Inferences
- **CoRL 2027:** by pattern, the deadline is about late May or early June 2027 (CoRL 2026 was May 28). UNVERIFIED.
- **IROS 2027:** the deadline is about March 1, 2027 by pattern, with RA-L+IROS around mid-February. UNVERIFIED.
- **Humanoids 2027:** the deadline is about July 2027 by pattern. UNVERIFIED.
- **RA-L and T-RO:** both take submissions year-round. RA-L is 8 pages and usually has a 6-month decision cycle. The conference options are tied to ICRA and IROS (UNVERIFIED specifics).
- **RSS 2027 fit:** its two-stage, hypothesis-first format suits a "findings about wrap grasps under impact" framing, for example a hypothesis that anticipatory grip force reduces slip per strike without lowering depth per strike. A December 4 extended abstract can describe planned real-robot trials as part of the Extension Charter.

### Gaps
- Not retrieved: exact IROS 2027, CoRL 2027 and Humanoids 2027 dates, the current RA-L page limits and turnaround, or the NeurIPS 2027 Datasets & Benchmarks track deadline (about May by pattern, UNVERIFIED).
