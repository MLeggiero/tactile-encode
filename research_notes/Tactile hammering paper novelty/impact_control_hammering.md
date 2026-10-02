# Robot hammering, nail driving and impact-aware manipulation control: prior-work map for novelty claims

Scope: closest prior work to a planned paper on a Franka FR3 at 1 kHz torque control swinging a **hand-held hammer** to drive a nail. The planned stack has a 1 kHz impedance layer with reference spreading, momentum-observer impact detection, velocity-feedback gating during ringing and passivity-constrained stiffness. Above it sit a 100-200 Hz reactive layer that outputs pose, stiffness, wrench feedforward and grip force, and a limit-aware swing planner that brakes at contact and allows arc (curved) strikes.
Research date: 2026-10-02. All arXiv IDs below were checked against the arXiv API on this date unless marked UNVERIFIED.

## Q1. Which robot papers/demos drive nails or hammer, and what did each measure or control?

### Takeaway
Many systems "hammer", but none combines (a) a real torque-controlled arm at kHz rates, (b) a tool held in a gripper that can slip, (c) measured or estimated impact force and grasp slip, and (d) closed-loop impact or grip control. The learning papers (Grasp-to-Act, SimToolReal, Adroit) control slip at 30-60 Hz with no impact model. The model-based paper (Ti et al.) optimises only the pre-strike posture, under joint-velocity control, and could not measure impact force. ARMADA reports strikes per nail but no grip or impact control. Demo or competition systems (Torobo, the Beijing games) are not peer-reviewed papers.

### Cited Findings
**Ti, Gao, Zhao, Calinon: "An Optimal Control Formulation of Tool Affordance Applied to Impact Tasks", arXiv 2402.05502 (Feb 2024; journal-length, 17 pp)**: VERIFIED
- Claim: iLQR + ADMM motion planning with a cost on *directional velocity manipulability*. The plan picks where to grasp the hammer handle (a "viapoint with a desired range") and reaches a pre-strike posture that maximises hammer-head speed along the nail axis. — [arXiv 2402.05502](https://arxiv.org/abs/2402.05502)
- Robot: real 7-DoF Franka Emika, run with a **joint velocity controller**. The control constraint is the Franka joint-velocity limit (no torque, torque-rate or impact model). — [arXiv HTML](https://arxiv.org/html/2402.05502)
- Grip: two-finger gripper. "To simulate the connection between the human hand and the hammer, we stuck two pieces of **sponge** on the two grippers". The hammer is a 3D-printed ABS hammer with a cylindrical handle. — [arXiv HTML](https://arxiv.org/html/2402.05502)
- Target: a nail driven into **plastic foam** with a **pilot hole**. The foam was chosen to "mitigate the impulse of the hammering impact on the type of robot we used". — [arXiv HTML](https://arxiv.org/html/2402.05502)
- Impact force: "Due to experimental equipment constraints, we could not record impact forces accurately… as the connection between the grippers and the tool remains soft, it is hard to fully transmit the impact force to the robot's sensors." Instead they fit a depth-to-peak-force model (15 samples per hole tightness) and inferred force from nail insertion depth. They state that a force sensor on the hammer head would "increas[e] the grip instability". — [arXiv HTML](https://arxiv.org/html/2402.05502)
- Model: impulse-momentum with an assumed post-impact hammer velocity of 0, so pre-impact speed is "the key factor". There is no grasp-slip measurement, no impact detection and no post-impact control. — [arXiv HTML](https://arxiv.org/html/2402.05502)
- The related work cites an RRT hammering planner with joint-acceleration limits and non-zero start/goal velocities (their ref. [34]). The authors were not resolved here. — [arXiv HTML](https://arxiv.org/html/2402.05502)

**ARMADA: Kim, Kim, Lee, Jang, B. Kim, "A low-cost and lightweight 6 DoF bimanual arm for dynamic and contact-rich manipulation", arXiv 2502.16908 (Feb 2025)**: VERIFIED (the arXiv v2 title starts "Design of…" on ar5iv)
- Hardware paper: custom 6-DoF quasi-direct-drive (QDD) back-drivable arm built for $6,100 (both arms), with peak end-effector speed 6.16 m/s (std 0.472) and 2.5 kg payload. The claim is that back-drivability lets the joints "absorb impact forces through natural compliance". — [arXiv 2502.16908](https://arxiv.org/abs/2502.16908)
- Gripper: custom 1-DoF jaw gripper (Dynamixel XM430) with flexible **TPU** pads. — [arXiv HTML](https://arxiv.org/html/2502.16908)
- Hammering: the nail is pre-set at 20 mm in a wooden board. ARMADA drove it 20 mm in **10 ± 1.26 strikes**, against 13.15 ± 6.41 for 14 humans using the same hammer. A human straightens the nail if it bends. — [arXiv HTML](https://arxiv.org/html/2502.16908)
- Impact force: measured only as a safety test, by striking a push-pull gauge near maximum speed (mean 50.5 N, max 52.2 N). It was not measured during nail driving. — [arXiv HTML](https://arxiv.org/html/2502.16908)
- The swing uses pre-computed trajectories. Controller rate: 200 Hz, with a 20 Hz policy for the RL tasks. There is no impact detection, grip control or slip measurement. The authors note that letting objects slip in the gripper "renders the grasp less robust". — [arXiv HTML](https://arxiv.org/html/2502.16908)

**Grasp-to-Act: Gupta, Mirzaee, Yuan, arXiv 2602.20466 (Feb 2026), IEEE RA-L 11(5):6288-6295, 2026**: VERIFIED
- Claim: physics-based grasp optimisation seeded by human demonstrations, plus an RL residual controller that issues joint corrections "to prevent in-hand slip". Tested on hammering, sawing, cutting, stirring and scooping. — [arXiv 2602.20466](https://arxiv.org/abs/2602.20466)
- Robot: 16-DoF LEAP hand on a 6-DoF UR5 driven by IK, with wrist and finger updates at **30 Hz**. — [arXiv HTML](https://arxiv.org/html/2602.20466)
- Hammer task: steel claw hammer (30 cm, 520 g). The task is to drive a nail standing 2 cm above a drywall stack in **4 strikes**, scored as the fraction of the nail embedded. In simulation the nail is modelled as immovable. — [arXiv HTML](https://arxiv.org/html/2602.20466)
- Measures in-hand translational and rotational slip (sim hammer, G2A: E_t 0.69 cm, E_θ 1.35°; without adaptation: 7.33°). Notes that "under repeated application of task-specific forces, particularly in hammering, cumulative pose errors accrue". — [arXiv HTML](https://arxiv.org/html/2602.20466)
- Impact force, strike speed and an impact model were not reported in the text checked. — [arXiv HTML](https://arxiv.org/html/2602.20466)

**SimToolReal: Kedia, Lum, Bohg, C. K. Liu, arXiv 2602.16863 (Feb 2026), which also introduces DexToolBench**: VERIFIED
- A single sim-to-real RL policy that drives procedurally generated tool primitives to goal poses. Reports 120 real rollouts over 24 tasks, 12 objects and 6 tool categories. — [arXiv 2602.16863](https://arxiv.org/abs/2602.16863)
- Robot: 22-DoF Sharpa five-fingered hand on a 7-DoF KUKA iiwa 14. Sim/control rate 120/60 Hz, object pose estimated at 30 Hz, goal trajectory from human video downsampled to 3 Hz. — [arXiv HTML](https://arxiv.org/html/2602.16863)
- DexToolBench hammer tasks: "Grasp the hammer…, rotate it by 90° into a striking configuration, swing down [or sideways] onto a nail 3 times". Tools: 3D-printed claw hammer (36 g) and rubber mallet (331 g). The heavier tool performed worse, and drops were most common on heavy objects. — [arXiv HTML](https://arxiv.org/html/2602.16863)
- The paper motivates dexterous hands by noting that parallel-jaw grasps rely "primarily on friction and grip force" against impact torques. Success is a pose-tracking "task progress" score. No nail-depth, impact-force or strike-speed metric was found. — [arXiv HTML](https://arxiv.org/html/2602.16863)

**Adroit Hammer: Rajeswaran et al., RSS 2018 (DAPG), arXiv 1709.10087**
- Simulated 24-DoF ShadowHand on a 4-DoF arm in MuJoCo. The task is to pick up a hammer and drive a nail whose dry friction absorbs up to 15 N. Success means the entire nail is inside the board. Simulation only. — [Gymnasium-Robotics docs](https://robotics.farama.org/envs/adroit_hand/adroit_hammer/); [arXiv 1709.10087](https://arxiv.org/abs/1709.10087)

**Torobo (Tokyo Robotics) demo video, ~2024, not a paper**
- Humanoid with 7-axis dual arms and joint torque sensors, shown hammering a nail into wood. The company says hammer rebound is absorbed by "the elasticity of the rubber material securing the hammer, the deflection in torque sensors and harmonic gears, backdrivability, and impedance control". No quantitative data was published. — [Interesting Engineering](https://interestingengineering.com/innovation/torobo-humanoid-robot-hammers-nail); [GlobalSpec](https://insights.globalspec.com/article/22764/this-humanoid-robot-can-hammer-nails)

**World Humanoid Robot Games, Beijing (Aug 2025 first edition; 2026 second edition)**
- 2025: robots were given hammers to drive nails into a corkboard. One report says even teleoperated (motion-capture glove) robots failed. — [The Ticker](https://theticker.org/19624/science/robots-from-beijing-humanoid-games-can-beat-records-but-cant-hammer-nails/); [Wikipedia](https://en.wikipedia.org/wiki/World_Humanoid_Robot_Games)
- 2026 (2nd edition, 666 teams, 2,056 robots): rules are to place the nail with the left hand and hammer with the right. A nail counts only if fully driven with no protruding head and no obvious bending, and the score is valid nails in 5 minutes. A referee says many robots drove nails at angles. No numeric results were given. — [Bastille Post](https://www.bastillepost.com/global/article/6104693-robots-compete-to-hammer-nails-with-precision-at-humanoid-robot-games-in-beijing)
- A Nature news piece ("'Robot Olympics' reveal humanoids' rapid progress — and flaws") exists but was paywalled and not read. — [Nature](https://www.nature.com/articles/d41586-026-02713-z)

**Older hammering hardware and control work (cited by ARMADA and Ti)**
- Garabini et al., "Optimality principles in variable stiffness control: The VSA hammer" (IROS 2011). Optimal stiffness profile to maximise link velocity at a given final position; about 30% higher hammer velocity. — [IEEE Xplore](https://ieeexplore.ieee.org/document/6094870/); [ResearchGate](https://www.researchgate.net/publication/221065759_Optimality_principles_in_variable_stiffness_control_The_VSA_hammer)
- Romanyuk et al., multiple-working-mode hammering with a modular reconfigurable robot (and a Robotica follow-up). Joints switch to a passive mode during impact to keep the harmonic-drive impulse below a threshold, rather than to maximise hammering performance. — [Robotica](https://www.cambridge.org/core/journals/robotica/article/abs/multiple-working-mode-approach-to-robotic-hammering-analysis-and-experiments/391360FB28559678E395DF5428D43314); [ResearchGate](https://www.researchgate.net/publication/335500693_A_Multiple_Working_Mode_Approach_to_Hammering_with_a_Modular_Reconfigurable_Robot)
- ARMADA's reference list also includes "Impulse modeling and analysis of dual arm hammering task: Human-like manipulator" and "Control of a hitting velocity and direction for a hammering robot using a flexible link". These are listed by ARMADA but were not individually checked. — [arXiv HTML 2502.16908](https://arxiv.org/html/2502.16908)
- Kirner & Ott, "The Nonsmooth Impact Direction (NSID) of Robotic Systems", arXiv 2607.13768 (Jul 2026, under review). Theory of pre-/post-impact velocity directions with experiments (OptiTrack 1 kHz, F/T sensor 2 kHz). It names "repetitive hammering" as an application and argues that momentum transfer strictly along the contact normal "prevents the nail from bending". There is no actual nail driving or held tool. — [arXiv 2607.13768](https://arxiv.org/abs/2607.13768)

### Inferences
- Under a quick rubric (real hardware + held tool + nail-depth metric + impact force + grip/slip control + kHz impact-aware control), no single prior paper scores more than 2-3 out of 6. Ti et al. is closest on platform (also a Franka) and on its compliant (sponge) grip, but it is a velocity-controlled pre-strike planner with no impact handling. Grasp-to-Act is closest on slip control, but at 30 Hz with a dexterous hand and a UR5.
- Ti et al.'s own statement that impact force could not be measured through a soft grip, and that a head-mounted sensor destabilises the grip, motivates an in-grasp or proprioceptive impact estimate.
- Strike speed is reported quantitatively only by ARMADA (6.1 m/s for batting, not hammering). Strike speed during nail driving appears unreported in all the learning papers.

### Gaps
- "Zhang/Fazeli impact work": no hammering or impact paper with these authors was found on arXiv or the web. UNVERIFIED; ask the requester for the exact title.
- The Nature 2026 article was not readable (paywall). No peer-reviewed data on any WHRG hammering team was found.
- No paper was found reporting nail depth per strike together with measured strike speed on a torque-controlled arm.
- Torobo control details (rates, strategy) are unpublished.

## Q2. State of the art in reference spreading, impact-aware QP control, momentum-observer detection and post-impact ringing: has any of it been applied to a hand-held tool, or combined with grip-force control?

### Takeaway
Reference spreading (RS) has matured from simulation (2021-22) to real dual-arm and hit-and-push experiments on Franka arms (T-RO 2024; 600 experiments in 2411.09870). In every case, though, the impact is made by the robot's own end effector or pads on a box or surface. No paper applies RS, impact-aware QP, or momentum-observer impact gating to a tool held in a gripper, and none couples impact handling with grip-force control.

### Cited Findings
- **van Steen, van den Brandt, van de Wouw, Kober, Saccon, arXiv 2305.08643**, accepted to IEEE T-RO in June 2024: VERIFIED. QP-based RS for dual-arm manipulation with planned *simultaneous* impacts. Overlapping ante-/post-impact references consistent with impact dynamics are built from teleoperation data. Adds a novel **interim mode** for when unplanned sequential impacts occur. The authors call it the "first time an experimental evaluation of reference spreading control on a robotic setup", compared against three baselines. — [arXiv 2305.08643](https://arxiv.org/abs/2305.08643)
- **van Steen, Stokbroekx, van de Wouw, Saccon, arXiv 2411.06319**, submitted to T-RO: VERIFIED. "Impact-Aware Robotic Manipulation: Quantifying the Sim-To-Real Gap for Velocity Jumps". Generates rigid impact maps with a physics engine and validates them experimentally using RS: the correct map minimises net feedback. Reports 3.1% average error in post-impact velocity. It explicitly notes that feedback during impact-induced vibrations makes velocity-only evaluation unreliable. — [arXiv 2411.06319](https://arxiv.org/abs/2411.06319)
- **van Steen, van de Wouw, Saccon, arXiv 2411.09870**, submitted to T-RO: "Impact-Aware Control using Time-Invariant Reference Spreading". Uses vector-field references and a physics-engine impact model. Its interim mode uses a position feedback signal derived from the ante-impact velocity reference to promote contact completion. Validated on 600 robotic hit-and-push and dual-arm grabbing experiments. — [arXiv 2411.09870](https://arxiv.org/abs/2411.09870)
- Earlier RS work: QP-based RS in simulation (arXiv 2111.05211); time-invariant RS (arXiv 2206.04852); dual-arm impact-aware grasping (IFAC 2023, arXiv 2212.00877, simulation only, interim phase "without using velocity error feedback"). — [arXiv 2111.05211](https://arxiv.org/abs/2111.05211); [arXiv 2206.04852](https://arxiv.org/abs/2206.04852); [arXiv 2212.00877](https://arxiv.org/abs/2212.00877)
- RS has been carried over to bipedal locomotion with hybrid Lyapunov feedback (Bertollo et al., arXiv 2405.02184). — [arXiv 2405.02184](https://arxiv.org/abs/2405.02184)
- **Aouaj, Padois, Saccon, arXiv 2010.08220** (ICRA 2021, Xi'an): VERIFIED. "Predicting the Post-Impact Velocity of a Robotic Arm via Rigid Multibody Models: an Experimental Study". A 7-DoF torque-controlled robot intentionally impacts a rigid surface. They compare a frictionless inelastic rigid impact map with the recorded *damped oscillatory* post-impact response. Data from 18 experiments are released. — [arXiv 2010.08220](https://arxiv.org/abs/2010.08220)
- **Wang, Dehio, Tanguy, Kheddar, "Impact-aware task-space quadratic-programming control"**, IJRR 2023 (arXiv 2006.01987; RSS 2019 precursor "Impact-friendly robust control design with task-space quadratic optimization"). The QP gets constraints for hardware-affordable impact bounds and post-impact feasible sets. A one-step preview assumes "the impact will occur at the next iteration", which makes it robust to impact time and location. Tested on a Panda (moderate impacts) and HRP-4 (swift grabbing). — [arXiv 2006.01987](https://arxiv.org/abs/2006.01987); [IJRR DOI](https://dl.acm.org/doi/abs/10.1177/02783649231198558); [RSS 2019](https://www.roboticsproceedings.org/rss15/p32.pdf)
- Wang, Dehio, Kheddar: predicting impact-induced joint-velocity jumps (250 Panda experiments, arXiv 2202.12646) and an inverse-inertia/contact-force model for normal impacts (arXiv 2109.04756). Wang, Tanguy, Kheddar: impact-aware multi-contact balance on HRP-4 hitting a wall (arXiv 2308.06784). — [arXiv 2202.12646](https://arxiv.org/abs/2202.12646); [arXiv 2109.04756](https://arxiv.org/abs/2109.04756); [arXiv 2308.06784](https://arxiv.org/abs/2308.06784)
- **Impact-invariant control** (Yang & Posa, arXiv 2103.06907; 2303.00817 "Maximizing Control Authority During Impacts"). Projects out the velocity-feedback components that are uncertain during impacts, for legged robots. This is the closest published analogue to gating joint-velocity feedback during post-impact transients. — [arXiv 2303.00817](https://arxiv.org/abs/2303.00817); [arXiv 2103.06907](https://arxiv.org/abs/2103.06907)
- **Momentum observer**: Haddadin, De Luca, Albu-Schäffer, "Robot Collisions: A Survey on Detection, Isolation, and Identification", IEEE T-RO 33(6):1292-1312, 2017. This is the standard reference for proprioceptive generalised-momentum residuals for collision detection, including a flexible-joint variant. It targets *unintended* collisions and safety reactions, not intentional repeated tool impacts. — [PDF](http://www.diag.uniroma1.it/~labrob/pub/papers/TRO_Collision_Dec2017.pdf)
- Impact-aware catching (Yan, Stouraitis, Moura, Xu, Gienger, Vijayakumar, arXiv 2403.17249) jointly optimises end-effector motion, **stiffness** and contact force through multi-mode trajectory optimisation (MMTO). The impact is between the robot and a free object, not a held tool. — [arXiv 2403.17249](https://arxiv.org/abs/2403.17249)
- Kirner & Ott, "Impact Analysis for the Planning of Targeted Non-Slippage Impacts of Robot Manipulators", RA-L 9(3):2750-2757, 2024. Stamping-like impacts where the end effector must not slip off the target. — [TU Wien repositUm](https://repositum.tuwien.at/handle/20.500.12708/194644); [DLR PDF](https://elib.dlr.de/208498/1/Impact_Analysis_for_the_Planning_of_Targeted_Non-Slippage_Impacts_of_Robot_Manipulators.pdf)

### Inferences
- The planned stack's 1 kHz RS with an interim mode is an *application and adaptation* of van Steen et al. (T-RO 2024 / 2411.09870), not a new concept, and should be cited that way. The novelty must come from the **tool-in-grasp** setting. The impact is then transmitted through a compliant, slip-prone grasp, so the rigid impact map is ill-defined: the hammer–gripper interface decouples, and Ti et al. explicitly report poor force transmission. Repeated impacts, coupling with grip force, and the 100-200 Hz reactive layer that sets stiffness, feedforward and grip are also new.
- Gating velocity feedback for about 50 ms is conceptually related to RS's interim mode ("without using velocity error feedback", 2212.00877) and to impact-invariant projection (Yang & Posa). A reviewer will expect these comparisons.
- Passivity-constrained stiffness variation is standard in variable-impedance control. Its use across impact transitions with a held tool was not found in the literature searched.

### Gaps
- No paper found applying RS, impact-aware QP, or momentum-observer detection where the impacting body is a tool held in a gripper.
- No paper found coupling any impact-aware controller with grip-force modulation. Grasp-to-Act adapts finger joints with RL at 30 Hz with no impact model.
- "Jongeneel et al." (presumably Maarten Jongeneel, TU/e, impact-aware tossing/TOSS) could not be located through the arXiv author search, which returned only Wouter Jongeneel. UNVERIFIED; treat as related I.AM.-project tossing work and confirm separately.
- Whether 2411.06319 and 2411.09870 have since been published in T-RO could not be confirmed.

## Q3. Intentional or repeated impacts with a held tool (nailing, chopping, tapping, whipping) and adjacent racket sports

### Takeaway
Intentional-impact research overwhelmingly uses the robot's own links or pads (pushing, grabbing, hitting walls, stamping, kicking). Held-tool impact work is limited to the hammering papers in Q1 and to racket sports. In racket sports the racket is rigidly mounted or treated as part of the end effector, with a single high-speed impact per stroke. Repeated-impact control with a compliant grasp was not found.

### Cited Findings
- Table tennis: DeepMind's "Achieving Human Level Competitive Robot Table Tennis" (D'Ambrosio et al., arXiv 2408.03906) uses a learned agent with the paddle as the end effector. — [arXiv 2408.03906](https://arxiv.org/abs/2408.03906)
- Gossard et al., racket-ball bounce dynamics across rubbers (arXiv 2604.11349) and event-time hybrid optimal control for serves (arXiv 2608.08157). These model the impact timing and event for a racket-held strike. — [arXiv 2604.11349](https://arxiv.org/abs/2604.11349); [arXiv 2608.08157](https://arxiv.org/abs/2608.08157)
- ARMADA batting: a ping-pong ball hit with the dorsum of the end effector at an average 6.135 m/s, using a pre-computed trajectory. — [arXiv HTML 2502.16908](https://arxiv.org/html/2502.16908)
- Grasp-to-Act groups hammering (repeated impacts) with sawing and cutting (sustained resistance) and finds the largest benefit from online adaptation in those tasks. — [arXiv HTML 2602.20466](https://arxiv.org/html/2602.20466)
- Kirner & Ott (NSID) name repetitive hammering as a target application and note that NSID helps "align pre- and post-impact velocities for repetitive tasks". This is theory with no held-tool experiment. — [arXiv 2607.13768](https://arxiv.org/abs/2607.13768)

### Inferences
- Racket sports should be framed as adjacent but distinct: a rigid racket attachment, a moving target, and no requirement to keep a friction grasp through repeated impacts into a near-rigid target.

### Gaps
- No robotics papers were found on chopping, tapping or whipping with a held tool under impact-aware control. Coverage was limited to a few targeted arXiv searches, so a dedicated search on these terms is still needed.

## Q4. Swing trajectory planning under actuator limits for maximum impact

### Takeaway
Maximum-velocity and impact motion planning exists for elastic and variable-stiffness joints (Haddadin, Garabini), for pre-strike posture (Ti et al., directional manipulability) and for impact-direction choice (Kirner & Ott NSID). None of the papers found jointly enforces joint-velocity, torque *and torque-rate* limits for a held hammer, plans arrival with braking at contact, or compares curved (arc) with straight strike paths.

### Cited Findings
- Haddadin et al., "Kick it with elasticity: Safety and performance in human-robot soccer", Robotics and Autonomous Systems 2009. Exploits joint elasticity to increase kick speed. — [search summary referencing the paper](https://arxiv.org/pdf/2212.14741)
- Haddadin, Weis, Wolf, Albu-Schäffer, "Optimal control for maximizing link velocity of robotic variable stiffness joints", IFAC World Congress 2011. — [BSA paper reference list, arXiv 2212.14741](https://arxiv.org/pdf/2212.14741)
- Garabini et al., VSA hammer (IROS 2011) and "VSA kick". These choose inputs that maximise link velocity at a given final position and time; varying stiffness gave about 30% more hammer velocity. — [ResearchGate VSA hammer](https://www.researchgate.net/publication/221065759_Optimality_principles_in_variable_stiffness_control_The_VSA_hammer); [ResearchGate VSA kick](https://www.researchgate.net/publication/254041302_Optimality_principles_in_stiffness_control_The_VSA_kick)
- Ti et al. optimise the pre-hammering posture for directional velocity manipulability under joint-velocity limits only, via ADMM-iLQR. — [arXiv 2402.05502](https://arxiv.org/abs/2402.05502)
- Kirner & Ott (RA-L 2024; arXiv 2607.13768): the approach direction relative to the NSID sets the impulsive-force direction. It is a planning criterion for impact direction, not swing speed. — [arXiv 2607.13768](https://arxiv.org/abs/2607.13768)
- "Maximum Impulse Approach to Soccer Kicking for Humanoid Robots" (arXiv 2412.01480) is an adjacent maximum-impulse formulation for legged kicking. Its abstract was not reviewed in detail. — [arXiv 2412.01480](https://arxiv.org/pdf/2412.01480)

### Inferences
- A planner that caps strike speed and acceleration from the Franka FR3's joint velocity, torque and torque-rate limits, brakes at contact (so post-impact rebound and ringing stay within limits), and chooses arc versus straight paths appears new for held-tool hammering on rigid, torque-controlled arms. Prior maximum-velocity work relies on elastic or variable-stiffness actuators.

### Gaps
- The details of the hammering RRT planner with acceleration limits cited as [34] by Ti et al. were not retrieved.
- No paper comparing curved against straight strike trajectories for a robot hammer was found.

## Summary of verified citations (requested checks)
| arXiv ID | Exists | Actual title / claim |
|---|---|---|
| 2305.08643 | Yes | van Steen et al., QP-based RS for dual-arm planned simultaneous impacts with interim mode; first experimental RS; T-RO (accepted Jun 2024) |
| 2411.06319 | Yes | van Steen et al., sim-to-real gap for velocity jumps; RS used to identify impact map; 3.1% error; submitted to T-RO |
| 2010.08220 | Yes | Aouaj, Padois, Saccon, post-impact velocity prediction via rigid multibody models; ICRA 2021; 18 experiments |
| 2502.16908 | Yes | ARMADA (Kim et al., B. Kim lab): low-cost QDD bimanual arm; hammering 10 strikes/20 mm |
| 2402.05502 | Yes | Ti, Gao, Zhao, Calinon: optimal-control tool affordance for impact tasks; Franka; sponge-padded gripper; foam + pilot hole; impact force not measured |
| 2602.20466 | Yes | Gupta, Mirzaee, Yuan: Grasp-to-Act; LEAP hand + UR5; slip reduction in 5 tool tasks incl. hammering; RA-L 2026 |
| (extra) 2602.16863 | Yes | SimToolReal + DexToolBench (Kedia, Lum, Bohg, Liu); Sharpa hand + KUKA iiwa |
| (extra) 2411.09870 | Yes | Time-invariant RS, 600 experiments |
| (extra) 2607.13768 | Yes | Kirner & Ott NSID; mentions repetitive hammering |
| Zhang/Fazeli impact | NOT FOUND | — |
| Jongeneel et al. | NOT FOUND via arXiv author search | — |
