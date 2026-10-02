# Dexterous / humanoid hands under impulsive tool loads, actuator properties, and impact simulation (novelty map)

Scope note: about 22 tool calls, done 2026-10-02. Primary sources were fetched for Grasp-to-Act, SimToolReal, the WUJI Hand 2 spec and docs pages, arXiv 2606.28805 and arXiv 2601.02778. The other entries rest on search-result snippets and are flagged where it matters.

## Q1. Dexterous-hand tool use with impacts or high external loads (2022-2026); did anyone measure slip under impact or actuator loads?

### Takeaway
The closest prior work is **Grasp-to-Act (G2A, arXiv 2602.20466)**. It measures in-hand slip during real hammering with a LEAP hand. It does not model impact forces explicitly, and it does not discuss actuator torque limits, hard-stop loads, or backdrivability. **SimToolReal / DexToolBench** includes hammer *motions* on a Sharpa hand, but it explicitly does not evaluate high-force functional contact. I found no paper that reports the *actuator or gearbox load relative to rating* during a hammer blow. That is the clearest gap the planned paper fills.

### Cited Findings
- G2A (Gupta, Mirzaee, Yuan) targets "dynamic forces such as impacts, torques, and continuous resistance". It combines physics-based grasp optimization with an RL residual controller that issues joint corrections to prevent in-hand slip. It transfers zero-shot to hardware on five tasks (hammering, sawing, cutting, stirring, scooping) — [arXiv abs](https://arxiv.org/abs/2602.20466)
- G2A setup: 16-DoF LEAP hand on a UR5, simulated in Isaac Lab, 30 Hz control. For hammering, "we instead model the nail as immovable and let contact dynamics generate task disturbances". Grasp candidates are stress-tested with wrenches up to a predefined F_max / τ_max. Real hammering slip was E_t = 0.69 cm and E_θ = 1.35°, with 100% completion over 10 trials (4 strikes each). The fetched text does not discuss actuator torque limits or backdrivability — [arXiv HTML](https://arxiv.org/html/2602.20466)
- SimToolReal (arXiv 2602.16863) uses a 22-DoF Sharpa hand on a KUKA iiwa 14. It reached 79.2% average task progress over 120 real rollouts on DexToolBench (24 tasks, 6 tool categories, hammer included). The authors state "We focus on trajectory following ... rather than evaluating functional task completion" and that it "does not guarantee functional task completion, especially for high-force interactions". It reports no force or impact measurements and no actuator limits. Failures were 43.7% pose-tracking loss and 34.5% drops — [arXiv HTML](https://arxiv.org/html/2602.16863); [project](https://simtoolreal.github.io/); [Sharpa blog](https://www.sharpa.com/blogs/research/moving-beyond-specialist-training-simtoolreal-enables-zero-shot-transfer-for-general-purpose-tool-use)
- The fetch summary said SimToolReal trained in "MuJoCo-based environments". **Unverified:** I could not confirm the simulator, and it may be Isaac Gym/Lab.
- DexScrew (arXiv 2512.02011, ICRA 2026; Hsieh, ..., Malik, Sreenath, Qi) does screwdriving and nut tightening. It uses a simplified simulator for motion primitives plus real-world tactile behavior cloning, and is robust to external perturbations. The task is quasi-static torque, not impulsive — [arXiv](https://arxiv.org/abs/2512.02011)
- An ICLR 2025 paper (SGFT) lists real-world "dynamic, contact-rich manipulation tasks including hammering". I did not verify the hand type or what was measured — [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/e68274fc4f158dbcbd4dddc672f7ee9c-Paper-Conference.pdf)
- Other 2026 dexterous tool or impact-adjacent work surfaced by search but not read: DexDrummer (in-hand, contact-rich drumming, [2603.22263](https://arxiv.org/pdf/2603.22263)) and Mana (articulated tools, [2606.13677](https://arxiv.org/pdf/2606.13677)). DexDrummer involves repeated impacts and **should be checked** as possibly close prior work.
- Benchmark lineage: Adroit Hammer (Rajeswaran et al. RSS 2018) uses a 24-DoF ShadowHand plus 4-DoF arm in MuJoCo. The nail has dry friction "capable of absorbing up to 15N", and actions are absolute joint positions. No actuator torque-rate limits and no grip-load reporting — [Gymnasium-Robotics docs](https://robotics.farama.org/envs/adroit_hand/adroit_hammer/); [RSS paper](https://www.roboticsproceedings.org/rss14/p49.pdf)
- World Humanoid Robot Games: the 2026 Beijing edition added eight dexterous-hand events, including "hammering a nail" and power-tool assembly. These sources are secondary and aggregator-level — [ainchina blog](https://www.ainchina.com/blog/world-humanoid-robot-games-beijing-2026/); [Wikipedia](https://en.wikipedia.org/wiki/World_Humanoid_Robot_Games). The 2025 edition had 26 events, mostly locomotion and sports plus some scenario tasks. I could not confirm a 2025 hammering event — [Wikipedia](https://en.wikipedia.org/wiki/World_Humanoid_Robot_Games)
- Figure 02 hand: 16 DoF, claimed 25 kg payload per hand, tactile sensing down to 3 g. These are company claims via press — [UST](https://www.unmannedsystemstechnology.com/2024/02/humanoid-robot-showcases-hand-dexterity/). I found no Optimus or Figure hammer-and-nail demo (search returned none).

### Inferences
- The prior art tunes grasps or adds learned residuals to *limit slip kinematically*. None of it audits whether rated finger torques or drives can physically sustain the impulse. "At rated torques a power wrap cannot hold, and holding requires self-locking drives loaded 20-40x rating" is a hardware-feasibility result that G2A and SimToolReal do not address. G2A's success on a LEAP hand (Dynamixel servos, backdrivable but relatively high-reduction) probably involved actuator overloading that was never reported. This is speculative and needs confirmation from their paper or hardware.
- G2A models the nail as immovable and lets contact generate disturbances, so it has no ground-truth force channel. The planned paper's force truth at 8 kHz is a methodological difference.

### Gaps
- Not verified: whether DexDrummer, SGFT or Mana report grip forces or actuator loads during impact.
- No primary source found for a Tesla Optimus, Figure or Unitree hammering demo. The 2025 Games task list is unconfirmed.
- No result found for hammering with Inspire, Ability or Allegro hands that reported actuator loads.

## Q2. Power-grasp holding capacity vs actuator limits; self-locking vs backdrivable; power-tool grip forces

### Takeaway
Self-locking (worm or wedge-clutch) finger transmissions are well established for holding load without power. The literature frames this qualitatively ("resist forces up to material strength") and does not set it against impulsive tool loads or quantify gearbox overload versus rating. I could not locate the Lindenmann et al. 84-156 N power-tool grip-force source.

### Cited Findings
- Worm gears are widely used in robotic hands for their high reduction and self-locking property. A non-backdrivable gripper stays closed when power is lost — [ResearchGate figure, non-backdrivable worm gear](https://www.researchgate.net/figure/Scheme-of-a-non-backdrivable-worm-gear-mechanism_fig1_221144291); [Self-locking underactuated mechanism](https://www.researchgate.net/publication/319369000_Self-locking_underactuated_mechanism_for_robotic_gripper)
- A patent states that non-backdrivable worm trains let "large contact forces be resisted to the limit of the strength of the materials" — [US 4957320](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/4957320)
- A wedge-based non-backdrivable clutch in a prosthetic joint improves manipulation stability and transmits more efficiently than worm gears — [PMC11169929](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11169929/); also [Mech. Mach. Theory, miniaturized NBD mechanism](https://www.sciencedirect.com/science/article/abs/pii/S0094114X10000947)
- WUJI Hand generation 1 was described as non-backdrivable, with joint torque IP/MCP 0.3 / 1.0 Nm. WUJI Hand 2 moved to direct-drive backdrivable joints. Source is a MIDAS Hand paper's comparison, via search snippet — [MIDAS Hand arXiv 2607.14487](https://arxiv.org/html/2607.14487)
- Passive reaction analysis for grasp stability addresses passive (non-backdrivable) joint grasps analytically — [arXiv 1801.06558](https://arxiv.org/pdf/1801.06558)
- Kengoro's musculoskeletal hand with machined springs aimed for self-weight support (56.4 kg robot), an example of high-load hand design — [arXiv 2403.17459](https://arxiv.org/pdf/2403.17459)

### Inferences
- "Self-locking holds but overloads the gearbox 20-40x" directly qualifies the patent-style claim of "limited only by material strength". Quantifying how much material margin a dexterous-scale gearbox would need under a hammer impulse appears new.
- The WUJI generation switch (non-backdrivable to backdrivable direct drive) makes the self-locking ablation especially relevant: it contrasts the two WUJI generations' design philosophies.

### Gaps
- **Lindenmann et al. (84-156 N power-tool grip forces): not found** in two searches. The caller should verify the citation (it may be a German ergonomics or DGUV study).
- iCub, Schunk SVH and Robotiq backdrivability and brake specs were not checked. Known from memory, unverified: Robotiq 2F-85 is self-locking (non-backdrivable) on power loss, and Schunk SVH reportedly uses non-backdrivable drives.
- I found no paper computing a grasp wrench space bounded by actuator torque ratings for impulsive tool loads.

## Q3. WUJI Hand 2 and Sharpa Wave specs; published use for tool impacts

### Takeaway
WUJI Hand 2 is a 20-DoF backdrivable direct-drive hand with a 1 kHz control rate and fingertip peak force of at least 12 N. Public pages do **not** list per-joint torques, so the paper's ~0.2-2 Nm ratings need a citable source. Sharpa Wave is a 22-DoF hand with 20 N fingertip force, a 500 Hz control rate and 180 FPS tactile. It is used in SimToolReal (hammer trajectories without impact measurement).

### Cited Findings
- WUJI Hand 2: 20 DOF, "Direct-drive rotary, back-drivable", 1000 Hz across 20 axes over 100BASE-T Ethernet, fingertip peak force ≥ 12 N, ≥ 4 open/close cycles/s, joint backlash < 1°, 12 V DC, about 180 × 80 × 40 mm, multi-axis F/T fingertip sensing. Specs are labeled "Beta1", subject to change — [wuji.tech/en/hand2](https://www.wuji.tech/en/hand2)
- WUJI docs (Hand 2.2 beta): FOC control, about 800 g (650 g bare), 4 DoF per finger. No per-joint torques, gear ratios or payload are listed. "Fingertip soft-pad hardness is still being tuned" — [docs.wuji.tech](https://docs.wuji.tech/docs/en/wuji-hand/latest/overview/)
- Sharpa Wave: 22 DoF, max active fingertip force 20 N, a tactile sensor and torque sensor in each finger, more than 1000 taxels per fingertip, 0.005 N sensitivity, DTA at 180 FPS with under 1 mm resolution, 1.3 kg, GigE at 500 Hz control, 15 W static / 180 W peak — [CNX Software](https://www.cnx-software.com/2026/06/02/sharpa-wave-high-end-dexterous-robotic-hand-with-22-dof-high-sensitivity-dynamic-tactile-array/); [sharpa.com/pages/wave](https://www.sharpa.com/pages/wave)
- Sharpa hands were added to an NVIDIA humanoid reference design — [Engineering.com](https://www.engineering.com/sharpa-hands-added-to-nvidia-humanoid-robot-design/)

### Inferences
- With fingertip forces of 12-20 N per finger, even ideal five-finger summation sits in or below the range typically cited for human power-tool grip. That is consistent with the planned finding that rated torques cannot hold a hammer blow.

### Gaps
- No primary source found for WUJI Hand 2 per-joint torque ratings (~0.2-2 Nm). If the paper uses them, cite the datasheet or SDK file, or state where the numbers came from.
- Sharpa Wave backdrivability and per-joint torques were not found.
- No published work found using WUJI Hand 2 for tool impacts.

## Q4. Impact simulation fidelity (MuJoCo/Isaac/Newton) and hammering benchmarks

### Takeaway
Impact sim-to-real is studied for rigid-body velocity jumps (van Steen et al., AGX Dynamics) and for simulator validation on single-object impacts (arXiv 2110.00541). Simulators capture inelastic impacts reasonably well but fail on elasticity. No hammering benchmark (Adroit, DexToolBench, G2A's Isaac Lab setup) provides force ground truth with actuator limits enforced. arXiv 2606.28805 does exist, but it is Sony AI's table-tennis physics paper, not a hand-contact paper.

### Cited Findings
- van Steen, Stokbroekx, van de Wouw, Saccon (arXiv 2411.06319, submitted to T-RO) generate and validate "impact maps" from physics-engine simulation (AGX Dynamics, nonsmooth time-stepping). The model assumes instantaneous rigid contact and neglects impact-induced vibrations — [arXiv](https://arxiv.org/abs/2411.06319)
- "Validating Robotics Simulators on Real-World Impacts" (arXiv 2110.00541) finds simulators including MuJoCo "capture inelastic impacts surprisingly well, though generally fail to reproduce elasticity" — [arXiv PDF](https://arxiv.org/pdf/2110.00541)
- MuJoCo's own docs say its soft constraints limit accuracy for rigid contacts and stick-slip transitions. Raising stiffness past a threshold makes the system unstable or noisy. Restitution comes from solref stiffness and damping, and energy is not exactly conserved because contact spans several timesteps — [MuJoCo modeling docs](https://mujoco.readthedocs.io/en/latest/modeling.html); [GitHub discussion #2347](https://github.com/google-deepmind/mujoco/discussions/2347); [arXiv 1702.07252](https://arxiv.org/pdf/1702.07252)
- arXiv 2606.28805 is Conti et al. (Sony AI), "Physics Models for Sim-to-Real Transfer in Professional-Level Robot Table Tennis". It adds residual neural corrections to normal and tangential restitution and spin damping for racket-ball contact, and reports a 59% median landing-error reduction — [arXiv](https://arxiv.org/abs/2606.28805). It is relevant as a learned velocity-dependent restitution precedent, but it does not involve grasped tools or hands.
- Adroit Hammer: nail friction up to 15 N, position actions, no actuator-load reporting — [Gymnasium docs](https://robotics.farama.org/envs/adroit_hand/adroit_hammer/)

### Inferences
- An 8 kHz MuJoCo testbed reporting peak contact forces and hard-stop loads should acknowledge that MuJoCo's soft contact under-resolves short impulses and ringing. That supports framing results as relative comparisons (rated vs self-locking vs gripper) rather than absolute force predictions, or validating peak force against a drop or impact test.

### Gaps
- Not checked: Newton or Isaac Lab impact fidelity, and ManiSkill, RLBench, Meta-World or DexArt hammer-task details. From memory, unverified: Meta-World "hammer" and RLBench have hammer-like tasks with position or end-effector control and no force truth.

## Q5. Testbeds that enforce real hardware limits (torque-rate, velocity, interface rate)

### Takeaway
Actuator-realistic simulation (torque-speed saturation, backlash, delay, current-to-torque) is common for sim-to-real *training*, especially in locomotion and in one dexterous force-grasping paper. I found no manipulation *evaluation testbed* that jointly enforces published torque, torque-rate, velocity and interface-rate limits across arm, hand and humanoid for impact tasks.

### Cited Findings
- Zhao et al. (arXiv 2601.02778) model actuator dynamics, backlash and torque-speed saturation, and calibrate current to torque. They report the "first demonstration of controllable grasping on a multi-finger dexterous hand trained entirely in simulation and transferred zero-shot". The tasks are force tracking and reorientation, with no tools or impacts — [arXiv](https://arxiv.org/abs/2601.02778)
- Actuator Reality Shaping (arXiv 2607.02205) and trajectory-based actuator identification via differentiable simulation (arXiv 2604.10351) target the actuator-level sim-to-real gap (friction, backlash, saturation, delays). Search snippets only — [2607.02205](https://arxiv.org/pdf/2607.02205); [2604.10351](https://arxiv.org/html/2604.10351v1)
- Legged sim-to-real practice clips torques to measured torque-speed curves — [arXiv 2509.06342](https://arxiv.org/html/2509.06342v1); [Booster Lab 2606.27813](https://arxiv.org/pdf/2606.27813)

### Inferences
- The "hardware-faithful" angle is incrementally novel: enforcing Franka FR3 limits and a 100 Hz position-only humanoid interface for *impact tool use* is not covered by the sources found. Its novelty is application and integration rather than method. The humanoid result (about 1.1 m/s strike from a 100 Hz position interface) has no direct comparator in the sources found.

### Gaps
- Not checked: robosuite or ManiSkill controller-limit options, or Franka-specific sim wrappers that enforce the libfranka jerk and torque-rate limits.
