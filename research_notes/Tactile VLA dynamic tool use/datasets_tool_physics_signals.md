# Verified catalog: public SIGNAL datasets for power-tool / screwdriving / drilling / hammering / impact physics

Purpose: candidate real-signal sources for (a) calibrating a phenomenological "drill plant" (motor + clutch + bit-workpiece state machine + vibration synthesis), (b) training engagement / cam-out / stripped / seated classifiers, (c) training chatter / bind / kickback detectors, (d) deriving vibration spectra for synthesis, (e) setting human grip-force priors, (f) hammer/impact force-pulse physics.

Verification date: 2026-09-28. Every entry below was checked by opening the dataset landing page, the repository README, or the paper PDF (downloaded and text-extracted). "not stated" = not on the page/paper I could open. Resolvability rule used throughout: a ~3 ms impact pulse needs >= 1-2 kHz sampling (>= 3-6 samples/pulse); a 50-300 Hz vibration with harmonics to ~1 kHz needs >= 2 kHz.

Flags: **[DRILL-TOP5]** = top 5 for calibrating the drill plant; **[IMPACT-TOP3]** = top 3 for hammer/impact physics.

---

## 1. Summary table (sorted by relevance to the project)

| # | Dataset (year, institution) | Domain | Key signals @ rate | Size | Labels | Access | Resolves 3 ms pulse? / 50-300 Hz + harmonics? | Primary use here |
|---|---|---|---|---|---|---|---|---|
| 1 | **PyScrew** (2025, TU Dortmund IPS / RIF e.V.) **[DRILL-TOP5]** | Industrial screwdriving | Torque, angle, gradient, time, process step @ 833.33 Hz | 34,082 runs, 6 scenarios, 296 MB | 1 + 8 + 26 + 25 + 42 + 44 scenario classes (incl. driver slippage, deformed/stripped threads, enlarged holes, washers, torn screws, torque/speed changes), OK/NOK | Open, CC BY 4.0, Zenodo | Pulse: marginal (~2.5 samples); vibration: torque ripple only to ~400 Hz | Torque-angle regime curves; engagement / slip / strip / seat classifier |
| 2 | **PHM Society 2010 Data Challenge – milling** (2010) **[DRILL-TOP5]** | High-speed milling tool wear | Fx,Fy,Fz dynamometer; Vx,Vy,Vz accel; AE-RMS @ 50 kHz | 6 cutters x ~315 cuts (~800 MB each) | Flank wear per cut (c1,c4,c6) | Official links dead; Kaggle mirror (login); IEEE DataPort (subscription) | Yes / Yes | Cutting-force + vibration texture vs. wear; vibration synthesis |
| 3 | **Turning chatter dataset** (2019, Michigan State Univ. + TU Chemnitz; Khasawneh, Otto, Yesilli) **[DRILL-TOP5]** | Turning chatter | Accelerometers (2 uniaxial + triaxial), microphone, tachometer @ 160 kHz raw; conditioned channel @ 10 kHz | 83 tagged time series over 4 stick-outs | stable / intermediate chatter / chatter (+unknown) | Open, CC BY 4.0, Mendeley Data | Yes / Yes | Chatter / bind detector; chatter spectral model |
| 4 | **Rzeszow milling tool-life dataset** (2025, Rzeszow Univ. of Technology; Sci Data) **[DRILL-TOP5]** | Milling to tool failure | 8 accelerometers @ 25 kHz; 12 motor-current transformers @ 0.5 kHz | 14 tools, 968 cycles, run-to-failure | Failure cycle, normalized cycles-to-failure | Open, CC BY 4.0, figshare (+Kaggle) | Yes (accel) / Yes; current only to 250 Hz | Motor-current <-> vibration coupling as load/wear grows |
| 5 | **AURSAD** (2021, Aarhus Univ. + Technicon) **[DRILL-TOP5]** | Robotic screwdriving anomalies | 125 UR3e + OnRobot channels (joint currents, 6-axis TCP force, screwdriver torque, ...) @ 100 Hz | 2,045 tightening + 2,049 loosening samples; 6.4 GB HDF5 | normal / damaged screw / extra component / missing screw / damaged thread | Open, CC BY 4.0, Zenodo | No / No (regime-level only) | Engagement-failure (missing screw = free-spinning bit) and obstruction signatures at controller rate |
| 6 | **Mondragon MU-TCM face-milling** (2025, Mondragon Unibertsitatea; Sci Data) | Milling tool condition | Kistler 9139AA forces @ 50 kHz; PCB accel @ 50 kHz; Kistler 8152C AE @ 1 MHz (filtered/RMS); CNC internals @ 250 Hz | 67 cuts, 4 wear levels, 2 materials | Flank wear 0-0.3 mm | Open, CC BY 4.0, eBiltegia (MU repository) | Yes / Yes | Force/vibration/AE vs. wear; AE burst statistics |
| 7 | **Bosch CNC Machining** (2022, Bosch Research) | Brownfield CNC vibration anomalies | Tri-axial accel @ 2 kHz | 3 machines x 15 processes, 6 six-month periods | good / bad (anomalous) | Open, CC BY 4.0, GitHub (archived 2026-02) | Marginal (~6 samples) / Yes to ~1 kHz | Machine vibration "normal vs. anomalous" spectra |
| 8 | **NASA/BEST Lab Milling Data Set** (2007, UC Berkeley / NASA Ames) | Milling tool wear (entry, regular, exit cuts) | AC & DC spindle current, table & spindle vibration, table & spindle AE (RMS-processed) @ 250 Hz | 167 runs (16 cases), 9,000 samples/run | Flank wear VB | Open ("use at own risk"), 14.7 MB zip | No / No (envelopes only) | Current-vs-wear envelopes; cheap sanity set |
| 9 | **QIT-CEMC end-mill wear** (2024, Qilu Inst. of Technology; Sci Data) | Milling Ti6Al4V | Kistler 9170B251 force + torque @ 10 kHz; vibration @ 10 kHz; sound @ 10 kHz | 68 samples x ~5 M rows | Wear (VBmax, area), 4 regions | Open, CC BY-NC-ND 4.0, figshare | Yes / Yes | Torque + vibration + sound joint statistics vs. wear |
| 10 | **HES-SO SOON milling tool wear** (2022, HES-SO Saint-Imier) | Milling quality/tool wear | AE @ 200 kHz; 9 accelerometers @ 20 kHz; 6 motor currents @ 1 kHz | 14.1 GB | Tool wear / quality | Open (license text "Open"; specific license not stated), Zenodo | Yes / Yes | AE + current + vibration for multi-rate sensor fusion |
| 11 | **Tool Tracking Dataset** (Fraunhofer IIS; Mutschler) | Hand-held power tools in use | IMU acc/gyr ~102 Hz, magnetometer ~155 Hz, microphone (rate not stated) | 4 tools x 30-60 min; 100-200 instances/action | tightening, untightening, motor CW/CCW, manual rotation, shaking, undefined | Open non-commercial, CC BY-NC-SA 4.0, GitHub + Fraunhofer ownCloud | No (IMU) / mic possibly | Action segmentation priors for real electric/pneumatic screwdrivers, riveting gun, torque wrench |
| 12 | **REASSEMBLE** (2025, TU Wien ASL; RSS) | Contact-rich assembly (NIST Task Board 1) | 6-axis F/T (AIDIN AFT200-D80-C), 3 microphones, event camera, Franka state; rates "native", not stated | 4,551 demos (4,035 successful), 781 min, 54.8 GiB | Success/failure, 2-level action segments | Open, CC BY 4.0, TU Wien Research Data | not stated | Insertion/removal contact transients + audio; NOTE nuts (threading) were excluded |
| 13 | **TU Delft impact-hammer test of Al plate** (2022) **[IMPACT-TOP3]** | Modal impact testing | PCB 086C02 impact hammer force (N) + 352A24 accel @ ~8.5 kHz (32,768 samples / 3.84 s) | 25 impact points, 13.8 MB TDMS | Point locations | Open, CC BY 4.0, Zenodo | Yes / Yes | Calibrated hammer force-pulse shapes (rise/duration/peak) |
| 14 | **Brno impact-echo hammer dataset** (2021, Brno Univ. of Technology) **[IMPACT-TOP3]** | Impact-echo on concrete beam | Hammer-embedded piezo + receiving piezo, time arrays @ 193 kHz | 1 beam; blunt/sharp tip x handle/no-handle x several force levels; 9.9 MB .mat | Tip type, handle type, excitation voltage | Open, CC BY 4.0, figshare | Yes / Yes | Pulse shape vs. tip hardness/handle; impact + structural ring-down |
| 15 | **RealImpact** (2023, Stanford; CVPR) **[IMPACT-TOP3]** | Impact sounds of real objects | PCB 086E80 impact-hammer force profiles + 48 kHz audio (force rate not stated) | 50 objects x 600 mic positions; 150,000 recordings | Impact location, material, mic pose, RGBD | Open via project page; license not stated | Audio yes; force: rate not stated | Force-pulse -> acoustic-emission mapping across materials |
| 16 | **NTNU hammer-tap F/T + IMU** (2024, NTNU) | Robot end-effector estimation with hammer taps | ATI Gamma F/T ~700 Hz; MPU6886 IMU 254 Hz; KUKA LBR Med 14 | 12 CSV files, 1.4 MB | Condition (no tap / taps / taps + manual force) | Open, CC BY 4.0, Zenodo | Marginal (700 Hz) / partly | Hammer-tap disturbance seen through a wrist F/T sensor |
| 17 | **Sensorized hammerstone knapping** (2024, Univ. of Wollongong; PLOS ONE) | Human striking force | Pneumatic-chamber pressure @ 100 kHz, force plate @ 1 kHz (in study); **shared file = peak pairs only** | 240 strikes (24 trials); S5 = 180 peak pressure/force pairs | Model vs. validation | Open (PLOS CC BY) - peaks only | Time series NOT shared | Human strike-force distribution prior (135-730 N) |
| 18 | **I.AM. box-impact archive / Impact-Aware Robotics Database** (2021-2023, TU/e) | Robot-induced impacts (tossing/dropping boxes, arm impacts) | Motion capture + F/T sensor (rates not stated) | 2.5 GB HDF5 (box drops) + further 4TU datasets | Experiment metadata | Open, CC BY-NC-SA 4.0, 4TU.ResearchData | not stated | Impact-transition modelling reference; not tool-specific |
| 19 | **KIT IPEK hand-arm impedance dataset** (2025, KIT) | Hand-arm biodynamics vs grip/push force | Shaker multisine 10-500 Hz (5 Hz steps) @ 40 m/s2 RMS; accelerations + 3-axis forces top/bottom; grip 10-135 N, push 10-110 N (10 levels each) | 6 subjects; raw .mat 17.1 GB | Grip/push level, anthropometrics, max forces | Open, CC BY 4.0, RADAR4KIT | n/a (steady-state) / Yes (10-500 Hz) | Hand-arm impedance = boundary condition for the tool-handle model; grip-force priors |
| 20 | **Grip/load-force coupling during step-down** (2016, UCLouvain; Dryad/Zenodo) | Precision-grip GF/LF under brisk load increase | ATI Mini40 GF/LF @ 1 kHz | 12 participants x 8 trials; 448 kB | Task phases | Open, CC0 | Yes (1 kHz) / partly | Human GF-LF coupling latency (~70 ms) prior for reactive grip layer |
| 21 | **KIT IPEK hammer-drill posture/muscle dataset** (2025, KIT + TUM) | Hammer drill user forces | Grip & push force (means), EMG RMS (8 muscles), joint angles | 3 participants x handle configs x 3 trials; 1.8 MB xlsx | Handle configuration | Open, CC BY 4.0, RADAR4KIT | No (summary values) | Grip/push-force priors for hammer drilling |
| 22 | **VIBTOOL** (2026, Univ. Politecnica de Madrid) | Smartwatch IMU on 9 vibrating hand tools | 6 inertial channels @ 100 Hz | 19 participants x 9 tools x ~30 s; 28.1 MB | Tool identity | Open, CC BY 4.0, Zenodo | No / No (aliased) | Wrist-level tool-vibration signatures only |
| 23 | **F.A.I.R. brushed DC-motor faults** (2020, CARTIF) | Small DC motors run to failure | Vibration, current, voltage, temperature, noise @ 51.2 kHz | 483 MB HDF5 | Fault evolution (commutator, winding, brush) | Open, CC BY 4.0, Zenodo | Yes / Yes | Brushed-motor current/vibration signature priors (commutation ripple) |
| 24 | **Induction-motor current signatures under load** (2022, MUET) | Motor current vs load | 3-phase current @ 10 kHz | 39 datasets; loads 100/200/300 W; healthy + bearing + broken-bar | Health + load | Open, CC BY 4.0, Mendeley Data | Yes / Yes | Load-dependent current signature statistics (not a tool motor) |
| 25 | **ext-sense (Extended Tactile Perception)** (2021, NUS CLeAR) | Vibration through held rods/fork into finger sensors | NUSkin event-based taxels @ 4 kHz; BioTac micro-vibration (rate not stated in README) | Raw 946 MB + preprocessed 696 MB (Dropbox) | Tap location on 20/30/50 cm rods; handover grasp stability; food class | Open download; license not stated | Yes (4 kHz) / Yes | Tool-borne vibration into fingertip sensors; localization of impacts along a held tool |
| 26 | **Penn Haptic Texture Toolkit (HaTT)** (2014, UPenn) | Tool-surface vibration | Force, speed, high-freq acceleration @ 10 kHz, 10 s per texture | 100 textures | Texture identity | Non-commercial research use; UPenn repository (403 at check) | Yes / Yes | Accelerometer-on-tool vibration realism; texture/friction models |
| 27 | **Cluster Haptic Texture Dataset** (2026, Sci Data; figshare) | Tool-surface vibration under controlled velocity/direction | Audio, acceleration, force, position, video (rates not stated on landing page) | 118 textures, 18,880 recordings, 15.2 GB | Velocity (20-60 mm/s), direction, material | Open, CC BY 4.0 | likely (not verified) | Alternative to LMT/HaTT with force + acceleration |
| 28 | **Vibro-Sense** (2026, Leibniz Univ. Hannover / L3S) | Vibro-acoustic touch localization on robot hand | 7 piezo microphones @ 50 kHz raw (20 kHz used) | "over 20,000 interactions per indenter" | Contact location / trajectory | Open (HF dataset); license not stated | Yes / Yes | Structure-borne impulse-response library for a hand/tool |
| 29 | **M2/M3 screw torque-angle curves** (2024, UACJ/UTCJ Mexico) | Small-screw tightening | Torque-angle raw curves (rate/tool not stated) | 83 screws, 22 MB | Gaussian fits, GPR models | Open, CC BY 4.0, Zenodo | not stated | Seating-curve shape prior for small screws |
| 30 | **TalTech BLDC hub-motor currents** (2026, IEEE DataPort) | BLDC phase currents under torque | 3-phase currents, voltage, torque, speed @ 100 kHz (+ down-sampled copies) | 144 files; 200-350 rpm; 5-10 Nm | Injected imbalance faults | **Subscription** | Yes / Yes | BLDC current ripple vs. load (if subscription available) |
| 31 | **NJMU Bone Drilling Force Dataset** (2024, IEEE DataPort) | Bone drilling forces (bovine) | Radial Fx/Fy, thrust, torque paired with 99x99 images (rate not stated) | 14 paths / 3 bones (9,162 images) + 600 synthetic paths | Force values (breakthrough not stated) | **Subscription** | not stated | Thrust/torque vs. depth in layered material (if accessible) |

Not included above (rejected or unverifiable): see Section 3 and 4.

---

## 2. Per-dataset verified details

### 2.1 Screwdriving

#### PyScrew - Industrial screw driving dataset collection **[DRILL-TOP5]**
- Year / institution: 2025 (Zenodo latest v1.2.3, 2025-07-21; arXiv 2505.11925). TU Dortmund, Institute of Production Systems, and RIF Institute for Research and Transfer e.V. (N. West, J. Deuse).
- URLs: code/docs https://github.com/nikolaiwest/pyscrew ; concept DOI https://doi.org/10.5281/zenodo.14729547 ; v1.1.0 record https://zenodo.org/records/14769379 ; paper https://arxiv.org/abs/2505.11925 ; per-scenario docs e.g. https://github.com/nikolaiwest/pyscrew/blob/main/docs/scenarios/s04_assembly-conditions-2.md
- Signals: torque (Nm), angle (deg), time, gradient (Nm/deg), step (process phase index), class. Sampling frequency 833.33 Hz ("approximately 3-4 data points per degree"). Also torqueRed/angleRed monitoring values.
- Sensors/tool: Bosch Rexroth CS351S-D compact tightening system with EC302 servo motor, 2GE26 planetary gearbox, 2DMC006 measuring transducer (integrated torque/angle). Target torque 1.4 Nm (window 1.2-1.6 Nm). Process phases: finding, driving-in, pre-tightening, final tightening.
- Workpiece: EJOT DELTA PT 40x12 thread-forming screws into thermoplastic (glass-fibre reinforced) motor-control-unit housings (upper/lower part).
- Size: 34,082 runs total - s01 thread degradation 5,000 (1 class, 100 workpieces, 25 reuses each); s02 surface friction 12,500 (8); s03 assembly conditions 1: 1,700 (26); s04 assembly conditions 2: 5,000 (25 = control + 24 errors, interleaved 5 OK / 5 NOK); s05 upper-workpiece fabrication 2,400 (42); s06 lower-workpiece fabrication 7,482 (44). Total 296.4 MB (JSON per run + labels.csv).
- Labels: class per run, station OK/NOK, workpiece usage count, screw position (left/right), scenario condition. Error classes directly relevant to this project (from s03/s04 docs): 301 material in screw head -> "driver slippage ... inconsistent torque transfer" (closest public analogue to cam-out); 401/302 enlarged pilot hole -> reduced thread engagement (strip-like); 202/203/101 deformed thread; 404/405 torn-off screw shaft; 101-103 washers -> reduced insertion depth (premature seating); 305 gap between parts (no clamp); 201/202 damaged/broken contact surface; 501/502 angular velocity +/-10%; 503/504 target torque +/-0.1 Nm; s02 surface friction (lubricant, moisture, sanded 40/400, adhesive, scratched).
- License / access: CC BY 4.0, open (Zenodo; `pip install pyscrew` loader).
- Resolvability: 833 Hz -> Nyquist ~417 Hz: captures torque ripple / thread-forming periodicity up to ~400 Hz; a 3 ms pulse is only ~2.5 samples (marginal). No vibration/current channel.
- Use: fit torque-angle regime curves for the plant's bit-workpiece state machine (thread forming -> run-down -> head seating -> clamp -> over-torque), and train engagement / slip / strip / seated classifiers with 40+ labelled fault conditions.

#### AURSAD - Universal Robot Screwdriving Anomaly Detection Dataset **[DRILL-TOP5]**
- Year / institution: 2021. Aarhus University (Dept. ECE) + Technicon ApS (B. Leporowski, D. Tola, C. Hansen, A. Iosifidis).
- URLs: https://zenodo.org/records/4487073 (DOI 10.5281/zenodo.4487073); tech report https://arxiv.org/abs/2102.01409 ; conference paper https://arxiv.org/abs/2107.01955 ; loader https://pypi.org/project/aursad/ ; code https://github.com/CptPirx/AURSAD-source
- Signals: 125 raw channels sampled at 100 Hz over UR RTDE + OnRobot: target/actual/control joint currents, joint positions/velocities, TCP pose and 6-axis TCP force, safety-board robot current, screwdriver current torque, etc. Continuous stream with per-event labels.
- Tool: OnRobot Screwdriver (0.15-5 Nm, embedded Z-axis) on UR3e. Workpiece: not stated on landing page.
- Size: 2,045 tightening samples (normal 1,420 = 69.4%; damaged screw 221; extra assembly component 183; missing screw 218; damaged thread 3) plus 2,049 loosening/screw-pick samples; one 6.4 GB HDF5 file.
- Labels: 4 anomaly types + normal (+ loosening as optional 6th class); 'full' / 'partial' / 'tighten' labelling modes.
- License / access: CC BY 4.0, open.
- Resolvability: 100 Hz -> no pulse or vibration resolution; regime-level torque/current trajectories only.
- Use: robot-side signature of engagement failure (missing screw = free-spinning bit) and obstruction (extra component); low-rate proxy of what a slow VLA layer would see.

#### CMU "Data-driven classification of screwdriving operations" (Aronson et al., ISER 2016) - NOT PUBLIC
- Paper: https://publications.ri.cmu.edu/storage/publications/pub_files/2016/10/final_screw_ISER2016.pdf (also Springer 10.1007/978-3-319-50115-4_22).
- Verified from paper: 1,862 runs; 6-axis force/torque (ATI Mini40), motor current, motor speed, all at 100 Hz, plus robot position and video; hand-labelled stages and results. Result classes: success 84.4%, no screw 9.7%, no hole found 2.6%, crossthreaded 1.8%, stripped 0.8%, partial 0.3% (plus stage labels such as hole finding, no-screw spinning, stripped engaging/rundown/tightening).
- Data availability: no data-availability statement or URL in the paper; the CMU Manipulation Lab site (mlab.ri.cmu.edu) did not resolve on 2026-09-28; no repository found by search. Treat as request-only (contact authors).

#### M2/M3 Screw Torque-Angle Curve Dataset
- 2024-10-02; Universidad Autonoma de Ciudad Juarez / Universidad Tecnologica de Ciudad Juarez. https://zenodo.org/records/13878636 (DOI 10.5281/zenodo.13878636). CC BY 4.0. Raw torque-angle curves for 83 M2/M3 screws + Gaussian fits + GPR models (Datasets.zip 22.3 MB). Screwdriver, sensors and sampling rate: not stated. Use: small-screw seating-curve shape prior only.

#### REASSEMBLE (TU Wien)
- 2025 (RSS). https://researchdata.tuwien.ac.at/records/0ewrv-8cb44 (DOI 10.48436/0ewrv-8cb44); https://github.com/TUWIEN-ASL/REASSEMBLE ; https://arxiv.org/abs/2502.05086
- Wrist 6-axis F/T AIDIN ROBOTICS AFT200-D80-C, 3 microphones (gripper mic OSA K1T), DAVIS346 event camera, RGB cameras, Franka Emika Panda state; each modality stored at its native rate in HDF5 (rates not stated). 4,551 demos (4,035 successful), 781 min; 17 objects of NIST Assembly Task Board 1 - the three nuts (M12, M8, M4) were excluded, so there is no threading task. CC BY 4.0; 54.8 GiB.
- Use: contact-transient F/T + audio priors for insertion/removal; not a screwdriving source.

#### Tool Tracking Dataset (Fraunhofer IIS)
- Pages: https://mlps-lab.de/datasets/tool-tracking-dataset (redirect target of cmutschler.de); repo https://github.com/mutschcr/tool-tracking (README fetched 2026-09-28: CC BY-NC-SA 4.0; data pulled from a Fraunhofer ownCloud via script).
- Tools: electric screwdriver, pneumatic screwdriver, pneumatic riveting gun, torque wrench. Sensors: accelerometer + gyroscope ~102.3 Hz mean, magnetometer ~154.6 Hz mean, microphone (rate not stated). 30-60 min raw data per tool, 100-200 instances per action; labels (electric screwdriver): tightening, untightening, motor CW/CCW, manual motor rotation, shaking, undefined, garbage. Year not stated (2020 update noted).
- Use: real hand-held power-tool action labels; IMU too slow for vibration physics, microphone may carry motor/clutch acoustics.

### 2.2 Drilling / milling / chatter (no dedicated open drilling thrust/torque dataset was found - see Section 4)

#### PHM Society 2010 Data Challenge (CNC milling) **[DRILL-TOP5]**
- Pages: https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/ ; IEEE DataPort https://ieee-dataport.org/documents/2010-phm-society-conference-data-challenge (DOI 10.21227/jdxd-yy51) and https://ieee-dataport.org/documents/phm2010-dataset (DOI 10.21227/bh7t-qe79); Kaggle mirror https://www.kaggle.com/datasets/rabahba/phm-data-challenge-2010
- Signals: 7 channels at 50 kHz/channel - force X/Y/Z (dynamometer), vibration X/Y/Z (accelerometers), AE-RMS. Machine: high-speed CNC, 6 mm ball-nose tungsten-carbide 3-flute cutter, 10,400 rpm, 1,555 mm/min, radial DOC 0.125 mm, axial DOC 0.2 mm, stainless steel (HRC52) workpiece.
- Size: 6 cutters c1-c6 (~800 MB each), ~315 cuts each; flank-wear file per cut for c1, c4, c6.
- Access (verified 2026-09-28): the six zip links on the PHM Society page point to cdn.teamholistic.net and return a parked-domain redirect (dead). IEEE DataPort entries require a subscription and showed no uploaded files. The Kaggle mirror page exists (content needs login). License: none specified.
- Resolvability: yes / yes.
- Use: cutting-force and vibration texture vs. wear; vibration synthesis reference for a rotating cutter.

#### Turning Dataset for Chatter Diagnosis (Khasawneh / Otto / Yesilli) **[DRILL-TOP5]**
- 2019-05-29; Michigan State University + TU Chemnitz. https://data.mendeley.com/datasets/hvm4wh3jzx/1 (DOI 10.17632/hvm4wh3jzx.1). CC BY 4.0. Paper: https://arxiv.org/abs/1908.01678
- Signals: two perpendicular uniaxial accelerometers, one triaxial accelerometer, microphone, laser tachometer, raw at 160 kHz (NI USB-6366); conditioned single accelerometer channel low-pass filtered and down-sampled to 10 kHz with tags.
- Setup: Al 6061 turned on Clausing-Gamet 33 cm lathe; 0.015 in TiN insert on S10R-SCLCR3S boring bar; stick-outs 5.08 / 6.35 / 8.89 / 11.43 cm (varies stiffness and chatter frequency).
- Size/labels: 83 tagged series - stable 44, intermediate 18, chatter 21 (per stick-out 36/14/11/22) plus "unknown".
- Resolvability: yes / yes.
- Use: chatter/bind detector training and validation; source spectra for a self-excited-vibration regime in the plant.

#### Rzeszow University of Technology milling tool-life dataset (Sci Data 2025) **[DRILL-TOP5]**
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12006439/ (Sci Data 12, 650). Data: https://doi.org/10.6084/m9.figshare.28589216 (also Kaggle). CC BY 4.0.
- Haas VF-1, Beckhoff PAC; 42CrMo4 (38 +/- 2 HRC) blocks; 14 tools from two makers run to failure; 968 milling cycles. 8 accelerometers (spindle and axes) at 25 kHz; 12 current transformers (3-phase, multi-axis) at 0.5 kHz; raw + 120 aggregated features per cycle; failure cycle and normalized cycles-to-failure.
- Resolvability: accel yes / yes; current only to 250 Hz.
- Use: how motor current tracks vibration and load as the tool degrades - the plant's motor-current-vs-load channel.

#### Mondragon MU-TCM face-milling dataset (Sci Data 2025)
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12102343/ ; data DOI 10.48764/3hdp-gf23 (eBiltegia, Mondragon Unibertsitatea). CC BY 4.0.
- LAGUN L1000 VMC; 80 mm face mill with 9 Ayma SPKR M55 inserts; GG30 cast iron and 316L. Kistler 9139AA dynamometer 50 kHz (X/Y/Z); PCB J356A45 accelerometer 50 kHz; Kistler 8152C AE at 1 MHz (filtered + RMS); 16 CNC internal signals (speed, current, torque, position) at 250 Hz. 32 planned experiments, 67 cuts, 4 flank-wear levels (0-0.3 mm).
- Use: high-rate force/vibration/AE triplets vs wear; AE burst statistics for synthesis.

#### Bosch CNC Machining dataset
- 2022 (Procedia CIRP 107:131-136). https://github.com/boschresearch/CNC_Machining (archived read-only 2026-02-19). Data CC BY 4.0, code BSD-3.
- Bosch CISS tri-axial accelerometer inside a brownfield CNC milling machine at 2 kHz; 3 machines (M01-M03) x 15 processes (OP00-OP14); 6 six-month periods Oct 2018-Aug 2021; labels good/bad; HDF5 per process example. Number of files: not stated.
- Resolvability: 3 ms pulse ~6 samples (marginal); 50-300 Hz + harmonics to ~1 kHz yes.
- Use: normal-vs-anomalous machine vibration spectra; scalability of anomaly models across machines.

#### NASA / UC Berkeley BEST Lab Milling Data Set
- Agogino & Goebel (2007). https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip (14.7 MB; mirror index https://data.phmsociety.org/nasa/). Repository statement: users employ the data at their own risk (no formal license).
- Verified by loading mill.mat: 167 runs (16 cases), fields case, run, VB, time, DOC, feed, material, smcAC, smcDC, vib_table, vib_spindle, AE_table, AE_spindle; 9,000 samples/run (one run 15,360) at 250 Hz. Matsuura MC-510V, KC710 inserts, 826 rpm; DOC 0.75/1.5 mm; feed 0.25/0.5 mm/rev; cast iron and steel. AE and vibration channels pass through RMS meters (envelopes). Readme states entry cuts and exit cuts were investigated alongside regular cuts.
- Resolvability: no / no (250 Hz envelopes).
- Use: current-vs-wear envelope curves; quick sanity check for engagement (entry/exit) envelope shapes.

#### QIT-CEMC coated end-mill wear dataset (Sci Data 2024/2025)
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11704274/ ; data DOI 10.6084/m9.figshare.27323346 ; CC BY-NC-ND 4.0 (no derivatives - check before redistribution).
- VDF-850 VMC; 4-flute TiAlN end mill (10 mm) in Ti6Al4V; Kistler 9170B251 rotating dynamometer force X/Y/Z + torque at 10 kHz; Donghua HS922D vibration X/Y/Z 10 kHz; microphone 10 kHz; 68 samples ~5 M rows each; wear labels (VBmax, VB at 1/2 depth, wear area), tool-region classes.
- Use: joint torque + vibration + sound statistics; torque-ripple vs wear.

#### HES-SO SOON milling tool-wear dataset (Zenodo 6505073)
- 2022-04-29; Haute Ecole Arc Ingenierie, HES-SO. https://zenodo.org/records/6505073 (DOI 10.5281/zenodo.6505073); code https://github.com/soon-project/MillingToolWear . License shown as "Open" (specific license not stated).
- One internal AE sensor at 200 kHz; nine accelerometers at 20 kHz (five on spindle axis, three on nearest part axis, sync); six motor currents (5 axes + spindle) at 1 kHz from CNC monitoring. Files: acc 1.1 GB, AE 11.2 GB, axes 1.7 GB.
- Use: multi-rate fusion (AE/vibration/current) reference.

### 2.3 Bone / surgical drilling (analogue for breakthrough)

#### NJMU Bone Drilling Force Dataset (IEEE DataPort, 2024) - SUBSCRIPTION
- https://ieee-dataport.org/documents/njmu-bone-drilling-force-dataset . Real set: 14 drilling paths from 3 bovine bones, 9,162 99x99 images each paired with radial force X/Y, thrust force, torque (sensor-recorded thrust) - 27.8 MB; synthetic set: 600 virtual paths, 985,356 images, 2.85 GB. Sample rate: not stated. Breakthrough labels: not stated. Access: IEEE DataPort subscription.

#### Balgrist / UZH / TUM acoustic drill-breakthrough study (Sci Rep 2021) - DATA NOT SHARED
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7889943/ . Custom piezo contact microphones at 44.1 kHz/24-bit (up to 4 synchronous); 6 human cadaveric hips, 136 drill holes with breakthrough events; two-class labels cortical vs breakthrough. No data-availability statement or repository - would need author contact. Listed because it is the closest labelled analogue to board breakthrough.

#### Southeast University robotic femur drilling (2020, PMC7593762) - REQUEST ONLY
- Hans Robot 6-DOF arm, 2.5 mm bit, 1000/1500 rpm, 110 mm/min; Kistler 9272A four-component dynamometer + vibration sensor; bovine, porcine, Sawbones 3310 femurs. "Available from the corresponding author upon request."

#### Smart bone-screwdriver torque-limit study (2025, HFU / Univ. Basel; PMC12252061) - REQUEST ONLY
- NCTE-2300 rotary torque sensor + 360 CPR encoder; 80 insertions into 8 PU-foam densities; sampling rate not stated. "Dataset available on request from the authors."

### 2.4 Hand-arm vibration / power-tool spectra

#### KIT IPEK: Mechanical impedance and vibration power absorption for varying grip and push forces (RADAR4KIT, 2025)
- https://radar.kit.edu/radar/en/dataset/uwz0qdyzh8h8uekw (DOI 10.35097/uwz0qdyzh8h8uekw). C. Spengler, S. Saurbier, S. Matthiesen (IPEK). CC BY 4.0.
- Shaker excitation, multisine 10-500 Hz in 5 Hz steps at 40 m/s2 RMS; accelerations (top/bottom) and forces Fx/Fy/Fz (top/bottom); grip force 10-135 N and push force 10-110 N in ten levels each; 6 subjects, randomized; raw .mat files (17.1 GB) + result matrices + anthropometrics/max forces.
- Resolvability: steady-state 10-500 Hz band (covers 50-300 Hz + first harmonics).
- Use: hand-arm impedance as the boundary condition for the handle model; grip/push force operating ranges.

#### KIT IPEK: hammer-drill design changes vs posture and muscle stress (RADAR4KIT, 2025)
- https://radar.kit.edu/radar/en/dataset/d072bjn9s8ghkpah (DOI 10.35097/d072bjn9s8ghkpah). Sutschet, Rack, Bengler, Matthiesen. CC BY 4.0. 3 participants, hammer drills with varied handle configurations, 3 trials each; mean joint angles, grip and push force data, EMG RMS of 8 muscles; xlsx (1.8 MB). Summary values, not raw time series.

#### VIBTOOL (Zenodo 21702329, 2026)
- Universidad Politecnica de Madrid. https://zenodo.org/records/21702329 (DOI 10.5281/zenodo.21702329). CC BY 4.0. 19 participants operating 9 handheld vibrating tools (types not enumerated on the landing page; README inside record) for ~30 s each at max power without contacting material; smartwatch accelerometer/gyro/magnetometer/high-frequency accelerometer/linear acceleration/gravity at 100 Hz nominal; 28.1 MB .mat. 100 Hz cannot resolve tool vibration spectra (aliased); useful only for tool-identity features.

#### Lindenmann et al. 2021 (Applied Ergonomics 95:103430) - PAYWALLED, NO DATA
- PubMed 33957304. 15 professional test persons; gripping-force medians 84-156 N (main handle), 10-30 N (auxiliary handle) at feed forces 100/150/200 N; posture effect on handle RMS acceleration; increasing grip force lowers RMS and a_hv. No dataset located (ScienceDirect blocked; no repository found). Useful as a published prior only.

### 2.5 Hammering / impact

#### TU Delft impact-hammer test of an Al 6082-T6 plate (Zenodo 7758683) **[IMPACT-TOP3]**
- 2022-07-09. https://zenodo.org/records/7758683 (DOI 10.5281/zenodo.7758683). CC BY 4.0. PCB 086C02 impact hammer (force in N) + PCB 352A24 accelerometer (g), NI 9234; 5x5 grid = 25 impact points; 32,768 samples over 3.840 s per record (~8.5 kHz); TDMS files + Python plate model; 13.8 MB.
- Resolvability: yes (about 25 samples across a 3 ms pulse) / yes.
- Use: calibrated hammer force-pulse library (rise time, duration, peak) and the impulse-response coupling to a thin structure.

#### Brno impact-echo hammer dataset (figshare 13653284) **[IMPACT-TOP3]**
- 2021-01-28; Brno University of Technology (Materials 2021, 14(3):606). https://doi.org/10.6084/m9.figshare.13653284 . CC BY 4.0. Concrete beam 100x100x400 mm; hammer with piezo sensor in the head and a receiving piezo sensor; blunt (300 mm radius) vs sharp tip, with/without handle, several excitation levels; each row = tip type, handle type, excitation voltage, time array, hammer-sensor signal, receiver signal; 193 kHz; PublicDataset.mat 9.9 MB.
- Caveat: hammer channel is sensor voltage, not calibrated newtons.
- Use: pulse shape vs tip hardness and handle compliance; impact + ring-down for synthesis.

#### RealImpact (Stanford, CVPR 2023) **[IMPACT-TOP3]**
- https://samuelpclarke.com/realimpact/ ; https://github.com/samuel-clarke/RealImpact ; https://arxiv.org/abs/2306.09944 . 150,000 impact-sound recordings of 50 objects, 600 microphone positions per object, audio at 48 kHz; automated striking mechanism using a PCB 086E80 impact hammer with recorded force profiles; impact locations, material labels, RGBD. Force sampling rate: not stated; license: not stated on page/paper.
- Use: force-pulse -> acoustic response mapping across materials; acoustic-emission synthesis reference.

#### NTNU accelerometer + F/T with hammer taps (Zenodo 11096791)
- 2024-05-01; NTNU. https://zenodo.org/records/11096791 (DOI 10.5281/zenodo.11096791). CC BY 4.0. KUKA LBR Med 14, ATI Gamma F/T ~700 Hz, MPU6886 IMU 254 Hz, robot ~100 Hz; three iterations (no disturbance; gentle hammer taps to robot body; taps + manual end-effector force); 12 CSV, 1.4 MB. 700 Hz is marginal for a 3 ms pulse.

#### Sensorized hammerstone for stone knapping (PLOS ONE 2024, Univ. of Wollongong)
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310520 . Honeywell SSCDANN060PGAA5 pressure sensor at 100 kHz (2nd-order Butterworth LP 600 Hz), AMTI force plate at 1 kHz, NI USB-6003; 24 trials x 10 strikes = 240 strikes; forces ~135-730 N; RMSE 38.7 N. Data statement: "All relevant data are within the manuscript and its Supporting Information files." Verified S5 File (pone.0310520.s005.txt): 180 rows of PressurePeak / ForcePeak / DataType only - no time series. STL files of the device are shared.
- Use: human strike-force magnitude prior only.

#### TU/e Impact-Aware Robotics Database and I.AM. box-impact archive
- Database https://www.impact-aware-robotics-database.tue.nl/ ; 4TU collection https://data.4tu.nl/collections/Impact-Aware_Robotics_Datasets/5405187 ; box drops https://data.4tu.nl/articles/dataset/Impact_Aware_Manipulation_I_AM_archive_containing_box_impact_recordings/17122553/1 (DOI 10.4121/17122553.v1, CC BY-NC-SA 4.0, 2.5 GB HDF5: UR10 tossing/dropping boxes on a conveyor, motion capture + F/T sensor; sampling rates not stated). Also Franka Panda impact experiments (10.4121/21655631). Not tool-specific, but the only open robot-impact F/T corpus found.

### 2.6 Human grip force

#### Precision grip control while walking down a stair step (UCLouvain, 2016)
- Zenodo mirror https://zenodo.org/records/4959013 ; Dryad DOI 10.5061/dryad.kf4tr ; paper 10.1371/journal.pone.0165549. CC0. Arsalis grip-lift manipulandum with ATI Mini40 sensors, GF and LF at 1000 Hz; 12 participants x 8 step-down trials; GF peaks ~70 ms after LF during the brisk load increase. 448 kB.
- Use: human GF/LF anticipatory-coupling prior (latency and gain) for the reactive grip layer.

#### Northeastern 3D-printed unconstrained grip-force device (IEEE ToH 2025)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12527031/ ; data/code https://github.com/mahdiaredraki/A-Novel-3D-Printed-Device-for-Unconstrained-Grip-Force-Measurements . 50 kg beam load cell + HX711 at 80 Hz; 7 participants, 30 x 60 s force-tracking trials; license not stated. Low rate; grip-force variability prior only.

(KIT RADAR4KIT hammer-drill and impedance datasets above also carry grip/push forces.)

### 2.7 Vibration transmitted through held tools / high-rate tactile

#### ext-sense - Extended Tactile Perception (NUS, IROS 2021)
- https://github.com/clear-nus/ext-sense ; paper https://arxiv.org/abs/2106.00489 . Raw data (~946 MB) and preprocessed (~696 MB) on Dropbox links in README. NUSkin event-driven multi-taxel sensor at 4 kHz; BioTac micro-vibration (hydrophone) channel (rate not stated in README; paper reports frequency response up to ~1 kHz). Tasks: tap localization along 20/30/50 cm held rods, grasp-stability during handover (rod, box, plate), food identification through a fork. Paper finding: >2 kHz multi-taxel sampling gave the best performance. License: not stated.
- Use: how impacts along a held tool arrive at fingertip sensors; localization target for tool-borne vibration.

#### Penn Haptic Texture Toolkit (HaTT, 2014)
- Record https://repository.upenn.edu/entities/publication/c1b8168b-b8a7-4f22-9e38-6430c233a600 (returned 403 to automated fetch on 2026-09-28); mirrors/pointers https://sites.usc.edu/culbertson/dataset/ , https://www.grasp.upenn.edu/projects/the-penn-haptic-texture-toolkit/ , https://srl.mcgill.ca/hat-box/tools/penn-haptic-texture-toolkit/ (license: "Non-Commercial Research Use Only"). 100 textures; 10 s recordings of force, speed and high-frequency acceleration of a handheld tool at 10 kHz, plus models and rendering code.
- Use: accelerometer-on-tool vibration realism and friction/texture models.

#### LMT Haptic Texture Database (TUM, 2014) - SERVER UNAVAILABLE AT CHECK
- Info page https://www.ce.cit.tum.de/en/lmt/forschung/datensaetze/texture-database/ ; download https://zeus.lmt.ei.tum.de/downloads/texture/ returned HTTP 503 (twice, 2026-09-28). 108-texture "old" database + 69-texture set; stylus accelerometer during controlled and free-hand scans; sampling rate: not verifiable today.

#### Cluster Haptic Texture Dataset (Sci Data 2026)
- https://doi.org/10.6084/m9.figshare.29438288 (CC BY 4.0; full 15.2 GB, mini 760 MB); code https://github.com/cluster-lab/Cluster-Haptic-Texture-Dataset ; paper https://arxiv.org/abs/2407.16206 . 118 textures, 9 categories, 5 velocities (20-60 mm/s) x 8 directions, 18,880 synchronized recordings of audio, acceleration, force, position, video. Sampling rates: not stated on landing page.

#### Vibro-Sense (Leibniz Univ. Hannover / L3S, 2026)
- https://arxiv.org/abs/2601.20555 ; project https://wzaielamri.github.io/publication/vibrosense ; data https://huggingface.co/datasets/wzaielamri/vibrosense ; code https://github.com/wzaielamri/vibrosense . Seven low-cost piezo microphones on a robotic hand; raw 50 kHz, analysed at 20 kHz; >20,000 interactions per indenter; localization <5 mm. License: not stated.

### 2.8 Motor current

#### F.A.I.R. open dataset of brushed DC-motor faults (CARTIF, Zenodo 4314249)
- 2020-12-10. https://zenodo.org/records/4314249 . CC BY 4.0. Inexpensive brushed DC motors run out of nominal conditions to failure (30 min-6 h); vibration (g), current (A), voltage, surface/ambient temperature, noise, all at 51.2 kHz; HDF5 482.8 MB + processing sheet. Load conditions: "beyond nominal" (not quantified on page).
- Use: commutation-ripple and current/vibration co-signatures of a brushed motor (cordless-tool analogue).

#### Current signature dataset of three-phase induction motor under varying load (Mendeley 2022)
- https://data.mendeley.com/datasets/gxdd74czwh/1 (DOI 10.17632/gxdd74czwh.1). MUET Jamshoro. CC BY 4.0. 3-phase currents at 10 kHz, 1,000 samples/channel per record; 39 records; loads 100/200/300 W; healthy, bearing inner/outer race (0.7-1.7 mm), broken rotor bar. Not a tool motor.

#### BLDC hub-motor fault-detection dataset (TalTech, IEEE DataPort 2026) - SUBSCRIPTION
- https://ieee-dataport.org/documents/bldc-hub-motor-fault-detection-dataset-multi-rate-sampling-phase-current-measurements . 3-phase currents (TA018 hall sensors), voltage, torque, speed at 100 kHz plus copies at 50 k/25 k/10 k/5 k/2 k/1 k/500/250 Hz; 4 operating points 200-350 rpm, 5-10 Nm; 144 files. Subscription required.

---

## 3. Datasets / sources checked and rejected (with reasons)

| Source | Why rejected |
|---|---|
| CMU screwdriving dataset (Aronson et al. 2016, 1,862 runs) | Paper only; no data-availability statement or repository; lab site offline. Keep as request-only lead. |
| PHM 2010 official download links (cdn.teamholistic.net c1-c6.zip) | Domain parked; links return an HTML redirect, not data. Use Kaggle/IEEE DataPort mirrors. |
| IEEE DataPort "PHM2010 dataset" (10.21227/bh7t-qe79) and "2010 PHM Society Conference Data Challenge" (10.21227/jdxd-yy51) | Subscription; pages showed no uploaded files at check. |
| UniWear (katulu-io) | Derivative merge of NUAA + PHM2010 aggregated to ~2-25 Hz; no raw signals. |
| Kaggle "CNC Mill Tool Wear" (Univ. of Michigan SMART lab) | 100 ms (10 Hz) controller data on wax blocks; cannot resolve tool physics. |
| SIMTech/NUS gun-drilling dataset (Inconel 718, 20 tools, force/torque/12 vibration channels) | Described in arXiv 1804.10801 / 1805.00367 but no public download found. |
| Sci Rep 2026 drill-bit condition monitoring (XTRON-544 VMC, spindle accelerometer 20 kHz, 7 wear classes, 6,657 samples) | "Available from the corresponding author on reasonable request." |
| Fraunhofer IWU multi-sensory tool holder chatter tests (10 kHz bending moment/axial force/accel; 4 stable + 4 chatter tests) | "Data are contained within the article"; no raw release. |
| Balgrist/UZH/TUM acoustic drill-breakthrough (44.1 kHz, 136 holes, cortical/breakthrough labels) | No data-availability statement or repository. Best labelled breakthrough analogue if authors will share. |
| Southeast Univ. bovine/porcine/artificial femur drilling forces (Kistler 9272A) | Request only. |
| MDPI Eng 2024 "Bone drilling ... bone layer classification using vibration signal" | Publisher blocked automated access (403); could not verify data statement. |
| Human-inspired robotic craniotomy framework (arXiv 2607.21058) | Multi-sensor bone-milling with video ground truth, 100 Hz control loop; no dataset release found. |
| Smart bone screwdriver torque-limit algorithm (HFU/Basel 2025) | "Dataset available on request." |
| NJMU Bone Drilling Force Dataset | Kept in table but flagged: IEEE DataPort subscription; sample rate and breakthrough labels not stated; force paired to images rather than plain time series. |
| Lindenmann et al. 2021 hammer-drill grip force (KIT) | Paywalled; no dataset; abstract values usable as priors only. |
| KIT manual angle-grinder tool-force prediction (Sensors 2021; 20 kHz Kistler forces, 1 kHz IMU/current/voltage, 64 tests) | "Available from the corresponding author upon reasonable request." |
| SDSU wearable hand-arm-vibration monitoring (arXiv 2509.16536; 4 subjects hammer-drilling concrete) | No dataset release; consumer IMUs 200 Hz with 256 Hz internal filter. |
| NIOSH Power Tools (Sound & Vibration) Database | Live site https://wwwn.cdc.gov/niosh-sound-vibration/ returned 404 on 2026-09-28; only web-archive copies and a CDC noise-summary xlsx (returned 403) remain; vibration entries were summary levels, not time series. |
| Swedish National Vibration Database (vibration.db.umu.se) | JavaScript application, content not retrievable; holds tool-level a_hv summary values per ISO 5349, not raw signals. |
| VIBTOOL | Kept in table but 100 Hz wrist IMU cannot give tool spectra; tool identity only. |
| ETH Zurich / PoliMi lab-scale structure impact tests (Zenodo 15516419) | Impact-hammer excitation but acceleration sampled at 100 Hz - cannot resolve pulses. |
| Sensorized hammerstone knapping (Wollongong 2024) | Kept for peak-force prior only: shared S5 file holds 180 peak pairs, not the 100 kHz traces. |
| "Knapping force as a function of stone heat treatment" (PLOS ONE 2022) | Indentation tests, no strikes; all data in tables. |
| Instrumented surgical hammer / acetabular cup impaction (Sensors 2018; 51.2 kHz strain sensors) and Summers-osteotomy mallet (arXiv 2511.10126) | No data statements or repositories. |
| "Effect of hammer mass on upper-extremity joint moments" (Applied Ergonomics 2016; mocap + force plate hammering) | Paywalled; no data release found. |
| RH20T | General manipulation dataset; no hammering/tool-physics tasks verified; F/T rate not stated in paper (images 10 Hz). Out of scope. |
| Minari/D4RL Adroit "hammer" | Simulation only. |
| Junggy/HAMMER-dataset | Depth-sensing/3D reconstruction dataset; name collision only. |
| REASSEMBLE for threading | Kept for insertion transients, but the NIST-board nuts (M4/M8/M12) were excluded, so no nut-threading signals. |
| Sound of Touch (arXiv 2602.16846; 44.1 kHz string acoustics) and MicCheck (arXiv 2511.18299; 48 kHz pin mics) | Methods papers; no dataset release found. |
| "SpectRobot" | No such dataset found by search. |
| Aircraft fastener assembly failure prediction (arXiv 2505.03917, aeronautical collars) | No public data release found. |
| ARTiS adaptive gripper screwdriver torque data (arXiv 2609.03362) | F/T torque during tightening/unscrewing described; no dataset release found. |
| Kaggle "screw dataset" (ruruamour) | Images (normal/anomaly), not signals. |
| Kaggle "BLDC Motor Dataset" (ziya07) | Unverified provenance (likely synthetic); not checked further. |
| IEEE DataPort "Roughness of Milling Process" | Spindle accelerometer for roughness; off-topic. |
| Mendeley/Zenodo/figshare API sweeps | Zenodo REST API returned 403 ("unusual traffic") for this network; figshare search endpoint returned unrelated newest items regardless of query; Mendeley search API 400. Coverage of those repositories therefore relied on web search + direct record fetches. |

---

## 4. Gaps I could not fill or verify

1. **No open dataset of human/robot screwdriving with a hand-held power drill/driver** (clutch, cam-out, bit-slip, motor current at >= 1 kHz). PyScrew is a fixed industrial spindle at 833 Hz without current or vibration; AURSAD is 100 Hz. Cam-out exists only as PyScrew's "material in the screw head -> driver slippage" class.
2. **No open drilling dataset with thrust/torque/vibration and breakthrough labels** (wood, metal, or bone). All bone-breakthrough sources are unpublished or request-only; NJMU is subscription-only and its labels/rate are not stated. Drilling-specific open sets (gun drilling, drill-bit wear at 20 kHz) are not downloadable.
3. **No open time-series hand-arm-vibration recordings of hammer drills / impact drivers / demolition or riveting hammers.** Only summary databases (NIOSH - offline; Swedish DB - JS app) and IMU-at-100 Hz sets exist. Closest usable physics: KIT shaker-based hand-arm impedance (10-500 Hz) and Lindenmann's published grip/feed-force values.
4. **No open human hammering biomechanics dataset with strike force + grip force + kinematics.** Knapping study shares peaks only; hammering mocap/EMG studies are paywalled or unshared. Hammer-pulse physics must come from instrumented modal hammers (TU Delft, Brno, RealImpact, NTNU).
5. **No open grip-force-during-power-tool-use time series.** KIT RADAR4KIT hammer-drill data are means; the impedance dataset is shaker-based; GF/LF coupling comes from a step-down manipulandum task.
6. **Motor-current datasets are not power-tool motors** (brushed toy DC motors, an induction motor, a subscription-only e-scooter BLDC). No cordless-drill BLDC current-vs-load dataset was found.
7. **Unverified items:** LMT texture database sampling rate and availability (server 503); HaTT record contents at the UPenn repository (403 to automated fetch; mirrors confirm it exists); Cluster Haptic Texture Dataset sampling rates (not on landing page); RealImpact force sampling rate and license; I.AM./TU-e F/T rates; VIBTOOL tool list (inside README); ext-sense license; exact Bosch CNC file count.
8. **Access caveats to re-check before relying on them:** PHM2010 (mirrors only), IEEE DataPort items (subscription), QIT-CEMC (CC BY-NC-ND forbids derivatives), Tool Tracking (CC BY-NC-SA), I.AM. archive (CC BY-NC-SA).

---

## Appendix: quick picks by project need

- Drill-plant torque/angle regime curves and screw-fault classifier: PyScrew (primary), AURSAD (controller-rate view), CMU (if obtainable).
- Chatter / bind detector: Khasawneh turning set (labelled); PHM2010 + MU-TCM for stable-cut vibration statistics; Bosch CNC for machine-level anomalies.
- Motor-current vs load/wear: Rzeszow (0.5 kHz current + 25 kHz vibration), HES-SO (1 kHz current + 20 kHz accel + 200 kHz AE), NASA (250 Hz envelopes), F.A.I.R. brushed DC motor (51.2 kHz).
- Vibration synthesis spectra: PHM2010 / MU-TCM / QIT-CEMC (rotating cutter), Khasawneh (chatter), HaTT / Cluster (tool-surface), ext-sense and Vibro-Sense (tool -> finger transmission).
- Hammer pulse physics: TU Delft (calibrated N, ~8.5 kHz), Brno (193 kHz, tip hardness), RealImpact (force + 48 kHz sound), NTNU (F/T view of taps), knapping (human peak-force range 135-730 N).
- Human grip priors: KIT impedance dataset (grip 10-135 N, push 10-110 N), Lindenmann 2021 (main handle 84-156 N, auxiliary 10-30 N, feed 100-200 N), UCLouvain GF/LF coupling (~70 ms), KIT hammer-drill posture set (means).
