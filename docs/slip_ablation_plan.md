# Slip ablation: is fast reactive tactile control the right lever?

Status (2026-10-06): built. The screen is running on the desktop (`mleggiero-desk`: Ryzen 7 7700, 16 threads, 30 GB,
MuJoCo 3.14.0). Branch `exp/slip-ablation`; nothing committed yet.

## Question

What keeps a held tool from turning or sliding in the hand, for a given grip effort? The candidate levers:

- a fast reactive grip loop driven by 1 kHz tactile data;
- the anticipatory grip ramp;
- passive compliance between the flange and the hand;
- the arm controller's rotational stiffness;
- where on the handle the hand grips.

## Hypotheses (from the first-order model in the design discussion)

- **H1 reactive.** At equal mean grip, a 1 kHz reactive grip loop barely changes per-strike hammer tilt, even with
  oracle slip detection: the blow lasts ~4 ms, shorter than the hand's force response.
- **H2 wrist compliance.** Passive compliance between flange and hand cuts per-strike tilt substantially. The
  first-order model predicts the peak grasp moment falls from ~51 N m (rigid) to ~10 N m (200 N m/rad) at 1.1 m/s.
- **H3 active stiffness.** The arm controller's rotational stiffness barely matters within the blow.
- **H4 grip location.** Gripping near the hammer's centre-of-percussion point (the default 190 mm) gives less slip
  than choking up.
- **H5 positive control.** On slow saw and driver loads the same reactive loop does reduce slip, so a null hammer
  result is not a broken controller.

## Scope (decided 2026-10-06)

- **Hammer** on all three platforms:
  - FR3 + WUJI Hand 2 (backdrivable);
  - FR3 + Franka Hand;
  - Vega U + WUJI Hand 2.
- **Saw, driver driving a screw, driver drilling a hole** on FR3 + Franka Hand, the only hand with saw and driver
  grasps today. WUJI tool grasps need their own synthesis; that is a follow-up, and a chance to test form-closure
  grasps (fingers through the saw's D-handle).
- **Reactive variants:** oracle (an upper bound) and the TaxelScan / pad-array detector. Hand actuation stays as
  modelled: 1 kHz joint torque on the WUJI (optimistic), so reactive results are an upper bound for the real hand.
- **Runs:** a screen first (randomization off, seed 0), then the full grid (domain randomization on, seeds 0-4).

## Conditions (one factor at a time from a common baseline, paired by seed)

### Baseline B

- Constant grip at the hand's hold setpoint: WUJI 40 N summed patch force, Franka Hand 55 N per pad (60 N for the
  saw and driver).
- Anticipatory ramp off.
- Cross-strike grip-margin adaptation off (`grip_margin_step = 0`). The default swing adds 10 N after every strike
  that slips, which would confound every comparison.
- Default wrist: 1500 N m/rad on the FR3, rigid on the Vega.
- Default controller stiffness; grip 190 mm from the head axis.

### Factors

| Factor | Hammer levels | Saw / driver levels |
|---|---|---|
| F fixed grip (the frontier) | WUJI 20, 30, 50, 60 N; Franka 25, 40, 70 N | 30, 45, 70 N |
| A anticipatory ramp | pre-load gain 6, 12, 24 N/(kg m/s) | - |
| R oracle / taxel reactive | each at the low and baseline grip | each at 30 and 60 N |
| W wrist (passive, flange to hand) | FR3: rigid, 400, 100, 30; Vega: 1500, 400, 100, 30 N m/rad | rigid, 400, 100 |
| K controller rotational stiffness | FR3: 30, 1000 N m/rad (nominal and strike); Vega: servo P x0.25, x2 | 30, 1000 |
| L grip location | 130, 160, 220 mm from the head axis | - |
| adapt | baseline + the existing +10 N cross-strike margin | - |
| combinations | A12 + oracle, A12 + W100 | - |

The wrist keeps the default's damping ratio. A soft wrist (k < 1500) gets +/-0.6 rad of travel instead of +/-0.1.

Screen: 110 episodes. Full grid: 550.

## Reactive loop (`tactile_sim/control/reactive_slip.py`, 1 kHz)

The response to a detection is the hand's maximum grip, at once, bypassing the 500 Hz grip loop:

- 140 N per pad on the Franka Hand;
- full synergy torque on the WUJI.

The boost holds for 200 ms after the last detection, then ramps back over 200 ms. Fast detectors (impacts, jerks)
need 2 consecutive ticks over threshold. Slow detectors (creep) pass two cascaded 0.5 s low-passes, and their
references reset at every detection.

| Detector | Fast | Slow |
|---|---|---|
| oracle (simulation truth) | tool angular / linear speed in the hand | low-passed tool pose drift |
| taxel (the hand's own sensors) | centre-of-pressure speed, patch load drop, pad accelerometer jolt | low-passed CoP drift, rotation of the pressure pattern |

### Calibration (log mode, seed 0)

Each threshold sits just above the largest excursion seen without slip. Hammer: 6 strikes; tools: one episode.

| Signal | Franka hammer | Franka saw / driver | WUJI hammer (FR3, Vega) |
|---|---|---|---|
| oracle rate, swing peak -> impact peak | 0.5 -> 9.8 rad/s | strokes 1.9 rad/s; clutch 37 rad/s | 1.2 / 0.19 -> 11.6 / 6.8 rad/s |
| oracle rate threshold | 2.5 rad/s, 0.08 m/s | 2.5 rad/s, 0.08 m/s | 1.5 rad/s, 0.04 m/s |
| CoP speed swing peak -> impact | 1.5 -> 3.2 m/s | 1.2 m/s | 0.27 -> 2.8 m/s |
| CoP speed threshold | 1.8 m/s | 1.8 m/s | 0.35 m/s |
| accel swing peak -> impact | 114 -> 310 m/s^2 | 41 (saw), 266 (clutch) | 20 -> 247 m/s^2 |
| accel threshold | 140 m/s^2 | 140 m/s^2 | 30 m/s^2 |
| patch load drop | swing 16 % | 10 % | **swing 99 %: cue off** |
| pattern rotation without slip | swing 1.2 deg | stroke ripple 0.3 deg | swing 1.0-1.6 deg |
| pattern threshold | 1.5 deg | 0.5 deg | 2.0 deg |

### Findings from the calibration

1. **Load drop is useless on the WUJI wrap.** Its taxel patches lose nearly all load in every swing: the windup's
   deceleration moves the handle onto the bare fingers.
2. **A rate threshold cannot see the saw's creep.** The handle creeps 0.18 deg/s while the pads flex at up to
   1.9 rad/s each stroke, and a threshold set above the flexing misses the creep entirely.
3. **The centre of pressure does not follow the saw's creep either.** Low-passed, it ripples 1.9 mm and does not
   trend, because the handle turns about the pad normal, which barely moves the CoP.
4. **The pressure pattern's orientation does see the creep, weakly.** It trends 0.04 deg/s against the true
   0.18 deg/s, with 0.3 deg ripple. That is the cue a normal-pressure array gives for rotational creep.
5. **The oracle catches hammer impacts during the blow.** It first crosses 1 rad/s 0.4-2 ms after contact.

## First screen attempt (v0) and the fixes it forced

The first screen was stopped after 21 episodes. Its results are kept in `runs/slip_ablation/old/screen_v0`. It found
three problems in the set-up, not the levers:

1. **A reactive step to full grip reshapes the WUJI wrap.** Full synergy torque folds the fingers and turns the
   hammer: 14 deg in one swing, when a detection fired 124 ms before contact.
   - The re-seating then drifts the pressure centre. The slow detector reads that as slip and boosts again, so the
     grip stays up for whole strikes.
   - Vega U + WUJI, TaxelScan reactive at 40 N: 20 deg tilt per strike against 1.8 deg at baseline.
   - **Fix:** the boost goes to the hand's pre-impact ceiling, the anticipatory ramp's own cap (`boost_to = "pre"`),
     so reaction and anticipation reach the same squeeze and differ only in timing. The slow detectors are blind
     while the boost acts and re-reference once it has released (`ReactiveCfg.boost_to`, blanking in
     `ReactiveSlip.tick`).
2. **A soft wrist servoed through its own spring breaks the task.** On the Vega U the hammer dropped in the first
   swing at 30 N m/rad and slid 8-12 mm per strike at 100 N m/rad. The arm controller servoed the TCP on the far side
   of the wrist hinges.
   - **Fix:** `ArmCfg.wrist_control = "flange"`. The controller servoes the TCP's zero-deflection pose, carried by the
     F/T body, so the compliance is a passive element after the flange, as a real compliant wrist is. Every condition
     of the experiment uses it, so all factors share one control frame.
3. **"Grip" summed every normal force on the taxel patches, impact slaps included** (peaks of 450 N on the WUJI
   palm). That is contact load, not effort.
   - **Fix:** `HandIO.command()`, the grip the hand is commanded to apply (Franka Hand: drive force per pad; WUJI:
     synergy scalar x full-synergy force). Its mean is the frontier's axis (`effort_mean`); the contact load stays as
     a secondary metric.

Early v0 numbers, Vega U + WUJI, seed 0 (directional only):

| Condition | Tilt per strike | Peak grasp pitch moment |
|---|---|---|
| Fixed grip 20 / 30 / 40 / 50 N | 2.8 / 2.2 / 1.8 / 1.6 deg | |
| Anticipatory ramp | 1.1-1.4 deg | |
| Oracle reactive at 40 N | 1.5 deg | |
| Wrist 1500 N m/rad | 1.1 deg | 17.6 N m |
| Rigid wrist (B) | 1.8 deg | 38.4 N m |

## Screen v1 results (seed 0, no randomization; directional)

110 episodes, 0 failed, 28 min on 13 workers. Report: `runs/slip_ablation/screen/report/`.

- **FR3 + Franka Hand, hammer.** The fixed-grip frontier is monotonic: 25 / 40 / 55 / 70 N give 6.3 / 4.2 / 3.1 / 1.9
  deg per strike.
  - Anticipation lands below it: 1.2-2.2 deg at 60-66 N.
  - The 30 N m/rad wrist gives 2.0 deg at the baseline effort, against 3.1.
  - Reactive loops give the lowest tilt (0.7-1.3 deg), but at 85-104 N mean effort.
  - Stiffer control is worse (1000 N m/rad: 4.8 deg).
- **Vega U + WUJI, hammer.**
  - Anticipation sits below the frontier (1.1 deg against ~1.5).
  - The 100 / 30 N m/rad wrists give 0.45 / 0.57 deg, and the peak grasp moment falls 38 -> 13 N m, but at about
    twice the effort (the closed grip loop squeezes harder once the patches read less load).
  - The oracle reflex at 40 N gives 1.1 deg. The TaxelScan reflex makes things worse (4-10 deg).
  - Choking up (130 mm) is worse.
- **FR3 + WUJI, hammer.** Chaotic: 5-19 deg per strike whatever the lever, and a non-monotonic frontier. The wrap is
  close to failing.
- **Saw and driver.**
  - At 30 N per pad the fixed grip lost the tool on all three tasks; reactive loops kept it in hand.
  - Short reflex boosts don't reduce saw creep at 60 N.
  - On the screw, the reflex cut the clutch-seating twist (0.5 vs 2.3 deg, below the frontier).

### Why the full grid differs from the screen

1. **The sim's grip actuation is instant.** A grip command reaches the pads and finger joints in ~1-2 ms, so a
   reflex can act inside the 4 ms blow. That is what makes reactive work on the Franka hammer, and it is exactly the
   assumption H1 disputes. The full grid adds a first-order grip lag (`GripperCfg.grip_lag`) of 0 (the ideal upper
   bound), 10 and 30 ms on the reactive and anticipatory conditions.
2. **The WUJI's regulated grip loop confounds effort.** It regulates the summed taxel-patch force, and that sum
   includes the hammer's weight and swing loads on the palm. The full grid runs every WUJI platform both ways:
   `fr3_wuji` / `vega_wuji` regulated, `fr3_wuji_open` / `vega_wuji_open` with a constant commanded squeeze
   (`ControllerCfg.grip_loop = "open"`).
3. **A 200 ms reflex can't fix creep.** The full grid adds a latched variant (`ReactiveCfg.latch`): slip raises the
   grip, which then decays over ~2 s, a raised safety margin like a hand's after a slip.
4. **Housekeeping.** Saw and driver episodes flag a lost tool (`dropped` / `lost`), and the report excludes those
   from slip and effort means. The TaxelScan skin model skips rays that cannot reach a handle slice (a bounding-sphere
   cut; the taxel load shapes are bit-identical, see the test).

### Full grid

Seeds:

| Platform | Grip loop | Seeds |
|---|---|---|
| `fr3_wuji` | regulated | 10 |
| `fr3_wuji_open` | open (commanded squeeze) | 10 |
| `fr3_franka` | regulated | 5 |
| `vega_wuji` | regulated | 5 |
| `vega_wuji_open` | open (commanded squeeze) | 5 |

Hammer conditions per platform:

- B, the fixed-grip levels;
- A6, A24, and A12 at 0 / 10 / 30 ms lag;
- reflex (R) and latched (RL) loops, each with the oracle and taxel detectors: at the baseline grip for 0 / 10 / 30 ms
  lag, at the low grip without lag;
- wrist (W x4), stiffness (K or P x2), grip location (L x3), adapt;
- A12 + oracle reflex at 0 and 30 ms lag; A12 + W100.

Saw / screw / hole: B, F x3, R and RL as above, W x3, K x2.

## Metrics

Per strike, strikes 2-10 (strike 1 is the setting tap):

- **Primary:** tilt of the tool in the hand per strike (swing start to strike end).
- **Secondary:**
  - slide per strike;
  - tilt in the 100 ms after contact (includes elastic deflection that recovers);
  - grip effort (mean commanded grip; the frontier's axis) and contact load (summed patch / pad normal force);
  - nail advance;
  - aim offset at contact;
  - impact-flag latency;
  - drops;
  - hand hard-stop load;
  - arm-limit ratios.
- **Mechanism:** peak grasp wrench in the 60 ms after contact, the moment split into pitch (swing plane), twist
  (about the handle) and yaw.
- **Saw and driver:** net and peak tool tilt, net slide, mean grip, time to goal, reactive detections.

## Analysis (`tactile_sim/experiments/slip_report.py`)

- The tables and the slip-against-mean-grip plot are the decisive output. The fixed-grip levels trace a frontier,
  and a lever helps only if it lands below it.
- Full stage: each condition minus B, paired by seed, with a bootstrap 95% interval over seeds.

## Reproduction notes

- **MuJoCo pinned at 3.14.0**, the repo's development version. 3.15.0 fails `test_hold_is_stable` by a hair.
- **FR3 + Franka Hand reproduces the README:** 13.6 mm vs 13.8 mm in 10 strikes, tilt 0.6-2.9 deg.
- **FR3 + WUJI does not match the README's row** (7.8 vs 12.3 mm, tilt 3-16 deg). That row predates commit
  7cdf3f3 (Oct 4), which made the TaxelScan skins the WUJI default; the runs here use the current code.

## Code changes

All new knobs default to the old behaviour, so existing results are unchanged.

- `config.py`:
  - `SwingCfg.grip_profile` (`template` | `constant`);
  - `SwingCfg.strike_k_rot` (was hard-coded 250);
  - `ArmCfg.wrist_range` (was hard-coded +/-0.1 rad);
  - `ArmCfg.wrist_control` (`tcp` | `flange`);
  - `GripperCfg.grip_lag` (first-order grip actuation lag);
  - `ControllerCfg.grip_loop` (`closed` | `open`);
  - new `ReactiveCfg` (incl. `boost_to`, `latch`, `latch_tau`) and `SimConfig.reactive`.
- `sim/hands.py`: the grip lag (`HandIO.actuate`, stepped by `World.step`) and an exact bounding-sphere ray cut in the
  TaxelScan load model.
- `tool_task.py`: `dropped` / `lost` flags.
- `sim/world.py`: `ctrl_pose` / `ctrl_jacobian`, the frame the arm servoes. `arm_state`, both L1s, the swing and the
  tool behaviours use it; slip and aim are still measured on the hand's TCP.
- `sim/hands.py`: `HandIO.command()`, the commanded grip (effort).
- `control/reactive_slip.py`: the reactive loop.
- `sim/testbed.py`: a grip floor (`grip_boost`); the reactive loop scheduled between the grip loop and the hand.
- `sim/truth.py`: `contact_wrench` (checked: the hand carries the hammer's weight to 0.1 %).
- `episode.py`: per-strike `extra` metrics (slip by window, grip mean, grasp-wrench peaks, aim, reactive detections).
- `tool_task.py`: reactive counts in the summary.
- `experiments/slip_ablation.py` (runner) and `experiments/slip_report.py` (report).
- `tests/test_reactive_slip.py`: 16 tests.

## Running

    python -m tactile_sim.experiments.slip_ablation list --stage screen
    python -m tactile_sim.experiments.slip_ablation run --stage screen --workers 13
    python -m tactile_sim.experiments.slip_report --stage screen
    python -m tactile_sim.experiments.slip_ablation run --stage full --workers 13

Results land in `runs/slip_ablation/<stage>/<platform>/<task>/<condition>/seed<k>.json`. A run resumes where it
stopped.

## Risks

- **Contact at the 4 ms timescale.** MuJoCo's soft contact is the weakest part of the sim exactly here. Report
  relative comparisons, and sweep contact parameters on the winning condition.
- **WUJI actuation is optimistic.** The sim drives the hand with 1 kHz joint torque; the real Hand 2 takes position
  targets with a current limit. Reactive results are an upper bound.
- **Side effects of a soft wrist.** It can hurt aim and delay F/T impact detection (the F/T sits before the
  compliance). Hits, aim offset and flag latency are logged.
