# tactile-encode

Research and simulation for a tactile, force-aware control stack for dynamic tool use
(hammering, power drilling). The research lives in `reports/`, `research_notes/` and
`docs/control_flowchart.html`. The simulation plan is `docs/simulation_plan.md`.

`tactile_sim` is a headless MuJoCo simulation of the testbed for **Task A: a hammer driving a
pre-started nail**. It provides the plant, the sensor models and the 1 kHz L1 controller that the
learned layers (L2 reactive, L3 World-Action-Model / VLA planner) will train against.

- **Robot:** the Franka FR3 arm from MuJoCo Menagerie at a pinned commit, with torque-controlled joints and a
  wrist F/T sensor body, and one of two hands:
  - the **Franka Hand** (Menagerie; the default), with compliant rubber layers on its real 17 x 17 mm pads;
  - the **WUJI Hand 2** (WUJI's published MJCF, Beta 2, MIT license) holding the hammer in a power wrap. Its
    20 joints are torque motors limited to WUJI's per-joint ratings and run a 1 kHz joint law (MIT mode).
- **Tool:** the YCB 048_hammer scan (steel claw hammer, wooden handle, 665 g, CC BY 4.0), gripped
  190 mm from its head on the flat, widest part of the handle. The full scan is drawn; contacts use convex
  hulls cut from the scan (gripped handle section, striking face, whole head). Offline, a primitive
  hammer with the same masses is used.
- **Plant:** a nail in a vertical board. The nail resists with Coulomb friction that grows with depth.
  The hammer-nail contact gives a ~4 ms blow at ~500-730 N.
- **Sensors:** wrist F/T at 4 kHz, accelerometers at 8 kHz (+-16 g) under each taxel patch, 8 x 8 pressure
  arrays (64 taxels) at 1 kHz, joint encoders and torques at 1 kHz, and the momentum observer's external
  torque. Each has its own band-limit, latency, noise, bias, quantization and saturation. Contact forces
  are spread over the taxels with a 3 mm kernel, the way a rubber layer spreads load. The Franka Hand has a
  patch on each pad; the WUJI hand has two (128 taxels), placed where the wrap loads it: the palm and the
  thumb segment that carries the handle.
- **Control:** a 1 kHz Cartesian impedance law with a reference limiter, stiffness slew limit, payload
  compensation and 50 ms velocity gating after impact; momentum observer and three-channel impact
  detector; reference spreading around the strike; 500 Hz grasp-force loop with drop detection.
- **Behavior:** a scripted swing that stands in for L2/L3, with a human-template grip profile (ramp
  150 ms before contact, peak 60 ms after), a setting tap, and iterative re-aiming between strikes. The
  swing is planned inside the arm's limits (below).
- **Hardware limits** are enforced where the real system enforces them and monitored everywhere else
  (`tactile_sim/limits.py`); every episode reports its worst ratio per limit and the tests fail when one
  is exceeded.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .[dev,gym]
make assets        # fetch the pinned Menagerie FR3 + Franka Hand, the WUJI Hand 2 and the YCB hammer into ~/.cache
make test          # full suite, ~2 min
```

Without the caches everything still runs on stand-ins with identical kinematics and inertias
(capsule arm, box hand, primitive hammer); the WUJI hand needs its cache. Set `TACTILE_SIM_ASSETS=/path` to
use a different asset cache.

## Run

```bash
python -m tactile_sim.run_strikes --n 10 --out runs/demo.h5          # 8 kHz episode + HDF5 force-truth log
python -m tactile_sim.run_strikes --n 10 --fast --dr --seed 3         # 4 kHz, domain randomized
python -m tactile_sim.run_strikes --n 10 --hand wuji2                 # WUJI Hand 2 power wrap
python -m tactile_sim.run_strikes --n 10 --hand wuji2 --self-locking  # ... with non-backdrivable joints
python -m tactile_sim.calibrate pulse                                 # free-hammer contact sweeps
python -m tactile_sim.viewer.export --n 6 --preset default --preset wuji2 --preset wuji2-selflock
python -m tactile_sim.viewer.live --n 5 --slowdown 20                 # native MuJoCo window (needs a display)
```

`runs/replay.html` is a standalone 3D replay (three.js, loaded from a CDN) of the exported episodes with
the real robot meshes, per-strike records, time-aligned traces, a live pressure heatmap per taxel patch and
the episode's hardware-limit verdict. `gymnasium.make("TactileHammer-v0")`
exposes the L2 -> L1 interface (TCP offset, stiffness scale, strike-axis feedforward, grasp force) at 200 Hz.

On a machine with an NVIDIA GPU, set `MUJOCO_GL=egl` before starting Python to enable the optional
cameras (`sensors.cameras = True`).

## Hardware limits

| Limit | Value | How |
|---|---|---|
| FR3 joint torque | 87 Nm (joints 1-4), 12 Nm (5-7) | enforced: command clip |
| FR3 torque rate | 1000 Nm/s per joint (libfranka `kMaxTorqueRate`) | enforced: L1 rate-limits every command |
| FR3 joint velocity | 2.62 rad/s (1-4), 5.26 / 4.18 / 5.26 rad/s (5-7) | monitored (a velocity reflex on the real arm) |
| FR3 joint range | Menagerie joint ranges | monitored |
| FR3 payload | 3 kg beyond the flange | checked |
| Franka Hand | 140 N peak, 70 N continuous per finger | enforced |
| WUJI Hand 2 joint torque | MCP flexion 2.0, abduction 0.2, PIP/DIP 0.3, thumb CMC 0.6, thumb MCP/IP 0.3 Nm | enforced |
| WUJI hard stops / gearbox | torque carried by the stops (self-locking: the gearbox), relative to the joint's rating | monitored; WUJI publishes no stop rating, so the joint rating is used |

The swing is planned to fit: strike speed is capped at 80 % of what the joint velocity limits allow along
the strike axis at the hover pose, swing acceleration at 60 % of what the joint torque limits can push
there, and the swing peaks just before the nail and arrives braking. A 1000 Nm/s torque-rate limit needs
~170 ms to reverse a saturated joint, so a swing still accelerating at contact keeps driving the arm into
the nail and the tool through the grasp. Each hand has a hover pose chosen for these caps.

## Results (8 kHz, seed 0, 10 strikes)

| Metric | Franka Hand | WUJI Hand 2 | WUJI Hand 2, self-locking | Report target |
|---|---|---|---|---|
| Strikes on the nail | 10 / 10 | 9 / 10, hammer lost on the 10th | 10 / 10 | >= 90 % |
| Nail driven | 13.6 mm (0.9-1.5 mm per blow) | 10.8 mm in 9 | 15.4 mm | 20 mm in <= 10 strikes |
| Face speed at contact (after the 1.2 m/s tap) | 1.7-1.8 m/s | 1.75-1.9 m/s | 1.7-1.8 m/s | |
| Impact flag after contact | 0.4-1.1 ms | 0.5-1.5 ms | 0.25-1.0 ms | <= 2 ms |
| Tool tilt in the grasp per strike | 0.6-2.9 deg (over 1.4 deg on 3) | 2.9-16.6 deg | 0.1-0.5 deg | <= 1.4 deg |
| Tool slip per strike | 0.5-1.1 mm | 2.2-7.0 mm | 0.2-1.0 mm | <= 7 mm |
| FR3 limits | all held | joint 3 speed 1.17x (strikes 8-9) | joint 6 speed 1.25-1.31x after every blow | none exceeded |
| Hand joint loads | within rating | thumb CMC hard stops up to 26x rating | gearboxes up to 29x rating | |

## Findings so far

- **The earlier results were not achievable on a real FR3.** Before the limits were enforced, the
  controller stepped joint torques at up to 43x libfranka's 1000 Nm/s limit and the wrist spun past its
  velocity limit after each blow; the arm would have stopped with a reflex. Within the limits the FR3 hits
  at ~1.8 m/s (not 2.2), drives ~1.4 mm per blow against the default nail, and needs ~14 strikes for 20 mm.
- The strike pose matters as much as the controller. With the Franka Hand pointing down, the wrist joints'
  12 Nm limit caps the swing's acceleration (6.5 m/s^2 at the original hover pose, 11.6 m/s^2 at the
  current one, where the joints can push 123 N along the strike axis). With the WUJI hand pointing along the strike axis, the strike becomes an elbow extension and the
  joint velocity limits cap it at 1.3 m/s unless the hover pose is moved (2.6 m/s at the current one).
- **A power wrap with WUJI's rated torques does not hold a hammer blow on its own.** At rest the wrap holds
  (0.01 mm drift in 0.5 s at 40 N). A blow puts 500-800 N on the face, 0.19 m from the grip: a moment
  of ~100 Nm for a few milliseconds, against a wrap whose 2 Nm finger joints resist a few Nm, so the handle turns 3-17 deg in the hand
  per blow and is lost after nine strikes. More squeeze, grippier skin and gripping farther from the head
  did not change this. The blows also drive the thumb's CMC joints into their hard stops at up to 26x
  their rated torque.
- **If WUJI's joints really are self-locking, the wrap is the best grasp tested:** tilt 0.1-0.5 deg per
  blow, 10/10 hits. The price is load: the gearboxes hold 20-29x the joints' rated torque during blows,
  and the now-rigid grasp passes the blow into the arm, spinning FR3 joint 6 to 1.3x its velocity limit.
  Whether WUJI's drives are self-locking, and what their gearboxes and stops can hold, decides this;
  neither is in WUJI's published model or specifications.
- The Franka Hand's 17 mm pads hold the real 665 g claw hammer poorly: the tool twists in the pads on
  every blow, over the 1.4 deg target on 3 of 10 strikes at 55 N.
- The wooden handle is curved (its centreline drifts ~1.3 mm per cm). Pads aligned with the whole
  handle's axis load only one edge; the Franka grasp frame follows the handle's local direction at the
  grip, and the wrap uses short convex slices along the handle instead of one hull.
- Only the patch accelerometer meets the 2 ms impact-flag budget; the wrist F/T and the momentum observer
  see the blow 3.4-4.5 ms after contact.
- MuJoCo needs the noslip solver pass for a static grasp: without it soft friction lets the hammer creep
  ~11 deg/s under its own weight.

## Status

| Milestone | Scope | Status |
|---|---|---|
| M0 | packaging, Menagerie fetch + manifest, fallback arm | done |
| M1 | scene composition, `World`, grasp settle | done |
| M2 | hammer-nail contact calibration | done |
| M3 | rate-limited sensor models, scheduler | done |
| M4 | impedance controller, momentum observer, impact detector | done |
| M5 | grip force loop, drop detection, slip metrics | done |
| M6 | scripted swing, reference spreading, strike loop, replay viewer | done |
| M7 | HDF5 logging, `run_strikes` CLI | done |
| M8 | Gymnasium env, domain randomization | done |
| M9 | drill plant (Task B) | stub only |
| D0-D6 | WUJI Hand 2: fetch, import, grasp synthesis, taxel patches, joint control, strikes, viewer | done |
| D7 | experiments E1-E9 of `docs/dexterous_hand_plan.md` | partly (E2, E3, E7) |
| | Sharpa Wave | not started |

Not built: the series-elastic joint mode from the plan (`flex_mode="sea"`), the drill/screw plant, and
any learned L2/L3 layer.

## Licenses

Menagerie models are Apache-2.0 (Franka FR3, Franka Hand) and the WUJI Hand 2 model is MIT; their LICENSE
files are fetched alongside the meshes and are not redistributed in this repository. The YCB hammer scan is
CC BY 4.0.
