# tactile-encode

Research and simulation for a tactile, force-aware control stack for dynamic tool use
(hammering, power drilling). The research lives in `reports/`, `research_notes/` and
`docs/control_flowchart.html`. The simulation plan is `docs/simulation_plan.md`.

`tactile_sim` is a headless MuJoCo simulation of the testbed for **Task A: a hammer driving a
pre-started nail**. It provides the plant, the sensor models and the 1 kHz L1 controller that the
learned layers (L2 reactive, L3 World-Action-Model / VLA planner) will train against.

- **Robot:** the Franka FR3 arm and the Franka Hand, both from MuJoCo Menagerie at a pinned commit, with
  torque-controlled joints. The only additions are a wrist F/T sensor body and compliant rubber layers on
  the hand's real 17 x 17 mm fingertip pads.
- **Tool:** the YCB 048_hammer scan (steel claw hammer, wooden handle, 665 g, CC BY 4.0), gripped
  190 mm from its head on the flat, widest part of the handle. The full scan is drawn; contacts use convex
  hulls cut from the scan (gripped handle section, striking face, whole head). Offline, a primitive
  hammer with the same masses is used.
- **Plant:** a nail in a vertical board. The nail resists with Coulomb friction that grows with depth.
  The hammer-nail contact gives a ~4 ms blow at ~500-730 N.
- **Sensors:** wrist F/T at 4 kHz, pad accelerometers at 8 kHz (+-16 g), an 8 x 8 pressure array
  (64 taxels) on each pad at 1 kHz, joint encoders and torques at 1 kHz, and the momentum observer's
  external torque. Each has its own band-limit, latency, noise, bias, quantization and saturation. Pad
  contact forces are spread over the taxels with a 3 mm kernel, the way a rubber layer spreads load.
- **Control:** a 1 kHz Cartesian impedance law with a reference limiter, stiffness slew limit, payload
  compensation and 50 ms velocity gating after impact; momentum observer and three-channel impact
  detector; reference spreading around the strike; 500 Hz grasp-force loop with drop detection.
- **Behavior:** a scripted swing that stands in for L2/L3, with a human-template grip profile (ramp
  150 ms before contact, peak 60 ms after), a setting tap, and iterative re-aiming between strikes.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .[dev,gym]
make assets        # fetch the pinned Menagerie FR3 + Franka Hand and the YCB hammer into ~/.cache
make test          # full suite, ~2 min
```

Without the caches everything still runs on stand-ins with identical kinematics and inertias
(capsule arm, box hand, primitive hammer). Set `TACTILE_SIM_ASSETS=/path` to use a different asset cache.

## Run

```bash
python -m tactile_sim.run_strikes --n 10 --out runs/demo.h5          # 8 kHz episode + HDF5 force-truth log
python -m tactile_sim.run_strikes --n 10 --fast --dr --seed 3         # 4 kHz, domain randomized
python -m tactile_sim.calibrate pulse                                 # free-hammer contact sweeps
python -m tactile_sim.viewer.export --n 8 --preset default --preset weak-grip --out runs/replay.html
python -m tactile_sim.viewer.live --n 5 --slowdown 20                 # native MuJoCo window (needs a display)
```

`runs/replay.html` is a standalone 3D replay (three.js, loaded from a CDN) of the exported episodes with
the real robot meshes, per-strike records and time-aligned traces. `gymnasium.make("TactileHammer-v0")`
exposes the L2 -> L1 interface (TCP offset, stiffness scale, strike-axis feedforward, grasp force) at 200 Hz.

On a machine with an NVIDIA GPU, set `MUJOCO_GL=egl` before starting Python to enable the optional
cameras (`sensors.cameras = True`).

## Results on the default testbed (8 kHz, seed 0, 10 strikes)

| Metric | Result | Report target |
|---|---|---|
| Strikes on the nail | 10 / 10 | >= 90 % |
| Nail driven | 18.6 mm (1.7-2.1 mm per full blow) | 20 mm in <= 10 strikes |
| Impact flag after true contact | 0.5-1.1 ms (pad accelerometer) | <= 2 ms |
| Tool tilt in the grasp per strike | 0.3-4.4 deg; over 1.4 deg on 5 of 10 strikes | <= 1.4 deg |
| Tool slip per strike | 0.8-4.3 mm | <= 7 mm |
| Grip ramp before contact | >= 150 ms | >= 150 ms |
| Drops | 0 | 0 |

## Findings so far

- The FR3's joint torque limits cap strike speed. At this pose, joints 1 and 3 reach 87 Nm and the
  hammer face arrives at ~2.1-2.2 m/s however fast the swing is commanded. Against the default nail
  resistance that takes ~12 strikes to drive 20 mm.
- The Franka Hand's 17 mm pads hold the real 665 g claw hammer poorly. Its centre of mass sits 137 mm
  from the grip toward the head, and the striking face is offset from the handle axis, so each blow twists
  the handle in the pads; tilt exceeds the 1.4 deg target on half the strikes even with the hold raised to the hand's 70 N
  continuous rating. At 55 N the hammer survives a 1 g shake but twists ~9 deg at 2 g; at 70 N it holds.
  With slippery pads or a 15 N hold the hammer twists out within a few strikes.
- The wooden handle is curved (its centreline drifts ~1.3 mm per cm). Pads aligned with the whole
  handle's axis load only one edge; the grasp frame follows the handle's local direction at the grip.
- Only the pad accelerometer meets the 2 ms impact-flag budget. The wrist F/T and the joint-torque
  momentum observer see the blow 3.4-4.5 ms after contact, because the compliant pads and wrist pass it on
  late. An F/T slope test cannot tell blows from the tool ringing in the grasp during ordinary swings.
- Post-impact ringing on the wrist F/T is 15-70x its pre-impact level, and it comes from the pads, not
  the wrist: a rigid wrist rings about as much.
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

Not built: the series-elastic joint mode from the plan (`flex_mode="sea"`), the drill/screw plant, and
any learned L2/L3 layer.

## Licenses

Menagerie models are Apache-2.0 (Franka FR3, Franka Hand); their LICENSE files are fetched alongside the
meshes and are not redistributed in this repository.
