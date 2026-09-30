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
- **Plant:** a 0.6 kg hammer and a nail in a vertical board. The nail resists with Coulomb friction that
  grows with depth. The hammer-nail contact gives a ~4.5 ms blow at ~500-700 N.
- **Sensors:** wrist F/T at 4 kHz, pad accelerometers at 8 kHz (+-16 g), 4 x 4 pad pressure arrays at
  500 Hz, joint encoders and torques at 1 kHz, and the momentum observer's external torque. Each has its
  own band-limit, latency, noise, bias, quantization and saturation.
- **Control:** a 1 kHz Cartesian impedance law with a reference limiter, stiffness slew limit, payload
  compensation and 50 ms velocity gating after impact; momentum observer and three-channel impact
  detector; reference spreading around the strike; 500 Hz grasp-force loop with drop detection.
- **Behavior:** a scripted swing that stands in for L2/L3, with a human-template grip profile (ramp
  150 ms before contact, peak 60 ms after), a setting tap, and iterative re-aiming between strikes.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .[dev,gym]
make assets        # fetch the pinned Menagerie FR3 + Franka Hand (~35 MB) into ~/.cache
make test          # full suite, ~2 min
```

Without the Menagerie cache everything still runs on stand-ins with identical kinematics and inertias
(capsule arm, box hand). Set `TACTILE_SIM_ASSETS=/path` to use a different asset cache.

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
| Nail driven | 17.7 mm (1.7-2.6 mm per full blow) | 20 mm in <= 10 strikes |
| Impact flag after true contact | 0.25-1.0 ms (pad accelerometer) | <= 2 ms |
| Tool tilt in the grasp per strike | 2.6 and 2.0 deg on the first two contacts, then <= 0.34 deg | <= 1.4 deg |
| Tool slip per strike | <= 0.75 mm | <= 7 mm |
| Grip ramp before contact | >= 150 ms | >= 150 ms |
| Drops | 0 | 0 |

## Findings so far

- The FR3's joint torque limits cap strike speed. At this pose, joints 1 and 3 reach 87 Nm and the
  hammer face arrives at ~2.1-2.2 m/s however fast the swing is commanded. Against the default nail
  resistance that takes ~12 strikes to drive 20 mm.
- The Franka Hand's 17 mm pads resist rotation about the approach axis weakly. The first contacts
  seat the handle by 2-3 deg; after that the per-strike target holds. Holding the hammer needs ~55 N per
  pad: at 40 N a 2 g shake twists it ~7 deg.
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
