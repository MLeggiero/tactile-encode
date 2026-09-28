# Plan: MuJoCo simulation of the tactile + World-Action-Model (WAM) testbed — Task A (hammer → nail)

## Context

`tactile-encode` is currently research only: a synthesized report (`reports/Tactile VLA dynamic tool use.md`), eight research notes, and a control-design flowchart (`docs/control_flowchart.html`). No code, README, packaging or tests exist. The report designs a three-layer stack (L3 VLA / World-Action-Model planner at 1–10 Hz, L2 reactive net at 100–200 Hz, L1 real-time controller at 1 kHz, L0 tool–target estimator) for dynamic tool use and names simulation as stage 3 of its roadmap (`reports/...md:117`). The flowchart pins the sim target: **MuJoCo at 0.1–0.25 ms, soft pads, joint flexibility, sensor models for bandwidth/noise/latency/saturation** (`docs/control_flowchart.html:157-158, 263-264, 334, 627`).

Scope decisions confirmed with the project owner:
- "WAM" = **World-Action Model**, not the Barrett arm. The sim is the plant + sensors + L1 controller that the learned layers will later train against.
- Simulator = **MuJoCo** (CPU, headless). Verified: `mujoco==3.14.0` wheel downloads here (Python 3.11, 4 cores, 15 GB RAM, no GPU).
- First scope = **Task A: hammer → pre-started nail in a soft board**. Task B (driver/screw) is out of scope but gets an architectural slot.

Task A is not "viable" without an arm that swings, sensors that see the impact, and a controller that survives it. So minimal sensor models, the L1 controller, a scripted swing generator (standing in for L2/L3), force-truth logging, and a thin Gymnasium wrapper are in scope. Learned L2/L3 layers are not.

Asset facts: `raw.githubusercontent.com` works (Menagerie `franka_fr3/fr3.xml`, `franka_emika_panda/hand.xml`, and their `.obj`/`.stl` meshes fetch fine; 28 OBJ + 8 STL for the arm, ~1.4 MB per visual mesh), but `codeload.github.com` tarballs and `api.github.com` return 403. Assets are therefore fetched file-by-file at a pinned commit, never as a tarball. Menagerie FR3 ships `implicitfast`, position actuators (to be replaced with torque motors), `armature 0.195`, `frictionloss 1.137`, and an `attachment_site` on link 7. The Panda hand has slide-joint fingers (0–0.04 m) with small box fingertip pads.

## Design anchors → sim decisions

| Constraint (report / flowchart) | Sim decision |
|---|---|
| 3 ms pulse, 50–700 N peaks, 0.45 kg head | tune hammer-face/nail-head `solref`; test asserts 1–6 ms pulse |
| 20 mm in ≤10 strikes | nail resistance calibrated so a 2.5 m/s strike advances ~2 mm |
| Ringing must exist; rigid sim has none | explicit compliant pad bodies + wrist compliance (optional series-elastic joints) |
| Sensor rates F/T 1–8 kHz, accel 8 kHz ±16 g, pressure ≥200 Hz, joints 1 kHz, τ_ext ≥100 Hz | physics at **8 kHz (dt = 0.125 ms)** so every rate is an integer divisor |
| L1 1 kHz, grip 200–500 Hz, L2 100–200 Hz | scheduler ticks every 8 / 16 / 40 physics steps |
| Impact flag ≤2 ms, gate q̇ 50 ms, grip ramp at t_c − 150 ms, grip peak +50–70 ms | controller/swing defaults and test thresholds |
| Watchdog: stale L2 > 20 ms → hold pose, F_grip = F_hold, F_ff = 0 | `control/supervisor.py` |

## Package layout (new, all under repo root)

```
pyproject.toml   # tactile-sim; deps mujoco>=3.2,<4 numpy scipy h5py; extras gym, dev(pytest ruff)
README.md  Makefile (assets | test | strikes | lint)
tactile_sim/
  config.py        # dataclasses: SimConfig{physics, arm, gripper, hammer, plant, sensors, controller, swing, dr, logging}
  names.py         # canonical body/site/geom/joint/actuator names
  assets/  fetch_menagerie.py MANIFEST.json (pinned sha + file hashes) fallback_arm.xml hammer.xml nail_board.xml gripper_pads.xml
  model/   builder.py arm.py gripper.py tool_hammer.py sensors_mjcf.py plants/{base,nail,drill(stub)}.py
  sim/     world.py scheduler.py truth.py randomize.py
  sensors/ base.py ft.py accel.py pressure.py joints.py camera.py(optional)
  control/ interface.py kinematics.py impedance.py reference_spreading.py momentum_observer.py gripper.py supervisor.py
  behaviors/ trajectories.py swing.py
  logging/ schema.py writer.py strike_metrics.py
  env/gym_env.py   run_strikes.py   calibrate.py
tests/  conftest.py test_{assets,model,contact,sensors,controller,observer,gripper,swing,logging,env}.py
```

Boundaries: `model/` only emits XML; `sim/` owns MjModel/MjData; `sensors/` and `control/` are pure NumPy over `World` accessors (portable to hardware later); `behaviors/` emits the same `L2Command` a learned L2 would.

## MJCF modeling

- **Composition** (`model/builder.py`): `xml.etree` composition (not `MjSpec`, still churning). Load FR3, strip `<actuator>`, add `<motor>` per joint (±87 / ±12 Nm). Disable collision on links 0–5. Under link 7 attach `ft_sensor_body` (F/T site; MuJoCo force/torque sensors read the parent-child wrench) → optional `wrist_flex` body → Panda hand → pad bodies. Hammer is a free body with an inactive `grasp_weld` equality used only to settle at reset. Meshes passed via `MjModel.from_xml_string(xml, assets=...)`. Options: `timestep 0.000125 integrator implicitfast cone elliptic impratio 10`.
- **Nail** (`plants/nail.py`): slide joint (range 0–30 mm), penetration resistance as **`frictionloss` (Coulomb), not a spring** (a spring returns energy). Depth-dependent: `dof_frictionloss = R0 + R1·depth` set in `pre_step`; viscous `damping` for rate-dependent wood crushing. R0 ≈ 400–600 N calibrated by `calibrate.py`, DR range 100–1500 N. Head cylinder with `margin 2 mm` against tunnelling; shank visual-only; board collides with hammer (misses land on the board).
- **Pads** (`model/gripper.py`): explicit compliant pad bodies per finger: normal slide (k ≈ 5e4 N/m, ζ ≈ 0.3, 10 g) + two tangential slides; box geom `friction 1.3`, `condim 4`, `priority 1`. Gives a controllable ~65 Hz tool-in-grasp resonance and Coulomb slip. Fallback `pad_mode="soft_contact"` flag if unstable. Finger coupling via split tendon + joint equality; `grip_motor` on the tendon (±150 N).
- **Joint flexibility** (`cfg.arm.flex_mode`): `rigid` (baseline), `wrist` (default: 2 hinges + 1 slide with stiffness/damping under the F/T body, ~40–80 Hz ring), `sea` (series-elastic rewrite of all 7 joints, for the ringing study only).
- **Hammer**: 0.45 kg head + 0.15 kg handle capsule (r 14 mm). Sites `hammer_face`, `hammer_ref` (slip), `hammer_imu`. Face `solref="0.0015 0.4" priority 2`; `calibrate.py` sweeps timeconst 0.5–4 ms and dampratio 0.2–1 to pin a ~3 ms pulse.
- **Velocity-dependent restitution**: per step, set `geom_solref[nail_head,1] = ζ0 + ζ1·v_approach` before contact, frozen during the pulse.
- **Sensors in MJCF**: `force`/`torque` at `ft_site`, `accelerometer` at pad and hammer sites, `jointpos/jointvel/jointactuatorfrc`, `touch` 4×4 grid per pad for the pressure array.

## Sensor model (`sensors/base.py`)

`SensorSpec(rate_hz, dim, bandwidth_hz, latency_s, noise_std, bias_std, saturation, quant_step, decimation)` and `RateLimitedSensor.step()` called every physics step: bandwidth filter at physics rate (scipy butter, per-step IIR) → decimate every `round(f_phys/f_s)` steps (`mean` boxcar or `zoh`) → latency deque → bias + noise → quantize → clip. `latest()` returns ZOH-delayed sample with `(t_sample, t_avail, seq)`; `window(n)` for the 27–64-sample F/T windows. Defaults: F/T 4 kHz / 0.5 ms latency / ±500 N; pad accel 8 kHz / ±16 g / 3.5 kHz bandwidth; pressure 500 Hz; joints 1 kHz with encoder quantization; `tau_ext` fed by the observer. `camera.py` probes EGL/osmesa and degrades to disabled.

## L1 controller (`control/`)

- `interface.py`: `L2Command(x_eq, xd_eq, K, D, F_ff, F_grip, mode, t_c_pred)` = the L2→L1 boundary from the report (`reports/...md:104`).
- `impedance.py` at 1 kHz: `e = clip(pose_error, ±Δmax)` (reference limiter), `F = K_t e + D ė + F_ff`, `τ = Jᵀ F + N(K_q(q_post − q) − D_q q̇) + qfrc_bias`, torque clip. K in the strike frame, slew-limited (`K̇_max`, passivity), `D = 2ζ√(KΛ)` overdamped default. Gating: 50 ms after the impact flag, velocity feedback uses the reference velocity, not measured q̇. Stale-input fallback in `supervisor.py`.
- `momentum_observer.py`: `r_k = K_O[p_k − p_0 − Σ(τ_m + Ṁq̇ − qfrc_bias + qfrc_passive + r_{j−1})Δt]` with `M` from `mj_fullM`, `K_O` ≈ 300–500 s⁻¹ (documented as sim-optimistic, DR'd). `ImpactDetector` flags on projected `r` along the strike axis > 30 N or F/T slope; latches peak load and `t_flag`; 100 ms refractory; logs `flag_source`.
- `reference_spreading.py`: ante reference continues the swing through the nail; post reference starts at predicted contact pose, time-shifted to the actual flag; interim feedforward from t_c − 10 ms; switch only on the flag (clock timeout t_c + 100 ms → miss).
- `gripper.py`: 500 Hz PI + feedforward force loop on summed pad normal force (through the pressure sensor model); drop detection (force collapse + pad accel spike → freeze arm).

## Scripted swing (`behaviors/swing.py`)

200 Hz FSM standing in for L2 + a trivial L3, vertical stroke onto a horizontal board: `approach` (min-jerk to hover 60 mm above the head) → `windup` (0.12–0.20 m) → `swing` (quintic timed so speed = v_strike 1.5–3 m/s at head crossing; `t_c_pred` from current nail depth; K high along z) → `strike` (from t_c − 20 ms, slip detector muted, wait for flag/timeout) → `recover` → `settle` → repeat until depth ≥ 20 mm or n strikes. Grip schedule follows the human template: hold until t_c − 150 ms, ramp to `F_pre = f(tool mass, v_strike, μ̂)` by t_c − 50 ms, peak at t_flag + 60 ms, decay over 200 ms; slip on the previous strike raises the next hold margin.

## Logging (`logging/`), HDF5 per episode

Groups `/physics` (truth at physics rate: hammer pose/vel, nail depth, hammer–nail contact force, pad forces, hammer-in-hand pose), `/sensors/<name>` (native rate, `t_sample`, `t_avail`, `value`, `seq`), `/control` (1 kHz: τ_cmd, q, q̇, x, x_eq, K, F_ff, F_grip, mode, impact_flag, τ_ext_obs, gated), `/events`, `/strikes` compound rows (t_contact_truth, t_flag, flag latency, hit, peak force, impulse, pulse width, depth increment, slip_trans, slip_rot, drop, peak joint torque, grip timings, ringing energy 30–300 Hz), `/summary` attrs (hit rate, strikes_to_20mm, max slip, drops). Root attrs include config JSON, seed, mujoco version, git sha. `npz` fallback. `StrikeAccumulator` computes rows from truth (pulse width = force > 5 % of peak; slip = `hammer_ref` pose change in the TCP frame over the strike).

## Milestones (each independently testable; `pytest -q` < 3 min on CPU using `cfg_fast`: 0.25 ms step, fallback arm when Menagerie is absent)

| # | Deliverable | Tests |
|---|---|---|
| M0 | Scaffolding, `fetch_menagerie.py` (raw URLs, pinned sha, sha256 manifest, cache `~/.cache/tactile_sim/menagerie/<sha>`, `TACTILE_SIM_ASSETS` override), capsule `fallback_arm.xml` | manifest hashes; offline fallback; both arms load |
| M1 | Scene composition, keyframe, weld-settle reset, `World` | names exist; no NaN 2 s under hold; hammer stays in grasp after weld release (< 2 mm); energy drift < 5 % at both timesteps |
| M2 | Contact calibration (free hammer, no arm), `truth.py`, `calibrate.py` | 2.5 m/s strike: pulse 1–6 ms, peak 100–2000 N, advance 0.5–5 mm; no tunnelling at 6 m/s; restitution falls with dampratio; increment shrinks with depth |
| M3 | Sensor models + scheduler | sample counts, latency, ZOH, noise std, saturation, −3 dB point, monotonic timestamps |
| M4 | Impedance + observer + detector + supervisor | holds pose < 1 mm; step response overshoot < 10 %; limiter bounds force; K slew; observer matches applied push within 15 %; flag ≤ 2 ms after truth contact |
| M5 | Grip loop + slip metrics | settles ±5 % in < 50 ms; 40 N + shake → slip < 0.7 cm; 5 N → drop flagged |
| M6 | Swing FSM + reference spreading, full strike loop | 10 strikes: hit ≥ 90 %, ≥ 1 mm/strike, no drop, slip ≤ 0.7 cm / 1.4°, grip ramp ≥ 150 ms before contact, q̇ gated 50 ms; ringing energy post-impact ≥ 5× pre and ≥ 3× `rigid` mode |
| M7 | HDF5 writer + `run_strikes` CLI | CLI writes all groups, 3 strike rows, rates match spec |
| M8 | Gym env + domain randomization | reset/step contracts, seeded DR reproducible, in-range params |
| M9 | Drill plant stub + README | stub raises `NotImplementedError` |

M2 and M3 can proceed in parallel after M1; M6 needs M4 + M5.

## Risks specific to MuJoCo

1. Pulse width is set by `solref` × effective mass, not material; timeconst < 0.5 ms with a 0.45 kg head gives > 5 kN peaks and chatter. Sweep in M2 before the controller depends on it.
2. Nail `frictionloss` jitter at rest (µm) → report depth as a 1 ms average.
3. Pad tangential springs / `condim 6` may destabilize → `pad_mode="soft_contact"` fallback.
4. Arm `frictionloss` shows up as external torque unless `qfrc_passive` is in the observer; include it and DR the mismatch.
5. Observer ≤ 2 ms at 1 kHz needs high gain, optimistic vs. a real gearbox → `flag_source` logged so the F/T path is evaluated separately.
6. Runtime ~30 DoF at 8 kHz ≈ 4–8 s wall per 10-strike episode; CI uses `cfg_fast`.
7. DR of `body_mass` needs `mj_setConst` + re-settle; `geom_solref` mutation at runtime does not.
8. Cameras need EGL/osmesa, likely absent; never fail import.

## Verification

- `make assets` fetches Menagerie (or reports fallback); `make test` runs the full pytest suite headless.
- `python -m tactile_sim.run_strikes --n 10 --out runs/demo.h5 --seed 0` then inspect `/summary` attrs: expect `strikes_to_20mm ≤ 10`, `hit_rate ≥ 0.9`, `drops = 0`, `max_slip_trans ≤ 0.007`.
- `python -m tactile_sim.calibrate pulse` prints pulse width/peak/restitution across the `solref` sweep; the default is pinned to the ~3 ms row.
- Compare `--flex-mode rigid` vs `wrist` on the logged 30–300 Hz post-impact F/T energy to confirm ringing is present only with compliance.
- Add a short "Simulation" section to README linking back to `docs/control_flowchart.html:334` so the sim's numbers stay traceable to the report.
