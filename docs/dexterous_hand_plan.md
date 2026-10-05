# Plan: dexterous-hand variants of the hammer testbed (Sharpa Wave, WUJI Hand 2)

## Goal

Replace the Franka Hand's two-pad pinch with a power wrap grasp by a five-finger hand, and test whether a
wrap lets a hand with low fingertip force (15-20 N) hold the YCB hammer through repeated blows better than
the Franka Hand does at 55-70 N. Two hand variants are built and run separately on the same FR3 testbed:

- `hand = "sharpa_wave"`: Sharpa Wave, 22 DoF, 500 Hz joint control, 20 N fingertip force.
- `hand = "wuji2"`: WUJI Hand 2, 20 DoF, 1 kHz joint control in MIT mode (joint torque + PD), ~15 N fingertip force.

The Franka Hand stays as the baseline (`hand = "franka"`). Everything downstream of the hand (arm, impedance
controller, observer, impact detector, swing, hammer, nail plant, logging, viewer) is reused.

Tactile sensing: the user's own sensor is modelled on each hand as 8 x 8 taxel patches at 1 kHz (the
current pressure model), placed where the handle actually loads the hand. Built-in sensing is modelled at
vendor spec where it is known, and used only where it is good enough (see Sensors).

## What the vendor models give (checked)

| | Sharpa Wave | WUJI Hand 2 |
|---|---|---|
| Source | `github.com/sharpa-robotics/sharpa-urdf-usd-xml` @ `0d19cac6` | `github.com/wuji-technology/wuji-description` @ `c2cd7f8d` |
| License | Apache-2.0 | MIT |
| MJCF used | `wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.xml` | `hand2/hand2_beta2/body/mjcf/right_with_mount.xml` (Beta 2 = Beta 1 + a tactile pad body per fingertip) |
| Mass | 1.25 kg | 0.62 kg (0.69 kg with mount) |
| Joints / actuators | 22 hinges, 22 `position` actuators (kp 0.9-13.2, dampratio 0.9) | 20 hinges, 20 PD actuators (kp 0.18-0.69, kv 0.008-0.03) |
| Joint torque limits in the model | **none** | MCP flexion 2.0 Nm, MCP abduction 0.2 Nm, PIP/DIP 0.3 Nm, thumb CMC 0.6 Nm, thumb MCP/IP 0.3 Nm |
| Joint friction / armature | calibrated per joint class (frictionloss 0.004-0.13 Nm, armature 1e-4-3e-3) | armature 2e-4-5e-4, no friction |
| Collision | 27 convex-hull meshes; fingertip elastomer `solref 0.06 0.9` | 26 convex-hull meshes, every link collides, 10 assembly pairs excluded |
| Sites / sensors | none (tactile frames `*_elastomer` exist as bodies) | 5 fingertip sites, fingertip sensor-pad bodies |
| Model timestep | 2 ms | 2 ms, RK4 |

Consequences:
1. **Sharpa torque limits must be supplied.** The model's actuators are unlimited. Limits are set from
   the 20 N fingertip spec: each flexion joint gets `20 N x (joint-to-fingertip distance)` at full
   extension. This is marked as an assumption in the config and swept (x0.5-x2) in the experiments.
2. **Neither model has a real soft skin.** Sharpa's elastomer uses a 60 ms `solref`, which is far too soft
   and slow for 4 ms blows at an 8 kHz step. WUJI puts contact on the rigid distal hull. Both get the same
   explicit compliant skin layer used on the Franka pads (below).
3. **Convex hulls of the palm cannot seat a handle.** A convex palm hull fills the hollow a wrap relies on.
   Contact geometry is rebuilt as primitives fitted to the meshes (capsules per phalanx, boxes for the
   palm pads); the vendor meshes stay as visuals.
4. **WUJI's self-locking joints are not in the model.** Its joints are modelled as backdrivable motors. A
   self-locking variant (below) is added and both are tested.
5. Both models were tuned at 2 ms. Armature and friction are kept; stability at 0.125 ms is checked.

Geometry: WUJI index and middle fingers are 106 mm from MCP to tip. At 2 Nm per MCP that is ~19 N at the
tip with the finger straight, consistent with the 15 N spec. The YCB handle is ~35 mm across at the grip
(~110 mm around), so either hand can close around it.

## Architecture changes

```
tactile_sim/
  assets/
    fetch_hands.py        # per-file raw.githubusercontent.com fetch at pinned commits, sha256 manifest,
    HANDS_MANIFEST.json   #   same scheme as fetch_menagerie.py; cache ~/.cache/tactile_sim/hands/<vendor>/<sha>
  model/
    hands/
      base.py             # HandSpec: joints, actuators, torque limits, flange transform, palm frame,
                          #   grasp frame, skin patches, contact proxies, control rate, sensor layout
      franka.py           # current gripper.py wrapped as a HandSpec (baseline, unchanged behaviour)
      sharpa_wave.py      # import MJCF, strip its actuators/options/collision, add proxies + skin + motors
      wuji2.py            # same for WUJI Hand 2 Beta 2
      proxies.py          # fit capsules/boxes to link meshes (PCA + extent), cached per hand
      skin.py             # compliant skin patches: body on n/t springs, box geom, IMU site, taxel frame
    gripper.py            # becomes a dispatcher over HandSpec
  control/
    hand_joint.py         # per-joint MIT-mode law at the hand's rate: tau = kp(q*-q) + kd(dq*-dq) + tau_ff,
                          #   clipped to the torque limits; ZOH between hand ticks
    wrap_grasp.py         # grasp synergy: one grip scalar -> per-joint torques; tactile loop per finger
  sensors/
    pressure.py           # generalised from 2 pads to N named patches
    hand_builtin.py       # vendor tactile at vendor rate (Sharpa DTA 180 Hz), joint torque/current
  behaviors/
    grasp_synthesis.py    # offline: find the wrap pose for a hand + hammer, store as a keyframe
```

Selection is `cfg.gripper.hand in {"franka", "sharpa_wave", "wuji2"}`. `World`, the swing and the
controllers only see the grasp frame, the grip scalar and the named taxel patches, so they do not change
per hand.

### Mounting

- Sharpa: the `with_flange` variant is rooted at its flange; it goes on the F/T body with an adapter
  transform (flange axis = FR3 flange z).
- WUJI: `right_with_mount` includes WUJI's mount; the published palm mounting-interface drawing
  (`hand2/hand2_beta2/attachment/`) sets the adapter offset.
- Payload check at build time: F/T body + adapter + hand + hammer <= 3 kg (Sharpa ~2.1 kg, WUJI ~1.5 kg).

### Contact model

- Proxies: one capsule per phalanx (proximal, middle, distal) and 2-3 boxes over the palm. They are fitted
  to each link's mesh vertices, then shrunk so the skin sits where the real surface is.
- Skin: every patch that carries a taxel array is a separate body on a normal spring and two tangential
  springs (the Franka pad model), stiffness 2e5 N/m normal by default. Other link surfaces use soft contact
  with the same parameters.
- Contact pairs are explicit: hammer grip hull vs every proxy and skin geom; hammer head vs board; face vs
  nail. Finger-finger and finger-palm pairs stay on so the wrap cannot pass through itself.
- `noslip_iterations 5` and `impratio 100` as now.

### Grasp synthesis (offline, per hand)

1. Place the handle in the palm: handle axis across the palm diagonal (index MCP -> hypothenar), head out on
   the thumb/index side, handle pressed on the palm skin. Candidate poses on a small grid (roll about the
   handle, offset along the palm).
2. With the hammer welded, close all fingers under a joint torque ramp until each phalanx touches (or its
   joint stops), thumb over the index/middle middle phalanges.
3. Score: number of phalanges in contact, palm contact, a form-closure margin (smallest wrench the contacts
   cannot resist at a unit grip, computed from the contact normals), and fingertip force headroom.
4. Keep the best pose as a keyframe (joint angles, hammer pose in the palm frame, contact list). Release
   the weld and settle 0.5 s under gravity; reject if the hammer moves more than 1 mm / 0.5 deg.

The grasp frame for the swing is the handle frame at the palm's centre of contact, so the existing aiming
code works unchanged.

### Hand control

- `hand_joint.py` runs at 500 Hz (Sharpa) or 1 kHz (WUJI) with one-tick latency and ZOH, per the vendor
  rate. Torques are clipped to each joint's limit.
- `wrap_grasp.py` maps one grip scalar G to joint torques through the grasp keyframe's contact Jacobians
  (`tau = sum J_c^T n_c g_c(G)`), plus a position hold on the keyframe angles. The finger torque limits cap
  G, and the limit that binds first is logged.
- Tactile loop: per finger, a PI loop on the summed normal force of that finger's patches (the current
  `GripForceLoop` logic, per finger), plus the human grip template from the swing (ramp 150 ms before
  contact, peak 60 ms after). Drop detection uses all patches.
- **Self-locking (WUJI only):** optional `lock_mode = "self_locking"`. Each joint may close under motor
  torque but only opens if the external torque exceeds a breakaway value; below it the joint holds at its
  current angle (a one-sided joint limit moved every step). The breakaway torque is unknown and swept.

### Sensors

- **User's tactile patches (default for both hands):** 8 x 8 taxels, 1 kHz, 0.1 N noise, 20 N/taxel, with
  the 3 mm load-spread kernel, plus a patch accelerometer at 8 kHz. Patch size follows the segment it sits
  on (about 16 x 16 mm on a phalanx, 30 x 30 mm on the palm).
- Default layout = **2 patches (128 taxels, the user's current capacity):** palm (under the handle) and
  thumb distal (over the handle). Alternative layouts: `palm+index` and `full` (palm, thumb, and the
  proximal and middle phalanges of index and middle; 7 patches, 448 taxels).
- **Built-in, at vendor spec:**
  - Sharpa DTA: fingertip arrays at 180 Hz (spec says >1000 pixels; modelled at 16 x 16, since the grip
    doesn't touch the fingertips in a wrap). Too slow for impact timing; used only for grasp confirmation.
  - Sharpa joint torque sensing: 500 Hz.
  - WUJI Hand 2 fingertip tactile pad: body present in the model, spec not published; modelled as a 3-axis
    fingertip force at 1 kHz until the spec is known.
  - WUJI joint current -> torque: 1 kHz, 5 % noise.
- Wrist F/T, arm joints and the momentum observer are unchanged.

## Experiments

Each runs for `franka` (baseline), `sharpa_wave` and `wuji2`, at 8 kHz, 10 strikes, seeds 0-4.

| # | Experiment | Output |
|---|---|---|
| E1 | Static hold: gravity, then 1 g and 2 g shakes along the strike axis | slip/tilt vs grip scalar; minimum grip that holds |
| E2 | Default strike series | same table as the README (hits, depth, flag latency, tilt, slip, drops) |
| E3 | Grip sweep: 20-100 % of each hand's torque limit | tilt/slip per strike vs grip; the grip needed for tilt <= 1.4 deg |
| E4 | Contact-point loads | peak torque per finger joint during a blow vs its limit; backdrive angle per strike |
| E5 | Sensor layout: none / built-in only / 2 patches / full | drop detection time, grip-loop tracking, impact flag latency from patch accel |
| E6 | Control rate: hand loop at 1 kHz / 500 Hz / 200 Hz; tactile at 1 kHz / 180 Hz | same metrics as E2 |
| E7 | WUJI backdrivable vs self-locking (breakaway sweep) | tilt/slip, joint load |
| E8 | Sharpa torque-limit assumption x0.5 / x1 / x2 | sensitivity of E2/E3 |
| E9 | Domain randomization (friction 0.6-1.4, skin stiffness x0.5-2, hammer mass +-10 %) | robustness of E2 |

The main questions:
- Does the wrap keep tilt under 1.4 deg per strike at a grip within the hand's torque limits? (The Franka
  Hand misses on half the strikes at 70 N.)
- Which joints saturate first under a blow, and do they backdrive? With WUJI's DIP at 0.2 Nm the distal
  segments may give way while the palm and proximal phalanges hold.
- Is a 500 Hz hand loop enough, or does the 1 kHz WUJI loop measurably help?
- Is 128 taxels in the right two places enough, or does more coverage change the outcome?
- Does the heavier hand (Sharpa +0.6 kg over WUJI) lower the FR3's strike speed?

## Milestones

| # | Deliverable | Tests |
|---|---|---|
| D0 | `fetch_hands.py` + manifest at the pinned commits; both MJCFs load from the cache; offline stand-in hand (capsule fingers, same joint layout) | hashes; both models load; stand-in used offline |
| D1 | `HandSpec`; Franka refactored onto it with identical results | full existing test suite passes unchanged |
| D2 | Hand import, mounting, proxies, skin, joint motors with limits; 0.125 ms stability | hands hold a pose 2 s with no NaN; finger joint torque stays within limits; payload check |
| D3 | Grasp synthesis + keyframes for both hands | wrap has >= 8 phalanx contacts + palm; settled hammer moves < 1 mm / 0.5 deg |
| D4 | `hand_joint` + `wrap_grasp` controllers, per-finger tactile loop, self-locking option | per-finger force step settles in < 50 ms (WUJI) / < 80 ms (Sharpa); 1 g shake holds |
| D5 | N-patch pressure sensor + built-in sensors; layouts | taxel counts per layout; rates; patch accel sees the blow |
| D6 | Full strike loop per hand; viewer shows the hand meshes and all patches as heatmaps | E2 runs end to end for both hands |
| D7 | E1-E9 runs, results table + findings in README | — |

D2-D5 can be built for one hand first (WUJI: smaller, has published torque limits), then repeated for
Sharpa.

## Risks

1. **Sharpa torque limits are assumed.** Results that depend on them are reported as a sweep (E8), not a
   single number. Ask Sharpa for per-joint continuous and peak torque.
2. **Joint loads during a blow may exceed what the fingers are rated for.** The sim reports peak joint
   torque per blow (E4); hardware tests should not start until those are known to be within spec.
3. **Contact-rich wraps are expensive.** 20+ finger joints and ~20 contacts at 8 kHz may run 3-5x slower
   than the Franka setup; tests use `fast_config` and fewer strikes.
4. **Primitive proxies approximate the real finger shapes.** Contact positions may be off by a few mm.
   The proxy fit is checked against the mesh surface (max deviation reported).
5. **Beta models.** WUJI Hand 2 is at Beta 2 and its conventions are frozen from Beta 1 on; a pinned
   commit keeps results reproducible, and a later revision is a manifest update.
6. **Self-locking and the WUJI tactile pad** are modelled from descriptions, not data. Both are optional
   and reported separately.

## Status: WUJI Hand 2 (built)

Built as planned, with these changes:

- **Hardware limits are part of the test.** `tactile_sim/limits.py` enforces the FR3's torque and
  1000 Nm/s torque-rate limits and each hand's joint ratings in the command path, and monitors FR3 joint
  velocity, range, payload and the load on the hand's hard stops (or, self-locking, its gearboxes).
  Enforcing the torque-rate limit showed the original swing was infeasible on a real FR3, so the swing is
  now planned inside the arm's velocity and torque caps and arrives at the nail already braking; each
  hand has its own hover pose chosen for those caps. The Franka results changed accordingly (README).
- **Contact geometry:** the vendor's per-link convex hulls are used as-is (the palm hull did not block
  the handle in practice); the hammer handle is cut into 15 mm convex slices so the wrap follows its
  curve. Hand-handle contacts use soft contact (the skin), not spring-mounted patch bodies.
- **Grasp synthesis** closes the hand on the welded hammer at full synergy torque, releases the weld and
  lets the hammer seat under gravity; the seated hammer pose is the tool frame. The keyframe records each
  joint's hard-stop load: the first wrap tried (thumb abduction closing too) held only because the thumb
  sat on its stops at up to 7x rating, so the default keeps thumb abduction as a held shaping joint. At
  the 40 N hold the worst stop load is 0.44x rating (1.5x at full squeeze). Default: handle across the palm
  under the MCP line, fingers curled around it, thumb flexed over it (CMC abduction held at -0.8 rad).
- **Taxel patches** go where the seated grasp loads the hand (force-weighted contact centroid, facing the
  handle): the palm and the thumb's proximal segment for the default 2 x 64 layout.
- **Self-locking** is a ratchet: a closing joint's opening-side limit follows it closed, shaping joints are
  held; the limit constraint force is the gearbox load.

Results (8 kHz, 10 strikes) are in the README. In short: with backdrivable joints the wrap turns 3-17 deg in
the hand per blow and the thumb's hard stops carry up to 26x their rating; with self-locking joints the
tilt is 0.1-0.5 deg but the gearboxes carry up to 29x rating and the stiff grasp spins FR3 joint 6 past
its velocity limit. What WUJI's drives and stops can hold decides whether the wrap works; ask WUJI.

Not done yet: E1 (shake tests), E4 in full, E5 (sensor layouts), E6 (control rates), E8/E9, and the
Sharpa Wave.

## Status: WUJI Hand 2 on the Dexmate Vega U (built)

- **Model:** Dexmate's Vega U URDF (`dexmate-ai/dexmate-urdf`, Apache-2.0, pinned) compiled by MuJoCo: a fixed
  pedestal, a lift (0-0.4 m) and a torso flip (0-1 rad) under the head and two 7-joint arms. Dexmate ships
  the pedestal, lift and torso meshes only as GLB; `tactile_sim/assets/glb.py` converts them to OBJ. Dexmate's
  Vega U robot profile drives only the upper body (arms, head), so the lift (0 m, shoulders 1.24 m up) and
  flip (upright) are set before a run and compiled as fixed joints. A WUJI Hand 2 on each wrist: the right
  one (with the grasp synthesis, patches and joint law above) strikes, the left one holds a relaxed pose on
  position servos at its joint ratings; the left arm and head hold a pose. The wheeled Vega-1P (same arms,
  a three-joint torso, wheels locked) builds from the same code.
- **Interface, as Dexmate exposes it:** `dexcontrol` takes joint position targets (optionally with velocity
  feedforward) at 100 Hz and lets the user scale the factory P gains by 0.1-4; no torque mode. The sim
  runs every Vega joint on a torque-limited PD servo (ratings from the URDF) and a host loop
  (`tactile_sim/control/position_l1.py`) that detects impacts at 1 kHz from the wrist F/T and the patch
  accelerometer and every 10 ms sends differential-IK targets, offset by the gravity droop of the arm and of
  every held joint, rate-limited to 90 % of the joint velocity limits. Factory gains, drive inertia and any
  torque-rate limit are not published; the sim's values are assumptions and the P multiplier is swept.
- **Strike:** a vertical nail in a board lying on a tabletop (top at 0.84 m), driven straight down by the
  right arm in front of the robot. The path is an arc (`StrikeGeometry`): the tool turns about a pivot
  0.6 m behind the face along the handle, through ~25 deg over a 0.20 m windup, and the face meets the nail
  square, moving straight down. Hover pose, handle direction and arc radius were searched for the speed the
  joint velocity and torque limits allow: a longer arc is faster (0.8 m: ~1.9 m/s cap) but the windup is
  then limited by joint 7's range; 0.6 m allows ~1.5 m/s, and the swing reaches ~1.1 m/s. The grasp is
  synthesised under gravity in the tool frame of this pose (the hammer hangs differently than in a forward
  strike). An earlier forward (horizontal) strike reached ~0.85 m/s.
- **Fixes found on the way, which also changed the FR3 + WUJI numbers:** the learned lateral aim drift was
  applied at the requested rather than the achievable strike speed; and the WUJI drop check fired when the
  two patches lost the load while the fingers still held the tool. With the downward strike the windup's
  deceleration presses the handle onto the bare fingers, so the drop check now needs the fingers to close
  into the space a lost tool would leave for any WUJI hand, not only a self-locking one.

Results are in the README.
