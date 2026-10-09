# The Makita DF333D in simulation

The drill task's tool is a real 12 V drill/driver, the Makita DF333D (12V max CXT; US: FD09), modelled from
Makita's published data and its product renders: its drivetrain (`tactile_sim/model/plants/drivetrain.py`), its
body (`tactile_sim/model/df333d.py`), and how the WUJI Hand 2 holds it (`DrillCfg.pistol_*`, the drill task's
`gripper.thumb_preshape`). `DrillCfg.pistol_body = "primitive"` and the presets in `config.py` bring back the
earlier primitive drill.

## Sources

| What | Value | Source |
|---|---|---|
| Speed, low gear | 0-450 rpm (0-1,700 high) | Makita catalogue / manual |
| Torque, ISO 5393 | 14 Nm soft joint, 30 Nm hard joint | Makita catalogue |
| Clutch | 0.5-3.5 Nm over 20 settings, plus drill mode | Makita catalogue |
| Output | 170 W | Makita catalogue |
| Size with BL1016 / BL1021B | 179 x 66 x 212 mm (L x W x H) | Makita catalogue |
| Weight | 1.1-1.2 kg with the pack | Makita catalogue |
| Pack (BL1021B, CXT slide) | 87 x 64 x 37 mm, 0.21 kg | retailer data, consistent across sources |
| Shape | side and rear silhouettes | Makita's renders `DF333D_C8C0` (right side) and `DF333D_C7C0` (rear), transparent backgrounds |

The silhouettes were scaled to the catalogue's length and height (0.734 / 0.771 mm per pixel along / across the bit)
and checked against the pack's 64 mm width in the rear view (0.753 mm per pixel): the scales agree within 2.5 %.

## Measured shape

Side view, X forward along the bit from the tool's rear, Z down from its top (mm):

| Part | Measurement | In the model |
|---|---|---|
| Bit axis | 29 below the top | `axis_z` 29 |
| Motor housing | 55-56 tall, 53-55 wide, X 15-110 | capsule r 27.5, X 27.5-112 (touchable) |
| Gearbox | 45 tall, X 110-128 | cylinder r 22 (drawing, mass) |
| Clutch ring, chuck | 42, then 40-32, X 128-172 | cylinder r 18 (drawing, mass) |
| Grip centreline | X = 84.8 - 0.190 Z: raked 10.7 deg | two capsules along it |
| Grip section | 30-31 wide, 35-44 deep (40 over the hand) | a 30 x 40 stadium: two r 15 capsules 10 apart |
| Trigger | face 13 proud of the grip, centre 80 below the top (51 below the bit axis) | box 18 x 6 x 12, 6 mm travel |
| Base, pack | Z 148-175, Z 175-212, ~80 long | boxes (drawing, mass) |

Mass split (not published; 1.15 kg in all, inside the catalogue's 1.1-1.2): motor and head shell 0.34, gearbox and
clutch 0.22, chuck 0.14, grip 0.12, base 0.12, pack 0.21.

## Collision geometry in the tool frame

The tool frame is the grasp point: `pistol_grip_from_head` (66 mm) below the bit axis, `pistol_grasp_offset`
(the grip's centreline 4 mm behind it and 4 mm to its -z side); x down (perpendicular to the bit), y back along the bit
(the tip at -y), z across. With the defaults (mm):

| Geom | Type | Size | Centre | Axis |
|---|---|---|---|---|
| `hammer_handle` (grip, back) | capsule | r 15, half-length 36.64 | (8.07, 10.62, -4.0) | (-0.9825, -0.1864, 0) |
| `drill_grip_front` | capsule | r 15, half-length 36.64 | (9.93, 0.79, -4.0) | same |
| `drill_housing` | capsule | r 27.5, half-length 42.25 | (-66.0, 1.03, -4.0) | (0, 1, 0) |
| `trigger_face` | box on a slide joint | half (9, 3, 6) | (-15.0, -29.38, -4.0) at rest | slides along (-0.1864, 0.9825, 0) |

The bit tip is at (-66.0, -158.2, -4.0); the tool's mass 1.15 kg, centre of mass (-14.3, -20.9, -4.0), principal
inertia diag (7.56, 6.22, 1.75) x 1e-3 kg m^2 (the trigger body, 8 g, apart).

## Drivetrain

`stall_torque` 14 Nm (the soft joint), `free_speed` 47.12 rad/s (450 rpm), `spindle_inertia` 5.5e-3 kg m^2 fitted
so the virtual dyno's ISO 5393 hard joint gives 30 Nm (`python -m tactile_sim.calibrate drill --fit-hard 30`). The
dyno then reads 448 rpm no-load (spindle drag), 13.99 / 30.02 Nm soft / hard, and the motor line's peak power
14 x 47.12 / 4 = 165 W against the rated 170 W, which the fit did not use. The clutch's nominal 1.5 Nm is about
setting 7 of 20.

## How the WUJI holds it

A real drill's shape forced a new hold:

- The hammer's side wrap puts the thumb along the handle toward the head; on the DF333D it runs 14-34 mm into the
  motor housing at any grasp height.
- A person's pistol grip (palm behind the grip) clears the housing, but the WUJI's index finger then wraps the grip
  diagonally and its tip lands 24 mm below the trigger.
- The WUJI's fingertip joints are rated 0.3 Nm: a fingertip pushes at most ~7 N, under the trigger's 8 N preload. The
  index finger's middle segment, closer to its 2 Nm knuckle, has to cross the trigger.
- With the thumb closing round the grip's front, its tip pressed the trigger's far side and the two clamped it.

The hold was found by searching the wrap's roll about the grip (`pistol_wrap_roll`), the grasp height and offset,
and the thumb's CMC abduction, in three rounds:

1. In the grasp-synthesis scene, scored on the trigger's travel at a full pull (>= 90 %) and after letting go
   (< 5 %), the thumb's force on the trigger (none), its clearance of the housing (> 2 mm) and the tool's turn under
   60 N of push-back along the bit (< 2-3 deg). Holds that passed also seated randomized screws, yet in the task
   world (the arm, the drives' MIT model, the full hold) every pull of the trigger turned the tool 1-5 deg further in
   the hand, pressing the trigger into the resting finger until the motor ran at rest.
2. In the task world, on four pull - let go - rest cycles. One hold (roll -35 deg) held still. But the WUJI's
   fingers did not collide with each other: wrapped round the grip, the thumb passed 15 mm through the ring finger.
3. With the fingers colliding (`GripperCfg.finger_self_contact`, the thumb now meeting the ring and little fingers
   too), in the task world, on three pull cycles under the screw's 60 N of push-back (243 holds; 40 the arm cannot
   reach at the hover pose). The best seated randomized screws only slowly, the index finger pulling the trigger
   34-84 % of the way under the push; pulling at the finger's full rated torque (`trigger_full` 1.0, was 0.8)
   fixed it.

The result: roll -25 deg (toward a pistol grip), 66 mm below the bit axis, the grip's centreline 4 mm behind and
4 mm to the side of the grasp point, the thumb's CMC abduction held at -0.6 rad (the hammer's wrap: -0.8), the
fingers colliding (`VEGA_DRILL_OVERRIDES["gripper"]`), the index finger resting on the trigger at 0.12 (at 0.15 a
finger that has just let go can settle 1.3 mm into the trigger, past the dead band) and pulling at 1.0. Over ten
free-air cycles the tool turns 0.06 deg, then ~0.04 deg more per cycle (0.41 after ten), the trigger releasing fully
each time; the fingers overlap 0.14 mm at most.

Randomized drives on the Vega U + WUJI (the residual experiments' randomization: hand friction 0.6-1.4, the tool's
masses, the drill's class, the task shifted +-3 cm; the scripted feed; the trigger pulled by the index finger):

| | screws (42) | holes (10) |
|---|---|---|
| Done | 42 | 10 |
| Trigger pulled fully | 42 | 10 |
| Cam-outs | 0 | 0 |
| Median time | 2.51 s seated | 13.75 s through (feed-rate limited) |
| Tool turn in the hand (max, incl. the pads' give) | 5.3-7.6 deg | 5.2-7.5 deg |
| Tool slide in the hand (max) | 3.3 mm | 2.3 mm |
| First contact off the screw's axis | <= 0.68 mm | <= 2.83 mm |

In the keyframe the thumb's base carries most of the grip (56 N), then the little and ring fingers' base segments
(28, 25 N); the middle finger's knuckle and the ring and little fingers' fingertip joints rest at their stops (the
stop carries the closing torque), so those joints cannot squeeze harder.

## Limits

- The mass split, the trigger's width, travel and spring, and the grip's taper (35 mm deep at its bottom) are
  estimates or simplifications; the grip is a constant 30 x 40 stadium.
- The hold is sensitive: in the search, at most a fifth of a good point's neighbours also passed. Randomized drives
  are the check that matters, and a VLA's hand targets must put the index finger on the trigger as precisely.
- Vibration amplitudes have no published anchor (see `DrillCfg` vib_*).
