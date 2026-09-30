# tactile-encode

Research and simulation for a tactile, force-aware control stack for dynamic tool use
(hammering, power drilling). The research lives in `reports/`, `research_notes/` and
`docs/control_flowchart.html`. The simulation plan is `docs/simulation_plan.md`.

`tactile_sim` is a headless MuJoCo simulation of the testbed for **Task A: a hammer
driving a pre-started nail**. It provides the plant, the sensor models and the 1 kHz L1
controller that the learned layers (L2 reactive, L3 World-Action-Model / VLA planner)
will later train against.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .[dev,gym]
make assets        # fetch the pinned MuJoCo Menagerie FR3 + hand (~35 MB) into ~/.cache
make test          # fast suite
```

Without the Menagerie assets everything still runs on a capsule fallback arm that has
the same kinematics, inertias and joint parameters as the FR3. Set
`TACTILE_SIM_ASSETS=/path` to use a different asset cache.

## Status

| Milestone | Scope | Status |
|---|---|---|
| M0 | packaging, Menagerie fetch + manifest, fallback arm | done |
| M1 | scene composition, `World`, grasp settle | pending |
| M2 | hammer-nail contact calibration | pending |
| M3 | rate-limited sensor models | pending |
| M4 | impedance controller, momentum observer, impact detector | pending |
| M5 | grip force loop, slip metrics | pending |
| M6 | scripted swing, reference spreading, full strike loop | pending |
| M7 | HDF5 logging, `run_strikes` CLI | pending |
| M8 | Gymnasium env, domain randomization | pending |
| M9 | drill plant stub | pending |

## Licenses

Menagerie models are Apache-2.0 (Franka FR3, Franka hand); their LICENSE files are fetched
alongside the meshes and are not redistributed in this repository.
