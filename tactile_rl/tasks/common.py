"""What the residual tasks share: their observation groups, rewards, terminations and reset randomization."""

from __future__ import annotations

import os

from mjlab.envs.mdp import time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from tactile_rl.mdp import rewards as R
from tactile_rl.mdp.observations import (
    drive_internals,
    event_truth,
    load_truth,
    nail_state,
    patch_contact,
    randomization,
    student_obs,
    tool_in_hand,
)
from tactile_rl.mdp.randomization import (
    pair_solref_solimp,
    randomize_forecast,
    randomize_plant,
    randomize_taxelscan,
    randomize_wuji_drive,
)
from tactile_rl.mdp.terminations import base_done, nonfinite_state, thermal_fault, tool_left_hand
from tactile_sim.control.obs_spec import OBS_SPEC_VERSION

ALL_GROUPS = ("student", "teacher", "critic", "aux_labels")
# a student group in an earlier layout (obs_spec.OBS_LAYOUTS), beside the current one: evaluating or distilling a
# student of that layout in the current env
LAYOUT_GROUPS = {"student_v2": 2}
# what a teacher's PPO run needs: rsl_rl stores every group the env returns for the whole rollout (888 floats per step
# instead of 2532 on the load generator), and the critic reads the teacher's group (the plan: the same observation)
TEACHER_GROUPS = ("teacher",)


def observation_groups(task: str, groups: tuple[str, ...] = ALL_GROUPS, nan_policy: str = "disabled",
                       student_version: int = OBS_SPEC_VERSION) -> dict[str, ObservationGroupCfg]:
    """`student` (obs_spec's layout `student_version`, the current one by default), `teacher` and `critic` (the current
    layout's student part plus the privileged state, whatever the student's layout), `aux_labels` (the students'
    auxiliary targets: the event truth and, on the load generator, the wrench) and `student_v2` (layout 2, beside the
    student's); `groups` picks which the env returns. `nan_policy` is mjlab's per group ("warn": a non-finite value is
    reported and zeroed, the plan's NaN guard for long runs; "error" in tests), checked once per group and step."""
    out = {k: v for k, v in _all_groups(task, student_version).items() if k in groups}
    for g in out.values():
        g.nan_policy = nan_policy
        g.nan_check_per_term = False
    return out


def _all_groups(task: str, student_version: int = OBS_SPEC_VERSION) -> dict[str, ObservationGroupCfg]:
    student = {"obs": ObservationTermCfg(func=student_obs)}  # the current layout (the teacher's student part)
    priv = {"tool_in_hand": ObservationTermCfg(func=tool_in_hand),
            "patch_contact": ObservationTermCfg(func=patch_contact),
            "event_truth": ObservationTermCfg(func=event_truth)}
    if task in ("hammer", "mixed"):
        priv["nail_state"] = ObservationTermCfg(func=nail_state)
    if task in ("loadgen", "mixed"):
        priv["load_truth"] = ObservationTermCfg(func=load_truth)
    priv.update({"drive_internals": ObservationTermCfg(func=drive_internals),
                 "randomization": ObservationTermCfg(func=randomization)})
    aux = {"event_truth": ObservationTermCfg(func=event_truth)}
    if task in ("loadgen", "mixed"):
        aux["load_truth"] = ObservationTermCfg(func=load_truth)
    out = {"student": ObservationGroupCfg({"obs": ObservationTermCfg(func=student_obs,
                                                                    params={"version": student_version})}),
           "teacher": ObservationGroupCfg({**student, **priv}),
           "critic": ObservationGroupCfg({**student, **priv}),
           "aux_labels": ObservationGroupCfg(aux)}
    for g, v in LAYOUT_GROUPS.items():
        out[g] = ObservationGroupCfg({"obs": ObservationTermCfg(func=student_obs, params={"version": v})})
    return out


def reward_terms(task: str) -> dict[str, RewardTermCfg]:
    """The plan's rewards for `task`, substep-integrated (tactile_rl.mdp.rewards): the effort penalties, the task's
    terms, the drop's and the thermal fault's penalties. $TACTILE_EFFORT_SCALE (default 1) multiplies the I^2 t
    penalty's weight: the heating-price experiment (heating dearer against slip)."""
    effort = float(os.environ.get("TACTILE_EFFORT_SCALE", "1"))
    out = {k: RewardTermCfg(func=getattr(R, k), weight=w * (effort if k == "effort_i2t" else 1.0))
           for k, w in R.WEIGHTS_EFFORT.items()}
    weights = R.WEIGHTS_HAMMER if task in ("hammer", "mixed") else R.WEIGHTS_LOADGEN
    out.update({k: RewardTermCfg(func=getattr(R, k), weight=w) for k, w in weights.items()})
    out["drop"] = RewardTermCfg(func=R.termination_penalty, weight=R.DROP_PENALTY, params={"term": "tool_dropped"})
    out["thermal_fault"] = RewardTermCfg(func=R.termination_penalty, weight=R.THERMAL_FAULT_PENALTY,
                                         params={"term": "thermal_fault"})
    return out


def termination_terms(task: str) -> dict[str, TerminationTermCfg]:
    out = {"time_out": TerminationTermCfg(func=time_out, time_out=True)}
    if task in ("hammer", "mixed"):
        out["base_done"] = TerminationTermCfg(func=base_done, time_out=True)
    out.update({"tool_dropped": TerminationTermCfg(func=tool_left_hand),
                "thermal_fault": TerminationTermCfg(func=thermal_fault),
                "nonfinite": TerminationTermCfg(func=nonfinite_state)})
    return out


def dr_events(scene_key: str, hand_friction: tuple[float, float] | None = None) -> dict[str, EventTermCfg]:
    """The reset randomization (tactile_rl.mdp.randomization), in the order they apply; `hand_friction` the hand/tool
    friction's range (None: the scene's ResidualDRCfg)."""
    p = {"scene_key": scene_key}
    pc = {**p, "hand_friction": None if hand_friction is None else tuple(hand_friction)}
    return {"dr_contact": EventTermCfg(func=pair_solref_solimp, mode="reset", params=pc),
            "dr_drive": EventTermCfg(func=randomize_wuji_drive, mode="reset", params=p),
            "dr_taxelscan": EventTermCfg(func=randomize_taxelscan, mode="reset", params=p),
            "dr_plant": EventTermCfg(func=randomize_plant, mode="reset", params=p),
            "dr_forecast": EventTermCfg(func=randomize_forecast, mode="reset", params=p)}
