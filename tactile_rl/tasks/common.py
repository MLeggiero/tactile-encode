"""What the residual tasks share: their observation groups, rewards, terminations and reset randomization."""

from __future__ import annotations

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

ALL_GROUPS = ("student", "teacher", "critic", "aux_labels")
# what a teacher's PPO run needs: rsl_rl stores every group the env returns for the whole rollout (888 floats per step
# instead of 2532 on the load generator), and the critic reads the teacher's group (the plan: the same observation)
TEACHER_GROUPS = ("teacher",)


def observation_groups(task: str, groups: tuple[str, ...] = ALL_GROUPS) -> dict[str, ObservationGroupCfg]:
    """`student` (obs_spec's layout), `teacher` and `critic` (the student's plus the privileged state), and
    `aux_labels` (the students' auxiliary targets: the event truth and, on the load generator, the wrench); `groups`
    picks which the env returns."""
    return {k: v for k, v in _all_groups(task).items() if k in groups}


def _all_groups(task: str) -> dict[str, ObservationGroupCfg]:
    student = {"obs": ObservationTermCfg(func=student_obs)}
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
    return {"student": ObservationGroupCfg(dict(student)),
            "teacher": ObservationGroupCfg({**student, **priv}),
            "critic": ObservationGroupCfg({**student, **priv}),
            "aux_labels": ObservationGroupCfg(aux)}


def reward_terms(task: str) -> dict[str, RewardTermCfg]:
    """The plan's rewards for `task`, substep-integrated (tactile_rl.mdp.rewards): the effort penalties, the task's
    terms, the drop's and the thermal fault's penalties."""
    out = {k: RewardTermCfg(func=getattr(R, k), weight=w) for k, w in R.WEIGHTS_EFFORT.items()}
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


def dr_events(scene_key: str) -> dict[str, EventTermCfg]:
    """The reset randomization (tactile_rl.mdp.randomization), in the order they apply."""
    p = {"scene_key": scene_key}
    return {"dr_contact": EventTermCfg(func=pair_solref_solimp, mode="reset", params=p),
            "dr_drive": EventTermCfg(func=randomize_wuji_drive, mode="reset", params=p),
            "dr_taxelscan": EventTermCfg(func=randomize_taxelscan, mode="reset", params=p),
            "dr_plant": EventTermCfg(func=randomize_plant, mode="reset", params=p),
            "dr_forecast": EventTermCfg(func=randomize_forecast, mode="reset", params=p)}
