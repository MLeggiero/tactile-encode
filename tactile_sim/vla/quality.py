"""Quality gates for recorded VLA demos (vla.demos): what a demo must meet, beyond completing its task, to be exported.

A demo that drives the nail home can still teach the wrong thing: an arm faster than the real Vega allows, a hammer
twisting in the hand, blows landing off the nail head, or action labels with steps and alternation that the model
then reproduces (the documented closed-loop failure is chunk-to-chunk alternation, docs/vla_in_sim.md). Each demo's
metrics and the reasons it fails are kept in its metadata; vla.lerobot_export skips failing demos with the failed
ones, and vla.coverage counts the reasons.

- Arm limits (limits.LimitMonitor over the episode): joint velocity, range and torque within their ratings. The WUJI's
  hard-stop and joint loads are recorded but not gated: the power wrap rests its knuckles on their stops and normal
  blows load them past the joint's rating (README "Findings"), so a gate would reject every hammer demo.
- The tool in the hand: no strike (hammer) or episode (drill) slips or turns the tool more than the thresholds.
- Blows (hammer): the share of blows that land off the nail head (the face centre farther from the nail's axis than the
  face radius less the head's) or miss it.
- Labels: the arm's recorded targets never ask a joint to move faster than its velocity limit between frames, and
  rarely oscillate: frames where a joint's second difference (above `alt_min`) flips sign `alt_run` times in a row.
  A clean strike flips once at each change of phase (windup, swing, brake, recover); the closed-loop failure flips on
  every frame. Real scripted and planned demos score 0, a 0.03 rad frame-to-frame alternation 1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ARM_LIMITS = ("arm_velocity", "arm_range", "arm_torque")


@dataclass(frozen=True)
class QualityCfg:
    strike_slip_trans: float = 0.007  # m, per strike (the hammer's grasp target, README)
    strike_slip_rot: float = 5.0  # deg, per strike (scripted and planned demos: under 3 deg)
    drill_slip_trans: float = 0.005  # m, over the episode
    # deg, over the episode, the pads' elastic twist included: the clutch's ratchet rolls the drill in the fingers, and
    # scripted and planned demos alike turn it 4.7-6.5 deg (2026-10-31, seeds 100-103)
    drill_slip_rot: float = 10.0
    off_centre_share: float = 0.25  # of the blows
    target_rate: float = 1.0  # x the joint velocity limit, between recorded frames
    alt_min: float = 0.005  # rad: a second difference this large counts toward alternation
    alt_run: int = 3  # consecutive sign flips that make an oscillation
    alternation: float = 0.05  # share of frames inside an oscillation


DEFAULT = QualityCfg()


def label_metrics(action: np.ndarray, fps: float, v_limit: np.ndarray, alt_min: float, alt_run: int = 3) -> dict:
    """Smoothness of the recorded arm targets (the first 7 action columns): the fastest move between frames as a
    multiple of the joint's velocity limit, and the share of frames where a joint's second difference, above alt_min,
    flips sign alt_run times in a row (sustained frame-to-frame oscillation)."""
    a = np.asarray(action, float)[:, :7]
    if len(a) < 4:
        return {"target_rate": 0.0, "alternation": 0.0}
    rate = float(np.max(np.abs(np.diff(a, axis=0)) * fps / np.asarray(v_limit, float)))
    dd = np.diff(a, n=2, axis=0)
    big = np.abs(dd) > alt_min
    flips = (np.sign(dd[1:]) != np.sign(dd[:-1])) & big[1:] & big[:-1]
    n = len(flips) - alt_run + 1
    if n <= 0:
        return {"target_rate": rate, "alternation": 0.0}
    run = np.ones_like(flips[:n])
    for k in range(alt_run):
        run &= flips[k:k + n]
    return {"target_rate": rate, "alternation": float(np.mean(run.any(axis=1)))}


def hammer_metrics(result, cfg) -> dict:
    """Per-episode metrics from an Episode.run result (EpisodeResult)."""
    from tactile_sim.model.tool_hammer import hammer_geometry

    allow = max(hammer_geometry(cfg.hammer).face_radius - cfg.plant.head_radius, 0.0)
    recs = result.strikes
    aims = np.array([r.extra.get("aim_offset", np.nan) for r in recs], float)
    hits = np.array([r.hit for r in recs], bool)
    trans = np.array([r.slip_trans for r in recs], float)
    rot = np.degrees(np.array([r.slip_rot for r in recs], float))
    return {"n_blows": int(len(recs)), "missed": int((~hits).sum()),
            "off_centre": int(np.sum(hits & np.isfinite(aims) & (aims > allow))),
            "max_aim_offset_mm": float(np.nanmax(aims) * 1e3) if np.isfinite(aims).any() else float("nan"),
            "max_strike_slip_mm": float(np.nanmax(trans) * 1e3) if np.isfinite(trans).any() else float("nan"),
            "max_strike_rot_deg": float(np.nanmax(rot)) if np.isfinite(rot).any() else float("nan"),
            **_limits(result.summary)}


def drill_metrics(summary: dict) -> dict:
    """Per-episode metrics from a ToolTask summary."""
    return {"max_slip_mm": float(summary["max_slip_trans"] * 1e3),
            "max_rot_deg": float(np.degrees(summary["max_slip_rot"])), "cam_outs": int(summary.get("cam_outs", 0)),
            **_limits(summary)}


def _limits(summary: dict) -> dict:
    return {f"limit_{k}": float(v) for k, v in ((k[6:], v) for k, v in summary.items() if k.startswith("limit_"))
            if isinstance(v, (int, float))}


def gate(task: str, m: dict, q: QualityCfg | None = None) -> list[str]:
    """The reasons a demo fails the gates, each "gate: detail" (empty: it passes)."""
    q = q or DEFAULT
    out = []
    for k in ARM_LIMITS:
        if m.get(f"limit_{k}", 0.0) > 1.0 + 1e-6:
            out.append(f"{k}: {m[f'limit_{k}']:.2f}x")
    if task == "hammer":
        if m.get("max_strike_slip_mm", 0.0) > q.strike_slip_trans * 1e3:
            out.append(f"strike_slip: {m['max_strike_slip_mm']:.1f} mm")
        if m.get("max_strike_rot_deg", 0.0) > q.strike_slip_rot:
            out.append(f"strike_turn: {m['max_strike_rot_deg']:.1f} deg")
        n = max(m.get("n_blows", 0), 1)
        if (m.get("off_centre", 0) + m.get("missed", 0)) / n > q.off_centre_share:
            out.append(f"blows: {m.get('off_centre', 0)} off-centre and {m.get('missed', 0)} missed of {n}")
    else:
        if m.get("max_slip_mm", 0.0) > q.drill_slip_trans * 1e3:
            out.append(f"slip: {m['max_slip_mm']:.1f} mm")
        if m.get("max_rot_deg", 0.0) > q.drill_slip_rot:
            out.append(f"turn: {m['max_rot_deg']:.1f} deg")
        if m.get("cam_outs", 0) > 0:
            out.append(f"cam_outs: {m['cam_outs']}")
    if m.get("target_rate", 0.0) > q.target_rate:
        out.append(f"target_rate: {m['target_rate']:.2f}x")
    if m.get("alternation", 0.0) > q.alternation:
        out.append(f"alternation: {m['alternation']:.2f}")
    return out


def assess(task: str, metrics: dict, action: np.ndarray, fps: float, v_limit: np.ndarray,
           q: QualityCfg | None = None) -> dict:
    """The demo's quality record: its metrics (episode and labels), whether it passes, and why not."""
    q = q or DEFAULT
    m = dict(metrics, **label_metrics(action, fps, v_limit, q.alt_min, q.alt_run))
    reasons = gate(task, m, q)
    return {"passed": not reasons, "reasons": reasons, "metrics": m, "thresholds": asdict(q)}
