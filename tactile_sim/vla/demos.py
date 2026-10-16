"""Scripted demos for the VLA (L3) on the Vega U + WUJI Hand 2.

Each episode runs a task's scripted base in the CPU testbed the experiment evaluates in (MIT drives, the TaxelScan
fast profile, the residual slot at zero), with the residual experiment's physics randomization, the task shifted on
the table (TASK_SHIFT), and a VisualDR draw: the hammer's swing (its strike speed varied) until the nail is home, or
the drill's feed driving a screw, the trigger pulled by the index finger (DrillCfg trigger_mode "finger"). The head
stays aimed at the unshifted task, so the task moves in the image. A recorder on the 1 kHz clock samples, at
VlaCfg.fps (frame k at the first millisecond at or after k / fps):

- the head camera, streamed to an h264 mp4;
- the state: the arm's joint_pos and the hand's hand_q, as the host reads them;
- the action: the arm's joint targets (PositionL1.q_ref, before droop compensation) and the hand's
  table-equivalent targets of the command the hand was sent (vla.action.HandTable).

The vectors, timing, outcome and draws go to an .npz beside the mp4; vla.lerobot_export turns a directory of
them into a LeRobot v2.1 dataset.

    python -m tactile_sim.vla.demos --task hammer --episodes 500 --workers 14 --out runs/vla_demos/hammer
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INSTRUCTIONS = {"hammer": "hammer the nail into the board", "drill": "drive the screw into the board"}
TASK_SHIFT = 0.03  # m, uniform on each horizontal axis: the whole task (board, nail, hover) moved on the table
STRIKE_SPEED = (0.85, 1.15)  # x the nominal strike speed
MAX_STRIKES = 20  # an episode ends when the nail is home, or after this many strikes
APPROACH_TIME = 0.05  # s, the swing's hold before the first windup (its default 0.6, plus 0.1)
DRILL_T_MAX = 10.0  # s: a screw that has not seated by then is a failed demo
VIDEO_KEY = "observation.images.ego_view"
START = 0.02  # s: recording starts after the grip loop and the arm's first commands have run
# DemoPerturbation: per joint scale of a push (rad; shoulders and elbow, then the wrist) and its duration (s)
PUSH_SCALE = np.array([0.06, 0.06, 0.06, 0.06, 0.10, 0.10, 0.10])
PUSH_TIME = (0.30, 0.55)


def ffmpeg_exe() -> str:
    import shutil

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("no ffmpeg: pip install imageio-ffmpeg") from None
        return exe


class VideoWriter:
    """RGB frames piped to ffmpeg as h264 / yuv420p (what GR00T's and LeRobot's decoders read), a keyframe every
    `gop` frames for random access."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 18, gop: int = 15):
        self.path = Path(path)
        self.size = (height, width, 3)
        self.n = 0
        self.proc = subprocess.Popen(
            [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
             "-g", str(gop), "-f", "mp4", str(self.path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self.size or frame.dtype != np.uint8:
            raise ValueError(f"frame {frame.shape} {frame.dtype}, expected {self.size} uint8")
        self.proc.stdin.write(frame.tobytes())
        self.n += 1

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


class ImageStats:
    """Per-channel pixel statistics over [0, 1]-scaled frames, with 256-bin histograms for exact quantiles."""

    def __init__(self):
        self.hist = np.zeros((3, 256), dtype=np.int64)

    def update(self, frame: np.ndarray) -> None:
        for c in range(3):
            self.hist[c] += np.bincount(frame[..., c].ravel(), minlength=256)

    @staticmethod
    def summarize(hist: np.ndarray) -> dict:
        """mean / std / min / max / q01 / q99 per channel, shaped (3, 1, 1) as LeRobot keeps image stats."""
        x = np.arange(256) / 255.0
        n = hist.sum(axis=1)
        mean = (hist * x).sum(axis=1) / n
        std = np.sqrt(np.maximum((hist * x**2).sum(axis=1) / n - mean**2, 0.0))
        cdf = np.cumsum(hist, axis=1) / n[:, None]

        def q(p):
            return np.array([x[np.searchsorted(cdf[c], p)] for c in range(3)])

        lo = np.array([x[np.flatnonzero(hist[c])[0]] for c in range(3)])
        hi = np.array([x[np.flatnonzero(hist[c])[-1]] for c in range(3)])

        def shp(v):
            return np.asarray(v, float).reshape(3, 1, 1).tolist()

        return {"mean": shp(mean), "std": shp(std), "min": shp(lo), "max": shp(hi), "q01": shp(q(0.01)),
                "q99": shp(q(0.99)), "count": [int(n[0])]}


class DemoRecorder:
    """The VLA's view, state and action at `fps`, sampled on the 1 kHz host clock (register after the residual, so
    a tick sees the command the hand was just sent)."""

    def __init__(self, tb, camera, table, fps: float, video: VideoWriter):
        self.tb, self.cam, self.table, self.video = tb, camera, table, video
        self.period = 1.0 / fps
        self.t0: float | None = None
        self.k = 0
        self.t: list[float] = []
        self.state: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.stats = ImageStats()

    def __call__(self, t: float) -> None:
        sen = self.tb.sensors
        if self.t0 is None:
            # the first frame once the host has read every input and the hand has been commanded under its grip
            if t < START or self.tb.residual.last_cmd is None or sen["joint_pos"].latest().seq < 0 or \
                    sen["hand_q"].latest().seq < 0:
                return
            self.t0 = t
        if t + 1e-9 < self.t0 + self.k * self.period:
            return
        cmd = self.tb.residual.last_cmd
        q_arm = np.asarray(sen["joint_pos"].latest().value, float)
        q_hand = np.asarray(sen["hand_q"].latest().value, float)
        hand_target = self.table.equivalent_targets(cmd.array(), q_hand)
        frame = self.cam.render(self.tb.world.data)
        self.video.write(frame)
        self.stats.update(frame)
        self.t.append(t)
        self.state.append(np.concatenate([q_arm, q_hand]))
        self.action.append(np.concatenate([self.tb.l1.q_ref, hand_target]))
        self.k += 1


class DemoPerturbation:
    """DART-style recoveries for the demos: now and then the arm's servos get a smooth push (a raised-cosine bump in
    joint space, PUSH_SCALE x U(0.4, 1) along a random direction, PUSH_TIME long) that the scripted base does not
    know about, while the recorded action stays the base's own target (PositionL1.q_ref, through exec_offset). The
    demos then show the arm off its path and the expert's target pulling it back.

    Pushes start only where the task can take them: for the hammer, at the start of a strike's settle, ending before
    the next downswing (settle + windup); for the drill, in the approach, ending before the bit meets the screw. Each
    such window starts a push with probability `p`."""

    def __init__(self, task: str, base, p: float, rng: np.random.Generator):
        self.task, self.base, self.p, self.rng = task, base, p, rng
        self.bumps: list[tuple[float, float, np.ndarray]] = []
        self._window: object = None  # the window the last decision was made in

    def _window_now(self, t: float) -> tuple[object, float] | None:
        """(an id for the current window, the time left in it) if pushes may start now."""
        if self.task == "hammer":
            sw = self.base.swing
            c = self.base.tb.cfg.swing
            if sw.phase == "settle":
                left = c.settle_time - (t - sw.phase_t0) + c.windup_time + 0.05
                return (sw.k, sw.phase_t0), left - 0.05
            return None
        feed = self.base.feed
        if feed.phase == "approach":
            left = self.base.tb.cfg.drill.approach_time - (t - feed.t_phase)
            return ("approach", feed.t_phase), left - 0.05
        return None

    def __call__(self, t: float) -> np.ndarray:
        w = self._window_now(t)
        if w is not None and w[0] != self._window:
            self._window = w[0]
            if self.rng.uniform() < self.p and w[1] >= PUSH_TIME[0]:
                d = self.rng.normal(size=len(PUSH_SCALE))
                dq = PUSH_SCALE * self.rng.uniform(0.4, 1.0) * d / np.linalg.norm(d)
                self.bumps.append((t, float(min(self.rng.uniform(*PUSH_TIME), w[1])), dq))
        off = np.zeros(len(PUSH_SCALE))
        for t0, dur, dq in self.bumps:
            if t0 <= t <= t0 + dur:
                off += dq * 0.5 * (1.0 - np.cos(2.0 * np.pi * (t - t0) / dur))
        return off

    def to_list(self) -> list[dict]:
        return [{"t0": t0, "duration": dur, "dq": dq.tolist()} for t0, dur, dq in self.bumps]


@dataclass(frozen=True)
class DemoJob:
    task: str
    index: int
    seed: int
    out: str
    fast: bool = False
    hammer: str = "ycb"  # config.HAMMERS: the real YCB hammer, or "light" (datasets v1-v3)
    perturb: float = 0.0  # DemoPerturbation's chance of a push per window (0: none, datasets up to v4)

    @property
    def stem(self) -> Path:
        return Path(self.out) / f"episode_{self.index:06d}"


def demo_config(task: str, seed: int, fast: bool = False, hammer: str = "ycb"):
    """One episode's configuration and what was drawn for it: the hammer task with `hammer` (config.HAMMERS; the
    datasets before v4 used the "light" one)."""
    from tactile_sim.config import fast_config, residual_config
    from tactile_sim.sim.randomize import apply_residual_dr, residual_dr
    from tactile_sim.vla.camera import vla_config

    cfg = vla_config(residual_config(task=task, base=fast_config() if fast else None, hammer_kind=hammer))
    rng = np.random.default_rng(seed)
    dr = residual_dr(cfg, rng)
    cfg = apply_residual_dr(cfg, dr)
    shift = rng.uniform(-TASK_SHIFT, TASK_SHIFT, 2)
    over = {"scene": {"hover_tcp": tuple(float(v) for v in np.asarray(cfg.scene.hover_tcp) + [*shift, 0.0])}}
    if task == "hammer":
        # the first windup starts at once: an idle hold looks the same in every frame, so a policy cannot tell when
        # it ends (the grasp has settled before the clock starts, in World.reset)
        over["swing"] = {"v_strike": float(cfg.swing.v_strike * rng.uniform(*STRIKE_SPEED)),
                         "approach_time": APPROACH_TIME}
    return cfg.replace(**over), {"dr": dr, "overrides": over}


def _hammer_episode(cfg, seed: int):
    """The scripted swing until the nail is home (or MAX_STRIKES): testbed, base, run() -> outcome."""
    from tactile_sim.behaviors.base_sources import SwingBase
    from tactile_sim.episode import Episode

    ep = Episode(cfg, seed=seed)
    base = SwingBase(ep.tb, ep.swing, rng=np.random.default_rng(10_000 + seed))

    def run() -> dict:
        res = ep.run(MAX_STRIKES)
        tb = ep.tb
        return {"success": bool(tb.world.plant.done() and not tb.grip.dropped), "dropped": bool(tb.grip.dropped),
                "n_strikes": len(res.strikes)}

    return ep.tb, base, run


def _drill_episode(cfg, seed: int):
    """The scripted drill feed (screw): approach, press, pull the trigger with the index finger, release when the
    clutch clicks, back out."""
    from tactile_sim.behaviors.base_sources import DrillBase
    from tactile_sim.tool_task import ToolTask

    tt = ToolTask(cfg, seed=seed, t_max=DRILL_T_MAX)
    base = DrillBase(tt.tb, tt.behavior, rng=np.random.default_rng(10_000 + seed))

    def run() -> dict:
        s = tt.run().summary
        return {"success": bool(s["done"] and not s["lost"]), "dropped": bool(s["lost"]), "t_done": s["t_done"],
                "cam_outs": s["cam_outs"]}

    return tt.tb, base, run


EPISODES = {"hammer": _hammer_episode, "drill": _drill_episode}


def run_demo(job: DemoJob) -> dict:
    from tactile_sim.vla.action import HandTable
    from tactile_sim.vla.camera import EgoCamera

    npz = job.stem.with_suffix(".npz")
    if npz.exists():
        return {"status": "skip", "path": str(npz)}
    t_wall = time.time()
    cfg, draw = demo_config(job.task, job.seed, job.fast, job.hammer)
    tb, base, run = EPISODES[job.task](cfg, job.seed)
    tb.sensors.set_noise_scale(draw["dr"]["_runtime"]["sensor_noise_scale"])
    tb.residual.base_fn = base
    pert = None
    if job.perturb > 0:
        pert = DemoPerturbation(job.task, base, job.perturb, np.random.default_rng(30_000 + job.seed))
        tb.l1.exec_offset = pert
    cam = EgoCamera(tb.world.model, cfg)
    visual = cam.randomize(np.random.default_rng(20_000 + job.seed)) if cfg.vla.visual_dr else None
    table = HandTable.for_hand(tb.world.hand, cfg)
    v = cfg.vla
    mp4 = job.stem.with_suffix(".mp4")
    mp4.parent.mkdir(parents=True, exist_ok=True)
    video = VideoWriter(mp4.with_suffix(".mp4.part"), v.cam_width, v.cam_height, v.fps)
    rec = DemoRecorder(tb, cam, table, v.fps, video)
    tb.sched.register("vla_demo", 1000.0, rec)
    try:
        outcome = run()
    finally:
        video.close()
        cam.close()
    meta = {"task": job.task, "instruction": INSTRUCTIONS[job.task], "seed": job.seed, "fps": v.fps, **outcome,
            "depth": float(tb.world.plant.depth), "draw": draw, "visual": visual, "hand_table": table.to_dict(),
            "hand_joints": list(tb.world.hand.joint_names),
            "head_pose": list(cfg.arm.head_pose), "config": cfg.to_dict(), "wall_s": time.time() - t_wall,
            "perturb": job.perturb, "pushes": pert.to_list() if pert is not None else []}
    mp4.with_suffix(".mp4.part").rename(mp4)
    tmp = npz.with_suffix(".part.npz")
    np.savez(tmp, state=np.asarray(rec.state, np.float32), action=np.asarray(rec.action, np.float32),
             t=np.asarray(rec.t), image_hist=rec.stats.hist, meta=json.dumps(meta, default=_jsonable))
    tmp.rename(npz)
    return {"status": "ok", "path": str(npz), "success": outcome["success"], "frames": rec.k,
            "wall_s": meta["wall_s"]}


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def load_demo(npz: Path) -> dict:
    with np.load(npz) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def prewarm(cfgs) -> None:
    """Synthesize every grasp keyframe the episodes need once, serially, before they run in parallel."""
    from tactile_sim.assets.fetch_hands import hand_xml
    from tactile_sim.model.hands.wuji_grasp import cache_path, wrap_grasp

    hand = hand_xml("wuji2")
    seen = set()
    for cfg in cfgs:
        key = cache_path(cfg, hand)
        if key not in seen:
            seen.add(key)
            wrap_grasp(cfg, hand)


def run_all(jobs: list[DemoJob], workers: int) -> None:
    todo = [j for j in jobs if not j.stem.with_suffix(".npz").exists()]
    print(f"{len(jobs)} demos, {len(jobs) - len(todo)} done, {len(todo)} to run on {workers} workers", flush=True)
    if not todo:
        return
    prewarm([demo_config(j.task, j.seed, j.fast, j.hammer)[0] for j in todo])
    t0 = time.time()
    n_ok = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        futs = {ex.submit(run_demo, j): j for j in todo}
        for k, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                r = f.result()
                n_ok += bool(r.get("success"))
                msg = f"{'ok' if r.get('success') else 'FAILED'} {r.get('frames', 0)} frames {r.get('wall_s', 0):.0f}s"
            except Exception:  # noqa: BLE001 - report and keep going
                msg = "ERROR\n" + traceback.format_exc()
            print(f"[{k}/{len(todo)}] {j.task} #{j.index} seed {j.seed}: {msg} ({n_ok} successful) "
                  f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", default="hammer", choices=sorted(INSTRUCTIONS))
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--first", type=int, default=0, help="first episode index (seed = seed0 + index)")
    ap.add_argument("--seed0", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--fast", action="store_true", help="the fast physics config (no noslip, 4 kHz)")
    ap.add_argument("--hammer", default="ycb", choices=("ycb", "light"),
                    help="the hammer task's hammer: the real YCB claw hammer, or the light primitive one "
                         "(datasets v1-v3)")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="DemoPerturbation: chance of a recovery push per window (0: none)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or f"runs/vla_demos/{a.task}"
    jobs = [DemoJob(a.task, i, a.seed0 + i, out, a.fast, a.hammer, a.perturb)
            for i in range(a.first, a.first + a.episodes)]
    run_all(jobs, a.workers)


if __name__ == "__main__":
    main()
