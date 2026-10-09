"""Closed loop with a served VLA in the CPU testbed: the VLA drives the arm and gives the residual its base.

VlaBase, on the 1 kHz host clock:

- at VlaCfg.fps it renders the head camera and reads the state, as the demos record them (vla.demos); every
  `exec_steps` frames it asks the policy for a chunk (H, 27) of absolute targets at the frame times from the
  request on;
- the chunk takes effect `latency` later in simulated time (the simulator waits for the server in wall time, so
  the server's own speed does not matter); from then the robot follows it, linearly interpolated between frames
  and held at its last action, until the next chunk takes effect;
- the arm through PositionL1's joint stream (the host still adds droop compensation and the rate limit); the hand
  as the residual's base command: the chunk's hand targets under the dataset's HandTable (vla.action);
- the lookahead block from the chunk (hand targets relative to now, arm targets relative to the arm's current
  target, at +10/+25/+50 ms); no forecast (the VLA has no forecast head), no trigger.

Every chunk is logged (request time, effect time, state, actions) so VLA runs become residual bases.

Policies: GrootPolicy speaks GR00T's policy server protocol (ZeroMQ REQ, msgpack, numpy as .npy bytes;
gr00t/eval/run_gr00t_server.py); OpenPiPolicy speaks openpi's (websocket, msgpack with raw numpy buffers;
scripts/serve_policy.py); ReplayPolicy serves a recorded demo's own actions, which checks the plumbing without a
model.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np

from tactile_sim.control.obs_spec import FORECAST_DIM, LOOKAHEAD, N_ARM, N_JOINTS
from tactile_sim.vla.action import ACTION_DIM, ARM, HAND, HandTable


# ---------------------------------------------------------------------- policies
class GrootPolicy:
    """A GR00T N1.7 policy server (run_gr00t_server.py) for the Vega U + WUJI modality config."""

    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 120_000,
                 language_key: str = "annotation.human.task_description"):
        import zmq

        self.zmq = zmq
        self.ctx = zmq.Context()
        self.addr = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.language_key = language_key
        self._connect()

    def _connect(self) -> None:
        self.sock = self.ctx.socket(self.zmq.REQ)
        self.sock.setsockopt(self.zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(self.zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(self.zmq.LINGER, 0)
        self.sock.connect(self.addr)

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return obj

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    def call(self, endpoint: str, data: dict | None = None):
        import msgpack

        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        try:
            self.sock.send(msgpack.packb(req, default=self._encode))
            msg = self.sock.recv()
        except self.zmq.error.Again:
            self.sock.close()
            self._connect()  # a REQ socket that timed out cannot send again
            raise
        out = msgpack.unpackb(msg, object_hook=self._decode)
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"policy server: {out['error']}")
        return out

    def reset(self) -> None:
        self.call("reset", {"options": None})

    def __call__(self, t: float, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        s = np.asarray(state, np.float32)
        obs = {"video": {"ego_view": np.asarray(image, np.uint8)[None, None]},
               "state": {"right_arm": s[None, None, ARM], "right_hand": s[None, None, HAND]},
               "language": {self.language_key: [[instruction]]}}
        action, _ = self.call("get_action", {"observation": obs, "options": None})
        return np.concatenate([action["right_arm"][0], action["right_hand"][0]], axis=-1).astype(float)


class OpenPiPolicy:
    """A pi0.5 policy served by openpi (scripts/serve_policy.py; websocket, msgpack with numpy as raw buffers -- the
    openpi_client wire format) for the Vega U + WUJI config in vla/finetune/openpi_vega_wuji.py."""

    def __init__(self, host: str = "localhost", port: int = 8000, timeout: float = 120.0):
        import websockets.sync.client

        self.ws = websockets.sync.client.connect(f"ws://{host}:{port}", compression=None, max_size=None,
                                                 open_timeout=timeout)
        self.timeout = timeout
        self.metadata = self._unpack(self.ws.recv(timeout=timeout))

    @staticmethod
    def _pack(obj):
        if isinstance(obj, np.ndarray):
            return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
        if isinstance(obj, np.generic):
            return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
        return obj

    @staticmethod
    def _unhook(obj):
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj

    def _unpack(self, data):
        import msgpack

        if isinstance(data, str):
            raise RuntimeError(f"policy server: {data}")
        return msgpack.unpackb(data, object_hook=self._unhook)

    def reset(self) -> None:
        pass

    def __call__(self, t: float, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        import msgpack

        obs = {"observation/image": np.asarray(image, np.uint8), "observation/state": np.asarray(state, np.float32),
               "prompt": instruction}
        self.ws.send(msgpack.packb(obs, default=self._pack))
        out = self._unpack(self.ws.recv(timeout=self.timeout))
        return np.asarray(out["actions"], float)[:, :ACTION_DIM]


class ReplayPolicy:
    """A recorded demo's own actions as chunks: the recording interpolated at the request time and every frame after
    (held before the first frame and after the last), so the replay keeps the demo's timing to the millisecond."""

    def __init__(self, actions: np.ndarray, fps: float, horizon: int = 16, t_start: float = 0.0):
        self.actions = np.asarray(actions, float)
        self.fps, self.horizon, self.t_start = fps, horizon, t_start

    def reset(self) -> None:
        pass

    def __call__(self, t: float, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        x = np.clip((t - self.t_start) * self.fps + np.arange(self.horizon), 0, len(self.actions) - 1)
        k = np.minimum(np.floor(x).astype(int), max(len(self.actions) - 2, 0))
        w = np.clip(x - k, 0.0, 1.0)[:, None]
        return (1 - w) * self.actions[k] + w * self.actions[np.minimum(k + 1, len(self.actions) - 1)]


# ---------------------------------------------------------------------- chunks
@dataclass
class Chunk:
    t_req: float  # time of the observation, and of action 0
    t_eff: float  # when the robot starts following it
    dt: float  # between actions
    actions: np.ndarray  # (H, 27)
    state: np.ndarray  # (27,) at the request

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Targets and their rate at time t: linear between actions, held before the first and after the last."""
        x = (t - self.t_req) / self.dt
        H = len(self.actions)
        if x <= 0:
            return self.actions[0].copy(), np.zeros(ACTION_DIM)
        if x >= H - 1:
            return self.actions[-1].copy(), np.zeros(ACTION_DIM)
        k = int(np.floor(x))
        a, b = self.actions[k], self.actions[k + 1]
        return a + (x - k) * (b - a), (b - a) / self.dt


@dataclass
class ChunkLog:
    t_req: list = field(default_factory=list)
    t_eff: list = field(default_factory=list)
    state: list = field(default_factory=list)
    actions: list = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, float) for k, v in self.__dict__.items()}


def smooth_chunk(a: np.ndarray, passes: int) -> np.ndarray:
    """A chunk low-passed along its horizon by `passes` of the [1, 2, 1]/4 kernel (its ends held): removes the
    frame-to-frame alternation a policy's samples can carry (the Nyquist rate, 15 Hz at 30 fps), which the arm would
    otherwise follow as acceleration spikes, and keeps slow motion."""
    for _ in range(passes):
        p = np.concatenate([a[:1], a, a[-1:]])
        a = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
    return a


# ---------------------------------------------------------------------- the base
class VlaBase:
    def __init__(self, tb, policy, table: HandTable, instruction: str, camera, latency: float = 0.1,
                 exec_steps: int = 8, fps: float | None = None, blend: float = 0.0, smooth: int = 0):
        self.tb, self.policy, self.table, self.instruction, self.cam = tb, policy, table, instruction, camera
        self.latency, self.exec_steps = latency, exec_steps
        self.blend = blend  # s over which a new chunk fades in over the one it replaces (0: it takes over at once)
        self.smooth = smooth  # passes of smooth_chunk over each chunk before it runs (0: as the policy gave it)
        self.fps = fps or tb.cfg.vla.fps
        self.dt = 1.0 / self.fps
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.chunks: list[Chunk] = []
        self.log = ChunkLog()
        self.k = 0  # frames seen
        self.t0: float | None = None
        l1 = self.tb.l1
        self.q_arm0 = l1.q_ref.copy()
        self.q_hand0: np.ndarray | None = None

    def attach(self) -> None:
        """Hand the arm and the residual's base to the VLA: no scripted swing, the arm on the joint stream, this
        object as the residual's base_fn, and a 1 kHz frame task (registered after the residual's)."""
        tb = self.tb
        tb.l2_callbacks[:] = []
        tb.l1.set_joint_stream(self.arm_stream)
        tb.residual.base_fn = self
        tb.sched.register("vla", 1000.0, self.frame)

    # ------------------------------------------------------------------ frames and requests
    def state(self) -> np.ndarray:
        sen = self.tb.sensors
        return np.concatenate([sen["joint_pos"].latest().value, sen["hand_q"].latest().value]).astype(float)

    def frame(self, t: float) -> None:
        sen = self.tb.sensors
        if self.t0 is None:
            if sen["joint_pos"].latest().seq < 0 or sen["hand_q"].latest().seq < 0:
                return
            self.t0 = t
            self.q_hand0 = np.asarray(sen["hand_q"].latest().value, float)
        if t + 1e-9 < self.t0 + self.k * self.dt:
            return
        if self.k % self.exec_steps == 0:
            state = self.state()
            image = self.cam.render(self.tb.world.data)
            a = np.asarray(self.policy(t, image, state, self.instruction), float)
            if a.ndim != 2 or a.shape[1] != ACTION_DIM or not np.all(np.isfinite(a)):
                raise ValueError(f"policy returned {a.shape} actions (finite: {np.all(np.isfinite(a))})")
            a = smooth_chunk(a, self.smooth)
            c = Chunk(t, t + self.latency, self.dt, a, state)
            self.chunks.append(c)
            L = self.log
            L.t_req.append(t)
            L.t_eff.append(c.t_eff)
            L.state.append(state)
            L.actions.append(a)
        self.k += 1

    def active(self, t: float) -> int | None:
        """Index of the chunk the robot follows at t."""
        for i in range(len(self.chunks) - 1, -1, -1):
            if self.chunks[i].t_eff <= t + 1e-12:
                return i
        return None

    # ------------------------------------------------------------------ outputs
    def targets(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """The 27 targets and their rates at t: the active chunk, faded in over the previous one for `blend` s
        (smoothstep; consecutive chunks disagree, and a step in the arm's targets jolts the tool in the hand), or
        the pose at the start until one arrives."""
        i = self.active(t)
        if i is None:
            hand = self.q_hand0 if self.q_hand0 is not None else self.tb.sensors["hand_q"].latest().value
            return np.concatenate([self.q_arm0, hand]), np.zeros(ACTION_DIM)
        q, qd = self.chunks[i].at(t)
        s = (t - self.chunks[i].t_eff) / self.blend if self.blend > 0 and i > 0 else 1.0
        if s < 1.0:
            q_old, qd_old = self.chunks[i - 1].at(t)
            w, dw = s * s * (3.0 - 2.0 * s), 6.0 * s * (1.0 - s) / self.blend
            qd = qd_old + w * (qd - qd_old) + dw * (q - q_old)
            q = q_old + w * (q - q_old)
        return q, qd

    def arm_stream(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        q, qd = self.targets(t)
        return q[ARM], qd[ARM]

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        q, _ = self.targets(t)
        if self.active(t) is None:
            # until the first chunk takes effect the hand holds where it is at the table's squeeze (the scripted base's
            # own grip, HandTable.for_hand); an unread sensor (the first ms) gives zero angles, targets that would open
            # the hand, so until its first reading the squeeze alone
            reading = self.tb.sensors["hand_q"].latest()
            base = self.table.hold(np.asarray(reading.value, float) if reading.seq >= 0 else None)
        else:
            base = self.table.command(q[HAND])
        look = []
        for dt in LOOKAHEAD:
            qa, _ = self.targets(t + dt)
            look.append(np.concatenate([qa[HAND] - q[HAND], qa[ARM] - self.tb.l1.q_ref]))
        look = np.concatenate(look)
        assert look.shape == ((N_JOINTS + N_ARM) * len(LOOKAHEAD),)
        return base, look, False, np.zeros(FORECAST_DIM)


# ---------------------------------------------------------------------- episodes
def strike_events(truth: dict, thresh: float = 5.0, gap: float = 0.15) -> list[int]:
    """Indices into the truth log where hammer-nail contact starts (force above thresh after gap of none)."""
    t, f = truth["t"], truth["contact_force"]
    on = np.flatnonzero(f > thresh)
    out, last = [], -np.inf
    for i in on:
        if t[i] - last > gap:
            out.append(int(i))
        last = t[i]
    return out


def strike_metrics(truth: dict, pre: float = 0.02, post: float = 0.1) -> list[dict]:
    """Per strike: nail advance, and the tool's slip in the hand from just before contact to `post` after."""
    t = truth["t"]
    out = []
    ev = strike_events(truth)
    for n, i in enumerate(ev):
        i0 = max(0, int(np.searchsorted(t, t[i] - pre)) - 1)
        i1 = min(len(t) - 1, int(np.searchsorted(t, t[i] + post)))
        j = ev[n + 1] if n + 1 < len(ev) else len(t) - 1
        p, r = truth["hammer_in_hand_pos"], truth["hammer_in_hand_rotvec"]
        out.append({"t_contact": float(t[i]), "peak_force": float(truth["contact_force"][i:i1 + 1].max()),
                    "advance": float(truth["nail_depth"][j] - truth["nail_depth"][i0]),
                    "slip_trans": float(np.linalg.norm(p[i1] - p[i0])),
                    "slip_rot_deg": float(np.degrees(np.linalg.norm(r[i1] - r[i0])))})
    return out


def run_vla_episode(cfg, policy, seed: int, latency: float = 0.1, exec_steps: int = 8, max_time: float = 30.0,
                    residual_policy=None, visual_seed: int | None = None, after_done: float = 1.0,
                    blend: float = 0.0, smooth: int = 0, table: HandTable | None = None,
                    sensor_noise_scale: float | None = None) -> dict:
    """One episode with the VLA in charge, on the configured task (ResidualCfg.task): until the nail is home or the
    screw seated (then `after_done` more seconds, for the VLA to let go and back out), the tool drops, or max_time.
    The residual runs `residual_policy` (default: the configured one, zero unless set)."""
    from tactile_sim.logging.strike_metrics import rotvec_diff
    from tactile_sim.vla.camera import EgoCamera
    from tactile_sim.vla.demos import INSTRUCTIONS

    task = cfg.residual.task
    if task == "drill":
        from tactile_sim.tool_task import ToolTask

        tb = ToolTask(cfg, seed=seed, t_max=max_time).tb  # the drill's testbed; the VLA replaces its feed
        step_cb = None
    else:
        from tactile_sim.episode import Episode

        ep = Episode(cfg, seed=seed)
        tb = ep.tb
        step_cb = ep._step_cb  # noqa: SLF001 - the episode's truth recorder, without its scripted swing
    if sensor_noise_scale is not None:  # the seed's draw, as the demos ran it (demo_config's dr["_runtime"])
        tb.sensors.set_noise_scale(sensor_noise_scale)
    if residual_policy is not None:
        tb.residual.check_policy(residual_policy)
        tb.residual.policy = residual_policy
    cam = EgoCamera(tb.world.model, cfg)
    if cfg.vla.visual_dr:
        cam.randomize(np.random.default_rng(20_000 + (seed if visual_seed is None else visual_seed)))
    # the hand table the VLA's actions were recorded under: the given one (a dataset's, LeRobot sidecar
    # meta/vega_wuji.json "hand_tables"), else the task's own (as the demos build it)
    table = table or HandTable.for_hand(tb.world.hand, cfg)
    vla = VlaBase(tb, policy, table, INSTRUCTIONS[task], cam, latency, exec_steps, blend=blend, smooth=smooth)
    vla.attach()
    w, plant = tb.world, tb.world.plant
    p0, r0 = w.hammer_in_hand()
    slip_t = slip_r = 0.0
    aim, trig = np.inf, 0.0  # the drill: the bit tip's closest approach to the screw's axis, the deepest pull
    t_done = float("nan")
    n_max = int(round(max_time / w.dt))
    try:
        for k in range(n_max):
            tb.step()
            if step_cb is not None:
                step_cb(tb)
            elif k % 8 == 0:  # the drill: the tool's travel in the hand, and its aim
                p, r = w.hammer_in_hand()
                slip_t, slip_r = max(slip_t, float(np.linalg.norm(p - p0))), max(slip_r, rotvec_diff(r0, r))
                if plant.tip_ax > -0.01:  # within 10 mm of the head, along the axis
                    aim = min(aim, plant.tip_lat)
                trig = max(trig, plant.trigger)
            if np.isnan(t_done) and plant.done():
                t_done = float(w.t)
            if tb.grip.dropped or (np.isfinite(t_done) and w.t - t_done >= after_done):
                break
    finally:
        cam.close()
    out = {"seed": seed, "task": task, "t_end": float(w.t), "done": bool(plant.done()), "t_done": t_done,
           "dropped": bool(tb.grip.dropped), "depth": float(plant.depth), "n_chunks": len(vla.chunks),
           "residual": {"ticks": tb.residual.n_ticks, "stale": tb.residual.n_stale,
                        "nonfinite": tb.residual.n_nonfinite},
           "chunks": vla.log.arrays()}
    if task == "drill":
        out.update(cam_outs=int(plant.cam_outs), state=plant.state, max_slip_trans_mm=1e3 * slip_t,
                   max_slip_rot_deg=float(np.degrees(slip_r)), closest_aim_mm=1e3 * aim, max_trigger=trig)
    else:
        out["strikes"] = strike_metrics(ep.truth.arrays())
    return out
