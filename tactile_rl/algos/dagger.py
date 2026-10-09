"""DAgger distillation of a privileged teacher into the student (the plan's pipeline, step 2).

Rollouts run in Tactile-Residual's C3 (70 % hammer swings, 30 % load generator, the base at 0.3-0.8 of its hold, the
full randomization). Each env episode is driven either by the teacher (with probability beta, drawn at every reset) or
by the student; the teacher labels every step. The schedule follows the plan: behaviour cloning first (beta = 1:
the teacher drives everything), then beta falls linearly to 0, then on-policy (the student drives, the teacher only
labels). One rollout of `rollout_s` (0.128 s) per iteration, as the teachers' PPO, so iterations are comparable in
simulated time across rates.

Loss on every step (truncated backpropagation through each env's rollout, the student's state carried across
rollouts and zeroed at episode starts):
- Huber on the squashed action (tanh of the means, the action the env applies) against the teacher's, divided by
  its threshold (smooth L1 with beta = `huber_delta`: quadratic below it, the absolute error above);
- the auxiliary targets, from the teacher's privileged observation: each patch's slip over the next 20 ms (the
  tool's tangential velocity over the skin at the patch's contacts, summed over the next 20 ms of steps; masked
  where the window leaves the rollout or the episode), the load regime (the base's true plan: 5 regimes, none when
  no event is planned), contact (each patch carrying > 0.5 N; the tool on the workpiece).

Logged per iteration (tensorboard): the losses, beta, the learning rate, the episode return, and the env's own
metrics under their usual tags (Metrics/base/*, Episode_Metrics/*), so tactile_rl.bench.curves reads a student run as
it reads a teacher's.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from tactile_rl.algos.student_model import AUX_LAYOUT, REGIME_CLASSES, StudentNet, count_parameters

N_SLIP = AUX_LAYOUT["slip_next_mm"]
N_CONTACT = AUX_LAYOUT["contact_logits"]


@dataclass
class DaggerCfg:
    rate: float = 1000.0
    num_envs: int = 4096
    iterations: int = 300
    bc_iterations: int = 60  # beta = 1
    mix_iterations: int = 100  # beta from 1 to 0, then on-policy
    rollout_s: float = 0.128
    epochs: int = 2
    minibatches: int = 8
    lr: float = 5e-4
    lr_final: float = 1e-4  # cosine to this over the run
    huber_delta: float = 0.1  # on squashed actions
    w_slip: float = 0.5
    w_regime: float = 0.05
    w_contact: float = 0.05
    slip_horizon_s: float = 0.02
    contact_n: float = 0.5  # N: a patch in contact
    max_grad_norm: float = 1.0
    norm_until: int = 100  # iterations the student's normalizer keeps updating
    hidden: int = 192
    seed: int = 1
    save_every: int = 50


def beta_at(cfg: DaggerCfg, it: int) -> float:
    if it < cfg.bc_iterations:
        return 1.0
    k = it - cfg.bc_iterations
    return max(0.0, 1.0 - k / max(cfg.mix_iterations, 1))


def lr_at(cfg: DaggerCfg, it: int) -> float:
    c = 0.5 * (1.0 + math.cos(math.pi * min(it / max(cfg.iterations - 1, 1), 1.0)))
    return cfg.lr_final + (cfg.lr - cfg.lr_final) * c


def teacher_label_slices(student_dim: int) -> dict[str, slice]:
    """Where the privileged blocks the labels read sit in the mixed task's teacher observation."""
    from tactile_rl.mdp.observations import PRIVILEGED_BY_TASK

    out, k = {}, student_dim
    for name, w in PRIVILEGED_BY_TASK["mixed"].items():
        out[name] = slice(k, k + w)
        k += w
    return out


def step_labels(o_t: torch.Tensor, sl: dict[str, slice], contact_n: float) -> tuple[torch.Tensor, ...]:
    """From the teacher's observation at one step: each patch's slip speed (m/s), the regime class, contact (12)."""
    pc = o_t[:, sl["patch_contact"]]
    force = pc[:, :N_SLIP] * 10.0
    speed = (pc[:, N_SLIP:].reshape(-1, N_SLIP, 3) * 0.1).norm(dim=-1)
    ev = o_t[:, sl["event_truth"]]
    valid = ev[:, 0] > 0.5
    regime = torch.where(valid, ev[:, 2:7].argmax(dim=1), torch.full_like(valid, REGIME_CLASSES - 1, dtype=torch.long))
    contact = torch.cat([force > contact_n, (ev[:, 7] > 0.5)[:, None]], dim=1)
    return speed, regime, contact


def future_slip(speed: torch.Tensor, done: torch.Tensor, k: int, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """speed (T, N, P) m/s, done (T, N): the episode ended with step t's action. Slip over the next k steps (mm) at
    each step and whether the window stays within the rollout and the episode."""
    T = speed.shape[0]
    cs = torch.cat([torch.zeros_like(speed[:1]), speed.cumsum(0)])  # cs[j] = sum of speed[:j]
    cd = torch.cat([torch.zeros_like(done[:1], dtype=torch.long), done.long().cumsum(0)])
    out = torch.zeros_like(speed)
    ok = torch.zeros(done.shape, dtype=torch.bool, device=done.device)
    if T > k:
        t = torch.arange(T - k, device=speed.device)
        out[: T - k] = (cs[t + k + 1] - cs[t + 1]) * dt * 1e3
        ok[: T - k] = (cd[t + k] - cd[t]) == 0  # no episode end among steps t .. t+k-1
    return out, ok


class DaggerTrainer:
    def __init__(self, cfg: DaggerCfg, teacher_ckpt: str, library: str | None, log_dir: str,
                 device: str = "cuda:0"):
        from mjlab.envs import ManagerBasedRlEnv
        from torch.utils.tensorboard import SummaryWriter

        from tactile_rl.export.onnx_export import load_actor
        from tactile_rl.tasks.residual import residual_env_cfg
        from tactile_sim.control.obs_spec import obs_dim, stack_for

        torch.manual_seed(cfg.seed)
        self.cfg, self.device = cfg, device
        self.stack = stack_for(cfg.rate)
        env_cfg = residual_env_cfg(cfg.rate, library=library, num_envs=cfg.num_envs, start_stage=3, max_stage=3,
                                   obs_groups=("student", "teacher"))
        env_cfg.seed = cfg.seed
        self.env = ManagerBasedRlEnv(env_cfg, device=device)
        self.teacher = load_actor(teacher_ckpt, device=device).to(device).eval()
        self.student = StudentNet(self.stack, hidden=cfg.hidden).to(device)
        self.opt = torch.optim.Adam(self.student.parameters(), lr=cfg.lr)
        self.T = max(1, round(cfg.rollout_s * cfg.rate))
        self.dt = 1.0 / cfg.rate
        self.k_slip = max(1, round(cfg.slip_horizon_s * cfg.rate))
        self.D = obs_dim(self.stack)
        self.sl = teacher_label_slices(self.D)
        self.teacher_ckpt = str(teacher_ckpt)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        (self.log_dir / "cfg.json").write_text(json.dumps({**asdict(cfg), "teacher": self.teacher_ckpt,
                                                           "library": library, "stack": self.stack,
                                                           "params": count_parameters(self.student)}, indent=1))
        N, T, dev = cfg.num_envs, self.T, device
        self.buf = {"obs": torch.zeros(T, N, self.D, device=dev), "a_t": torch.zeros(T, N, 60, device=dev),
                    "speed": torch.zeros(T, N, N_SLIP, device=dev),
                    "regime": torch.zeros(T, N, dtype=torch.long, device=dev),
                    "contact": torch.zeros(T, N, N_CONTACT, dtype=torch.bool, device=dev),
                    "done": torch.zeros(T, N, dtype=torch.bool, device=dev)}
        self.h = self.student.initial_state(N, dev)
        self.use_teacher = torch.ones(N, dtype=torch.bool, device=dev)
        self.ep_ret = torch.zeros(N, device=dev)
        self.it = 0
        self.start_it = 0

    # ------------------------------------------------------------------ rollouts
    @torch.no_grad()
    def collect(self, obs: dict, beta: float) -> tuple[dict, dict]:
        from tensordict import TensorDict

        cfg, b = self.cfg, self.buf
        N = cfg.num_envs
        self.h0 = self.h.clone()
        logs: dict[str, list[float]] = {}
        rets, agree = [], []
        for t in range(self.T):
            o_s, o_t = obs["student"], obs["teacher"]
            if self.it < cfg.norm_until:
                self.student.norm.update(o_s)
            mu_t = self.teacher(TensorDict({"teacher": o_t}, batch_size=[N]))
            mu_s, self.h, _ = self.student.step(o_s, self.h)
            a_t = torch.tanh(mu_t)
            agree.append(float((torch.tanh(mu_s) - a_t).abs().mean()))
            b["obs"][t] = o_s
            b["a_t"][t] = a_t
            b["speed"][t], b["regime"][t], b["contact"][t] = step_labels(o_t, self.sl, cfg.contact_n)
            act = torch.where(self.use_teacher[:, None], mu_t, mu_s)
            obs, rew, term, trunc, extras = self.env.step(act)
            done = term | trunc
            b["done"][t] = done
            self.ep_ret += rew
            if bool(done.any()):
                rets.append(self.ep_ret[done].clone())
                self.ep_ret[done] = 0.0
                self.teacher.reset(done)
                self.h[:, done] = 0.0
                self.use_teacher[done] = torch.rand(int(done.sum()), device=self.device) < beta
            for k, v in extras.get("log", {}).items():
                logs.setdefault(k, []).append(float(v))
        out = {k: sum(v) / len(v) for k, v in logs.items()}
        if rets:
            out["Train/mean_reward"] = float(torch.cat(rets).mean())
        out["Student/action_gap_rollout"] = sum(agree) / len(agree)
        return obs, out

    # ------------------------------------------------------------------ the update
    def update(self) -> dict[str, float]:
        cfg, b = self.cfg, self.buf
        slip, slip_ok = future_slip(b["speed"], b["done"], self.k_slip, self.dt)
        starts = torch.zeros_like(b["done"])
        starts[1:] = b["done"][:-1]
        N = cfg.num_envs
        mb = N // cfg.minibatches
        tot = {"Loss/action": 0.0, "Loss/slip": 0.0, "Loss/regime": 0.0, "Loss/contact": 0.0}
        n = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=self.device)
            for i in range(cfg.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                mu, aux = self.student.sequence(b["obs"][:, idx], self.h0[:, idx], starts[:, idx])
                l_act = F.smooth_l1_loss(torch.tanh(mu), b["a_t"][:, idx], beta=cfg.huber_delta)
                ok = slip_ok[:, idx]
                pred = aux[..., :N_SLIP]
                l_slip = (F.huber_loss(pred, slip[:, idx], delta=1.0, reduction="none").mean(-1) * ok).sum() / \
                    ok.sum().clamp(min=1)
                l_reg = F.cross_entropy(aux[..., N_SLIP:N_SLIP + REGIME_CLASSES].reshape(-1, REGIME_CLASSES),
                                        b["regime"][:, idx].reshape(-1))
                l_con = F.binary_cross_entropy_with_logits(aux[..., N_SLIP + REGIME_CLASSES:],
                                                           b["contact"][:, idx].float())
                loss = l_act + cfg.w_slip * l_slip + cfg.w_regime * l_reg + cfg.w_contact * l_con
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), cfg.max_grad_norm)
                self.opt.step()
                for k, v in zip(tot, (l_act, l_slip, l_reg, l_con), strict=True):
                    tot[k] += float(v.detach())
                n += 1
        return {k: v / n for k, v in tot.items()}

    # ------------------------------------------------------------------ the loop
    def save(self, path: Path) -> None:
        torch.save({"student_state_dict": self.student.state_dict(), "optimizer": self.opt.state_dict(),
                    "iter": self.it, "cfg": asdict(self.cfg), "stack": self.stack, "hidden": self.cfg.hidden,
                    "rate": self.cfg.rate, "teacher": self.teacher_ckpt}, path)

    def resume(self, path: str) -> None:
        """Continue a run from its checkpoint: the student (normalizer included) and the optimizer; the schedules go on
        from the checkpoint's iteration (the env and the student's state start afresh)."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.student.load_state_dict(ck["student_state_dict"])
        self.opt.load_state_dict(ck["optimizer"])
        self.start_it = int(ck["iter"]) + 1

    def train(self) -> None:
        cfg = self.cfg
        obs, _ = self.env.reset()
        self.teacher.reset()
        self.student.norm.frozen = self.start_it >= cfg.norm_until
        for it in range(self.start_it, cfg.iterations):
            self.it = it
            if it == cfg.norm_until:
                self.student.norm.frozen = True
            beta = beta_at(cfg, it)
            lr = lr_at(cfg, it)
            for g in self.opt.param_groups:
                g["lr"] = lr
            t0 = time.time()
            obs, logs = self.collect(obs, beta)
            t1 = time.time()
            losses = self.update()
            t2 = time.time()
            sim_s = (it + 1) * cfg.num_envs * self.T * self.dt
            rec = {**logs, **losses, "Student/beta": beta, "Loss/learning_rate": lr, "Perf/collection_s": t1 - t0,
                   "Perf/learning_s": t2 - t1, "Perf/sim_s": sim_s}
            for k, v in rec.items():
                self.writer.add_scalar(k, v, it)
            ret = logs.get("Train/mean_reward", float("nan"))
            print(f"it {it:4d} beta {beta:.2f} lr {lr:.1e} action {losses['Loss/action']:.4f} slip "
                  f"{losses['Loss/slip']:.3f} regime {losses['Loss/regime']:.3f} contact {losses['Loss/contact']:.3f} "
                  f"gap {logs['Student/action_gap_rollout']:.3f} return {ret:.2f} ({t1 - t0:.1f} + {t2 - t1:.1f} s)",
                  flush=True)
            if (it + 1) % cfg.save_every == 0 or it == cfg.iterations - 1:
                self.save(self.log_dir / f"student_{it}.pt")
        self.writer.flush()


def load_student(path: str, device: str = "cpu") -> StudentNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = StudentNet(int(ck["stack"]), hidden=int(ck.get("hidden", 192)))
    net.load_state_dict(ck["student_state_dict"])
    net.norm.frozen = True
    return net.to(device).eval()
