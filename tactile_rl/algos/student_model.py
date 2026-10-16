"""The student (the plan's "Distill into a small recurrent student"): what the robot senses, with its timing, into a
GRU of 192 units, then an action head and auxiliary heads; 0.42 M parameters at 1 kHz, 0.46 M at 250 Hz.

Per 1 kHz sample (a 250 Hz student reads `stack` = 4 of them, newest first, through the same weights): the palm's
128 taxels as an 8 x 16 image through two convolutions, each finger patch's 32 taxels (and, in observation layout 2,
its 6 features) through one shared MLP (10 patches), the palm's features (layout 2), the boards' status, the joint
angles and currents; each sample projected, then the stacked samples mixed; the policy-rate blocks of the layout
(last action, the base's hand target, feedforward and lookahead, the task, the trigger enable; layout 2 also the
forecast) through a small MLP. `version` picks the layout (obs_spec.OBS_LAYOUTS). Inputs are normalized first
(running mean and variance, a floor on the scale, clipped, as the teacher's RobustNormalization), which the ONNX export
folds in.

Outputs per step: the action's pre-tanh mean (40; the env squashes it; the last layer starts at zero so a fresh
student is the base) and the auxiliary predictions (AUX_LAYOUT): each patch's slip over the next 20 ms (mm), the
load regime (5 regimes and none, logits), contact (each patch and the tool on the workpiece, logits).
"""

from __future__ import annotations

import torch
from torch import nn

from tactile_sim.control.obs_spec import (
    ACTION_DIM,
    BOARDS,
    N_JOINTS,
    OBS_SPEC_VERSION,
    TAXEL_PATCHES,
    TAXELS,
    obs_blocks,
    obs_dim,
    obs_slices,
)

PALM_GRID = (8, 16)
FINGER_TAXELS = 32
N_FINGER_PATCHES = TAXEL_PATCHES - 1
REGIME_CLASSES = 6  # obs_spec.REGIMES and "none"
AUX_LAYOUT = {"slip_next_mm": TAXEL_PATCHES, "regime_logits": REGIME_CLASSES, "contact_logits": TAXEL_PATCHES + 1}
AUX_DIM = sum(AUX_LAYOUT.values())


def slow_blocks(version: int = OBS_SPEC_VERSION) -> tuple[str, ...]:
    """The layout's policy-rate blocks."""
    return tuple(name for name, _, fast, _ in obs_blocks(version) if not fast)


SLOW_BLOCKS = slow_blocks()


class RunningNorm(nn.Module):
    """Running mean and variance (count-weighted, Chan's update) with a floor on the scale and a clip, as the
    teacher's RobustNormalization; `frozen` stops the statistics."""

    def __init__(self, dim: int, std_floor: float = 0.1, clip: float = 5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.std_floor, self.clip = float(std_floor), float(clip)
        self.frozen = False

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if self.frozen:
            return
        x = x.reshape(-1, x.shape[-1]).float()
        n = float(x.shape[0])
        m, v = x.mean(0), x.var(0, unbiased=False)
        c = float(self.count)
        tot = c + n
        d = m - self.mean
        self.mean += d * (n / tot)
        self.var.copy_((self.var * c + v * n + d * d * (c * n / tot)) / tot)
        self.count += n

    def scale(self) -> torch.Tensor:
        return torch.clamp(self.var.sqrt(), min=self.std_floor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / self.scale(), -self.clip, self.clip)


class StudentNet(nn.Module):
    def __init__(self, stack: int = 1, hidden: int = 192, fast_dim: int = 160, slow_dim: int = 64,
                 sample_dim: int = 80, version: int = OBS_SPEC_VERSION):
        super().__init__()
        self.stack, self.hidden, self.version = int(stack), int(hidden), int(version)
        self.obs_dim = obs_dim(self.stack, self.version)
        sl = obs_slices(self.stack, self.version)
        self._sl = {k: (v.start, v.stop) for k, v in sl.items()}
        self.features = "patch_features" in sl  # layout 2: the patches' features beside their codes
        self.slow_blocks = slow_blocks(self.version)
        self.norm = RunningNorm(self.obs_dim)
        self.palm = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ELU(), nn.Conv2d(8, 16, 3, stride=2, padding=1),
                                  nn.ELU(), nn.Flatten(), nn.Linear(16 * (PALM_GRID[0] // 2) * (PALM_GRID[1] // 2), 64),
                                  nn.ELU())
        nf = 6 if self.features else 0
        self.finger = nn.Sequential(nn.Linear(FINGER_TAXELS + nf, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU())
        fast_in = 64 + N_FINGER_PATCHES * 32 + nf + 2 * BOARDS + 2 * N_JOINTS
        self.fast = nn.Sequential(nn.Linear(fast_in, fast_dim), nn.ELU())
        # each sample to sample_dim, then the stack mixed: the size grows only with the stack's last layer
        self.proj = nn.Sequential(nn.Linear(fast_dim, sample_dim), nn.ELU())
        self.mix = nn.Sequential(nn.Linear(sample_dim * self.stack, fast_dim), nn.ELU())
        slow_in = sum(self._sl[b][1] - self._sl[b][0] for b in self.slow_blocks)
        self.slow = nn.Sequential(nn.Linear(slow_in, slow_dim), nn.ELU())
        self.gru = nn.GRU(fast_dim + slow_dim, self.hidden)
        self.head = nn.Sequential(nn.Linear(self.hidden, 128), nn.ELU(), nn.Linear(128, ACTION_DIM))
        self.aux = nn.Linear(self.hidden, AUX_DIM)
        with torch.no_grad():
            self.head[-1].weight.zero_()
            self.head[-1].bias.zero_()

    # ------------------------------------------------------------------ the encoder
    def _block(self, x: torch.Tensor, name: str) -> torch.Tensor:
        a, b = self._sl[name]
        return x[..., a:b]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x (..., obs_dim), normalized -> (..., fast_dim + slow_dim)."""
        lead = x.shape[:-1]
        s = self.stack
        codes = self._block(x, "taxel_codes").reshape(-1, s, TAXELS)  # (M, s, 448), newest sample first
        board = self._block(x, "board_status").reshape(-1, s, 2 * BOARDS)
        q = self._block(x, "hand_q").reshape(-1, s, N_JOINTS)
        cur = self._block(x, "hand_current").reshape(-1, s, N_JOINTS)
        M = codes.shape[0]
        n_palm = PALM_GRID[0] * PALM_GRID[1]
        palm = self.palm(codes[..., :n_palm].reshape(M * s, 1, *PALM_GRID)).reshape(M, s, -1)
        fing = codes[..., n_palm:].reshape(M, s, N_FINGER_PATCHES, FINGER_TAXELS)
        if self.features:
            feats = self._block(x, "patch_features").reshape(-1, s, TAXEL_PATCHES, 6)
            fing = self.finger(torch.cat([fing, feats[:, :, 1:]], dim=-1)).reshape(M, s, -1)
            per = self.fast(torch.cat([palm, fing, feats[:, :, 0], board, q, cur], dim=-1))  # (M, s, fast_dim)
        else:
            fing = self.finger(fing).reshape(M, s, -1)
            per = self.fast(torch.cat([palm, fing, board, q, cur], dim=-1))
        fast = self.mix(self.proj(per).reshape(M, -1))
        slow = self.slow(torch.cat([self._block(x, b) for b in self.slow_blocks], dim=-1).reshape(M, -1))
        return torch.cat([fast, slow], dim=-1).reshape(*lead, -1)

    # ------------------------------------------------------------------ inference and training
    def step(self, obs: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One policy step: obs (B, D) raw, h (1, B, H) -> (pre-tanh mean (B, 60), h_next, aux (B, AUX_DIM))."""
        z = self.encode(self.norm(obs))
        y, h = self.gru(z[None], h)
        y = y[0]
        return self.head(y), h, self.aux(y)

    def sequence(self, obs: torch.Tensor, h0: torch.Tensor, starts: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """obs (T, B, D) raw, h0 (1, B, H) the state before step 0, starts (T, B) bool: an episode begins at that step
        (the state is zeroed before it). -> pre-tanh means (T, B, 60), aux (T, B, AUX_DIM)."""
        z = self.encode(self.norm(obs))
        keep = (~starts).float()[..., None]  # (T, B, 1)
        h = h0
        ys = []
        for t in range(obs.shape[0]):
            h = h * keep[t][None]
            y, h = self.gru(z[t:t + 1], h)
            ys.append(y[0])
        y = torch.stack(ys)
        return self.head(y), self.aux(y)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden, device=device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class StudentGraph(nn.Module):
    """The exportable student (the plan's signature): obs, h_in -> squashed action, h_out, aux. The normalizer's
    statistics are copied in as constants."""

    def __init__(self, net: StudentNet):
        super().__init__()
        self.net = net
        self.register_buffer("mean", net.norm.mean.detach().clone())
        self.register_buffer("scale", net.norm.scale().detach().clone())
        self.clip = float(net.norm.clip)

    def forward(self, obs: torch.Tensor, h_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.clamp((obs - self.mean) / self.scale, -self.clip, self.clip)
        z = self.net.encode(x)
        y, h = self.net.gru(z[None], h_in[None])
        y = y[0]
        return torch.tanh(self.net.head(y)), h[0], self.net.aux(y)
