"""Distil a teacher into the student (tactile_rl.algos.dagger), in Tactile-Residual's C3:

    python -m tactile_rl.distill --rate 1000 --teacher <run>/model_999.pt --library <lib.npz> --log-dir <dir> \
        [--iterations 300] [--num-envs 4096] [--seed 1]

Checkpoints (student_<it>.pt), tensorboard scalars and cfg.json go to --log-dir.
"""

from __future__ import annotations

import argparse
from dataclasses import fields


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - training script
    from tactile_rl.algos.dagger import DaggerCfg, DaggerTrainer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--library", default=None)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    for f in fields(DaggerCfg):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    a = ap.parse_args(argv)
    cfg = DaggerCfg(**{f.name: getattr(a, f.name) for f in fields(DaggerCfg)})
    trainer = DaggerTrainer(cfg, a.teacher, a.library, a.log_dir, a.device)
    try:
        trainer.train()
    finally:
        trainer.env.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
