"""Write and read episode files (HDF5, or npz when h5py is unavailable). Layout: see schema.py."""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from tactile_sim.logging.schema import SCHEMA_VERSION
from tactile_sim.logging.strike_metrics import STRIKE_DTYPE, to_array

try:  # pragma: no cover - import guard
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def episode_tables(res, cfg, seed) -> tuple[dict, dict[str, np.ndarray]]:
    """Flatten an EpisodeResult into (root attrs, {'group/name': array})."""
    tb = res.testbed
    attrs = {"schema_version": SCHEMA_VERSION, "config_json": cfg.to_json(), "mujoco_version": mujoco.__version__,
             "git_sha": _git_sha(), "seed": -1 if seed is None else int(seed), "dt_phys": tb.world.dt,
             "arm_source": tb.world.arm_source, "created": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    out: dict[str, np.ndarray] = {}
    for k, v in res.truth.items():
        out[f"physics/{k}"] = np.asarray(v, dtype=np.float64 if k == "t" else np.float32)
    for name, s in tb.sensors.sensors.items():
        h = s.history()
        out[f"sensors/{name}/t_sample"] = h["t_sample"]
        out[f"sensors/{name}/t_avail"] = h["t_avail"]
        out[f"sensors/{name}/value"] = h["value"].astype(np.float32)
        out[f"sensors/{name}/seq"] = h["seq"]
    L = tb.l1.log
    for k in ("t", "tau", "q", "qd", "x", "x_eq", "xd_eq", "K", "F_ff", "F_grip", "mode", "impact_flag", "gated",
              "tau_ext", "f_ext"):
        v = getattr(L, k)
        dt = np.float64 if k == "t" else (np.int8 if k == "mode" else (np.bool_ if k in ("impact_flag", "gated")
                                                                         else np.float32))
        out[f"control/{k}"] = np.asarray(v, dtype=dt)
    G = tb.grip.log
    for k in ("t", "setpoint", "measured", "command"):
        out[f"grip/{k}"] = np.asarray(getattr(G, k), dtype=np.float64 if k == "t" else np.float32)
    ev = np.array([(n.encode()[:32], t) for n, t in res.events], dtype=[("name", "S32"), ("t", "f8")])
    out["events"] = ev
    out["strikes"] = to_array(res.strikes) if res.strikes else np.zeros(0, dtype=STRIKE_DTYPE)
    return attrs, out


def write_episode(res, path: str | Path, cfg, seed: int | None = None, compress: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    attrs, tables = episode_tables(res, cfg, seed)
    summary = {k: (np.nan if v is None else v) for k, v in res.summary.items()}
    if h5py is None or path.suffix == ".npz":
        path = path.with_suffix(".npz")
        flat = dict(tables)
        flat["__attrs__"] = np.array(json.dumps(attrs))
        flat["__summary__"] = np.array(json.dumps(summary))
        np.savez_compressed(path, **flat)
        return path
    kw = {"compression": "gzip", "compression_opts": 4} if compress else {}
    with h5py.File(path, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        for key, arr in tables.items():
            f.create_dataset(key, data=arr, **(kw if arr.size > 64 and arr.dtype.kind in "fiub" else {}))
        sg = f.create_group("summary")
        for k, v in summary.items():
            sg.attrs[k] = v
    return path


def read_episode(path: str | Path) -> dict:
    """Load an episode file back into {'attrs', 'summary', 'group/name': array}."""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        out = {k: z[k] for k in z.files if not k.startswith("__")}
        out["attrs"] = json.loads(str(z["__attrs__"]))
        out["summary"] = json.loads(str(z["__summary__"]))
        return out
    out: dict = {}
    with h5py.File(path, "r") as f:
        out["attrs"] = dict(f.attrs)
        out["summary"] = dict(f["summary"].attrs)

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]

        f.visititems(visit)
    return out
