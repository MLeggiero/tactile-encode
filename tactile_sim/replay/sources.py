"""Recorded tool motions to replay in the testbed: where they come from, how they are fetched, and how each is read
into a ToolMotion (the tool's striking face and orientation over time, in the source's own world).

Sources (fetched file by file at pinned commits, verified against REPLAY_MANIFEST.json, cached outside the repo):

- "adroit": the 25 human hammering demonstrations of DAPG / D4RL hammer-human (Rajeswaran et al., recorded in VR
  with a CyberGlove driving the Adroit hand in MuJoCo; aravindr93/hand_dapg, Apache-2.0). 100 Hz observations
  (MuJoCo's default 2 ms step x frame_skip 5) hold the hammer's position and Euler angles (intrinsic XYZ, relative to
  its resting orientation in DAPG_hammer.xml) and the nail head. A touch sensor on the nail marks contacts.
- "dextoolbench": the hammer task trajectories of DexToolBench (Lum et al., SimToolReal; tylerlum/simtoolreal, MIT):
  tool poses tracked from human RGB-D videos by FoundationPose, in the benchmark's world frame, downsampled to
  ~3 Hz ([x, y, z, qx, qy, qz, qw]). There is no nail: the strike point is taken where the face reaches farthest
  along its outward normal.

Usage:
    python -m tactile_sim.replay.sources                    # fetch everything into the cache
    python -m tactile_sim.replay.sources --regen-manifest   # rebuild the manifest at the pinned commits
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tactile_sim.assets.fetch_menagerie import RAW_BASE, _write_atomic, download, sha256

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "assets" / "REPLAY_MANIFEST.json"
ENV_VAR = "TACTILE_SIM_REPLAY"

# source -> (repo, commit, [paths], license file)
REPLAY_SOURCES: dict[str, tuple[str, str, list[str], str]] = {
    "adroit": ("aravindr93/hand_dapg", "9a1de23df6574e1b94e1da742e3326eb1f61c013",
               ["dapg/demonstrations/hammer-v0_demos.pickle"], "LICENSE"),
    "dextoolbench": ("tylerlum/simtoolreal", "313d5aea1f507c6cfe097b672b62945d7b0bbff5",
                     ["dextoolbench/trajectories/hammer/claw_hammer/swing_down.json",
                      "dextoolbench/trajectories/hammer/claw_hammer/swing_side.json",
                      "dextoolbench/trajectories/hammer/mallet_hammer/swing_down.json",
                      "dextoolbench/trajectories/hammer/mallet_hammer/swing_side.json"], "LICENSE"),
}

# Source tool geometry: the striking-face centre in the tool's frame, and this repository's tool axes (x along the
# handle away from the head, y opposite the face's outward normal, z = x cross y) written in the source tool's frame.
#  - Adroit (DAPG_hammer.xml, body "Object"): head cylinder at x = -0.24 along the body z axis; the face used is its
#    -z end (the "tool" site sits on that side).
#  - DexToolBench meshes: the head is at +x along y; the flat face is at -y, the claw (or second face) at +y.
SOURCE_TOOLS: dict[str, dict] = {
    "adroit_hammer": {"face_local": (-0.24, 0.0, -0.04), "axes": ((1, 0, 0), (0, 0, 1), (0, -1, 0))},
    "claw_hammer": {"face_local": (0.126, -0.038, 0.0), "axes": ((-1, 0, 0), (0, 1, 0), (0, 0, -1))},
    "mallet_hammer": {"face_local": (0.205, -0.046, 0.0), "axes": ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
                      "other_face": {"face_local": (0.205, 0.041, 0.0), "axes": ((-1, 0, 0), (0, -1, 0), (0, 0, 1))}},
}
# DAPG_hammer.xml: the hammer's resting orientation and the nail board's (the nail is driven along board -z)
ADROIT_HAMMER_QUAT = (0.707388, 0.706825, 0.0, 0.0)
ADROIT_BOARD_QUAT = (0.583833, 0.583368, -0.399421, -0.399104)
ADROIT_DT = 0.01
DEXTOOLBENCH_DT = 1.0 / 3.0


@dataclass
class ToolMotion:
    """A tool's striking face and orientation over time, in the source's world.

    R holds this repository's tool axes (x along the handle away from the head, -y the face's outward normal)."""
    name: str
    t: np.ndarray  # (N,) s
    face: np.ndarray  # (N, 3) face centre
    R: np.ndarray  # (N, 3, 3)
    contact_index: int  # sample at which the face meets the target
    axis: np.ndarray  # (3,) strike direction (into the target)
    contacts: list[int] = field(default_factory=list)  # every recorded contact onset
    meta: dict = field(default_factory=dict)

    @property
    def normal(self) -> np.ndarray:
        """Outward face normal per sample."""
        return -self.R[:, :, 1]


# ----------------------------------------------------------------- cache
def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}


def cache_dir(source: str) -> Path:
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env) / source
    return Path.home() / ".cache" / "tactile_sim" / "replay" / source / REPLAY_SOURCES[source][1]


def available(source: str) -> bool:
    entry = _manifest().get(source)
    if entry is None:
        return False
    root = cache_dir(source)
    return all((root / f["path"]).is_file() and sha256((root / f["path"]).read_bytes()) == f["sha256"]
               for f in entry["files"])


def fetch(source: str) -> Path:
    entry = _manifest()[source]
    root = cache_dir(source)
    for f in entry["files"]:
        target = root / f["path"]
        if target.exists() and sha256(target.read_bytes()) == f["sha256"]:
            continue
        data = download(RAW_BASE.format(repo=entry["repo"], commit=entry["commit"], path=f["path"]))
        if sha256(data) != f["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {source}:{f['path']}")
        _write_atomic(target, data)
    return root


def regen_manifest(sources: list[str]) -> dict:
    manifest = _manifest()
    for src in sources:
        repo, commit, paths, lic = REPLAY_SOURCES[src]
        files = []
        root = cache_dir(src)
        for p in [lic] + paths:
            data = download(RAW_BASE.format(repo=repo, commit=commit, path=p))
            _write_atomic(root / p, data)
            files.append({"path": p, "sha256": sha256(data), "size": len(data)})
        manifest[src] = {"repo": repo, "commit": commit, "files": files}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _require(source: str) -> Path:
    if not available(source):
        raise FileNotFoundError(f"replay source {source!r} not cached; run `python -m tactile_sim.replay.sources`")
    return cache_dir(source)


def _tool_rotation(spec: dict, R_src: np.ndarray) -> np.ndarray:
    M = np.column_stack([np.asarray(a, float) for a in spec["axes"]])
    return R_src @ M


# ----------------------------------------------------------------- Adroit / DAPG
def adroit_demos() -> list[dict]:
    """The raw DAPG hammer demonstrations (a pinned, hash-verified pickle of numpy arrays)."""
    path = _require("adroit") / REPLAY_SOURCES["adroit"][2][0]
    with open(path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301 - pinned upstream file, sha256-verified above


def load_adroit(demo: int = 0, lift: float = 0.03, tail: float = 0.5) -> ToolMotion:
    """One DAPG demonstration, from the moment the hammer is lifted to `tail` s after its last contact."""
    e = adroit_demos()[demo]
    o = np.asarray(e["observations"], float)
    q0 = np.asarray(ADROIT_HAMMER_QUAT, float)
    R0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
    R_src = Rotation.from_euler("XYZ", o[:, 39:42]).as_matrix() @ R0
    R = _tool_rotation(SOURCE_TOOLS["adroit_hammer"], R_src)
    face_local_src = np.asarray(SOURCE_TOOLS["adroit_hammer"]["face_local"])
    face = o[:, 36:39] + R_src @ face_local_src
    touch = o[:, 45] > 1e-6  # the clipped touch sensor (rounding noise is ~1e-18)
    onsets = [int(i) for i in np.flatnonzero(touch & ~np.r_[False, touch[:-1]])]
    if not onsets:
        raise ValueError(f"Adroit demo {demo} has no nail contact")
    qb = np.asarray(ADROIT_BOARD_QUAT, float)
    axis = Rotation.from_quat([qb[1], qb[2], qb[3], qb[0]]).as_matrix() @ np.array([0.0, 0.0, -1.0])
    lifted = np.flatnonzero(o[:, 38] > o[0, 38] + lift)
    i0 = int(lifted[0]) if len(lifted) else 0
    i0 = max(0, min(i0, onsets[0] - int(round(0.3 / ADROIT_DT))))  # keep the approach to the first blow
    i1 = min(len(o), onsets[-1] + int(round(tail / ADROIT_DT)) + 1)
    sl = slice(i0, i1)
    t = np.arange(i1 - i0) * ADROIT_DT
    contacts = [i - i0 for i in onsets if i0 <= i < i1]
    return ToolMotion(f"adroit/demo{demo}", t, face[sl], R[sl], contacts[0], axis, contacts,
                      {"source": "adroit", "demo": demo, "nail": o[onsets[0], 42:45].tolist(), "dt": ADROIT_DT})


# ----------------------------------------------------------------- DexToolBench
def dextoolbench_tasks() -> list[str]:
    return [p.split("trajectories/")[1][:-5] for p in REPLAY_SOURCES["dextoolbench"][2]]


def load_dextoolbench(task: str = "hammer/claw_hammer/swing_down") -> ToolMotion:
    """A DexToolBench hammer trajectory; the strike is where the face reaches farthest along its mean normal."""
    root = _require("dextoolbench")
    d = json.loads((root / "dextoolbench" / "trajectories" / f"{task}.json").read_text())
    poses = np.asarray([d["start_pose"]] + d["goals"], float)
    tool = task.split("/")[1]
    R_src = Rotation.from_quat(poses[:, 3:7]).as_matrix()  # scalar-last, as SimToolReal writes them
    specs = [SOURCE_TOOLS[tool]] + ([SOURCE_TOOLS[tool]["other_face"]] if "other_face" in SOURCE_TOOLS[tool] else [])
    best = None
    for spec in specs:  # a two-faced head strikes with whichever face points along the swing
        face = poses[:, :3] + R_src @ np.asarray(spec["face_local"])
        R = _tool_rotation(spec, R_src)
        n_mean = (-R[:, :, 1]).mean(axis=0)
        n_mean /= np.linalg.norm(n_mean)
        reach = face @ n_mean
        k = int(np.argmax(reach))
        align = float(-R[k, :, 1] @ n_mean)
        if best is None or align > best[0]:
            best = (align, face, R, n_mean, reach, k)
    _, face, R, n_mean, reach, k = best
    # every local maximum within 1 cm of the deepest reach counts as a strike
    peaks = [i for i in range(1, len(reach) - 1)
             if reach[i] >= reach[i - 1] and reach[i] >= reach[i + 1] and reach[i] > reach[k] - 0.01]
    t = np.arange(len(poses)) * DEXTOOLBENCH_DT
    return ToolMotion(f"dextoolbench/{task}", t, face, R, k, n_mean, peaks or [k],
                      {"source": "dextoolbench", "task": task, "dt": DEXTOOLBENCH_DT})


def load(source: str, **kw) -> ToolMotion:
    if source == "adroit":
        return load_adroit(**kw)
    if source == "dextoolbench":
        return load_dextoolbench(**kw)
    raise ValueError(f"unknown replay source {source!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", default=list(REPLAY_SOURCES))
    ap.add_argument("--regen-manifest", action="store_true")
    args = ap.parse_args(argv)
    if args.regen_manifest:
        regen_manifest(args.sources)
        return 0
    for s in args.sources:
        print(f"{s}: {fetch(s)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
