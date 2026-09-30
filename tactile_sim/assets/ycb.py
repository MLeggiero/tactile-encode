"""YCB 048_hammer (claw hammer, steel head, wooden handle; CC BY 4.0) for the Task A tool.

The Google 16k scan is downloaded once from the YCB benchmark bucket, verified by sha256, and
preprocessed into the simulator's hammer frame (see tool_hammer.py):

    origin  on the handle centreline, `grip_from_head` from the head axis (the grasp point)
    x       along the handle, away from the head
    y       along the head axis; the striking face points to -y
    z       x cross y (the hammer's thickness direction)

Outputs (in the cache): the full scan split into head and handle visual meshes (STL), vertex sets
for the convex contact hulls (grip section, striking face, whole head, handle) and a metadata JSON
with the face centre and the handle's half-width at the grasp.

    python -m tactile_sim.assets.ycb            # fetch + preprocess
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import tarfile
from pathlib import Path

import numpy as np

URL = "https://ycb-benchmarks.s3.amazonaws.com/data/google/048_hammer_google_16k.tgz"
SHA256 = "4745093ebe83e71e4ec80f5fa438a42a7175a0df6e5e2c6964a23085bc6ae195"
STL_MEMBER = "048_hammer/google_16k/nontextured.stl"
MASS = 0.665  # YCB object mass (kg)
HEAD_MASS = 0.45  # 16 oz steel head; the wooden handle carries the rest
HEAD_DEPTH = 0.045  # the head occupies the top 45 mm along the handle axis
PREPROC_VERSION = 4


def cache_dir() -> Path:
    env = os.environ.get("TACTILE_SIM_YCB")
    return Path(env) if env else Path.home() / ".cache" / "tactile_sim" / "ycb" / "048_hammer"


def read_stl(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Binary STL -> (unique vertices, faces)."""
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    tri = rec["v"].reshape(-1, 3).astype(np.float64)
    verts, inv = np.unique(tri, axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 3)


def write_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(faces), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = nrm
    rec["v"] = tri
    buf = io.BytesIO()
    buf.write(b"tactile_sim ycb 048_hammer".ljust(80, b" "))
    buf.write(struct.pack("<I", len(faces)))
    buf.write(rec.tobytes())
    path.write_bytes(buf.getvalue())


def _centreline(V: np.ndarray, u: np.ndarray, t: np.ndarray, lo: float, hi: float, step: float = 0.01):
    """Centres of thin slabs across the tool between t-offsets lo..hi (from the head end)."""
    out = []
    for d in np.arange(lo, hi, step):
        sl = V[np.abs((t - t.min()) - d) < 0.003]
        if len(sl) > 20:
            out.append((d, sl.mean(axis=0)))
    return out


def hammer_frame(V: np.ndarray, grip_from_head: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rotation R (columns = hammer axes in scan coordinates), grasp origin, and landmarks.

    The handle axis comes from the centreline (centres of thin slabs), so neither the head's mass of
    vertices nor the handle's curve tilts it; the frame's x axis is the handle's local direction at
    the grasp, so the pads sit parallel to its sides there.
    """
    _, _, W = np.linalg.svd(V - V.mean(axis=0), full_matrices=False)
    u, thick = W[0], W[2]  # longest and thinnest directions of the whole tool
    t = (V - V.mean(axis=0)) @ u
    if t[np.argmax(V[:, 1])] > 0:  # in the scan the head sits at the high-y end: point u away from it
        u = -u
    for _ in range(3):  # refine the handle axis from the centreline, clear of the head
        t = (V - V.mean(axis=0)) @ u
        pts = np.array([c for _, c in _centreline(V, u, t, 0.08, np.ptp(t) - 0.02)])
        _, _, Wc = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
        u = Wc[0] if Wc[0] @ u > 0 else -Wc[0]
    t = (V - V.mean(axis=0)) @ u
    h = np.cross(thick, u)
    h /= np.linalg.norm(h)
    head = V[t < t.min() + HEAD_DEPTH]
    ph = head @ h
    # the striking face is the round end of the head; the claw end is thin
    caps = {s: head[(ph > ph.max() - 0.010) if s > 0 else (ph < ph.min() + 0.010)] for s in (1, -1)}
    size = {s: np.ptp(caps[s] @ u) for s in caps}
    h = h * (1 if size[1] > size[-1] else -1)
    ph = head @ h
    face_cap = head[ph > ph.max() - 0.003]
    F = face_cap.mean(axis=0)
    F = F + h * (ph.max() - F @ h)  # face centre on the outermost face plane
    # grasp on the centreline, grip_from_head from the face's axis along the handle
    d_face = (F - V.mean(axis=0)) @ u - t.min()
    d_grip = d_face + grip_from_head
    local = _centreline(V, u, t, d_grip - 0.03, d_grip + 0.031, 0.005)
    pts = np.array([c for _, c in local])
    _, _, Wl = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    x_ax = Wl[0] if Wl[0] @ u > 0 else -Wl[0]
    ds = np.array([d for d, _ in local])
    G = np.array([np.interp(d_grip, ds, pts[:, k]) for k in range(3)])
    y_ax = -(h - (h @ x_ax) * x_ax)
    y_ax /= np.linalg.norm(y_ax)
    z_ax = np.cross(x_ax, y_ax)
    R = np.column_stack([x_ax, y_ax, z_ax])
    return R, G, {"t": t, "t_min": float(t.min()), "face": F}


def preprocess(stl_bytes: bytes, out: Path, grip_from_head: float) -> dict:
    V, F = read_stl(stl_bytes)
    R, G, lm = hammer_frame(V, grip_from_head)
    t = lm["t"]
    is_head_v = t < lm["t_min"] + HEAD_DEPTH
    # the wooden handle is curved and flared: centre the grasp origin on the gripped section itself
    L = (V - G) @ R
    sec = L[(np.abs(L[:, 0]) < 0.03) & ~is_head_v]
    mid = 0.5 * (sec.min(axis=0) + sec.max(axis=0))
    G = G + R @ np.array([0.0, mid[1], mid[2]])
    L = (V - G) @ R  # hammer-frame vertices
    face_local = (lm["face"] - G) @ R
    # visual: split faces into steel head and wooden handle by their vertices
    head_face = is_head_v[F].any(axis=1)
    out.mkdir(parents=True, exist_ok=True)
    write_stl(out / "hammer_head_visual.stl", L, F[head_face])
    write_stl(out / "hammer_handle_visual.stl", L, F[~head_face])
    # contact / mass hulls (vertex clouds; MuJoCo builds convex hulls)
    grip = L[(np.abs(L[:, 0]) < 0.03) & ~is_head_v]
    face = L[L[:, 1] < face_local[1] + 0.012]
    face = face[np.abs(face[:, 0] - face_local[0]) < 0.03]
    hulls = {"grip": grip, "face": face, "head": L[is_head_v], "handle": L[~is_head_v]}
    np.savez(out / "hulls.npz", **{k: v.astype(np.float32) for k, v in hulls.items()})
    meta = {
        "version": PREPROC_VERSION, "sha256": SHA256, "grip_from_head": grip_from_head,
        "face_local": face_local.tolist(),
        "face_radius": float(np.max(np.linalg.norm((face[face[:, 1] < face_local[1] + 0.002] - face_local)[:, [0, 2]],
                                                   axis=1))),
        "grip_half_width": float(np.max(np.abs(grip[:, 1]))),  # along the finger closing axis (y)
        "grip_half_thickness": float(np.max(np.abs(grip[:, 2]))),
        "length": float(np.ptp(L[:, 0])), "mass": MASS, "head_mass": HEAD_MASS,
        "license": "YCB object set, CC BY 4.0 (https://www.ycbbenchmarks.com)",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _download() -> bytes:
    from tactile_sim.assets.fetch_menagerie import download

    data = download(URL, timeout=120)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("YCB hammer archive failed its sha256 check")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        return tf.extractfile(STL_MEMBER).read()


def ensure_hammer(grip_from_head: float, download: bool = False) -> tuple[Path, dict] | None:
    """Return (dir, meta) for the preprocessed hammer, fetching/preprocessing if allowed."""
    root = cache_dir()
    raw = root / "nontextured.stl"
    out = root / f"grip_{grip_from_head * 1000:.0f}mm_v{PREPROC_VERSION}"
    if (out / "meta.json").exists():
        return out, json.loads((out / "meta.json").read_text())
    if not raw.exists():
        if not download:
            return None
        root.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(_download())
    return out, preprocess(raw.read_bytes(), out, grip_from_head)


def load_hulls(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path / "hulls.npz")
    return {k: z[k] for k in z.files}


def main() -> int:
    from tactile_sim.config import HammerCfg

    res = ensure_hammer(HammerCfg().grip_from_head, download=True)
    path, meta = res
    print(f"YCB 048_hammer ready in {path}: face at {np.round(meta['face_local'], 4).tolist()} m, "
          f"face radius {meta['face_radius'] * 1e3:.1f} mm, grip half-width {meta['grip_half_width'] * 1e3:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
