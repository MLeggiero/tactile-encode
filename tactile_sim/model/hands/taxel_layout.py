"""Conforming taxel layout for the WUJI Hand 2: a 128-taxel sheet on the palm and a 32-taxel patch on the palmar face
of every finger's distal pad and middle segment (thumb included): 448 taxels, each read at 1 kHz by a TaxelScan board.

Each patch is a grid laid on the link's actual surface, not on a plane. In the vendor model at its zero pose (open
hand), a patch frame is set up on the link: u along the link (proximal to distal), n the palmar normal (the direction
the link moves when its flexion joint closes; the palm's +y for the palm), v = n x u across it. Every grid point
(s_u, s_v) casts a ray from outside the palmar side back along -n onto the link's visual mesh (the detailed mesh,
not the convex hull); the hit is the taxel's position and the hit triangle's normal its sensing direction. Patch
extents follow the link: a fraction of the link's length and of its width at mid-length, so a patch stays on the
pad's front and does not wrap round onto the sides. Positions and normals are stored in the patch body's frame.

The fingertip pad covers the distal link and the fingertip shell (the `*_tip_sensor_frame` body, rigidly fixed to
it): both meshes are cast against, in the distal link's frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

PALM_BODY = "r_wrist"
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
# body -> (flexion joint, distal end body used for the link's length)
_LINKS = {}
for _f in FINGERS:
    if _f == "thumb":
        _LINKS["r_thumb_middle"] = ("r_thumb_mcp", "r_thumb_distal")
        _LINKS["r_thumb_distal"] = ("r_thumb_ip", None)
    else:
        _LINKS[f"r_{_f}_middle"] = (f"r_{_f}_pip", f"r_{_f}_distal")
        _LINKS[f"r_{_f}_distal"] = (f"r_{_f}_dip", None)


@dataclass(frozen=True)
class TaxelPatchSpec:
    """A conforming taxel patch on one body (plus rigidly attached bodies that share it)."""

    name: str
    body: str
    bodies: tuple[str, ...]  # bodies whose collision geoms load the patch
    grid: tuple[int, int]  # rows along u, columns along v
    pos: np.ndarray = field(repr=False)  # (rows*cols, 3) taxel centres, body frame, row-major
    normal: np.ndarray = field(repr=False)  # (rows*cols, 3) outward unit normals, body frame
    pitch: tuple[float, float]  # taxel spacing along u, v (m)
    center: tuple[float, float, float]  # patch frame (body frame): origin at the middle taxel's surface point
    u: tuple[float, float, float]
    v: tuple[float, float, float]
    accel: bool = False  # carries the hand's patch accelerometer

    @property
    def n_taxels(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def half(self) -> tuple[float, float]:
        return (0.5 * self.grid[0] * self.pitch[0], 0.5 * self.grid[1] * self.pitch[1])


def _ray_mesh(origin: np.ndarray, d: np.ndarray, tris: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Nearest hit of a ray on a triangle soup (Moller-Trumbore): (distance, triangle normal) or None."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - v0
    uu = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    vv = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (t > 1e-9)
    if not hit.any():
        return None
    i = np.flatnonzero(hit)[np.argmin(t[hit])]
    nrm = np.cross(e1[i], e2[i])
    return float(t[i]), nrm / np.linalg.norm(nrm)


def _body_tris(m: mujoco.MjModel, d: mujoco.MjData, body: int, frame_body: int) -> np.ndarray:
    """Triangles of a body's detailed (visual) meshes, in `frame_body`'s frame."""
    Rf, pf = d.xmat[frame_body].reshape(3, 3), d.xpos[frame_body]
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g]:
            continue
        mid = m.geom_dataid[g]
        va, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        v = m.mesh_vert[va:va + nv] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        v = (v - pf) @ Rf
        out.append(v[m.mesh_face[fa:fa + nf]])
    if not out:
        raise ValueError(f"body {m.body(body).name!r} has no visual mesh")
    return np.concatenate(out)


def _cast_grid(tris: np.ndarray, c0: np.ndarray, u: np.ndarray, v: np.ndarray, n: np.ndarray, su: np.ndarray,
               sv: np.ndarray, standoff: float = 0.04) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos, nrm, ok = [], [], []
    for a in su:
        for b in sv:
            o = c0 + a * u + b * v + standoff * n
            hit = _ray_mesh(o, -n, tris)
            if hit is None:
                pos.append(o - standoff * n)
                nrm.append(n)
                ok.append(False)
                continue
            t, tn = hit
            pos.append(o - t * n)
            nrm.append(tn if tn @ n > 0 else -tn)
            ok.append(True)
    return np.array(pos), np.array(nrm), np.array(ok)


def _width_along(tris: np.ndarray, c: np.ndarray, u: np.ndarray, v: np.ndarray, band: float = 0.003) -> float:
    pts = tris.reshape(-1, 3)
    s = (pts - c) @ u
    sel = np.abs(s) < band
    if sel.sum() < 3:
        sel = np.abs(s) < 3 * band
    w = (pts[sel] - c) @ v
    return float(np.ptp(w)), float(0.5 * (w.max() + w.min()))


def _max_rect(ok: np.ndarray) -> tuple[int, int, int, int]:
    """Largest all-True axis-aligned rectangle in a boolean map: (row0, row1, col0, col1), inclusive."""
    h = np.zeros(ok.shape[1], dtype=int)
    best, out = 0, (0, 0, 0, 0)
    for r in range(ok.shape[0]):
        h = np.where(ok[r], h + 1, 0)
        stack = []  # histogram largest rectangle
        for c in range(len(h) + 1):
            hc = h[c] if c < len(h) else 0
            start = c
            while stack and stack[-1][1] >= hc:
                start, hs = stack.pop()
                if hs * (c - start) > best:
                    best, out = hs * (c - start), (r - hs + 1, r, start, c - 1)
            stack.append((start, hc))
    return out


def _covered_rect(tris, c, u, v, n, step=0.001, inset=0.002, standoff=0.06, u_lim=None):
    """The largest rectangle in (u, v) about `c` whose every point has surface under it along -n:
    (u_lo, u_hi, v_lo, v_hi), shrunk by `inset` on every side."""
    pts = tris.reshape(-1, 3) - c
    su = np.arange(*(u_lim if u_lim is not None else np.quantile(pts @ u, [0, 1])), step)
    sv = np.arange(*np.quantile(pts @ v, [0, 1]), step)
    # the ray test is per-sample; restrict to triangles facing +n so back faces do not count
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    front = tris[fn @ n > 0]
    ok = np.array([[_ray_mesh(c + a * u + b * v + standoff * n, -n, front) is not None for b in sv] for a in su])
    r0, r1, c0, c1 = _max_rect(ok)
    return su[r0] + inset, su[r1] - inset, sv[c0] + inset, sv[c1] - inset


def _patch(name, body_name, frame_body, tris, c0, u, n, length, grid, len_frac, wid_frac, bodies, accel=False):
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    width, v_mid = _width_along(tris, c0 + 0.5 * length * u, u, v)
    rows, cols = grid
    lu, lv = len_frac[1] - len_frac[0], wid_frac
    su = length * (len_frac[0] + (np.arange(rows) + 0.5) / rows * lu)
    sv = v_mid + width * lv * ((np.arange(cols) + 0.5) / cols - 0.5)
    pos, nrm, ok = _cast_grid(tris, c0, u, v, n, su, sv)
    if not ok.all():
        raise ValueError(f"{name}: {np.sum(~ok)} taxel rays missed the {body_name} surface")
    mid = pos.reshape(rows, cols, 3)[rows // 2, cols // 2]
    return TaxelPatchSpec(name, body_name, tuple(bodies), (rows, cols), pos, nrm,
                          (length * lu / rows, width * lv / cols), tuple(mid), tuple(u), tuple(v), accel)


def _patch_rect(name, body_name, tris, c, u, v, n, u_rng, v_rng, grid, bodies, accel=False):
    rows, cols = grid
    pu, pv = (u_rng[1] - u_rng[0]) / rows, (v_rng[1] - v_rng[0]) / cols
    su = u_rng[0] + (np.arange(rows) + 0.5) * pu
    sv = v_rng[0] + (np.arange(cols) + 0.5) * pv
    pos, nrm, ok = _cast_grid(tris, c, u, v, n, su, sv)
    if not ok.all():
        raise ValueError(f"{name}: {np.sum(~ok)} taxel rays missed the {body_name} surface")
    mid = pos.reshape(rows, cols, 3)[rows // 2, cols // 2]
    return TaxelPatchSpec(name, body_name, tuple(bodies), (rows, cols), pos, nrm, (pu, pv), tuple(mid), tuple(u),
                          tuple(v), accel)


@lru_cache(maxsize=4)
def conforming_layout(hand_xml: str, palm_grid: tuple[int, int] = (8, 16), finger_grid: tuple[int, int] = (8, 4),
                      width_frac: float = 0.85) -> tuple[TaxelPatchSpec, ...]:
    """The TaxelScan layout on a WUJI right hand MJCF (vendor file): palm + distal pad + middle segment per finger."""
    m = mujoco.MjModel.from_xml_path(str(Path(hand_xml)))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = []

    # palm: the +y face of the mount frame (the hand frame), fingers along -z
    pb = m.body(PALM_BODY).id
    Rp = d.xmat[pb].reshape(3, 3)
    root = m.body("r_mount").id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_mount") >= 0 else 0
    Rr = d.xmat[root].reshape(3, 3)
    n_w, u_w = Rr @ np.array([0.0, 1.0, 0.0]), Rr @ np.array([0.0, 0.0, -1.0])
    n, u = Rp.T @ n_w, Rp.T @ u_w
    tris = _body_tris(m, d, pb, pb)
    # the palm is not a rectangle (knuckle notches, thenar mound): fit the sheet to the largest fully covered one
    v = np.cross(n, u)
    c = tris.reshape(-1, 3).mean(axis=0)
    u_lo, u_hi, v_lo, v_hi = _covered_rect(tris, c, u, v, n, step=0.002)
    out.append(_patch_rect("palm", PALM_BODY, tris, c, u, v, n, (u_lo, u_hi), (v_lo, v_hi), palm_grid,
                           (PALM_BODY,), accel=True))

    for f in FINGERS:
        short = "thumb" if f == "thumb" else f.replace("_finger", "")
        for seg in ("middle", "distal"):
            bname = f"r_{f}_{seg}"
            b = m.body(bname).id
            jn, end_body = _LINKS[bname]
            j = m.joint(jn).id
            Rb, xb = d.xmat[b].reshape(3, 3), d.xpos[b]
            anchor = Rb.T @ (d.xanchor[j] - xb)
            axis = Rb.T @ d.xaxis[j]
            bodies = [bname]
            tris = _body_tris(m, d, b, b)
            if seg == "distal":
                tip = m.body(f"r_{f}_tip_sensor_frame").id
                tris = np.concatenate([tris, _body_tris(m, d, tip, b)])
                bodies.append(f"r_{f}_tip_sensor_frame")
                # the link runs from the joint to the farthest point of the pad along the finger
                pts = tris.reshape(-1, 3)
                far = pts[np.argmax(np.linalg.norm(pts - anchor, axis=1))]
                r = far - anchor
            else:
                eb = m.body(end_body).id
                r = Rb.T @ (d.xpos[eb] - xb) - anchor
            length = float(np.linalg.norm(r))
            u = r / length
            n = np.cross(axis, u)  # closing (+angle) moves the link along axis x r: the palmar side
            n -= (n @ u) * u
            n /= np.linalg.norm(n)
            frac = (0.12, 0.88) if seg == "middle" else (0.10, 0.80)
            v = np.cross(n, u)
            u_lo, u_hi, v_lo, v_hi = _covered_rect(tris, anchor, u, v, n, step=0.0005, inset=0.001,
                                                   u_lim=(frac[0] * length, frac[1] * length))
            # stay on the pad's front: no wider than the given fraction of the covered width, centred on it
            vc, vh = 0.5 * (v_lo + v_hi), 0.5 * min(v_hi - v_lo, width_frac * (v_hi - v_lo + 2e-3))
            out.append(_patch_rect(f"{short}_{seg}", bname, tris, anchor, u, v, n, (u_lo, u_hi), (vc - vh, vc + vh),
                                   finger_grid, bodies))
    return tuple(out)


def palm_view(hand_xml: str, layout: tuple[TaxelPatchSpec, ...]) -> list[dict]:
    """A hand-shaped map of the skins: each patch unrolled flat at its true pitch, centred where it sits on the open
    hand (zero pose) seen from the palm, and turned so its rows run along the link as projected. x toward the
    thumb, y toward the fingertips (mm). Returns per patch the taxel centres "xy" (row-major), the taxel size
    "cell" (across, along) and the rows' direction "angle" (rad from +y)."""
    m = mujoco.MjModel.from_xml_path(str(Path(hand_xml)))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    root = m.body("r_mount").id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_mount") >= 0 else 0
    Rr, xr = d.xmat[root].reshape(3, 3), d.xpos[root]

    def proj(v_hand):  # hand frame -> view (x, y)
        return np.array([v_hand[0], -v_hand[2]])

    out = []
    for p in layout:
        b = m.body(p.body).id
        Rb = d.xmat[b].reshape(3, 3)
        ph = ((p.pos @ Rb.T + d.xpos[b]) - xr) @ Rr
        c = proj(ph.mean(axis=0)) * 1e3
        u2 = proj(Rr.T @ Rb @ np.asarray(p.u))
        u2 = u2 / np.linalg.norm(u2) if np.linalg.norm(u2) > 1e-6 else np.array([0.0, 1.0])
        v2 = np.array([u2[1], -u2[0]])  # across the rows, to the right of the rows' direction
        v_proj = proj(Rr.T @ Rb @ np.asarray(p.v))
        if v_proj @ v2 < 0:
            v2 = -v2
        rows, cols = p.grid
        pu, pv = p.pitch[0] * 1e3, p.pitch[1] * 1e3
        r, k = np.meshgrid(np.arange(rows) - (rows - 1) / 2, np.arange(cols) - (cols - 1) / 2, indexing="ij")
        xy = c + (r.reshape(-1, 1) * pu) * u2 + (k.reshape(-1, 1) * pv) * v2
        out.append({"xy": np.round(xy, 2), "cell": (round(pv, 2), round(pu, 2)),
                    "angle": round(float(np.arctan2(u2[0], u2[1])), 4)})
    return out
