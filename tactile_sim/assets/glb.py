"""Minimal binary glTF (GLB) mesh reader: triangle geometry only, node transforms applied, written as OBJ.

MuJoCo reads OBJ/STL but not glTF. Dexmate ships some Vega U links (pedestal, lift, torso) only as GLB.
Supports uncompressed float32 POSITION and uint8/16/32 indices, mode 4 (triangles); materials ignored.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_COMP = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _accessor(gl: dict, blob: bytes, idx: int) -> np.ndarray:
    acc = gl["accessors"][idx]
    view = gl["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(_COMP[acc["componentType"]])
    n = _NCOMP[acc["type"]]
    off = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", 0) or dtype.itemsize * n
    count = acc["count"]
    raw = np.frombuffer(blob, dtype=np.uint8, count=stride * (count - 1) + dtype.itemsize * n, offset=off)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dtype.itemsize * n), strides=(stride, 1))
    return np.ascontiguousarray(rows).view(dtype).reshape(count, n)


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T  # glTF is column-major
    M = np.eye(4)
    if "scale" in node:
        M = np.diag(list(node["scale"]) + [1.0]) @ M
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        T = np.eye(4)
        T[:3, :3] = R
        M = T @ M
    if "translation" in node:
        T = np.eye(4)
        T[:3, 3] = node["translation"]
        M = T @ M
    return M


def read_glb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """All triangles of the default scene: (vertices N x 3, faces M x 3)."""
    data = Path(path).read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path} is not a binary glTF file")
    n_json = struct.unpack("<I", data[12:16])[0]
    gl = json.loads(data[20:20 + n_json])
    off = 20 + n_json
    blob = b""
    if off < len(data):
        n_bin = struct.unpack("<I", data[off:off + 4])[0]
        blob = data[off + 8:off + 8 + n_bin]
    for ext in gl.get("extensionsRequired", []):
        raise ValueError(f"{path}: compressed glTF ({ext}) is not supported")
    verts, faces = [], []
    n_v = 0

    def visit(i: int, parent: np.ndarray) -> None:
        nonlocal n_v
        node = gl["nodes"][i]
        M = parent @ _node_matrix(node)
        if "mesh" in node:
            for prim in gl["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                v = _accessor(gl, blob, prim["attributes"]["POSITION"]).astype(float)
                v = v @ M[:3, :3].T + M[:3, 3]
                f = (_accessor(gl, blob, prim["indices"]).reshape(-1, 3) if "indices" in prim
                     else np.arange(len(v)).reshape(-1, 3))
                verts.append(v)
                faces.append(f.astype(np.int64) + n_v)
                n_v += len(v)
        for c in node.get("children", []):
            visit(c, M)

    scene = gl["scenes"][gl.get("scene", 0)]
    for r in scene["nodes"]:
        visit(r, np.eye(4))
    return np.concatenate(verts), np.concatenate(faces)


def glb_to_obj(src: Path, dst: Path) -> Path:
    v, f = read_glb(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v] + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
    tmp = dst.with_suffix(".part")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(dst)
    return dst
