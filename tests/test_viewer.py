"""Replay export: mesh simplification, scene geometry and the standalone page."""
import json

import numpy as np
import pytest
from conftest import requires_menagerie

from tactile_sim.config import fast_config
from tactile_sim.sim.world import World
from tactile_sim.viewer.export import build_html, cluster_decimate, run_and_export, scene_geometry


def test_cluster_decimate_keeps_shape():
    rng = np.random.default_rng(0)
    u, v = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))
    verts = np.c_[u.ravel(), v.ravel(), np.full(u.size, 0.025)] * 0.1  # sheet mid-cell in z
    idx = np.arange(u.size).reshape(60, 60)
    faces = np.r_[np.c_[idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()],
                  np.c_[idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()]]
    v2, f2 = cluster_decimate(verts + rng.normal(0, 1e-5, verts.shape), faces, 0.005)
    assert len(f2) < len(faces) / 4 and len(f2) > 0
    assert np.allclose(v2.min(0)[:2], 0.0, atol=0.003) and np.allclose(v2.max(0)[:2], 0.1, atol=0.003)
    assert np.all(f2[:, 0] != f2[:, 1])


@requires_menagerie
def test_real_robot_meshes_are_exported():
    w = World(fast_config())
    assert w.arm_source == "fr3" and w.hand_source == "franka"
    bodies, geoms, meshes = scene_geometry(w.model)
    mesh_bodies = {g["body"] for g in geoms if g["type"] == "mesh"}
    assert {"fr3_link1", "fr3_link7", "hand", "left_finger", "right_finger"} <= mesh_bodies
    assert all(g["mesh"] in meshes for g in geoms if g["type"] == "mesh")
    assert len(json.dumps(meshes)) < 2_000_000


def test_fallback_scene_draws_capsules():
    w = World(fast_config(arm={"source": "fallback"}, gripper={"hand_source": "box"}))
    _, geoms, meshes = scene_geometry(w.model)
    assert all(k.startswith("hammer") for k in meshes)  # only the tool is a mesh
    assert any(g["type"] == "capsule" and g["body"] == "fr3_link2" for g in geoms)


def test_standalone_page(tmp_path):
    data = run_and_export(["default"], n=1, fast=True)
    ep = data["episodes"][0]
    F = ep["frames"]
    import base64
    nb = len(ep["bodies"])
    assert F["n"] == len(F["t"])
    assert len(base64.b64decode(F["pos"])) == 2 * 3 * nb * F["n"]  # int16 positions
    assert len(base64.b64decode(F["quat"])) == 2 * 4 * nb * F["n"]
    assert ep["strikes"][0]["hit"]
    tac = ep["tactile"]
    assert (tac["rows"], tac["cols"]) == (8, 8)
    assert [p["name"] for p in tac["patches"]] == ["L", "R"]
    raw = base64.b64decode(tac["data"])
    assert len(raw) == len(tac["t"]) * 2 * 64 and max(raw) > 0  # both pads, one byte per taxel
    assert ep["limits"]["violations"] == ""
    out = build_html(data, tmp_path / "replay.html")
    html = out.read_text()
    assert "<title>Nail Strike Replay</title>" in html and "/*__REPLAY_DATA__*/" not in html
    assert '"episodes"' in html


@pytest.mark.skipif(not __import__("tactile_sim.assets.fetch_hands", fromlist=["x"]).hand_available("wuji2"),
                    reason="WUJI Hand 2 not cached")
def test_wuji_scene_and_patches_export():
    from tactile_sim.config import hand_config

    w = World(hand_config("wuji2", fast_config()))
    bodies, geoms, meshes = scene_geometry(w.model)
    assert any(m.startswith("wuji_") for m in meshes)  # the vendor's hand meshes are drawn
    # TaxelScan skins: every taxel drawn where it sits on the hand, no flat plates
    assert not any(g["name"].startswith("patch_") for g in geoms)
    assert sum(g["name"].startswith("taxel_") for g in geoms) == 448
    assert not any(g["name"].startswith("wuji_col") for g in geoms)  # collision hulls are not
