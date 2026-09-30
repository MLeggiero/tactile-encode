"""Replay export: mesh simplification, scene geometry and the standalone page."""

import json

import numpy as np
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
    assert meshes == {}
    assert any(g["type"] == "capsule" and g["body"] == "fr3_link2" for g in geoms)


def test_standalone_page(tmp_path):
    data = run_and_export(["default"], n=1, fast=True)
    ep = data["episodes"][0]
    assert len(ep["frames"]["t"]) == len(ep["frames"]["pos"]) == len(ep["frames"]["quat"])
    assert len(ep["frames"]["pos"][0]) == 3 * len(ep["bodies"])
    assert ep["strikes"][0]["hit"]
    out = build_html(data, tmp_path / "replay.html")
    html = out.read_text()
    assert "<title>Nail Strike Replay</title>" in html and "/*__REPLAY_DATA__*/" not in html
    assert '"episodes"' in html
