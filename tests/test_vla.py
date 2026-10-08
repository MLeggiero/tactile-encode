"""The VLA's side of the simulation: the hand's MIT table and table-equivalent targets, the head camera (aim,
render, visual randomization), scripted demos and their LeRobot export."""

import json

import numpy as np
import pytest

import tactile_sim.vla  # noqa: F401 - selects EGL before mujoco's first import, when this runs first
from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.config import residual_config
from tactile_sim.control.hand import WrapSynergy
from tactile_sim.vla.action import ACTION_DIM, STATE_DIM, HandTable

requires_vega_wuji = pytest.mark.skipif(not (hand_available("wuji2") and hand_available("vega_1u")),
                                        reason="WUJI Hand 2 or the Vega U not cached")

FINGER = ["r_index_finger_mcp_flex", "r_index_finger_mcp_abd", "r_index_finger_pip", "r_index_finger_dip"]


def synergy():
    return WrapSynergy.for_wuji([(n, t) for n, t in zip(FINGER, (2.0, 1.0, 0.3, 0.3), strict=True)],
                                thumb_close=(1, 0, 1, 1))


def egl_renderer_or_skip(model, cfg):
    from tactile_sim.vla.camera import EgoCamera

    try:
        return EgoCamera(model, cfg)
    except Exception as e:  # noqa: BLE001 - no GL backend: nothing to test here
        pytest.skip(f"offscreen rendering unavailable ({type(e).__name__}: {e}); run with MUJOCO_GL=egl")


def test_table_targets_reproduce_the_synergy_hold():
    syn = synergy()
    kp_nom = (0.37, 0.43, 0.22, 0.55, 0.22)
    table = HandTable.from_synergy(syn, 0.6, FINGER, kp_nom)
    assert table.kp.tolist() == pytest.approx([0.37, syn.kp, 0.22, syn.coupling_kp])  # closing joints: vendor Kp
    q = np.array([1.1, 0.05, 0.9, 0.5])
    # at the table's grip level the closing joints hold where they are, the shaping one at its keyframe, the
    # coupled fingertip at its share of the middle joint
    q_star = table.equivalent_targets(np.stack(syn.mit_command(0.6, q)), q)
    assert q_star == pytest.approx([1.1, syn.q_hold[1], 0.9, syn.coupling_ratio * 0.9])
    # any grip level: the same static torque at q
    for s in (0.0, 0.3, 1.0):
        base = np.stack(syn.mit_command(s, q))
        q_star = table.equivalent_targets(base, q)
        tau_base = base[2] * (base[0] - q) + base[4]
        cmd = table.command(q_star)
        assert cmd.shape == (5, 4) and np.all(cmd[1] == 0)
        assert cmd[2] * (cmd[0] - q) + cmd[4] == pytest.approx(tau_base)
    assert HandTable.from_dict(json.loads(json.dumps(table.to_dict()))).tau == pytest.approx(table.tau)


def test_the_experiment_base_holds_open_loop():
    c = residual_config()
    assert c.controller.grip_loop == "open" and c.swing.grip_profile == "constant" and c.swing.grip_margin_step == 0


@requires_vega_wuji
def test_the_head_is_aimed_at_the_task():
    import xml.etree.ElementTree as ET

    import mujoco

    from tactile_sim import names
    from tactile_sim.model.builder import add_ego_camera
    from tactile_sim.model.robots import load_vega_tree
    from tactile_sim.vla.camera import HEAD_JOINTS, aim_head, look_point, project, vla_config

    cfg = residual_config()
    pose = aim_head(cfg)
    assert vla_config(cfg).arm.head_pose == pose and pose[2] == 0.0
    assert 0.2 < pose[0] < 1.2  # pitched down at the table
    root, _ = load_vega_tree(cfg.arm)
    add_ego_camera(root, cfg)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    for j, v in zip(HEAD_JOINTS, pose, strict=True):
        d.qpos[m.joint(j).qposadr[0]] = v
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    cam = m.camera(names.EGO_CAMERA).id
    u, v, depth = project(m, d, cam, look_point(cfg), cfg.vla.cam_width, cfg.vla.cam_height)
    assert (u, v) == pytest.approx((cfg.vla.cam_width / 2, cfg.vla.cam_height / 2), abs=1.0)
    assert 0.3 < depth < 1.5
    fixed = cfg.replace(vla={"head_pose": (0.4, 0.0, 0.0)})
    assert vla_config(fixed).arm.head_pose == (0.4, 0.0, 0.0)


@requires_vega_wuji
def test_ego_camera_renders_and_randomizes_only_the_scene():
    import mujoco

    from tactile_sim.model.builder import build_scene

    cfg = residual_config()
    m = mujoco.MjModel.from_xml_string(build_scene(cfg).xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = egl_renderer_or_skip(m, cfg)
    try:
        img = cam.render(d)
        assert img.shape == (cfg.vla.cam_height, cfg.vla.cam_width, 3) and img.dtype == np.uint8
        assert img.std() > 5.0  # not a blank frame
        rgba0, pos0 = m.geom_rgba.copy(), m.cam_pos[cam.cam].copy()
        draw = cam.randomize(np.random.default_rng(3))
        changed = np.flatnonzero(np.any(m.geom_rgba != rgba0, axis=1))
        scene = np.concatenate(cam.groups)
        assert len(changed) and set(changed) <= set(scene.tolist())  # the robot keeps its color
        assert np.abs(m.geom_rgba[:, :3] - rgba0[:, :3]).max() <= cfg.vla.color_jitter + 1e-9
        assert np.abs(m.cam_pos[cam.cam] - pos0).max() <= cfg.vla.cam_pos_jitter
        assert len(draw["color_offsets"]) == len(cam.groups)
        mujoco.mj_forward(m, d)
        assert not np.array_equal(cam.render(d), img)
        cam.nominal()
        assert np.array_equal(m.geom_rgba, rgba0) and np.array_equal(m.cam_pos[cam.cam], pos0)
    finally:
        cam.close()


def fake_raw(tmp_path, n_ok=2, n_fail=1, frames=8):
    """Recorded-demo files as vla.demos writes them, with tiny videos."""
    pytest.importorskip("pyarrow")
    from tactile_sim.vla.demos import ImageStats, VideoWriter, ffmpeg_exe

    try:
        ffmpeg_exe()
    except RuntimeError:
        pytest.skip("no ffmpeg")
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg = residual_config()
    rng = np.random.default_rng(0)
    for i in range(n_ok + n_fail):
        stem = raw / f"episode_{i:06d}"
        v = VideoWriter(stem.with_suffix(".mp4"), cfg.vla.cam_width, cfg.vla.cam_height, cfg.vla.fps)
        st = ImageStats()
        for _ in range(frames):
            f = rng.integers(0, 256, (cfg.vla.cam_height, cfg.vla.cam_width, 3), dtype=np.uint8)
            v.write(f)
            st.update(f)
        v.close()
        meta = {"task": "hammer", "instruction": "hammer the nail into the board", "seed": i, "fps": cfg.vla.fps,
                "success": i < n_ok, "hand_table": {"kp": [1.0] * 20, "kd": [0.01] * 20, "tau": [0.0] * 20},
                "hand_joints": [f"j{k}" for k in range(20)], "head_pose": [0.5, 0.0, 0.0],
                "config": cfg.to_dict()}
        np.savez(stem.with_suffix(".npz"), state=rng.normal(size=(frames, STATE_DIM)).astype(np.float32),
                 action=rng.normal(size=(frames, ACTION_DIM)).astype(np.float32), t=np.arange(frames) / 30.0,
                 image_hist=st.hist, meta=json.dumps(meta))
    return raw


def test_lerobot_export_writes_the_groot_layout(tmp_path):
    import pyarrow.parquet as pq

    from tactile_sim.vla.lerobot_export import export

    raw = fake_raw(tmp_path)
    out = tmp_path / "ds"
    s = export([raw], out)
    assert s == {"episodes": 2, "frames": 16, "skipped": 1, "tasks": ["hammer the nail into the board"]}
    info = json.loads((out / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["total_episodes"] == 2 and info["total_frames"] == 16
    assert info["fps"] == 30.0 and info["features"]["observation.state"]["shape"] == [STATE_DIM]
    assert info["features"]["observation.images.ego_view"]["shape"] == [240, 384, 3]
    t = pq.read_table(out / "data/chunk-000/episode_000001.parquet")
    assert t.column("observation.state").type.list_size == STATE_DIM
    assert t.column("index").to_pylist() == list(range(8, 16))
    assert t.column("timestamp").to_pylist() == pytest.approx([k / 30.0 for k in range(8)])
    assert (out / "videos/chunk-000/observation.images.ego_view/episode_000001.mp4").stat().st_size > 0
    eps = [json.loads(x) for x in (out / "meta/episodes.jsonl").read_text().splitlines()]
    assert [e["length"] for e in eps] == [8, 8] and all(e["success"] for e in eps)
    stats = json.loads((out / "meta/stats.json").read_text())
    assert set(stats["action"]) >= {"mean", "std", "min", "max", "q01", "q99"}
    assert np.array(stats["observation.images.ego_view"]["mean"]).shape == (3, 1, 1)
    mod = json.loads((out / "meta/modality.json").read_text())
    assert mod["action"]["right_arm"] == {"start": 0, "end": 7} and mod["action"]["right_hand"]["end"] == ACTION_DIM
    assert len((out / "meta/episodes_stats.jsonl").read_text().splitlines()) == 2
    assert export([raw], out, only_success=False, overwrite=True)["episodes"] == 3
    with pytest.raises(FileExistsError):
        export([raw], out)


@requires_vega_wuji
def test_a_demo_records_the_vla_view_state_and_action(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import tactile_sim.vla.demos as demos

    try:
        demos.ffmpeg_exe()
    except RuntimeError:
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(demos, "MAX_STRIKES", 1)
    job = demos.DemoJob("hammer", 0, 7, str(tmp_path), fast=True)
    try:
        r = demos.run_demo(job)
    except Exception as e:  # noqa: BLE001
        if "EGL" in str(e) or "GL" in type(e).__name__ or "gl" in str(e).lower():
            pytest.skip(f"offscreen rendering unavailable: {e}")
        raise
    assert r["status"] == "ok" and not r["success"]  # one strike cannot drive the nail home
    d = demos.load_demo(job.stem.with_suffix(".npz"))
    m = d["meta"]
    n = len(d["t"])
    assert n == r["frames"] and d["state"].shape == (n, STATE_DIM) and d["action"].shape == (n, ACTION_DIM)
    assert np.allclose(np.diff(d["t"]), 1 / 30.0, atol=1.1e-3)  # 30 Hz on the 1 ms clock
    assert m["n_strikes"] == 1 and m["depth"] > 1e-3
    closing = np.array(m["hand_table"]["kp"]) < 1.0  # the vendor-Kp joints
    # the base holds its grip at the table's level: its closing joints' targets are where they are
    assert d["action"][:, 7:][:, closing] == pytest.approx(d["state"][:, 7:][:, closing], abs=1e-5)
    assert abs(d["action"][:, :7] - d["state"][:, :7]).max() < 0.2  # arm targets lead the joints a little
    assert d["image_hist"].sum() == 3 * n * 240 * 384
    assert job.stem.with_suffix(".mp4").stat().st_size > 0
    assert demos.run_demo(job)["status"] == "skip"


def test_chunks_interpolate_and_hold():
    from tactile_sim.vla.client import Chunk, ReplayPolicy

    a = np.arange(3 * ACTION_DIM, dtype=float).reshape(3, ACTION_DIM)
    c = Chunk(1.0, 1.1, 0.5, a, np.zeros(ACTION_DIM))
    q, qd = c.at(1.25)
    assert q == pytest.approx(a[0] + 0.5 * (a[1] - a[0])) and qd == pytest.approx((a[1] - a[0]) / 0.5)
    assert c.at(0.0)[0] == pytest.approx(a[0]) and c.at(9.0)[0] == pytest.approx(a[2])
    assert not c.at(9.0)[1].any()
    p = ReplayPolicy(a, fps=2.0, horizon=4, t_start=1.0)
    assert p(1.5, None, None, "")[:, 0].tolist() == [a[1, 0], a[2, 0], a[2, 0], a[2, 0]]
    mid = 0.5 * (a[:-1, 0] + a[1:, 0])  # between frames the replay interpolates the recording
    assert p(1.25, None, None, "")[:, 0] == pytest.approx([mid[0], mid[1], a[2, 0], a[2, 0]])
    assert p(0.0, None, None, "")[0] == pytest.approx(a[0])


def test_a_new_chunk_fades_in_over_the_last():
    from tactile_sim.vla.client import Chunk, VlaBase

    base = VlaBase.__new__(VlaBase)  # targets() needs only the chunks once one is active
    base.blend = 0.1
    old = Chunk(0.0, 0.1, 0.05, np.zeros((16, ACTION_DIM)), np.zeros(ACTION_DIM))
    new = Chunk(0.2, 0.3, 0.05, np.ones((16, ACTION_DIM)), np.zeros(ACTION_DIM))
    base.chunks = [old, new]
    assert not base.targets(0.3 - 1e-6)[0].any() and not base.targets(0.3)[0].any()  # no step at the switch
    q, qd = base.targets(0.35)
    assert q == pytest.approx(0.5) and qd == pytest.approx(15.0)  # smoothstep: half way, at its steepest
    assert base.targets(0.4)[0] == pytest.approx(1.0) and not base.targets(0.4)[1].any()
    base.blend = 0.0
    assert base.targets(0.3)[0] == pytest.approx(1.0)  # without a blend the new chunk takes over at once


def test_smoothing_a_chunk_removes_alternation_and_keeps_a_ramp():
    from tactile_sim.vla.client import smooth_chunk

    ramp = np.linspace(0.0, 1.5, 16)[:, None] * np.ones(ACTION_DIM)
    jitter = 0.03 * (-1.0) ** np.arange(16)[:, None] * np.ones(ACTION_DIM)
    out = smooth_chunk(ramp + jitter, 1)
    assert np.abs(out - ramp)[1:-1].max() < 1e-12  # the kernel cancels a period-2 alternation and passes a line
    assert smooth_chunk(ramp, 0) is ramp


@requires_vega_wuji
def test_a_replayed_demo_hammers_through_the_vla_client(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import tactile_sim.vla.demos as demos
    from tactile_sim.vla.client import ReplayPolicy, run_vla_episode

    try:
        demos.ffmpeg_exe()
    except RuntimeError:
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(demos, "MAX_STRIKES", 1)
    job = demos.DemoJob("hammer", 0, 11, str(tmp_path), fast=True)
    try:
        demos.run_demo(job)
    except Exception as e:  # noqa: BLE001
        if "gl" in str(e).lower():
            pytest.skip(f"offscreen rendering unavailable: {e}")
        raise
    d = demos.load_demo(job.stem.with_suffix(".npz"))
    cfg, _ = demos.demo_config("hammer", job.seed, fast=True)
    policy = ReplayPolicy(d["action"], d["meta"]["fps"], horizon=16, t_start=float(d["t"][0]))
    r = run_vla_episode(cfg, policy, seed=job.seed, latency=0.0, exec_steps=8, max_time=float(d["t"][-1]) + 0.3)
    assert not r["dropped"] and r["residual"]["nonfinite"] == 0
    assert len(r["strikes"]) == 1 and r["depth"] > 1e-3  # the replayed swing lands, on time
    # but softer: 30 Hz targets cannot carry the strike profile's braking in the last ~30 ms, so a replayed (or VLA)
    # blow meets the nail at 0.7-0.9 m/s where the scripted one did at 1.16, and drives it 40-55 % as far
    assert r["depth"] < 1.3 * d["meta"]["depth"]
    assert r["n_chunks"] == len(r["chunks"]["t_req"]) >= len(d["t"]) // 8


@requires_vega_wuji
def test_a_drill_demo_pulls_the_trigger_through_the_index_targets(tmp_path):
    pytest.importorskip("pyarrow")
    import tactile_sim.vla.demos as demos
    from tactile_sim.control.hand import TRIGGER_FINGER

    try:
        demos.ffmpeg_exe()
    except RuntimeError:
        pytest.skip("no ffmpeg")
    job = demos.DemoJob("drill", 0, 3, str(tmp_path), fast=True)
    try:
        r = demos.run_demo(job)
    except Exception as e:  # noqa: BLE001
        if "gl" in str(e).lower():
            pytest.skip(f"offscreen rendering unavailable: {e}")
        raise
    d = demos.load_demo(job.stem.with_suffix(".npz"))
    m = d["meta"]
    assert r["success"] and m["instruction"] == demos.INSTRUCTIONS["drill"] and m["cam_outs"] <= 3
    mcp = 7 + m["hand_joints"].index(TRIGGER_FINGER[0])
    lead = d["action"][:, mcp] - d["state"][:, mcp]  # the index finger's target ahead of where it is
    assert abs(lead[0]) < 0.05  # resting on the trigger: the table's own level
    assert lead.max() > 0.4  # a full pull, as a target past the finger
    assert abs(lead[-1]) < 0.1  # let go by the end


def test_openpi_wire_format_round_trips():
    pytest.importorskip("msgpack")
    import msgpack

    from tactile_sim.vla.client import OpenPiPolicy

    obs = {"observation/image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
           "observation/state": np.linspace(-1, 1, 27).astype(np.float32), "prompt": "drive the screw into the board",
           "n": np.float32(2.5)}
    back = msgpack.unpackb(msgpack.packb(obs, default=OpenPiPolicy._pack), object_hook=OpenPiPolicy._unhook)
    assert back["prompt"] == obs["prompt"] and back["n"] == np.float32(2.5)
    for k in ("observation/image", "observation/state"):
        assert back[k].dtype == obs[k].dtype and np.array_equal(back[k], obs[k])


@requires_vega_wuji
def test_a_replayed_drill_demo_seats_the_screw_through_the_vla_client(tmp_path):
    pytest.importorskip("pyarrow")
    import tactile_sim.vla.demos as demos
    from tactile_sim.vla.client import ReplayPolicy, run_vla_episode

    try:
        demos.ffmpeg_exe()
    except RuntimeError:
        pytest.skip("no ffmpeg")
    job = demos.DemoJob("drill", 0, 3, str(tmp_path), fast=True)
    try:
        demos.run_demo(job)
    except Exception as e:  # noqa: BLE001
        if "gl" in str(e).lower():
            pytest.skip(f"offscreen rendering unavailable: {e}")
        raise
    d = demos.load_demo(job.stem.with_suffix(".npz"))
    assert d["meta"]["success"]
    cfg, _ = demos.demo_config("drill", job.seed, fast=True)
    policy = ReplayPolicy(d["action"], d["meta"]["fps"], horizon=16, t_start=float(d["t"][0]))
    r = run_vla_episode(cfg, policy, seed=job.seed, latency=0.0, exec_steps=8, max_time=float(d["t"][-1]) + 0.5)
    # the trigger is pulled by the index finger's replayed targets alone, and the screw goes home
    assert r["task"] == "drill" and r["done"] and not r["dropped"] and r["cam_outs"] <= 6
    assert r["max_slip_rot_deg"] < 15.0 and r["residual"]["nonfinite"] == 0
