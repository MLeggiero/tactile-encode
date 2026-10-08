"""E2 (the VLA with and without the residual): conditions, the paired report, and the GR00T client's protocol
against a stand-in server."""

import io
import json
import threading

import numpy as np
import pytest

from tactile_sim.experiments.vla_experiment import Job, conditions, make_vla_policy, per_seed, report


def test_conditions_pair_the_vla_with_each_residual():
    from pathlib import Path

    tiny = str(Path(__file__).parent / "data" / "tiny_policy.onnx")
    c = conditions({"r": tiny})
    assert [x.name for x in c] == ["vla", "vla+residual:r"] and c[1].overrides["residual"]["rate"] == 1000.0
    j = Job("drill", c[1], 7, "groot://localhost:5555")
    assert j.path(Path("out")) == Path("out/vla__residual_r/seed7.json")
    with pytest.raises(ValueError):
        make_vla_policy("http://x:1")


def _result(cond, seed, task, done, slip):
    r = {"job": {"condition": cond, "seed": seed}, "task": task, "done": done, "dropped": False,
         "t_done": 3.0 if done else float("nan")}
    if task == "drill":
        r.update(cam_outs=1, max_slip_rot_deg=slip, max_slip_trans_mm=1.0)
    else:
        r["strikes"] = [{"advance": 0.002, "slip_rot_deg": slip}]
    return r


def test_report_pairs_seeds(tmp_path):
    for seed in range(4):
        for cond, slip in (("vla", 5.0 + seed), ("vla+residual:r", 3.0 + seed)):
            d = tmp_path / cond.replace(":", "_").replace("+", "__")
            d.mkdir(exist_ok=True)
            (d / f"seed{seed}.json").write_text(json.dumps(_result(cond, seed, "drill", seed > 0, slip)))
    ps = per_seed([json.loads(p.read_text()) for p in tmp_path.glob("*/seed*.json")])
    assert ps["vla"][1]["slip_rot_deg"] == 6.0 and np.isnan(ps["vla"][0]["t_done"])
    text = report("drill", tmp_path)
    assert "| vla+residual:r | 4 |" in text and "(-2 [-2, -2])" in text  # 2 degrees less slip on every seed


def test_groot_client_speaks_the_server_protocol():
    zmq = pytest.importorskip("zmq")
    msgpack = pytest.importorskip("msgpack")

    def enc(o):
        if isinstance(o, np.ndarray):
            b = io.BytesIO()
            np.save(b, o, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": b.getvalue()}
        return o

    def dec(o):
        if isinstance(o, dict) and "__ndarray_class__" in o:
            return np.load(io.BytesIO(o["as_npy"]), allow_pickle=False)
        return o

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    seen = {}

    def serve():
        for _ in range(2):
            req = msgpack.unpackb(sock.recv(), object_hook=dec)
            if req["endpoint"] == "get_action":
                obs = req["data"]["observation"]
                seen["video"] = obs["video"]["ego_view"].shape
                seen["arm"] = obs["state"]["right_arm"].shape
                seen["text"] = obs["language"]["annotation.human.task_description"]
                b = obs["state"]["right_arm"].shape[0]
                act = {"right_arm": np.ones((b, 16, 7), np.float32), "right_hand": np.zeros((b, 16, 20), np.float32)}
                sock.send(msgpack.packb([act, {}], default=enc))
            else:
                sock.send(msgpack.packb({"status": "ok"}))

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    p = make_vla_policy(f"groot://127.0.0.1:{port}")
    p.reset()
    a = p(0.0, np.zeros((240, 384, 3), np.uint8), np.zeros(27), "hammer the nail into the board")
    th.join(timeout=10)
    sock.close()
    assert a.shape == (16, 27) and a[:, :7].min() == 1.0 and a[:, 7:].max() == 0.0
    assert seen == {"video": (1, 1, 240, 384, 3), "arm": (1, 1, 7), "text": [["hammer the nail into the board"]]}
