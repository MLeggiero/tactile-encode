"""Recorded demos (vla.demos) -> a GR00T-flavored LeRobot v2.1 dataset, laid out like the user's G1_INSPIRE set
(~/groot_n1_7_redcup) so the same GR00T N1.7 and openpi tooling reads it:

    data/chunk-000/episode_000000.parquet      observation.state (27), action (27), timestamp, frame_index,
                                               episode_index, index, task_index, annotation.human.task_description
    videos/chunk-000/observation.images.ego_view/episode_000000.mp4
    meta/info.json  episodes.jsonl  tasks.jsonl  stats.json  episodes_stats.jsonl  modality.json
    meta/vega_wuji.json                        what the vectors mean, the camera, the hand's MIT table per task

State and action are absolute (vla.action): GR00T's modality config makes the arm relative itself
(ActionRepresentation.RELATIVE), openpi's DeltaActions transform does the same. Only successful episodes are
exported unless asked otherwise.

    python -m tactile_sim.vla.lerobot_export runs/vla_demos/hammer --out runs/lerobot/vega_wuji_hammer
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tactile_sim.vla.action import ACTION_DIM, ARM, ARM_NAMES, HAND, STATE_DIM
from tactile_sim.vla.demos import VIDEO_KEY, ImageStats, load_demo

CODEBASE_VERSION = "v2.1"
ROBOT_TYPE = "vega_1u_wuji2"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
SCALARS = ("timestamp", "frame_index", "episode_index", "index", "task_index", "annotation.human.task_description")


def modality() -> dict:
    """GR00T's meta/modality.json: slices of the state and action vectors, the camera and the instruction."""
    groups = {"right_arm": {"start": ARM.start, "end": ARM.stop}, "right_hand": {"start": HAND.start, "end": HAND.stop}}
    return {"state": groups, "action": json.loads(json.dumps(groups)),
            "video": {"ego_view": {"original_key": VIDEO_KEY}},
            "annotation": {"human.task_description": {"original_key": "annotation.human.task_description"}}}


def tool_of(cfg: dict) -> dict:
    """The tool an episode's configuration (SimConfig.to_dict) held and the wrap it held it in, for the sidecar
    (nothing the randomization draws per episode, such as the head's mass)."""
    g = cfg["gripper"]
    wrap = {"wrap_tcp": list(g["wrap_tcp"]), "wrap_yaw": g.get("wrap_yaw", 0.0),
            "thumb_close": list(g["thumb_close"]), "thumb_preshape": list(g["thumb_preshape"]),
            "finger_dip": g.get("finger_dip")}
    if cfg["plant"]["kind"] == "drill":
        d = cfg["drill"]
        return {"drill": d.get("model"), "form": d.get("form"), "body": d.get("pistol_body", "primitive"), **wrap}
    h = cfg["hammer"]
    return {"hammer": h["model"], "grip_from_head": h["grip_from_head"], **wrap}


def vector_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64)
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(), "max": x.max(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(), "q99": np.quantile(x, 0.99, axis=0).tolist(),
            "count": [int(len(x))]}


def _write_parquet(path: Path, columns: dict[str, np.ndarray]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrays = {}
    for k, v in columns.items():
        if v.ndim == 2:
            arrays[k] = pa.FixedSizeListArray.from_arrays(pa.array(v.astype(np.float32).ravel()), v.shape[1])
        else:
            arrays[k] = pa.array(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path)


def export(raw: list[Path], out: Path, only_success: bool = True, chunk_size: int = 1000,
           overwrite: bool = False) -> dict:
    """Write the dataset; returns its summary (episodes, frames, skipped)."""
    out = Path(out)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (pass overwrite=True / --overwrite)")
        shutil.rmtree(out)
    files = sorted(p for d in raw for p in Path(d).glob("episode_*.npz") if not p.name.endswith(".part.npz"))
    tasks: dict[str, int] = {}
    episodes, ep_stats = [], []
    all_state, all_action, scalars = [], [], {k: [] for k in SCALARS}
    hist = np.zeros((3, 256), np.int64)
    fps = shape = None
    first_meta = None
    tables: dict[str, dict] = {}  # the hand's MIT table, per task (the drill's trigger finger runs stiffer)
    tools: dict[str, dict] = {}  # the tool and the wrap holding it, per task
    n_skipped = 0
    index = 0
    for f in files:
        demo = load_demo(f)
        meta = demo["meta"]
        if only_success and not meta["success"]:
            n_skipped += 1
            continue
        fps = fps or meta["fps"]
        if abs(meta["fps"] - fps) > 1e-9:
            raise ValueError(f"{f}: {meta['fps']} fps, the dataset is {fps}")
        cfg_v = meta["config"]["vla"]
        shape = shape or (cfg_v["cam_height"], cfg_v["cam_width"], 3)
        first_meta = first_meta or meta
        state, action = demo["state"], demo["action"]
        T = len(state)
        if state.shape != (T, STATE_DIM) or action.shape != (T, ACTION_DIM) or T == 0:
            raise ValueError(f"{f}: state {state.shape}, action {action.shape}")
        e = len(episodes)
        task = meta["instruction"]
        ti = tasks.setdefault(task, len(tasks))
        if tables.setdefault(task, meta["hand_table"]) != meta["hand_table"]:
            raise ValueError(f"{f}: its hand table differs from the rest of {task!r}")
        tool = tool_of(meta["config"])
        if tools.setdefault(task, tool) != tool:
            raise ValueError(f"{f}: its tool ({tool}) differs from the rest of {task!r} ({tools[task]})")
        chunk = e // chunk_size
        cols = {"observation.state": state, "action": action,
                "timestamp": (np.arange(T) / fps).astype(np.float32), "frame_index": np.arange(T, dtype=np.int64),
                "episode_index": np.full(T, e, np.int64), "index": np.arange(index, index + T, dtype=np.int64),
                "task_index": np.full(T, ti, np.int64), "annotation.human.task_description": np.full(T, ti, np.int64)}
        _write_parquet(out / DATA_PATH.format(episode_chunk=chunk, episode_index=e), cols)
        video = out / VIDEO_PATH.format(episode_chunk=chunk, video_key=VIDEO_KEY, episode_index=e)
        video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f.with_suffix(".mp4"), video)
        episodes.append({"episode_index": e, "tasks": [task], "length": T, "success": bool(meta["success"]),
                         "source": f"{f.parent.name}/{f.stem}"})
        st = {"observation.state": vector_stats(state), "action": vector_stats(action),
              VIDEO_KEY: ImageStats.summarize(demo["image_hist"])}
        st.update({k: vector_stats(cols[k][:, None].astype(np.float64)) for k in SCALARS})
        ep_stats.append({"episode_index": e, "stats": st})
        all_state.append(state)
        all_action.append(action)
        for k in SCALARS:
            scalars[k].append(cols[k])
        hist += demo["image_hist"]
        index += T
    if not episodes:
        raise ValueError(f"no {'successful ' if only_success else ''}episodes in {[str(d) for d in raw]}")
    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    n_ep, n_frames = len(episodes), index
    vec = {"dtype": "float32", "shape": [STATE_DIM], "names": list(ARM_NAMES) + list(first_meta["hand_joints"])}
    info = {
        "codebase_version": CODEBASE_VERSION, "robot_type": ROBOT_TYPE, "total_episodes": n_ep,
        "total_frames": n_frames, "total_tasks": len(tasks), "total_videos": n_ep,
        "total_chunks": (n_ep - 1) // chunk_size + 1, "chunks_size": chunk_size, "fps": float(fps),
        "splits": {"train": f"0:{n_ep}"}, "data_path": DATA_PATH, "video_path": VIDEO_PATH,
        "features": {
            VIDEO_KEY: {"dtype": "video", "shape": list(shape), "names": ["height", "width", "channel"],
                        "info": {"video.height": shape[0], "video.width": shape[1], "video.codec": "h264",
                                 "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": float(fps),
                                 "video.channels": 3, "has_audio": False}},
            "observation.state": vec, "action": dict(vec),
            **{k: {"dtype": "float32" if k == "timestamp" else "int64", "shape": [1], "names": None}
               for k in SCALARS},
        },
    }
    stats = {"observation.state": vector_stats(np.concatenate(all_state)),
             "action": vector_stats(np.concatenate(all_action)), VIDEO_KEY: ImageStats.summarize(hist)}
    stats.update({k: vector_stats(np.concatenate(scalars[k])[:, None].astype(np.float64)) for k in SCALARS})
    fm = first_meta
    robot = {"state": "right arm joint angles (7, rad) then WUJI Hand 2 joint angles (20, rad), as the host reads",
             "action": "right arm joint position targets (7, rad, before the host's gravity-droop compensation) then "
                       "WUJI MIT position targets q* (20, rad) under the task's hand table below",
             "hand_tables": tables, "tools": tools, "head_pose": fm["head_pose"], "camera": fm["config"]["vla"],
             "arm_command_rate": fm["config"]["arm"]["command_rate"]}
    _dump(meta_dir / "info.json", info)
    _dump(meta_dir / "stats.json", stats)
    _dump(meta_dir / "modality.json", modality())
    _dump(meta_dir / "vega_wuji.json", robot)
    _jsonl(meta_dir / "episodes.jsonl", episodes)
    _jsonl(meta_dir / "tasks.jsonl", [{"task_index": i, "task": t} for t, i in tasks.items()])
    _jsonl(meta_dir / "episodes_stats.jsonl", ep_stats)
    return {"episodes": n_ep, "frames": n_frames, "skipped": n_skipped, "tasks": list(tasks)}


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", nargs="+", help="directories of recorded demos (vla.demos --out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="export failed episodes too (flagged in episodes.jsonl)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    s = export([Path(r) for r in a.raw], Path(a.out), only_success=not a.all, overwrite=a.overwrite)
    print(f"{a.out}: {s['episodes']} episodes, {s['frames']} frames ({s['skipped']} failed episodes skipped)")


if __name__ == "__main__":
    main()
