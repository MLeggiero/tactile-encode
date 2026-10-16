"""The VLA's view: the Vega U head camera (VlaCfg), aimed at the task, with per-episode visual randomization.

aim_head(cfg) solves the head joints that put the tool's working point, raised `rise` back along the working
direction so the motion above it stays in view, at the image centre (the robot alone, at the configured lift and
flip). vla_config(cfg) puts that pose, or VlaCfg.head_pose, into ArmCfg.head_pose, which the testbed holds.

EgoCamera renders the camera offscreen and draws VisualDR per episode: the camera's mount pose, the lights'
intensity and direction, and the colors of the tool, the workpiece and its table, and the floor. Rendering
needs MUJOCO_GL=egl (or osmesa) set before mujoco is first imported.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from tactile_sim import names
from tactile_sim.config import SimConfig

HEAD_JOINTS = ("head_j1", "head_j2", "head_j3")


def project(m, d, cam: int, p: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel (u right, v down) of the world point p in camera `cam`, and its depth along the view axis."""
    R = d.cam_xmat[cam].reshape(3, 3)
    v = R.T @ (np.asarray(p, float) - d.cam_xpos[cam])  # camera frame: x right, y up, looking along -z
    depth = -v[2]
    f = 0.5 * height / np.tan(np.deg2rad(m.cam_fovy[cam]) / 2)
    if depth <= 1e-6:
        return np.array([np.nan, np.nan, depth])
    return np.array([width / 2 + f * v[0] / depth, height / 2 - f * v[1] / depth, depth])


def look_point(cfg: SimConfig, rise: float = 0.10) -> np.ndarray:
    from tactile_sim.model.builder import tool_target

    p, axis = tool_target(cfg)
    return np.asarray(p, float) - rise * np.asarray(axis, float)


def aim_head(cfg: SimConfig, rise: float = 0.10) -> tuple[float, float, float]:
    """Head joints (pitch head_j1, yaw head_j2; head_j3 level) that put look_point(cfg, rise) in the ego image's
    middle column, VlaCfg.aim_v of the way down."""
    return _aim_head(cfg.to_json(), rise)


@lru_cache(maxsize=16)
def _aim_head(cfg_json: str, rise: float) -> tuple[float, float, float]:
    import json
    import xml.etree.ElementTree as ET

    import mujoco

    from tactile_sim.model.builder import add_ego_camera
    from tactile_sim.model.robots import load_vega_tree

    cfg = SimConfig.from_dict(json.loads(cfg_json))
    if not cfg.arm.robot.startswith("vega"):
        raise ValueError("the ego camera is on the Vega's head")
    root, _ = load_vega_tree(cfg.arm)
    add_ego_camera(root, cfg)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, names.EGO_CAMERA)
    adr = [m.joint(j).qposadr[0] for j in HEAD_JOINTS]
    lo = np.array([m.joint(j).range[0] for j in HEAD_JOINTS[:2]])
    hi = np.array([m.joint(j).range[1] for j in HEAD_JOINTS[:2]])
    target = look_point(cfg, rise)
    w, h = cfg.vla.cam_width, cfg.vla.cam_height
    centre = np.array([w / 2, cfg.vla.aim_v * h])

    def err(x):
        d.qpos[adr] = [x[0], x[1], 0.0]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        return project(m, d, cam, target, w, h)[:2] - centre

    x = np.array([0.5, 0.0])
    for _ in range(30):
        e = err(x)
        if not np.all(np.isfinite(e)):
            raise ValueError(f"the task point {target} is behind the head camera")
        if np.abs(e).max() < 0.05:
            break
        J = np.stack([(err(x + dx) - e) / 1e-4 for dx in np.eye(2) * 1e-4], axis=1)
        x = np.clip(x - np.linalg.lstsq(J, e, rcond=None)[0], lo, hi)
    if np.abs(err(x)).max() > 1.0:
        raise ValueError(f"the head cannot centre {target} in its camera (joint limits)")
    return (round(float(x[0]), 4), round(float(x[1]), 4), 0.0)


def vla_config(cfg: SimConfig) -> SimConfig:
    """cfg with the head where the VLA looks: VlaCfg.head_pose, else aimed at the task."""
    pose = cfg.vla.head_pose if cfg.vla.head_pose is not None else aim_head(cfg)
    return cfg.replace(arm={"head_pose": tuple(float(v) for v in pose)})


def _small_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation by independent uniform angles about x, y and z."""
    import mujoco

    q = np.array([1.0, 0.0, 0.0, 0.0])
    for axis in np.eye(3):
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, axis, np.deg2rad(rng.uniform(-max_deg, max_deg)))
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, q, dq)
        q = out
    return q


class EgoCamera:
    def __init__(self, model, cfg: SimConfig):
        import mujoco

        self.m = model
        self.v = cfg.vla
        self.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, names.EGO_CAMERA)
        if self.cam < 0:
            raise ValueError("the model has no ego camera (the Vega robots have one)")
        self.renderer = mujoco.Renderer(model, self.v.cam_height, self.v.cam_width)
        m = model
        vis = m.vis.headlight
        self._nominal = {
            "cam_pos": m.cam_pos[self.cam].copy(), "cam_quat": m.cam_quat[self.cam].copy(),
            "light_diffuse": m.light_diffuse.copy(), "light_specular": m.light_specular.copy(),
            "light_ambient": m.light_ambient.copy(), "light_dir": m.light_dir.copy(),
            "head": (vis.ambient.copy(), vis.diffuse.copy(), vis.specular.copy()),
            "geom_rgba": m.geom_rgba.copy(),
        }
        self.groups = self._color_groups()
        self.draw: dict | None = None

    def _color_groups(self) -> list[np.ndarray]:
        """Geoms recolored together: each body of the tool and of the workpiece (board, nail, screw, table), and
        the floor. The robot keeps its color."""
        import mujoco

        m = self.m
        robot_root = int(m.body_rootid[m.body(names.EGO_CAMERA_BODY).id])
        robot = {b for b in range(m.nbody) if int(m.body_rootid[b]) == robot_root}
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        groups: dict[tuple, list[int]] = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b in robot:
                continue
            groups.setdefault(("geom", g) if b == 0 else ("body", b), []).append(g)  # world geoms: floor, table
        assert floor < 0 or ("geom", floor) in groups
        return [np.array(v) for v in groups.values()]

    def nominal(self) -> None:
        m, n = self.m, self._nominal
        m.cam_pos[self.cam], m.cam_quat[self.cam] = n["cam_pos"], n["cam_quat"]
        m.light_diffuse[:], m.light_specular[:] = n["light_diffuse"], n["light_specular"]
        m.light_ambient[:], m.light_dir[:] = n["light_ambient"], n["light_dir"]
        for field, value in zip(("ambient", "diffuse", "specular"), n["head"], strict=True):
            getattr(m.vis.headlight, field)[:] = value
        m.geom_rgba[:] = n["geom_rgba"]
        self.draw = None

    def randomize(self, rng: np.random.Generator) -> dict:
        """A new VisualDR draw (VlaCfg) on top of the nominal scene; returns what was drawn."""
        import mujoco

        self.nominal()
        v, m, n = self.v, self.m, self._nominal
        dpos = rng.uniform(-v.cam_pos_jitter, v.cam_pos_jitter, 3)
        dq = _small_rotation(rng, v.cam_rot_jitter)
        m.cam_pos[self.cam] = n["cam_pos"] + dpos
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, n["cam_quat"], dq)
        m.cam_quat[self.cam] = q
        light = rng.uniform(1 - v.light_jitter, 1 + v.light_jitter, m.nlight)
        m.light_diffuse[:] = np.clip(n["light_diffuse"] * light[:, None], 0, 1)
        m.light_specular[:] = np.clip(n["light_specular"] * light[:, None], 0, 1)
        for k in range(m.nlight):
            tilt = _small_rotation(rng, 20.0)
            out = np.zeros(3)
            mujoco.mju_rotVecQuat(out, n["light_dir"][k], tilt)
            m.light_dir[k] = out / np.linalg.norm(out)
        head = rng.uniform(1 - v.light_jitter, 1 + v.light_jitter)
        amb, dif, _ = n["head"]
        m.vis.headlight.ambient[:] = np.clip(amb * head, 0, 1)
        m.vis.headlight.diffuse[:] = np.clip(dif * head, 0, 1)
        colors = []
        for geoms in self.groups:
            dc = rng.uniform(-v.color_jitter, v.color_jitter, 3)
            m.geom_rgba[geoms, :3] = np.clip(n["geom_rgba"][geoms, :3] + dc, 0, 1)
            colors.append(dc.tolist())
        self.draw = {"cam_dpos": dpos.tolist(), "cam_dquat": dq.tolist(), "light_scale": light.tolist(),
                     "headlight_scale": float(head), "color_offsets": colors}
        return self.draw

    def render(self, data) -> np.ndarray:
        """(H, W, 3) uint8 from the camera at data's current state."""
        self.renderer.update_scene(data, camera=self.cam)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()
