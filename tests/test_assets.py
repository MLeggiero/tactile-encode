import json
import re
import shutil

import mujoco
import numpy as np
import pytest
from conftest import requires_menagerie

from tactile_sim import names
from tactile_sim.assets import (
    FALLBACK_ARM_XML,
    MANIFEST_PATH,
    cache_dir,
    ensure_menagerie,
    fr3_xml_path,
    hand_xml_path,
    load_arm_model,
    load_manifest,
    menagerie_available,
)
from tactile_sim.assets.fetch_menagerie import fetch, referenced_files

# attachment_site of the Menagerie FR3 at q = 0 (computed once from fr3.xml at the pinned commit)
FR3_ATTACH_Q0 = np.array([0.088, 0.0, 0.926])


def test_manifest_schema():
    m = json.loads(MANIFEST_PATH.read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", m["commit"])
    paths = {f["path"] for f in m["files"]}
    for p in ("franka_fr3/fr3.xml", "franka_fr3/LICENSE",
              "franka_emika_panda/hand.xml", "franka_emika_panda/LICENSE"):
        assert p in paths
    for f in m["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] > 0


@requires_menagerie
def test_manifest_covers_every_referenced_file():
    paths = {f["path"] for f in load_manifest()["files"]}
    for xml, d in ((fr3_xml_path(), "franka_fr3"), (hand_xml_path(), "franka_emika_panda")):
        for f in referenced_files(xml.read_text()):
            assert f"{d}/assets/{f}" in paths


def _fallback():
    return mujoco.MjModel.from_xml_path(str(FALLBACK_ARM_XML))


def test_fallback_arm_structure():
    m = _fallback()
    assert m.nq == 7 and m.nu == 0
    assert [m.joint(i).name for i in range(m.njnt)] == names.ARM_JOINTS
    assert all(m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE for i in range(m.njnt))
    assert m.site(names.ATTACHMENT_SITE).id >= 0
    assert np.all(m.geom_contype == 0) and np.all(m.geom_conaffinity == 0)


def test_fallback_kinematics_match_fr3():
    m = _fallback()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert np.linalg.norm(d.site(names.ATTACHMENT_SITE).xpos - FR3_ATTACH_Q0) < 0.02


@requires_menagerie
def test_fallback_dynamics_match_fr3_live():
    a, b = _fallback(), mujoco.MjModel.from_xml_path(str(fr3_xml_path()))
    for q in (np.zeros(7), b.key_qpos[0], np.array([0.3, -0.5, 0.2, -2.0, 0.4, 1.8, 0.1])):
        da, db = mujoco.MjData(a), mujoco.MjData(b)
        da.qpos[:] = db.qpos[:] = q
        mujoco.mj_forward(a, da)
        mujoco.mj_forward(b, db)
        assert np.allclose(da.site(names.ATTACHMENT_SITE).xpos, db.site(names.ATTACHMENT_SITE).xpos, atol=1e-9)
        assert np.allclose(da.qfrc_bias, db.qfrc_bias, atol=1e-9)


def test_fallback_steps_finite():
    m = _fallback()
    m.opt.timestep = 0.00025
    m.opt.gravity[:] = 0
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    for _ in range(2000):
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))


def test_offline_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("TACTILE_SIM_ASSETS", str(tmp_path / "empty"))
    assert cache_dir() == tmp_path / "empty"
    assert ensure_menagerie(download=False) is None
    with pytest.warns(RuntimeWarning, match="fallback"):
        m, src = load_arm_model("fr3")
    assert src == "fallback" and m.nq == 7


@requires_menagerie
def test_corrupted_cache_detected(tmp_path, monkeypatch):
    dst = tmp_path / "copy"
    shutil.copytree(cache_dir(), dst)
    assert menagerie_available(dst)
    f = dst / "franka_fr3" / "LICENSE"
    b = bytearray(f.read_bytes())
    b[0] ^= 0xFF
    f.write_bytes(bytes(b))
    assert not menagerie_available(dst)
    monkeypatch.setenv("TACTILE_SIM_ASSETS", str(dst))
    with pytest.warns(RuntimeWarning):
        assert load_arm_model("fr3")[1] == "fallback"


@requires_menagerie
def test_menagerie_models_load():
    arm, src = load_arm_model("fr3")
    assert src == "fr3" and arm.nq == 7
    hand = mujoco.MjModel.from_xml_path(str(hand_xml_path()))
    assert hand.nq == 2


@pytest.mark.network
def test_fetch_licenses(tmp_path):
    n_new, n_ok = fetch(tmp_path, only="LICENSE", verbose=False)
    assert n_new == 2 and n_ok == 0
    assert fetch(tmp_path, only="LICENSE", verbose=False) == (0, 2)
