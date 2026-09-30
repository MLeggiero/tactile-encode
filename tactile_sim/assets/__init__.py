"""Asset locations, the Menagerie cache, and arm-model loading with an offline fallback."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import mujoco

ASSET_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ASSET_ROOT / "MANIFEST.json"
FALLBACK_ARM_XML = ASSET_ROOT / "fallback_arm.xml"
ENV_VAR = "TACTILE_SIM_ASSETS"


def asset_root() -> Path:
    return ASSET_ROOT


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def cache_dir(commit: str | None = None) -> Path:
    """Directory holding the Menagerie checkout subset.

    `TACTILE_SIM_ASSETS` overrides it; otherwise ~/.cache/tactile_sim/menagerie/<commit>.
    """
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    commit = commit or load_manifest()["commit"]
    return Path.home() / ".cache" / "tactile_sim" / "menagerie" / commit


def menagerie_available(root: Path | None = None, check_hashes: bool = True) -> bool:
    """True if every manifest file exists under `root` (and matches its sha256)."""
    root = Path(root) if root else cache_dir()
    for entry in load_manifest()["files"]:
        p = root / entry["path"]
        if not p.is_file():
            return False
        if check_hashes and hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]:
            return False
    return True


def ensure_menagerie(download: bool = False) -> Path | None:
    """Return the cache dir if assets are present and verified, else optionally fetch them."""
    root = cache_dir()
    if menagerie_available(root):
        return root
    if not download:
        return None
    from tactile_sim.assets.fetch_menagerie import fetch

    fetch(root, verbose=False)
    return root if menagerie_available(root) else None


def fr3_xml_path(root: Path | None = None) -> Path:
    return (Path(root) if root else cache_dir()) / "franka_fr3" / "fr3.xml"


def hand_xml_path(root: Path | None = None) -> Path:
    return (Path(root) if root else cache_dir()) / "franka_emika_panda" / "hand.xml"


def load_arm_model(prefer: str = "fr3") -> tuple[mujoco.MjModel, str]:
    """Load the bare arm. Falls back to the capsule arm (with a warning) if Menagerie is absent."""
    if prefer not in ("fr3", "fallback"):
        raise ValueError(f"prefer must be 'fr3' or 'fallback', got {prefer!r}")
    if prefer == "fr3":
        root = ensure_menagerie(download=False)
        if root is not None:
            return mujoco.MjModel.from_xml_path(str(fr3_xml_path(root))), "fr3"
        warnings.warn(
            "Menagerie FR3 assets not found; using the capsule fallback arm. "
            "Run `python -m tactile_sim.assets.fetch_menagerie` to fetch them.",
            RuntimeWarning,
            stacklevel=2,
        )
    return mujoco.MjModel.from_xml_path(str(FALLBACK_ARM_XML)), "fallback"
