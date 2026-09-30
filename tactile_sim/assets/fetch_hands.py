"""Fetch pinned dexterous-hand models (WUJI Hand 2) into a local cache.

Same scheme as fetch_menagerie: files come one at a time from raw.githubusercontent.com at a pinned commit
and are verified against HANDS_MANIFEST.json. Only the MJCF used here, the meshes it references and the
LICENSE are fetched.

Usage:
    python -m tactile_sim.assets.fetch_hands                      # fetch into the cache
    python -m tactile_sim.assets.fetch_hands --regen-manifest     # rebuild the manifest at the pinned commits
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tactile_sim.assets.fetch_menagerie import RAW_BASE, _write_atomic, download, referenced_files, sha256

HANDS_MANIFEST_PATH = Path(__file__).resolve().parent / "HANDS_MANIFEST.json"
ENV_VAR = "TACTILE_SIM_HANDS"

# hand key -> (repo, commit, MJCF path, license path)
HAND_SOURCES = {
    "wuji2": ("wuji-technology/wuji-description", "c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95",
              "hand2/hand2_beta2/body/mjcf/right_with_mount.xml", "LICENSE"),
}


def load_hands_manifest() -> dict:
    return json.loads(HANDS_MANIFEST_PATH.read_text())


def hand_cache_dir(hand: str) -> Path:
    """~/.cache/tactile_sim/hands/<hand>/<commit>, or $TACTILE_SIM_HANDS/<hand>."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env) / hand
    commit = load_hands_manifest()[hand]["commit"]
    return Path.home() / ".cache" / "tactile_sim" / "hands" / hand / commit


def hand_available(hand: str, check_hashes: bool = True) -> bool:
    entry = load_hands_manifest().get(hand)
    if entry is None:
        return False
    root = hand_cache_dir(hand)
    for f in entry["files"]:
        p = root / f["path"]
        if not p.is_file() or (check_hashes and sha256(p.read_bytes()) != f["sha256"]):
            return False
    return True


def hand_xml(hand: str) -> Path | None:
    """Path of the cached MJCF, or None when the hand has not been fetched."""
    if not hand_available(hand):
        return None
    return hand_cache_dir(hand) / load_hands_manifest()[hand]["mjcf"]


def regen_manifest(hands: list[str]) -> dict:
    manifest = load_hands_manifest() if HANDS_MANIFEST_PATH.exists() else {}
    for hand in hands:
        repo, commit, mjcf, lic = HAND_SOURCES[hand]
        xml_bytes = download(RAW_BASE.format(repo=repo, commit=commit, path=mjcf))
        text = xml_bytes.decode()
        meshdir = ""
        if 'meshdir="' in text:
            meshdir = text.split('meshdir="', 1)[1].split('"', 1)[0]
        base = Path(mjcf).parent
        paths = [mjcf, lic] + [os.path.normpath(str(base / meshdir / f)) for f in referenced_files(text)]
        files = []
        dest = Path.home() / ".cache" / "tactile_sim" / "hands" / hand / commit
        for p in paths:
            data = xml_bytes if p == mjcf else download(RAW_BASE.format(repo=repo, commit=commit, path=p))
            _write_atomic(dest / p, data)
            files.append({"path": p, "sha256": sha256(data), "size": len(data)})
        manifest[hand] = {"repo": repo, "commit": commit, "mjcf": mjcf, "files": files}
        print(f"{hand}: {len(files)} files, {sum(f['size'] for f in files) / 1e6:.1f} MB")
    HANDS_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def fetch(hand: str, verbose: bool = True) -> tuple[int, int]:
    entry = load_hands_manifest()[hand]
    root = hand_cache_dir(hand)
    n_new = n_ok = 0
    bad = []
    for f in entry["files"]:
        target = root / f["path"]
        if target.exists() and sha256(target.read_bytes()) == f["sha256"]:
            n_ok += 1
            continue
        data = download(RAW_BASE.format(repo=entry["repo"], commit=entry["commit"], path=f["path"]))
        if sha256(data) != f["sha256"]:
            bad.append(f["path"])
            continue
        _write_atomic(target, data)
        n_new += 1
        if verbose:
            print(f"  fetched {f['path']} ({len(data)} B)")
    if bad:
        raise RuntimeError(f"sha256 mismatch for: {bad}")
    return n_new, n_ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hands", nargs="*", default=list(HAND_SOURCES))
    ap.add_argument("--regen-manifest", action="store_true")
    args = ap.parse_args(argv)
    if args.regen_manifest:
        regen_manifest(args.hands)
        return 0
    for hand in args.hands:
        try:
            n_new, n_ok = fetch(hand)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"{hand} in {hand_cache_dir(hand)}: {n_new} downloaded, {n_ok} already verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
