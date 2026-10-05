"""Fetch the pinned MuJoCo Menagerie files (Franka FR3 arm + Franka hand) into a local cache.

Files are downloaded one at a time from raw.githubusercontent.com at a pinned commit and
verified against MANIFEST.json. Tarball and GitHub API endpoints are not used because
they are often blocked behind egress proxies.

Usage:
    python -m tactile_sim.assets.fetch_menagerie                 # fetch into the cache
    python -m tactile_sim.assets.fetch_menagerie --dest DIR      # fetch elsewhere
    python -m tactile_sim.assets.fetch_menagerie --only LICENSE  # substring filter
    python -m tactile_sim.assets.fetch_menagerie --regen-manifest --commit SHA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

from tactile_sim.assets import MANIFEST_PATH, cache_dir, load_manifest

RAW_BASE = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
DEFAULT_REPO = "google-deepmind/mujoco_menagerie"
MODEL_DIRS = {"franka_fr3": "fr3.xml", "franka_emika_panda": "hand.xml"}
BACKOFF_S = (2, 4, 8, 16)


def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if cafile and Path(cafile).exists():
        return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def download(url: str, timeout: float = 60.0) -> bytes:
    """GET a URL with retries and exponential backoff."""
    ctx = _ssl_context()
    last: Exception | None = None
    for attempt in range(len(BACKOFF_S) + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - network errors are varied
            last = e
            if attempt < len(BACKOFF_S):
                time.sleep(BACKOFF_S[attempt])
    raise RuntimeError(f"download failed after retries: {url}: {last}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def referenced_files(xml_text: str) -> list[str]:
    """Return every file="..." attribute in an MJCF document (mesh/texture files)."""
    return sorted(set(re.findall(r'file="([^"]+)"', xml_text)))


def regen_manifest(repo: str, commit: str, dest: Path) -> dict:
    """Walk the model XMLs at `commit`, download everything they reference, and hash it."""
    files = []
    for model_dir, xml_name in MODEL_DIRS.items():
        xml_path = f"{model_dir}/{xml_name}"
        xml_bytes = download(RAW_BASE.format(repo=repo, commit=commit, path=xml_path))
        entries = [xml_path, f"{model_dir}/LICENSE"]
        entries += [f"{model_dir}/assets/{f}" for f in referenced_files(xml_bytes.decode())]
        for p in entries:
            data = xml_bytes if p == xml_path else download(RAW_BASE.format(repo=repo, commit=commit, path=p))
            _write_atomic(dest / p, data)
            files.append({"path": p, "sha256": sha256(data), "size": len(data)})
            print(f"  {p}  {len(data)} B")
    manifest = {
        "repo": repo,
        "commit": commit,
        "license": "Apache-2.0 (see each model's LICENSE)",
        "files": files,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def fetch(dest: Path | None = None, only: str | None = None, verbose: bool = True) -> tuple[int, int]:
    """Fetch manifest files into `dest`. Returns (n_downloaded, n_already_ok). Raises on mismatch."""
    manifest = load_manifest()
    dest = Path(dest) if dest else cache_dir()
    n_new = n_ok = 0
    bad = []
    for entry in manifest["files"]:
        if only and only not in entry["path"]:
            continue
        target = dest / entry["path"]
        if target.exists() and sha256(target.read_bytes()) == entry["sha256"]:
            n_ok += 1
            continue
        url = RAW_BASE.format(repo=manifest["repo"], commit=manifest["commit"], path=entry["path"])
        data = download(url)
        if sha256(data) != entry["sha256"]:
            bad.append(entry["path"])
            continue
        _write_atomic(target, data)
        n_new += 1
        if verbose:
            print(f"  fetched {entry['path']} ({len(data)} B)")
    if bad:
        raise RuntimeError(f"sha256 mismatch for: {bad}")
    return n_new, n_ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=None, help="target dir (default: the asset cache)")
    ap.add_argument("--only", default=None, help="only fetch manifest paths containing this substring")
    ap.add_argument("--regen-manifest", action="store_true", help="rebuild MANIFEST.json at --commit")
    ap.add_argument("--commit", default=None, help="Menagerie commit sha (regen only)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    args = ap.parse_args(argv)

    if args.regen_manifest:
        if not args.commit:
            ap.error("--regen-manifest needs --commit")
        dest = args.dest or cache_dir(args.commit)
        m = regen_manifest(args.repo, args.commit, dest)
        print(f"wrote {MANIFEST_PATH} with {len(m['files'])} files; cached in {dest}")
        return 0

    dest = args.dest or cache_dir()
    try:
        n_new, n_ok = fetch(dest, args.only)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"menagerie assets in {dest}: {n_new} downloaded, {n_ok} already verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
