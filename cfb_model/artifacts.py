"""Fingerprint model artifacts and preserve immutable prediction runs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from uuid import uuid4

import pandas as pd

from .config import BASE_DIR

SCHEMA_VERSION = 2


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity():
    files = sorted((BASE_DIR / "cfb_model").rglob("*.py"))
    files += sorted(BASE_DIR.glob("requirements*.txt"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(BASE_DIR)).encode())
        digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "cfb_model", "requirements.txt", "requirements-dev.txt"],
            cwd=BASE_DIR, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"git_commit": commit, "git_source_dirty": dirty, "source_sha256": digest.hexdigest()}


def manifest_path(path):
    return Path(str(path) + ".manifest.json")


def write_manifest(path, inputs=(), **metadata):
    path = Path(path)
    versions = {}
    for package in ("pandas", "numpy", "scipy", "scikit-learn", "lightgbm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        **code_identity(),
        "artifact_sha256": sha256(path),
        "inputs": {str(Path(p).resolve()): sha256(p) for p in inputs},
        "packages": versions,
        **metadata,
    }
    manifest_path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload


def validate_artifact(path, inputs=()):
    path = Path(path)
    try:
        saved = json.loads(manifest_path(path).read_text())
        valid = (saved["schema_version"] == SCHEMA_VERSION
                 and saved["source_sha256"] == code_identity()["source_sha256"]
                 and saved["artifact_sha256"] == sha256(path)
                 and saved["inputs"] == {
                     str(Path(p).resolve()): sha256(p) for p in inputs})
    except (OSError, KeyError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"Stale or unversioned artifact {path.name}; rebuild features/backtest")
    return saved


def new_run_dir(root):
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def snapshot_inputs(inputs):
    return {str(Path(p).resolve()): sha256(p) for p in inputs}


def require_unchanged(inputs, identity):
    if (snapshot_inputs(inputs) != identity["inputs"] or
            code_identity()["source_sha256"] != identity["source_sha256"]):
        raise ValueError("Inputs or model source changed during this run; refusing to label stale results")
