"""Structured experiment run manifests."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from pet_ai.reproducibility.hashing import sha256_or_not_available

NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    timestamp: str
    git_commit: str | None
    dataset_manifest_sha256: str | None
    split_manifest_sha256: str | None
    config_sha256: str | None
    python_version: str
    platform: str
    gpu_name: str | None
    seed: int | None
    command: list[str]
    exit_status: int | None
    checkpoint_sha256: str | None
    notes: str | None

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def current_git_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def detect_gpu_name() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return "; ".join(names) if names else None


def create_run_manifest(
    *,
    repo_root: Path,
    dataset_manifest: Path | None,
    split_manifest: Path | None,
    config: Path | None,
    checkpoint: Path | None,
    seed: int | None,
    command: list[str],
    exit_status: int | None,
    notes: str | None = None,
    run_id: str | None = None,
    gpu_name: str | None = None,
) -> RunManifest:
    return RunManifest(
        run_id=run_id or f"run-{uuid.uuid4().hex}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=current_git_commit(repo_root),
        dataset_manifest_sha256=sha256_or_not_available(dataset_manifest),
        split_manifest_sha256=sha256_or_not_available(split_manifest),
        config_sha256=sha256_or_not_available(config),
        python_version=sys.version,
        platform=platform.platform(),
        gpu_name=gpu_name if gpu_name is not None else detect_gpu_name(),
        seed=seed,
        command=command,
        exit_status=exit_status,
        checkpoint_sha256=sha256_or_not_available(checkpoint),
        notes=notes,
    )


def write_run_manifest(manifest: RunManifest, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
