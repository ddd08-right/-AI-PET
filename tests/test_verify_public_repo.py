from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def _scanner_module():
    script = Path(__file__).parents[1] / "scripts" / "verify_public_repo.py"
    spec = importlib.util.spec_from_file_location("verify_public_repo", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def test_untracked_blocked_artifact_is_a_scan_candidate(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    safe_file = tmp_path / "README.md"
    safe_file.write_text("synthetic fixture\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    blocked = tmp_path / "SYNTHETIC_FORBIDDEN.nii.gz"
    blocked.write_bytes(b"synthetic test bytes")

    scanner = _scanner_module()
    candidates = scanner.candidate_files(tmp_path)

    assert blocked in candidates
    assert scanner.is_blocked_artifact(blocked)


def test_explicit_ignored_delivery_file_is_scanned(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text("*.ckpt\n", encoding="utf-8")
    ignored = tmp_path / "SYNTHETIC_FORBIDDEN.ckpt"
    ignored.write_bytes(b"synthetic test bytes")

    scanner = _scanner_module()
    candidates = scanner.candidate_files(tmp_path, [Path(ignored.name)])

    assert ignored in candidates
    assert scanner.is_blocked_artifact(ignored)
