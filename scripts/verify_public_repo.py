
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

BLOCKED_SUFFIXES = (
    ".dcm",
    ".nii",
    ".nii.gz",
    ".mha",
    ".mhd",
    ".nrrd",
    ".raw",
    ".pt",
    ".pth",
    ".ckpt",
)
SENSITIVE_STRINGS = (
    "Patient" + "Name",
    "Patient" + "Birth" + "Date",
    "Accession" + "Number",
    "Medical" + "Record" + "Number",
)
SECRET_PATTERNS = (
    re.compile(r"(?i)\bTOKEN\s*="),
    re.compile(r"(?i)\bAPI_KEY\s*="),
    re.compile(r"(?i)\bPASSWORD\s*="),
)
MACHINE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"\b" + "Users" + r"\\"),
    re.compile(r"\b" + "D" + "YL" + r"\b"),
    re.compile("Codex" + "Sandbox" + "Offline"),
)
EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "data",
    "datasets",
    "raw",
    "images",
    "predictions",
    "checkpoints",
    "weights",
    "logs",
}


def git_tracked_files(repo_root: Path) -> list[Path] | None:
    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    files = [repo_root / line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return files or None


def filesystem_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_root.rglob("*"):
        rel_parts = path.relative_to(repo_root).parts
        if any(part in EXCLUDED_DIRS for part in rel_parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def candidate_files(repo_root: Path) -> list[Path]:
    return git_tracked_files(repo_root) or filesystem_files(repo_root)


def is_blocked_artifact(path: Path) -> bool:
    lowered = path.name.lower()
    return any(lowered.endswith(suffix) for suffix in BLOCKED_SUFFIXES)


def scan_text(path: Path, repo_root: Path) -> list[str]:
    findings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return [f"could not read text: {exc}"]
    rel = path.relative_to(repo_root).as_posix()
    scanner_file = rel == "scripts/verify_public_repo.py"
    if not scanner_file:
        for marker in SENSITIVE_STRINGS:
            if marker in text:
                findings.append(f"sensitive metadata marker found: {marker}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"probable secret pattern found: {pattern.pattern}")
        for pattern in MACHINE_PATH_PATTERNS:
            if pattern.search(text):
                findings.append(f"machine-specific path marker found: {pattern.pattern}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Guardrail scan for public repository content. This is not a formal PHI detector."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    errors: list[str] = []
    for path in candidate_files(repo_root):
        if not path.exists():
            continue
        rel = path.relative_to(repo_root)
        if is_blocked_artifact(path):
            errors.append(f"{rel}: blocked medical/model artifact extension")
            continue
        if path.is_file():
            for finding in scan_text(path, repo_root):
                errors.append(f"{rel}: {finding}")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        print("FAIL: public repository guardrail found blocked content", file=sys.stderr)
        return 1
    print("PASS: public repository guardrail found no blocked content")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
