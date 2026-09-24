from __future__ import annotations

from pet_ai.reproducibility.hashing import sha256_file
from pet_ai.reproducibility.run_manifest import create_run_manifest


def test_sha256_deterministic(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 7\n", encoding="utf-8")

    assert sha256_file(path) == sha256_file(path)


def test_run_manifest_contains_required_provenance_fields(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 7\n", encoding="utf-8")

    manifest = create_run_manifest(
        repo_root=tmp_path,
        dataset_manifest=None,
        split_manifest=None,
        config=config,
        checkpoint=None,
        seed=7,
        command=["pytest", "-q"],
        exit_status=0,
        gpu_name=None,
        run_id="test-run",
    )
    data = manifest.to_json_dict()

    for field in (
        "run_id",
        "timestamp",
        "git_commit",
        "git_dirty",
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "config_sha256",
        "python_version",
        "platform",
        "gpu_name",
        "seed",
        "command",
        "exit_status",
        "checkpoint_sha256",
        "dependency_versions",
        "notes",
    ):
        assert field in data
    assert data["run_id"] == "test-run"
    assert data["config_sha256"] == sha256_file(config)
    assert data["dataset_manifest_sha256"] is None
    assert data["git_dirty"] is None
    assert {"numpy", "nibabel", "PyYAML"} <= data["dependency_versions"].keys()
