from __future__ import annotations

import torch

from pet_ai.datasets.synthetic_petct import SyntheticPETCTDataset


def test_synthetic_petct_dataset_returns_expected_shapes_and_dtypes() -> None:
    dataset = SyntheticPETCTDataset(num_samples=2, spatial_shape=(16, 16, 16), seed=123)

    sample = dataset[0]

    assert sample["image"].shape == (2, 16, 16, 16)
    assert sample["label"].shape == (1, 16, 16, 16)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].dtype == torch.float32
    assert sample["sample_id"] == "synthetic_petct_0000"
    assert set(torch.unique(sample["label"]).tolist()).issubset({0.0, 1.0})


def test_synthetic_petct_dataset_same_seed_is_deterministic() -> None:
    first = SyntheticPETCTDataset(num_samples=1, spatial_shape=(16, 16, 16), seed=42)[0]
    second = SyntheticPETCTDataset(num_samples=1, spatial_shape=(16, 16, 16), seed=42)[0]

    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["label"], second["label"])
    assert first["sample_id"] == second["sample_id"]
