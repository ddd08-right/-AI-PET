"""Deterministic synthetic PET/CT tensors for software tests.

The generated volumes are not physiologically realistic, are not clinical
simulation, and must not be used for scientific model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
from torch.utils.data import Dataset


class SyntheticPETCTSample(TypedDict):
    image: torch.Tensor
    label: torch.Tensor
    sample_id: str


@dataclass(frozen=True)
class SyntheticPETCTConfig:
    num_samples: int = 4
    spatial_shape: tuple[int, int, int] = (32, 32, 32)
    seed: int = 20260916


class SyntheticPETCTDataset(Dataset[SyntheticPETCTSample]):
    """Create deterministic two-channel PET/CT-like tensors and binary labels."""

    def __init__(
        self,
        *,
        num_samples: int = 4,
        spatial_shape: tuple[int, int, int] = (32, 32, 32),
        seed: int = 20260916,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if len(spatial_shape) != 3 or any(size < 8 for size in spatial_shape):
            raise ValueError("spatial_shape must contain three dimensions of at least 8 voxels")
        self.config = SyntheticPETCTConfig(num_samples=num_samples, spatial_shape=spatial_shape, seed=seed)

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> SyntheticPETCTSample:
        if index < 0 or index >= len(self):
            raise IndexError(f"sample index out of range: {index}")

        generator = torch.Generator().manual_seed(self.config.seed + index)
        depth, height, width = self.config.spatial_shape
        label = torch.zeros((1, depth, height, width), dtype=torch.float32)

        lesion_size = max(3, min(self.config.spatial_shape) // 8)
        max_starts = [size - lesion_size - 1 for size in self.config.spatial_shape]
        starts = [
            int(torch.randint(1, max_start + 1, (1,), generator=generator).item())
            for max_start in max_starts
        ]
        z0, y0, x0 = starts
        label[:, z0 : z0 + lesion_size, y0 : y0 + lesion_size, x0 : x0 + lesion_size] = 1.0

        z_axis = torch.linspace(-1.0, 1.0, depth, dtype=torch.float32).view(depth, 1, 1)
        y_axis = torch.linspace(-1.0, 1.0, height, dtype=torch.float32).view(1, height, 1)
        x_axis = torch.linspace(-1.0, 1.0, width, dtype=torch.float32).view(1, 1, width)
        ct_background = 0.30 * z_axis + 0.15 * y_axis - 0.10 * x_axis
        ct_noise = 0.03 * torch.randn((depth, height, width), generator=generator, dtype=torch.float32)
        ct = ct_background + ct_noise

        pet_noise = 0.05 * torch.randn((depth, height, width), generator=generator, dtype=torch.float32)
        pet = 0.20 + pet_noise + 1.50 * label[0]
        image = torch.stack([pet, ct], dim=0).to(dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "sample_id": f"synthetic_petct_{index:04d}",
        }
