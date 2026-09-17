from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.pytorch

from pet_ai.models.unet3d import SmallUNet3D  # noqa: E402


def test_small_unet3d_output_shape_matches_input_spatial_shape() -> None:
    model = SmallUNet3D()
    images = torch.randn(2, 2, 32, 32, 32)

    logits = model(images)

    assert logits.shape == (2, 1, 32, 32, 32)


def test_small_unet3d_runs_on_cpu_without_cuda() -> None:
    model = SmallUNet3D().cpu()
    images = torch.randn(1, 2, 16, 16, 16, device=torch.device("cpu"))

    logits = model(images)

    assert logits.device.type == "cpu"
    assert logits.shape == (1, 1, 16, 16, 16)
