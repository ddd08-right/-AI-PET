from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from pet_ai.datasets.synthetic_petct import SyntheticPETCTDataset
from pet_ai.losses.segmentation import bce_dice_loss
from pet_ai.models.unet3d import SmallUNet3D
from pet_ai.training.engine import train_step


def test_backward_populates_finite_gradients() -> None:
    torch.manual_seed(20260916)
    model = SmallUNet3D()
    batch = next(iter(DataLoader(SyntheticPETCTDataset(num_samples=2), batch_size=2)))

    logits = model(batch["image"])
    loss = bce_dice_loss(logits, batch["label"])
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        bool(torch.isfinite(gradient).all().item())
        for gradient in gradients
        if gradient is not None
    )


def test_optimizer_step_changes_at_least_one_parameter() -> None:
    torch.manual_seed(20260916)
    model = SmallUNet3D()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
    batch = next(iter(DataLoader(SyntheticPETCTDataset(num_samples=2), batch_size=2)))
    before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]

    result = train_step(
        model=model,
        batch=batch,
        criterion=bce_dice_loss,
        optimizer=optimizer,
        device="cpu",
    )
    after = [parameter.detach() for parameter in model.parameters() if parameter.requires_grad]

    assert result.batch_size == 2
    assert result.device == "cpu"
    assert result.output_shape == (2, 1, 32, 32, 32)
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))
