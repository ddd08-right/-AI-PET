"""Small native PyTorch training loop helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    batch_size: int
    device: str
    output_shape: tuple[int, ...]


def train_step(
    *,
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> TrainStepResult:
    """Run one explicit native PyTorch optimization step."""

    selected_device = torch.device(device)
    model.to(selected_device)
    model.train()

    images = batch["image"].to(selected_device)
    labels = batch["label"].to(selected_device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    return TrainStepResult(
        loss=float(loss.detach().cpu().item()),
        batch_size=int(images.shape[0]),
        device=str(selected_device),
        output_shape=tuple(int(dim) for dim in logits.shape),
    )
