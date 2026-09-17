from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from pet_ai.datasets.synthetic_petct import SyntheticPETCTDataset
from pet_ai.losses.segmentation import bce_dice_loss
from pet_ai.models.unet3d import SmallUNet3D
from pet_ai.training.engine import train_step


def _count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _parameters_changed(before: list[torch.Tensor], model: torch.nn.Module) -> bool:
    current = [parameter.detach().cpu() for parameter in model.parameters() if parameter.requires_grad]
    return any(not torch.equal(old, new) for old, new in zip(before, current, strict=True))


def _gradients_are_finite(model: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a native PyTorch synthetic PET/CT demo.")
    parser.add_argument("--device", default="cpu", help="Torch device to use. Use 'cpu' for CI.")
    parser.add_argument("--steps", type=int, default=2, help="Small number of optimization steps.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device!r}, but CUDA is not available")

    torch.manual_seed(20260916)
    device = torch.device(args.device)
    dataset = SyntheticPETCTDataset(num_samples=2, spatial_shape=(32, 32, 32), seed=20260916)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    model = SmallUNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    initial_parameters = [
        parameter.detach().cpu().clone() for parameter in model.parameters() if parameter.requires_grad
    ]

    with torch.no_grad():
        initial_logits = model(batch["image"].to(device))
        initial_loss = bce_dice_loss(initial_logits, batch["label"].to(device))

    final_result = None
    for _ in range(args.steps):
        final_result = train_step(
            model=model,
            batch=batch,
            criterion=bce_dice_loss,
            optimizer=optimizer,
            device=device,
        )

    if final_result is None:
        raise RuntimeError("training loop did not run")

    print(f"PyTorch version: {torch.__version__}")
    print(f"device: {device}")
    print(f"input shape: {tuple(batch['image'].shape)}")
    print(f"target shape: {tuple(batch['label'].shape)}")
    print(f"model output shape: {tuple(initial_logits.shape)}")
    print(f"trainable parameter count: {_count_trainable_parameters(model)}")
    print(f"initial loss: {float(initial_loss.detach().cpu().item()):.6f}")
    print(f"optimization steps: {args.steps}")
    print(f"final loss: {final_result.loss:.6f}")
    print(f"gradients finite: {_gradients_are_finite(model)}")
    print(f"model parameters changed: {_parameters_changed(initial_parameters, model)}")
    print()
    print("[PASS] native PyTorch forward")
    print("[PASS] loss computation")
    print("[PASS] backward gradients")
    print("[PASS] optimizer parameter update")
    print("[PASS] synthetic training loop")
    print()
    print("Native PyTorch demo completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
