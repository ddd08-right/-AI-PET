# Native PyTorch Core

This repository includes a small native PyTorch 3D segmentation baseline to expose the mechanics of tensors, `Dataset`, `nn.Module`, `Conv3d`, logits, segmentation loss, autograd, gradients, and optimizer updates.

This is an engineering and educational baseline. It uses synthetic data only. It is not physiologically realistic, not clinical simulation, not external validation, and not a performance benchmark.

## Tensor convention

The model uses channels-first 3D tensors:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| input image | `[B,2,D,H,W]` | Batch, PET/CT channels, depth, height, width |
| label | `[B,1,D,H,W]` | Binary synthetic target mask |
| output logits | `[B,1,D,H,W]` | Raw model scores before sigmoid |

PET/CT is represented as two input channels. Channel 0 is synthetic PET-like intensity with a brighter synthetic lesion. Channel 1 is a simple synthetic CT-like background pattern. These are software test tensors, not clinical images.

## Model path

`src/pet_ai/models/unet3d.py` defines `SmallUNet3D`, a compact two-level 3D U-Net-style module:

- encoder level 1: `Conv3d` block at full spatial resolution
- downsample: `MaxPool3d`
- encoder level 2: `Conv3d` block at half spatial resolution
- downsample: `MaxPool3d`
- bottleneck: `Conv3d` block at quarter spatial resolution
- decoder: `ConvTranspose3d` upsampling
- skip connections: channel concatenation with matching encoder features
- output: final `1x1x1 Conv3d`

The final layer returns raw logits. It does not apply sigmoid.

## Example shape table

For input `[B,2,32,32,32]` with default `base_channels=4`:

| Stage | Shape |
| --- | --- |
| input | `[B,2,32,32,32]` |
| encoder level 1 | `[B,4,32,32,32]` |
| downsample 1 | `[B,4,16,16,16]` |
| encoder level 2 | `[B,8,16,16,16]` |
| downsample 2 | `[B,8,8,8,8]` |
| bottleneck | `[B,16,8,8,8]` |
| upsample 2 | `[B,8,16,16,16]` |
| concatenate skip 2 | `[B,16,16,16,16]` |
| decoder level 2 | `[B,8,16,16,16]` |
| upsample 1 | `[B,4,32,32,32]` |
| concatenate skip 1 | `[B,8,32,32,32]` |
| decoder level 1 | `[B,4,32,32,32]` |
| output logits | `[B,1,32,32,32]` |

## Losses

`src/pet_ai/losses/segmentation.py` implements:

- soft Dice loss from logits
- combined BCE plus soft Dice segmentation loss

`BCEWithLogitsLoss` semantics are used through `torch.nn.functional.binary_cross_entropy_with_logits`, so logits are passed directly. The soft Dice loss applies sigmoid internally. Epsilon smoothing is explicit, and empty-target behavior is kept finite for software testing rather than reported as a clinical Dice result.

## Training mechanics

`src/pet_ai/training/engine.py` keeps the native PyTorch sequence visible:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

The training helper explicitly moves image and label tensors to the selected device. It does not require CUDA, distributed training, PyTorch Lightning, MONAI engines, or pretrained weights.

## Synthetic test boundary

`src/pet_ai/datasets/synthetic_petct.py` creates deterministic synthetic PET/CT tensors and binary masks from a seed. Unit tests and the demo use these synthetic samples only. No patient data, DICOM, NIfTI patient images, masks, checkpoints, model weights, PHI, or private metadata are needed.

Run the CPU demo:

```powershell
python -m pip install -e ".[test,pytorch]"
python scripts/demo_pytorch_train.py --device cpu
```
