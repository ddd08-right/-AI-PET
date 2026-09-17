
# Programming Evidence

| Engineering problem | Implementation | Test | Evidence |
| --- | --- | --- | --- |
| Patient leakage | `src/pet_ai/data/split_validation.py` | `tests/test_split_validation.py` | Detects duplicate cases, duplicate rows, invalid split/label values, and patient keys crossing train/validation/test splits. |
| PET/CT geometry mismatch | `src/pet_ai/qc/geometry.py` | `tests/test_geometry.py` | Compares shape, voxel spacing, orientation, and affine on synthetic NIfTI files. |
| Invalid segmentation labels | `src/pet_ai/qc/labels.py` | `tests/test_labels.py` | Detects invalid labels, non-integer values, NaN/Inf values, empty masks when forbidden, and geometry mismatch. |
| Voxel metrics | `src/pet_ai/evaluation/segmentation.py` | `tests/test_segmentation_metrics.py` | Reports TP, FP, FN, Dice, FPV_mL, and FNV_mL with undefined Dice for empty ground truth. |
| Lesion-level evaluation | `src/pet_ai/evaluation/lesion_metrics.py` | `tests/test_lesion_metrics.py` | Labels connected components, computes lesion volumes, applies deterministic one-to-one overlap matching, counts TP/FP/FN lesions, and reports split/merge ambiguity. |
| End-to-end modular execution | `scripts/demo_pipeline.py` | Synthetic demo command | Imports and calls manifest, split, QC, voxel metric, lesion metric, and provenance modules using temporary synthetic NIfTI files. |
| Experiment provenance | `src/pet_ai/reproducibility/run_manifest.py` | `tests/test_run_manifest.py` | Records command, exit status, file hashes, platform/Python details, seed, optional GPU/checkpoint evidence, and notes. |
| Failure diagnosis | `docs/FAILURE_ANALYSIS.md` | Historical evidence records | Separates software bugs, environment failures, resource failures, and scientific/model failures without claiming new reruns. |
| Native PyTorch model | `src/pet_ai/models/unet3d.py` | `tests/test_unet3d.py` | Checks native `nn.Module` forward propagation from `[B,2,D,H,W]` tensors to raw logits shaped `[B,1,D,H,W]`. |
| Segmentation loss | `src/pet_ai/losses/segmentation.py` | `tests/test_pytorch_loss.py` | Computes BCE-with-logits and soft Dice loss from raw logits with finite synthetic examples. |
| Backward gradients | `src/pet_ai/training/engine.py` | `tests/test_pytorch_training.py` | Exercises `loss.backward()` and verifies at least one trainable parameter receives finite gradients. |
| Parameter updates | `src/pet_ai/training/engine.py` | `tests/test_pytorch_training.py` | Runs an optimizer step and verifies at least one trainable parameter changes. |
| Multimodal synthetic Dataset | `src/pet_ai/datasets/synthetic_petct.py` | `tests/test_synthetic_petct_dataset.py` | Returns deterministic synthetic PET/CT image tensors, binary labels, and synthetic sample IDs without patient data. |
| End-to-end PyTorch execution | `scripts/demo_pytorch_train.py` | `.github/workflows/ci.yml` | Runs native PyTorch forward, loss computation, backward gradients, optimizer update, and synthetic CPU training loop in the `pytorch-core` CI job. |
