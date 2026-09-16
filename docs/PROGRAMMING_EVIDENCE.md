
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
