
# Architecture

The public release is organized as a small pipeline of testable modules. Each module has one responsibility and returns structured results rather than silently changing data.

## Pipeline

manifest -> split validation -> image QC -> label QC -> voxel evaluation -> lesion evaluation -> provenance -> failure analysis

## Module responsibilities

| Stage | Module | Inputs | Outputs | Responsibility |
| --- | --- | --- | --- | --- |
| Manifest | `pet_ai.data.manifest` | Public-safe CSV rows | `ManifestValidationResult` | Check required fields, duplicate case IDs, modality availability, allowed labels/splits, and forbidden private fields. |
| Split validation | `pet_ai.data.split_validation` | Manifest rows | `SplitValidationResult` | Detect patient-level leakage across train/validation/test. |
| Image QC | `pet_ai.qc.geometry` | NIfTI image paths | `GeometryQCResult` | Compare shape, spacing, orientation, and affine. |
| Label QC | `pet_ai.qc.labels` | Segmentation NIfTI path and optional reference | `LabelQCResult` | Check binary labels, NaN/Inf values, non-empty masks when required, and optional geometry alignment. |
| Voxel evaluation | `pet_ai.evaluation.segmentation` | Prediction/ground-truth arrays or NIfTI paths | `SegmentationMetrics` | Count TP/FP/FN and compute Dice, FPV_mL, and FNV_mL. |
| Lesion evaluation | `pet_ai.evaluation.lesion_metrics` | Prediction/ground-truth arrays | `LesionEvaluation` | Label connected components, match lesions, count TP/FP/FN lesions, and report small missed lesions and ambiguity. |
| Provenance | `pet_ai.reproducibility` | File paths, command, seed, status | SHA256 strings and `RunManifest` | Record hashes and run metadata for reproducibility. |
| Failure analysis | `docs/FAILURE_ANALYSIS.md` | Evidence-supported records | Documented categories | Separate software, environment, resource, and scientific/model failures. |

## Why the modules are separated

Manifest and split validation operate on metadata, not images. Geometry and label QC operate on NIfTI files before evaluation. Voxel and lesion metrics answer different questions: voxel overlap measures spatial agreement, while lesion matching measures detection behavior. Provenance is kept independent so any script can record hashes and run context.

This separation makes each behavior independently testable with synthetic data and reduces the risk of hidden exclusions or silent repairs.
