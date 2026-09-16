
# Code Walkthrough

## `src/pet_ai/data/manifest.py`

Purpose: validate public-safe dataset manifest CSV rows. Inputs are CSV paths or row dictionaries. Outputs are `ManifestValidationResult` objects with rows, errors, warnings, and an `ok` property. Major functions are `read_manifest`, `validate_manifest_rows`, and `validate_manifest`. Key decisions: the module rejects forbidden private fields and checks availability flags before any image processing. Failure modes include missing required columns, duplicate case IDs, invalid labels, invalid splits, and positive cases without segmentation references. Tests: `tests/test_manifest.py`.

## `src/pet_ai/data/split_validation.py`

Purpose: detect patient-level leakage and duplicate metadata. Inputs are manifest rows. Output is `SplitValidationResult` with errors and label counts by split. Key decisions: `val` is normalized to `validation`, and patient keys may not cross train/validation/test. Failure modes include invalid split names, invalid label values, duplicate rows, duplicate case IDs, and missing patient keys. Tests: `tests/test_split_validation.py`.

## `src/pet_ai/qc/geometry.py`

Purpose: compare PET/CT/segmentation NIfTI geometry. Inputs are reference and moving NIfTI paths. Output is `GeometryQCResult`. Major functions are `load_nifti_geometry` and `compare_nifti_geometry`. Key decisions: geometry mismatch is reported rather than repaired. Failure modes include shape, spacing, orientation, or affine mismatch. Tests: `tests/test_geometry.py`.

## `src/pet_ai/qc/labels.py`

Purpose: validate segmentation mask labels. Input is a NIfTI segmentation path, with optional reference image. Output is `LabelQCResult`. Major function: `validate_segmentation_labels`. Key decisions: binary labels are expected by default, non-empty masks can be required explicitly, and reference geometry is checked through the geometry module. Failure modes include NaN, Inf, non-integer labels, invalid labels, empty masks, and geometry mismatch. Tests: `tests/test_labels.py`.

## `src/pet_ai/evaluation/segmentation.py`

Purpose: compute voxel-level segmentation metrics. Inputs are arrays or NIfTI paths plus voxel volume. Output is `SegmentationMetrics`. Major functions are `voxel_counts`, `dice_from_counts`, `evaluate_binary_segmentation`, and `evaluate_nifti_segmentation`. Key decisions: empty-ground-truth Dice is undefined, while FPV_mL separately captures false positives. Failure modes include shape mismatch and invalid voxel volume. Tests: `tests/test_segmentation_metrics.py`.

## `src/pet_ai/evaluation/lesion_metrics.py`

Purpose: compute lesion-level detection metrics. Inputs are 3D prediction and ground-truth arrays plus voxel volume. Output is `LesionEvaluation`. Major functions are `connected_component_labels`, `match_lesions`, and `evaluate_lesions`. Key decisions: connected components are implemented locally to avoid a large dependency, connectivity is explicit, and one-to-one matching is deterministic. Failure modes include non-3D inputs, shape mismatch, invalid connectivity, invalid voxel volume, and split/merge ambiguity. Tests: `tests/test_lesion_metrics.py`.

## `src/pet_ai/reproducibility/hashing.py`

Purpose: compute SHA256 hashes. Inputs are file paths. Outputs are hex digests or `None` for unavailable optional paths. Major functions are `sha256_file` and `sha256_or_not_available`. Failure modes include invalid chunk size or unreadable files. Tests: `tests/test_run_manifest.py`.

## `src/pet_ai/reproducibility/run_manifest.py`

Purpose: record reproducible run metadata. Inputs include repository root, optional artifact paths, command, seed, exit status, and notes. Output is a `RunManifest` dataclass that can be written as JSON. Key decisions: Git commit and GPU name are optional because they may be unavailable in public CI or an unpacked release directory. Failure modes include unwritable output paths. Tests: `tests/test_run_manifest.py`.

## `scripts/demo_pipeline.py`

Purpose: run the full public-safe engineering path in one command. Inputs are none. Outputs are PASS lines and temporary synthetic files that are automatically removed. Key decisions: the script imports project modules instead of duplicating QC or metric logic. Failure modes raise explicit exceptions if any expected synthetic check fails.
