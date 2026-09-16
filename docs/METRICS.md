
# Metrics

## Voxel-level metrics

`pet_ai.evaluation.segmentation` reports:

- TP: predicted foreground voxels that overlap ground truth foreground.
- FP: predicted foreground voxels outside ground truth.
- FN: ground truth foreground voxels missed by prediction.
- Dice: `2 * TP / (2 * TP + FP + FN)` when ground truth is non-empty.
- FPV_mL: false-positive voxels multiplied by voxel volume.
- FNV_mL: false-negative voxels multiplied by voxel volume.

For empty ground truth, Dice is undefined and represented internally as `NaN`; JSON-style output converts it to `null` with status `UNDEFINED_EMPTY_GT`. Dice is not forced to 1 for negative cases because a negative case with false-positive uptake should not look perfect. Use FPV_mL to evaluate negative cases.

## Lesion-level metrics

`pet_ai.evaluation.lesion_metrics` identifies connected components in 3D binary masks using configurable 6, 18, or 26 connectivity. It reports GT lesion count, prediction lesion count, true-positive lesions, false-positive lesions, false-negative lesions, lesion sensitivity, lesion volumes in mL, and small-lesion failures.

The matching policy is deterministic one-to-one overlap matching. Candidate pairs require at least one overlapping voxel and are sorted by descending overlap, then ascending GT component ID, then ascending prediction component ID. Split/merge ambiguity is reported explicitly because overlap matching alone cannot fully resolve biological lesion identity.
