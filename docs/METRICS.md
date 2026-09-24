
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

Array entry points require same-shaped, finite, strictly binary 3D arrays; they
cannot establish physical alignment. NIfTI evaluation additionally requires a
recognized spatial unit and the same validated physical grid. Units are
converted to millimetres, comparisons use absolute tolerances with `rtol=0`,
and voxel volume is `abs(det(affine[:3, :3]))`. Inputs are not resampled,
thresholded, or repaired. A same-grid result does not prove anatomical
registration. Different native PET/CT grids are not automatically corrupt, but
cannot be fused by matching array indices without justified alignment.

## Lesion-level metrics

`pet_ai.evaluation.lesion_metrics` identifies connected components in 3D binary masks using configurable 6, 18, or 26 connectivity. It reports GT lesion count, prediction lesion count, true-positive lesions, false-positive lesions, false-negative lesions, lesion sensitivity, lesion volumes in mL, and small-lesion failures.

The legacy matching policy is deterministic greedy one-to-one overlap matching.
Candidate pairs require at least one overlapping voxel and are sorted by
descending overlap, then ascending IDs. This is a detection rule, not contour
accuracy and not a claim of equivalence to an AutoPET official implementation.
The overlap matrix `[[9, 8], [8, 0]]` is a counterexample: greedy returns one
match although a two-match assignment exists. Maximum-cardinality matching is
not implemented. Split/merge ambiguity remains explicitly reported.
