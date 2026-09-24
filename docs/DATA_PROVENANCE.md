
# Data Provenance

Public executable artifacts contain schemas and synthetic tests; the repository also documents one public-safe aggregate development case. It does not include DICOM, patient NIfTI files, clinical spreadsheets, segmentation masks from patients, model checkpoints, or raw private logs.

A public-safe manifest should use de-identified `case_id` and `patient_key` values. It should not contain names, dates of birth, accession numbers, medical record numbers, hospital numbers, addresses, phone numbers, local absolute paths, or raw DICOM metadata dumps.

Development-exposed data must not be described as an independent test set. If evidence is unavailable, use `NOT_VERIFIED`.

Run manifests record the actual Git commit and dirty state, command and exit
code, supplied artifact hashes, and required dependency versions. Unused or
unavailable evidence remains null or `NOT_AVAILABLE`.

## Real-data preflight: staged status

Phase 2 found authorized local evidence for FDG-PET-CT-Lesions Version 2 and
verified the collection identity and DOI against the official
[TCIA collection page](https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/).
The local conversion manifest exposes four available CT/PET/reference triplets.
Strict read-only QC found all four readable with finite non-empty binary
references, but all four NIfTI headers have unknown spatial units. Physical
volume remains blocked; historical values that assumed millimetres were not
promoted to current verified results.

The OOF19 records define 19 grouped examinations across four folds. Current
artifacts contain five Fold 0 predictions with prediction/checkpoint hashes;
the other 14 records are explicitly `NOT_GENERATED`. The five prediction and
reference artifacts were relocated by recorded SHA256 and are readable,
finite, binary, and numerically same-grid. Their spatial units are unknown, so
no strict segmentation metric or physical volume was calculated. No patient
row, clinical metric, confidence interval, or independent-test conclusion is
published from these checks.

Phase 3 traced the unknown-unit origin without changing source files. Direct
DICOM geometry checks used Pixel Spacing, Image Orientation, and projected
Image Position differences; SliceThickness was not substituted for inter-slice
distance. DICOM LPS coordinates were converted to RAS before comparing world
corners. The intermediate CT/PET NIfTI files retain an explicit millimetre
unit, while SUV, resampled CT, SEG, and downstream OOF outputs retain the same
numeric millimetre geometry but omit the NIfTI unit label. These files are
classified `UNIT_VERIFIED` for a metadata-only derived-copy migration. No
derived image was written because explicit write authorization has not yet
been granted.

The five existing OOF files are one prediction for each of five distinct Fold
0 held-out examinations from one checkpoint. They are not five members for one
case and not a five-seed ensemble. The historical protocol did not explicitly
set a seed. Single-member Dice/quantification can be evaluated only after the
strict unit-tagged derived copies exist; ensemble disagreement, CV/SD, and
reliability AURC remain unavailable.
## Phase 4 authorized metadata migration and technical evaluation (2026-09-23)

The protected audit records were reread before execution. All 22 manifest source hashes matched; 22 non-overwriting derived copies passed checks for shape, dtype, voxel values, affine, qform/sform and codes, scaling, time units, and non-target header fields. Only the NIfTI spatial unit label was set to `mm`; originals were not modified and no resampling, registration, inference, or training was run.

P01–P05 each have exactly one Fold 0 prediction from one checkpoint. Strict same-grid evaluation on the validated derived pairs completed for 5/5 cases and is limited to `REAL_DATA / FOLD0_SINGLE_PREDICTION_TECHNICAL_EVALUATION`. The protected result file contains per-case physical volumes, signed/absolute errors, relative errors when reference volume is positive, Dice, FPV/FNV, and an independent volume identity check. Cases with zero reference volume retain undefined relative error/Dice with reasons.

`patient_in_training=NO` is supported by the recorded membership evidence; development/model-selection exposure remains `UNKNOWN`, so this is not described as a fully independent test. No disagreement, SD/CV, entropy, AURC, confidence intervals, significance tests, or extrapolation to the other 14 OOF19 examinations were performed.

## Separate 40-epoch development baseline

The [40-epoch development baseline case](DEVELOPMENT_BASELINE_40E_CASE.md) is a
separate, `DEVELOPMENT_EXPOSED` single-seed, fold-0 evaluation over five
development validation cases. Its public-safe aggregate record is
[`development_baseline_40e_summary.json`](development_baseline_40e_summary.json).
The final checkpoint (`current_epoch=40`) and the checkpoint selected by online
EMA pseudo Dice (`current_epoch=36`) both have real protected predictions and
evaluations; patient mappings, per-case rows, images, predictions, and weights
remain outside the public repository. This case does not complete the 19-case
OOF plan, independent testing, external validation, or reliability research.

The historical `baseline_config.status=PREPARATION_ONLY_NOT_RUN` value is a
preparation-stage field and is preserved rather than rewritten. Final run
status is determined from the actual checkpoint counters and logging records,
the frozen result files `final_metrics_post40.json`,
`best_metrics_epoch36_post40.json`, and `nnunet_summary.json`, plus the wrapper
records `continue_wrapper_result.json`, `manual_wrapper_result.json`, and
`ema_selected_evaluation_result.json`. Their recorded digests are exposed in
the aggregate JSON for identity and traceability, not as independent scientific
validation.
