# v0.4 Real Public PET Quantitative-Reliability Proof of Concept

## Scope and research question

The primary clinical workflow is **fixed-budget selective review**. The core technical mechanism is
**patient/examination-level reliability ranking**. The primary question is whether a reliability
score can prioritize PET/CT examinations with high whole-body lesion-mask volume error.

The decision unit is one PET/CT examination. When a patient has multiple examinations, all data
splitting and all statistical resampling must be grouped by patient. Patient grouping must be
verified from provenance; it must not be inferred from filenames or imaging similarity.

The reference standard is an expert imaging segmentation reference. It is **not pathology ground
truth**.

This protocol defines analysis only. It contains no cohort results, model results, or claims of
validated reliability.

## Quantitative endpoints

The primary quantitative error is absolute whole-body lesion-mask volume error in mL:

`|V_pred - V_ref|`

The secondary quantitative error is relative volume error and is defined only when `V_ref > 0`.
No arbitrary epsilon will be introduced for zero-reference examinations. Negative-reference
examinations will be evaluated with absolute volume error and false-positive volume.

## Reliability candidates

The primary reliability-risk candidates are:

1. Segmentation disagreement: `1 - mean pairwise Dice` across independent predictions.
2. Predicted-volume coefficient of variation.

Predictive entropy is exploratory only and may be evaluated only when genuine probability maps
exist. Binary masks must never be converted into fake probability maps.

## Fixed-budget selective-review evaluation

The primary evaluation is the repository-defined discrete risk-coverage AURC. A higher
reliability-risk score means less reliable, and the highest-risk examinations are reviewed first.
Under the repository convention, lower AURC indicates better ranking.

Secondary evaluations are Risk@90% coverage, Risk@80% coverage, and Risk@70% coverage.

The required controls are:

- a reproducible random ranking;
- a non-deployable oracle ranking based on true quantitative error; and
- a deliberately reversed ranking.

The oracle is an analysis control and must never be described as deployable.

## Data and analysis safeguards

External medical images and references remain outside the public repository. Preflight auditing is
read-only: failures are recorded explicitly, and images are not repaired, resampled, thresholded,
clipped, or silently excluded. Empty binary expert references are valid negative examinations.
PET quantitative units, dataset version, provenance, patient grouping, and eligibility must be
supported by direct evidence before an analysis that depends on them. In particular, a PET image
must not be treated as SUV based on its filename or modality alone.

Development-exposed examinations must not be called independent test examinations. Any future
split and any confidence-interval or other statistical resampling procedure must preserve verified
patient groups.

## Claim boundaries

v0.4 does not establish clinical validity, clinical utility, prospective workflow benefit,
pathology-level truth, benchmark superiority, algorithmic novelty, multi-center generalization,
cross-tracer generalization, scanner robustness, reconstruction robustness, disease robustness,
or PSMA generalization. Each requires separate testing before it may be claimed.
