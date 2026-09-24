# Patient-level Quantitative Reliability

The reliability candidates summarize variation across multiple binary mask or
probability predictions for one synthetic case:

- Mean pairwise Dice: higher means greater prediction agreement.
- Segmentation disagreement: `1 - mean pairwise Dice`; higher means greater predicted risk.
- Volume coefficient of variation: population SD divided by mean; higher means greater predicted risk.
- Predictive entropy: binary entropy of the ensemble-mean probability map.

Dice uses these explicit conventions: two empty masks score `1.0`, one empty
and one non-empty mask score `0.0`, and two non-empty masks use twice their
intersection divided by the sum of foreground counts. An all-zero volume
ensemble has coefficient of variation `0.0`, reflecting perfect stability at
zero. Negative physical volumes are rejected. CV uses population SD (`ddof=0`)
because the supplied prediction set is treated as the complete current
ensemble for this engineering metric; this is not a claim that population CV
is universally preferable to sample CV.

Predictive entropy requires probabilities in `[0, 1]`, not logits. It computes
`-p log(p) - (1-p) log(1-p)`. Values are clipped to float64 machine epsilon and
`1-epsilon` only while evaluating logarithms; conceptual probabilities are not
changed. The API returns a voxel-wise entropy map, not a patient-level scalar.
Predictive entropy reflects combined predictive uncertainty and is not
equivalent to pure epistemic disagreement or mutual information.

The synthetic demo uses the full-volume mean of that map only as a demonstration
aggregation. Background dominance can reduce its sensitivity to focal lesion
uncertainty; it is not presented as an optimal PET reliability score.

## Error and risk--coverage convention

Absolute error is `|predicted - reference|`. Relative error divides this by
`|reference|`; when reference is zero it returns `NaN`, because the quantity is
undefined and no arbitrary epsilon is used.

Larger risk scores mean less reliable cases. Stable ascending sorting covers
the lowest-risk cases first, so removing coverage corresponds to reviewing the
highest-risk cases. The coverage grid is `[1/n, 2/n, ..., 1]`; zero coverage is
excluded because its mean error is undefined. Risk is mean true error among
covered cases, and full-coverage risk equals cohort mean error.

NaN or infinite errors and risk scores raise `ValueError`; no observation is
silently dropped. AURC uses a right-endpoint Riemann sum across equal intervals
of width `1/n`: `AURC_discrete = (1/n) * sum(R_k, k=1..n)`. This is the
repository's discrete risk--coverage convention, not a universal continuous
AURC definition. Lower AURC means a better ranking for this convention.

## Tie, coverage, and ensemble-output policies

Exact risk-score ties use analytic expected risk over all within-tie
permutations. This avoids using row order, true errors, labels, or ground truth
to break ties. The explicit `tie_policy="stable"` option preserves the legacy
row-order-dependent result for reproduction only. Scores are not rounded to
manufacture ties. AURC has the same unit as its error. The expected AURC of a
fully random ordering is the cohort mean error.

Risk@target coverage must report the integer retained count and resulting
actual coverage. If `floor(n * target)` is zero, risk is undefined. Repeated
random permutations describe ranking variation, not a patient-sampling
confidence interval.

Mean member volume, majority-vote mask volume, and thresholded ensemble-mean
probability-mask volume are different estimators. The demo uses the first. Its
0.9/0.1 maps are explicitly synthetic mathematical inputs, not model outputs;
real entropy is `NOT_AVAILABLE` without genuine probability maps. An all-zero
CV means stability only: every member can share the same miss. Disagreement,
SD/CV, and entropy remain separate candidates, and an unvalidated sum is not an
optimized reliability model. On one grid,
`|V_pred - V_ref| = |FPV - FNV|`; equal FPV and FNV can therefore hide spatial
errors behind zero net volume error.

The synthetic demo compares a candidate score with fixed-seed random ranking,
best-case reference ranking based on true error, and deliberately reversed
ranking. Best-case reference ranking is a methodological bound, not deployable.
