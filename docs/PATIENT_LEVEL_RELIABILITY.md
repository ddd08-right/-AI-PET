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
zero. Negative physical volumes are rejected.

Predictive entropy requires probabilities in `[0, 1]`, not logits. It computes
`-p log(p) - (1-p) log(1-p)`. Values are clipped to float64 machine epsilon and
`1-epsilon` only while evaluating logarithms; conceptual probabilities are not
changed.

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
of width `1/n`. Lower AURC means a better ranking for this convention.

The synthetic demo compares a candidate score with fixed-seed random ranking,
best-case reference ranking based on true error, and deliberately reversed
ranking. Best-case reference ranking is a methodological bound, not deployable.
