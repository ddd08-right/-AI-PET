# PET Quantification Boundaries

The v0.3 utilities convert a three-dimensional non-zero mask into a generic
physical volume. For spacing in `(D, H, W)` millimetres:

`voxel volume (mL) = spacing_D * spacing_H * spacing_W / 1000`

The division follows from `1000 mm^3 = 1 mL`. Mask volume is foreground voxel
count multiplied by voxel volume. Empty masks have volume `0.0 mL`. Spacing
must contain exactly three finite, strictly positive values.

`masked_mean` and `masked_max` summarize finite image values inside a non-zero
mask. Both reject an empty mask because the statistic is undefined.
`uptake_volume_product` multiplies the generic masked mean by generic mask
volume and therefore has units of image-value times mL.

These names deliberately do not imply MTV, TLV, TMTV, or SUV. SUV terminology
is valid only when the PET image has already been correctly converted and
validated in SUV units. Study-specific biological or clinical interpretations
are NOT VERIFIED by these generic calculations.
