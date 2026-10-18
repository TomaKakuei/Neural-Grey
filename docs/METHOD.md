# Method and release scope

Neural Grey estimates a reusable camera–lens shading map from four approximate
roll captures of a static, light-colored surface. It models capture `k` as

```text
I_k(x) = g_k V(x) L(W_k x) + b_k.
```

`V` is fixed in sensor coordinates, `L` is fixed in scene coordinates, and `W_k`
combines the nominal roll with a small learned translation, rotation and scale
departure. Per-capture gains and biases model exposure and black-level drift.

## Representation and fitting

The shading envelope is an elliptical cosine-power profile with a learned center,
orientation, normalized focal scale and exponent. Bounded coordinate-MLP and
angular residuals supply supported asymmetry. Illumination uses a second-order
polynomial plus a bounded Tanh MLP, evaluated in the transformed scene frame.
These compact fields favor smooth variation over reproducing wall texture.

The default implementation has 24,861 stored parameters, including the retained
inactive auxiliary branch described in the training guide. The coordinate fit
jointly optimizes the fields and nuisance variables using four-image reconstruction,
physical, illumination, residual, fused-anchor and nuisance regularization.
The fused anchor is built from the observations, not from a flat-field target.

## What the rolls cannot determine

A radial field is unchanged under roll. Consequently, camera rolls alone cannot
uniquely assign a broad radial attenuation to the lens or the wall illumination.
Neural Grey explicitly supplies prior information through focal metadata.

Before radial selection, D2 stabilization smoothly suppresses broad non-symmetric
low-frequency energy using horizontal/vertical reflection averages. This is a
regularization assumption: genuine broad asymmetry may also be attenuated.
The stabilized log map is decomposed into a 48-bin radial median and a spatial
remainder. A continuous focal-conditioned rule mixes the fitted radial profile
with wide and standard cosine-power profiles, then restores the spatial remainder
and normalizes the peak to one. It does not re-fit the network or query Lensfun.

The selector is a fixed empirical prior, not a claim that roll observations
uniquely identify radial shading. Controlled flat-field measurement remains the
preferred reference when suitable equipment and illumination are available.

## Implementation and evidence

The source follows the current manuscript, **Neural Grey: Practical Vignetting
Calibration from Casual Camera Rolls**, by Yueze Wang and Spencer Smith:

- The production physical–neural field, preprocessing and losses are retained.
- The complete D2 + continuous focal-conditioned selector is included.
- The package adds explicit device selection, array-based I/O, validation,
  epoch checkpoints, a correction API and a small analytic runnable example.

The original production path and this extraction were compared on a bounded
two-epoch, 1,024-location example on CPU and CUDA. Model parameters, joint-fit
maps and final selected maps matched exactly in that environment. This verifies
the extraction on that check; it is not a new full-benchmark evaluation or a
promise of bitwise agreement across PyTorch/device versions.

The README's PSNR values are the manuscript's existing results. Real-camera
evaluation consists of 21 takes from seven optical configurations. These counts
do not describe 21 independent lenses. Full original RAW datasets and benchmark
outputs are not bundled with this code release. The repository contains no pretrained
population model: fitting starts anew for each calibration set.

Use a calibration only at its matched camera/lens, aperture, focus and crop.
Large scene changes, textured or non-static walls, extensive saturation and
unmodeled capture motion can violate the observation model. This release does
not claim unrestricted handheld scene calibration or separate RGB shading recovery.
