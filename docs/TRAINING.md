# Capture, train and reuse

Neural Grey fits one optical configuration from four observations. Its training
target is the observed intensity stack itself. A ground-truth shading map is
neither accepted by `FitSession` nor read by the fitting CLI.

## Capture and input format

Use a static, light-colored surface and approximately 0°, 90°, 180° and 270°
camera roll. Keep the camera/lens, aperture, focus and image crop fixed. Avoid
strong texture, extensive clipping and changes in illumination during capture.
Small exposure and alignment departures are fitted jointly with the two fields.

Inputs must remain in **sensor coordinates**. Do not rotate each image upright
with display/EXIF orientation: the camera roll is represented in the model's
scene coordinates. Its transforms are `(x,y)`, `(-y,x)`, `(-x,-y)`, `(y,-x)` in
capture order. Clockwise versus counterclockwise acquisition must follow this
coordinate convention.

Supported input layouts:

- `.npy`: one float array with shape `(4, H, W)`.
- `.npz`: the same array under `captures`; unrelated arrays are not training inputs.
- A directory containing `sim_capture_0.npy`, `sim_capture_90.npy`,
  `sim_capture_180.npy`, `sim_capture_270.npy`, each `(H, W)`. These names retain
  compatibility with the prepared paper inputs; the images can be real captures.

Use finite, black-subtracted **linear** intensities in `[0,1]`. JPEG/sRGB values
are not linear sensor measurements. All frames must share one sensor size and
crop. The current path fits luminance shading rather than separate color-channel
shading; grayscale fitting can still be applied uniformly to a linear RGB image.

## Optional Bayer RAW preparation

```shell
python -m pip install ".[raw]"
neural-grey prepare-raw roll0.NEF roll90.NEF roll180.NEF roll270.NEF --long-edge 1024 --output captures.npz
```

The adapter reads the visible Bayer mosaic, subtracts per-channel black levels,
normalizes by channel white levels, averages the two green sites and computes
linear luminance as `0.2126 R + 0.7152 G + 0.0722 B`. It then area-resizes to the
requested long edge. It does not demosaic through a tone-mapped display pipeline,
auto-rotate the sensor frame, apply a camera shading correction, or use GT.
This adapter supports Bayer 2 × 2 layouts; other CFA arrangements need their own
linear preprocessing.

## Fitting recipe

```shell
neural-grey fit captures.npz --focal-mm 28 --sensor-diag-mm 43.3 --epochs 200 --sample-count 240000 --batch-size 8192 --device cuda --output outputs/lens
```

`--device cpu` runs the same fitting code on CPU. `auto` selects CUDA when
available. `--threads` controls CPU execution in the CLI.

| Setting | Default |
| --- | --- |
| Epochs / sample pool / batch | 200 / up to 240,000 pixels / 8,192 coordinates |
| Shading MLP | 7 features, three hidden layers of width 96, Tanh |
| Illumination field | Second-order polynomial + two hidden layers of width 48, Tanh |
| Optimizer | Adam, cosine annealing |
| Shading neural LR | 1e-3 |
| Physical LR | 1.2e-4 |
| Illumination LR | 6e-5 |
| Exposure / warp LR | 1e-4 |
| Cosine scheduler floor | 1e-4, as in the production implementation |
| Seed | 42 |

The production scheduler uses a shared final learning rate of 1e-4, including
groups that begin below that value. The release preserves that behavior.

The regularization weights are: physical `1.0`, illumination `0.1`, residual
`3.0`, fused anchor `0.2`, gain/bias `0.15`, warp `0.2`, fitted focal scale `0.15`,
and center `0.05`. The anchor uses SmoothL1 with beta `0.03`. Loss weights remain
fixed through the fit. The sample pool
is selected once from the sensor grid and reshuffled each epoch. All four
observations participate at every selected coordinate.

Physical focal length in millimeters and sensor diagonal are used by the **post-fit
radial selector**. The production joint fit initializes its normalized optical
scale from a simple cosine fit to the fused observations, matching the released
paper path. Omitting `--focal-mm` returns the D2-stabilized field without radial
selection. Use the active sensor diagonal appropriate to the acquisition crop.

The model retains the production auxiliary weighted least-squares branch for
source compatibility. With the released settings its solver losses have zero
weight, its trust weights are constant, and its output is not mixed into the
shading prediction. It is not an additional learned correction stage.

## Pause and resume

```shell
# Run five epochs of the configured 200-epoch schedule, then pause.
neural-grey fit captures.npz --focal-mm 28 --output outputs/lens --stop-after 5

# Continue from the saved epoch boundary with the original schedule.
neural-grey resume outputs/lens/training.pt --output outputs/lens
```

`training.pt` contains the four inputs, configuration, model, optimizer,
scheduler, RNG states and completed epoch count. It is self-contained: resuming
does not search for data in local research directories. A checkpoint is written
atomically after every epoch; an interrupted epoch is repeated from the last
saved boundary. CPU interrupted/resumed updates are tested against uninterrupted
updates. CUDA reproducibility is subject to the chosen kernels and environment.

An output from a paused fit is labeled `complete: false`; it is usable for
inspection but has not finished the configured training budget. Starting a new
fit in a directory that already contains a fit is rejected.

## Saved map and correction

| File | Contents |
| --- | --- |
| `calibration.npz` | Final shading, joint-fit map, stabilized map and acquisition metadata |
| `training.pt` | State needed to continue training |
| `fit.json` | Configuration, completion state and loss history |
| `shading.png` | Visualization of the final map |

```shell
neural-grey correct outputs/lens/calibration.npz linear_image.npy --output corrected.npy
```

Correction divides linear image values by the saved transmittance. It supports
grayscale and channel-last RGB arrays. A map can be interpolated to another
resolution with the same aspect ratio, provided the sensor crop is unchanged.
No output clipping or exposure renormalization is applied. The default minimum
transmittance `0.05` limits amplification to 20×; change it explicitly with
`--minimum-transmittance` if appropriate to your acquisition.

For programmatic access, use `FitConfig`, `FitSession.run`,
`FitSession.save`/`resume` and `Calibration.save`/`load`. The ordinary PyTorch
model and loss function are exposed in `model.py` and `training.py` for method
development. The small analytic demo is reproducible from source and is not a
substitute for the paper's full experimental datasets.
