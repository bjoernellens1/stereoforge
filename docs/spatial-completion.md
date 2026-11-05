# Spatial completion and continuous refinement

`DaxaRefinement` provides separate measured and completed observation frames.
`observations()` retains measured stereo. With `planar_recovery=true`,
`completed_observations()` adds plane-supported predictions tagged
`PlanarInferred` (6), with their current source ID and zero temporal age. It
preserves measured geometry and provenance. These additional predictions do not
increase measured completeness. Ultra-fast mode still publishes only left-eye
metric observations.

The planar pass uses coarse stereo priors, local measured anchors, disparity
plane fitting, ray/edge checks and photometric agreement. Its confidence is a
heuristic, not a calibrated error probability or metric uncertainty. Broader
spatial-completion quality and physical accuracy remain unqualified.

## Explicit continuous subpixel refinement

`continuous_subpixel=true` requires `subpixel=true`. It enables the existing
five-position, half-pixel, zero-mean 3x3 photometric refinement and blends that
estimate with the Census subpixel estimate. The explicit option is honored with
adaptive search, including ultra-fast mode; ultra-fast continues to refine only
the left eye. Implicit non-ultra precision behavior is unchanged.

The option defaults to false. No quality-reference or live-quality profile is
selected by this change. Enabling it adds matching work; live latency and
throughput require measurement on the selected native input and calibration.
Existing CPU and Vulkan paths consume the same shader implementation.

## Development evidence and limitations

A fixed analytic audit covers ten ray-rendered scenes: a plane, textureless
patch, foreground silhouette, thin object, gap, slanted plane, dim lighting,
missing support, 0.5/3 m layers, and an extent not divisible by 16. It runs two
existing local-matcher settings on synthetic 256x192 and 258x194 grids. These are
independently rendered measurements at those grids, not resized ZED images or a
physical/native-ZED qualification corpus.

Enabling planar completion preserves all measured/diagnostic/RGB buffers
bit-for-bit and retains measured pixels in the completed layer. It leaves
unsupported pixels invalid. Nevertheless, the original settings add 231 errors
larger than 0.1 m across this audit. Separate provenance does not make these
predictions accurate.

The explicit continuous option previously had no effect in adaptive ultra-fast
mode: all 300 measured-plane hashes matched its disabled counterpart. A native
GPU regression now exercises a fractional-depth plane rendered from its declared
camera models. The regression checks that the option changes the result and
increases correct measured support within the fixed 1%/10 mm tolerance.

With the option disabled, all 600 measured-plane hashes, 100 completed-array
files, and 240 score rows in the audit remain identical. With it enabled, the
adaptive setting gains 19,890 correct measured predictions and 23,105 correct
completed predictions over the fixed support. Gap and dim cohorts nevertheless
add catastrophic predictions. This is development evidence for an available
option, not a default/profile promotion or a verified accuracy-coverage frontier.
Native 720p/1080p factory geometry, independent scene groups, calibrated absolute
uncertainty, clock qualification and physical/live acceptance remain required.

Run the native regression with the existing Daxa build:

```sh
ctest --test-dir build/ros-daxa-native -R daxa_refinement_gpu_contracts --output-on-failure
```
