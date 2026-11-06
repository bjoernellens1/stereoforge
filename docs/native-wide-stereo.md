# Native D288 reference stereo

The native `packed-specialized` family supports its existing D1–D256 range and
the explicit D288 extension. D257–D287 and ranges above D288 remain unsupported.
Generic `StereoConfig::validate()` and the RHI/baseline/wave/overwrite kernels
retain their D256 limit. No default disparity range or quality profile changes.

D288 selects separately compiled twelve-register aggregation kernels for both
independent eyes. Existing ranges retain the eight-register kernels. Costs,
recurrence, selection and subpixel math are shared; packed penalties remain
bounded to 8 or 16 bits. Masked subgroup lanes are retained through all shuffles.

Integrated D288 depth requires independent right references and either the
`sequential` or `shared-census-sequential` schedule. Layered, paired-per-path,
hierarchical, derived-right and overwrite depth configurations reject D288.
The 2 GiB allocation ceiling remains unchanged.

For ZED SN26041639's native 1080p factory rectification, `fx=1062.31148 px` and
`B=0.119444 m` imply 253.77346 px at 0.5 m. Adding the campaign's 16 px margin
and rounding up to 32 selects D288. This derives a search range; it does not
establish physical accuracy or qualify that calibration.

A one-slot native 1920×1080 Census four-path/P2=32 shared-Census pipeline
requests 1,609,114,264 bytes and fits the existing ceiling. Two slots do not.
Eight-path/P2=31 retains 8-bit packing; eight-path/P2=32 requires wider storage
and exceeds the combined ceiling at native 1080p, even with one slot.

The existing `daxa_depth_benchmark` accepts this explicit reference configuration:

```sh
daxa_depth_benchmark --manifest native1080/full/manifest.json \
  --disparities 288 --paths 4 --p1 8 --p2 32 --slots 1 \
  --descriptor-mode census --right-reference-mode independent-sgm \
  --stereo-schedule shared-census-sequential --validation \
  --frames 20 --warmup 3 --output fresh-result.json --dump-prefix fresh-depth
```

The manifest must contain native input geometry and its matching calibration.
Repeated-pair timings do not establish fresh-frame throughput. D288 availability
does not promote an estimator, qualify uncertainty or change the standalone
estimator's independence from RealSense.
