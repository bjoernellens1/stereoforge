# Fast60 partial-prior and tile-work feasibility evidence

This is CPU-only analysis of one frozen 1280×720 analytic development fixture,
not held-out or physical-camera evidence. It does not establish GPU latency,
live throughput, production quality, or eligibility for promotion.

## Partial-prior experiment

The result in `artifacts/daxa-refinement-20261028/partial-prior-experiment-v3.json`
is **rejected** by its predeclared exploratory gates. The implementation is
`benchmarks/fast60_partial_prior_experiment.py`; it attempts a partial prior
only where the original guided prior is invalid. It requires at least two
in-bounds, valid, finite and sufficiently confident coarse taps, at least 0.5
combined original bilinear weight, and at most two coarse pixels of disparity
span. It rescales the renormalized disparity by two and searches radius two
independently for each reference eye. It merges these matches with the original
raw matches and performs one independent left/right consistency check. Truth
is loaded only after matching and does not select pixels or candidates.

The fixed acceptance criteria require, in **each eye**, at least a five
percentage-point gain in thin correct metric completeness, at most a one-point
loss in both all-region and near-region completeness, and an all-region false
LR-accept increase no greater than both 10% and 200 counts. The all and near
gates pass; thin gain and false-accept gates fail in both eyes. Separate edge
and region-specific false-geometry gates were not predefined. Even a passing
result from this harness would not authorize promotion.

| Correct metric completeness | Eligible pixels L/R | Baseline L/R | Partial prior L/R |
|---|---:|---:|---:|
| All | 816,257 / 819,070 | 54.499% / 54.369% | 72.284% / 72.088% |
| Near | 33,768 / 34,036 | 52.168% / 51.889% | 56.275% / 56.008% |
| Thin | 3,164 / 3,164 | 0.126% / 0.126% | 2.971% / 2.939% |
| Edges | 10,119 / 9,235 | 11.711% / 9.930% | 19.804% / 18.917% |

Completeness counts depth-valid pixels with disparity error at most one pixel
over the fixed visible, in-range centre-ray truth support with full
Census/remap support and no mixed raw footprint; invalid estimates remain in
the denominator. It is not a percentage of all image pixels.

All-region false LR accepts / total LR accepts are **1,759 / 446,607** and
**1,806 / 447,128** in the baseline, versus **3,907 / 593,927** and
**3,927 / 594,375** in the candidate, left and right respectively. These
counts include LR-accepted pixels outside the clean completeness domain; the
denominator is all LR accepts in the region, not clean truth support. They
must not be represented as a clean-support false-accept rate. The partial
policy selected 127,697 / 129,758 pixels and evaluated an additional 638,312 /
648,669 supported candidates, left/right. Candidate output planes are not
persisted; the artifact contains aggregate results and hashes only.

The script recomputes Census support independently and checks it exactly
against the recorded native support, rejects nonbinary masks, checks the full
native profile and recorded device/report identity, and compares CPU control
matching and LR masks to the native output. The explicit freeze digest and
recorded successful comparison bind the referenced report; this is a
self-consistency check, not independent attestation of the source or device.
The underlying fixed-support evaluator is `benchmarks/compare_daxa_refinement.py`.
The initial failure attribution is in `benchmarks/fast60_prior_diagnostic.py`.

## Tile-work estimate

`artifacts/daxa-refinement-20261028/tile-work-feasibility-support.json` records
the output of `benchmarks/fast60_tile_work.py` on the same fixture. Selection
uses observed missing prior, invalid raw match, or provisional LR failure;
the provisional LR result is recomputed from both independent raw eye maps.
No truth labels drive selection. The unrestricted policy selects **474,993 /
474,472** left/right pixels; requiring independently verified full-resolution
Census support selects **467,009 / 466,488**. In either policy, every tile
is touched: **3,600 / 3,600** for 16×16, **920 / 920** for 32×32 and
**240 / 240** for 64×64, left/right.
That artifact records the hash of the earlier version of the script; the
subsequent risk-concentration artifact below records the newer version.

The table is a structural hypothetical: every selected tile executes D128 and
four state updates per candidate, for each eye independently. Expanded tile
rectangles are clipped to the frame and overlapping halos count once per tile.
Ratios use the full-frame two-eye D128/four-path total of **943,718,400** state
updates as denominator. Both selection policies touch the same tiles and
therefore have identical ratios:

| Tile size | No halo | 4-pixel halo | 8-pixel halo |
|---|---:|---:|---:|
| 16×16 | 1.000× | 2.224× | 3.931× |
| 32×32 | 1.000× | 1.548× | 2.215× |
| 64×64 | 1.000× | 1.255× | 1.540× |

These are counts, not timings or evidence of a working tiled SGM recurrence.
They exclude full preprocessing, LR checking, dispatch overhead, memory traffic,
and other pipeline work. The estimate does not prove that halos reproduce
full-frame SGM path state or quality, and it does not address interactions
between independent eye selections. The artifact records input and script
hashes, but its provenance is limited to the recorded synthetic native planes
and metadata; it is not an independent measurement of live camera performance.

The subsequent
`artifacts/daxa-refinement-20261028/tile-work-risk-concentration-v2.json`
(SHA256 `e4c8a2c481cf70ba7334948aed097f0ae8fbe616061ad9540604309658b5a2b5`)
uses script SHA256 `9be2169761f6f711128f0bc1e597262db36ddf4de885a92dfa720b88a7911903`
and adds tile occupancy thresholds. Tiles are selected independently for each
eye. Risk coverage counts selected tile interiors only, even when halo work is
included; halo overlap increases work and does not add risk coverage. For 16×16
tiles, zero halo, and the Census-supported risk mask, selecting tiles with at
least **25%** risk pixels covers **914,678 / 933,497 (97.98%)** observable risk
pixels while requiring **95.03%** of the hypothetical full-frame dense work.
At a **50%** threshold, the model uses **38.37%** of that work and covers
**512,689 / 933,497 (54.92%)** risk pixels, leaving **420,808** risk pixels
outside selected tiles. Risk here is the observable mask described above, not
ground-truth error or reconstructed depth quality. This single-scene structural
estimate omits other work and GPU timing, and does not establish an accepted
runtime profile.

## Reproduction

From the repository root, with the frozen artifacts present and the project
Python environment installed, use new output paths because both tools refuse
to overwrite files:

```bash
F=artifacts/fast60-refinement-20261028/analytic-family-720p
A=artifacts/daxa-refinement-20261028/root-frozen-numeric-matrix-v1/r2-subpixel0
TMPDIR=/dev/shm .venv/bin/python benchmarks/fast60_partial_prior_experiment.py \
  --coarse-manifest "$F/half/manifest.json" --target-manifest "$F/full/manifest.json" \
  --family "$F/family.json" --report "$A/native.json" --dump-root "$A" \
  --freeze artifacts/tmp-r0/refinement-dispatch-freeze-candidate-v1.json \
  --freeze-sha256 c38eb117e14c2d4e67767b5d83203e82146cd32bc4a2e91333cc2acabcdd854f \
  --comparison "$A/comparison.json" --output /dev/shm/fast60-partial-new.json
TMPDIR=/dev/shm .venv/bin/python benchmarks/fast60_tile_work.py \
  --coarse-manifest "$F/half/manifest.json" --target-manifest "$F/full/manifest.json" \
  --report "$A/native.json" --dump-root "$A" --output /dev/shm/fast60-tiles-new.json
```
