# FP1 tile hypotheses: CPU candidate-recall screen

The first tile-hypothesis initialization fails both provisional screens on the inspected synthetic scene. The union of coarse predictions and full-resolution tile hypotheses represents fewer thin pixels than the pinned old compact intervals. This experiment produces candidate sets only: it does not estimate depth, perform independent LR verification, run SGM on those sets, or demonstrate a speedup.

For each eye independently, each 8×8 tile uses its supported center as a grid point and up to three additional nonzero-gradient points, ranked by gradient then row/column, with Chebyshev separation ≥2. If the center is unsupported, at most three gradient points are used. Each point scans D128 using both-eye Census support. A contiguous equal-cost local-minimum plateau contributes its lowest disparity and a plateau flag; distinct equal-cost minima are not called plateau ties. At most two minima per point, separated by ≥4 disparities, become at most eight hypotheses per tile. Each expands ±4 disparities; duplicate candidates are removed. The tile proposes the same windows to its pixels, which still require both-eye Census support at each disparity. These proposals share same-frame imagery and are not independent observations.

Five predeclared sets were compared: the existing valid coarse prediction windows, tile-local proposals with **no coarse input**, their union, the old pinned compact intervals (including their full-search fallback for missing priors), and full D128. An absent coarse prior contributes an empty coarse-only set; it is not a full-search recovery. All sets share the same support and fixed clean truth domain. Candidate existence within ±1 disparity pixel of truth is evaluated only after all five packed two-eye sets, selected-point and retained-peak identity metadata (excluding peak costs), work and policy manifest were written and hashed.

| Thin candidate recall | Coarse only | No-coarse tile | Union | Old compact | Full D128 |
|---|---:|---:|---:|---:|---:|
| Left, out of 3,164 | 152 | 1,970 | 1,976 | 2,611 | 3,164 |
| Right, out of 3,164 | 132 | 1,608 | 1,654 | 2,822 | 3,164 |

Of the old compact set's missing thin pixels, union recovers 165/553 left (29.8%) and 95/342 right (27.8%), below the provisional ≥50% requirement. It also misses many pixels represented by old compact; recovered counts do not imply an overall gain. Per-surface union recall for IDs 6/7/8 is 42/452, 386/904, 1,548/1,808 left and 80/452, 388/904, 1,186/1,808 right. For surface 6 specifically, union represents 42 / 80 left/right versus 303 / 318 with old compact: old compact uses edge-expanded intervals and broad fallback, while coarse-only here is the narrower valid-prior window. The full per-eye/surface partition is retained in the result. There are 22,650 visible clean-support truth pixels beyond D128 per eye in the all-region population; D128 cannot represent them. No thin-domain pixels were flagged out of range in this particular scene. Even the full-D control misses a supported truth-near candidate at 1,432 eligible all-region pixels per eye because opposite-eye candidate support is absent. Occluded pixels are excluded from this recall domain and remain unmeasured.

The other provisional screen requires 1–32 supported candidates on ≥90% of own-supported pixels; excluding empty sets prevents an empty result from appearing cheap. Union achieves this on 153,597/913,616 left (16.8%) and 154,766/913,616 right (16.9%), failing both eyes. All union pixels have at least one supported candidate here. The screen is candidate count, not actually processed recurrence work or a release criterion.

Per eye, the point search used 57,036 points, 7,300,608 attempted disparity checks, and 6,924,084 / 6,923,098 supported Hamming operations (left/right); 113,721 / 113,732 point minima were retained, with 12,063 / 11,701 flagged plateaus and zero tile overflow. Census extraction implies 21,926,784 descriptor comparisons per eye; point selection evaluates 913,616 supported interior gradients and makes 14,400 grid probes per eye. Building the four main masks shares one `128*720*1280` candidate-filter pass per eye; building the old compact control uses a second such pass. These counts do not include sorting, feature materialization, coarse generation, downstream recurrence, GPU traffic or latency. Five packed two-eye mask outputs request 29,491,200 bytes each (147,456,000 bytes total), plus 2,764,800 bytes of retained tile metadata; transient and other buffers are additional. These are requested array sizes, not GPU allocations or measured hardware bandwidth.

The fixed synthetic source is record 0/frame 7. The pinned v2 input/selection/report hashes, calibration/fixture loading, native plane hashes and source dependencies are checked. Output work and candidate decisions were frozen before truth. The reviewed CPU run exited 0; `PYTHONPATH=python:. .venv/bin/python -m pytest -q tests/test_fp1_candidates.py` passed 9 tests. It does not qualify physical accuracy, held-out generalization, metric RGB-D, 60 Hz operation, or the genuinely compact recurrence in FP2.

Ignored local artifacts are in `artifacts/fp1-candidates-20261028-v1`: `pretruth.json` SHA-256 `4712775b7ed1816af1278d9683b50310a84143fc837217957e6c198c5dc200b7` and `result.json` `1df583d1816361405dc0e102baea0c505dc6c8dad56a6f0796c6da83d70c9f55`. The manifest records all five candidate and tile-metadata hashes. Source `benchmarks/fp1_candidates.py` SHA-256 `ac712e9d5ac5e4ef93873389a32baa7b4177811ecdf92b80c81317cab81d3eba`. Artifacts may be absent from a fresh checkout.

To reproduce with pinned local inputs and a fresh output directory:

```bash
F=artifacts/fast60-refinement-20261028/analytic-family-720p
N=artifacts/daxa-refinement-20261028/root-frozen-numeric-matrix-v1/r2-subpixel0
A=artifacts/compact-sgm-step-20261028-v2
O=/tmp/stereoforge-fp1-repro
PYTHONPATH=python:. .venv/bin/python benchmarks/fp1_candidates.py \
  --artifact-root "$A" --family-root "$F" --native-report "$N/native.json" \
  --dump-root "$N/planes" --output-root "$O"
```
