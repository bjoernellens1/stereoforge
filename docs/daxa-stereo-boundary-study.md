# Frozen stereo boundary CPU study

The four predefined CPU models show a tradeoff: 3x3 Census with adaptive P2
recovers substantially more thin geometry, while reducing global correct coverage
and increasing raw disparity error. 5x5 adaptive P2 rejects more false matches,
but also has raw-error regressions. No unconditional best model or default change
follows. Production defaults remain unchanged; no candidate is promoted.

Local evidence lives in `artifacts/daxa-depth-boundary-study-20261027/`:
`report.md`, `summary.json`, `study.json`, and `model-freeze.json`. These local
artifacts are not published repository links.

## Frozen models and scope

All models use 96x32 images, D32, P4, P1=8, maximum/base P2=32 and subpixel
selection. The four choices were frozen before reading baseline outputs or
candidate evaluation:

| Model | Census | P2 |
|---|---|---|
| 5x5_constant | Strict-less 5x5, 24 bits | Constant 32 |
| 3x3_scaled_constant | Strict-less 3x3, 8-bit Hamming multiplied by 3 | Constant 32 |
| 5x5_adaptive | Strict-less 5x5, 24 bits | Adaptive |
| 3x3_scaled_adaptive | Strict-less 3x3, 8-bit Hamming multiplied by 3 | Adaptive |

All four enforce the same radius-2 Census validity mask and correspondence
bounds. Only valid Hamming costs are scaled; unsupported costs remain 10000.
Thus 3x3 receives no extra valid border area. The exact adaptive rule is:

```text
P2_current = max(P1, P2_max * 16 // (16 + abs(I_current - I_previous)))
```

The intensity difference uses the same reference eye at neighboring path
positions. Independent Right aggregation uses mirrored Right luma; derived
Right retains Left aggregation. Intensity adaptation is motivated by
[Hirschmüller's SGM paper](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Hirschmueller2008PAMI.pdf).
The integer formula and factor 16 here are our frozen heuristic, not a claim
that the paper specifies these parameters or guarantees their behavior. Random
texture edges need not coincide with depth edges. GT enters scoring only.

The study evaluated 128 CPU model/input combinations from 32 generated stereo
scenes: development seed 42 and fresh texture seeds 46, 47, 48, each on the
same four plane/slope/thin-occluder/photometric families, clean and with one
fixed noise condition. Noise is independent uniform integer [-4,4] per eye,
clipped after the ideal GT warp; PCG64 seeds are recorded in the freeze. These
are repeated synthetic geometries, not twelve independent physical scenes or
physical camera noise. No new candidate GPU experiment ran.

The baseline prerequisite matched frozen actual RHI GPU raw validity masks and
valid integer winners on all four clean seed-42 cases for Left, independent
Right and derived Right: 12 comparisons. Baseline float outputs differed by at
most one ULP; this bound applies only to those baseline comparisons, not the
candidate models. Separately, all 32 seed-42 LR-mask checks matched actual GPU
masks exactly (four fixtures, two estimators, two eyes, two thresholds).

## Fresh derived Right results

All rows below use LR tolerance 0.75. All-region raw MAE covers visible raw-valid
pixels before LR rejection. LR correct means geometric winner-correct accepted
correspondences; false includes wrong visible matches and acceptance outside
visible geometry. Accepted-visible completeness and MAE include wrong visible
matches, whereas correct-visible coverage counts only correct matches. Full
JSON retains both eyes, both estimators, all/thin/boundary regions and both
frozen LR thresholds (1.0 and 0.75).

| Condition | Model | All LR correct / false | Correct-visible coverage | All raw MAE (px) | Thin raw correct / valid | Thin LR correct / false |
|---|---|---:|---:|---:|---:|---:|
| clean | 5x5_constant | 28855 / 269 | 82.917% | 0.1075 | 75 / 162 | 67 / 74 |
| clean | 3x3_scaled_constant | 28826 / 247 | 82.833% | 0.1093 | 115 / 161 | 110 / 39 |
| clean | 5x5_adaptive | 28866 / 114 | 82.948% | 0.1089 | 113 / 161 | 85 / 27 |
| clean | 3x3_scaled_adaptive | 28749 / 118 | 82.612% | 0.1334 | 142 / 166 | 128 / 14 |
| noise4 | 5x5_constant | 28855 / 250 | 82.917% | 0.1087 | 79 / 165 | 71 / 74 |
| noise4 | 3x3_scaled_constant | 28794 / 256 | 82.741% | 0.1134 | 107 / 164 | 100 / 51 |
| noise4 | 5x5_adaptive | 28865 / 121 | 82.945% | 0.1116 | 111 / 161 | 87 / 31 |
| noise4 | 3x3_scaled_adaptive | 28672 / 111 | 82.391% | 0.1480 | 141 / 166 | 120 / 10 |

The combined 3x3 adaptive model increases clean raw-correct thin disparities
75→142 while raw-valid thin count changes 162→166. Thin LR correct rises
67→128 and false falls 74→14; correct-visible thin coverage rises
34.90%→66.67% of 192 GT thin pixels. With noise, raw-correct thin rises
79→141 and LR correct 71→120, with false 74→10. This demonstrates recovery
before rejection in this CPU model. It also lowers global correct counts
28,855→28,749 clean and 28,855→28,672 noisy, while raw MAE rises
0.1075→0.1334 and 0.1087→0.1480. Independent Right also regresses in raw error.

5x5 adaptive reduces derived false counts 269→114 clean and 250→121 noisy,
with small global correct gains (28,855→28,866 and 28,855→28,865). However,
its all-region raw MAE increases, and independent Right raw thin MAE worsens
from approximately 1.115 to 1.901 px on clean data. Lower post-LR error can
partly reflect rejection. The 3x3 constant model also trades global coverage
and noise robustness for thin recovery. None is an established replacement.

## Qualification boundary and identities

Sources and captured inputs match before/after evaluation; all 32 generated
inputs were regenerated and matched saved hashes. Literal checks cover equality,
bit order, common border domain, valid-only cost scaling, constant-image P2,
all-u8 gradient bounds, mirrored Right cost equivalence, and absence of GT from
the matcher API. No fifth model or new threshold was tuned from these results.

Candidate GPU implementation would need independent cost/aggregation/selection
oracles and explicit float/mask checks before production parity qualification.
Unseen geometry, broader texture/noise and physical calibrated captures remain
necessary for accuracy qualification. Transfer-inclusive GPU timing, full pipeline
throughput and sustained live 60 Hz remain unqualified. This CPU study supplies
no candidate GPU speed, physical accuracy or release qualification.

SHA256 identities (verified from current local artifact bytes):

| Identity | SHA256 |
|---|---|
| Frozen configuration | `760a28a41b13aaf1ebf6f3d825fb1bfa1392b71753795bc8c6f9b2295193a285` |
| model-freeze.json | `534de6cbfcac3b350f5201d8ab74fd1b9bb6f084a122b79104c489f49663233c` |
| study.py | `5585a56c357b48ebfe976e10e48e585fabc01cbc16c2d6debe8ac844a47ffe37` |
| report.md | `84ece23f70087a57524448e7e09a84388196c6658b1d6f85eb3fa91773e2dc7a` |
| summary.json | `f80ac73d83b1fd774e3d6f62e94ad712d0d6c7f784432f9a722ad3740d8d73e7` |
| study.json | `b31ff7bca019f6e03e0549bfd6dfdfbde9a5625342e2981924b1b6cf9abd04b3` |
