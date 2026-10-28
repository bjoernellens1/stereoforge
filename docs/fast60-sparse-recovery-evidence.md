# Sparse recovery selectors: analytic development evidence

This CPU experiment compares four image-only point selectors on one previously
inspected synthetic 1280×720 development scene. It is not a held-out evaluation,
dense-depth completeness, physical accuracy, GPU timing, or 60 FPS evidence.
FS1 remains open. Both eyes use D224 (integer disparities 0–223), an independent
reverse Census search, 16×16 parent cells, and `min_margin=1`. Combined unions
and deduplicates gradient and descriptor selections. Combined-edge additionally
samples gradient points on a full-frame 4×4 subcell grid; this is denser global
sampling, **not** a selective difficult-region fallback.

Policy was saved before matching, then all matching decisions before truth
loading. Because the scene and previous results were already inspected, the
policy is exploratory rather than blind preregistration. Correct means within
one disparity pixel on visible, in-range pair-supported analytic truth. Precision
is correct divided by **all accepted**, including accepted points outside eligible
truth. False accepted counts below refer only to eligible truth; outside-eligible
acceptances contribute to the precision denominator but are not counted as false
eligible. See [anchor evidence](fast60-sparse-anchor-evidence.md) for eligibility
and nearest-integer opposite-eye Census support details.

| Selector | Eye | Selected | Accepted | Correct / all accepted | Thin correct / false accepted eligible | Surface 6 correct / accepted |
|---|---|---:|---:|---:|---:|---:|
| gradient | left | 3,570 | 1,306 | 1,260 / 1,306 (96.48%) | 8 / 0 | 3 / 3 |
| gradient | right | 3,570 | 1,345 | 1,300 / 1,345 (96.65%) | 24 / 4 | 0 / 0 |
| descriptor | left | 3,570 | 2,414 | 2,383 / 2,414 (98.72%) | 3 / 2 | 0 / 2 |
| descriptor | right | 3,570 | 2,513 | 2,477 / 2,513 (98.57%) | 18 / 4 | 0 / 0 |
| combined | left | 7,052 | 3,666 | 3,589 / 3,666 (97.90%) | 11 / 2 | 3 / 5 |
| combined | right | 7,044 | 3,800 | 3,721 / 3,800 (97.92%) | 41 / 6 | 0 / 0 |
| combined-edge | left | 58,821 | 27,540 | 26,936 / 27,540 (97.81%) | 40 / 7 | 3 / 8 |
| combined-edge | right | 58,819 | 27,642 | 27,038 / 27,642 (97.81%) | 81 / 13 | 3 / 4 |

Thin ground-truth eligibility is 3,164 pixels per eye, surface 6 eligibility
452 pixels per eye; counts are sparse points and must not be interpreted as dense
reconstruction. These regions overlap. Combined-edge passes the provisional
existence checks (one correct anchor on each eligible thin surface and at least
one each in near and edge regions, plus 95% overall all-accepted precision).
Those weak checks do **not** constitute a quality gate: on surface 6, five of
its eight accepted left anchors and one of four accepted right anchors are
incorrect. The dense selector is not suitable for promotion. Combined misses
surface 6 on the right; neither gradient nor descriptor passes the per-surface
existence check in both eyes.

| Selector | Forward candidate comparisons | Reverse candidate comparisons |
|---|---:|---:|
| gradient | 1,456,662 | 740,263 |
| descriptor | 1,456,298 | 1,208,750 |
| combined | 2,874,097 | 1,921,064 |
| combined-edge | 23,962,648 | 14,440,906 |

Counts sum both eyes and are CPU descriptor comparisons, not GPU timing or
SGM aggregation work. The denser selector increases candidate work substantially.
The next experiment needs stronger independent matching evidence and an explicit
per-surface false-accept/coverage gate, together with lower-work targeted recovery.
Evaluate on frozen held-out scenes and independent physical truth before promotion.

## Reproduction and identity

From the repository root, with the local-only analytic fixture and environment
available, choose a **new nonexistent** output directory:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/evaluate_sparse_recovery.py \
  --coarse artifacts/fast60-refinement-20261028/analytic-family-720p/half/manifest.json \
  --target artifacts/fast60-refinement-20261028/analytic-family-720p/full/manifest.json \
  --family artifacts/fast60-refinement-20261028/analytic-family-720p/family.json \
  --output artifacts/fast60-refinement-20261028/sparse-recovery-rerun
```

The run output is the ignored local directory
`artifacts/fast60-refinement-20261028/sparse-recovery-development-v2/`;
it and the input fixture may be absent from a fresh checkout. Its policy SHA-256
is `815c94e2f6442df00571aed935a7216c3b0ba44d23fd69cb8a609f11a6ef1bc9`;
decisions SHA-256 is
`22d2d9bae13a3df6555b9f4eda0f40b340b2fa814b98454ca9ec94592e8fabf4`.
Matcher and evaluator source SHA-256 are respectively
`762e696747176c84c4ab6072140376391633968bd59ad44491763b285e517559` and
`2b5cbb9bf61dcb0631d98565e833a9dd35290b4c7b652911a9a32921f0a0214a`.
Coarse, target and family manifest SHA-256 are respectively
`98e76b180aaad43a38cc8bdfafa8f5e73d045e568279242c118719270877d5ad`,
`2799d18f1bfa13bf862817aabaf05c917055af371472e162467437bc837c7ea0`, and
`db7f6265a1be538d6a01091cea54f699a3564a39d289ca2dc2fe3d54080c9b9b`.
The report additionally records luma, support and scoring-source hashes. A
separate v3 repeat produced byte-identical decisions and reports (`cmp`),
confirming determinism for this fixture only. The corrected v2 evaluator rejects
policy drift. Its results are identical to the earlier v1 profiles; their source
identities differ. The prior anchor experiment remains evidence for its own
recorded matcher source, which differs after selector changes.
