# Sparse Census and luma: analytic development evidence

This paired CPU experiment compares single-row Census with lexicographic Census
and center-pixel luma matching, independently in both eyes and their reverse
searches, on one previously inspected 1280×720 analytic development scene.
Both matchers use D224, cell size 16, margin 1, and identical selected coordinates
within each `combined` or `combined_edge` pair, checked before truth is loaded.
**Neither matcher passes the per-surface thin quality gate in either eye.**

The luma score is `256 * Hamming + abs(left_luma - right_luma)` for uint8
luma: one Census-bit difference outweighs any luma difference. Its margin of 1
is in encoded score units, unlike the baseline's margin of 1 Hamming bit;
the matchers therefore do not apply numerically equivalent ambiguity tests.
Brightness and exposure differences may change the luma ordering, and integer
matches cannot resolve subpixel or mixed-footprint boundary geometry. Census
and luma reuse the same images and are correlated, not independent evidence.

For each eye the exploratory gates require at least 95% correct over all
accepted anchors and, for each eligible thin surface, at least two correct and
90% correct over all accepted on that surface. Empty denominators fail. Correct
means at most one pixel of disparity error on eligible analytic truth. Settings
and decisions were persisted before truth loading, but the scene was already
inspected: this is neither blind preregistration nor held-out evaluation. Even
the 90% per-surface gate is a sparse diagnostic, not a deployment criterion.

| Selector | Matcher | Eye | All correct / accepted | Near correct / false eligible | Thin correct / false eligible | Surface 6 correct / accepted |
|---|---|---|---:|---:|---:|---:|
| combined | single | left | 3,589 / 3,666 | 23 / 0 | 11 / 2 | 3 / 5 |
| combined | single | right | 3,721 / 3,800 | 25 / 0 | 41 / 6 | 0 / 0 |
| combined | luma | left | 4,714 / 4,863 | 267 / 1 | 12 / 2 | 4 / 6 |
| combined | luma | right | 4,832 / 4,972 | 273 / 0 | 42 / 7 | 0 / 1 |
| combined_edge | single | left | 26,936 / 27,540 | 75 / 7 | 40 / 7 | 3 / 8 |
| combined_edge | single | right | 27,038 / 27,642 | 71 / 9 | 81 / 13 | 3 / 4 |
| combined_edge | luma | left | 37,494 / 38,621 | 1,598 / 14 | 56 / 8 | 4 / 10 |
| combined_edge | luma | right | 37,645 / 38,818 | 1,606 / 14 | 86 / 14 | 3 / 5 |

False counts are accepted eligible points with incorrect disparity; all-accepted
includes points outside eligible truth. Surface 6 contains 452 eligible pixels
per eye. Near and thin regions can overlap. These are sparse point counts, not
dense-depth completeness. Luma finds more near anchors in this scene but does
not establish acceptable thin-surface accuracy.

| Selector | Matcher | Forward + reverse candidates | Hamming upper bound | Luma upper bound |
|---|---|---:|---:|---:|
| combined | single | 4,795,161 | 4,795,161 | 0 |
| combined | luma | 5,495,831 | 5,495,831 | 5,495,831 |
| combined_edge | single | 38,403,554 | 38,403,554 | 0 |
| combined_edge | luma | 44,966,288 | 44,966,288 | 44,966,288 |

Counters sum both eyes and exclude selection and other work. The luma matcher
compares both Hamming and center luma for every supported candidate when a
search has at least two; the candidate-based counts are upper bounds because
short searches skip comparisons. Reverse counts differ when forward outcomes
change. These are CPU algorithmic work counts, not GPU timings. No physical
accuracy, dense reconstruction, live performance or 60 FPS claim follows.

## Reproduction and identity

The recorded output and fixtures are ignored local artifacts and may be absent
in a fresh checkout. Choose a new output directory:

```bash
PYTHONPATH=python:. .venv/bin/python benchmarks/evaluate_sparse_luma.py \
  --coarse artifacts/fast60-refinement-20261028/analytic-family-720p/half/manifest.json \
  --target artifacts/fast60-refinement-20261028/analytic-family-720p/full/manifest.json \
  --family artifacts/fast60-refinement-20261028/analytic-family-720p/family.json \
  --output artifacts/fast60-refinement-20261028/sparse-luma-rerun
```

Recorded output: `artifacts/fast60-refinement-20261028/sparse-luma-development-v1/`.
Verified SHA-256:

- Policy: `43fd41a42709951b0a87fabaf1fc6e6f8dc9dcdea2d76c24315279955d09a4fc`
- Decisions: `72f40dbfc50fac3309b60778630efd710a16e7b9bb4b4c92832312038e2a2470`
- Single matcher: `762e696747176c84c4ab6072140376391633968bd59ad44491763b285e517559`
- Luma matcher: `cc8578b14c260b32c59f84391e87b3a6d75a8d31a0c4e66a95c7c0272afdcff7`
- Scorer: `6d1b4e4f358546b4cdfda652e5881b0c275c728f7887ba6780c68166d953a2bf`
- Thin scorer: `78ec765113540e1f5a615054239074e9d8934445055089419f6c01118dc9a041`
- Evaluator: `e8aff67e27bfa14d1554659c23d81b732150a1ca3d0ea6e46053e0e69ea1cdcf`
- Coarse manifest: `98e76b180aaad43a38cc8bdfafa8f5e73d045e568279242c118719270877d5ad`
- Target manifest: `2799d18f1bfa13bf862817aabaf05c917055af371472e162467437bc837c7ea0`
- Family manifest: `db7f6265a1be538d6a01091cea54f699a3564a39d289ca2dc2fe3d54080c9b9b`

The report additionally records input luma and support hashes, fixture loader and
refinement hashes, source/capture/calibration associations and NumPy 2.5.4.
An independent v2 repeat produced byte-identical decisions and report files;
this verifies determinism for this fixture and environment only.
