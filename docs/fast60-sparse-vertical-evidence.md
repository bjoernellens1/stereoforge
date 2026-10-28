# Sparse vertical matching: analytic development evidence

This paired CPU experiment compares ordinary single-row Census cost with the sum
of three vertically adjacent Census costs. Both independently search each eye
and its reverse match at 1280×720, D224 and `min_margin=1`. It compares both
`combined` and globally denser `combined_edge` selections on one previously
inspected synthetic development scene. **All profiles fail the provisional
per-surface thin gate in both eyes.** They are not candidates for promotion.

Each pair used the same selected coordinates, verified in the persisted decision
file. For each eye, the provisional thresholds were at least 95% correct over
all accepted anchors, and, on **each** eligible thin surface, at least two
correct anchors and at least 90% correct over all accepted anchors on that
surface. Empty denominators fail. Correct means disparity error at most one
pixel on eligible analytic truth. The thresholds were written before inputs and
decisions before truth, but the scene was previously inspected; this is not
blind preregistration or held-out evidence.

| Selector | Matcher | Eye | All correct / accepted | Near correct / false eligible | Thin correct / false eligible | Surface 6 correct / accepted |
|---|---|---|---:|---:|---:|---:|
| combined | single | left | 3,589 / 3,666 | 23 / 0 | 11 / 2 | 3 / 5 |
| combined | single | right | 3,721 / 3,800 | 25 / 0 | 41 / 6 | 0 / 0 |
| combined | vertical3 | left | 5,115 / 5,206 | 112 / 1 | 17 / 2 | 2 / 3 |
| combined | vertical3 | right | 5,251 / 5,345 | 123 / 0 | 46 / 2 | 0 / 1 |
| combined_edge | single | left | 26,936 / 27,540 | 75 / 7 | 40 / 7 | 3 / 8 |
| combined_edge | single | right | 27,038 / 27,642 | 71 / 9 | 81 / 13 | 3 / 4 |
| combined_edge | vertical3 | left | 42,385 / 43,007 | 540 / 9 | 82 / 9 | 4 / 11 |
| combined_edge | vertical3 | right | 42,610 / 43,235 | 556 / 12 | 103 / 4 | 2 / 4 |

False counts in the table refer to accepted, eligible points with incorrect
matches; all-accepted denominators also include accepted points outside eligible
truth. Surface 6 has 452 eligible pixels per eye; counts describe sparse
anchors, not dense-depth completeness. Near and thin regions may overlap.
Although every profile passes the overall 95% gate, surface 6 fails the stricter
per-surface gate in every eye. Vertical cost recovers more near anchors in this
scene but also changes false matches; it does not resolve the one-pixel surface.

| Selector | Matcher | Forward + reverse candidate hypotheses | Forward + reverse Hamming upper bound |
|---|---|---:|---:|
| combined | single | 4,795,161 | 4,795,161 |
| combined | vertical3 | 5,506,151 | 16,518,453 |
| combined_edge | single | 38,403,554 | 38,403,554 |
| combined_edge | vertical3 | 45,349,353 | 136,048,059 |

These are both-eye CPU search counters, excluding point selection and other
work, not GPU latency. Single uses an upper bound because searches with fewer
than two supported candidates skip comparisons; vertical3 currently performs
exactly three Hamming comparisons per counted candidate;
reverse candidate counts can differ because forward decisions differ. Both
matchers use the same imagery and their three vertically adjacent costs are
correlated, not independent evidence. The larger stencil may cross boundaries.
The scoring eligibility checks central Census footprints in both eyes; it does
not certify all additional vertical stencil samples as same-surface truth.

Use stronger, independently validated matching evidence and bounded, targeted
recovery next; preserve per-surface false-accept gates and test previously unseen
scenes and independent physical ground truth. No physical accuracy, dense
reconstruction, GPU performance or 60 FPS claim follows from this result.

## Reproduction and identity

With the local-only fixture available, choose a new nonexistent output directory:

```bash
PYTHONPATH=python:. .venv/bin/python benchmarks/evaluate_sparse_vertical.py \
  --coarse artifacts/fast60-refinement-20261028/analytic-family-720p/half/manifest.json \
  --target artifacts/fast60-refinement-20261028/analytic-family-720p/full/manifest.json \
  --family artifacts/fast60-refinement-20261028/analytic-family-720p/family.json \
  --output artifacts/fast60-refinement-20261028/sparse-vertical-rerun
```

The recorded output is the ignored local directory
`artifacts/fast60-refinement-20261028/sparse-vertical-development-v3/`;
it and the fixture may be missing in a fresh checkout. Policy SHA-256 is
`577ab7668998737ea88fe3ba365cd7f4d0a82ad0f371dc1326db53c2c07498ed`;
decisions SHA-256 is
`b8c41583af21c30e8cf549f0cee9ade3178f26ddc0b02915609ee8cd2cada9c5`.
Single matcher, vertical matcher, scorer and evaluator source SHA-256 are:

- `762e696747176c84c4ab6072140376391633968bd59ad44491763b285e517559`
- `ee4ceffa133be4cf021d9838d3446797ddf725fda78ed31b58ee9aace935196c`
- `6d1b4e4f358546b4cdfda652e5881b0c275c728f7887ba6780c68166d953a2bf`
- `78ec765113540e1f5a615054239074e9d8934445055089419f6c01118dc9a041`

Coarse, target and family manifest SHA-256 are respectively
`98e76b180aaad43a38cc8bdfafa8f5e73d045e568279242c118719270877d5ad`,
`2799d18f1bfa13bf862817aabaf05c917055af371472e162467437bc837c7ea0` and
`db7f6265a1be538d6a01091cea54f699a3564a39d289ca2dc2fe3d54080c9b9b`.
The report also records luma, support, loader and refinement source hashes and
NumPy version. Policy, decision and source hashes were checked against the
recorded files. The v4 repeat produced byte-identical decisions and report
to v3 (`cmp`), establishing determinism for this fixture and environment only.
