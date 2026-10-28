# Population-level thin-region diagnosis

The diagnosis uses the already-inspected analytic development scene and its frozen fixed-P2 SGM selections. It reports associations, not reasons why path aggregation chose a result. No per-pixel compact-to-full transition, physical accuracy, generalization, GPU speed, or 60 Hz claim follows from these counts.

Both modes use the same synthetic record 0 (frame ID 7, synthetic camera clock, capture SHA-256 `0b7e45a21752c6a0fcf867b327f58b2d0d7b5c8d8f1949044e873b9935c42078`) and the v2 experiment's independently estimated left/right SGM with LR checks. Compact uses the frozen edge-expanded bounds; full-D uses [0,128). Per eye, the fixed evaluation support is 3,164 pixels with visible, valid, in-range, unmixed thin truth and own Census/remap support.

| Count (left / right) | Compact fixed | Full-D fixed |
|---|---:|---:|
| Supported candidate within one disparity of truth | 2,611 / 2,822 | 3,164 / 3,164 |
| Raw selected wrong, no supported near candidate | 542 / 337 | 0 / 0 |
| Raw selected wrong, truth-near minimum local Census cost higher | 583 / 601 | 690 / 666 |
| Raw selected wrong, truth-near cost ties local minimum | 134 / 165 | 98 / 123 |
| Raw selected wrong, truth-near candidate uniquely minimizes local cost | 87 / 81 | 38 / 37 |
| Raw correct but rejected by LR check | 294 / 407 | 311 / 246 |

The four wrong-match rows partition raw wrong matches, including pixels with no supported truth-near candidate. A truth-near local minimum refers to the lowest Census cost among candidates within ±1 pixel of truth; tied and unique categories count pixels where at least one supported truth-near candidate attains the overall local minimum; higher-cost cases are separate. LR uses frozen masks, not a new correspondence. Raw-invalid pixels have no retained winner: `select_eye` has already discarded ties, too few candidates, zero disparity and restricted interval endpoints. Aggregate path scores and runner output winner/best/second were not retained, so these categories cannot distinguish their causes or establish how aggregation changed a winner.

The reports include per-eye and per-surface-ID marginal and joint partitions. Surface eligible counts reconcile to the total: IDs 6, 7 and 8 have 452, 904 and 1,808 pixels per eye. For compact left, their supported-near counts are 303, 606, 1,702; compact right 318, 726, 1,778. Full-D has supported-near candidates for every eligible pixel of each surface. Surface labels are synthetic identifiers.

The bounded-memory pass first computes local minima, deterministic smallest-disparity tie winners, tie counts and frozen raw-winner Census costs without truth, and persists them before `load_truth`. Truth-near cost comparisons are computed **after** truth is opened; they do not affect matcher output. The mode-specific artifact identities, input manifest hashes, source dependency hashes, native plane checks, and truth loader validation are checked. This is not an exhaustive rehash of every input after truth loading and does not attest a rebuilt binary. The two mode runs and their enclosing shell exited 0.

Tests: `PYTHONPATH=python:. .venv/bin/python -m pytest -q tests/test_compact_sgm_population.py` (6 passed). A separate review was performed before full-data runs.

Local artifacts (not distributed, may be absent on a fresh checkout):

```
compact-fixed.json           7ccc100f601ad3adb8c27de2e35867f39a251447a1605c1f5c8229a94ecd97a1
full-fixed.json              72796419564f51885aec0ee01d737863a4d7f4543ebefcd649b27f7a28d49755
compact-fixed-truthfree.npz  5b6f25c985cbfc1d7b8593bfdef0f811f4132b704071f69f3e2cb3fd060e597a
full-fixed-truthfree.npz     8fbadf4f1654e401ce7396971497cf98ebe3f39697ea9db78a1480f9b4116df8
```

The files are under `artifacts/compact-sgm-population-20261028-v1`. Diagnostic source SHA-256: `bda216bfe0d3ecdc6c62f314bce5e73c2a5176e85c4948f043e775ecc50d0ed3`. Pinned v2 selections, intervals and reports are specified in the diagnostic source.

Reproduce with the existing frozen fixtures and new output names:

```bash
F=artifacts/fast60-refinement-20261028/analytic-family-720p
N=artifacts/daxa-refinement-20261028/root-frozen-numeric-matrix-v1/r2-subpixel0
A=artifacts/compact-sgm-step-20261028-v2
O=/tmp/stereoforge-population-repro
mkdir -p "$O"
for mode in compact-fixed full-fixed; do
  PYTHONPATH=python:. .venv/bin/python benchmarks/compact_sgm_population.py \
    --mode "$mode" --artifact-root "$A" --family-root "$F" \
    --native-report "$N/native.json" --dump-root "$N/planes" \
    --truthfree "$O/$mode-truthfree.npz" --output "$O/$mode.json"
done
```
