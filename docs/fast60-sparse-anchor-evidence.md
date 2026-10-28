# Sparse stereo anchors: analytic development evidence

This is an exploratory **CPU** experiment on a single 1280×720 analytic scene,
not dense depth completeness, physical accuracy, GPU performance or a 60 Hz
result. The images in the two views arise from the same synthetic scene;
independent reverse matching supplies a second check, not independent imagery
or independent ground truth. The scene and earlier experiment outputs had already
been examined before this run's policy was persisted. Therefore the provisional
checks below are development diagnostics, not pre-registered acceptance gates.
FS1 remains open.

The matcher selects at most one point per 16×16 cell using either gradient or
Census-descriptor variation, separately per eye; searches Census over D128 or
D224 (`0..127` / `0..223`); rejects insufficient best-versus-second-best margin
(`min_margin=1`); and performs an explicit opposite-eye search. Decisions were
written before loading analytic truth in this evaluator run. A correct point is
within 1 disparity pixel of truth, on pair-supported eligible truth. Geometric
eligibility requires finite, in-range, visible, non-occluded, on-image, non-mixed
truth with source remap and Census support. Pair-supported eligibility additionally
requires the truth correspondence on the same row, rounded by `floor(x + 0.5)`,
to lie on the opposite integer grid and have opposite-eye Census support. This
is an integer-footprint criterion, not bilinear support. Precision below uses
**correct / all accepted**, including accepted
points outside eligible truth in the denominator. False eligible counts exclude
accepted points outside eligible truth; those are reported separately.

## D224 results

| Selector | Eye | Correct / all accepted | Precision | False eligible | Accepted outside eligible |
|---|---|---:|---:|---:|---:|
| gradient | left | 1260 / 1306 | 96.48% | 38 | 8 |
| gradient | right | 1300 / 1345 | 96.65% | 37 | 8 |
| descriptor | left | 2383 / 2414 | 98.72% | 17 | 14 |
| descriptor | right | 2477 / 2513 | 98.57% | 18 | 18 |

For each region the four numbers are **correct / selected eligible / eligible
pixels / false accepted eligible**. Regions can overlap, so rows are not additive.
They quantify sparse selected points and pair-supported truth, not reconstructed
dense pixels. For D224, geometric eligibility over the whole image is 838,907
left / 841,720 right; pair-supported eligibility is 837,475 / 840,288, a
difference of 1,432 per eye. All listed near, thin and edge region eligibility
counts happen to equal their geometric eligibility on this fixture.

| Selector | Eye | Near | Thin | Edges |
|---|---|---|---|---|
| gradient | left | 0 / 193 / 56,418 / 0 | 8 / 23 / 3,164 / 0 | 14 / 90 / 10,869 / 4 |
| gradient | right | 0 / 199 / 56,686 / 0 | 24 / 42 / 3,164 / 4 | 26 / 85 / 9,985 / 5 |
| descriptor | left | 23 / 205 / 56,418 / 0 | 3 / 14 / 3,164 / 2 | 6 / 79 / 10,869 / 6 |
| descriptor | right | 25 / 199 / 56,686 / 0 | 18 / 37 / 3,164 / 4 | 10 / 65 / 9,985 / 7 |

There is one accepted point outside eligible edge support per eye for gradient;
all listed descriptor region results have zero outside-eligible accepts. Correct
thin points per eligible thin surface ID 6/7/8 are gradient left **3/5/0**, right
**0/6/18**; descriptor left **0/0/3**, right **0/0/18**. Thus all four rows
fail the provisional requirement of at least one correct point on *every* eligible
thin surface. Gradient also fails the provisional near gate in both eyes. Every
row passes the weak 95% all-accepted precision and at-least-one edge checks;
descriptor passes the at-least-one near check. Passing these weak checks does
not qualify the algorithm or show useful dense reconstruction.

D224 searched 1,456,662 forward and 740,263 reverse candidates for gradient,
and 1,456,298 forward and 1,208,750 reverse for descriptor, summed across eyes.
These are CPU comparison counts, not GPU aggregation work or latency estimates.

## D128 control

D128 has a different eligible range and must not be compared as if it covered the
same near-field truth. Gradient left/right precision is **95.50% / 96.78%**;
near correct is **0 / 0**, thin correct **8 / 24**, edges correct **15 / 27**.
Descriptor precision is **98.22% / 97.95%**; near correct **20 / 24**, thin
correct **3 / 18**, edges correct **7 / 10**. Both selectors still fail the
per-surface thin gate for both eyes; gradient also fails near for both eyes.
Gradient uses 867,058 forward and 473,158 reverse comparisons; descriptor uses
866,896 forward and 746,526 reverse comparisons. Counts include both eyes.

A **post hoc** gradient-only diagnostic (`/tmp/stereoforge-anchor-review.py`,
not a frozen artifact) found every selected near point ambiguous: 193 left and
199 right. The rounded true match had supported Census footprints for all these
points; the rounded true disparity was among minimum-cost ties for 144/193 left
and 150/199 right. This points to ambiguity in this fixture, not proof of a
universal cause. Descriptor selection recovers some near anchors but loses thin
and edge correct points versus gradient; it does not pass the thin gate. Broader
matching evidence, independent validation and bounded fallback for unanchored
surfaces remain necessary before advancement. Measure precision and coverage on
fixed held-out synthetic and independent physical truth before promotion.

## Reproduction and identity

From the repository root, with the local analytic fixture and Python environment
already available, choose a **new nonexistent** output directory:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/evaluate_sparse_anchors.py \
  --coarse artifacts/fast60-refinement-20261028/analytic-family-720p/half/manifest.json \
  --target artifacts/fast60-refinement-20261028/analytic-family-720p/full/manifest.json \
  --family artifacts/fast60-refinement-20261028/analytic-family-720p/family.json \
  --output artifacts/fast60-refinement-20261028/sparse-anchors-rerun
```

The original local-only, ignored directory is
`artifacts/fast60-refinement-20261028/sparse-anchors-development-v3/`; it and the
fixture may not exist on a fresh checkout. Its policy SHA-256 is
`f5a21e434ec9df2e61d2349ef01fa133f861b9a9dcad47ac759d6e4556bc1993` and
its serialized decisions SHA-256 is
`ccc8fb581df38f45ec29d6f1d2089322e83f15b3da4b676978156fd651d723a3`.
The report records source, fixture, luma and support hashes; evaluator and matcher
source hashes are respectively
`6d1b4e4f358546b4cdfda652e5881b0c275c728f7887ba6780c68166d953a2bf` and
`f5bf9382c13c29dc167f90962b5f3c8f92863109908dc44fbb95db46b411b467`.
These identities bind this exploratory run; they do not qualify subsequent edits.
An independent v4 repeat with the same inputs and source produced byte-identical
v3 decisions and report files. This confirms determinism for this fixture, not
generalization to new scenes.
