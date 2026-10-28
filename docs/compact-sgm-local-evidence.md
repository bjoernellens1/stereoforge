# Local stereo evidence under synthetic exposure changes

The joint Census/luma/gradient cost reduces ambiguity on the unmodified analytic scene, but a +25 right-eye brightness shift reverses much of that gain and raises wrong unique minima. A vertical, center-subtracted cost changes less under these shifts but still produces more wrong unique minima than Census alone. None of these local costs has been used by SGM, LR checks or metric-depth output; this experiment does not establish downstream quality or speed.

The input is the previously inspected synthetic frame 7, 720p/eye, with frozen compact or full-D fixed-P2 intervals. Each eye's thin evaluation domain contains 3,164 visible, in-range, unmixed truth pixels with valid Census/remap support. Every candidate for every method requires valid 5×5 Census footprints in both eyes, which also supports the vertical and horizontal stencils. This holds selection support fixed across methods. Vertical stencils can nevertheless cross physical depth surfaces.

Four predeclared costs are Census Hamming; `256*Hamming + abs(center_left-center_right)`; `4096*Hamming + sum(abs((left_i-left_center)-(right_i-right_center)))` for five vertical samples; and `2*Hamming + floor(min(abs(center_left-center_right),32)/4) + floor(min(abs(horizontal_gradient_left-horizontal_gradient_right),64)/8)`, using `I(x+1)-I(x-1)` for horizontal gradient. The first two augmented costs use Hamming as the primary lexicographic key. The joint cost changes the primary ordering.

Each cell below is unique-correct / unique-wrong / ambiguous local minima, left and right respectively. Correct means within ±1 disparity pixel of analytic truth; ambiguous covers every local tie, without silently treating it as correct.

| Full-D local cost | Unmodified left | Unmodified right | +25 clipped left | +25 clipped right |
|---|---:|---:|---:|---:|
| Census | 1,038 / 563 / 1,563 | 1,028 / 637 / 1,499 | 1,025 / 566 / 1,573 | 1,021 / 641 / 1,502 |
| Census + center | 1,359 / 950 / 855 | 1,374 / 1,024 / 766 | 1,217 / 1,248 / 699 | 1,231 / 1,172 / 761 |
| Census + vertical | 1,303 / 902 / 959 | 1,323 / 1,000 / 841 | 1,297 / 909 / 958 | 1,313 / 1,004 / 847 |
| Joint | 1,588 / 738 / 838 | 1,596 / 813 / 755 | 1,084 / 1,179 / 901 | 1,134 / 1,142 / 888 |

The +25 transform clips **45,725 / 921,600** right-image pixels to 255 (4.96%). The additional fixed transform is `clip(rint(0.8*x+20))`; full-D joint unique-correct / unique-wrong becomes 1,389 / 940 left and 1,322 / 1,064 right. These deterministic synthetic changes approximate neither real exposure noise nor camera ISP response. Center subtraction is invariant to an unclipped additive offset, but clipping and gain changes can alter it and the Census descriptor. Both costs use correlated pixels from the same images.

Results are also recorded per surface ID and for compact intervals. Full-D unmodified surface 6, for example, changes from Census 52 / 158 correct/wrong to joint 83 / 247 on the left and from 72 / 187 to 120 / 251 on the right. The compact candidate sets omit a supported truth-near candidate for 553 left and 342 right thin pixels; the full-D sets retain one for each. Candidate comparisons on eligible thin pixels are 328,917 / 353,986 compact and 404,992 / 404,992 full-D per configuration (left/right). These are candidate counts, not actual GPU work, bytes moved or latency measurements.

All 24 mode/exposure/cost configurations, each with both-eye local minima and ties, were persisted and hashed before loading truth. Truth-near costs and surface scores were then computed post-hoc. The frozen v2 artifact hashes, source dependencies, native report and calibrated fixture support are checked. The result records all configuration hashes and saturation counts; it does not attest a rebuilt binary or rehash every input after truth loading. The run exited 0 after independent review.

`PYTHONPATH=python:. .venv/bin/python -m pytest -q tests/test_compact_sgm_local_evidence.py` passed 20 tests, including both-eye scalar comparisons, support/borders, tie handling, exposure clipping and frozen-artifact rejection. The local-only output is `artifacts/compact-sgm-local-evidence-20261028-v1/result.json` (SHA-256 `307b03f57291937b525aefd8aed41b36a0f6a8a5813c85468c12239b531c76ae`); its 24 adjacent `.npz` files contain the pre-truth selections. These ignored local artifacts may be absent on a fresh checkout.

Source hashes: `compact_sgm_local_evidence.py` `e80833d0bd9607f79987c44c359efec224f4643a3990cdfc982da5e8c378ec8b`; imported `compact_sgm_population.py` `bda216bfe0d3ecdc6c62f314bce5e73c2a5176e85c4948f043e775ecc50d0ed3`.

To reproduce with the recorded fixture, use an empty output directory:

```bash
F=artifacts/fast60-refinement-20261028/analytic-family-720p
N=artifacts/daxa-refinement-20261028/root-frozen-numeric-matrix-v1/r2-subpixel0
A=artifacts/compact-sgm-step-20261028-v2
O=/tmp/stereoforge-local-evidence-repro
PYTHONPATH=python:. .venv/bin/python benchmarks/compact_sgm_local_evidence.py \
  --artifact-root "$A" --family-root "$F" --native-report "$N/native.json" \
  --dump-root "$N/planes" --output-root "$O"
```
