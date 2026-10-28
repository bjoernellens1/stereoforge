# Edge-expanded compact SGM: offline development evidence

A fixed image-edge expansion slightly improves thin completeness over the radius-2 control, but remains far below full-D quality and has substantially more false accepted thin matches. This is one previously inspected synthetic 1280×720 scene; it is **not** held-out or physical evidence. The candidate's recorded provisional gates were **not formally evaluated**; no production promotion, live performance, GPU timing, or 60 Hz conclusion follows.

The candidate independently computes four-path left/right Census SGM with D128, P1=8, P2=32, integer winners and one-pixel LR tolerance. At each guided pixel within a 2-pixel dilation of a luma difference of at least 64, it widens the guided bounds by eight disparities on either side; a missing prior retains full D. This is an image-edge policy, not a confidence- or temporal-guided scheduler. Settings were chosen after the scene had been inspected.

| Correct metric completeness (%, left / right) | Radius-2 control | Edge-expanded candidate | Full-D control |
|---|---:|---:|---:|
| All | 94.884 / 94.931 | 95.279 / 95.317 | 95.339 / 95.371 |
| Near, within D128 support | 99.488 / 99.459 | 99.526 / 99.310 | 99.138 / 98.669 |
| Thin | 44.501 / 45.480 | 45.101 / 46.239 | 60.714 / 62.263 |
| Edges | 60.747 / 61.299 | 61.073 / 61.451 | 65.273 / 64.894 |

Thin-region false LR accepts / total LR accepts were **847/2,255 and 668/2,107** for radius 2, **830/2,257 and 693/2,156** for the candidate, and **402/2,323 and 384/2,354** for full D, respectively left and right. Overall false LR accepts were 33,210 / 33,054, 32,673 / 32,578 and 31,001 / 31,181. The false counts are over the named whole region, including LR accepts outside the clean domain, and mean invisible or disparity error above one pixel. They are not a clean-support false-accept rate.

Completeness counts correct metric-valid estimates within one disparity pixel over fixed visible, in-range, unmixed truth with Census and remap support; the thin denominator is 3,164 pixels per eye. Near truth beyond disparity 127 is excluded, so the near row cannot qualify the camera's full near-field range. The candidate's clean-support surface 6 reconstructs only **44/452** pixels in either eye; among that surface's metric-accepted pixels, **210/254 left and 206/250 right** are false LR accepts. Independent physical truth and held-out scenes remain necessary.

## Where the candidate fails

On fixed thin truth, **553 left / 342 right** pixels have no integer disparity within one pixel inside the chosen interval. Of the remaining **2,611 / 2,822** representable pixels, **1,427 / 1,463** are correctly metric-valid, **346 / 369** are metric-valid but wrong, and **838 / 990** are metric-invalid. Representability alone does not guarantee opposite-image support, correct aggregation, or LR consistency. These are post-truth diagnostic counts; truth did not choose the intervals. The same missing-candidate counts were reported for radius 2 in [prior feasibility](daxa-sgm-overwrite-evidence.md#separate-compact-window-feasibility).

Four-path interval slots are **411,849,632** across both eyes, or **43.64%** of 943,718,400 dense slots; radius 2 uses **333,315,528** (35.32%). Supported candidate updates are **190,319,284 / 189,599,732** for the candidate versus **151,200,236 / 150,282,652** for radius 2 and **444,492,800 / 444,492,800** for full D. Slots include unsupported positions; updates exclude them. These counters omit other processing. The CPU runner still allocates a dense aggregate score volume and performs full-disparity scans for predecessor minima, state clearing, and final selection. Per-eye runner wall times include I/O and exclude preprocessing; none of these numbers implies GPU speed or memory scaling.

## Post-hoc thin-match diagnosis

A subsequent CPU diagnostic classified the saved candidate selection, with truth loaded solely for reporting. On representable thin pixels, raw matching was invalid for **86 / 105**, wrong for **804 / 847**, and correct but rejected by LR consistency for **294 / 407**, left/right. Of the raw-wrong pixels, **346 / 369** nevertheless passed LR. The 2-pixel dilated edge mask covered **435 / 452 left and 452 / 452 right** pixels on surface 6; **149 / 134** guided pixels on that surface still had no candidate within one disparity pixel of truth after expansion. Edge-mask coverage and candidate representability are distinct from raw-match validity and LR acceptance. The nearest-integer ground-truth cross-view association has clean same-surface support for all 452 surface-6 pixels in each eye; both raw matches are correct for only **44** pairs. These truth-associated correspondences are diagnostic, not independent observations or an algorithm input.

The analysis was chosen **after** inspection of this development fixture, with no pre-frozen gates; it provides no physical, held-out or runtime inference. The ignored local artifacts `artifacts/compact-sgm-edge-posthoc-20261028-v1/{diagnostic.py,run.py,result.json,repeat.json}` may be absent on a fresh checkout. The diagnostic script SHA-256 is `a09c8abf120ab519b0263f4459da9cfd24d66af7c73980561481664afeb4362b`, wrapper SHA-256 is `f4173e715ca0dc0c4b95cbe3f640208058e58d8f4607bbc7588fb53f92acf870`, and both independently produced JSON files hash to `3969a795142a18d1313ccd16b2f25d5839ac73a59685a5521eeff3004098e16b`. From the repository root with the frozen fixture and `.venv` available, the invocation was:

```sh
PYTHONPATH=python:. .venv/bin/python artifacts/compact-sgm-edge-posthoc-20261028-v1/run.py > artifacts/compact-sgm-edge-posthoc-20261028-v1/result.json
```

The JSON records relevant source, manifests and frozen-plane hashes; matching repeat output demonstrates determinism for these inputs and environment, not broader validity.

## Comparison binding and limits

Candidate report `artifacts/compact-sgm-edge-20261028-v1/result.json` has SHA-256 `17fa3448920fd4a845c9c6de2a8fe657bb9e83a6b59a39b8ec28f2e3427a10d0`. The controls are local ignored artifacts `artifacts/daxa-refinement-20261028/compact-sgm-quality-offline/r2-v3.json` (`39d95e86ed3b8f66ff6b3dc59fa44cac8bdf08c2d3b90ee0b9f8d5ffa9822a58`) and `full-v3.json` (`a000dff01e45930ac781cc30561d2f810e1d1f561d0f704a5fdcabd134c2fb4a`). These artifacts may be absent from a fresh checkout.

All three reports record matching coarse/target manifest, truth-family, native report and runner-binary hashes. The candidate's pre-truth freeze also records the same 30 native-plane hashes, runner source hash and compact quality harness source hash as both control reports. Its frozen configuration agrees with its report; both control reports specify the same D, penalties, LR tolerance, integer selection and independent per-eye processing. The candidate calls the same runner and selection code. The pre-truth `freeze.npz` SHA-256 is `48ff587d27f8b3ddf54637194bb9c84e41a359cc90a482813e0386b56a1a5fc3`; its selection file hash is `a48ce724042ca3598f39fcf4544114ced53b49130566bf234637315600d566a0`. The candidate source SHA-256 recorded by this run is `1a5b4a398aa41773ebcf4042638ad866431f8fba7b0a8f1e571b0c83e85575eb`. These checks establish recorded identity and equivalent CPU control semantics, not an independent attestation of how earlier reports were generated.

Both controls and candidate use the offline runner's infinity for unsupported costs. The native SGM uses a finite invalid sentinel and may use subpixel fitting; the full-D row is therefore a **CPU full-D control**, not byte-parity to native SGM. The earlier reports do not persist their output planes, so per-pixel matched outcome comparison cannot be independently reconstructed from their hashes. A fresh matched run, frozen thresholds and held-out evaluation are required before promotion. See [compact SGM evidence](compact-sgm-quality-evidence.md) for the control's original scope.

With the local fixture, native dumps, and runner build manifest available, reproduce
the candidate with fresh output paths (the runner identity is checked):

```bash
F=artifacts/fast60-refinement-20261028/analytic-family-720p
N=artifacts/daxa-refinement-20261028/root-frozen-numeric-matrix-v1/r2-subpixel0
O=/tmp/stereoforge-compact-edge-repro
mkdir -p "$O"
PYTHONPATH=. .venv/bin/python benchmarks/compact_sgm_edge_experiment.py \
  --runner /dev/shm/stereoforge-compact-sgm-v3 \
  --build-manifest /dev/shm/stereoforge-compact-build-v3.json \
  --coarse-manifest "$F/half/manifest.json" --target-manifest "$F/full/manifest.json" \
  --family "$F/family.json" --report "$N/native.json" --dump-root "$N/planes" \
  --frozen "$O/freeze.npz" --output "$O/result.json"
```
