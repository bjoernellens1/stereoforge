# Directional gradient P2: offline development evidence

The adaptive penalty improves thin-region completeness slightly and reduces false LR accepts on one previously inspected synthetic scene, but substantially reduces edge completeness. Do not promote it to the GPU or replace the full-SGM quality baseline on this evidence. This CPU-only post-hoc development experiment does not establish physical accuracy, held-out generalization, GPU performance, or 60 Hz operation. Its provisional gates were not formally evaluated.

Both eyes independently use D128, four-path Census SGM, P1=8, integer winners and one-pixel LR tolerance. At each path step the candidate uses `max(8, floor(32*16/(16+abs(I(p)-I(previous)))))`; the controls use constant P2=32. The compact comparison uses the same edge-expanded per-pixel intervals; the full-D comparison searches all disparities. Configuration was chosen after inspecting this scene.

| Correct metric completeness (%, left / right) | Compact fixed P2 | Compact adaptive P2 | Full-D fixed P2 | Full-D adaptive P2 |
|---|---:|---:|---:|---:|
| All | 95.279 / 95.317 | 94.882 / 94.881 | 95.339 / 95.371 | 94.191 / 94.216 |
| Thin | 45.101 / 46.239 | 46.271 / 47.503 | 60.714 / 62.263 | 61.884 / 63.432 |
| Edges | 61.073 / 61.451 | 55.173 / 53.904 | 65.273 / 64.894 | 55.302 / 54.846 |

Thin-region false LR accepts / total LR accepts were 830/2,257 and 693/2,156 for compact fixed, 413/1,877 and 385/1,888 for compact adaptive, 402/2,323 and 384/2,354 for full-D fixed, and 189/2,147 and 175/2,182 for full-D adaptive, left/right respectively. False LR accepts cover the named whole region, including outside clean support; they are invisible or have disparity error above one pixel. Correct completeness uses the same fixed visible, in-range, unmixed truth with Census and remap support; thin support is 3,164 pixels per eye. D128 excludes farther-disparity near-field truth.

The compact adaptive run visits 411,849,632 of 943,718,400 dense four-path interval slots (43.64%); supported updates total 190,319,284 left and 189,599,732 right. These are not GPU work or timing: this CPU runner allocates dense aggregate scores and also scans all disparities for predecessor minima and output selection.

## Witnesses and controls

Four deterministic first raw-wrong thin witnesses were selected after examining truth, all with full-D intervals. The truth-nearest supported disparity had a lower local Census cost than the wrong winner in three examples, but lost after aggregation: respectively left (x=857,y=134) 0/76 versus 10/73, left (857,136) 0/65 versus 8/59, and right (766,134) 0/61 versus 7/60, where each pair is local/aggregate cost. At right (714,135), truth-nearest 3/123 loses to 2/8, so local evidence also favors the wrong match. These selected cases are not representative and do not establish that adaptive P2 will correct them. The diagnostic reconstructs directional scores; it checks its winner against the frozen selection rather than comparing those scores against the original compiled run. Small-input tests compare compiled and scalar paths, sums and winners, including gaps and both-eye image mapping.

The adaptive runner matches an independently structured scalar recurrence in the focused tests. Candidate bounds and selections were persisted before loading truth. The old compact control binds the same bounds, source frames, calibration, native input report and matching configuration. The old full-D CPU report records matching inputs and settings but did not retain output planes, so its per-pixel outputs cannot be independently reconstructed from the report. Neither CPU control is byte-parity to native finite-sentinel/subpixel SGM. A fresh matched full-D fixed-P2 run with retained selections and frozen thresholds is required for formal evaluation. Further development should preserve thin-object and boundary completeness and false-accept gates, then evaluate held-out and independently measured physical scenes.

## Artifacts and reproducibility

Local ignored artifacts may be absent on a fresh checkout. Their SHA-256 values are:

| Artifact | SHA-256 |
|---|---|
| `artifacts/compact-sgm-gradient-20261028-v3/result.json` | `2ca2e52f66e2c921050714f509defaf7d3f7390ffb7109fd0059a8e62e94ec70` |
| `artifacts/compact-sgm-gradient-20261028-v3/freeze.npz` | `4266a5d568c999e12bc439b6292be68f45d04f91712f1fc8ed556aaa38441071` |
| `artifacts/compact-sgm-gradient-20261028-v3/freeze.npz.selection.npz` | `92fe50faac46bf955f34b220c02832aa9f45dafeacfb5fcc4fa5694501e17b48` |
| `artifacts/compact-sgm-gradient-20261028-full-v2/result.json` | `fc70ff333aca8c84690a0c3a8508725e3864083599a0e1f9a6b4f9fb5d899369` |
| `artifacts/compact-sgm-gradient-20261028-full-v2/freeze.npz` | `22df36f18f78962ad2e72c1b8223e5c6243675d75c95ad52e12fae01f8eca30a` |
| `artifacts/compact-sgm-gradient-20261028-full-v2/freeze.npz.selection.npz` | `6d8672d5124c22222318acd9ed9d3822972273ed2a039a51752624e3978f5e6b` |
| `artifacts/compact-sgm-edge-20261028-v1/witness-v3.json` | `6350c538237e58ad058c0d048ac66c0a2b83b3ef31caf41061c6cd2e44b52b28` |

The runner and compiler manifest are in the compact adaptive directory; their SHA-256 values are `3b225393d7d8f6d0de7dc1020a61000814b22995270f25078a4aef57f1268670` and `dd8f7fe8ed999299b9cb245e51923dd57d572aba264a040bd595503bbe998e1a`. Frozen inputs record and recheck source dependencies, including the scalar reference containing path directions. Full file identities and experiment settings are contained in the frozen files and reports.

To reproduce, with the existing local analytic fixture, native dumps, baseline freeze and result, runner and build manifest present, use fresh output paths:

```bash
F=artifacts/fast60-refinement-20261028/analytic-family-720p
N=artifacts/daxa-refinement-20261028/root-frozen-numeric-matrix-v1/r2-subpixel0
B=artifacts/compact-sgm-edge-20261028-v1
R=artifacts/compact-sgm-gradient-20261028-v3
O=/tmp/stereoforge-gradient-repro
mkdir -p "$O"
for mode in compact full; do
  extra=()
  if [ "$mode" = full ]; then extra=(--full-d); fi
  PYTHONPATH=python:. .venv/bin/python benchmarks/compact_sgm_gradient_experiment.py \
    --runner "$R/runner" --build-manifest "$R/build.json" \
    --coarse-manifest "$F/half/manifest.json" --target-manifest "$F/full/manifest.json" \
    --family "$F/family.json" --report "$N/native.json" --dump-root "$N/planes" \
    --baseline-freeze "$B/freeze.npz" --baseline-report "$B/result.json" \
    --frozen "$O/$mode.npz" --output "$O/$mode.json" "${extra[@]}"
done
PYTHONPATH=python:. .venv/bin/python benchmarks/compact_sgm_witness.py \
  --family-root "$F" --frozen "$B/freeze.npz" \
  --selection "$B/freeze.npz.selection.npz" --result "$B/result.json" \
  --output "$O/witness.json"
```
