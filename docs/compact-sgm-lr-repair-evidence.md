# Opposite-reference recheck: offline development evidence

The recheck recovers few correct thin pixels while accepting additional false geometry. Do not promote it to the GPU or live pipeline. This is a CPU-only experiment on one previously inspected synthetic 720p fixture; it does not establish held-out or physical quality, GPU performance, or 60 Hz operation.

The fixed-P2 and gradient-adaptive compact SGM selections are the two starting points. A valid original winner may request an opposite pixel only when that pixel's original winner is invalid. Requests are deduplicated; targets with multiple requests are rejected. Each remaining target competes over supported disparities 0–127 using opposite-reference Census costs; ties and zero winners are rejected. Original valid values are preserved, and the existing one-pixel LR check is applied to both final maps. The initiating match chooses the target, and the two searches share the same image pair: this is correlated evidence, not independent photometric or SGM validation. Rejecting duplicate requests is only partial collision handling; it does not implement general occlusion or depth-order reasoning, nor correct mutually consistent wrong original matches.

| Thin region, left / right | Fixed original | Fixed + recheck | Adaptive original | Adaptive + recheck |
|---|---:|---:|---:|---:|
| Correct metric completeness | 45.101% / 46.239% | 45.259% / 46.429% | 46.271% / 47.503% | 46.650% / 47.977% |
| False LR accepts | 830 / 693 | 835 / 695 | 413 / 385 | 433 / 398 |

Completeness uses the fixed, visible, in-range, unmixed truth domain with Census and remap support, 3,164 pixels per eye, and a one-pixel disparity correctness threshold. False LR accepts count the whole thin region, including outside this clean domain; they mean invisible or disparity error above one pixel. On the clean thin domain, invalid-to-correct transitions were 5 / 6 for fixed and 12 / 15 for adaptive; invalid-to-false transitions were 5 / 2 and 20 / 13 respectively. Previously metric-valid pixels did not change category in these saved selections. This preservation is not a guarantee of correctness: previously accepted false values remain false.

Across both eyes, the fixed case requested 14,621 rechecks at 10,012 distinct targets and rejected 2,374 multiply requested targets; it evaluated 968,101 supported Census candidates. The adaptive case requested 19,350 at 11,270 targets, rejected 3,503 multiply requested targets, and evaluated 984,888 supported candidates. These are CPU diagnostic work counts, not GPU timings or a complete cost accounting.

## Reproducibility and limitations

The script checks baseline report hashes for the supplied interval and selection files, checks source and input identities, and writes the candidate selections before loading truth. This sequencing does not make the previously inspected fixture held out. The source-current local artifacts are ignored and may be absent on a fresh checkout:

- `artifacts/compact-lr-repair-20261028-v4/result.json`: SHA-256 `968f759b0665a1d3fae88cc5f8b440bd77f081b16fbbc630cacc94df16b8fbbe`.
- `artifacts/compact-lr-repair-20261028-v4/selection.npz`: SHA-256 `524d8782a4f0d7fab9ae87e2afe59c274839fc984d8ef2c15730d81c6a5d6616`.

The result records hashes of the inputs and source dependencies used. Its baseline reports establish recorded identity, not independent attestation of earlier runs. The earlier full-D CPU control did not retain output planes and is not a matched control for this recheck; native finite-sentinel and subpixel behavior also differs. A fresh matched full-D control, fixed quality gates, held-out scenes and independent physical truth are needed for formal evaluation.

With the existing local fixture, native dumps and both prior frozen selections/reports available, reproduce into a fresh directory:

```bash
F=artifacts/fast60-refinement-20261028/analytic-family-720p
N=artifacts/daxa-refinement-20261028/root-frozen-numeric-matrix-v1/r2-subpixel0
B=artifacts/compact-sgm-edge-20261028-v1
G=artifacts/compact-sgm-gradient-20261028-v3
PYTHONPATH=python:. .venv/bin/python benchmarks/compact_sgm_lr_repair_experiment.py \
  --coarse-manifest "$F/half/manifest.json" --target-manifest "$F/full/manifest.json" \
  --family "$F/family.json" --report "$N/native.json" --dump-root "$N/planes" \
  --fixed-freeze "$B/freeze.npz" --fixed-report "$B/result.json" \
  --adaptive-freeze "$G/freeze.npz" --adaptive-report "$G/result.json" \
  --output-dir /tmp/stereoforge-lr-repair-repro
```
