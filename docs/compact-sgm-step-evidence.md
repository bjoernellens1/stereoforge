# Sustained-step directional penalty: exploratory CPU evidence

Do not promote this experiment. A reference-eye directional P2 reduction at sustained high-contrast steps regresses thin and edge completeness against fixed P2 on the inspected synthetic development scene. The trial does not establish physical-camera quality, held-out generalization, GPU speed or 60 Hz operation.

Each of four independent paths and both eyes use D128, P1=8, and integer winner selection. The step policy uses P2=8 only when the path crosses an intensity difference >=64, the two same-side changes are each <=16, and all four reference-eye Census support pixels are valid and in bounds. Otherwise it uses P2=32. A fixed-P2 control and the previously used gradient P2 control ran through the same compiled runner, inputs, bounds and output logic. Thresholds were specified before scoring, but this scene had already been examined. Each policy's bounds and selected output were persisted before that policy loaded truth; the six outputs were not all frozen before the first truth access.

| Correct metric completeness, % (left / right) | Compact fixed | Compact gradient | Compact step | Full-D fixed | Full-D gradient | Full-D step |
|---|---:|---:|---:|---:|---:|---:|
| All | 95.279 / 95.317 | 94.882 / 94.881 | 95.206 / 95.234 | 95.339 / 95.371 | 94.191 / 94.216 | 95.166 / 95.183 |
| Thin | 45.101 / 46.239 | 46.271 / 47.503 | 40.487 / 41.308 | 60.714 / 62.263 | 61.884 / 63.432 | 51.138 / 52.086 |
| Edges | 61.073 / 61.451 | 55.173 / 53.904 | 58.029 / 58.690 | 65.273 / 64.894 | 55.302 / 54.846 | 59.215 / 59.524 |

Correct completeness uses visible, in-range, unmixed truth with Census and remap support. LR false accepts cover the whole named region including outside clean support, and have invisible truth or disparity error >1 pixel. In thin regions, compact fixed / gradient / step false accepts were 830/693, 413/385, and 786/650; full-D counts were 402/384, 189/175, and 435/454 (left/right). Counts by surface ID and their full domain are in each JSON report. For example, surface ID 4 false accepts increased from 6,846/6,687 to 9,200/9,252 compact, and from 7,113/7,190 to 8,614/8,603 full-D, comparing fixed to step; this is a label-based diagnostic, not a physical surface claim.

All policies used the same interval slots: compact 411,849,632 of 943,718,400 dense four-path slots (43.64%), full-D 100%. Supported candidate updates were 190,319,284 / 189,599,732 compact and 444,492,800 / 444,492,800 full-D (left/right). These are CPU accounting numbers, not GPU performance; D128 excludes truth requiring disparities above 127. The matched controls retain selections, but this CPU runner is not qualified for bit parity with native subpixel SGM. No provisional quality gate was promoted.

The scalar reference and compiled paths were checked with `PYTHONPATH=python:. .venv/bin/python -m pytest -q tests/test_compact_sgm_step_ablation.py` (12 passed), including both-eye Census mapping, unsupported and out-of-bounds samples, exact thresholds, reverse direction, disconnected intervals, and independent fixed/gradient references.

Local ignored artifacts are `artifacts/compact-sgm-step-20261028-v2/{compact,full}-{fixed,gradient,step}.json`, each with a corresponding `.npz` frozen input and `.npz.selection.npz`. The batch exited 0. The report hashes are respectively:

```
compact-fixed     3884c9772b8027d345e4b36b64dd34738f0253c10fd4d2b799084fe78cb64a52
compact-gradient  a2377d06819938c910ed9e9e556441bf0a5de11be6bdf470c4e84d8703a3e7ce
compact-step      78300b23b78a58178b322d76752d92bf1d534ff2a49728863fe991a1ec74ec14
full-fixed        129b54718c04ef7b05f841319455cb70fd8e0445ad0b5f2d9d7b3a5df22c6319
full-gradient     a75da0851ecaade28a986153f4cfe68d1fe263929233cd3974647aa592d3b407
full-step         f4edfb5db3c1db54ee09e3a50c58dc9291830c2cd06b511a67f69ff3393370e2
```

Runner and build manifest are in `artifacts/compact-sgm-step-20261028-v1`; their hashes are `26c220fc87b79b1f9f2e4fe8a077d426b33e2e9019891b2503175aece9f24143` and `a250d53277ac021894db449013cce3f98fd1f08315cf1aba9cab7505e6f0c94c`. The frozen inputs record source, runner, fixture and selected wrapper dependencies. The files are local and may not exist on a fresh checkout.

To reproduce with the existing fixture and baseline artifacts, use a fresh output directory:

```bash
F=artifacts/fast60-refinement-20261028/analytic-family-720p
N=artifacts/daxa-refinement-20261028/root-frozen-numeric-matrix-v1/r2-subpixel0
B=artifacts/compact-sgm-edge-20261028-v1
R=artifacts/compact-sgm-step-20261028-v1
O=/tmp/stereoforge-step-repro
mkdir -p "$O"
for mode in compact full; do
  extra=()
  if [ "$mode" = full ]; then extra=(--full-d); fi
  for policy in fixed gradient step; do
    PYTHONPATH=python:. .venv/bin/python benchmarks/compact_sgm_step_experiment.py \
      --runner "$R/runner" --build-manifest "$R/build.json" \
      --coarse-manifest "$F/half/manifest.json" --target-manifest "$F/full/manifest.json" \
      --family "$F/family.json" --report "$N/native.json" --dump-root "$N/planes" \
      --baseline-freeze "$B/freeze.npz" --baseline-report "$B/result.json" \
      --policy "$policy" --frozen "$O/$mode-$policy.npz" --output "$O/$mode-$policy.json" "${extra[@]}"
  done
done
```

No activation counters were collected; these observations do not establish why particular matches changed. Source hashes at the time of the v2 runs were:

```
compact_sgm_step_ablation.py      da50872265b1f5f663123d9f8bf5554df9ad8626b73752c3c15a635882553974
compact_sgm_step_experiment.py    6257aa1baa108c3eabe6ff2d658dd1f78589dfbb3c6884eff621c6b165d7a70d
compact_sgm_step_runner.cpp       56710dd21d3b476236fe3bb9b1571f92538aa8e3189f44ff105fe484f14342fe
compact_sgm_gradient_ablation.py  1371c71231547a9b0254433bca7980ec86db50cb1d729db80dafc49767fa5172
```
