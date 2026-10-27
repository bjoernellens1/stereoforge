# Explicit LR consistency threshold evidence

The diagnostic CLI and derived comparison harness expose the existing native
`DaxaDepthConfig::lr_tolerance` through `--lr-tolerance`. The default remains
1.0 pixels; 0.75 is an explicit rejection setting. This changes neither the raw
matcher nor shader code, resource allocation or the real-time frame path.
Independent SGM remains the default estimator. No mode is promoted.

## Development and frozen rule

The local development precision report
`artifacts/daxa-depth-derived-20261027/precision-analysis-report.md`
examines four deterministic 96x32, D32 development probes on texture seed 42.
At tolerance 0.75, derived Right false correspondences fall from 162 to 82,
while correct correspondences fall from 9,621 to 9,620. Left correct remains
9,616. This exploratory comparison selected the fixed 0.75 rule before new
texture seeds were evaluated; other rejection rules were also explored on the
development set. It did not repair thin foreground geometry.

Here, correct means an LR-accepted rounded correspondence matches independent
geometric winner identity. False includes a wrong visible correspondence or
acceptance outside visible geometry. Accepted-visible completeness includes
wrong visible matches, so it must be distinguished from correct-visible coverage.

## Fresh texture evaluation

The local fixed-rule fresh texture report
`artifacts/daxa-depth-derived-20261027/precision-heldout-report.md`
uses seeds 43, 44 and 45 on the same four geometry/photometric families: plane,
slope, thin occluder and photometric change. Each case is 96x32, D32, P4,
p1=8, p2=32 with subpixel estimates. These are three fresh textures on fixed
geometry families, not twelve independent physical scenes. Plane and
photometric cases share underlying geometry and texture within each seed.

Scoring uses actual frozen RHI raw GPU output bytes, followed by CPU scoring
of the fixed LR rule. It does not use CPU scalar matcher output as the source
of fresh-seed estimates: the scalar prerequisite had exact masks but output
word differences of up to one ULP. These quality fixtures did not execute the
production Daxa pipeline.

| Estimator and tolerance | Eye | Correct | False | Correct-visible coverage | Post-LR visible MAE (px) |
|---|---|---:|---:|---:|---:|
| Derived, 1.0 | Left | 28,848 | 494 | 82.897% | 0.07571 |
| Derived, 0.75 | Left | 28,844 | 414 | 82.885% | 0.07493 |
| Derived, 1.0 | Right | 28,865 | 456 | 82.945% | 0.08304 |
| Derived, 0.75 | Right | 28,858 | 237 | 82.925% | 0.07628 |
| Independent SGM, 1.0 | Right | 28,961 | 177 | 83.221% | 0.07767 |

Derived Right false matches decrease by 48.0%, retaining 28,858/28,865 correct
matches (99.976%). All three fresh texture seeds reduce false Right matches.
The candidate still has more false matches and fewer correct matches than
independent Right SGM at tolerance 1.0.

Thin Right correct matches fall from 76 to 74 and false matches from 72 to 68;
accepted-visible completeness falls from 77.08% to 73.96%. This removes some
accepted pixels and does not recover thin geometry. Independent Right SGM at
1.0 retains 96 correct thin matches. Tightening LR cannot detect two estimates
agreeing on the same wrong surface.

## Reproduction and native qualification

For an exact production comparison at the explicit threshold, use a fresh
output directory with the existing 720p fixture and compiler:

```bash
VK_LAYER_VALIDATE_SYNC=1 PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_depth_derived.py \
  --manifest artifacts/daxa-preprocess-20261027/fixture-720p/manifest.json \
  --compiler build/standalone/_deps/sf_slang_sdk-src/bin/slangc \
  --output artifacts/new-derived-consistency-075-exact \
  --frames 30 --warmup 3 --slots 2 --wave-size 32 \
  --disparities 128 --paths 4 --p1 8 --p2 32 \
  --stereo-schedule shared-census-sequential \
  --right-reference-mode left-aggregated-selection \
  --lr-tolerance 0.75 --validation
```

Root completed five exact comparisons, each passing all nine observation planes
and six raw stereo planes under synchronization validation with zero errors and
warnings and empty harness stderr:

- `small-075-sync`
- `720p-075-sync`
- `small-01-sync` (0.1 normalized to float32 value 0.10000000149011612)
- `small-default-derived-sync`
- `small-default-independent-sync`

Local evidence is recorded in
`artifacts/daxa-depth-tolerance-20261027/exact-summary.json` and
`artifacts/daxa-depth-tolerance-20261027/root-exact.log`; the root runner exited
successfully. Root checked all actual hashes for both default controls: their
fifteen planes are byte-exact to the earlier
`artifacts/daxa-depth-derived-20261027/small-derived-sync` and
`artifacts/daxa-depth-derived-20261027/legacy-small-sequential-core`, respectively.
Earlier immutable artifacts remain separate. The default tolerance is still
1.0 and the default estimator is independent SGM.

CPU verification passes 1,327 tests, with 37 GPU/camera tests deselected, in
28.58 seconds, plus all seven CPU CTest contracts. The frozen native timed-loop
payload of 59 runtime/header/shader/CMake files and all 49 SPIR-V files remains
unchanged. Current source/binary freezes are recorded locally in
`artifacts/daxa-depth-tolerance-20261027/pre-gpu-freeze.json` and
`artifacts/daxa-depth-tolerance-20261027/review.json`; documentation is outside
that payload. This qualification establishes exact implementation agreement on
these fixtures. It adds no performance measurement or performance claim.

The observed quality benefit applies only to these fresh texture seeds. No
unseen geometry, noise or camera qualification, production Daxa quality on these
fixtures, statistical population interval, physical depth accuracy, wide RGB-D
intake or sustained live 60 Hz claim follows. Exact native byte parity would
qualify implementation agreement, not accuracy or physical performance.
