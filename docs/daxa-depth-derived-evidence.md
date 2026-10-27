# Left-aggregated Right selection: evidence and limits

The opt-in `left-aggregated-selection` estimator reduces repeated-fixture
720p latency and requested buffer bytes, but changes the Right estimate and
makes both layers' LR checks correlated. Controlled development fixtures expose
quality regressions, including more false accepted correspondences around thin
occluders. **This mode is experimental and is not promoted.** Independent Right
SGM remains the default; exact implementation agreement does not establish
accuracy equivalence or physical depth accuracy.

Evidence is retained under `artifacts/daxa-depth-derived-20261027/`. The
[depth contract](daxa-depth-contract.md) specifies the unchanged layout and
metric support rules; [shared Census evidence](daxa-depth-shared-census-evidence.md)
describes the independent-SGM scheduling control.

## Estimator and ownership contract

Select the new mode with both options:

```text
--stereo-schedule shared-census-sequential \
--right-reference-mode left-aggregated-selection
```

`DaxaDepthConfig::right_reference_mode` defaults to
`DaxaRightReferenceMode::IndependentSGM`. The derived mode is admitted only with
`SharedCensusSequential`. All four old schedules retain their independent-SGM
behavior when the new option is omitted. It adds no CUDA runtime dependency.

Left retains its complete Census, packed SGM recurrence and selection. For a
Right pixel `iR` and candidate disparity `d`, let `iL=iR+d`. The Right score is:

```text
S_right(iR,d) = paths * C_left(iL,d) + unpack(P_left(iL,d))
C_left(iL,d) = popcount(LeftDesc[iL] XOR RightDesc[iR])
```

Image extent, disparity extent, `d < width-xR`, Right Census validity and shifted
Left Census validity are checked before descriptor or penalty loads.
Unsupported candidates are excluded. Selection requires at least two supported
candidates, chooses the lowest disparity on ties, and retains the existing
`best>0`, `best_cost<10000`, `second>best_cost` validity predicate and confidence
formula. Subpixel refinement uses the existing SGM helper only when both
neighboring disparities are supported; each neighbor gathers its own shifted
Left score. Otherwise the result remains integer. Raw remap support does not
veto selection; it still separately gates metric observations.

The Right slot has output/parameter storage but no independent penalty
allocation, clear or path recurrence. A bounded, noncyclic shared storage owner
retains Left penalties, Census and the foundation, including after a Right-only
producer lease outlives the outer pipeline and Left engine. Aligned admission
prevents reuse while either producer is retained. The combined graph registers
the physical Left penalty buffer once and runs `paths+5` phases: two Census,
Left clear, Left paths, Left select and derived Right select. This is nine
phases for P4 or thirteen for P8, with joint phase timings and unavailable
isolated reference timings. Enqueue adds no allocation, wait or readback.

Typed StereoFrames/readbacks report Right mode and estimator. Left's estimator
is `IndependentSGM`; Right's is `LeftAggregatedSelection`. DepthFrames/readbacks
report `LeftAggregatedCorrelated` LR dependency. Both metric layers use
`DaxaDepthProvenance::LeftAggregatedStereo=5` when valid and 0 when invalid.
Source IDs and age behavior are unchanged. Raw Left outputs remain byte
identical to the control, but Left LR outputs may change because they depend on
the changed Right result.

Derived Right `buffer(PackedPenalties)` and intermediate readback reject:
Left penalties must never be presented as an independent Right recurrence.
Raw Right disparity/confidence/validity readback remains available.
`check_stereo_integer()` rejects this mode; `check_derived_stereo_integer()`
checks independent scalar Left Census/integer sums/P, the derived Right mask,
and LR/provenance metadata. Native report flags keep both-independent checks,
Right integer-winner checks and compound CPU float checks false. These flags
remain honest even when the separate GPU-byte comparison passes.

## Functional evidence and byte bindings

Native GPU contracts pass under core and synchronization validation, with empty
stderr and zero validation warnings/errors through teardown. They include 32
literal production-selector combinations and 32 integrated combinations covering
packing widths, P4/P8, W32/W64, one/two slots, retained Right leases, metadata and
provenance5. The literal tests cover excluded unsupported candidates, singleton
support, ties, bounds and supported/unsupported subpixel neighbors.

The new `benchmarks/compare_daxa_depth_derived.py` harness compares all nine
observation planes and six raw stereo planes against the independent
`apps/native_derived_right.cpp` Slang RHI driver. Its reference selector is in
`python/stereoforge/shaders/reference/derived_right.slang`, separate from the
production selector. The driver uses the existing pinned Vulkan-only RHI and
Left SGM control; it adds no copied dependency source. The old legacy harness
remains unchanged.

All four derived comparisons pass exact nine-plus-six output checks:

| Artifact directory | Profile/validation |
| --- | --- |
| `small-derived-sync` | Small fixture, synchronization validation |
| `720p-derived-core` | 720p/D128/P4/W32, core validation |
| `720p-derived-sync` | 720p/D128/P4/W32, synchronization validation |
| `dynamic-p8-w64-derived-sync` | Changed fixture, P8/W64, synchronization validation |

`legacy-small-sequential-core` also passes the old nine-plus-six comparison.
`old-spv-hash-diff.json` verifies all 46 preceding build SPVs and all 33 prior
control SPVs remain byte identical. The three additive SPVs are the packed8 and
packed16 derived selectors and the provenance5 metric wrapper. The canonical
metric body uses a provenance macro whose default value remains 1; the new
wrapper defines 5 and renames the entry before inclusion, sharing metric math
without calling another shader entrypoint.

`native-freeze.json` binds native sources, binaries and SPVs at the pre-GPU
freeze. Its `gpu_executed=false` describes that freeze point; subsequent GPU
results are in the named comparison artifacts and native logs. Each comparison
binds source/compiler/binary/SPV identities, reports and dumps; metadata was
rebound after the reviewed quality-region correction. The CPU-only
`--derived-shader-source-identities` getter exposes
`stereo_derived_shader_source_sha256_triple` (selector, parameters, SGM helper)
and `depth_derived_shader_source_sha256_triple` (wrapper, canonical depth,
observation helper). Legacy source identity keys and legacy report schema remain
unchanged. New reports use `stereoforge.daxa_depth.derived.v1`.

The 300-sample synchronization stress run (`derived-300-sync.json`) completes
303 retirements including warmups, with empty stderr and zero teardown
warnings/errors. Left integer/P, Right mask, LR metadata and preprocessing
checks pass; both-independent and Right integer-winner flags remain false.
GPU mean/p95 are 19.057477/19.473831 ms; host mean/p95 are
19.642098/20.119201 ms. Its 4070.334892 ms diagnostic work occurs after timing.

Final CPU verification is recorded in `root-full-python-final.log`, using the
qualified native capture directory, repository Python path and `.venv`, with
GPU/camera tests explicitly excluded: **1263 passed, 37 deselected** in
28.72 seconds (exit 0).

## Matched latency and buffer measurements

`abba-summary.json` records serial independent/derived/derived/independent runs
on AMD Radeon 8060S Graphics (RADV Strix Halo), Mesa 26.2.3, official Slang
2026.12 and pinned Daxa. Both modes use the shared-Census schedule, the same
retained 1280×720 per-eye fixture, D128/P4/W32, two slots, 100 samples and 20
warmups. Runs omit validation, reference diagnostics and dumps.

| Order/mode | GPU queue-span mean (ms) | Host-wall mean (ms) | Requested buffers (B) |
| --- | ---: | ---: | ---: |
| 1 Independent SGM | 31.442786 | 31.931006 | 811,009,328 |
| 2 Derived Right | 19.124390 | 19.470896 | 575,079,728 |
| 3 Derived Right | 19.065169 | 19.457884 | 575,079,728 |
| 4 Independent SGM | 31.559834 | 32.057641 | 811,009,328 |

This is approximately 39% lower latency for a different estimator. The complete
GPU measure includes upload, preprocessing, stereo, LR, metric metadata and
intergraph gaps on the same queue. Host wall includes submission, completion
wait and timing queries. Neither mode meets the 16.67 ms frame budget.

Removing Right penalties saves 235,929,600 requested bytes at this profile.
Actual allocation accounting equals the requested buffer sum and retains the
2 GiB guard. These are buffer bytes only, excluding allocator/driver overhead,
queries, diagnostic exports and CPU capture storage. They are not peak resident
memory or end-to-end live camera throughput measurements.

## Development quality regression

`quality-development/summary.json` reports independent RHI algorithm references
on controlled integer forward-warp geometry at 96×32, D32/P4. **Production Daxa
was not executed on these new quality fixtures.** The plane, slope,
thin-occluder and photometric cases are development data, not physical captures,
held-out data or promotion evidence; every report retains `promoted=false`.
Eye-specific region masks were corrected during review, and report bindings
were refreshed before these numbers were accepted.

| Right-layer development measurement | Independent SGM | Derived Right |
| --- | ---: | ---: |
| Thin-occluder: false LR geometric correspondences, whole image | 55 | 105 |
| Thin-occluder: mean absolute disparity error, whole image (px) | 0.131514 | 0.149288 |
| Thin foreground: mean absolute disparity error (px) | 1.611648 | 2.239916 |
| Thin foreground: false LR geometric correspondences | 21 | 29 |
| Plane: false LR geometric correspondences, whole image | 0 | 19 |

These are observed quality regressions. False LR geometric correspondence is
the development harness's ground-truth correspondence check, not a production
telemetry counter. The unchanged raw Left control does not make its LR result
independent: both checks reuse information from the Left aggregation. Exact
GPU implementation parity therefore cannot justify equal accuracy or treating
LR acceptance as an independent corroboration.

## Reproduction and remaining qualification

Build options and fixture preparation follow the
[combined-depth instructions](daxa-depth-evidence.md#reproduction-and-remaining-qualification).
Use fresh output destinations for the new harness:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_depth_derived.py \
  --manifest artifacts/daxa-preprocess-20261027/fixture-720p/manifest.json \
  --compiler build/standalone/_deps/sf_slang_sdk-src/bin/slangc \
  --output artifacts/new-derived-exact \
  --frames 30 --warmup 3 --slots 2 --wave-size 32 \
  --disparities 128 --paths 4 --p1 8 --p2 32 \
  --stereo-schedule shared-census-sequential \
  --right-reference-mode left-aggregated-selection --validation
```

Set `VK_LAYER_VALIDATE_SYNC=1` for synchronization validation. Timing-only native
runs omit validation, `--check-reference` and dumps; their new-mode flags are
the same. Held-out production quality, physical/calibration accuracy, fresh
camera throughput and age, wide RGB-D/view intake, temporal/adaptive work and
sustained live 60 Hz remain unqualified. The independent-SGM default is retained.
