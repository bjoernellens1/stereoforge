# Refinement evidence, 2026-10-28

Evidence is static analytic development evidence on AMD Radeon 8060S / RADV
Mesa 26.2.3. No promotion, 60 FPS claim, physical or live qualification follows.
Artifact paths below are relative to the repository root; `A` abbreviates
`artifacts/daxa-refinement-20261028` and `F` abbreviates
`artifacts/fast60-refinement-20261028/analytic-family-720p`.

`A/root-frozen-numeric-matrix-v1/r{2,4,8}-subpixel{0,1}/comparison.json`
contains six passing exact integer/mask/identity comparisons with checked
operation-specific float bounds (`conditional_implementation_agreement`).
Nominal float bits need not match; global arithmetic boundary qualification is
false. Each sibling `native.json` records the actual compiler, shaders and inputs.
`A/root-stress-r4-subpixel-sync-contended.json` records 300 measured frames,
303 refinement source retirements and zero validation errors/warnings after
teardown. Contended stress is lifecycle evidence, not performance qualification.
`A/root-full-cpu-integration.log`: 1707 passed, 50 skipped, 37 deselected.
`A/root-native-controls-existing-extension.log`: the 50 separately run controls
passed. These do not convert the 37 GPU deselections into coverage.
The existing 10-test / 10,000 mock-frame performance guard is host-only.

Frozen provenance (SHA256 unless identified as a commit):

| Item | Identity |
|---|---|
| Numeric dispatch freeze `artifacts/tmp-r0/refinement-dispatch-freeze-candidate-v1.json` | `c38eb117e14c2d4e67767b5d83203e82146cd32bc4a2e91333cc2acabcdd854f` |
| Separate performance-review freeze `A/performance-freeze-guided-r2-v2.json` | `7a78c367f7544e13e9f1a60fb7082aef4d835a3de5c31d7fe3f5c5d7c26a7f26` |
| Source checkpoint referenced by freeze | `ca36a59794b5501951a402cdd3a661ced83ae8e23abc2bc9f441af5414ac6f4f` |
| Freeze HEAD (working source is separately hashed) | commit `d233c66dfd520d179477989aa97d11e7c495d874` |
| Benchmark executable | `8cb823f96f4dd2d1160fbcbfc7bcb5fec264fd0044a79b82e87b106c4a1ed704` |
| Refinement wrapper / ops sources | `a1bb84326462959c87d0c3d2a126d3048ee0d906c31b56308f5aa15214d24084` / `dc4e735bfd0300d54bec71471403e3a0fb2742e9375f2e39053811d4db878c2b` |
| Capture | `0b7e45a21752c6a0fcf867b327f58b2d0d7b5c8d8f1949044e873b9935c42078` |
| `F/half/manifest.json` | `98e76b180aaad43a38cc8bdfafa8f5e73d045e568279242c118719270877d5ad` |
| `F/full/manifest.json` | `2799d18f1bfa13bf862817aabaf05c917055af371472e162467437bc837c7ea0` |
| Frozen coarse compiler 2026.12 | `82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad` |
| Refinement compiler 2026.19 | `b13606619c8959bbd490cb87e6e549e366a22a13c6cbd46cce3f798264778b60` |
| Official refinement tag `v2026.19` (unsigned) | commit `0fb6b7587a94fa33d88141ec9930766eedc6f59a` |
| Official Linux SDK archive | `bd6cfc47b7353b2cffa36a866b1584749f76f4b2dbebf6dda26bc3fb40dc3c0e` |
| SDK `libslang-compiler.so.0.2026.19` | `362b766c60b1819711ae770b193fb580d7422da31f99577ad94ea1b504c16602` |
| SDK `libslang-rt.so.0.2026.19` (current file hash) | `fa5d73a35b12a42e75f757f778bdc17ecc26c6dfe121a7efffd1c9f4bc3881cf` |

These are distinct freezes for distinct scopes. The numeric dispatch freeze binds
904 canonical files and 31 aliases, including both actual compiler dependency
closures; it is verified before and after every comparison. The performance-review
freeze records Ninja compiler/link inputs, runtime dependencies and
SPIR-V hashes; `changed_old_spirv` is empty. The runtime DSO hash above was read
from the extracted SDK; the freeze records the compiler DSO runtime dependency.
The 2026.19 lane is refinement-only; the old 2026.12 lane remains frozen.

R1 world quality below is **correct metric completeness (%)**, left/right,
using the comparison's 1-pixel disparity correctness threshold. Denominators
are clean visible in-range centre-ray truth with full Census/remap support and
no mixed raw footprint: all 816257/819070, near 33768/34036, thin 3164/3164,
edges 10119/9235. They exclude unsupported/mixed/out-of-range truth and are
not total image pixels (921600 per side). Thin false accepts are LR false
accepted / all LR accepted in the thin region, left/right.

| Radius / subpixel | All L/R % | Near L/R % | Thin L/R % | Edges L/R % | Thin false accepts L / R |
|---|---|---|---|---|---|
| r2-subpixel0 | 54.499/54.369 | 52.168/51.889 | 0.126/0.126 | 11.711/9.930 | 177/181 / 128/132 |
| r2-subpixel1 | 53.919/53.778 | 52.088/51.763 | 0.126/0.126 | 11.385/9.756 | 165/169 / 119/123 |
| r4-subpixel0 | 51.871/51.776 | 45.259/45.041 | 0.126/0.126 | 10.011/8.565 | 136/140 / 95/99 |
| r4-subpixel1 | 51.463/51.353 | 45.217/44.949 | 0.126/0.126 | 9.843/8.446 | 126/130 / 88/92 |
| r8-subpixel0 | 48.581/48.472 | 38.140/37.966 | 0.126/0.126 | 8.104/6.973 | 85/89 / 66/70 |
| r8-subpixel1 | 48.325/48.209 | 38.128/37.886 | 0.126/0.126 | 7.945/6.887 | 78/82 / 63/67 |

Thin completeness remains 4/3164 on each side in every case; exact implementation
agreement does not establish useful world quality or recovery of missing priors.

The unchanged F1 independent full-resolution SGM binary was also run on these
exact originals with integer/packed, preprocessing, LR metadata and synchronization
checks passing. `A/root-full-sgm-analytic-baseline-v1/quality.json` evaluates its
outputs against the same truth and Census-support denominator. Its supplementary
binding records the original F1 binary lineage and explicitly lists subsequent
CMake/header/test changes; it does not claim those historical source hashes still
match the current tree. Continuous compound SGM arithmetic remains unqualified.

| Correct metric completeness, L/R % | All | Near | Thin | Edges |
|---|---|---|---|---|
| Full-resolution independent SGM | 95.260/95.279 | 99.041/98.537 | 60.461/61.188 | 65.016/63.606 |
| Guided r2, one unrestricted CPU recovery round | 75.574/75.382 | 52.674/52.394 | 27.212/27.497 | 23.372/21.873 |

The recovery row is an offline CPU-oracle experiment, not a native implementation
or GPU performance estimate (`artifacts/tmp-r0/recovery-diagnostic/report.json`,
SHA256 `eeddca092fcdcd7aa917b27101689dbceae1a201441b87c9e803bf2902711b44`).
Selection uses observed initial validity/provisional LR, with no truth-derived ROI.
Supported candidate evaluations increase from 6,203,250 to 118,390,534; all-region
false LR accepts rise from 1,759/1,806 to 7,640/7,617. Automatic recovery also leaves
initial LR-agreeing false matches untouched. This is insufficient evidence for a
quality-preserving replacement; edge-aware support and/or regularized refinement
need controlled experiments before promotion.

All timing evidence is provisional from an incomplete interleaved matrix.
`A/root-matched-timings-v1/admission-rejected.json` rejected admission;
`A/root-matched-timings-v2/matrix.json` is `rejected_or_incomplete`, with foreign
GPU work at before/after admission checks. The following are exploratory means
from its six accepted runs with clean foreign-compute snapshots before/after
(desktop GPU clients remained). Each has 100 frames / 20 warmups, two slots,
wave32, resident output and no View. There are no ≥3 matched cycles.

| Variant | Radius/subpixel | Combined queue mean ms | Refinement mean ms | Host wall mean ms |
|---|---|---|---|---|
| coarse_alone | 4/0 | 5.988 | 0.000 | 6.193 |
| coarse_refine | 2/0 | 7.994 | 2.039 | 8.213 |
| coarse_refine | 4/0 | 8.201 | 2.125 | 8.430 |
| coarse_refine | 8/0 | 8.356 | 2.266 | 8.635 |
| coarse_refine | 2/1 | 7.976 | 2.033 | 8.189 |
| coarse_refine | 4/1 | 8.198 | 2.136 | 8.438 |

Reproduction templates (expand `A`, `F`, `R`, `S`, `V`; commands are recorded
verbatim in each comparison and the matrix; do not treat these as admission):

```sh
build/daxa/daxa_refinement_fixture --coarse-manifest F/half/manifest.json --target-manifest F/full/manifest.json --output A/check/native.json --dump-prefix A/check/planes/frame --radius R --subpixel S --slots 2 --wave-size 32 --validation
build/daxa/daxa_refinement_benchmark --coarse-manifest F/half/manifest.json --target-manifest F/full/manifest.json --output A/check/timing.json --radius R --subpixel S --slots 2 --wave-size 32 --frames 100 --warmup 20 --variant V
```

Recovery/ROI, View, held-out/physical quality and live scheduling/latency gates
remain open. Complete admitted matched timing cycles are still required; these
means cannot support a speedup, throughput or 60 FPS qualification.
