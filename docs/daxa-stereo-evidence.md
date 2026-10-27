# Native stereo optimization evidence — 2026-10-27

The resident Daxa SGM experiments preserve frozen RHI output bytes and reduce
one-reference GPU time from about 43.5 ms to 15.3 ms. This does not qualify the
complete pipeline for the 16.67 ms frame budget. These are exploratory measurements of
one stereo reference on a retained synthetic luma pair. They exclude acquisition,
rectification, the other reference, LR consistency, view synthesis and output
transfers. CPU floating-reference qualification remains open.

## Measured operating point

AMD Radeon 8060S/RADV, 1280×720, D128, four paths, P1=8/P2=32, subpixel enabled,
one retained output slot, 30 samples after three warmups:

| Implementation | GPU median | GPU p95 | Submit/wait/query wall median |
|---|---:|---:|---:|
| Dense workgroup baseline | 43.290 ms | 43.993 ms | 44.106 ms |
| Register recurrence, required wave64 | 39.669 ms | 40.533 ms | 40.682 ms |
| Register recurrence, required wave32 | 43.784 ms | 44.155 ms | 44.487 ms |

The wave64 case spends approximately 5.20 ms generating costs/clearing sums,
6.88–7.72 ms on each aggregation direction, and 5.50 ms selecting disparity.
Its CPU submission median is 0.185 ms and timestamp-query median 0.026 ms.
Requested retained buffers total 984,268,976 bytes; actual allocator/process peak
memory has not been measured. Each dense cost or sum plane is 471,859,200 bytes.

Frozen RHI took 2392.97 ms for 30 upload/compute/readback iterations, or 79.77 ms
per iteration on average. That transfer-inclusive aggregate is a different scope
from resident GPU timing. It is not evidence that language or backend alone
accounts for the difference. External shared-system activity was not locked.

The authoritative local reports are in
`artifacts/daxa-stereo-20261027/720p-d128-p4-exploratory` and
`720p-d128-p4-wave32-exploratory`. They bind shared input bytes, compiler, shader,
binary and dependency identities. Both runs compare all three final planes
byte for byte against frozen RHI. The separate small-fixture comparison also
checks dense costs/sums between Daxa variants. Diagnostic exports occur after
timing. Archived A binaries, SPIR-V and shader sources are retained locally in
`artifacts/daxa-stereo-20261027/archive-a`; binaries are not distributed here.

## Correctness limits and next intervention

The initial GPU scalar test failed CPU-RTE confidence bit equality: GPU 7/63 was
one ULP below the CPU result. A separate extraction bug changed 15 small-fixture
disparities by one ULP. Preserving the frozen RHI conditional update structure
restored exact RHI/Daxa output equality; that extraction change is independently
reviewed. Original failures remain in the local artifacts.

The confidence issue is separate. Vulkan permits bounded division error and
arithmetic contraction; CPU RTE evaluation does not define every permitted
execution. Integer Census, costs, SGM sums, masks and same-profile GPU/RHI byte
gates remain exact. An operation-level scalar floating contract is still needed;
there is no blanket floating tolerance or completed quality qualification.
[Vulkan shader precision rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation).

## Packed penalties and arithmetic specialization

The packed experiment computes matching costs on demand and stores only
`P = sum_paths(L - C)`. Every contribution lies in `[0,P2]`. The explicit
`paths*P2` bound selects lossless 8-bit fields when at most 255, or 16-bit fields
when at most 65535; larger bounds are rejected. Each lane owns complete packed
words. Selection reconstructs exactly `paths*C + P`. This changes storage, not
the SGM objective or quality settings. Dense diagnostic volumes reconstructed
from GPU Census and penalties are explicitly marked as reconstructed.

One matched 720p/D128/four-path/wave32 batch, 30 samples after three warmups:

| Implementation | GPU median | GPU p95 | Submit/wait/query wall median | Requested buffers |
|---|---:|---:|---:|---:|
| Dense baseline | 43.497 ms | 44.492 ms | 44.515 ms | 984,268,984 B |
| Packed penalties | 27.408 ms | 28.585 ms | 27.943 ms | 158,515,384 B |
| Packed with hoisted/static address arithmetic | 15.430 ms | 15.645 ms | 15.714 ms | 158,515,384 B |

The last variant passes 45 core plus 45 synchronization-validation cases,
including odd disparity counts, both subgroup sizes, four/eight paths and
packing boundaries. GPU Census, costs, reconstructed sums, masks and **actual
packed words** match independent scalar results. The full 720p comparison also
passes those integer checks and exact frozen-RHI final output bytes. The actual
15-SPIRV comparison and source identities are retained in
`artifacts/daxa-stereo-20261027/specialized-720p-wave32-performance`. Wave64 is
close at 15.578 ms GPU / 15.858 ms wall; these runs do not establish a robust
subgroup winner. External shared-system activity remains uncontrolled.

RADV compiler diagnostics report 96 VGPRs, zero spills/scratch and a 16-wave
compiler ceiling for specialized 8-bit aggregation. Selection reports 24 VGPRs,
zero spills/scratch and 3072 bytes LDS. These are compiled resource statistics,
not measured runtime occupancy, memory traffic or allocator peak. The raw dump
and stage mapping are retained locally as `specialized-isa.stderr` and
`isa-summary.json` under the same artifact root.

An independent exact-rational check accepts all 1620 valid confidence divisions
and 428 exact invalid zeros on each of four small-fixture variants. The original
CPU compound-float bit-equality gate remains unqualified. Single-reference
timing below 16.67 ms does not qualify the complete dual-reference/capture/view
pipeline for 60 Hz. No production adoption follows from these benchmarks alone.

## Fixed-D128 experiment: negative result

The next variant reduces the source recurrence state from eight to four
components and requires D128 explicitly. Matched runs use the same 30 samples,
three warmups, retained inputs and quality settings:

| Required subgroup | Specialized GPU median | D128-only GPU median |
|---|---:|---:|
| 32 | 15.228 ms | 16.023 ms |
| 64 | 15.607 ms | 16.528 ms |

Both variants pass exact frozen-RHI final output comparisons and independent
integer/actual-penalty checks at 720p. An additional 14 core plus 14
synchronization-validation cases pass with zero diagnostics through teardown.
The D128-only variant is retained as an explicit experiment, not adopted as the
preferred implementation. Reports are under `d128-720p-wave32-performance`,
`d128-720p-wave64-performance`, and `d128-validation` at the same artifact root.

Actual RADV/ACO diagnostics show both aggregation variants still allocate
96 VGPRs with a 16-wave compiler ceiling and no spills or scratch. D128 shrinks
static code from 3972 to 1756 bytes but leaves the high-index linear reduction
temporaries at v94/v95. Smaller source state therefore did not lower the final
register allocation. Compiler estimates and static instruction counts do not
establish runtime occupancy or a cause for the slowdown. The reused selection
shader has identical ISA in both variants, including three LDS barriers and
3072 bytes LDS; these facts do not prove barriers dominate its roughly 3.1 ms.

The optional native target remains disabled by default. Its full GPU CTest
scalar compound-float gate is still unqualified; successful explicit integer
checks and same-profile GPU byte comparisons must not be reported as that gate
passing. Capture, calibrated preprocessing, the second reference, consistency,
metadata and GPU view synthesis still need shared-context integration and
transfer-inclusive sustained validation before a 60 Hz claim.


## Quad selector: subgroup-specific benefit

The `packed-quad` experiment reuses specialized aggregation and penalty storage.
Four actual subgroup lanes cooperate on each pixel; register top-two selection
uses uniform shuffles. The compiled selectors have 64 threads, no workgroup
storage and no barriers. All seventeen earlier SPIR-V modules remain identical.

Matched 30-sample/three-warmup 720p/D128/four-path results:

| Subgroup | Specialized GPU / selection median | Quad GPU / selection median |
|---|---:|---:|
| 32 | 15.302 / 3.168 ms | 15.109 / 2.427 ms |
| 64 | 15.684 / 3.128 ms | 16.203 / 3.602 ms |

Both batches pass exact frozen-RHI outputs and integer/direct-penalty gates.
Separate 44 core/synchronization cases cover D1..256, odd counts, both subgroup
sizes, four/eight paths and packing bounds; all pass with zero diagnostics
through teardown. The full CPU suite passes 583 tests. Artifacts are in
`quad-validation`, `quad-small-parity` and `quad-720p-wave{32,64}-performance`.

A follow-up wave32 C/quad/quad/C ordering uses 100 samples after 20 warmups,
identical explicit retained inputs and no full output exports. Quad selection
medians are 2.263 and 2.350 ms, versus 3.088 and 3.236 ms for C. Total GPU medians
are 14.977/15.191 ms for quad and 15.382/16.669 ms for C. Unchanged aggregation
stages also vary, so these runs do not isolate a stable whole-pipeline gain or
justify summing stage medians. See `quad-wave32-abba/summary.json` for identities
and commands. External shared-system activity is not globally locked.

Quad remains an explicit experiment with a wave32 stage benefit; wave64 is a
negative result and no automatic selection changes. Aggregation still dominates.
All timings exclude the second reference, consistency, acquisition, calibrated
preprocessing and view synthesis. The scalar compound-float gate remains open.
