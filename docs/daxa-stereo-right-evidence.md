# Explicit right-reference Vulkan stereo

This experimental native extension adds true right-reference disparity to the
existing packed-specialized Daxa implementation. Positive disparity always means
`d = x_left - x_right`: left matching compares `L(x)` with `R(x-d)`, while right
matching compares `R(x)` with `L(x+d)`. Merely swapping ordinary inputs is wrong.
No runtime mirror/unmirror image passes are added.

## Interface and current boundary

`DaxaStereoReference::{Left,Right}` is appended to existing constructor and
validation arguments with Left as the default. Engine, output lease and explicit
readback expose the reference. The benchmark accepts `--reference left|right`
and reports both the reference and positive-disparity convention. Census planes
retain their ordinary physical left/right image ordering, independent of the
chosen matching reference. Source metadata and completed-input retirement retain
the existing owner and completion contracts.

Right currently supports only `PackedSpecialized`; other variants and right
metric calibration are explicitly rejected before allocation. Four/eight paths,
wave32/wave64 and existing exact 8/16-bit packing bounds remain supported.
Right disparity is a source-grid output, not virtual-view depth. Dual-reference
scheduling, LR consistency, metric observations/provenance and wide-view intake
remain subsequent integration gates.

The same aggregation and selection bodies specialize a compile-time direction
parameter. Recurrence, packing, tie selection, subpixel and confidence operations
are shared. The parameter block remains 168 bytes and push constants 16 bytes;
there is no runtime direction field in the shader. Four right entry points are
added. All 21 previous stereo SPIR-V files and both preprocessing SPIR-V files
remain byte-identical to the preceding `6ac9a12` baseline.

## Independent checks

The CPU oracle literally mirrors the right and left images, runs the existing
left-reference scalar matcher, then unmirrors its costs, sums, packed words and
final output rows. Those results match direct-right scalar addressing. Canonical
Census descriptors are checked separately on ordinary images because reflection
permutes their bits. CPU cases cover odd widths, D1 through D256 boundary/tail
cases, four/eight paths, both subgroup models, packing limits, known shifts, ties,
unsupported borders and explicit setup rejection.

Root GPU validation completed on AMD Radeon 8060S / RADV Strix Halo, Mesa 26.2.3,
using pinned Daxa and official Linux Slang 2026.12:

- Four native right-contract processes passed: core/synchronization validation
  at wave32/wave64, with exact Census, integer costs/sums, actual packed words,
  validity, output-domain checks and dynamic GPU-input/retained-output lifetimes.
- The independent frozen-RHI oracle receives mirrored right/left input bytes.
  Only its final output rows are unmirrored. Disparity, confidence and validity
  then match native right output bytes exactly on the small fixture, odd-width
  eight-path fixture and full 1280×720/D128/four-path fixture. Full-size parity
  passed under both core and synchronization validation.
- Existing left variants still match frozen RHI and exact integer/packed checks
  under synchronization validation. Native preprocessing contracts also pass
  again after the API extension.
- These validation runs captured no warnings/errors through teardown. Harnesses
  bind original and mirrored input bytes, source, compiler, dependencies, binary,
  all 25 stereo SPIR-V files and report bytes before and after execution.
- The full Python suite passed **730 tests**, and both native CPU contracts passed.

Native right tests do not apply an arbitrary CPU float tolerance. They explicitly
leave compound CPU floating-point qualification unchecked; the independent GPU
float gate is exact frozen-RHI byte parity. Existing strict left scalar-float
tests retain their original gate and remain unqualified. Matching GPU outputs
does not close that scalar contract.

## Measured separate-reference cost

Repeated retained luma, 1280×720, D128, four paths, subpixel, P1=8/P2=32,
one slot, 30 measured iterations after three warmups, validation disabled:

| Right-reference measurement | wave32 p50 / p95 | wave64 p50 / p95 |
|---|---:|---:|
| GPU graph | 15.113 / 15.207 ms | 15.653 / 16.731 ms |
| Submit/wait/query host wall | 15.460 / 15.796 ms | 16.022 / 17.066 ms |

Each requests 158,515,384 bytes of retained buffers, excluding allocator/driver
peaks and diagnostic storage. Source upload, capture, rectification, the other
reference, LR consistency, metric conversion and RGB synthesis are outside these
samples. Full diagnostic integer/P checks and exact RHI comparisons also pass
for both measured runs, after timing. The frozen-RHI timing is transfer-inclusive
and has no separate warmup; it is not a comparable end-to-end speed measure.
These separate-reference measurements are not a measured dual-reference pipeline
latency or a 60 Hz result. Shared-system GPU activity was not globally locked.

## Reproduction

Enable and build the optional native stereo target as in the
[left stereo evidence](daxa-stereo-evidence.md). With Khronos validation available:

```bash
ctest --test-dir build/daxa \
  -R '^daxa_stereo_packed-specialized-right(32|64)_contracts$' \
  --output-on-failure
PYTHONPATH=python .venv/bin/python benchmarks/compare_daxa_stereo_right.py \
  --compiler build/standalone/_deps/sf_slang_sdk-src/bin/slangc \
  --output artifacts/new-right-check --width 64 --height 32 \
  --disparities 16 --paths 4 --wave-size 32 --frames 1 --warmup 0 \
  --validation-only --validation --dump-intermediates
```

Use the configured compiler path if it differs and configure validation paths
from [validation evidence](vulkan-validation-evidence.md). Set
`VK_LAYER_VALIDATE_SYNC=1` for synchronization validation. For full-size timing,
use a fresh output directory, width 1280/height 720/D128, frames 30/warmup 3, and
omit validation-only/validation/intermediate dumps. The harness keeps final
diagnostic checks outside GPU samples.

Local ignored evidence: `artifacts/daxa-stereo-right-20261027/` contains native
and parity summaries, `small-core`, both `720p-w*-performance` runs,
`left-all-controls-sync` and the preprocessing regression logs. These are
qualification artifacts, not a binary release or bundled camera recording.
