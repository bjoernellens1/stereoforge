# Daxa + Slang foundation — 2026-10-27

Task 1 foundation qualification passes on the tested AMD device: Daxa CTest
**3/3**, the capability probe and the 300-execution primitive benchmark exit zero
with Khronos core validation loaded at instance and device level and no warning
or error diagnostics across both output streams through explicit device close.
The full Python suite passes **319 tests** (5.05 s); the existing standalone
Slang RHI CTest remains **5/5**, alongside Daxa **3/3**.
This qualifies the foundation primitive only. The full compositor/stereo pipeline,
synchronization validation, other devices and sustained 60 Hz remain open.

## Pinned independent build

The opt-in `daxa` preset builds Daxa tag `3.6`, source
`00761e80c1dd6d1972d5f4569ce49620a7b4d384`, with VMA `v3.1.0`, source
`009ecd192c1289c7529bff248a16cfe896254816`. The official Linux x86_64 Slang
2026.12 SDK compiles the foundation shader to SPIR-V offline. The runtime uses
Daxa's native SPIR-V pipeline API; it does not use Daxa's Windows-oriented Slang
bootstrap or link Python, SGL, Slang RHI or the Slang compiler. The compiler SDK
is a build dependency, separate from runtime linkage.

| Download | SHA256 |
|---|---|
| Daxa source archive | `0005d3601b4b3b01f7bdc0a1cacf957aa56dfdb70686fa84043ab79528672679` |
| VMA source archive | `5ed5125086a92666f1698df907a29f54f11197c382996094b556a1b22186ecaf` |
| Linux Slang 2026.12 SDK | `2409ecc7b6710783e05fd59792558c275c637e3e3c37fb5054fcfc858e570d82` |

Daxa's task graph and memory utilities are enabled. Pipeline managers, SPIR-V
validation utility, tests/tools, ImGui/ImPlot and FidelityFX integrations are
disabled. Source patches are retained under `cmake/patches`: the Linux headless
compatibility patch corrects the `u64` type and optional ImGui guards; the
external-resource-view patch corrects three returned view fields. The external-view regression has a clean unpatched runtime failure and patched
pass. Upstream runtime assertions remain
enabled in the optimized build.

Daxa and VMA actual source licenses are MIT; exact notices are retained. The
Slang SDK LICENSE matches the retained Apache-2.0 WITH LLVM-exception notice.
The selected-source review identifies no source-publication blocker. System
Vulkan/window-library and SDK binary redistribution inventory remains a separate
pending gate; see [license evaluation](license-evaluation.md).

## Capability probe

`artifacts/daxa-foundation-20261027/probe-final.json` records AMD Radeon 8060S Graphics
(RADV STRIX_HALO), Mesa 26.2.3. All 33 features required by the selected Daxa
configuration are available. Default subgroup size is 64, with supported sizes
32–64. Maximum compute invocations are 1024, shared memory is 65536 bytes and push
constants are limited to 256 bytes.
The readback allocation reports memory-type bits `4`, selecting explicit
host-visible coherent memory; CPU access waits for final completion. The probe
reports zero validation errors and zero warnings. This is a capability/core
validation result for this device and driver, not cross-vendor qualification.

## Regression and timing boundary

The clean unpatched run exits 1 with malformed external-view identity/index/
generation (`clean-unpatched-red.json`); the patched run passes. The initial API
exception was independently traced in `unpatched-throw-backtrace.txt`. The two
versioned patches cover Linux headless compilation and the three registration
return fields only. First-allocation, teardown masking, ignored completion signal
and command-pool exhaustion failure artifacts are retained rather than replaced.

The correctness fixture processes three fresh GPU frames with changing IDs 1, 2,
3 through producer and consumer graphs without an intermediate CPU wait. It checks
view identity, range and stale-generation handling using supported APIs, then
reads a final 12-byte counter after completion. Pinned `TaskGraph::submit` ignores
extra semaphore signals in `TaskSubmitInfo`; completion therefore uses Daxa's
supported `latest_queue_submit_index(QUEUE_MAIN)` / `wait_on_submit` API. No
scheduler patch was added. Explicit close destroys the device while the validation
messenger/counters remain alive, before the final warning/error check and success
publication.

The separate microbenchmark executes 300 tiny task graphs after three warmup
frames. Per-execute poll-only garbage collection is included in CPU enqueue timing.
A frozen benchmark-only batch limit of 16 avoids the pinned 128-command-pool cap;
its waits and extra reclamation cost are reported separately. The three-frame
correctness fixture has no such intermediate wait.

| Measurement | Final result |
|---|---:|
| CPU enqueue p50 / p95, including per-execute GC | 33.192 / 43.432 microseconds |
| Benchmark backpressure waits | 18; 0.328187 ms total |
| Extra batch reclamation CPU time | 0.113255 ms |
| Final completion wait | 0.027672 ms |
| Enqueue through final completion | 10.9643 ms |
| Final readback | 12 bytes |
| GPU timestamp timing | Unavailable; not instrumented |

Daxa allocates internal command-recorder objects per execution and reclaims them
through garbage collection; this evidence makes no allocation-free execution claim.
These tiny-graph measurements cover CPU submission/completion timing only.
These measurements are neither stereo latency nor pipeline FPS, and establish no
Daxa speedup against the Slang RHI baseline. Final runs used empty `DISPLAY`,
`WAYLAND_DISPLAY` and `PYTHONPATH`, with `PYTHONHOME=/nonexistent`.
`final-report.json` records commands, source hashes, loaded-layer evidence and
zero diagnostic findings; `ctest-final.stdout/.stderr`,
`benchmark-final.stdout/.stderr` and `probe-final.json` retain the raw outputs in
`artifacts/daxa-foundation-20261027/`.

The final report binds the complete source inventory. Built artifact SHA256 values:

After that run, the two dependency patches were regenerated with empty blank
context lines to pass Git's whitespace check. `post-format-equivalence.json`
records the final patch hashes, identical applied TaskGraph source and identical
binary/SPIR-V hashes, plus a fresh validation-enabled CTest pass (3/3).
The original timing report remains unchanged.

| Artifact | SHA256 |
|---|---|
| `build/daxa/daxa_foundation_contracts` | `823de2ac1826ab7b19a92ee58d28725665c8dffd902fbf3b9c9b8a2d1a72b36b` |
| `build/daxa/daxa_probe` | `0dd26421d0e5cb4ea51faf9ccda5203669ba16a7d32ac9abf34f8a414f0b9e35` |
| `build/daxa/daxa-foundation.spv` | `1a0ed4ee9d4ea851141d9f821d143c989c1e789a93874552ac0d929709b88b49` |

## Reproduction

```bash
cmake --preset daxa
cmake --build --preset daxa
mkdir -p artifacts/daxa-foundation-20261027
build/daxa/daxa_probe --validation --output artifacts/daxa-foundation-20261027/probe.json
ctest --preset daxa --output-on-failure
build/daxa/daxa_foundation_contracts --validation --benchmark
```

Install the Khronos validation layer separately, or use the locally extracted
layer described in [validation evidence](vulkan-validation-evidence.md):

```bash
export VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d"
export LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Set these variables before the probe and CTest commands when using that local
extraction. A separate negative probe with `VK_LAYER_PATH=build/daxa/empty-vulkan-layers`
and empty `VK_INSTANCE_LAYERS` exits 1 and writes `status: blocked` with the
Khronos-unavailable error. Retained evidence is
`artifacts/daxa-foundation-20261027/missing-validation-probe.json` and
`missing-validation.stdout/.stderr`. Missing validation is rejected rather than
silently counted as a pass.
