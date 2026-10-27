# Curated metadata variants and readback evidence — 2026-10-27

A bounded Daxa metadata comparison now preserves exact signed 64-bit observation
IDs and ages while exposing a large host-readback cost on this AMD target.
Cached coherent readback improves the measured round trip; no fused kernel is
adopted. This is shared-operation/variant preparation, not full Task 3 completion
or stereo/pipeline qualification.

## Contract and identity

`configs/kernel-variants.json` declares grid, units, precision, invalid borders,
layout, disjoint aliasing and retained ownership. Two real 64×1×1 entrypoint
variants gather target-grid metadata: fused ID+age and independent split ID/age.
Both use the same pure operation. IDs stay signed int64 (−1 invalid), ages stay
nanoseconds (0 invalid); transport uses exact low/high uint32 words. The fixture
includes high words, invalid flags, sentinel and out-of-range owners, and tail
lanes. Complete ID/age bytes are compared with a scalar oracle after each static
resident/transfer batch and across both variants. Repeated samples of this fixture
are not fresh camera observations.

The fail-closed identity binds physical GPU UUID, driver, Daxa commit plus actual
patch/applied-source/loaded-dependency and benchmark bytes, Slang version/compiler
bytes, selected compiled SPIR-V bytes, entrypoint/workgroup/compiler flags,
source/target dimensions, explicit D0/paths0 for this operation, precision, frozen
quality profile and queried subgroup. Host readback policy, actual memory type
and flags are also bound. Unknown variants, unsupported subgroups, incomplete or
changed keys, inconsistent memory choices and stale approvals are rejected.
The actual Daxa source directory is embedded in the executable; an unrelated
source override is rejected. Adoption requires manual review bound to a validated
measurement artifact; `approved` remains null.

## Primary controlled result

Target: AMD Radeon 8060S, RADV/Mesa 26.2.3, official Linux Slang 2026.12, pinned
Daxa `00761e80c1dd6d1972d5f4569ce49620a7b4d384` plus recorded patches. The primary
artifact is `artifacts/daxa-variants-20261027/readback-abba-summary.json` and its
four `abba-*` comparison directories: A/B/B/A policy order, two variants per case,
30 samples and 3 warmups each, 1280×720 metadata pixels. All eight cases pass
full byte parity. No other root build/GPU test ran during this sequence; external
shared-system load was not globally locked. Earlier screening that could overlap
CPU builds is excluded from the primary result.

Ranges below are **case medians**, spanning both kernels and repetitions:

| Readback policy | Actual type / Vulkan flags | CPU readback copy (ms) | Complete round trip (ms) | Compute GPU (ms) |
|---|---|---:|---:|---:|
| first-coherent | 2 / 6: host-visible, coherent | 67.413–70.446 | 69.432–72.420 | 0.167–0.173 |
| cached-coherent | 5 / 14: host-visible, coherent, cached | 0.503–0.540 | 1.719–1.775 | 0.170–0.177 |

Only the diagnostic readback allocation policy varies. Upload/parameter policies,
shader/fixture bytes and numerical gates remain fixed. Cached mode requires a
compatible host-visible/coherent/cached type and fails if unavailable. The
phase measurements localize the dominant improvement to CPU mapped readback
copying; compute GPU timing is essentially unchanged. This demonstrates a local
readback policy benefit, not zero-copy UMA or a whole-renderer speedup. Fused/split
GPU differences are small and mixed; neither kernel earns a confident adoption
ruling. The default first-coherent baseline remains available, and qualified
foundation/DaxaView allocation defaults are unchanged.

GPU timestamps bracket actual first/last scheduled compute callbacks, excluding
uploads/readback. Complete wall samples include CPU input staging, enqueue and
initial garbage collection, completion wait and post-wait collection, CPU output
copy, timestamp retrieval/validation. Per-row phases conserve the total; sums of
separate phase medians are not used to reconstruct median latency. Setup and file
dumps are separate. Requested application buffers total 73,728,056 bytes; allocator,
Daxa, Vulkan and process memory/growth are unmeasured. Register/LDS/spill/traffic
hardware counters are unavailable, represented as null; the logical load-byte
model is not a hardware counter.

## Validation, failures and reproduction

Both readback policies pass separate core-validation comparisons in
`readback-{first-coherent,cached-coherent}-core`. The earlier metadata baseline
also passes `core-final` and `sync-final`. Renewed synchronization validation also passes both policies in
`readback-{first-coherent,cached-coherent}-sync-final`: each fused/split child
reports exact parity and zero errors/warnings through teardown. Validation counts
include resource/device teardown. The original `core-first` setup assertion
(duplicate external names) and `core-second` loader-warning rejection are retained
as failures; unique-name guards fixed setup, and a clean environment resolved the
warning without filtering diagnostics.

```sh
cmake --preset daxa
cmake --build --preset daxa
ctest --test-dir build/daxa -R daxa_variant_cpu --output-on-failure
python benchmarks/compare_kernel_variants.py \
  --compiler build/standalone/_deps/sf_slang_sdk-src/bin/slangc \
  --readback-memory cached-coherent --validation \
  --output artifacts/NEW-variant-core
```

Use fresh directories, the configured pinned compiler, and an installed Khronos
layer (the local qualified layer may require `VK_LAYER_PATH`/library paths).
For a separate synchronization run add `VK_LAYER_VALIDATE_SYNC=1`. For timing,
remove validation/debug environment settings and omit `--validation`; run A/B/B/A
sequentially with `first-coherent` and `cached-coherent`. The command is bounded to
at most 1080p, 30–300 samples and 3–30 warmups. Actual hashes and identities are in
each comparison/native report. CPU contracts/guards passed 59 cases before root
hardware measurement. Final root regression passes 433 Python tests without skips;
fresh joint-capture and Daxa CPU CTests each pass 2/2. The Python command and log
are recorded in [native-capture-evidence.md](native-capture-evidence.md).

Compatible 8×8/16×8/16×16/32×4 candidates remain pending actual Task 4 preprocessing
kernels. Full held-out stereo/geometric quality, integrated GPU stereo and fresh
720p-per-eye/60-Hz operation remain open. Source additions follow repository
Apache-2.0 licensing; existing dependency notices remain applicable and binary
redistribution is a separate gate.
