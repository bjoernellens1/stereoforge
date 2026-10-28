# Experimental SinterGPU reduction migration

`VulkanStereo` can opt in to the SinterGPU 0.2.11 `common.block_reduce` helper
by setting `SINTERGPU_SHARED_REDUCTION=1` before engine construction and
installing the `sintergpu` extra. The extra pins the immutable release wheel by
SHA-256. The runtime also checks the expected package version and released
helper digest before creating a device, then composes the exact helper bytes
into the shader source. `shader_source_identity` records the composed source
hash and helper digest. The original reduction remains the default.

The shared path preserves the 256-thread uint-min tree and caller-owned shared
storage and barriers. It does not add a device, allocation, submission or
readback. Only the Python `VulkanStereo` SGM aggregate path is changed; native
Daxa implementations are separate. CPU tree/model and dual-compiler resource
reflection tests do not establish GPU output parity, timing or application
quality. The old 0.1.0 migration PR has historical Radeon fixture evidence,
but this changed consumer/release compilation needs fresh source-matched GPU
and controlled no-regression qualification before adoption.

A dedicated GPU runner must be independently reviewed for device identity,
exclusive locking, synchronization and teardown before any GPU parity run.

Standalone Slang 2026.12/2026.19 compile and reflect all five entry points
for both sources when entry points are discovered automatically. Explicit
`-entry select` does not resolve on the unchanged baseline in either compiler;
automatic discovery preserves the `select` entry name without altering source.
SlangPy runtime execution of the full pipeline still needs qualification.

## Proposed 4070 qualification protocol (execution blocked)

The target is the canonical `cps-wkstn-nvidia4070.local` RTX 4070 Ti. Use the
cross-project nonblocking `/tmp/sintergpu-gpu.lock` for the entire run, and
coordinate with other GPU users; this advisory lock cannot exclude processes
that do not honor it. Capture the actual Vulkan physical-device UUID, PCI bus
identity, vendor/device IDs, driver/ICD and SlangPy compiler identity before
and after the run. Record the exact source commit, dirty status, composed shader
hash, released wheel/helper hashes, fixture hashes, environment and executable
versions. Abort on an identity mismatch, changing source or missing evidence.

A reviewed executor must select and verify that physical device before any
work, hold the lock through synchronization and teardown, and abort cleanly on
failure. `VulkanStereo` defaults to constructing a device and exposes only an adapter
name; this does not establish physical identity or controlled device selection.
Its optional caller-owned `device` parameter permits the same device checked by
a future admission layer to be used for shader compilation and dispatch; the
caller owns synchronization and teardown and must destroy engine resources
before closing the device. For a borrowed device, `debug` does not configure
its validation layers; the caller must configure them when creating it. Mock tests verify object reuse only. The collector `benchmarks/sintergpu_qualification_runner.py` therefore
blocks its command-line hardware entry point before GPU initialization. Its
injected CPU-only callback tests establish collector behavior only; they are not
hardware evidence. No name-only or `nvidia-smi`-only inference is sufficient.

After that blocker is resolved and independently reviewed, test baseline and
candidate on identical deterministic stereo pairs including non-multiple-of-256
and disparity counts 1, 127, 128, 255 and 256; 4 and 8 paths; odd and border
sizes, bounded and unbounded disparity, flat and textured input. Compare
byte-exact left and right census descriptors and their validity masks, cost,
aggregate, disparity, confidence, raw uint32 validity and boolean validity results,
including old-before and old-after outputs. Save inputs and outputs or their
content-addressed hashes and retain mismatch witnesses. This verifies these
fixtures only and does not establish trajectory or application quality.

For each fixture, warm both paths, then collect at least three independently
controlled baseline/candidate/baseline runs with enough per-run samples for
stable medians and p95; retain every raw sample, warmup policy, run order and
clock/thermal/load metadata. Record GPU timestamp stage duration, complete
GPU pipeline time, and host transfer-inclusive latency separately. Do not sum
stage timestamps and present them as a measured pipeline wall time. Readback is
an explicit output/debug boundary and its cost is reported separately. Compare
paired medians and p95 across runs and use a predeclared statistical method and
confidence interval for improvement; reject performance adoption on any output
mismatch, a greater than 5% p95 regression, or evidence of a median regression.
Fusion adoption additionally requires the existing three-run threshold of at
least 5% lower median pipeline time and a statistically positive improvement.
The present collector records raw timing triples and fails on bitwise output
mismatches. If a witness directory is supplied, it saves the first mismatched
array pair and case/round identity in a uniquely named NPZ file; a future GPU
executor must require this directory and retain input and source identity too.
`VulkanStereo.result(debug=True, debug_census=True)` explicitly reads back
right census and both census validity masks in addition to the existing debug
outputs. This opt-in incurs three extra host readbacks; the default is unchanged.
The future executor must request these fields before claiming this coverage. The collector does not calculate a performance verdict
or establish statistical power. GPU qualification and optimized-default selection remain pending.

The CPU-only `sintergpu_device_receipt.py` scaffold checks the instrumented
runtime's device-owned receipt against that same device's three native handles
and an expected UUID, while holding the canonical nonblocking lock through idle
and close. It checks the lock inode and Linux fdinfo before and after use, retains
the device and lock on idle, lock-check or close failure, and preserves multiple
errors. These checks cannot prevent an uncooperative process replacing the lock
path temporarily, and a failed device factory may hide partially created resources. This receipt is not available in the ordinary released SlangPy API.
The scaffold does not authenticate the runtime, provider, receipt, or loader;
it is not wired to a GPU entry point. Independent provenance and live provider
verification are required before it can form part of hardware admission.

The fixture adapter `benchmarks/sintergpu_fixture_execution.py` connects the
collector to `VulkanStereo.enqueue` and `result` and requests all ten output
arrays, including raw uint32 validity as well as its boolean view. Its pipeline timing spans the first-to-last GPU stage timestamps;
transfer-inclusive host timing includes upload, enqueue, synchronization and
all debug readbacks, and readback time is retained separately. The first shape
can include allocation and compilation effects, so prewarm each shape and variant
and record the warmup policy; timestamp precision and GPU behavior remain unqualified. This adapter has
only been tested with mock engines. It does not establish runtime provenance,
physical device identity, validation layers, timestamp accuracy or GPU parity;
it has no standalone hardware entry point. Fixture construction, warmups,
engine lifetime, provider verification and qualification remain open.
