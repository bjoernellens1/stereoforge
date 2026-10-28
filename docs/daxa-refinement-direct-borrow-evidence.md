# Direct full-frame borrow: initial evidence

`DaxaRefinementConfig::borrow_full_frame` selects an experimental full-frame path;
the six-copy reuse path remains the default. The refinement slot retains both
the coarse observation and the full-resolution preprocessing frame until the
consumer completes and all output leases are released. Admission requires the
same capture lease, device context and exact setup-bound calibration owner.
Census reads the full frame's separate left and right luma and support buffers
on the GPU. The graph declares those reads and an explicit producer
compute-write to consumer compute-read barrier. Observation RGB views borrow
the corresponding full-frame RGB buffers.

A pair of separate eye buffers cannot be represented by the existing contiguous
`buffer(Luma)` or `buffer(Support)` API, so these calls reject in borrow mode.
The explicit diagnostic readback reconstructs the two logical `[eye][y][x]`
planes after completion. Other output planes retain their existing contracts.
The first stage has no packing dispatch: its stage timing is unavailable (`NaN`),
and the total spans the actual Census through metric timestamps. The existing
benchmark does not expose the borrow mode and needs explicit support for this
unavailable timing before it can measure that mode.

The path avoids the six per-frame GPU copies but still allocates the legacy
output buffers; no allocation savings are claimed. Native refinement CMake
compiles and hashes the tracked `python/stereoforge/shaders` source directly.
The built Census SPIR-V was byte-identical to a separate compilation of that
source using the same compiler and options.

## Validation and limits

- The refinement GPU contract compared all 30 logical planes bit-for-bit with
  both the copy path and the original raw path on two distinct synthetic 12×10
  frames. It exercised source and calibration-owner mismatch rejection,
  retained downstream DaxaView input, two-slot backpressure and recycling.
  Core and synchronization validation reported zero warnings and errors.
- Separately, the existing canonical rectified-input path passed its 300-frame
  synchronization-validation stress test. That test did **not** exercise borrow.
- The focused refinement static and CPU-control tests passed **34 tests**.
  A 10,000-frame actual-source fake-device probe measured approximately **99.4 ns
  per frame** of application bookkeeping, with address checks. It is not a
  measurement of real Daxa, GPU or driver time, nor end-to-end latency.
- No 720p borrow timing, sustained borrow stress test or live-camera validation
  has been established. These checks do not qualify 60 FPS or address the
  guided refinement's known thin-geometry quality regression.

The GPU contract uses `daxa_refinement_contracts --gpu`; focused static checks
are in `tests/test_daxa_refinement_hotpath.py`,
`tests/test_daxa_refinement_reuse_hotpath.py` and
`tests/test_daxa_refinement_borrow_hotpath.py`. See the
[raw reuse evidence](daxa-raw-frame-reuse-evidence.md) for the local validation
layer and GPU-lock setup.

Next gates are a matched 720p comparison against direct and copy paths, a
multi-slot sustained test and validation on the connected camera, all with
identical frame inputs and quality contracts. Record queue time, host overhead,
allocations and total latency independently; do not infer a speedup from the
removed copies alone.
