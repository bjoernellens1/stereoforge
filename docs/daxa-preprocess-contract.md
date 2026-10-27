# Native preprocessing contract

Status: the staged two-pass implementation passes native GPU contract tests
under core and synchronization validation, including retained consumers, dynamic
GPU stereo inputs and capture-pool recovery. Small-fixture GPU outputs match all
eight independent OpenCV planes exactly. Bounded synthetic full-size checks and timings are documented in
[preprocessing evidence](daxa-preprocess-evidence.md); full live-pipeline
qualification remains open. The CPU numerical profile below
has been checked against installed OpenCV 4.14.0 with optimization both enabled
and disabled. This document does not qualify a live pipeline,
physical calibration, pose estimation or 60 Hz operation.

## Processing and geometry

The native path consumes a retained `CaptureLease` from the existing packed
recording/replay or live acquisition path. It preserves the source frame ID,
camera and host timestamps, clock information, calibration identity, controls
and raw IMU association. Raw IMU samples are not a validated pose.

The fixed operation order is:

1. Decode side-by-side YUYV into both complete original RGB lens images.
2. Remap RGB using the four cached float32 calibration map planes.
3. Round the rectified RGB values, then convert to uint32 grayscale values 0–255.
4. Admit the leased luma buffers into stereo on the same Daxa foundation.

Both original RGB canvases remain available for wide-view synthesis. Rectified
RGB/luma carry the rectified calibration identity; original RGB carries the
original identity. Missing or mismatched calibration blocks processing. A set of
map files alone does not establish metric calibration.

The numerical profile is
`opencv414_yuyv20_remap32_rte_rgb_halfup_gray15`:

- Signed 20-bit limited-range YUYV conversion uses coefficients 1220542,
  1673527, -409993, -852492 and 2116026, followed by RGB saturation.
- Remapping quantizes float32 coordinates to 32 phases using nearest-even
  coordinate rounding. RGB interpolation uses constant-zero borders and
  half-up output rounding.
- RGB-to-gray uses coefficients 9798, 19235 and 3735, adds 16384, then shifts
  right by 15. Widening to uint32 follows the 8-bit result.

Interpolating raw Y or converting to gray before RGB remapping changes the
inputs and is outside this profile. Maps must be finite and within ±32768;
source and output dimensions must be below 32767. Lens widths must respect
YUYV chroma pairing. Padded row strides and byte extents require checked bounds.

A separate support mask reports whether every tap with **nonzero quantized
interpolation weight** is inside the source image. A zero-weight outside tap
must not invalidate an identity-map edge. The mask does not modify the frozen
stereo input or Census validity rules. RGB still follows the exact zero-border
interpolation contract where support is incomplete.

## Fixture and setup boundary

Fixture schema `stereoforge.daxa_preprocess_fixture.v1` reuses `capture.sfp` and
an exact record ordinal. A sidecar binds:

- source eye dimensions and padded stride, plus rectified dimensions;
- the complete recording SHA256 and selected record ordinal;
- original and rectified calibration identities and canonical JSON byte hashes;
- R1/R2/P1/P2 and their source/image conventions;
- four little-endian float32 map files, each with shape, byte count and SHA256;
- expected original RGB, rectified RGB, luma and support-mask planes.

The supported exporter emits the existing Python canonical calibration JSON
and IDs, without an added newline. The native importer hashes the exact supplied
bytes; it does not normalize JSON or promise semantic identity across alternative
serializations. Reformatting a valid calibration changes its byte identity and
requires explicit record/configuration rebinding; that new hash is not the
original Python `Calibration.identity`. Setup validates file sizes, hashes,
dimensions, rigid right-to-left pose, nonzero baseline, map domains and
calibration associations before GPU resource allocation. The
offline Python exporter/reference is allowed at configuration and test time;
it is absent from native frame processing.

The parser uses pinned nlohmann/json and OpenSSL Crypto 3.x only for setup.
Their notices are covered in the [license evaluation](license-evaluation.md).
No raw recordings or real camera calibration files belong in the source tree.

## Ownership, submission and retirement

A Daxa-specific typed frame carries concrete buffer IDs, packed formats,
dimensions, byte extents/strides, context identity, generation and producer
completion state. It retains its allocation owner and slot marker. Admission
rejects foreign contexts, stale generations and inconsistent geometry before
copy or dispatch; a physical device ID alone is insufficient. Stereo binds to
the exact immutable rectification/configuration shared owner used at setup.
Matching identity strings alone do not admit another owner's potentially
different maps or numerical profile. A separately loaded owner requires setup
or rebinding even when its bytes appear equal; interchangeability across such
owners is not promised by this integration. This check performs no per-frame
map hashing or comparison.

Maps, uploads, outputs, parameters and graphs are allocated at setup. A bounded
ring permits reuse only after consumers release the slot and submitted work
completes. Supported Daxa garbage collection remains bounded in standalone and
integrated paths; its internal completion scans are included in CPU/wall timing.
One admission snapshot is not a claim of one total Vulkan completion query.
The serial latency benchmark also polls readiness when retrieving preprocessing
and stereo timestamps; these checks belong to timestamp-query CPU and full host
wall timing, not to an unreported cost outside the iteration. Host-write, upload and compute dependencies are explicit. Importing
preprocessing outputs into stereo requires a compute-write to compute-read
memory dependency on the shared queue, without a CPU wait or intermediate
image readback.

Stereo retains GPU input leases until its GPU work completes. A bounded
`collect_completed_inputs()` operation uses one explicit application completion
snapshot and retires completed input leases even if stereo output leases remain retained.
The integration owner calls collection **before** acquiring or admitting the
next preprocessing frame. Otherwise, input retention can fill the preprocessing
ring before stereo gets a chance to recycle its slot.

Output source metadata retains its capture/configuration ownership separately
from the preprocessing GPU slot. Retiring an input slot must not erase output
frame identity, controls or IMU association. After GPU completion and release
of all output leases, collection must also drop the slot's source metadata
lease; retaining it until the next enqueue can exhaust the capture pool before
a new input exists. Collection releases stereo input leases, unleased stereo
metadata, then unleased preprocessing metadata before acquiring another frame.
A constructor's prototype input
must not remain secretly retained after graph setup. Shutdown drains submitted
work and retires input leases before closing the shared foundation.

## Qualification gates

The staged-upload, two-pass baseline is the first implementation gate. Exact
OpenCV output checks, repeated records, padded strides, identity/fractional
borders, missing calibration, hash failures, foreign contexts, delayed consumers,
ring pressure, retirement and teardown must pass before timing claims.
Independent functional and performance reviews precede root-owned GPU runs.

GPU validation must check exact plane bytes, dynamic GPU-luma-to-stereo parity
and synchronization with retained/delayed consumers. Full dumps and scalar
references run outside the retained-GPU timing interval. Report uploads, GPU
stages, submission, waits, frame age, drops and requested memory separately.

Manual fusion, 8×8/16×8/16×16/32×4 workgroups and staging versus direct host-visible
UMA policies require separate comparisons after baseline correctness. They are
pending measurements, not automatic improvements. Right-reference estimation,
LR consistency, metadata/depth outputs and shared-context wide-view intake remain
subsequent integration gates; preprocessing alone does not qualify the full goal.

## Benchmark diagnostic boundary

`compare_daxa_preprocess.py` checks source/compiler/binary/SPIR-V/input hashes
before and after a run and rejects validation messages through teardown. Its
normal diagnostic mode requests eight plane dumps and final scalar stereo
integer/packed-penalty checks. Stereo report flags distinguish these exact checks
from the still-open compound floating-point reference qualification.

`--resident-performance` omits dumps and stereo intermediate readback. The timed
iterations keep preprocessing and stereo outputs on the GPU. The harness retains
one final eight-plane preprocessing oracle after all samples; this is explicit
diagnostic work, and its cost is reported outside the timed samples. Consequently
the complete benchmark process is not described as having no readback.
