# Implementation ledger — StereoForge, 2026-10-27

Binding specification: the user-approved "Vulkan stereo, wide RGB-D, and temporal
reconstruction" plan in this chat. Original proposal is retained for historical
context; later Vulkan-only and research-first decisions supersede it.

## Implemented and locally exercised

- Typed Python/C++ frame, calibration, resource, completion, depth/provenance and
  timestamped-pose contracts; explicit CPU references and Vulkan selection.
- Native ZED Open Capture binding, lossless checksum-bound recording/replay,
  factory calibration download/cache, rectification and depth-to-raw remapping.
- Validated per-eye camera controls, explicit automatic/manual profiles, cached
  host readbacks and control metadata preserved through recordings and replay.
- Shared Slang Census/four- and eight-path SGM, disparity confidence/subpixel,
  Python left/right consistency and metric depth, scalar-stage regression oracles.
- CPU strict/preview RGB fusion, separate RGB/depth masks, source/approximation,
  exposure/seam/edge-aware compatible-surface blending, perspective/cylindrical
  virtual cameras, and timestamp-bound orientation stabilization.
- Shared Slang strict virtual projection and temporal reprojection/fusion, exposed
  through Python Vulkan adapters and native stage APIs. Conservative ancestry,
  occlusion, expiry, calibration/clock-reset and consumer-history ownership checks.
- Shared Vulkan per-eye visibility, confidence seams, compatible blending,
  robust exposure matching and marked preview, exposed through Python and native
  adapters. Optional stage timestamps and packed Python readback; see fusion-evidence.md.
- Adaptive bounds and four controlled ablations, input byte identities, frozen
  configuration, truth-support completeness and common-valid metric accuracy.
- Native C++ bounded persistent resource slots, asynchronous completion and GPU
  buffer leases, explicit readback, command-line stereo/geometry examples.
- Standalone source-built Vulkan-only Slang RHI against the matching official
  Slang SDK, with no Python/SGL discovery or linkage. Matched 720p native fusion
  separates retained GPU output from CPU readback; see standalone-rhi-comparison.md.
  It does not yet meet the 16.67 ms frame budget.
- CPU CI, locked Python dependencies, pinned capture sources/compiler helper
  hashes, runtime capability reporting, hardware/performance evidence.

## Decisions and corrections

- Ruling: use the empty requested repository as the project root; implement in
  an isolated Git worktree because parent AGENTS requires isolated worker changes.
  Native app worktree creation could not recognize the newly initialized repo;
  Git worktree fallback is used. Source publication followed the source-license
  evaluation; the GitHub repository is now public. Binary redistribution remains
  a separate gate in license-evaluation.md.
- TDD modules recorded initial missing implementation/failing behavioral tests,
  then passing verification. Integration review found returned-history aliasing,
  unbound pose timestamps and unmatched evaluation masks; regression tests precede
  each correction. Per-pixel calibration remap was replaced by vectorized z-buffer
  selection; performance guards and measured microbenchmarks record the boundary.
- Factory raw rotation needed transposition into the right-camera pose convention.
  Synthetic tests and same-byte camera correspondences support the correction.
  Failed initial evidence was retained and corrected evidence written separately.
- A native validation request initially succeeded without an installed Khronos
  layer because pinned RHI silently omitted it. Independent loader-layer preflight
  now fails such requests. Earlier numeric smokes are not validation-clean runs.
- Ruling: no CUDA/ROCm research baselines are executed. External results stay
  literature context; no superiority claim over unexecuted methods is permitted.

## Gates still open

- IMU strict image-interval association and calibrated motion: raw delivery and
  exact recording/replay now pass after the user corrected HID access, but a fresh
  48-frame run found 38 late-batched samples among 319 ordered observations.
  Its strict association gate remains failed. Physical synchronization accuracy,
  camera-IMU calibration and late-arrival handling remain open; see imu-evidence.md.
  No permissions were changed by the agent.
- Physical metric-depth/surface accuracy, held-out quality-loss limits and adoption
  decisions: no independent physical ground truth was supplied.
- Real-time qualification and fully GPU-resident end-to-end native ZED-to-mapping
  integration: research paths have CPU rectification, uploads, metadata gathering
  and readbacks; native stage APIs are building blocks, not that completed demo.
- Virtual-view observation identity/age export through Python remains open;
  source-depth identities are preserved, but FusedView needs corresponding planes.
- Pose estimation and adaptive candidate compaction: motion is supplied explicitly;
  dense SGM aggregation remains even when matching bounds narrow.
- Portable neural model conversion/operator qualification, ncnn/IREE adapter,
  ROS2, other GPU vendors, map feedback and binary redistribution packaging are
  optional later gates, not asserted as implemented or qualified.
- Full pipeline driver validation remains open. Native stereo now passes bounded
  core-validation runs using a locally extracted signed Khronos layer. Native
  compositor/temporal contract fixtures also pass core validation; synchronization
  validation and the integrated live chain require further checks.
  See vulkan-validation-evidence.md.

Detailed commands and measured limits are in README, hardware-evidence.md and
performance-evidence.md. Local raw recordings are ignored artifacts; numeric
summaries and their provenance are versioned. Future changes must preserve the
initial failed reports and the distinction between code tests and physical gates.

## Final local verification

`uv run pytest -q`: 98 passed, including actual AMD Vulkan shader execution and
four/eight-path Python/C++ output parity. `ctest --preset cpu`: 1/1 passed.
`ctest --preset vulkan`: 4/4 passed (CPU contracts, missing-layer rejection,
native GPU geometry/temporal, native GPU stereo/calibrated metric depth).
The missing-layer test verifies honest rejection; it is not driver validation.

Built wheel and source archive; installed-wheel Vulkan wide RGB-D smoke passed.
Source archive includes native CMake sources and shared shaders. These counts
describe the initial implementation; camera-control follow-up verification is
recorded in camera-controls-evidence.md. Source changes are integrated into the
original checkout and pushed to the configured GitHub remote.

Follow-up verification after shared Vulkan RGB fusion: **221 Python tests pass**,
native Vulkan CTest 4/4 and CPU CTest 1/1. Wheel and source archive build. Positive
native core validation passes for stereo and geometry/temporal contract fixtures.
The full requirements audit in requirements-audit.md remains incomplete; these
checks do not qualify the production chain or physical depth accuracy.

Standalone follow-up: **230 Python tests pass**, standalone CTest 4/4, and
standalone stereo/geometry core-validation runs emit zero diagnostics across
both streams. The benchmark executes under invalid PYTHONHOME/empty PYTHONPATH.


Resident native follow-up: **265 Python tests pass**, standalone CTest **5/5**,
and CPU CTest **1/1**. Native stages now share an explicit `NativeDevice` context
and queue. Geometry accepts retained GPU source planes and materializes selected
metric identity/age on the GPU. Matched 720p static fusion measures **1.427 ms
strict / 1.468 ms preview p50**, excluding one-time upload and CPU output readback.
All eleven final planes match the host-input path exactly. A three-frame actual
GPU stereo → bridge → compositor → downstream consumer fixture checks fresh
64-bit IDs/colors, metric masks and invalid defaults using only 16-byte final
counter downloads, including producer-slot reuse after high-level owners drop.
Stereo, geometry/temporal and handoff fixtures pass positive Khronos core
validation with zero diagnostics across both streams. These results do not
qualify synchronization validation, full native live capture/rectification/LR
consistency/temporal integration, or sustained 60 Hz operation. See
[native-resident-evidence.md](native-resident-evidence.md) for commands, hashes,
setup cost, p95 and the precise evidence boundary.
