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
- Adaptive bounds and four controlled ablations, input byte identities, frozen
  configuration, truth-support completeness and common-valid metric accuracy.
- Native C++ bounded persistent resource slots, asynchronous completion and GPU
  buffer leases, explicit readback, command-line stereo/geometry examples.
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

- IMU live access/synchronization and calibrated motion: device permissions blocked
  acquisition during initial tests. No permissions were changed by the agent.
- Physical metric-depth/surface accuracy, held-out quality-loss limits and adoption
  decisions: no independent physical ground truth was supplied.
- Real-time qualification and fully GPU-resident end-to-end native ZED-to-mapping
  integration: research paths have CPU rectification, uploads, metadata gathering
  and readbacks; native stage APIs are building blocks, not that completed demo.
- GPU exposure compensation/seam blending/preview: available in the CPU oracle;
  Vulkan/native renderers deliberately expose strict nearest-surface capability.
- Pose estimation and adaptive candidate compaction: motion is supplied explicitly;
  dense SGM aggregation remains even when matching bounds narrow.
- Portable neural model conversion/operator qualification, ncnn/IREE adapter,
  ROS2, other GPU vendors, map feedback and binary redistribution packaging are
  optional later gates, not asserted as implemented or qualified.
- Full driver validation: Khronos validation layer was absent on the local host.

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
