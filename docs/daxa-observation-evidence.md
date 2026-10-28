# Common native observation lease: qualification boundary

This integrates the common typed GPU observation interface needed by native
Temporal and View. It does **not** implement the native temporal stage, adaptive
matching or sustained live 60 Hz. The separate CPU temporal oracle is a diagnostic
reference with its own scope, not a Python component of the native runtime.

## Retained resources and correctness

Depth produces a `DaxaObservationFrame` without copying metric/RGB data or
allocating an adapter. View accepts it directly; old Depth-specific overloads
forward through `observations()`. Current RGB/grid identity and timestamp/clock
labels remain distinct from selected per-pixel observation IDs and ages. The
[resource contract](daxa-observation-contract.md) fixes the six metric planes,
four current RGB buffers, units, ownership, readiness and invalid conventions.

The lease retains the actual producer owner/slot control blocks. Child-before-
context destruction and by-value swap assignment protect empty, copy/move, self-
and cross-owner release. GPU tests exercise common-only retention after typed
Frame/facade release, bounded Depth slot admission, generation, foreign owner/
context rejection, exact six-plane/four-image handles and sizes, current source/
clock labels, and eleven-plane old/common View parity. The setup prototype is
not retained. No mutable Depth observation replacement API was introduced.

The native freeze at `artifacts/daxa-observation-20261027/native-freeze.json`
contains 24 source files, ten binaries and 58 SPIR-V payloads; its SHA256 is
`f23196312c612a1b9a575e1f221a435bce92f8f1b8bb8c511763672adb49f32b`.
All 58 SPVs are byte-identical to Task 6 v2. Source/compiler binding in the View
oracle now includes the observation implementation/header and actual Foundation
translation-unit dependencies; mutation tests reject changing either file.

Root-owned execution records:

- `root-{legacy-view,typed-common-view}-{core,sync}.log`: all four contracts pass.
- `root-oracle-matrix.json`: 24 passing runs (12 geometry profiles, core and sync),
  strict/preview eleven-plane exact checks. Seven perspective profiles also use
  preserved RHI numerical control; five cylindrical profiles use independent
  restricted known-point expectations only.
- `root-720p-sync.json`: 300 measured plus three warmup repeated synthetic 720p
  frames, zero native validation errors/warnings through teardown.

Native validation and frozen RHI numerical comparison remain separate. The
control is not validation-qualified; explicit `--native-only-validation` is
recorded, both child diagnostic streams remain fatal, and no automatic fallback
is permitted. See [the preserved validation boundary](daxa-view-depth-evidence.md).
These fixtures qualify defined geometry and metadata, not physical camera truth
or general floating-point equality.

## Performance and bounded host work

The independent review is `performance-review.json` in the same artifact
namespace. Actual-source CPU probes include the Depth adapter/common lease and
the generic View admission/bind/retirement path, with GPU/device/graph/producer
boundary mocks explicitly identified. Over 10,000 iterations the lease probe
records zero application allocations and zero implicit waits/polls; the View
probe records zero application allocations, reused graphs, bounded producer
references and 58 registry nodes per slot. View admission plus submission uses
two explicit snapshots per frame, plus a final collect. This does not assert zero
driver allocations or global single-poll behavior. CPU probe means establish no
GPU throughput or full enqueue speed claim.

The actual shared-device queue span includes staged input upload, preprocessing,
independent Left/Right SGM, LR/metric stages, inter-graph gaps and View. Four
post-interface ABBA runs use the same static synthetic 1280x720-per-eye input,
D128/P4/W32, Sequential/IndependentSGM and two slots. Each has 100 measured plus
20 warmup frames, validation and readback disabled in the loop.

| Measurement | Range of all four run means |
|---|---:|
| GPU chain queue span | 34.022–34.136 ms |
| Host completion/query wall | 34.500–34.596 ms |
| Strict View GPU stage | 1.982–1.986 ms |
| Preview View GPU stage | 2.046–2.053 ms |
| Requested chain buffer bytes | 1,032,292,504 |

`root-abba-summary.json` binds each underlying report hash and the native freeze;
post-run source/binary/SPIR-V checks match. Initialization, cache construction,
compilation and cold startup are excluded. Requested bytes are not peak driver or
process memory. There is no demonstrated GPU speedup from adding this interface.
The span remains above 16.67 ms, and repetitions of one input do not establish
fresh-frame throughput, age, drops or backpressure under live acquisition.

## CPU and temporal-reference boundary

The final full qualified CPU run after temporal-reference review fixes records
1,505 passing tests, 37 deselected, in 44.15 seconds without exclusions; ten
CPU/source CTests pass. The original 1,492-test snapshot remains preserved.
Focused guards preserve exact forwarding and actual common-path checks rather
than omitting the old overload. Final evidence is `performance-v2-review.json`
and `performance-v2-full-python.log` in the observation artifact namespace.
The depth-disabled View build and CPU contracts also pass with Vulkan disabled.

The new temporal CPU oracle covers timed absolute-world geometry, two-eye
transforms, ancestry/age/reuse limits, conservative confidence, motion loss and
transactional reset/rejection. Independent review found and corrected fresh-
current validation accepting accumulated metadata and nonfinite/zero weighted
mean output from finite extremes; the original failures and frozen evidence are
preserved. It remains a float64/rational diagnostic reference. General native
float32 arithmetic, native temporal GPU execution, complete ancestry propagation,
adaptive candidate/aggregation work, four held-out ablations, physical quality,
peak-memory and sustained live 720p/60 Hz remain open.
