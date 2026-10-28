# Native selective observation export implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development. This is a prerequisite of Task 7, not a replacement for ROS 2, sustained delivery or quality qualification. Execution is already authorized.

**Goal:** Supply a real bounded native GPU-to-CPU publication boundary that copies only requested left depth/RGB and retains source ownership until export completion.

**Architecture:** A setup-bound C++ consumer owns 2–3 persistent mapped coherent staging slots, preferring HOST_CACHED. Each export retains the native producer and submits ordered copies on the same foundation queue. Copy completion and destination lease ownership jointly control reuse. ROS-independent message/clock contracts prepare the next publisher integration; no diagnostic full-plane readback or CPU depth recreation.

**Tech Stack:** Existing C++20/Daxa/Vulkan and native CPU test mocks; no Python in the live path, no new shaders or inference runtime.

**Spec:** [Task 7](2026-10-28-zed-quality-benchmark.md#task-7-sustained-delivery-ros-2-and-final-report), [quality-first design](../specs/2026-10-28-zed-quality-benchmark-design.md).

## Constraints and review focus

- Production source depth is contiguous float32 axial Z metres `[eye][y][x]`; export exactly the first W*H*4 bytes for left depth. Preserve NaN/invalid payloads. No other depth/confidence/validity/provenance arrays are copied.
- Rectified-left RGB is a separate packed uint32 RGBA buffer (little-endian R,G,B,A); export exactly W*H*4 bytes only when requested. RGB-only export must accept `DaxaFrame` without stereo estimation.
- Retain producer/context/maps/source until the **copy** ticket completes. Producer completion alone cannot authorize host access or slot reuse. Destination leases remain valid after producer retirement and exporter object destruction.
- Prefer compatible HOST_VISIBLE|HOST_COHERENT|HOST_CACHED memory; fallback to coherent uncached only with actual type/flags recorded. No noncoherent memory unless an explicit invalidate implementation is qualified; initial version rejects it.
- No per-enqueue buffer/memory allocation, full readback, wait_idle or implicit producer wait. Command recording/submission may use Daxa/driver resources; do not claim their allocation activity is measured. One shared submission thread only.
- Full ring rejects before touching/submitting source work. Partial submit failure retains anything potentially referenced by a submitted copy until completion. Constructor/teardown failure must not leak buffers/memory or destroy pending resources.
- Wrong context/calibration owner/grid, stale leases, moved-from handles, old tickets, retained host frames, generation overflow and cross-thread submission are adversarial inputs, not excuses to silently reuse a slot.
- CPU message-copy and ROS publication are explicit later costs; this class does not claim zero-copy ROS, freshness, physical quality or 60 FPS.

## Task 1: Persistent native export consumer

**Owner:** Export worker owns `include/stereoforge/daxa_host_export.hpp`, `src/daxa_host_export.cpp`, `tests/daxa_host_export.cpp` and `tests/test_daxa_host_export_hotpath.py` only. Root owns CMake/build integration and shared GPU execution. Worker may add a small internal pure contract header if needed and reports it first.

Interfaces in namespace stereoforge:

```
struct DaxaHostExportConfig { unsigned slots=2; bool depth=true; bool rgb=false; };
struct DaxaHostExportMemory { uint32_t type_index,property_flags; size_t bytes_per_slot; bool host_cached; };
class DaxaHostExport {
  class Frame { /* copy/move default, checked generation; retains Impl+Slot */
    explicit operator bool() const;
    bool ready() const; void wait() const;
    std::span<const std::byte> depth_bytes() const; // ready required; empty if disabled
    std::span<const std::byte> rgba_bytes() const; // ready required; empty if disabled
    uint32_t width() const; uint32_t height() const;
    uint64_t producer_ticket() const; uint64_t copy_ticket() const;
    CaptureLease source() const;
    std::shared_ptr<const DaxaRectification> calibration() const;
  };
  DaxaHostExport(std::shared_ptr<DaxaFoundation>, const DaxaObservationFrame&, DaxaHostExportConfig);
  DaxaHostExport(std::shared_ptr<DaxaFoundation>, const DaxaFrame&, DaxaHostExportConfig); // depth=false only
  Frame enqueue(const DaxaObservationFrame&);
  Frame enqueue(const DaxaFrame&); // RGB-only; depth request rejects
  size_t collect(); void drain();
  DaxaHostExportMemory memory() const;
  size_t requested_bytes() const;
};
```

Reject slots outside2..3 or neither output. Setup binds exact immutable calibration **owner**, context identity and width/height. RGB dimensions must match rectified grid. The DaxaFrame constructor only permits RGB-only configuration; pass foundation explicitly and compare context_id. Observation source is native, never a fabricated CPU array. Export does not label temporal ancestry fresh; publisher must use its own known fresh producer/metadata admission.

Use persistent slot objects, mapped destinations and memory blocks allocated at construction. Slot payload can concatenate depth then RGBA (each requested plane W*H*4); disabled planes consume no storage. Record actual memory type/flags and sum requested bytes separately from allocator/driver memory. Bind on owner thread; enqueue/collect/drain reject a different thread. Frame readiness/read-only access may be used without submission; no mutation of live borrowed source buffers.

Same queue copy order: compute writes→transfer reads barrier; requested buffer copies; transfer writes→host reads barrier; submit; capture actual latest queue ticket. Do not call producer.wait() first. `Frame::wait` explicitly waits its copy ticket with a bounded timeout (30s consistent with native API). `ready` uses the existing inclusive completed_submission convention. Accessing bytes before completion rejects. `collect` drops producer leases only once the copy ticket is complete; it does not invalidate a retained host result. Reuse needs complete copy and exclusive slot ownership; increment generation with checked overflow.

Command recorder/list creation follows pinned Daxa APIs and existing submit conventions; never substitute a whole-device wait. If post-submit bookkeeping fails or submit throws after acceptance, inspect the latest ticket conservatively and retain the source+slot pending state. Destructor waits only outstanding copy tickets before destroying owned buffers; never free pending memory after a failed wait. Preserve existing native producer classes and their resource scopes.

- [x] RED/GREEN actual source CPU mock/probe tests: exact depth/RGB byte offsets/counts and NaN bytes, RGB-only no depth, cache preference/coherent fallback/no-compatible rejection, wrong context/owner/shape, full ring before GPU work, copy-versus-producer readiness, retained destination/source leases, producer retirement after copy, generation/moved-from/stale access, owner-thread rejection, constructor/submit/partial-submit cleanup, exporter destruction with retained result.
- [x] Actual-source hot-path guards/microprobe: setup-only buffer/memory allocations, no enqueue wait/readback/idle, bounded slots, fixed requested copies and explicit waits. Independent performance reviewer will verify against actual source separately.

## Task 2: ROS-independent clock and image contracts

**Owner:** Contract worker owns `include/stereoforge/ros_depth_contract.hpp`, `tests/ros_depth_contract.cpp`, `tests/test_ros_depth_contract.py` only. This provides executable contracts used by the future real ROS node, not a substitute publisher.

Define `RosCaptureClockMapping { std::string source_clock; int64_t offset_ns; uint64_t valid_from_ns,valid_until_ns,max_error_ns; }` with offset-only nanosecond mapping to ROS **system** time. `map_ros_capture_stamp(mapping,source_clock,source_ns)->RosCaptureStamp {int32_t sec;uint32_t nanosec;uint64_t max_error_ns;}`. Reject unknown/empty clocks, mapping bounds reversed/zero, values outside interval, overflow/negative result and ROS int32 second overflow. Use checked integer arithmetic, not float conversion. This initial type does not support drift/scale or sim time; caller must supply measured bounded mapping valid for the run, never infer camera time from publish time.

Define `RosDepthImageContract {width,height,step,encoding,frame_id,is_bigendian}` and `make_ros_depth_contract(width,height,frame_id)`: encoding32FC1, little-endian step=width*4 checked, nonzero/bounded dimensions≤8192×8192 and bytes≤512MiB, known nonempty optical coordinate frame string, no per-frame source ID in frame_id. Generic string cannot prove TF extrinsics; describe this as declared frame label. `validate_ros_depth_payload(contract,span<constbyte>)` verifies exact byte extent only and does not rewrite NaNs/invalid depth. RGB contract analogous `make_ros_rgba_contract` encodingrgba8 for the native packed format, not an unmeasured RGB conversion.

Define `RosDepthPublicationIdentity` source/frame/capture/GPU/copy/publication timestamps and a `FreshSourceTracker::admit(source_id,capture_ns,clock)` that rejects repeat/backwards/domain-change; allow initial arbitrary nonnegative signed-int64 IDs, require positive capture timestamp, at most one stored last identity. This tracker counts unique capture identities only; it cannot prove per-pixel freshness. Retain explicit API docs that publisher must reject historical/fused ancestry or use known fresh stereo producers; do not infer age0 from stable depth.

- [x] RED/GREEN boundary arithmetic/stamp mapping/units/payload/NaN identity, repeated/backward/clockchanged source, no per-frame dynamic storage. Native structs contain plain metadata; no ROS SDK dependency, fake ROS headers or fake GPU outputs.

## Task 3: Build, independent review and actual validation

**Root:** narrow CMake registration/library linking, test inventory, actual builds, resource arbitration/GPU smoke, docs/evidence/Git/CI. Existing default builds must not discover ROS, SDK or CUDA.

- [x] Add host-export library to Daxa build only, CPU mock contracts exposed through pytest and/or real named CTest target; no empty test selection accepted. Add pure ROS clock contract target to CPU build. Build actual linked Daxa component.
- [x] Independent functional review of producer/destination lifetime, barriers, failure paths and integer clock correctness; separate mandatory performance worker probes actual source, cache/allocation choice, copies and bounded storage before any GPU validation.
- [x] Root freeze source/dependencies/actual binary and existing analytic fixture, then run bounded same-source128×64 GPU correctness/synchronization test. Include byte-exact depth/RGBA copies, disabled plane suppression, retained destination slots and submission stress; no live/ROS/720p/FPS claim. GPU unavailable/contended leaves this gate open, not silently replaced with CPU mocks.
- [x] Full CPU regression, source/byte evidence and independent reviews. Publish this checkpoint and confirm its GitHub CI before reporting completion. Keep Task7 ROS2 app wiring, sustained delivery/report accounting, clock acquisition and600s actual topic benchmark open. The broader end-to-end goal remains active.

**Validation checkpoint 2026-10-28:** Native library and actual linked GPU contract built. The 128×64 analytic test passed exact left depth/RGBA bytes over 30 static cycles with core and explicit synchronization validation, selected cached coherent memory type5/flags14, and requested131072B across two slots. CPU regression:2546passed,37hardware tests deselected; default CPU CTest2/2passed. Independent functional and performance reviews accepted the source. This is a selective publication prerequisite; actual ROS publication, measured clock mapping, sustained600s delivery and 720p60Hz qualification remain open. [Evidence](../../native-host-export-evidence.md).
