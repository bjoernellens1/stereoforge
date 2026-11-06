# Native selective host export contracts and evidence

This qualifies bounded static native export correctness and CPU contracts. Live capture, 720p throughput, FPS, ROS delivery and SDK performance remain unqualified. Exact records and artifact hashes are in [the evidence JSON](evidence/native-host-export-2026-10-28.json).

## API and ownership

Use [DaxaHostExport](../include/stereoforge/daxa_host_export.hpp) with a native observation (depth and/or RGB) or native frame (RGB-only). Configure two or three persistent destination slots at construction. Select rectified-left depth, packed RGBA, or observation sidecars explicitly. Depth, RGBA, confidence, validity and provenance each occupy `width * height * 4` bytes; source IDs and ages each occupy `width * height * 8` bytes. No right-eye plane or temporal contributor-list plane is exported.

```cpp
stereoforge::DaxaHostExport exporter(context, observation,
    {.slots = 2, .depth = true, .rgb = true});
auto host = exporter.enqueue(observation);
// Poll host.ready(), or explicitly wait outside the submission hot path.
host.wait();
auto depth = host.depth_bytes();
auto rgba = host.rgba_bytes();
// Consume while retaining host. Release it before requesting slot reuse.
exporter.collect();
```

Optional `confidence`, `validity`, `source_ids` and `age_ns` flags default to false. Read them with `host.observation_bytes(DaxaObservationPlane::Confidence)` (or the corresponding plane enum). Unrequested planes return an empty span after copy completion. Observation sidecars require an observation producer; a preprocessing-only `DaxaFrame` supports RGB only. Existing depth/RGB/provenance layouts remain unchanged. Additional 64-bit planes are aligned to eight bytes, with any padding included in `bytes_per_slot` and `requested_bytes()`.

Confidence is the producer's unitless matching heuristic, **not calibrated metric uncertainty**. IDs and ages retain exact signed 64-bit storage, including invalid IDs and high bits. Per-pixel IDs/ages describe the producer's observation semantics; `source()` identifies the current capture/grid, not every temporal contributor. An age of zero does not qualify acquisition time or clock synchronization. A client must retain the host frame while consuming any returned span.

At 720p, selecting every available plane requests 33,177,600 bytes per slot (two slots: 66,355,200 bytes), versus 7,372,800 per slot for depth plus RGB. These are requested payload/storage bytes, not measured transfer bandwidth, serialization cost or a live-throughput claim. ROS publication of these additional sidecars remains a separate integration task.

Setup, enqueue, collect and drain share the native producer submission thread. Enqueue retains the source producer, capture, calibration and context through the actual copy completion ticket. Producer completion alone does not authorize host reads. `collect()` retires completed source ownership; a retained host frame independently protects destination bytes and context, including after exporter destruction. Reuse requires copy completion and exclusive slot ownership. A full ring rejects admission before recording/submission. There is no per-enqueue application buffer/memory allocation; Daxa recorder and driver allocations are outside that claim. Keep native producer graphs alive during steady-state publication: releasing the final producer lease may invoke existing producer teardown/`wait_idle`, so graph teardown is setup/shutdown cost outside the measured steady-state contract.

Memory selection pins a compatible HOST_VISIBLE|HOST_COHERENT type, preferring HOST_CACHED and recording any coherent uncached fallback. The actual tested allocation used type **5**, flags **14**, cached host memory: **65,536 bytes per slot**, **131,072 requested bytes total** for two 128×64 slots with both outputs.

Explicit `Frame::wait()` and `drain()` propagate failure. An uninspectable potentially accepted partial submission or failed final destructor wait calls `std::terminate`, retaining potentially pending resources rather than freeing them. This fail-closed policy can terminate the consumer process; it is not a recovery guarantee.

## Diagnostic extension checks (2026-11-06)

The extension was checked against actual-source CPU contracts and a native 128×64 Vulkan producer. The CPU test covers odd-grid 64-bit alignment, exact selective bytes, retained destinations, pending-copy read rejection and RGB-only producer rejection. The native check compares all six requested observation planes byte-for-byte to producer readback, including invalid values and source IDs above 2^32. Existing 30-cycle legacy export checks remain in place. This does not qualify calibrated uncertainty, ROS delivery or live performance. Full-suite verification and the fixed GPU microbenchmark gate are recorded separately in the campaign evidence.

## Earlier evidence and limits (2026-10-28)

- Linked native build passed after the supervisor fixed the local private VMA link. The build log contains compiler warnings; the final GPU validation logs contain no errors or warnings.
- Native GPU tests passed exact selective-byte checks over **30 static 128×64 cycles per run**, including retained outputs, disabled planes and slot reuse, with core validation and a separate explicit synchronization-validation run (`VK_LAYER_VALIDATE_SYNC=1`). The final sync-validation run explicitly selected RADV with `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`; its dependency freeze records 176 verified hashes covering sources, compiled shaders, binary, libraries, validation layer and driver, plus sysfs device IDs. Reservation was cooperative `flock` only; desktop renderers were present, with no exclusive global device reservation.
- CPU regression: **2,546 passed, 37 deselected in 150.17 s**. Default CPU build CTest: **2/2 passed**. Independent functional review separately passed **500,000 clock oracle cases** and **1/1 Release CTest**; sanitizer linkage was unavailable.
- Independent performance review: **30 tests passed**. The latest full-suite actual-source mock samples span **22.8789–33.5408 ns/enqueue+collect**; exact samples are preserved in JSON. These measure application control against bounded mocks and exclude real producer copy cost, Daxa, driver, GPU transfer/completion, ROS and capture. They establish neither real enqueue latency nor FPS.

[GPU contracts](../tests/daxa_host_export.cpp), [CPU export contracts](../tests/test_daxa_host_export_hotpath.py), [performance contracts](../tests/test_daxa_host_export_performance.py), and [ROS metadata contracts](../include/stereoforge/ros_depth_contract.hpp) support distinct boundaries. Metadata validation does not measure acquisition clocks, establish TF, or prove fresh pixel ancestry.

The [implementation plan](superpowers/plans/2026-10-28-native-observation-export.md) still requires Task 7's real ROS application, measured clock mapping and 600-second delivery report. The user-authorized SDK benchmark has not been executed here.
