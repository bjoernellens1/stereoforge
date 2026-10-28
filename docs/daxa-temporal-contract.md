# Native known-motion Temporal contract

This contract describes the native C++/Daxa/Slang path. It implements bounded
reconstruction from supplied poses; it does not estimate motion.
The [design specification](superpowers/specs/2026-10-28-native-temporal-design.md)
records the detailed implementation and acceptance requirements.

## API and dataflow

- Construct `DaxaTemporal(foundation, depth_prototype, config)` with compatible
  metric calibration, rectification maps and the same foundation as Depth.
  The prototype supplies configuration, not an input capture to retain.
- Submit raw measured `DaxaDepthPipeline::Frame` through
  `process(depth_frame, optional_pose)`. A Temporal output cannot become fresh
  measured input. Use one submission thread per shared foundation.
- Pass the returned `Frame::observations()` directly to View. The flow is
  **Depth → Temporal → View**, with GPU buffers retained between stages.
- Frames provide `ready()`, explicit `wait()`, `timings()`, generation and submit
  ticket accessors. `observations()` retains the producer slot and current RGB.
  `Frame::ancestry()` exposes its borrowed ancestry descriptor.
- `DaxaView::Frame::source_observations()` retains the generic input. A View
  owner index addresses that input's observation and ancestry; `UINT_MAX`
  preview owners have no metric lineage. Retain the View frame and source lease
  while consuming their borrowed buffers.

## Observation layout and identity

Each metric plane is contiguous `[eye][y][x]`, left eye then right eye, on the
current rectified grid. With `N = 2 * width * height`, planes contain `N` values:
`Depth` and `Confidence` are float32, `Validity` and `Provenance` uint32,
`SourceIds` and `AgeNs` int64. Depth is axial rectified Z in metres; ages are
nanoseconds. A fused metric buffer is not disparity.

`source()` identifies the **current RGB/grid capture**, including its time and
clock. Original and rectified RGB descriptors remain current, even when a metric
pixel comes from history. RGB uses packed uint32 pixels with R/G/B in the low 24 bits and
`width * 4` byte rows. Selected source ID and selected age describe the metric observation;
they do not describe all contributors or the RGB capture.

Raw Depth reports `SingleMeasured` ancestry with capacity 1, derived from its
six planes; requests for nonexistent sidecar buffers reject. Temporal reports
`BoundedContributors`, with these separate GPU sidecars:

| Sidecar | Physical layout |
| --- | --- |
| `ContributorIds` | `N * 4` int64, pixel-major `[observation][4]` |
| `Count` | `N` uint32 |
| `OldestAgeNs` | `N` int64 nanoseconds |
| `Reuse` | `N` uint32 |
| `OriginMask` | `N` uint32, OR of `1 << original_provenance` |

`capacity()` is the configured union bound, **not the physical ID stride**,
which is always 4. IDs are complete, sorted and unique; unused entries are -1.
Oldest age tracks the oldest contributor independently of selected age.
## Frozen defaults and geometry admission

Defaults are 2 slots, 4 contributors, 4 reuses, maximum age 200,000,000 ns,
relative depth tolerance 0.03 and confidence floor 0.05. Setup admits 2–3 slots,
1–4 contributors, positive reuse bound, nonnegative age and finite nonnegative
tolerance; confidence floor must be finite and in [0,1].

`DaxaTimedRigPose::world_from_rig` is a row-major double 4×4
**world-from-original-left** rigid transform. Supply known translation and
orientation. Its nonnegative timestamp and clock domain must exactly match the
capture's host-ingress timestamp and host clock identity. Device/clock identities
must be known, nonempty and bounded; capture ID/time must fit signed int64.
The borrowed clock string is consumed synchronously. Rigid checks require the
exact homogeneous bottom row and absolute 1e-10 rotation/determinant tolerance.

Calibration/map association must match. Each eye keeps its calibrated K and
original/rectified transform. Float-cast intrinsics must remain finite and fx/fy
strictly positive, including after casting; a double focal length that becomes
float zero rejects. Final composed GPU transforms must also be finite. Actual
Depth admission constrains rectified fx/fy/cy agreement; arbitrary unequal
rectified focal lengths require a separately labelled diagnostic harness.

## Selection, fusion and invalid values

History warps into the current eye, using positive Z, `floor(pixel + 0.5)`, current
support and nearest Z; equal-Z ties choose the lowest source linear index.
Elapsed host-ingress time increments selected and oldest ages; reuse increments
once per warp. Reject candidates on age overflow, oldest age above the bound,
reuse above the bound, invalid support/Z or confidence below the floor. Exact
age and reuse limits are inclusive; eligibility does not create confidence.

Current valid geometry wins incompatible surfaces. Eligible history can fill a
missing current observation: measured provenance 1 becomes reprojected 2;
derived 5 and fused 3 remain distinguishable. Historical selected ID/age remain.
This aged evidence does not establish visibility against a new dynamic occluder.
Consistent surfaces (`abs(historyZ-currentZ) <= tolerance*currentZ`) can fuse only
with adequate confidence, disjoint complete ancestry and a union within capacity.
Fusion uses positive confidence-weighted depth, maximum confidence, provenance 3,
current selected ID/age, maximum oldest age and maximum reuse; otherwise keep current.

The native numerical domain is **normal positive float32 depth**. Subnormal
current/history depth and transformed Z become canonical-invalid and increment
domain counters. Fresh support masking precedes domain counting. Float32 convex fusion bounds its rounded result to its input
range. This differs from the unchanged float64 oracle and is not general float
qualification. Unexpected NaN arithmetic preserves current and counts an
arithmetic fault; supported-domain qualification requires zero such faults.
Invalid metric values are canonical NaN `0x7fc00000`, confidence/validity/provenance
0, ID -1, age 0; invalid sidecars have IDs -1 and every other field 0.

## History, ownership and diagnostic boundaries

Missing pose emits current-only output and severs history; the first recovered
pose seeds history. Duplicate/late capture IDs or timestamps reject without
changing accepted history. A device/clock epoch change resets ordering;
incompatible calibration or GPU context requires a compatible new stage.
`reset()` severs history/sequence state and permits a new world origin. Excessive
time gaps reseed current-only. Failed validation/submission never promotes partial output.

The bounded output ring also stores history. Retained frames backpressure reuse;
active history and pending GPU readers/writers cannot be overwritten. Retained
outputs survive reset and facade destruction. Completed history-only storage can
release upstream Depth/RGB once external leases are gone, avoiding capture chains.
Call `collect()` **before upstream capture/upload**, then check `can_process()`;
the latter only checks admission. Explicit collection also performs nonblocking
shared-device deferred garbage collection, including after failed submissions.
`process()` performs no implicit wait, full-frame readback or per-frame allocation.
`readback(frame)` is an explicit diagnostic transfer/wait boundary, timed separately.

Qualification evidence must separately record the frozen fixtures, executed
validation modes, comparisons and timing scopes.
This contract claims neither sustained 60 FPS, adaptive stereo, physical depth
quality, NN readiness nor full GPU qualification.
