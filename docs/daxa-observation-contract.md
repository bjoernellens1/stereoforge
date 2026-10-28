# Native retained observation contract

`DaxaObservationFrame` is the common GPU input to the native View. The current
producer is `DaxaDepthPipeline::Frame::observations()`. The planned native temporal
producer is not implemented by this interface alone. Existing Depth-specific
View overloads forward through this value; legacy geometry fixtures remain a
separate input path.

## Geometry and resource layout

All six metric buffers are separate, contiguous two-eye arrays indexed as
`eye * width * height + y * width + x`, Left first. `width` and `height` are the
current rectified grid. Geometry is rectified pinhole axial Z in metres, with
that eye's intrinsics/rectification model retained in the exact calibration owner.
These buffers do not contain virtual-view depth or original-camera disparity.

| Plane | Element | Units / meaning | Canonical invalid value |
|---|---|---|---|
| `Depth` | float32 | Rectified axial Z, metres | NaN (`0x7fc00000`) |
| `Confidence` | float32 | Score in [0,1], not calibrated uncertainty | 0 |
| `Validity` | uint32 | 0 or 1 | 0 |
| `Provenance` | uint32 | Existing observation provenance code | 0 |
| `SourceIds` | signed int64 | Original selected observation identity | -1 |
| `AgeNs` | signed int64 | Selected observation age, nanoseconds | 0 |

Each scalar plane has `2 * width * height` elements. The first four use four
bytes per element; IDs and ages use eight. Validity must gate metric reads;
valid-looking RGB alone never establishes depth. Producer contracts establish
canonical contents: this adapter does not scan or read back every pixel.

Four image descriptors expose tightly packed decoded RGB: `OriginalLeft`,
`OriginalRight`, `RectifiedLeft` and `RectifiedRight`. Each uses a uint32 per pixel
with R/G/B in the low three bytes and the upper byte unused by RGB synthesis.
The format enum retains its existing `Rgba8PackedUint32` name. Original images
use the full per-eye capture dimensions; rectified images use the metric grid.
Row stride is image width times four bytes. A descriptor borrows its buffer;
retain the observation lease for the entire downstream use.

Raw stereo, disparity, LR consistency and other diagnostic planes remain on the
original Depth frame API. Converting to observations neither recomputes those
planes nor labels temporal/virtual depth as measured disparity.

## Identity, clocks and ownership

`source()` retains the current capture that supplies RGB and defines the output
grid, host/camera timestamps and their original clock-domain labels. It does not
replace the metric buffers' per-pixel source IDs or ages. Camera/host clocks are
not aligned or converted by this interface. Motion consumers must validate an
explicit timestamp/domain relationship independently.

The value retains the actual producer owner and slot control blocks, actual
shared `DaxaFoundation`, immutable map owner and current capture. It caches six
metric handles and four RGB descriptors without allocating an adapter. Copying
the value retains the same storage. The original typed Depth frame and producer
facade may be released while a common lease remains alive. Slot reuse still
requires downstream release and producer completion. Foreign context/map owner,
shape mismatch, empty or stale generation is rejected before View execution.
Only the closed producer set can construct populated values; there is no public
arbitrary-buffer constructor or observation replacement API.

Assignment uses a by-value `noexcept` swap. Destruction releases the producer slot
before its owner, and both before the retained foundation; empty/self/cross-owner
copy/move cases must preserve this ordering. `operator bool` indicates a populated
lease; accessors additionally validate its generation. Returned `BufferId`s are
borrowed read-only by contract, not a security boundary against an application
issuing its own writes.

## Completion and admission

`submit_ticket()` refers to the actual shared device's main queue. Same-queue
consumers enqueue directly with explicit write/read dependencies and retain the
lease through their own completion and output release. Conversion and ordinary
metadata/handle access do not wait or poll. `ready()` explicitly polls completion;
`wait()` is an explicit synchronization boundary. A ticket from one context is
not valid on another device or arbitrary queue.

The View retains the common lease in its preallocated slot. Admission precedes
upstream acquisition/upload, so full output rings reject before wasted work.
Resource/context creation, calibrated ray/map preparation and matrix checks stay
at setup; only bounded metadata/address binding occurs per submission. Cross-API,
cross-device and external-queue interoperability remain separate qualification.

A future temporal lease must additionally retain complete bounded contributor
ancestry, selected-source versus oldest-contributor age and reuse metadata. The
six existing planes alone do not describe compound ancestry. Their presence does
not implement or qualify temporal reconstruction, adaptive matching or live 60 Hz.
