# Fast60 independent full-resolution refinement

Status: experimental design; implementation and quality gates remain open.

## Objective and frozen comparisons

Keep the measured independent 640×360/D64 SGM front end and refine with actual
1280×720 image evidence. Start with full-frame coalesced Census searches at radii
±2, ±4 and ±8 **target pixels**, then compare selected regions and broader
recovery. Interpolated coarse disparity is only a prediction. No coarse depth
value becomes a new full-resolution observation merely through interpolation.

The measured coarse queue already includes upload, decode, rectification, both
SGM references, LR and metric metadata. Its 6.209–6.287 ms p99 is a static-fixture
measurement, not an additive component from which combined p99/FPS can be proven.
The refinement ≤5 ms and combined processing p99≤15 ms budgets are targets.
Retain full-resolution independent SGM as a comparison, separately from independent
scene truth. No profile changes the default estimator before quality qualification.

## Native interface and geometry

Add an opt-in `DaxaRefinement` child stage in a new header/source/CMake target.
It takes the shared foundation, target `DaxaRectification`, a coarse Depth-frame
prototype and fixed configuration. `process(coarse)`, `can_process()`, `collect()`
and an owning Frame follow existing closed native lease contracts.

Initial geometry is an exact 2× refinement in each dimension. Construction checks
that coarse/target describe identical original cameras, dimensions, distortion,
R1/R2 and rig pose, and that their projections obey the F0 half-pixel transform.
Per-frame admission binds the exact coarse calibration owner, device, source and
generation. The constructor retains geometry, not a permanent capture prototype.

Borrow original RGB from the coarse frame: no second upload, YUYV decode or host
round trip. Remap those originals directly into the target grid. Target image
metadata and output observations use the target map owner; coarse-configured View
instances must reject them. Original peripheral RGB remains available.

Outputs are two target-resolution disparity/confidence/validity layers and the
existing axial-Z/confidence/validity/provenance/source-ID/age planes. Provide
bounded diagnostics for prediction, evaluated search bounds/reasons/method and
actual candidate work. Invalid floats have the existing canonical NaN encoding;
invalid confidence/provenance/age are zero and source identity is -1.

Use a distinct `IndependentLocalCensus` right-reference/estimator identity, not
`IndependentSGM`. Existing Depth configuration continues rejecting unsupported
estimator modes. Fresh validated results have provenance1, current capture ID,
age0 and singleton ancestry. Coarse and refined evidence from one capture are
never counted as independent observations. Temporal intake stays unchanged in
this experiment; View consumes the generic retained observation lease.

## Prediction and matching

For target coordinate u, the coarse coordinate is `(u+0.5)/2-0.5`; predicted target
disparity is twice the bilinear coarse disparity. Every nonzero-weight neighbour
must be LR-valid, finite and pass the fixed prior-confidence gate. A configurable
coarse-disparity-span gate prevents interpolation across discontinuities. Missing
or incompatible prior is an explicit recovery reason, never an invented value.
Freeze these gates on development data before held-out evaluation.

Compute shared full-resolution 5×5/24-bit Census descriptors. A descriptor is
supported only when its complete footprint lies in the image and all contributing
rectification-support samples are valid. This stricter refinement support rule
does not silently change the frozen SGM implementation.

For a usable prediction p, centre=`floor(p+0.5)` and the integer search interval is
`[max(0,centre-r), min(D-1,centre+r)]`, D=128 initially. Each eye independently
evaluates Hamming costs: Left reads Right(x-d), Right reads Left(x+d). Share
descriptors, never winners or aggregated costs. Count actual supported candidate
evaluations separately from requested hypotheses and launched pixels.

Require at least two supported hypotheses; a collapsed/single-supported interval
is invalid. Select the lowest-disparity minimum deterministically. Second-best
is the minimum over all other supported hypotheses, including equal-cost ties.
Require integer `second-best - best >= min_margin`, initially min_margin=1;
ties therefore fail. Confidence is `(second-best-best)/max(second-best,1)` in
float32 and clipped to [0,1], an interval-dependent matching score, not calibrated uncertainty or
full-search uniqueness. A winner at a restricted interval endpoint is raw- and
metric-invalid when recovery is disabled, even if LR would agree. It can become
valid only through a subsequent accepted unrestricted search. Global domain
endpoints are distinguished from restricted endpoints.

Subpixel-on uses only supported, already-evaluated winner±1 costs. With both
neighbours, offset=`clamp(0.5*(c_minus-c_plus)/(c_minus-2*c_best+c_plus),-1,1)`;
a zero denominator gives zero offset. Otherwise retain the integer winner,
record subpixel_applied=false and MissingSubpixelNeighbour, and perform no extra
cost evaluations. This does not by itself invalidate a unique supported global
endpoint/interior winner. Never use sentinel costs. Subpixel off/on and applied
fractions are explicit profiles, not silently mixed comparisons.

Initial development settings are prior-confidence floor0.05, maximum coarse
neighbour disparity span2 pixels, min_margin1 and target LR tolerance1 pixel.
They are reported configurable experimental settings, held fixed across radius
comparisons; held-out thresholds are not tuned from test results.

After independent matches finish, use the target nearest-correspondence LR and
axial-depth conventions. Pin the coordinate graph to separately rounded float32:
`t=f32(f32(x) +/- f32(d))`, then `u=f32(t+f32(0.5))`, then `floor(u)`.
The new compiler profile forbids contraction/reassociation of those operations;
integer and mask agreement is still checked independently. Every metric-valid output
requires actual target-resolution evidence in both images plus final LR and
support acceptance. Failed verification remains invalid. Predictions have their
own diagnostic validity and never populate measured metric planes.

## Recovery and region experiments

First measure full-frame guided matching with recovery disabled, reporting missing
coverage honestly. Then add one bounded unrestricted-D128 recovery round for
missing/unreliable prior, discontinuities, restricted-boundary winners, ambiguity,
explicit selected regions and provisional LR failures. Re-run final LR after both
eyes finish recovery. No unbounded iteration and no reuse of unsupported surfaces.

Compare full-frame versus fixed, documented region fractions before introducing
compaction. Excluded pixels stay metric-invalid; report their completeness cost.
For edge/uncertain recovery, select from observed images/predictions, never truth.
Record guided/recovery fractions and accepted methods separately. A dense launch
with early return is acceptable when candidate loops actually skip excluded work.
Full D128 recovery cannot recover depths outside its calibrated disparity domain.

## Arithmetic and correctness

Independent CPU references cover pixel-centre geometry, prediction, supported
Census, costs, integer winners/second-best/ties, intervals, reasons, work counts,
independent Right geometry, LR, invalid encodings and source metadata. Integer
stages, masks and identities require exact agreement. Include threshold, border,
nonzero-principal-offset and missing-subpixel-neighbour cases before GPU execution.

Continuous confidence/subpixel/Z checks must have a reviewed arithmetic contract
before GPU qualification. Vulkan does not require correctly rounded FDiv:
[Khronos SPIR-V environment](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html)
permits a 2.5-ULP division error in its stated normal-denominator range. Use
per-operation error propagation with quantized input calibration and inspect the
compiled operation/contraction policy; do not add a generic post-hoc epsilon.
Report float64 world residuals separately from float32 execution agreement.
Ambiguous boundary predicates remain an open gate until resolved; exact mask
checks are not waived. Existing Stereo/Temporal float failures remain preserved.

## Lifetime, performance and qualification

Use two persistent output slots initially, with setup-created maps, graphs,
parameters, queries and buffers. Each slot retains its input through the actual
final submission ticket, including partial-submit exceptions. Reuse requires
inclusive completion and no external output leases. Common/View leases survive
producer-facade destruction. Admission rejects before upstream capture/upload.
No per-frame allocation, graph rebuild, wait or full output readback in process.
Benchmark terminal waits and diagnostic readback as separate explicit boundaries.

Require independent source/performance review and actual-source allocation/wait
guards before root GPU dispatch. Verify extraction/remap refactors against the
frozen old preprocessing outputs, with changed SPIR-V identities disclosed.

Develop exact shifted-texture algorithm probes and an independent analytic
two-camera textured-scene renderer. Render original rays separately and export
per-grid axial Z/disparity/surface/visibility/occlusion/region truth, rather than
resizing truth or treating full SGM as truth. Include near/far, slants, thin-object
width/phase, mixed footprints, repeated/weak texture, exposure and occlusions.
Keep the same original YUYV bytes across all profile comparisons. Physical RGB
recordings without ground truth establish appearance only.

Measure radius2/4/8 and region-fraction matrices on identical inputs: coarse-only,
coarse+refinement, then fixed720p View separately. Keep RGB/Temporal off for the
primary stereo comparison. Report direct combined GPU and transfer-inclusive host
samples/p99/max, all stage costs, actual work, requested/peak-memory scope, valid
coverage, correct completeness, false accepts, thin/edge recovery and baseline
disagreement. Freeze development thresholds before held-out quality evaluation.
No default promotion or live60 claim without the existing physical/quality and
ten-minute unique-frame/age/drop/bounded-queue gates.

## Current qualification boundary

The CPU reference and analytic original-camera truth generator pass their owning
checks and independent review. The thin-scene construction controls actual Left
full-grid phases 0/0.25/0.5 and centreline widths 1/2/4 pixels; Right projections are
recorded independently. A 720p/360p family is exported with 114 audited artifacts,
including a genuine thin surface lost by coarse Right centre sampling. This is
synthetic development truth, not physical held-out camera accuracy.

The guided native stage is being qualified separately. Slang 2026.12 does not
emit the required NoContraction decorations on this direct-SPIR-V path; the new
refinement-only build pins the official 2026.19 archive by SHA256. Its compiled
43 floating arithmetic operations carry NoContraction, with no relaxed precision,
fast-math or FMA, while 62 existing shader binaries remain unchanged. Coarse and
refinement compiler identities are reported separately. The tag is unsigned;
archive-digest verification is not a claim of signed-tag verification. Native
guided fixtures pass exact discrete checks and conditional operation-specific
float bounds at all three radii with/without subpixel refinement. Resource/lifetime
checks and a 303-frame synchronization stress pass. Global arithmetic boundaries,
repeatable performance, recovery/ROI and useful world quality remain open; see
the separate refinement evidence for the substantial quality gap against SGM.
