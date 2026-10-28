# Fast60 full-resolution refinement implementation plan

Follow [the design](../specs/2026-10-28-fast60-refinement-design.md) and parent
[Fast60 plan](2026-10-28-fast60.md). User authorized implementation and iteration;
no further approval menu is needed. Root owns architecture/GPU/publication.
Workers preserve shared dirty files and take one explicit ownership surface.

## 2026-10-28 quality-first amendment

The [common-input SDK benchmark](2026-10-28-zed-quality-benchmark.md) now precedes
further optimization or promotion under the revised [parent plan](2026-10-28-fast60.md).
R0–R2 are implemented; R3 has conditional six-case agreement and resource stress,
with global arithmetic boundaries still open. R4 has incomplete exploratory
timing and substantial measured world-quality loss, not a qualified Fast60 result.
Preserve these artifacts and the full-resolution SGM baseline. Do not automatically
implement the old next recovery step merely because it is listed below: first
establish common-input full-SGM/SDK quality, then choose the refinement change
from missing-prior, ambiguity, support and regularization evidence. Physical and
held-out truth gates cannot be replaced by agreement with proprietary depth.

1. **R0 contract and scalar reference.** Add independent Python CPU reference and
   tests for full-frame radii2/4/8, prediction/geometry/support/independent-eye
   matching, integer uniqueness, subpixel cases, LR/metric/provenance and bounded
   recovery. Pin operation-level float checks before GPU. Review design and oracle
   independently; do not alter the frozen SGM or earlier float failure reports.
2. **R1 independent development fixtures.** Add versioned analytic two-original-
   camera ray rendering plus existing shifted-texture probes, original SFP capture
   export, per-grid independent truth, region/occlusion/mixed-footprint masks and
   immutable provenance. Test analytic plane intersections and cross-eye geometry
   independently. Start small, then render the exact720p/360p family. Physical
   cache is optional and has no metric truth; never publish camera serials.
3. **R2 persistent native guided stage.** Add new header/source/CMake target,
   shared Slang operations/thin wrapper, target remap and Census, independent
   guided match, LR/metric outputs, distinct estimator metadata, retained generic
   observations and Frame lifetime/admission tests. Initially full-frame guided
   profiles, no Temporal generalization. Reuse originals without second upload.
4. **R3 qualification boundary.** Separate worker reviews hot path, zero-warm-
   allocation/wait guards, fixed-resource work accounting and partial-submit
   lifetimes. Build/CPU tests, then freeze actual source/binaries/compiler/deps/
   SPIR-Vs/fixtures. Root alone runs scalar/integer/float-contract core+sync
   comparisons, old preprocessing regressions and ≥300-frame bounded stress.
5. **R4 radius measurements and recovery.** Matched radii±2/4/8 first, full-frame
   versus configured region fractions next. Add one bounded recovery round and
   reason/method counters, requalify changed kernels before timing. Compare actual
   truth and full independent SGM separately, with support-aware completeness and
   false accepts; no hidden quality loss by dropping hard pixels.
6. **R5 view and live gate.** Construct View against target map owner, preserve
   original RGB and strict holes, measure combined direct latency/resources.
   Qualify held-out quality and physical geometry, then ≥10-minute nominal60Hz
   unique capture→GPU→consumer with frame age/drops/queues/sharing pressure. Keep
   targets unpromoted until measured. Publish bounded evidence incrementally.

SDK availability does not block portable benchmark contracts or independent-truth
evaluation, but executed SDK evidence is required for competitive parity claims.
The isolated reference runner never becomes a CUDA dependency of the production
pipeline. Native recovery/ROI, compact-SGM and portable neural candidates remain
unpromoted until the common quality/performance gates pass.
New reusable kernels enter SinterGPU only through separately authorized candidate
intake/review; no automatic production/default integration.
