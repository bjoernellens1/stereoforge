# Controlled temporal and calibration evaluation

These offline evaluators separate stable output from new measurements. They do
not run a camera, SDK, motion estimator or native temporal GPU pipeline. Physical
data collection and SDK quality qualification remain open.

```python
from benchmarks.zed_comparison.evaluate_temporal import evaluate_temporal
from benchmarks.zed_comparison.epipolar import (
    evaluate_epipolar, compare_epipolar_profiles,
)
from benchmarks.zed_comparison.schema import validate_physical_reference

report = evaluate_temporal(
    records, truth_document, policy,
    root=record_root, truth_root=truth_root,
    protocol=protocol_document, protocol_root=protocol_root,
)
```

All roots are explicit. `root` may be one directory or a directory list matching
record order. Portable record and independent-truth admission verify hashes,
calibration, source identity, timestamps and partitions. Paths inside a physical
reference manifest resolve against the truth root, including manifests stored in
subdirectories. Input files are checked again at frame and whole-run boundaries.

The exact protocol and reference schemas are in the
[implementation contract](superpowers/plans/2026-10-28-controlled-temporal-evaluation.md).
The [physical acquisition configuration](../configs/benchmarks/zed_physical_acquisition.json)
is an unfrozen collection plan with unavailable device/reference evidence. It
does not describe collected data or schedule acquisition.

## Effective settings and observation identity

Each implementation explicitly records effective settings:

```json
{"temporal": {
  "stabilization": {"status": "available", "value": 0},
  "tracking": {"status": "available", "value": false},
  "filling": {"status": "available", "value": false}
}}
```

This object belongs in `implementation.settings`. Missing fields stay unknown;
SDK defaults are never inferred. Instantaneous quality parity requires complete
temporal-off declarations for both methods. Temporal-on, off and unknown settings
form separate method groups.

Exact repeated measurement/output content counts once even when descriptor paths
differ. Conflicting outputs for a repeated method/input are rejected. A current
source ID, zero age and observed provenance identify current-source observations;
left-aggregation-derived observations are counted separately. Reprojected, fused,
aged or old-source pixels remain historical. Missing metadata cannot establish
freshness, and a single source ID does not establish fused ancestry.

## Jitter, ghosting and recovery

The hash-bound protocol declares rig/scene motion, independent tracks, known
poses, capture-gap limits and recovery targets. Implicit pixel tracks are allowed
only for stationary, static scenes with constant sampled truth. Known motion
requires time-aligned poses; static tracked world positions must agree with the
independent geometry. Estimated motion is reported separately.

Jitter uses streaming, bounded track moments of prediction minus current truth.
Output-error and fresh-observation jitter are separate. Each track segment needs
at least two unique captures. Capture gaps reset moments; pooled percentile
distributions are explicitly unavailable. Raw depth variation is reported only
for a static rig and scene.

Ghost-compatible output requires changed independent truth on a stationary image
grid, agreement with previous truth and incompatibility with current truth.
Unknown samples are never negative evidence. Moving-rig image-grid ghosting is
unavailable without independent dense reprojection/visibility information.

Recovery targets are fixed at the declared event start. First correct geometry
and first fresh correct observation have separate times. Unrecovered targets
remain censored, including targets no longer visible. Missing correctness
thresholds leave recovery and ghost decisions unavailable.

## Physical and calibration evidence

Physical references record independent survey/scanner/LiDAR artifacts,
registration and timestamp bounds, metre units and calibration identity. A
`metadata_ready` result validates declarations and files; it does not attest
survey accuracy. Sparse references cannot claim dense boundary labels, and
interpolated references are rejected. Measured truth without reference evidence
is scoreable but cannot qualify. Analytic truth cannot claim a physical reference.

Epipolar evaluation accepts finite `Nx4` rectified correspondences and a strict
geometry/provenance envelope. It reports left-minus-right signed and absolute
vertical medians, nearest-rank absolute p95 and maximum, alongside horizontal
disparity diagnostics. Empty populations are unavailable. Factory/effective
profile deltas require the same original pixels and declared correspondence
population, selection, method and count. These diagnostics never establish metric
truth or depth quality; selecting different inliers cannot demonstrate an
improvement on the same population.

Frozen policy/protocol, held-out data, complete settings and metadata, independent
motion/correspondence, physical evidence where applicable and bounded uncertainty
are required for controlled temporal qualification. Synthetic contract tests do
not qualify SDK parity, physical accuracy, VIO, native temporal processing or
720p/60 FPS delivery. CPU evaluation cost is reported separately from GPU timing.

## Reproduce the software evidence

```bash
PYTHONPATH=python:. .venv/bin/python docs/evidence/controlled-temporal-2026-10-28.py
PYTHONPATH=python:. .venv/bin/python -m pytest \
  tests/test_zed_evaluate_temporal.py tests/test_zed_epipolar.py \
  tests/test_zed_physical_reference.py tests/test_zed_task5_review.py -q
```

The adjacent [JSON evidence](evidence/controlled-temporal-2026-10-28.json) contains
source hashes and four tiny, deterministic synthetic cases. Historical fused
output contributes zero fresh observations. The recovery fixture reports 200 ns
on its artificial 100 ns capture interval; its supplied known poses are identity
poses, not a nonzero-motion accuracy test. Separate unit tests exercise moving
tracks and static-world pose consistency. The ghost fixture reports four
ghost-compatible pixels, and a declared common correspondence population changes
from a synthetic 2 px vertical offset to 0 px. These values test evaluator
semantics; they are not camera latency or calibration measurements.

Verified locally on 2026-10-28: 2,423 CPU tests passed, 37 GPU/camera tests were
deselected, with no skips (133.46 s). Independent functional and performance
reviews passed. The local 720p CPU moment probe measured 337.9 ms initial track
admission and 3.71–4.05 ms per subsequent update with unchanged IDs. File checks
scaled linearly and all mappings closed. These are offline component costs,
excluding truth/file admission and other evaluator stages; they are not native
stereo timings or end-to-end FPS.
