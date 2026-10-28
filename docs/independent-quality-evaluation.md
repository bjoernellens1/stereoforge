# Independent quality evaluation

This is a cold, offline Python/NumPy workflow for portable stereo observations and independently declared metric truth. It adds no dependency or evaluation work to the native resident GPU pipeline. Copies, dump readback and hashing belong to the explicit export/evaluation boundary.

## Library workflow

Use the repository virtual environment. The current implementation exposes library APIs; it has no evaluation CLI.

```python
import json
from pathlib import Path
from benchmarks.zed_comparison.native_observation import export_native_observation
from benchmarks.zed_comparison.compare_quality import compare_quality
from benchmarks.zed_comparison import schema

# Each export needs a fresh output directory with an existing parent.
record_path = export_native_observation(
    report_path, source_record_path, dump_root, fresh_output_dir
)
records = schema.load_records([record_path])
truth = json.loads(Path(truth_document_path).read_text())
policy = schema.load_policy(policy_path)
result = compare_quality(
    records, truth, policy,
    root=record_path.parent, truth_root=Path(truth_document_path).parent,
)
```

`export_native_observation(report_path, source_record_path, dump_root, output_dir)` admits executed canonical native `full_sgm` or `coarse_refine` reports and returns a new portable record path. It checks report/input/calibration/source bindings and dump sizes/hashes, preserves exact left-half float bytes from two-eye dumps, bulk-converts U32 masks/provenance, and preserves input bytes and input identity. Missing recorded commit metadata remains unavailable. It rejects existing output directories and malformed or changed evidence.

`compare_quality(records, truth, policy, *, root=None, truth_root=None)` takes record dictionaries. Supply `root` as a common directory or an ordered list matching the records, and supply explicit `truth_root`; there is no implicit working-directory binding. Missing roots yield an unqualified missing-binding result; invalid documents, bindings or observed mutations raise `ValueError`. Methods remain separate by implementation/settings, calibration stratum and exact input identity.

`load_truth(truth, records, *, root)` returns a `TruthBundle`. Its `frame(input_identity)` context manager materializes one frame of detached immutable arrays: axial `depth_m`, fixed `sample`, optional support/visibility/occlusion/negative/boundary masks and regions, plus metadata and uncertainty/registration declarations. Ray range converts on the exact rectified grid. Unknown truth is never interpolated. Mappings close after materialization; callers should release frame arrays after use. Declared plane extents are limited to 512 MiB per frame, which does not bound temporary NumPy copies. Fingerprints are checked around materialization and once across the whole bundle after evaluation; this detects observed changes without claiming an atomic filesystem snapshot.

## Denominators and decisions

Primary completeness uses fixed sampled truth in the requested range, including known truth outside remap sampling support or disparity search. Prediction validity cannot shrink this denominator. Full sampled, geometric-supported, search-supported and same-input intersection lanes are reported separately. Unknown accepts, known-negative false accepts and unsupported accepts remain distinct. Errors use accepted sampled predictions; completeness accounts for missing predictions.

Confidence curves include whole tied-score groups in the declared direction, with at most 101 thresholds. Scores are uncalibrated. Frame p95 uses nearest rank; pooled p95 and pooled confidence curves are explicitly unavailable to preserve the one-frame memory boundary. Paired decisions use scenes/sequences rather than treating frames as independent, with deterministic hierarchical bootstrap settings recorded in the result.

The default policy is provisional and supplies no accuracy threshold. Consequently ordinary error and completeness are reportable, but threshold correctness is unavailable and there is no correctness pass. SDK parity additionally requires an executed matching `ZED SDK` observation, frozen policy, held-out bindings, partition digest, independent uncertainty/registration and required region evidence, sufficient independent units and passing paired statistics.

## Measured development fixture

The [published evidence summary](evidence/quality-plane-2026-10-28.json) binds the evaluator source hashes and full local report digest. Evaluation v2 and its freeze manifest under `artifacts/zed-task4/` cover one 128×64 analytic infinite-plane development fixture. Its 1.44 m truth is independently procedural, derived from the known pinhole setup. It is not SDK truth, physical registration or held-out data. The fixed denominator is all 8192 known analytic samples; remap support contains 8001 samples and both-stereo-visible support contains 7497.

| Executed method | Valid / fixed samples | Completeness | MAE (m) | p95 absolute error (m) |
| --- | ---: | ---: | ---: | ---: |
| full_sgm | 6960 / 8192 | 84.9609375% | 0.0114341651386 | 0.03220444679 |
| coarse_refine | 5837 / 8192 | 71.2524414% | 6.19888305e-8 | 6.19888305e-8 |

Refinement has lower error on its accepted predictions and lower fixed-support completeness in this fixture; these data do not establish overall superiority. The report remains `status=unqualified`, `parity_qualified=false`, with actual reasons `missing_executed_reference` and `provisional_policy`. These are quality measurements, not GPU FPS results.

## Verification boundary

The independent functional review (`artifacts/zed-task4/functional-review.md`) independently verified 131 tests and decoded retained depth/validity bytes to reproduce the fixture metrics. The CPU performance review (`artifacts/zed-task4/performance/review.md`) records 150 passing component tests, including 19 performance checks. The final full CPU regression passed **2,290 tests**, with 37 GPU/camera tests deselected and no skips (132.74 s).

Review-driven regressions include the boundary-F1 RED/GREEN correction for disjoint nonempty edges, rejection of candidate-only subsets of reference inputs, required independent boundary evidence, policy-loader round trips and a final truth mutation recheck that catches an earlier frame changed during later evaluation. Static/runtime guards cover bounded hashing, bulk array conversion, closed mappings, one-frame lifetimes and absence of runtime GPU/SDK dependencies.

Warm dense 1280×720 `evaluate_frame` calls measured 371.7–376.6 ms on the review host, including all 101 confidence thresholds. This is offline CPU evaluator cost with no acceptance threshold, not native GPU runtime overhead or pipeline throughput. Tiny adapter and truth timings, source hashes and measurement limits are recorded in the performance review. Scientific qualification remains open under the [implementation contract](superpowers/plans/2026-10-28-independent-quality-evaluation.md).
