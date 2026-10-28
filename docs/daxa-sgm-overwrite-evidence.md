# Packed SGM overwrite experiment

The experimental `packed-specialized-overwrite` variant removes the separate
packed-penalty clear by overwriting the first path's contribution. In 24
single-reference, repeated synthetic-frame runs, its GPU graph mean was lower
in every paired cycle. These are exploratory results on a shared GPU, not a
dual-eye pipeline, live-camera, quality, or 60 FPS qualification.

## Matched single-eye timing

`artifacts/sgm-overwrite/run-abba.sh` runs three A-B-B-A cycles per eye:
A is `packed-specialized`, B is `packed-specialized-overwrite`. Each result
is a separate process with 20 warmups and 100 measured frames, retained luma,
1280×720, D128, four paths, P1=8/P2=32, subpixel on, wave32 and two slots.
The reported scope excludes capture, source upload, rectification, the other
reference, LR consistency, and metric calibration. The table averages the two
per-run GPU graph means of each variant within each cycle; delta is B minus A.

| Eye | Cycle | A mean (ms) | B mean (ms) | Delta (ms) | Delta (%) |
|---|---:|---:|---:|---:|---:|
| Left | 1 | 15.3786 | 13.1460 | -2.2326 | -14.52 |
| Left | 2 | 15.3930 | 13.2351 | -2.1579 | -14.02 |
| Left | 3 | 15.3987 | 13.2287 | -2.1700 | -14.09 |
| Right | 1 | 15.4123 | 13.1628 | -2.2494 | -14.60 |
| Right | 2 | 15.4291 | 13.2850 | -2.1441 | -13.90 |
| Right | 3 | 15.4176 | 13.2257 | -2.1919 | -14.22 |

The source reports and logs are `artifacts/sgm-overwrite/abba-{left,right}-<cycle>-<run>-<variant>.{json,log}`.
They retain per-run mean, p50 and p95, not per-frame samples; these data do not
support a pooled p99. Timing disabled Vulkan validation. Reports identify
Radeon 8060S, RADV Mesa 26.2.4, the same device and driver UUIDs, and the same
compiler and shader source pair. The scripts do not record a captured source
hash for the generated input; their common benchmark configuration is not an
independent input-identity attestation.

The separate `artifacts/sgm-overwrite/logs/gpu-busy-{before,during,after}.txt`
snapshots read 3, 95, and 89 respectively. The process and clock snapshots in
the same directory show the benchmark and desktop clients on the shared GPU.
These aggregate snapshots do not isolate competing work or establish clean
admission for each cycle. No live throughput or dual-eye saving follows from
summing the single-eye results.

## Correctness boundary

The validation-enabled `artifacts/sgm-overwrite/run-parity.sh` exercises both
references and wave32/64 at 37×11 for D31/P4/P2=32, D33/P8/P2=64,
D128/P4/P2=32, and D255/P8/P2=31. The validation-enabled
`artifacts/sgm-overwrite/run-720-parity.sh` checks both references at
1280×720/D128/P4/wave32. In every one of these 18 paired configurations,
the dumped costs, packed penalties, sums, disparity, confidence and validity
files are byte-identical between variants; reports record integer-reference
checks and zero validation errors after teardown. The result files use the
`left-*`, `right-*`, and `720-{left,right}-*` prefixes in that directory.
`dynamic-parity-wave{32,64}.log` records the separate exact GPU comparison
with changing input and two-slot reuse; the check also requires zero validation
warnings and errors. These are fixture-specific single-reference comparisons.

The CPU confidence bit-exact contract remains failing for both baseline and
candidate on both wave sizes: the `packed-specialized32-contracts.log`,
`packed-specialized64-contracts.log`, `packed-specialized-overwrite32-contracts.log`
and `packed-specialized-overwrite64-contracts.log` files record the D2/P4 example with GPU
`0x3de38e38` and CPU `0x3de38e39`, a one-ULP discrepancy for 7/63.
Baseline/candidate equality on the tested GPU fixtures does not resolve that
CPU floating-point contract or prove general numerical equivalence.

## Bound identities and reproduction

Every timing report records the following source-pair and compiler SHA-256
values. They are recorded source identities, not a complete immutable build,
binary, SPIR-V, driver, dependency, or input freeze:

* `shaders/daxa/stereo.slang`: `95cbb82d502c0f2220450bcdf72e52637ef282881b85c7797ebd8fab2d842b7b`
* `shaders/ops/sgm.slang`: `76518ce8e28480655d885b7e2fceb3710d52cffb58ed44ab2f1e485c45431b83`
* Slang compiler: `82e29ecd96714c64a31021160eaff4bf8604f44934e9f7b98448e962a5d12dad`

The recorded scripts expect their original container mounts: `/out` contains
the built binaries, `/evidence` is a writable fresh artifact directory, and
`/radv`, `/drm`, `/validation` contain the matching runtime libraries. With
those mounts configured and an independently reviewed build on the target GPU,
run `bash /evidence/run-parity.sh`, `bash /evidence/run-720-parity.sh`, and
`bash /evidence/run-abba.sh`. Use a fresh evidence directory and preserve its
script copies; these scripts write fixed filenames. Recorded script hashes are
`ba05dc8044f99c2ff6051c5d2d05181bb7cb2c73e7272889a200f626b13677f4`,
`1d916d549934ca235d708d5641e1f17cf220ed8d4df92743fb7f50c4b58cf8be`,
and `fe2d0e3535b1b6e8f77502cc415b9146591c910b540b804a704de2c59d912031`,
respectively. Clean matched runs and full pipeline validation remain necessary.
With the same runtime environment and validation layer available, the separate
dynamic check is `/out/daxa_stereo_overwrite_parity 32` and then
`/out/daxa_stereo_overwrite_parity 64`.

## Separate compact-window feasibility

`artifacts/daxa-refinement-20261028/compact-sgm-first-pass-work-v2.json`
(SHA-256 `94796bfa1b5d6c0cc18e3345657aa0222d59b03ab1e95dae882e810d4c585bfa`)
is an offline CPU estimate on one analytic fixture, unrelated to the GPU
overwrite timing above. Independent prior-valid windows and full-D fallback
produce 83,328,882 interval slots versus 235,929,600 dense slots across both
eyes (35.32%); fallback covers 32.69% of pixels. Under the stated hypothetical
one-pixel-per-wave model, separately rounded wave32 and wave64 slots total
116,826,624 and 156,527,616. These counts include slots without valid
opposite-image support and are neither GPU timings nor measured instructions.

Truth is loaded only after selecting windows. On the fixed clean truth support,
553/3,164 left and 342/3,164 right thin-region pixels lack an integer candidate
within ±1 disparity pixel. Representability for the remainder is an optimistic
necessary condition, not recovery: it does not establish candidate support,
matching, LR acceptance, confidence, or geometric quality. Wider intervals or
unrestricted fallback may change both the work and quality trade-off.

## Dual-eye canonical 720p experiment

A separate experiment runs sequential independent left/right SGM, LR
consistency, and metric metadata through the native depth pipeline. It uses
one repeated **synthetic canonical record** at 1280×720/D128, four paths,
P1=8/P2=32, subpixel enabled, wave32, and two slots. The combined timestamp
spans canonical producer upload through final metric output, including queue
gaps. It does not include camera capture, host-side canonical file loading,
wide RGB-D synthesis, ROS delivery, or camera-to-output age. It is not a live
throughput or 60 FPS result. The current default remains `packed-specialized`;
`packed-specialized-overwrite` is opt-in.

For the validation-enabled parity runs, each variant processed three frames
after one warmup. The nine dumped planes—LR disparity, confidence and
validity; metric depth, confidence and validity; provenance, source IDs and
ages—are byte-identical between the two variants at 720p. Both reports record
zero Vulkan validation errors and warnings through teardown. These are
same-input GPU-to-GPU comparisons, not independent physical depth validation.
See `artifacts/sgm-overwrite/depth-720-packed-specialized*.json` and matching
dumped planes. The candidate reports `effective_estimator: null` and
`estimator_identity_status: experimental_variant_not_qualified_for_canonical_estimator_identity`;
it must not be treated as a qualified baseline estimator profile.

Three A-B-B-A cycles use 20 warmups and 100 measured frames per process, with
Vulkan validation disabled. A is the baseline and B the overwrite candidate.
The table averages each cycle's two per-run values; its p99 column is the
**average of per-run p99s**, not a percentile computed from pooled samples.
Values are GPU combined queue spans in milliseconds.

| Cycle | A mean | B mean | Mean B − A | A p99 | B p99 |
|---|---:|---:|---:|---:|---:|
| 1 | 31.4874 | 27.2770 | -4.2105 | 32.1493 | 27.9083 |
| 2 | 31.4434 | 27.1402 | -4.3032 | 32.4055 | 28.0599 |
| 3 | 31.5420 | 27.2950 | -4.2470 | 32.1245 | 27.8249 |

Source reports are `artifacts/sgm-overwrite/depth-abba-<cycle>-<run>-<variant>.json`.
The shared desktop GPU may have competing work; these runs do not establish
controlled GPU admission or production stability. The candidate is faster on
this fixture but does not meet the 16.67 ms interval, and the earlier scalar
confidence exactness failure remains unresolved.

The recorded scripts use the `/out`, `/evidence`, `/radv`, `/drm`, and
`/validation` container mounts described above. With a reviewed build and a
fresh writable evidence directory containing the same synthetic bundle and
script copies, reproduce parity and timing with:

```bash
bash /evidence/run-depth-720-parity.sh
bash /evidence/run-depth-abba.sh
```

The scripts write fixed filenames; preserve earlier evidence separately.

## Post-refactor revalidation

The initialization entry points and recurrence now live separately in
`shaders/daxa/stereo_overwrite.slang` and `shaders/ops/sgm_overwrite.slang`.
The baseline `stereo.slang`, `stereo_layered.slang`, and `sgm_specialized.slang`
are restored to their committed source. A CPU guard compares the separate
recurrence with the frozen baseline body and permits only the change from add
to overwrite. Post-refactor source hashes are recorded in
`artifacts/sgm-overwrite/post-refactor/source-sha256.txt`.

The old and post-refactor build directories contain byte-identical SPIR-V for
all four initialization entries, both 8/16-bit left and right specialized
aggregation entries, and all four layered aggregation/selection entries.
This is a binary comparison of those twelve entries, not proof about every
shader in the build. The changed source identities are recorded separately;
results from the earlier build should not be presented as a new-build run.

The post-refactor dynamic checks, in
`artifacts/sgm-overwrite/post-refactor/dynamic-wave32.log` and
`dynamic-wave64.log`, pass exact candidate-versus-baseline output comparisons
for changing inputs and repeated two-slot reuse on both eye references. They
cover D3/D33, four/eight paths and 8/16-bit packed layouts, with Vulkan
validation enabled. The post-refactor 720p dual-eye parity reports and nine
plane pairs under `artifacts/sgm-overwrite/post-refactor/depth-720-*` also
match byte-for-byte; both reports record zero Vulkan validation errors and
warnings through teardown. These are synthetic GPU comparisons, not physical
accuracy measurements. Candidate `effective_estimator` remains null and its
identity remains explicitly unqualified.

The post-refactor A-B-B-A reports are
`artifacts/sgm-overwrite/post-refactor/depth-abba-<cycle>-<run>-<variant>.json`.
Each process uses 20 warmups and 100 measured frames, with validation disabled;
the same synthetic canonical input and combined queue-span scope above apply.
For each cycle and variant, the following ranges cover the three averages of
the two corresponding per-run statistics. Percentiles are therefore averages
of per-run percentiles, not pooled percentile estimates.

| Combined GPU span | Baseline A, ms | Overwrite B, ms |
|---|---:|---:|
| Mean | 31.581–31.743 | 27.048–27.247 |
| p50 | 31.514–31.670 | 27.012–27.212 |
| p95 | 32.112–32.311 | 27.485–27.686 |
| p99 | 32.244–32.851 | 27.594–28.043 |

These measurements use the shared desktop GPU and do not qualify sustained
operation or 60 FPS. The known baseline and candidate CPU confidence-bit
failure described above is still unresolved. The corrected full CPU suite is
pending; focused post-refactor pair and stereo checks passed, but they do not
replace the full suite.

The post-refactor scripts under `artifacts/sgm-overwrite/post-refactor/` use
`/fixture/native-input.json` for the canonical record and write to `/evidence`;
copy them to a fresh evidence directory and supply the matching `/out`,
`/fixture`, `/radv`, `/drm`, and `/validation` mounts before running
`run-depth-720-parity.sh` and `run-depth-abba.sh`.

## Physical ZED 2 ROS/Zenoh A-B-B-A observation

Four sequential 25-second camera runs used A=`packed-specialized` and
B=`packed-specialized-overwrite`, in A-B-B-A order. The publisher captured
720p per eye at nominal 60 Hz and ran sequential independent D128/P4 stereo,
P1=8/P2=32, subpixel on, wave32, LR consistency and metric output on Radeon
8060S (RADV Mesa 26.2.4). Depth `32FC1` and rectified RGBA were exported to
host memory and sent to a separate ROS Jazzy subscriber using
`rmw_zenoh_cpp`, best-effort keep-last-one QoS. Two processing slots, five
warmup frames, subscriber discovery wait and opt-in thread CPU timing were
used; Vulkan validation was disabled for these timing runs. This was an
uncontrolled live scene on a shared desktop GPU, not same-frame replay,
independent depth-quality validation, or a sustained 60 FPS qualification.

The table uses the summary's capture source-frame delta, publisher ring-drop
counter, and matched unique in-window subscriber deliveries. Delivery counts
are equal for depth and RGB in these runs. Source and delivery counters use
different window boundaries, so they should not be subtracted as a single
loss ledger. Ingress latencies are in milliseconds, p50/p95/p99.

| Run | Variant | Source frames | Ring drops | Delivered per output | Hz | Depth host ingress → arrival | RGB host ingress → arrival |
|---|---|---:|---:|---:|---:|---|---|
| 1 | A | 1,497 | 776 | 720 | 28.80 | 62.92 / 70.38 / 71.34 | 63.90 / 71.03 / 72.02 |
| 2 | B | 1,493 | 618 | 875 | 35.00 | 50.90 / 58.17 / 59.42 | 51.82 / 58.96 / 60.08 |
| 3 | B | 1,499 | 593 | 904 | 36.16 | 49.38 / 56.42 / 57.71 | 50.33 / 57.41 / 58.59 |
| 4 | A | 1,497 | 714 | 784 | 31.36 | 57.73 / 65.01 / 65.87 | 58.68 / 65.96 / 67.01 |

The B runs delivered more outputs in this short sequence, but changing scene
and desktop load prevent attributing the whole difference solely to the
kernel. The camera timestamp mapping was unverified: these timings begin at
**host ingress**, exclude earlier camera/USB time, and do not establish
exposure-to-subscriber age. All four summaries are unqualified because the
clock mapping is unverified, no frozen delivery policy was supplied, and
storage accounting is incomplete. The publisher logs also report
`SensorCapture` connection failure for the camera's sensor interface; IMU
operation was not established in these runs. This observation does not
identify the cause of that failure.

The candidate was selected explicitly with `--stereo-variant
packed-specialized-overwrite`, remains experimental, and records its
`effective_estimator` as unavailable with reason
`experimental_variant_not_qualified_for_canonical_estimator_identity`.
Baseline remains the default. Raw publisher/subscriber traces, logs and
summaries are in `artifacts/sgm-overwrite/live-abba/live-<1..4>-*`; the exact
commands and runtime configuration are in
`artifacts/sgm-overwrite/live-abba/run.sh`. Subscriber stop counters include
messages outside the publisher's measurement window; use the table's matched
in-window deliveries for the reported rate.
