# Overwrite + quad selector: development evidence

`packed-specialized-overwrite-quad` is an explicit opt-in candidate. The production default remains `packed-specialized`; the comparison below uses the separately opt-in `packed-specialized-overwrite` as A. Both eyes use independent SGM, sequentially. The candidate is rejected for paired, layered, shared-Census and derived-right schedules. Its canonical estimator identity is unavailable pending qualification.

## Exactness

On synthetic retained luma, the local ignored artifact `artifacts/sgm-overwrite-quad-parity/run.sh` describes 56 small profiles: both references, wave32/64, seven disparity/penalty/path profiles and subpixel on/off. The six exported planes (disparity, confidence, validity, reconstructed costs and sums, and actual packed penalties) match A byte-for-byte. Each case also checks the actual packed words and reconstructed integer volumes against the independent scalar oracle; the volumes are reconstructed on diagnostic readback, not directly exported GPU cost volumes. The separate `run-720.sh` case compares the same six planes for each eye at 720p. These scripts and outputs are local ignored artifacts, not part of the source distribution.

The source test `tests/daxa_stereo_quad_parity.cpp` runs six changing inputs in three two-slot batches, both references, wave32/64, odd pixel/disparity tails, four/eight paths, both packing widths, and subpixel on/off. Local `artifacts/sgm-overwrite-quad-parity/dynamic-wave{32,64}.log` records successful exact parity. `tests/daxa_depth_quad_parity.cpp` compares both stereo references and all nine dual-eye LR/depth/confidence/validity/provenance/source-ID/age planes with changing inputs and slot reuse; local `artifacts/sgm-overwrite-quad-depth/dynamic-wave{32,64}.log` records both passes. The local `run-720-parity.sh` compares those nine depth planes at 720p. Synchronization validation was enabled for the parity runs; these tests do not establish physical depth accuracy. An existing one-ULP CPU/GPU confidence assertion is a separate unresolved issue: exact comparison here is between the two GPU variants, not a claim of universal CPU floating-point bit parity.

## Matched timing

On AMD Radeon 8060S, the ignored local script `artifacts/sgm-overwrite-quad-depth/run-abba.sh` runs three ABBA cycles on the same repeated synthetic canonical 1280×720, D128, four-path record. Each arm has 20 warmups and 100 samples; the GPU was shared. The measured interval is the GPU queue span from producer upload start through metric completion, including queue gaps, not capture-to-consumer latency. Timing runs did not enable validation.

| Cycle | A mean, average of two arms | B mean, average of two arms | A individual-arm p99 range | B individual-arm p99 range |
|---|---:|---:|---:|---:|
| 1 | 27.269 ms | 25.201 ms | 27.986–28.013 ms | 25.737–25.900 ms |
| 2 | 27.219 ms | 25.378 ms | 27.626–28.211 ms | 25.975–26.263 ms |
| 3 | 27.204 ms | 25.573 ms | 27.927–28.345 ms | 26.063–26.604 ms |

The table averages arm means, not pooled percentiles; per-arm raw samples and configuration are in the ignored `depth-abba-*.json` reports. This evidence indicates a repeatable reduction on this fixture, but does not prove live 60 Hz, camera-to-output age, quality on real scenes, or sustained behavior.
