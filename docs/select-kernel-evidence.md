# Stereo selection reduction evidence

Measured on 2026-10-27 on AMD Radeon 8060S Graphics (RADV STRIX_HALO), using the standalone Slang RHI Vulkan build. This implementation supplies the benchmark baseline and shared Slang reference for the planned Daxa + Slang runtime; these results do not establish a Daxa implementation or its performance.

## Change and parity

The selection kernel assigns four adjacent lanes to one pixel, scans disparity candidates at stride four, and reduces each lane's best two costs through shared memory. Equal best costs select the lowest disparity index. All 256 threads, including padded pixels, participate in both reduction barriers. Validity thresholds, subpixel refinement and confidence arithmetic remain unchanged. Native dispatch remains `ceil(pixels/64)` groups; SlangPy requests four threads per pixel to obtain those groups.

The direct selection suite passes 25 tests: nine CPU structural/property cases and sixteen executed Vulkan cases, covering D=1,2,3,31,32,64,128,256, subpixel on/off, 67-pixel padding, tied minima, second minima within/across lanes, all-UINT_MAX costs, zero disparity and the 10000 rejection threshold. GPU disparity/confidence/validity match the previous scalar shader bit for bit, with an additional comparison against the unchanged scalar Python reference. Standalone CTest passes 5/5.

## Isolated selection kernel

Each case uses the same deterministic, retained 1280×720 uint32 sum-cost volume, with 3 warmup and 30 measured dispatches. GPU times come from timestamp queries. Submit-to-completion wall times are reported separately; source upload, setup and final plane downloads are excluded from both timing scopes. Baseline and candidate are separate processes, with execution order alternated across D. All three final output planes match exactly for each case.

| Disparities | GPU p50 baseline → candidate (ms) | GPU p95 baseline → candidate (ms) | Submit-to-completion p50 baseline → candidate (ms) |
|---|---:|---:|---:|
| 32 | 3.8938 → 0.7879 | 3.9948 → 0.8984 | 3.9765 → 0.8703 |
| 64 | 6.1566 → 1.5888 | 7.5444 → 1.6841 | 6.3100 → 1.6402 |
| 128 | 31.0485 → 5.6838 | 33.5602 → 6.3355 | 31.4536 → 5.8248 |

These are isolated selection measurements, not complete stereo latency. The retained D128 costs alone occupy 471,859,200 bytes; total reported resident buffer payload is 482,918,400 bytes. Each native child ran with `PYTHONHOME=/nonexistent` and empty `PYTHONPATH`; recorded linkage contains the standalone Slang compiler and system libraries, with no Python, SlangPy or SGL dependency.

Input identity uses FNV-1a64 over generated little-endian cost bytes. The fixture descriptor SHA256 binds dimensions, seed, that fingerprint and generator source; it is not a SHA256 of the cost bytes. Commands, distributions, hashes and raw samples are in [comparison.json](../artifacts/native-select-20261027/comparison.json) and the six adjacent timing JSON files.

## Full single-reference native stereo

These independent measurements run complete Census, cost construction, four/eight-path SGM and selection on synthetic 1280×720 input. Times are arithmetic means of 20 iterations, including first allocation, host uploads and complete disparity/confidence/validity readback each iteration. Engine initialization is excluded. They are not GPU timestamp medians and should not be combined with the isolated-kernel timings. Baseline is commit `451040e`; only selection differs in the candidate.

| Disparities | Paths | Transfer-inclusive mean baseline → candidate (ms) |
|---|---:|---:|
| 32 | 4 | 59.1740 → 55.6820 |
| 32 | 8 | 70.1505 → 67.0930 |
| 64 | 4 | 64.8440 → 60.2520 |
| 64 | 8 | 79.0905 → 75.5400 |
| 128 | 4 | 111.6670 → 81.1435 |
| 128 | 8 | 134.7985 → 109.8910 |

All eighteen baseline/candidate plane pairs match byte for byte. See [report.json](../artifacts/select-full-stereo-20261027/report.json). This fixture does not include camera delivery, rectification, left-right consistency, metric calibration, RGB synthesis or temporal integration. It does not qualify sustained 60 Hz operation or physical depth accuracy.

## Validation and artifact integrity

Positive Khronos core validation executed the selection benchmark with a partially filled final workgroup, the actual stereo→bridge→compositor→consumer fixture, and the native stereo self-test. All three processes returned zero, recorded the Khronos layer loaded, and emitted zero captured validation diagnostics across stdout and stderr. This does not qualify synchronization validation or GPU-assisted validation. See [validation report](../artifacts/native-select-core-validation-20261027/report.json).

Documentation verification recomputed all 28 source/binary/output SHA256 entries in the isolated comparison manifest and compared all 18 full-stereo plane pairs. Current candidate shader SHA256: `43bcd18b85cd6330605a4836bfe8bed7c845f0a712d4340c002ebba4201168fb`.

| Report | SHA256 |
|---|---|
| Isolated comparison | `56df0b46916ee042b26cbc2e8d4c57539a868440847860079963adf3ff0afdf6` |
| Full native stereo | `c1435f8fa2099baae486f7967f993f1230c3cacee99f0340ff39ad5dd2dc6e54` |
| Positive core validation | `b98dfc1ea2b7119ba54b9a9cce914de867b3bcf8a2a9047d4ee040903cb1a235` |

Artifacts are local ignored experiment outputs. Preserve the referenced files alongside this document when transferring or reproducing this evidence.
