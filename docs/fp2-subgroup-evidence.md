# FP2 subgroup reduction experiment

The isolated subgroup-assisted recurrence passed its scoped GPU parity gate, but regressed in matched timing against the existing parallel shader. GPU p50 in the parallel/subgroup/parallel sequence was 91.265 / 94.528 / 90.685 ms: the candidate was approximately 3.6–4.2% slower than the two controls. Retain the prior parallel implementation for this diagnostic.

The candidate has all 128 lanes take part in every path pixel. Each subgroup calculates a minimum, its elected leader performs a shared atomic minimum, and uniform workgroup barriers protect initialization and consumption. It does not assume a mapping from local lane numbers to subgroup IDs. It is not work-proportional compact recurrence. The test is synthetic and repeated-input; it does not establish live-camera throughput, transfer-inclusive latency, physical depth quality, unrestricted full SGM, or native/subpixel equivalence.

## Parity and device

The 7×5 and 320×180 GPU fixtures checked both eyes' four packed paths, sums and padding against an independent dense CPU oracle. The pinned analytic 1280×720 fixture compared final GPU raw selections and observations with frozen CPU selections and the integrated GPU output hashes; it did not compare 720p per-path arrays. The source ID was 7 and metric values use the scoped integer-disparity LUT described in [resident evidence](fp2-resident-evidence.md). The 720p host requested 656,178,404 GPU buffer bytes; requested bytes do not measure peak or total driver memory.

On AMD Radeon 8060S / RADV, the device reported subgroup size 64, supported-stages mask 16383 and supported-operations mask 1791; compute subgroup BASIC and ARITHMETIC passed admission. The reported limits were 1024 workgroup invocations, 1024 workgroup x size, 65,536 shared-memory bytes and 4,294,967,295 x dispatch groups. Both hosts fail closed when required subgroup capabilities or workgroup limits are unavailable. CPU models included wave32 and wave64 with shuffled lane partitions; only the reported AMD device was exercised on GPU.

A CPU decoder checked that the generated SPIR-V contains GroupNonUniform and GroupNonUniformArithmetic capabilities, subgroup-scope unsigned minimum and election, a workgroup-storage unsigned atomic minimum and five workgroup control barriers. Mutations removing each checked instruction/capability were rejected. This is a structural instruction and scope check, not a proof of full control flow or device behavior.

Parity and the matched run used the nonblocking shared GPU lock, exited 0 and checked fixture validation counters through teardown. Khronos instance/device validation was enabled; synchronization validation was requested but not independently attested. Loader notices are separate from fixture diagnostics. Source, binary, SPIR-V, manifest, baseline and prior-proof hashes matched before and after. Before GPU work, 311 CPU/identity and guard tests passed.

## Matched timing

One lock covered the three blocks. Each used the same pinned preuploaded analytic frame, one in-flight slot, 10 warmups and 100 timed samples; there was no timed full-image host readback. Each timing run was bound to its parity proof. GPU timestamps had 64 valid bits and a 10.019 ns period, with availability and wrap checks. Percentiles use NumPy linear interpolation. Values below are milliseconds; CPU wait overlaps GPU work and must not be added to it. Raw per-frame durations are retained, not raw timestamp ticks.

| Block | GPU p50 | GPU p95 | GPU p99 | GPU max | CPU record p50 | CPU submit p50 | CPU wait p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parallel 1 | 91.265 | 93.633 | 94.538 | 97.274 | 0.638 | 0.135 | 91.675 |
| Subgroup | 94.528 | 97.393 | 99.949 | 99.959 | 0.668 | 0.148 | 94.915 |
| Parallel 2 | 90.685 | 94.803 | 95.392 | 95.518 | 0.628 | 0.118 | 90.883 |

## Identities

| Artifact | SHA-256 |
|---|---|
| Shader source | `bc7a272abe6a2c2065c49e8e16451e89f4dcd104f08fd4f7a8cd01e062142a16` |
| 720p host source / binary | `462258cba5ffc27837d7076085698ab2648d7a7742e60c916013e59ef5e1c7a1` / `29390a32d85ebbc72f8398c88237c43ab8bd9b0660d33257aef241ea9453c7df` |
| Tiny host source / binary | `da90c8d1ab6d4e6617103e12bb12848d1abe787624c932596070ef5827093c50` / `c0e7a0b508f2f6c362c1e5c11594442d94f1c92d2d5dad49886b4738cb5590bc` |
| Diagnostic / matched harness | `b541cbe96ec5c1669407110e11f5c810310534e5eea38cbe0c9e05c360044c87` / `4fa1bb72c3873f534b1e8381739aeb33fc40a16d8b0b343b2c837eee13f9a0b4` |
| SPIR-V structural checker | `7ef96ad5d7387e3baaa3f4ba39b4d5a875bbc18a4e273e3386058bb6c0cb2639` |
| Parallel / subgroup parity proof | `ce5610937f578f7e7a15c302374bb5efa69687f041739f282f0b63cfe5f07c18` / `0510a49bd325684a228a5b3e26ed2dedf4b910a03df5bae23472013fe926161c` |
| Matched summary | `c892f943d5783955b07a740ebd376ffece04a0851f4b21974bf22621ae318e94` |
| Parallel 1 / subgroup / parallel 2 samples | `344fcb744659a40c607432c1f2c8041550ca2ae22e58b9bc9967ed321967367b` / `341b6c30d2011e2c3eecb2ad20f07e5e0dc0d753a1ce6a2ecfab9cc9fbfc2ed7` / `5018808598255efe3b6e630efa55f85797efdceaef99f6e0cfdc6989ad3dc852` |

The captured `summary.json` v1 incorrectly labels the candidate as using a tree reduction. The shader actually uses subgroup minimum plus one shared atomic minimum per subgroup leader, as described above. The captured artifact remains unchanged. After this run, the matched harness was corrected to record separate old tree-reduction and new subgroup/atomic labels; its current SHA-256 is `52d7e3a52a6abf01f2ea5d3c771a0c275ee52d04b1f422f9ea7333febe31d5e2`. The table records the original harness SHA used in the run. The subgroup, parallel and timing guard suites passed after the correction; the GPU run was not repeated. A reproduction with the corrected harness creates different harness-bound result hashes.

The ignored local `artifacts/fp2-subgroup-20261029-v1/` and `artifacts/fp2-subgroup-matched-20261029-v1/` directories retain identities, exit codes, logs, results and per-frame durations; the input payload is separately retained in `artifacts/fp2-720p-cpu-20261028-v1/prepared`; they may not exist in a fresh checkout. The prepared manifest SHA is `7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17`; the pinned baseline result SHA is `e690b1895b63cf2c9c158419bf0662e7e316426d1d3c164f8c078d3700090041`.

## Reproduction

Use the configured build, validation layer and pinned local artifacts described in [parallel evidence](fp2-parallel-evidence.md). Run all CPU and guard tests before GPU work:

```bash
TMPDIR=/dev/shm cmake --build build/daxa --target fp2_parallel_fixture fp2_parallel_tiny fp2_subgroup_fixture fp2_subgroup_tiny -j 4
TMPDIR=/dev/shm PYTHONPATH=python:. .venv/bin/python -m pytest -q \
  tests/test_fp2_subgroup_cpu.py tests/test_fp2_subgroup_hotpath.py tests/test_fp2_subgroup_matched.py tests/test_fp2_subgroup_spirv.py \
  tests/test_fp2_parallel_cpu.py tests/test_fp2_parallel_hotpath.py \
  tests/test_fp2_resident_cpu.py tests/test_fp2_resident_hotpath.py \
  tests/test_fp2_integrated_cpu.py tests/test_fp2_integrated_hotpath.py \
  tests/test_fp2_metric_lut_cpu.py tests/test_fp2_selection_cpu.py tests/test_fp2_selection_hotpath.py \
  tests/test_fp2_invalid_index_hotpath.py tests/test_fp2_720p_prepare.py \
  tests/test_fp2_720p_diagnostic.py tests/test_fp2_720p_hotpath.py \
  tests/test_fp2_census_cpu.py tests/test_fp2_census_hotpath.py \
  tests/test_fp2_image_cpu.py tests/test_fp2_image_hotpath.py \
  tests/test_fp2_slots_cpu.py tests/test_fp2_slots_hotpath.py \
  tests/test_fp2_prototype_cpu.py tests/test_fp2_dispatch_hotpath.py
```

Run fresh subgroup parity using the locked command in [resident evidence](fp2-resident-evidence.md), substituting `benchmarks/fp2_subgroup_diagnostic.py`, `build/daxa/fp2_subgroup_fixture`, and `build/daxa/fp2_subgroup_tiny`. Use a fresh output directory and preserve exit, logs and before/after identities. Stop on parity failure. Then run the proof-bound comparison with fresh output:

```bash
proof=/path/to/fresh/subgroup-parity/result.json
proof_sha=$(sha256sum "$proof" | cut -d ' ' -f 1)
TMPDIR=/dev/shm PYTHONPATH=python:. .venv/bin/python benchmarks/fp2_subgroup_matched.py \
  --output-root artifacts/fp2-subgroup-matched-new \
  --new-proof "$proof" --new-proof-sha256 "$proof_sha"
```

The comparison binds the existing parallel proof, candidate proof, current binaries, input and baseline, sets the AMD ICD and Khronos layer, requests synchronization validation, holds `/tmp/stereoforge-gpu.lock` nonblocking and applies a 300-second process-group watchdog per block. Retain exit statuses and inspect device and validation logs. Synchronization validation activation must be independently established before claiming it.
