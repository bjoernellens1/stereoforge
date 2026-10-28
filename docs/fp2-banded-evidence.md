# FP2 packed-band CPU recurrence

The offline packed-band runner matches the pinned 720p synthetic CPU selections for both eyes and both the old compact intervals and full-D intervals. It allocates aggregate entries by per-pixel bucket and iterates actual band widths for cost, predecessor minima, recurrence, aggregation and winner selection. This is a CPU correctness and accounting result; it does not establish GPU performance or a 60 Hz pipeline.

Each pixel's contiguous global disparity interval `[lo,hi)` has a prefix offset and a bucket capacity of 0, 16, 32, 64 or 128 entries. Padding is allocated but excluded from the active recurrence. Scanline predecessor and current scratch arrays are bounded by the largest actual band; their lookup maps each global `d`, `d−1` and `d+1` into the predecessor's own interval. Its valid minimum supplies the P2 transition even across disjoint intervals. An empty or all-invalid predecessor restarts at the raw cost; unsupported candidates remain `UINT32_MAX` and cannot win. P1=8/P2=32, independent references, integer winners and the existing LR and metric-validity rules are used. This representation handles contiguous bands only; non-contiguous FP1 hypotheses are not merged or qualified here.

| Requested per-eye CPU accounting | Compact left | Compact right | Full-D, each eye |
|---|---:|---:|---:|
| Active winner candidate visits | 51,653,921 | 51,308,487 | 117,964,800 |
| Packed aggregate entries, including padding | 58,459,840 | 58,150,496 | 117,964,800 |
| Packed aggregate bytes | 233,839,360 | 232,601,984 | 471,859,200 |
| Padding entries | 6,805,919 | 6,842,009 | 0 |
| Four-path cost checks | 206,615,684 | 205,233,948 | 471,859,200 |
| Supported recurrence updates | 190,319,284 | 189,599,732 | 444,492,800 |
| Predecessor-minimum entries visited | 206,103,684 | 204,721,948 | 471,347,200 |

In compact mode, per-eye bucket pixel counts for capacities 0/16/32/64/128 were 0/6,836/611,868/0/302,896 left and 0/6,682/615,270/0/299,648 right. Full-D uses bucket 128 for all 921,600 pixels per eye. Besides the aggregate array, each run requested 7,372,808 bytes of offsets, 7,372,800 bytes of bounds, 14,745,600 bytes of four descriptor/support planes, 14,745,600 bytes of four winner planes and 1,024 bytes of predecessor/current scratch. The fixture processes eyes sequentially. Diagnostic path buffers and packed input-cost buffers were disabled; their requested size is zero here. These figures are software-requested array sizes and loop counts, not peak process memory, measured memory traffic, GPU allocations or timing.

The compact and full-D fixture outputs match the pinned reference for raw disparity, raw validity, LR validity, metric validity and metric depth in both eyes. No truth is loaded. Tiny differential tests compare every directional path and aggregate against the independently structured dense masked scalar oracle, including shifted, disjoint, empty and all-invalid bands, holes, boundary disparities, ties, all bucket widths, full-D, and independent-eye Census geometry. The test-only diagnostic mode materializes packed paths and Python-unpacked dense arrays; it is disabled in the fixture. The source-level check only rejects a few obvious dense allocation/loop spellings; it is not a general proof against all dense work.

The CPU reference uses an infinite/`UINT32_MAX` invalid sentinel and integer disparity. Existing native SGM uses different finite-sentinel and subpixel semantics; parity with those is a separate unmet gate. The runner's source, build, frozen v2 artifacts and native fixture plane hashes are checked, but input checking is not an atomic snapshot and all fixture data are not rehashed after execution. Concurrent mutation would invalidate the result. No physical camera, held-out scene, GPU, or latency qualification follows from this run.

The reviewed fixture exited 0. `PYTHONPATH=python:. .venv/bin/python -m pytest -q tests/test_fp2_banded.py` passed 9 tests. The ignored local result `artifacts/fp2-banded-20261028-v1/results/result.json` has SHA-256 `8868bf7ba91a2d197f7a780d1d8f1fdd95e4444893010741c4b1c987202282`; the report records pinned inputs and output hashes. Runner SHA-256 `44d54351e8d36c3edd786d91dc675e6b6ac664e313842f6786fa04a33d7f5315`; build manifest `d1aacaedf4d346603d2ade74c45f99a973b1ee7428b0259669b2e96dc6db6739`. Source SHA-256: `fp2_banded_runner.cpp` `67920b63a5ea0c1e53d6934c82b603e19899b4cb98d7d443f87188b1055fef5e`, `fp2_banded.py` `737c4ac5d89c755cb296c680a5251dcf120bdaac03c119a73701405e5276707b`, and `fp2_banded_fixture.py` `7e3969b99c8437ee2841fe59a6253d8505f8c11b2b77f0b041798ca48817c586`. Local artifacts may not exist on a fresh checkout.

To reproduce with the pinned local inputs and an empty output directory:

```bash
F=artifacts/fast60-refinement-20261028/analytic-family-720p
N=artifacts/daxa-refinement-20261028/root-frozen-numeric-matrix-v1/r2-subpixel0
A=artifacts/compact-sgm-step-20261028-v2
R=artifacts/fp2-banded-20261028-v1
O=/tmp/stereoforge-fp2-banded-repro
PYTHONPATH=python:. .venv/bin/python benchmarks/fp2_banded_fixture.py \
  --runner "$R/runner" --build-manifest "$R/build.json" --artifact-root "$A" \
  --family-root "$F" --native-report "$N/native.json" --dump-root "$N/planes" --output-root "$O"
```
