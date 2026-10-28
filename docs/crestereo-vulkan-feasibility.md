# CREStereo portable-inference feasibility: CPU smoke and conversion attempt

This is an exploratory offline result, not a Vulkan inference implementation or a
neural-versus-SGM comparison. A fixed 320×240 CREStereo export executed on the
ONNX Runtime CPU provider, while an attempted pnnx conversion did not produce a
usable ncnn model. No neural GPU latency, model accuracy, output validity,
confidence or live throughput was established. The 320×240 input and its
nonuniform 720p resize are not equivalent to full-resolution stereo.

## Model and inputs

The [Qualcomm model card](https://aihub.qualcomm.com/models/crestereo) describes a
rectified stereo input and disparity output. The tested release archive was
identified as ai-hub-models v0.61.0, source commit
`5975a79b55b40f5cbc61f3ac5e52abe47d9d8bd5`; the [upstream repository](https://github.com/quic/ai-hub-models)
is the source of that release. Archive SHA-256:
`acfe65a00216c9eafc274f55b64acfd51b8d69c14c3e15a1425066b6dec6b549`.
The archive members used had these hashes:

- ONNX graph: `0f0c59b353c09b571508791b0ecf5dcf657a389d1934a0912db5d2c52e897de9`
- External weights data: `8e09346b8300891f72440b436eba8b1106f3eafd49fb7d0bad490ae6857dfd82`
- Metadata: `c142ae8a1e98d8772c1b17bad451a21665f63d932e8c799480f9e38ab91f4c7d`

The archive metadata declares two BGR float32 0–1 images at 320×240, two
at 160×120 and one 320×240 float32 output. The
[original CREStereo demo](https://github.com/megvii-research/CREStereo/blob/master/test.py)
constructs half-size inputs with bilinear interpolation and `align_corners=True`.
A geometric witness remains necessary to establish the packaged output sign and
units and the consequences of the resize before using it as metric depth.
The license and redistribution terms of the actual weights are not confirmed;
no weights or camera imagery are included in this repository.

## CPU smoke

[`benchmarks/crestereo_cpu_smoke.py`](../benchmarks/crestereo_cpu_smoke.py)
executed one private recorded ZED stereo frame repeatedly. It reads fixture RGB
planes, verifies their bytes against the fixture manifest, resizes each from
1280×720 to 320×240 without preserving aspect ratio, converts to BGR and `[0,1]`,
and creates half-size inputs with aligned-corner bilinear sampling. It resizes
the output to the source grid and multiplies by four as an explicitly unverified
pixel-disparity interpretation. This interpolation is not a validity-preserving
metric-depth conversion. Fixture hashes bind the bytes used but do not establish
calibration or physical truth. Model and script are hash checked before and after;
those checks do not create an atomic model-loading snapshot.

The recorded invocation used ONNX Runtime 1.31.1 CPUExecutionProvider only,
4 intra-op threads and 1 OpenCV thread, Python 3.13.12, NumPy 2.5.4 and OpenCV
5.0.0. There were two excluded warmups and ten serialized measured iterations;
session creation was separately measured at 599.55 ms. Median host times:

| Scope | Median |
|---|---:|
| CPU input preprocessing | 8.60 ms |
| CPU inference | 1034.87 ms |
| Output validation and resize | 0.48 ms |
| Combined in-memory iteration | 1043.71 ms |

File loading and session initialization are outside the measured interval.
These timings are not comparable to GPU-resident SGM timings. Ten iterations of
one image provide no sustained or tail qualification. Output was finite, but no
claim is made about correctness, disparity units or depth accuracy.

The source hash for this run is
`0b09adc2c850154c1a1555c5c7d9118e6495420705270684ab638a0767534296`;
report SHA-256 `3a865097b473a17d57f97f214d8845a34efbe18bcc4665b59d934a3daa6f0f62`.
The local report `/tmp/stereoforge-crestereo-cpu-smoke-reviewed.json` contains
raw samples and private frame-content hashes; it is temporary and not committed.
Focused CPU adapter tests passed: `TMPDIR=/dev/shm .venv/bin/python -m pytest -q tests/test_crestereo_cpu_smoke.py` (5 passed).

## ONNX graph and pnnx attempt

ONNX 1.20.1 reembedded the external weights and validated the inlined graph. The
pnnx diagnostic for that graph reported 8,375 ONNX native nodes,
including 120 GridSample, 832 Conv and 426 Concat nodes. This is a graph inventory,
not proof of ncnn operator support or Vulkan execution. The inlined graph hash was
`30d02ca0be3b3ebbfc0db45af4f4749a9f0d9a1fb1c2c0283f46157b765cd44a`.
The attempt used the isolated `pnnx==20261020` wheel (SHA-256
`d50d921e5897293d616444191c1a2dc5366423c81bdb4bd22a985fe8071cf17c`)
and executable SHA-256
`4db234cb90a27dc15e04e0d0edf5706c7a488ab8e6235cf8c146c6ec80fd9355`;
its upstream source commit and build identity were not established. From the
temporary audit directory it ran:

```bash
timeout 60s /usr/bin/time -v /tmp/stereoforge-onnx-audit-venv/lib/python3.13/site-packages/pnnx/pnnx crestereo-inline.onnx fp16=0 > pnnx-inline.log 2>&1
```

The 60-second-bounded conversion attempt reached pnnx's ncnn pass and logged
`write onnx failed` and `cat along batch axis 0 is not supported`; its ncnn param
and binary files were zero bytes. The timeout/exit status was not retained in the
log itself, so the log alone does not attest the exact termination cause. This
attempt yields no usable conversion and does not prove that every possible export
or conversion of CREStereo is unsupported. The temporary log hash is
`82794161ec84176ccc63f71d6d53c478daab8ba82839bade96c05634f756df4d`.

## Next gates

Choose a model whose weights and redistribution terms are verified and whose
operators are supported in a reproducible Vulkan conversion. Check native-versus-
converted numerical outputs on identical inputs, including a signed-disparity
geometric witness; independently test resizing, calibration, masks and missing
confidence. Then run analytic and recorded-camera same-input comparisons against
SGM. Report independent truth coverage and thin/edge errors separately from recorded
camera disagreement. Time actual GPU execution, preprocessing, copies and waits
with explicit device/resource ownership; compare matched resident and host-output
lanes. Until those checks pass, no neural backend or performance advantage is
qualified and the existing classical path stays the baseline.
