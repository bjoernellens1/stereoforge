# Compact SGM: offline development quality

A CPU-only experiment compares a frozen radius-2 coarse-guided disparity interval, with full D128 search where the prior is missing, against a full-D control on the same analytic 720p development fixture. Both independently compute left and right four-path SGM with P1=8, P2=32 and integer-only disparity, then apply a one-pixel LR tolerance. No GPU execution, physical or held-out quality, or 60 FPS claim follows from this test. The runner is `benchmarks/compact_sgm_runner.cpp`; the harness is `benchmarks/compact_sgm_quality.py` (`--help` lists its required inputs). The compiler command and source/binary hashes are embedded in the local ignored reports `artifacts/daxa-refinement-20261028/compact-sgm-quality-offline/{r2-v3,full-v3}.json`.

This CPU experiment uses infinity for unsupported candidates. Native SGM uses a finite invalid cost, so this full-D control is not a byte-equivalent native SGM baseline. Selection rejects restricted interval endpoints; both controls disable subpixel fitting.

| Metric | Radius 2 left / right | Full-D control left / right |
|---|---:|---:|
| Correct metric completeness, all | 94.884% / 94.931% | 95.339% / 95.371% |
| Correct metric completeness, thin region | 44.501% / 45.480% | 60.714% / 62.263% |
| Thin-region LR false accepted / total LR accepted | 847 / 2255; 668 / 2107 | 402 / 2323; 384 / 2354 |

Correct completeness uses fixed eligible clean truth support. The false-accepted counts and LR-accepted denominator are computed over the whole named region, not the clean-support denominator; false accepted means not visible or disparity error above one pixel, and does not mean every pixel outside the clean domain. The thin-region loss and increased false accepts argue against promoting this radius-2 configuration.

The radius-2 run reports 151.2/150.3 million candidate updates, versus 444.5/444.5 million for full D. Child wall times including file I/O were 1.248/1.193 s versus 2.474/2.480 s; process-lifetime child maximum RSS high-water was approximately 500,452 versus 500,272 KiB, not a per-eye memory measurement. These CPU figures are not GPU latency estimates. Reports retain output-plane hashes, but the generated raw output planes were not persisted; the hashes alone cannot reproduce a visual or per-pixel inspection.
