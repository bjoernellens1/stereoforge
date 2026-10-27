# Dependencies and reuse

The [license evaluation](../docs/license-evaluation.md) clears the present source
repository for public Apache-2.0 publication. It does not clear a bundled binary
release. Dependency licenses remain separate from StereoForge's own license.

Python versions/hashes are in `uv.lock`; CMake source revisions/hashes are in
`cmake/Capture.cmake` and `cmake/StandaloneRHI.cmake`. Full notice texts are in [licenses](licenses/).

| Component | Actual source license / election | Use |
|---|---|---|
| SlangPy 0.43.1 | Apache-2.0 WITH LLVM-exception (actual LICENSE; metadata omits exception) | Python research Vulkan; optional research native provider |
| Slang 2026.12 | Apache-2.0 WITH LLVM-exception | SPIR-V compiler and fetched helper headers |
| Slang RHI `1a9768741246b1f356116d5dfa305933034ce049` | Apache-2.0 WITH LLVM-exception | Source-built standalone native API; optional research provider |
| ZED Open Capture `93739154903a66c112ee52c3c6fbfd83b7218617` | MIT | Build-local modified capture/sensor sources |
| hidapi `d6b2a974608dec3b76fb1e36c189f22b9cf3650c` | **BSD-3-Clause elected** from upstream alternatives | Header plus separately obtained system runtime |
| pybind11 2.13.6 | BSD-3-Clause | Capture binding |
| OpenCV Python | Packaging MIT; core Apache-2.0; wheel has additional terms | CPU reference/rectification |
| NumPy | BSD family plus permissive source component terms; wheel runtime notices separate | Host arrays |
| Test/build dependencies | See evaluation and package notices | Tests/build only |

The capture build copies Open Capture, hidapi BSD and pybind11 notices into its
`native/licenses/` output. Retain them when redistributing those artifacts; this
is necessary attribution, not a complete binary redistribution certification.

No libSGM, OpenCL SGM, CODD, TemporalStereo or frame-generation source was copied.
They are research references. No pretrained weights, dependency binaries, raw
recordings, or camera calibration files are tracked. Future ncnn/IREE weights
and operators require separate qualification and license review.

SlangPy's unused proprietary OptiX headers and OpenCV's Intel IPP/FFmpeg terms
mean a complete environment is not uniformly Apache/BSD licensed. StereoForge
uses Vulkan explicitly; it does not require or initialize CUDA/ROCm. Audit the
actual contents and licenses before redistributing wheels, native bundles or
containers, including platform-specific and numerical runtime dependencies.
