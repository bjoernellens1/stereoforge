# Dependencies and reuse

Python distributions and their exact transitive versions are locked in `uv.lock`.
CMake capture downloads are revision- and SHA256-pinned in `cmake/Capture.cmake`.

| Component | Version/source | Use | License / handling |
|---|---|---|---|
| SlangPy | 0.43.1 | Python Vulkan execution; matching native RHI binaries/headers | Apache-2.0; retain upstream notices and bundled dependency licenses |
| Slang | bundled 2026.12 | SPIR-V compiler; two matching helper headers | Apache-2.0 WITH LLVM-exception |
| Slang RHI | bundled with SlangPy0.43.1 | Native Vulkan resource and dispatch API | Apache-2.0 WITH LLVM-exception; no independently upgraded ABI |
| ZED Open Capture | 93739154903a66c112ee52c3c6fbfd83b7218617 | Native camera/sensors | MIT; source copyright retained in build-local adapter copies |
| hidapi | revision/hash in Capture.cmake | API header; system runtime library | upstream multi-license terms; retain accompanying license on redistribution |
| pybind11 | 2.13.6 | Capture binding | BSD-3-Clause |
| OpenCV | locked Python wheel | Rectification, SGBM, validation | Apache-2.0 plus bundled wheel dependencies |
| NumPy | locked | Host reference arrays | BSD-3-Clause plus wheel dependencies |
| pytest | locked, tests only | Verification | MIT |

No libSGM, OpenCL SGM, CODD, TemporalStereo, or frame-generation source was copied.
They are research references only. No pretrained weights are distributed. Optional
ncnn/IREE models require their own runtime/operator/weight-license qualification.
Bundled multi-backend packages are configured for Vulkan; StereoForge does not
require or initialize CUDA/ROCm. Binary redistribution needs the complete upstream
wheel/runtime license inventory, not only this summary table.
