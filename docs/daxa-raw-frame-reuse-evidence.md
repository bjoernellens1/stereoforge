# Raw frame reuse: initial evidence

`DaxaRefinement` now has an opt-in path that consumes an already produced full-resolution raw-camera preprocessing frame alongside the coarse depth frame. It reuses that frame's rectified RGB, luma and sampling support via six GPU buffer copies. Original RGB remains available through the retained coarse input. Admission requires the same capture lease, the expected full-resolution calibration owner and the same device. Producer completion, downstream ownership and frame retirement remain explicit. This is not an atlas, a zero-copy path, or a change to the existing default.

A setup-only CPU helper derives per-eye static calibration sampling and 5×5 footprint masks from map values and source dimensions. It does not cache runtime resources, does not represent dynamic support, and is not the existing SGM validity contract. Callers must separately validate their calibration and geometry. A CPU differential against the existing preprocessing reference checks both eyes and an independently evaluated footprint; the separate Release CTest checks invalid maps, edges, ties and shape errors.

## Checks and boundaries

- The GPU contract test compared all 30 refinement readback planes bit-for-bit against the original path for two distinct synthetic 12×10 full-resolution inputs. It exercised admission and retained ownership with core and synchronization validation; the test destroys the owners and closes the context before checking validation counters. It reported zero validation errors and warnings. The raw reuse comparison covered two frames; this small synthetic test is not a 720p or live-camera qualification.
- Separately, the existing canonical rectified-input GPU path passed a 300-frame stress test with synchronization validation and zero validation errors or warnings. That stress test does not exercise the new raw reuse path.
- The host-path review used a fake GPU around the actual `raw_full_pack` control path: three 10,000-frame samples measured **2.46–2.62 µs/frame**, with zero tracked allocations and waits. It excludes real Daxa and GPU driver work and is not a GPU or end-to-end speed result.
- No 60 FPS, physical depth-quality, or real-time pipeline performance claim follows from these checks.

From the repository root, the local verification used:

```bash
cmake --build build/daxa --target daxa_refinement_contracts daxa_preprocess_contracts
ctest --test-dir /tmp/stereoforge-static-support-cmake -R '^static_calibration_support_contracts$' --output-on-failure
ctest --test-dir build/daxa -R '^daxa_preprocess_cpu_contracts$' --output-on-failure
TMPDIR=/dev/shm .venv/bin/python -m pytest -q tests/test_daxa_rectified_input_hotpath.py::test_actual_canonical_refinement_control_three_samples tests/test_daxa_refinement_reuse_hotpath.py
```

The last command reported seven passing tests. The GPU tests were run separately with the locally installed validation layer and RADV device, under the shared GPU lock:

```bash
env -u VK_LAYER_ENABLES \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
  LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  VK_LAYER_VALIDATE_SYNC=1 flock /tmp/stereoforge-gpu.lock \
  ./build/daxa/daxa_refinement_contracts --gpu
```

The same environment and lock were used for `./build/daxa/daxa_preprocess_contracts --gpu` and `./build/daxa/daxa_rectified_input_contracts --gpu` (the separate canonical-path stress test). These paths describe the tested local installation and may need adjustment elsewhere.

For this run, the SHA-256 of `src/daxa_refinement.cpp` was
`addba038f273a9c9ff56959ae326f1598738777bcc5ae9c016caa587a8cec8bc`;
the refinement contract executable was
`c7239bfd8c5d169b3429ed01cc95b66c3db099f646200c1f83d92735009e5c8a`.
These local hashes identify the tested files, not a portable binary release.

Remaining work includes matched 720p end-to-end timing, sustained multi-slot operation, actual atlas/cache integration and lifecycle tests, wider synchronization and device validation, and physical depth-quality evaluation. The full-resolution independent stereo baseline and the quality gates remain in force.
