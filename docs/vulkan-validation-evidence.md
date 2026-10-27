# Vulkan core validation — 2026-10-27

The host initially had no Khronos validation layer. A local, unprivileged
extraction of Fedora 44's signed `vulkan-validation-layers-1.4.341.0-2.fc44.x86_64`
RPM makes positive validation possible without installing system packages.
`rpm -K` reported **digests signatures OK**. RPM SHA256:
`d4978ebe2a247db8d3f4fb6e4e2d5984609879eb58eff904c1174cdbee94da1d`.
The extracted layer's shared-library dependencies all resolved on this host.
Neither the RPM nor its binary is included in source publication.

With `VK_LAYER_PATH` set to the local manifest directory and `LD_LIBRARY_PATH`
set to the extracted library directory, native four- and eight-path stereo
self-tests each completed 100 iterations on AMD Radeon 8060S (RADV STRIX_HALO).
Both reported `vulkan_validation:true`, exited zero, and emitted **zero bytes on
stderr**. Self-tests also exercised leases, foreign-device rejection, retained
completion ownership and calibrated metric-depth edge cases.

| Paths | Resolution / disparity count | Iterations | Process wall | Native processing total |
|---|---|---:|---:|---:|
| 4 | 64×32 / 16 | 100 | 941.2 ms | 89.18 ms |
| 8 | 64×32 / 16 | 100 | 930.1 ms | 110.76 ms |

Process wall includes initialization and additional self-test engines. Native
processing total includes the repeated transfer/readback loop and its initial
lease test. These are small synthetic validation runs, not camera throughput,
end-to-end runtime qualification, or directly comparable production timings.
This establishes **core validation for native stereo**, not synchronization
validation, GPU-assisted validation or a live native chain. Subsequent native
compositor and temporal contract fixtures also passed core validation; their
artifact, loader proof and scope are recorded in fusion-evidence.md.

Local evidence: `artifacts/validation-stereo-20261027/report.json`, plus stdout
and stderr per path count. Missing-layer rejection remains a separate CTest.

```bash
export VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d"
export LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
build/vulkan/native_stereo --validation --self-test --paths 4 --iterations 100
build/vulkan/native_stereo --validation --self-test --paths 8 --iterations 100
```

The layer was obtained with `dnf download vulkan-validation-layers`; only the
signature-verified x86_64 RPM was extracted using `rpm2cpio` and `cpio` into
`build/validation-layer`. No signing keys or system permissions were changed.

## Python device-creation limitation

A separate forced-layer run of Python compositor fixtures returned numerically
passing tests but printed `VUID-VkDeviceCreateInfo-pNext-pNext` and unknown-extension
diagnostics on **stdout** during SlangPy device creation. Empty stderr and exit
zero are therefore insufficient to qualify that run. The layer reports unknown
structure types 1000233000, 1000573000 and 1000231000, and extension names
`VK_KHR_shader_constant_data`, `VK_KHR_device_fault`, `VK_KHR_shader_abort`.
This remains an unresolved Python device/layer compatibility issue; no diagnostics
were filtered or treated as a clean pass. Native positive-validation results
above apply to their tested configuration only.

Retained evidence:
`artifacts/python-compositor-core-validation-20261027/report.json`, stdout and
stderr logs. The fixtures ran with capture disabled (`pytest -s`) so diagnostics
could not disappear inside passing-test capture.
