# License evaluation — public source repository

**Decision: GO for public source publication under Apache-2.0**, after the notice
remediations recorded below. No source-publication license blocker was identified
in this review. This decision does not approve redistribution of a complete binary
or container bundle, and does not describe the entire installed environment as
permissively licensed or free of proprietary components.

Reviewed on 2026-10-27 against source commit
`1f2c33a2688b79c2942b81a25b6daf0e92686733`, its complete two-commit history,
`uv.lock`, CMake download pins, and the locally installed Linux/Python 3.12
packages. The remediation commit adds notices/documentation and build-time notice
copying; it does not import dependency implementations or binary artifacts.

## What is being published

The reviewed tree contains 73 tracked files and the history contains 74 distinct
file blobs, all text. There are no tracked dependency binaries, wheels, pretrained
weights, raw camera recordings, calibration files, vendor SDK trees, fonts or
images. Images mentioned in the original user-supplied proposal are remote links,
not copied assets. Build directories, environments, recordings and distributions
are ignored. No CUDA, HIP, ROCm or TensorRT implementation is incorporated.

StereoForge implementation, shaders, tests and documentation are offered under
the repository's [Apache-2.0 license](../LICENSE). Separately licensed dependencies
retain their own terms; the project license does not relicense them.

## Dependency findings

| Component and inspected pin | License finding | Source-publication result / obligations |
|---|---|---|
| ZED Open Capture `93739154903a66c112ee52c3c6fbfd83b7218617` | MIT | Compatible with publication. Keep full copyright/permission/disclaimer when distributing its source or compiled capture module. |
| hidapi header `d6b2a974608dec3b76fb1e36c189f22b9cf3650c` and local hidapi 0.15.0 | User-selectable GPL-3.0 / BSD-style / original terms; **StereoForge elects BSD-3-Clause** | GPL is not mandatory for this use. Preserve the elected BSD notice and disclaimer. The local capture module links the hidraw backend dynamically. |
| pybind11 2.13.6 | BSD-3-Clause | Compatible; preserve its notice in redistributed bindings/binaries. |
| SlangPy 0.43.1, source `2f6c4625fdd2b3bd812ca6cd2802cf98bd89b248` | Actual LICENSE is **Apache-2.0 WITH LLVM-exception** | Compatible. Metadata's shorter Apache label omits the exception; the license text governs this evaluation. |
| Slang 2026.12, source `a7fbf1ab0e9ddd18b0a9eaa675ddc95f63532fee` | Apache-2.0 WITH LLVM-exception | Compatible. Includes the two fetched compiler helper headers; retain relevant notices. |
| RHI bundled with SlangPy, source `1a9768741246b1f356116d5dfa305933034ce049` | Apache-2.0 WITH LLVM-exception | Compatible. This is the actual bundled pin, not today's upstream HEAD. |
| opencv-python-headless 4.14.0.94 | Python packaging MIT; OpenCV core Apache-2.0; wheel components have additional terms | Calling/importing the separately obtained dependency does not block publication of this source. Do not represent the whole wheel as Apache-only. |
| NumPy 2.5.4, local Python 3.12 | Source/component metadata: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Compatible source dependency. Bundled numerical runtime binaries need a separate redistribution inventory. |
| NumPy 2.4.6, locked Python 3.11 alternative | NumPy source BSD family; this platform's binary wheel was not inspected | No source-repository blocker; no binary clearance is asserted for this alternative. |
| pytest 9.1.1, iniconfig 2.3.1, pluggy 1.7.0 | MIT | Test dependencies, fetched separately; preserve notices if later bundled. |
| packaging 26.3 | Apache-2.0 OR BSD-2-Clause | Permissive alternative licensing; fetched separately. |
| Pygments 2.21.0; colorama 0.4.6 | BSD-2-Clause; BSD-3-Clause respectively | Fetched test/platform dependencies; colorama is not installed in this Linux environment. |
| typing-extensions 4.16.0 | PSF-2.0 | Fetched dependency; retain its license if bundled. |

Setuptools is a separately resolved MIT-licensed build tool, not a vendored
runtime. CMake/Ninja, Python and system loader/runtime packages are also obtained
separately. A binary release must inventory the actual versions and system
libraries it packages rather than infer their licenses from this source table.

Primary evidence: [Open Capture MIT](https://github.com/stereolabs/zed-open-capture/blob/93739154903a66c112ee52c3c6fbfd83b7218617/LICENSE),
[hidapi license choice](https://github.com/libusb/hidapi/blob/d6b2a974608dec3b76fb1e36c189f22b9cf3650c/LICENSE.txt),
[hidapi BSD terms](https://github.com/libusb/hidapi/blob/d6b2a974608dec3b76fb1e36c189f22b9cf3650c/LICENSE-bsd.txt),
[pybind11 license](https://github.com/pybind/pybind11/blob/v2.13.6/LICENSE),
[SlangPy license](https://github.com/shader-slang/slangpy/blob/2f6c4625fdd2b3bd812ca6cd2802cf98bd89b248/LICENSE),
[Slang license](https://github.com/shader-slang/slang/blob/a7fbf1ab0e9ddd18b0a9eaa675ddc95f63532fee/LICENSE),
and [bundled RHI license](https://github.com/shader-slang/slang-rhi/blob/1a9768741246b1f356116d5dfa305933034ce049/LICENSE).
Installed OpenCV `LICENSE.txt` / `LICENSE-3RD-PARTY.txt` and NumPy distribution
`licenses/LICENSE.txt` supplied the wheel findings. Exact Python wheel hashes are
recorded in `uv.lock`.

## Remediations included before publication

- Add full reviewed MIT/BSD/LLVM-exception notices under
  [third_party/licenses](../third_party/licenses/) and a root [NOTICE](../NOTICE).
- Explicitly elect the BSD option for hidapi and correct the previous SlangPy
  summary that omitted the LLVM exception.
- Mark the two build-generated Open Capture header modifications while retaining
  upstream copyright headers. Copy capture dependency notices beside the native
  capture output.
- Include project and dependency notice files in Python wheel/source packaging.
- Keep research-only references separate: no libSGM, sgm_cl, CODD, TemporalStereo
  or frame-generation source or model weights were copied into this tree. In
  particular, noncommercial research code is not incorporated into the Apache core.

Verified notice SHA256 values:

| Notice | SHA256 |
|---|---|
| Open Capture MIT | `74e2b9fe5d883aec86d16546f68f7963232ceedd7e7b61c302303a5ae9a9f402` |
| hidapi BSD | `30eb1bef29b46f8ba7ab8b416035dbd93cb034a45481dd97815b944284582cd2` |
| hidapi license selector | `7d3b087c34f35d4d538e3bcddd1ff8f66e92f9ef336881999482800ddf840913` |
| pybind11 BSD | `83965b843b98f670d3a85bd041ed4b372c8ec50d7b4a5995a83ac697ba675dcb` |
| SlangPy / Slang / RHI LLVM-exception text | `b738d98b7725414b534c119ac66126876b8cbd40aa8ae48210db190714b81670` |

## Binary redistribution is a separate gate

The inspected environment has components that must not be described as uniformly
permissive:

- SlangPy includes unused OptiX SDK headers marked
  `LicenseRef-NvidiaProprietary`. They are not tracked by StereoForge or needed
  by its Vulkan path. Repacking a whole environment would redistribute them and
  needs an artifact-specific review of NVIDIA's terms.
- The OpenCV wheel contains FFmpeg LGPL-2.1 components and Intel IPP vendor
  binary-only terms. Its build reports IPP 2026, while an accompanying notice
  describes an older IPP version; reconcile that discrepancy before bundling.
- NumPy wheels include OpenBLAS/LAPACK notices and compiler runtimes, including
  libgfortran under GPL with the GCC runtime exception and libquadmath. Preserve
  notices and evaluate corresponding-source/replacement requirements for the
  exact redistributed libraries.
- Windows SDK binaries, bundled assets, containers, and future neural weights
  require their own inventories. No binary/model release is authorized by this
  evaluation.

These findings do not restrict public availability of StereoForge's present
source. They do prevent using this review as a blanket clearance for binary
redistribution, patent clearance, or ownership of third-party trademarks.
Re-evaluate when adding copied code, models, assets, changing dependency pins, or
shipping binaries. Public naming does not imply endorsement by Stereolabs, AMD,
NVIDIA, Khronos, or the dependency authors.
