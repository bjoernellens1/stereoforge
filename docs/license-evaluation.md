# License evaluation — public source repository

Native preprocessing follow-up (2026-10-27, integration in progress): the
optional fixture loader selects nlohmann/json `v3.11.3` (MIT) and separately
obtained OpenSSL Crypto 3.x (Apache-2.0) for setup-time JSON and SHA256 checks.
The exact upstream JSON license and installed Fedora OpenSSL 3.5.9 license
were inspected and retained under `third_party/licenses`. The JSON single header
SHA256 `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6`
was independently verified against the optional CMake pin. Embedded MIT SPDX
copyright declarations are preserved in `nlohmann-json.NOTICE`, including those
for Niels Lohmann, Evan Nemerson, The Abseil Authors, Björn Hoehrmann and Florian
Loitsch. No source-publication blocker is identified for these selections.
All three retained notices were verified byte-for-byte in the optional build
output `build/daxa/native/licenses`; complete binary redistribution remains
outside this source review. Neither dependency
performs per-frame image processing or adds a non-Vulkan compute backend.

Daxa foundation follow-up (2026-10-27): the independent Daxa/Slang build
selects Daxa tag `3.6`, revision `00761e80c1dd6d1972d5f4569ce49620a7b4d384`,
and its required Vulkan Memory Allocator (VMA) `v3.1.0`, resolved revision
`009ecd192c1289c7529bff248a16cfe896254816`. Both downloaded source license
files were inspected and retained byte-for-byte under `third_party/licenses`.
Their MIT terms introduce no identified source-publication blocker for this
selection. This finding is limited to the selected sources, not every optional
Daxa dependency or the contents of a future binary bundle. The Linux Slang
2026.12 SDK's actual LICENSE matches the existing retained Slang notice.

Standalone follow-up (2026-10-27): the production provider now source-builds
Slang RHI revision `1a9768741246b1f356116d5dfa305933034ce049` against the official
Slang 2026.12 SDK. Both archive hashes are pinned in cmake/StandaloneRHI.cmake;
both actual license files match the retained Apache-2.0 WITH LLVM-exception
notice byte-for-byte. No new binary/dependency archive is tracked or published.
Source-publication clearance below remains applicable; binary redistribution
still requires its separate audit. SlangPy is an optional research provider.

Refinement-only toolchain follow-up (2026-10-28):
`cmake/DaxaRefinement.cmake` separately selects the official Linux x86_64 Slang
2026.19 SDK, tag `v2026.19` (unsigned), source revision
`0fb6b7587a94fa33d88141ec9930766eedc6f59a`, archive SHA256
`bd6cfc47b7353b2cffa36a866b1584749f76f4b2dbebf6dda26bc3fb40dc3c0e`.
The extracted `build/daxa/_deps/sf_refinement_slang_sdk-src/LICENSE` is
byte-identical to both the existing 2026.12 SDK LICENSE and the retained
`slang-Apache-2.0-LLVM-exception.txt`: SHA256
`b738d98b7725414b534c119ac66126876b8cbd40aa8ae48210db190714b81670`.
This additional compiler does not change the frozen 2026.12 compiler lane.

Unlike the inspected 2026.12 SDK at `build/standalone/_deps/sf_slang_sdk-src`,
which has neither directory, the extracted 2026.19 SDK contains seven
`LICENSES/` files: `Apache-2.0.txt`, `BSL-1.0.txt`, `CC-BY-4.0.txt`,
`LLVM-exception.txt`, `LicenseRef-UOI-NCSA.txt`, `MIT.txt`, and `Unlicense.txt`;
and six `third-party-notices/` files: `cmark-COPYING`, `glslang-LICENSE.txt`,
`lz4-LICENSE`, `miniz-LICENSE`, `spirv-tools-LICENSE`, and
`unordered_dense-LICENSE`. These are additional notice inventory, not a
conclusion that every SDK component has only the primary Slang license.
In particular, `glslang-LICENSE.txt` (SHA256
`17e70c676e1521ff3e4686f04a2053d93a7e28a33be8de7ec37ab0ff72feb677`)
includes BSD/MIT/Apache terms, historical GPLv3 text with the Bison exception
(the notice states Bison was removed), and NVIDIA preprocessor terms permitting
use, modification and redistribution with retained notices. CC-BY-4.0 also
introduces attribution terms. No changed primary license or unidentified
restrictive notice was found in these files. This finding concerns the separately
fetched compiler used by the published source; any SDK/binary bundle still needs
component-to-artifact attribution, notice retention and generated-output review.
No SDK binaries or additional SDK license files are copied into the tracked tree.

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

## Optional proprietary SDK reference runner

Review dated 2026-10-28: an optional ZED SDK reference runner is planned, not
implemented. It would be isolated from the core, with no SDK/core linkage and
default OFF. The [official release page](https://www.stereolabs.com/developers/release)
currently offers SDK 5.5.1 installers with platform/CUDA/TensorRT choices; the official development image has been downloaded for isolated reference work
(`stereolabs/zed:5.5-devel-cuda12.8-ubuntu24.04`, manifest digest
`sha256:0297481b344437ad2874edaef4c32da821491c15cc761187c1d6e0c902148cde`).
No host-system SDK installer has been executed. Image availability is not
benchmark execution or a redistribution clearance.

The [sample repository LICENSE](https://raw.githubusercontent.com/stereolabs/zed-sdk/master/LICENSE)
is MIT, copyright 2021 Stereolabs. That covers the licensed sample source, not
the proprietary SDK or its models. Section 3.4 of the
[official Software and Services License Agreement](https://www.stereolabs.com/SoftwareandServicesLicenseAgreement)
(last modified November 15, 2024) preserves separate open-source component terms.
Section 3.5(h) restricts access or use to build a competitive product or service
unless Stereolabs gives express written permission. This is an actual condition
to resolve for the proposed StereoForge comparison, not merely a redistribution
notice obligation. The user states that permission is granted for the requested
task and all benchmarking against the Stereolabs SDK; this authorizes proceeding
with the isolated benchmark work. This records user-stated authorization, not a
reviewed vendor permission letter or blanket license clearance. The exact selected
SDK distribution terms and their applicability remain part of the dependency
inventory, including the competitive-use condition.

The proposed workflow would redistribute no SDK binaries, models, private SVO
recordings or camera serial numbers. Independently authored portable export-index
and evaluator work requiring no SDK can continue under the existing Apache-2.0
source-publication clearance. That clearance and the historical review scope and
counts below remain unchanged; external SDK parity remains unqualified.

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
| nlohmann/json `v3.11.3` | MIT; upstream LICENSE.MIT inspected | Optional native preprocessing fixture parser. Preserve copyright/permission/disclaimer when redistributing the header or compiled code. |
| OpenSSL Crypto 3.x; installed Fedora 3.5.9 | Apache-2.0; exact installed LICENSE.txt inspected | Optional setup-time SHA256 verification; separately discovered system library. Preserve applicable notices for the actual redistributed version; no complete binary-bundle clearance. |
| ZED Open Capture `93739154903a66c112ee52c3c6fbfd83b7218617` | MIT | Compatible with publication. Keep full copyright/permission/disclaimer when distributing its source or compiled capture module. |
| hidapi header `d6b2a974608dec3b76fb1e36c189f22b9cf3650c` and local hidapi 0.15.0 | User-selectable GPL-3.0 / BSD-style / original terms; **StereoForge elects BSD-3-Clause** | GPL is not mandatory for this use. Preserve the elected BSD notice and disclaimer. The local capture module links the hidraw backend dynamically. |
| pybind11 2.13.6 | BSD-3-Clause | Compatible; preserve its notice in redistributed bindings/binaries. |
| Daxa tag `3.6`, source `00761e80c1dd6d1972d5f4569ce49620a7b4d384` | MIT; actual source LICENSE inspected | Independent native Vulkan foundation. Preserve full notice in source/compiled distributions. |
| Vulkan Memory Allocator `v3.1.0`, source `009ecd192c1289c7529bff248a16cfe896254816` | MIT; actual source LICENSE.txt inspected | Required Daxa implementation dependency (`VMA_IMPLEMENTATION`); preserve full AMD notice. |
| SlangPy 0.43.1, source `2f6c4625fdd2b3bd812ca6cd2802cf98bd89b248` | Actual LICENSE is **Apache-2.0 WITH LLVM-exception** | Compatible. Metadata's shorter Apache label omits the exception; the license text governs this evaluation. |
| Slang 2026.12, source `a7fbf1ab0e9ddd18b0a9eaa675ddc95f63532fee` | Apache-2.0 WITH LLVM-exception | Compatible. Includes the two fetched compiler helper headers; retain relevant notices. |
| Slang 2026.19 refinement-only SDK, source `0fb6b7587a94fa33d88141ec9930766eedc6f59a` | Actual primary LICENSE matches retained Apache-2.0 WITH LLVM-exception text; additional SDK notices inventoried above | Separately fetched offline compiler; this source review does not clear redistribution of its SDK/binaries or all generated outputs. |
| RHI bundled with SlangPy, source `1a9768741246b1f356116d5dfa305933034ce049` | Apache-2.0 WITH LLVM-exception | Compatible. This is the actual bundled pin, not today's upstream HEAD. |
| opencv-python-headless 4.14.0.94 | Python packaging MIT; OpenCV core Apache-2.0; wheel components have additional terms | Calling/importing the separately obtained dependency does not block publication of this source. Do not represent the whole wheel as Apache-only. |
| NumPy 2.5.4, local Python 3.12 | Source/component metadata: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Compatible source dependency. Bundled numerical runtime binaries need a separate redistribution inventory. |
| NumPy 2.4.6, locked Python 3.11 alternative | NumPy source BSD family; this platform's binary wheel was not inspected | No source-repository blocker; no binary clearance is asserted for this alternative. |
| pytest 9.1.1, iniconfig 2.3.1, pluggy 1.7.0 | MIT | Test dependencies, fetched separately; preserve notices if later bundled. |
| packaging 26.3 | Apache-2.0 OR BSD-2-Clause | Permissive alternative licensing; fetched separately. |
| Pygments 2.21.0; colorama 0.4.6 | BSD-2-Clause; BSD-3-Clause respectively | Fetched test/platform dependencies; colorama is not installed in this Linux environment. |
| typing-extensions 4.16.0 | PSF-2.0 | Fetched dependency; retain its license if bundled. |

The selected Daxa foundation uses the separately obtained Linux Slang 2026.12
`slangc` to compile SPIR-V offline; Daxa's Slang/GLSL pipeline managers, SPIR-V
validation utility, tests, tools, ImGui/ImPlot and FidelityFX integrations are
disabled. Daxa's upstream Windows-only Slang bootstrap is therefore bypassed;
that bootstrap limitation is a build issue, not a new license restriction.
The task graph and memory utilities are enabled and add no further fetched
source dependencies in the inspected CMake files. At this pin, `cmake/deps.cmake`
and the selected `src`/`include` tree contain no volk, fmt, robin-map, glm or
SPIRV-Reflect dependency. Vulkan headers/loader come from system discovery rather
than a pinned Vulkan-Headers source fetch. Optional dependency licenses have not
been inventoried or cleared here. Changing
these options requires reviewing the newly selected source and license files.
Daxa also discovers Linux system dependencies (Vulkan headers/loader, X11 and
Wayland/xkbcommon where available); their exact linked versions and notices remain
pending for any binary redistribution inventory. They are not copied by this
source-publication change. Slang SDK binary contents and generated-shader release
obligations likewise require artifact-specific review before bundling.

Setuptools is a separately resolved MIT-licensed build tool, not a vendored
runtime. CMake/Ninja, Python and system loader/runtime packages are also obtained
separately. A binary release must inventory the actual versions and system
libraries it packages rather than infer their licenses from this source table.

Primary evidence: [nlohmann/json v3.11.3 MIT](https://github.com/nlohmann/json/blob/v3.11.3/LICENSE.MIT),
OpenSSL installed package `/usr/share/licenses/openssl-libs/LICENSE.txt`
(Fedora `openssl-libs`/`openssl-devel` 3.5.9-1.fc44; retained exact copy),
[Daxa MIT](https://github.com/Ipotrick/Daxa/blob/00761e80c1dd6d1972d5f4569ce49620a7b4d384/LICENSE),
[VMA MIT](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/blob/009ecd192c1289c7529bff248a16cfe896254816/LICENSE.txt),
[Open Capture MIT](https://github.com/stereolabs/zed-open-capture/blob/93739154903a66c112ee52c3c6fbfd83b7218617/LICENSE),
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
| nlohmann/json v3.11.3 MIT | `86b998c792894ccb911a1cb7994f7a9652894e7a094c0b5e45be2f553f45cf14` |
| Installed OpenSSL 3.5.9 Apache-2.0 | `7d5450cb2d142651b8afa315b5f238efc805dad827d91ba367d8516bc9d49e7a` |
| Daxa MIT | `566dadaf9e98c94b563f168488e0b696ebc9e216ee6e126650c73a9c7a3c3914` |
| VMA MIT | `f56ab0392925231e34dd008a7670bb5c5ead490203bc9e125de3106fef642a27` |
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
