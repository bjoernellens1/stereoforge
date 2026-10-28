# Canonical record input bridge, version 1

This is a cold, offline bridge from a validated portable analytic record to the
canonical RGB producer. It admits `source.kind == "analytic"` only. It does not
qualify SDK, SVO2, physical, or Middlebury geometry/support. Producer labels and
hashes bind provenance; they do not establish physical accuracy. Raw and rectified
stereo snapshots do not determine the producer's raw-to-rectified rotations;
`R1`/`R2` must be supplied explicitly, never inferred or replaced by a new
`stereoRectify` result.

## Python entry point and bundle

`benchmarks.zed_comparison.native_input.write_native_input(record_path,
supplement_path, output_dir) -> pathlib.Path` returns the new manifest path.
Call `schema.load_records([record_path])` before conversion. Require native
dimensions below 32767, even rectified width/height, positive capture timestamp,
known nonempty capture clock domain, a source ID in signed 64-bit range, and
metric `coordinate_frame == "left_optical"`.
Unavailable ingress remains unavailable: native timestamp zero/domain `unknown`.

Create a fresh output directory exclusively. Copy the exact record bytes and
every referenced FileDesc/PlaneDesc dependency recursively, retaining original
relative POSIX paths, byte contents, hashes, sizes, and row padding. Include
available resolved calibration, observation and implementation artifacts, so the
copied record still passes `schema.load_records`. Do not rewrite record JSON or
its `input_identity`. Copy the supplement bytes unchanged. Write only these new
root files:

* `native-input.json` — native manifest;
* `native-geometry.json` — exact supplied supplement bytes;
* `native-coarse-calibration.json` — derived normalized rectified calibration.

Reject any source record/dependency path equal to, beneath, or conflicting with
these reserved paths, including directory/file conflicts. Reject traversal,
absolute paths, backslashes, empty/dot components, symlink components, and
nonregular dependency files. Shared paths must describe identical bytes. Check
input snapshots before/after reading and copying; rehash copied bytes against
their descriptors. Publish no successful manifest for a failed conversion.

## Exact JSON contracts

All JSON uses strict parsing: reject duplicate keys, nonfinite numbers, booleans
where numbers are required, excessive size/nesting, and unknown keys in the new
manifest/supplement. Limit every parsed JSON file to 16 MiB and nesting depth 64;
check the limit before unbounded allocation. Limit the sum of the six stored
plane extents to 512 MiB, using checked integer arithmetic. Lowercase SHA256
strings have exactly 64 hex digits. A
FileDesc has exactly `path`, `sha256`, `size_bytes`, using the existing portable
record semantics. Matrices are nested row-major numeric arrays, not flat arrays;
elements must be finite numbers and rotations/projections must pass geometry
validation.

The manifest has exactly:

```json
{
  "schema": "stereoforge.canonical_record_input.v1",
  "source_record": {"path": "record.json", "sha256": "<64 hex>", "size_bytes": 1},
  "geometry_supplement": {"path": "native-geometry.json", "sha256": "<64 hex>", "size_bytes": 1},
  "coarse_calibration": {"path": "native-coarse-calibration.json", "sha256": "<64 hex>", "size_bytes": 1},
  "sampling_profile": "canonical_rgb8_box2x2_halfup_gray15_support_and4_halfpixel_v1",
  "original_pixel_identity": "<64 hex>",
  "rectified_pixel_identity": "<64 hex>"
}
```

Sizes/hashes above are placeholders. The supplement has exactly these keys:

| Key | Value |
| --- | --- |
| `schema` | `stereoforge.canonical_geometry.v1` |
| `input_identity` | Exact record `input_identity` |
| `record_sha256` | SHA256 of exact source record bytes |
| `raw_calibration_sha256` | Record `calibration.raw.sha256` |
| `rectified_calibration_sha256` | Record `calibration.rectified.sha256` |
| `R1`, `R2` | Explicit 3 by 3 raw-to-rectified rotations |
| `P1`, `P2` | Explicit 3 by 4 rectified projections |
| `support_qualification` | Exactly `analytic_known_geometry` |
| `producer` | Nonempty string naming the analytic geometry/support producer |

The qualification string is a scope restriction, not proof that arbitrary
declared support is correct. Initial fixtures must have independently specified,
tested analytic geometry/support; no vendor qualification label is accepted.

Compute each pixel identity as SHA256 of UTF-8
`json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)`.
`payload` maps selected image names to their complete PlaneDesc with **only**
`path` removed. Original selects `original_left`, `original_right`; rectified
selects `rectified_left`, `rectified_right`, `support_left`, `support_right`.
The remaining values are ASCII format/hash strings and integers, making this
serialization identical across Python/C++. Padding bytes remain hash-bound.

## Geometry and coarse derivation

Read the exact normalized portable raw/rectified calibration JSON referenced by
the record. Convert supported distortion objects to coefficient vectors without
changing their order: none=0, Brown-Conrady=4/5, rational polynomial=8. Preserve
camera dimensions, intrinsics and right-to-left poses. Full geometry identities
are the raw/rectified calibration **file hashes**. Combine those models with the
supplement matrices and call `DaxaRectifiedGeometry::validate()` and
`metric_calibration()`. Compare full models with record geometry/dimensions.

The coarse calibration file has the existing strict
`stereoforge.zed_comparison.calibration.v1` schema, `space == "rectified"`, and
unchanged conventions/pose. For each full rectified camera, derive:

* `width_c = width_f / 2`, `height_c = height_f / 2`;
* `fx_c = fx_f / 2`, `fy_c = fy_f / 2`;
* `cx_c = (cx_f + 0.5) / 2 - 0.5`, likewise `cy`;
* unchanged zero distortion and right-to-left pose.

Coarse geometry retains the exact original models/identity and `R1`/`R2`.
For each `P`, halve rows 0 and 1, except principal entries
`P[0][2]`, `P[1][2]`, which use the same half-pixel formula; retain row 2.
Its rectified identity is the coarse calibration file hash. Validate both the
derived geometry and the supplied coarse file against this derivation. No maps,
YUYV bytes, or additional coarse RGB files are generated. Coarse pixels are
producer box2x2 per-channel half-up, gray15 after RGB reduction, support AND4.

## C++ cold loader API and boundary

Add `include/stereoforge/canonical_record.hpp` and `src/canonical_record.cpp`,
using `canonical_rgb.hpp`, `daxa_calibration.hpp`, standard library headers,
private nlohmann JSON and OpenSSL hashing. Public headers must not include
Vulkan/Daxa frame/producer headers. Do not refactor the existing preprocess
fixture loader or its helpers.

```cpp
namespace stereoforge {
struct CanonicalRecordFile {
    std::filesystem::path path; // Absolute checked bundle path.
    std::string sha256;
    uint64_t size_bytes = 0;
};
struct CanonicalRecordInput {
    RectifiedRgbFrame frame;
    DaxaRectifiedGeometry full_geometry, coarse_geometry;
    uint64_t ordinal = 0;
    std::string manifest_sha256, record_sha256;
    std::string geometry_supplement_sha256, coarse_calibration_sha256;
    std::string producer;
    std::vector<CanonicalRecordFile> files;
};
CanonicalRecordInput load_canonical_record(const std::filesystem::path& manifest);
void verify_canonical_record_files(const CanonicalRecordInput&);
}
```

Native admission validates the strict manifest/supplement, all relevant record
fields/descriptors, all consumed hashes/extents, calibration bindings, matrix
shapes/values, derived coarse geometry, pixel identities and binary active mask
pixels. Verify six image names/formats and exact `size_bytes == stride*height`,
safe bounded extents, matching eye sizes, and record source/time/identity fields.
Python remains the complete portable-schema admission authority; native need not
reimplement observation/implementation semantics or Python input-identity
serialization. It must compare supplemental/record identities and exact hashes,
not silently recompute or replace the portable `input_identity`.

Load exactly six immutable byte vectors, preserving original padded extents,
and retain them through sidecar `backing`. Populate RGB in order original L/R,
rectified L/R and support L/R. Populate capture source kind
`CanonicalRgbMetadata`, logical original width `2*eye_width`, original height,
zero stride, empty YUYV, source ID, capture/ingress timestamps and truthful clock
domains. Set sidecar calibration identities from the three calibration hashes,
retain record/pixel identities/profile, and construct `RectifiedRgbFrame` only
after all cold validation. No I/O, hashing, support scan, or matrix traversal
belongs in enqueue.

`files` includes the manifest and every file actually checked by native
admission, with expected hashes/sizes. Detect observed changes during load by
checking file identity/stat snapshots around reads and hashes. Verification
rehashes those files at an explicit post-run boundary and rejects changes.
These checks detect observed mutation; they do not promise an atomic filesystem
snapshot. Admission owns copied immutable bytes independently of subsequent
filesystem changes.

## Verification

Python tests cover fresh-bundle round trip through `schema.load_records`, exact
six-plane preservation including awkward padding, unchanged portable identity,
known pixel digest vectors, coarse half-pixel derivation, and rejection of
nonanalytic/missing/mismatched geometry, duplicate keys, paths/symlinks,
collisions, hash mutations and invalid masks. Native CPU tests load that same
fixture, check all fields/geometry/owned lifetimes and digest vectors, reject
the same relevant malformed cases, and detect post-load file mutation. Build
the cold loader without Vulkan headers; producer/GPU qualification is separate.

## Usage, build and verification boundary

From the repository root, export an admitted analytic portable record and its
explicit geometry supplement into a directory that does not yet exist:

```bash
.venv/bin/python -m benchmarks.zed_comparison.native_input RECORD SUPPLEMENT FRESH_DIR
cmake --build build/daxa --target daxa_depth_rectified_benchmark
```

Run either `full_sgm` or `coarse_refine` with a fresh report path. This example
uses the locally extracted validation layer, with inherited layer enables
cleared as in the [native rectified input evidence](native-rectified-input.md):

```bash
env -u VK_LAYER_ENABLES \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
VK_LAYER_PATH="$PWD/build/validation-layer/usr/share/vulkan/explicit_layer.d" \
LD_LIBRARY_PATH="$PWD/build/validation-layer/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
VK_LAYER_VALIDATE_SYNC=1 build/daxa/daxa_depth_rectified_benchmark \
  --manifest bundle/native-input.json --variant full_sgm \
  --frames 3 --warmup 1 --output fresh-report.json --validation
```

Replace `full_sgm` with `coarse_refine` for the coarse/refinement path. Optional
`--dump-prefix fresh-prefix` writes result planes; use a fresh prefix for each
run. `--validation` enables Vulkan validation. Full SGM uses independent left
and right SGM with 128 disparities and four directions. The coarse path uses
independent 64-disparity, four-direction SGM followed by full-resolution
128-disparity refinement; `--radius` accepts 2, 4 or 8 (default 4). Primary
comparison runs keep RGB synthesis disabled.

The report measures repeated processing of one immutable record. Its timings
do not establish live FPS, sustained 720p60 performance, or camera ingestion
throughput. CPU exporter, loader and application guard tests cover admission
and configuration contracts. Bounded 128×64 validation runs, core stress and
raw-input regression checks provide runtime regression evidence; consult the
current run logs for their results and counts. A single analytic plane does
not qualify held-out depth accuracy or error-versus-coverage acceptance.

Root verification passed 2,140 CPU tests with 37 GPU/camera tests deselected
and no skips (130.89 s). Both benchmark variants completed three measured
iterations after one warmup on the 128×64 analytic fixture with zero validation
errors or warnings through teardown. The connected 300-record stress and all
three raw GPU regression suites also passed. Source, binary and fixture hashes
remained unchanged during the final benchmark runs. Evidence is in
`artifacts/zed-task3/{full-cpu-native-bundle.log,canonical-app-freeze-v3.json,canonical-app-full_sgm-gpu-v3.log,canonical-app-coarse_refine-gpu-v3.log,canonical-connected-gpu-v2.log,canonical-raw-gpu-v2.log}`.

SDK geometry/support qualification, Task 4 quality acceptance and sustained
720p60 remain open.
