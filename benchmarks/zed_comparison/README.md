# Portable ZED comparison records (v1)

This source-checkout module validates offline benchmark records without the SDK
or a GPU. The SDK exporter/indexer and native rectified adapter are not implemented.
See the [implementation plan](../../docs/superpowers/plans/2026-10-28-zed-quality-benchmark.md)
and [schema contract](schema.py) for the remaining work and exact required keys.

Run from the repository root with its Python environment (NumPy required):

```python
from pathlib import Path
from benchmarks.zed_comparison.schema import (
    input_identity, load_record, validate_partitions, load_policy,
)

# These paths refer to your own existing, complete records and their assets.
paths = [Path("artifacts/my-export/frame-0001.json")]
records = [load_record(path) for path in paths]
validate_partitions(records)  # metadata checks; load_record checked actual files
assert input_identity(records[0]) == records[0]["input_identity"]
policy = load_policy(Path("configs/benchmarks/zed_quality_policy.json"))
print(policy["status"])  # checked-in provisional policy: unqualified
```

`load_record(Path) -> dict` reads strict JSON and validates files relative to the
record's parent. `validate_record(dict, root: Path) -> None` validates without
mutation. `input_identity(dict) -> str` hashes input metadata without opening files
and can seal a record before its own identity field exists.
`validate_partitions(list[dict]) -> None` checks already-loaded records together;
it does not revalidate file contents. `load_policy(Path) -> dict` admits policy
structure. Invalid records/policies raise `ValueError`. JSON duplicate keys,
nonfinite numbers, and booleans used as numeric fields are rejected.

Each `FileDesc` has `path`, lowercase SHA256 `sha256`, and `size_bytes`. Paths are
relative POSIX paths to regular files; absolute paths, traversal and symlink
components reject. Hash and byte size must match. A plane adds positive integer
`width`, `height`, `stride_bytes`, and `format`; its exact file extent is
`stride_bytes * height`. Row padding is hashed but excluded from pixel checks.
Formats are `rgb8`, binary `u8_mask`, `f32_le_m` depth, `f32_le` confidence,
`u8_provenance`, and `i64_le` source IDs/age. Multi-byte formats are little-endian.
Source hashing uses bounded 1 MiB reads and hashes each repeated path once per
validation call. Separate calls still rehash a shared source; Task 2 will batch.

Raw/rectified calibration files are normalized JSON objects, each with exact keys
`schema`, `space`, `left`, `right`, `right_to_left`, `units`, `pixel_centres`,
`coordinate_axes`. Schema is `stereoforge.zed_comparison.calibration.v1`; space is
`raw` or `rectified`, units `m`, centres `integer_centres`, axes
`x_right_y_down_z_forward`. Each camera has `width`, `height`, `fx`, `fy`, `cx`,
`cy`, `projection: "perspective"`, and `distortion: {model, coefficients}`.
Supported models are `none` (0 coefficients), `brown_conrady` (4 or 5:
`k1,k2,p1,p2[,k3]`), and `rational_polynomial` (8:
`k1,k2,p1,p2,k3,k4,k5,k6`). Other models/counts reject; actual SDK normalization
has not been implemented or qualified. `resolved_file` retains opaque vendor bytes.

`right_to_left` is a finite homogeneous proper rigid 4×4 transform with a nonzero
translation. Both translation norms agree with the declared baseline within
relative `1e-6` or absolute `1e-9 m`: a representation admission tolerance, not a
physical/scientific accuracy waiver. Rectified cameras have no distortion, equal
`fy`/`cy`, shared `fx`, and identity rotation with +X baseline. Raw camera sizes
match original stereo images; rectified sizes/intrinsics match record geometry.
Binary support masks are declarations bound to known models. Task 3 must validate
their geometric map generation; structural admission does not prove support.

Input identity binds source content/codec/lossless metadata, frame ordinal/source
ID, calibration content/geometry, capture timestamp/domain and all six image
descriptors, excluding paths, implementation and observation outputs. Equivalent
record geometry int/float spellings normalize without rounding distinct integers;
calibration file bytes retain their actual hashes. Changed geometry creates a
separate stratum. Original capture identities remain fixed across strata;
rectified pixels/support cannot change within one stratum. The same source hash
cannot contradict kind/codec/lossless metadata. Scene, sequence or source-hash
overlap between `development` and `held_out` rejects, including renamed IDs.
Different implementations may share an input; duplicate implementation/frame
records and reused sequence source IDs for different captures reject.

Availability is explicit: `{status: "available", value: ...}` or
`{status: "unavailable", reason: ...}`. Unknown SDK age/ancestry stays unavailable.
Provenance codes alone do not prove freshness. Invalid raw float bits, including
NaN payloads/infinities, remain read-only; valid depth must be finite positive
left-referenced axial Z in metres. Confidence declares `low_is_good` or
`high_is_good` when present; scores are not calibrated probabilities.

Missing or unfrozen policy returns `unqualified`; frozen incomplete policy rejects.
A complete frozen policy returns only `policy_ready`: structural readiness, not
executed quality, SDK parity, physical accuracy or sustained 60 FPS qualification.
Correctness, regions, zero-reference handling, uncertainty, statistics and the
partition digest must be explicitly frozen; see the schema docstring for shapes.
