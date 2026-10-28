"""Portable, SDK-free records reject corrupt content and leaked experiments."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from benchmarks.zed_comparison.schema import (
    input_identity, load_policy, load_record, validate_partitions, validate_record,
)


def unavailable(reason="not exposed"):
    return {"status": "unavailable", "reason": reason}


def available(value):
    return {"status": "available", "value": value}


def independent_identity(r):
    """Independent spelling of the specified canonical identity payload."""
    def content(d):
        return {k: v for k, v in d.items() if k != "path"}
    c = r["calibration"]
    resolved = deepcopy(c["resolved_file"])
    if resolved["status"] == "available":
        resolved["value"] = content(resolved["value"])
    geometry = dict(c["geometry"])
    for name in ("fx", "fy", "baseline_m", "cx_left", "cx_right"):
        value = geometry[name]
        as_float = float(value)
        geometry[name] = as_float if as_float == value else value
        if value == 0:
            geometry[name] = 0.0
    payload = {
        "source": {**r["source"], "artifact": content(r["source"]["artifact"])},
        "ordinal": r["ordinal"], "source_id": r["source_id"],
        "calibration": {"raw": content(c["raw"]), "rectified": content(c["rectified"]),
                        "resolved_file": resolved, "geometry": geometry},
        "capture": r["timestamps"]["capture"],
        "images": {k: content(v) for k, v in r["images"].items()},
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


@pytest.fixture
def record(tmp_path):
    def file(name, data):
        (tmp_path / name).write_bytes(data)
        return dict(path=name, sha256=sha256(data).hexdigest(), size_bytes=len(data))
    def plane(name, data, fmt, stride):
        return dict(**file(name, data), width=2, height=2, stride_bytes=stride, format=fmt)
    rgb = bytes(range(16))  # 6 pixel bytes plus 2 padding bytes per row
    images = {k: plane(k, rgb, "rgb8", 8) for k in
              ("original_left", "original_right", "rectified_left", "rectified_right")}
    images.update({k: plane(k, b"\x01\x01\xff\x01\x00\xfe", "u8_mask", 3)
                   for k in ("support_left", "support_right")})
    # Preserve a noncanonical NaN payload and infinity on invalid pixels.
    depth = struct.pack("<IIIf", 0x3f800000, 0x7fc01234, 0x7f800000, 2.)
    camera = dict(width=2, height=2, fx=5., fy=5., cx=.5, cy=.5,
                  projection="perspective", distortion=dict(model="none", coefficients=[]))
    def calibration(space):
        return dict(schema="stereoforge.zed_comparison.calibration.v1", space=space,
                    left=deepcopy(camera), right={**deepcopy(camera), "cx": .6},
                    right_to_left=[[1., 0., 0., .12], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
                    units="m", pixel_centres="integer_centres", coordinate_axes="x_right_y_down_z_forward")
    r = dict(
        schema="stereoforge.zed_comparison.frame.v1", scene_id="scene-a",
        sequence_id="sequence-a", partition="development", ordinal=0, source_id=2**63-1,
        source=dict(kind="analytic", artifact=file("source", b"source bytes"),
                    codec="analytic", lossless=True),
        calibration=dict(raw=file("raw.json", json.dumps(calibration("raw")).encode()),
                         rectified=file("rect.json", json.dumps(calibration("rectified")).encode()),
                         resolved_file=unavailable(),
                         geometry=dict(width=2, height=2, fx=5., fy=5., baseline_m=.12,
                                       cx_left=.5, cx_right=.6, projection="perspective",
                                       depth_convention="axial_z", units="m",
                                       coordinate_frame="left_optical", pixel_centres="integer_centres",
                                       reference_camera="left", coordinate_axes="x_right_y_down_z_forward")),
        timestamps=dict(capture=dict(value_ns=100, clock_domain="camera"),
                        ingress=unavailable(), sensors=[]), images=images,
        observation=available(dict(depth=plane("depth", depth, "f32_le_m", 8),
                                   validity=plane("validity", b"\1\0\0\1", "u8_mask", 2),
                                   confidence=unavailable(), confidence_direction="unavailable",
                                   provenance=unavailable(), source_ids=unavailable(), age_ns=unavailable())),
        implementation=dict(name="test", version="1", commit=unavailable(),
                            device=unavailable(), settings={}, artifacts=[]), input_identity="pending")
    r["input_identity"] = independent_identity(r)
    return r


def reseal(r):
    r["input_identity"] = independent_identity(r)
    return r


def replace_plane(tmp_path, desc, values, dtype):
    data = np.asarray(values, dtype=dtype).tobytes()
    (tmp_path / desc["path"]).write_bytes(data)
    desc.update(size_bytes=len(data), sha256=sha256(data).hexdigest())


def test_v1_round_trip_hashes(record, tmp_path):
    before = deepcopy(record)
    validate_record(record, tmp_path)
    assert record == before
    assert input_identity(record) == independent_identity(record)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    assert load_record(path) == record
    assert (tmp_path / "depth").read_bytes()[4:8] == bytes.fromhex("3412c07f")


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside", "a/../depth", "a\\depth", "./depth", ""])
def test_reject_unsafe_paths_and_changed_hash(record, tmp_path, path):
    record["observation"]["value"]["depth"]["path"] = path
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


@pytest.mark.parametrize("change", ["hash", "size", "missing", "directory", "symlink", "parent_symlink"])
def test_bad_files(record, tmp_path, change):
    d = record["observation"]["value"]["depth"]
    if change == "hash":
        (tmp_path / "depth").write_bytes(b"!" * 16)
    elif change == "size":
        d["size_bytes"] = 15
    elif change == "missing":
        (tmp_path / "depth").unlink()
    elif change == "directory":
        d["path"] = "directory"
        (tmp_path / "directory").mkdir()
    elif change == "symlink":
        (tmp_path / "alias").symlink_to(tmp_path / "depth")
        d["path"] = "alias"
    else:
        (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
        d["path"] = "alias/depth"
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


@pytest.mark.parametrize("body", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'])
def test_strict_json(record, tmp_path, body):
    path = tmp_path / "record.json"
    path.write_text(body)
    with pytest.raises(ValueError):
        load_record(path)
    path.write_text(body)
    desc = record["calibration"]["raw"]
    desc.update(path=path.name, sha256=sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
    reseal(record)
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


@pytest.mark.parametrize("key,value", [("ordinal", True), ("ordinal", -1), ("source_id", True),
                                       ("source_id", 2**63), ("source_id", -1)])
def test_integer_domains(record, tmp_path, key, value):
    record[key] = value
    reseal(record)
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


@pytest.mark.parametrize("key,value", [("width", True), ("height", 0), ("fx", True), ("fy", float("inf")),
                                       ("baseline_m", 0), ("cx_left", float("nan"))])
def test_geometry_domains(record, tmp_path, key, value):
    record["calibration"]["geometry"][key] = value
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


def test_calibration_and_clock_identity_required(record, tmp_path):
    for section, key in [("calibration", "raw"), ("timestamps", "capture")]:
        r = deepcopy(record)
        del r[section][key]
        with pytest.raises(ValueError):
            validate_record(r, tmp_path)
    r = deepcopy(record)
    r["timestamps"]["capture"]["clock_domain"] = ""
    with pytest.raises(ValueError):
        validate_record(r, tmp_path)


@pytest.mark.parametrize("kind", ["mask", "depth", "format", "stride", "extent", "shape", "original_shape"])
def test_bad_plane_content_and_layout(record, tmp_path, kind):
    obs = record["observation"]["value"]
    if kind == "mask":
        replace_plane(tmp_path, obs["validity"], [1, 2, 0, 1], "u1")
    elif kind == "depth":
        replace_plane(tmp_path, obs["depth"], [0, np.nan, np.inf, -1], "<f4")
    elif kind == "format":
        obs["depth"]["format"] = "f32_be_m"
    elif kind == "stride":
        obs["depth"]["stride_bytes"] = 7
    elif kind == "extent":
        obs["depth"]["height"] = 3
    elif kind == "shape":
        obs["depth"].update(width=1, stride_bytes=8)
    else:
        record["images"]["original_right"]["width"] = 1
    reseal(record)
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


def test_all_invalid_depth_and_unknown_sdk_metadata(record, tmp_path):
    obs = record["observation"]["value"]
    replace_plane(tmp_path, obs["validity"], [0, 0, 0, 0], "u1")
    replace_plane(tmp_path, obs["depth"], [np.nan, np.inf, -np.inf, -1], "<f4")
    original = (tmp_path / "depth").read_bytes()
    validate_record(record, tmp_path)
    assert (tmp_path / "depth").read_bytes() == original
    record["observation"] = unavailable("SDK not executed")
    validate_record(record, tmp_path)


@pytest.mark.parametrize("name,fmt,values,dtype", [
    ("confidence", "f32_le", [np.nan, 1, 1, 1], "<f4"),
    ("age_ns", "i64_le", [0, -1, 0, 0], "<i8"),
    ("source_ids", "i64_le", [-1, 0, 0, 1], "<i8"),
    ("provenance", "u8_provenance", [0, 1, 1, 6], "u1"),
])
def test_optional_plane_domains(record, tmp_path, name, fmt, values, dtype):
    data = np.asarray(values, dtype=dtype).tobytes()
    (tmp_path / name).write_bytes(data)
    desc = dict(path=name, sha256=sha256(data).hexdigest(), size_bytes=len(data),
                width=2, height=2, stride_bytes=2*np.dtype(dtype).itemsize, format=fmt)
    obs = record["observation"]["value"]
    obs[name] = available(desc)
    if name == "confidence":
        obs["confidence_direction"] = "low_is_good"
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


def test_identity_excludes_outputs_and_implementation_but_binds_inputs(record, tmp_path):
    other = deepcopy(record)
    other["implementation"]["name"] = "another algorithm"
    other["observation"] = unavailable()
    other["images"]["rectified_left"]["path"] = "renamed"
    assert input_identity(other) == record["input_identity"]
    for mutate in [lambda r: r["timestamps"]["capture"].update(value_ns=101),
                   lambda r: r["calibration"]["geometry"].update(fx=6.),
                   lambda r: r["source"].update(lossless=None),
                   lambda r: r["images"]["rectified_left"].update(stride_bytes=7)]:
        r = deepcopy(record)
        mutate(r)
        assert input_identity(r) != record["input_identity"]
        with pytest.raises(ValueError):
            validate_record(r, tmp_path)


def test_partition_scene_sequence_leakage(record):
    for field in ["scene_id", "sequence_id", "source"]:
        r = deepcopy(record)
        r.update(scene_id="renamed", sequence_id="renamed", partition="held_out")
        r["source"]["artifact"]["sha256"] = "a"*64
        r[field] = deepcopy(record[field])
        reseal(r)
        with pytest.raises(ValueError):
            validate_partitions([record, r])


def test_partitions_allow_multi_implementation_reject_duplicates(record):
    other = deepcopy(record)
    other["implementation"]["name"] = "another"
    validate_partitions([record, other])
    with pytest.raises(ValueError):
        validate_partitions([record, deepcopy(record)])


def test_capture_invariants_across_geometry_strata(record):
    other = deepcopy(record)
    other["implementation"]["name"] = "other"
    other["calibration"]["geometry"]["fx"] = 6.
    other["images"]["rectified_left"]["sha256"] = "c"*64
    reseal(other)
    validate_partitions([record, other])
    other["images"]["original_left"]["sha256"] = "d"*64
    reseal(other)
    with pytest.raises(ValueError):
        validate_partitions([record, other])


def test_same_geometry_cannot_change_rectified_pixels(record):
    other = deepcopy(record)
    other["implementation"]["name"] = "other"
    other["images"]["support_left"]["sha256"] = "d"*64
    reseal(other)
    with pytest.raises(ValueError):
        validate_partitions([record, other])


def test_sequence_source_id_not_reused_for_different_ordinal(record):
    other = deepcopy(record)
    other["ordinal"] = 1
    reseal(other)
    with pytest.raises(ValueError):
        validate_partitions([record, other])


def policy():
    return dict(schema="stereoforge.zed_comparison.policy.v1", primary_reference="NEURAL_LIGHT",
                frozen=False, margins=dict(completeness_loss_pp=2., relative_mae_ratio=1.1,
                                           false_boundary_excess_pp=0.),
                required_regions=["all", "thin", "boundary", "near"], requested_range_m=[.3, 8.],
                freeze_parameters=dict.fromkeys(["correctness", "region_definitions", "zero_reference_error",
                                                "uncertainty", "statistics", "partition_digest"]))


def test_missing_policy_is_unqualified(tmp_path):
    assert load_policy(tmp_path / "missing.json") == dict(status="unqualified", reason="missing_policy")
    path = tmp_path / "policy.json"
    p = policy()
    path.write_text(json.dumps(p))
    result = load_policy(path)
    assert result["status"] == "unqualified" and result["frozen"] is False
    assert json.loads(path.read_text()) == p


def test_frozen_incomplete_policy_rejected(tmp_path):
    p = policy()
    p["frozen"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        load_policy(path)


def test_frozen_policy_ready_is_not_quality_pass(tmp_path):
    p = policy()
    p["frozen"] = True
    p["freeze_parameters"] = dict(
        correctness=dict(mode="depth_tolerance", absolute_m=.01, relative=.01, combine="max"),
        region_definitions={r: dict(description="predeclared mask") for r in p["required_regions"]},
        zero_reference_error=dict(method="absolute_margin", absolute_mae_margin_m=.001),
        uncertainty=dict(method="paired bootstrap", confidence_level=.95, registration_max_m=.001,
                         timestamp_max_ns=100),
        statistics=dict(method="paired bootstrap", unit="scene_sequence", min_scenes=2, min_sequences=2),
        partition_digest="a"*64)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(p))
    assert load_policy(path)["status"] == "policy_ready"
    for name, replacement in [("correctness", {}), ("region_definitions", {"all": {}}),
                              ("uncertainty", "known"), ("statistics", True),
                              ("zero_reference_error", {}), ("partition_digest", "unhashed")]:
        bad = deepcopy(p)
        bad["freeze_parameters"][name] = replacement
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_policy(path)


def test_malformed_format_is_validation_error(record, tmp_path):
    record["observation"]["value"]["depth"]["format"] = []
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


@pytest.mark.parametrize("location", ["source_size", "plane_width", "plane_stride", "timestamp", "settings"])
def test_nested_numeric_domains(record, tmp_path, location):
    if location == "source_size":
        record["source"]["artifact"]["size_bytes"] = True
    elif location == "plane_width":
        record["images"]["original_left"]["width"] = True
    elif location == "plane_stride":
        record["images"]["original_left"]["stride_bytes"] = True
    elif location == "timestamp":
        record["timestamps"]["capture"]["value_ns"] = True
    else:
        record["implementation"]["settings"] = {"nested": [float("inf")]}
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


def test_optional_metadata_planes_are_independent_and_read_only(record, tmp_path):
    obs = record["observation"]["value"]
    for name, fmt, dtype, values in [
        ("confidence", "f32_le", "<f4", [0., np.nan, np.inf, .5]),
        ("provenance", "u8_provenance", "u1", [1, 0, 0, 5]),
        ("source_ids", "i64_le", "<i8", [2**63-1, -1, -1, 0]),
        ("age_ns", "i64_le", "<i8", [0, 0, 0, 2**63-1]),
    ]:
        data = np.asarray(values, dtype=dtype).tobytes()
        (tmp_path / name).write_bytes(data)
        obs[name] = available(dict(path=name, sha256=sha256(data).hexdigest(), size_bytes=len(data),
                                   width=2, height=2, stride_bytes=2*np.dtype(dtype).itemsize, format=fmt))
        if name == "confidence":
            obs["confidence_direction"] = "low_is_good"
        validate_record(record, tmp_path)
        assert (tmp_path / name).read_bytes() == data


def test_confidence_availability_must_match_direction(record, tmp_path):
    record["observation"]["value"]["confidence_direction"] = "high_is_good"
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


@pytest.mark.parametrize("code", [1, 2, 3, 4, 5])
def test_existing_valid_provenance_codes_do_not_claim_freshness(record, tmp_path, code):
    obs = record["observation"]["value"]
    data = bytes([code, 0, 0, code])
    (tmp_path / "origin").write_bytes(data)
    obs["provenance"] = available(dict(path="origin", sha256=sha256(data).hexdigest(), size_bytes=4,
                                        width=2, height=2, stride_bytes=2, format="u8_provenance"))
    # Source ancestry and age remain explicitly unavailable even with a valid code.
    validate_record(record, tmp_path)
    assert obs["age_ns"]["status"] == obs["source_ids"]["status"] == "unavailable"


def test_disparity_correctness_policy(tmp_path):
    p = policy()
    p["freeze_parameters"]["correctness"] = dict(mode="disparity_pixels", max_error_pixels=1.)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(p))
    assert load_policy(path)["status"] == "unqualified"
    for error in [True, 0, -1, "1"]:
        p["freeze_parameters"]["correctness"]["max_error_pixels"] = error
        path.write_text(json.dumps(p))
        with pytest.raises(ValueError):
            load_policy(path)


def test_policy_strict_json_and_typed_fields(tmp_path):
    path = tmp_path / "policy.json"
    for invalid in ['{"schema":1,"schema":2}', '{"frozen":NaN}', '{"frozen":1e999}']:
        path.write_text(invalid)
        with pytest.raises(ValueError):
            load_policy(path)
    for key, value in [("frozen", "false"), ("required_regions", []), ("requested_range_m", [0, 8]),
                       ("margins", dict(completeness_loss_pp=True, relative_mae_ratio=1.1, false_boundary_excess_pp=0))]:
        p = policy()
        p[key] = value
        path.write_text(json.dumps(p))
        with pytest.raises(ValueError):
            load_policy(path)


def rewrite_calibration(record, tmp_path, which, change):
    desc = record["calibration"][which]
    path = tmp_path / desc["path"]
    payload = json.loads(path.read_bytes())
    changed = change(payload)
    data = json.dumps(payload if changed is None else changed).encode()
    path.write_bytes(data)
    desc.update(sha256=sha256(data).hexdigest(), size_bytes=len(data))
    reseal(record)


@pytest.mark.parametrize("which", ["raw", "rectified"])
def test_empty_calibration_with_all_valid_support_rejected(record, tmp_path, which):
    for name in ("support_left", "support_right"):
        desc = record["images"][name]
        replace_plane(tmp_path, desc, [1, 1, 1, 1, 1, 1], "u1")
    rewrite_calibration(record, tmp_path, which, lambda _: {})
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


@pytest.mark.parametrize("mutation", ["missing_cy", "unknown_distortion", "wrong_coefficients", "boolean_coefficient",
                                       "reflection", "nonrigid", "bad_homogeneous", "zero_baseline", "bad_shape",
                                       "wrong_dimensions", "unknown_projection", "missing_model", "baseline_mismatch"])
def test_raw_calibration_models_and_rigid_transform(record, tmp_path, mutation):
    def change(c):
        if mutation == "missing_cy":
            del c["left"]["cy"]
        elif mutation == "unknown_distortion":
            c["left"]["distortion"]["model"] = "vendor_unknown"
        elif mutation == "wrong_coefficients":
            c["left"]["distortion"] = dict(model="brown_conrady", coefficients=[0, 0, 0])
        elif mutation == "boolean_coefficient":
            c["left"]["distortion"] = dict(model="brown_conrady", coefficients=[True, 0, 0, 0])
        elif mutation == "reflection":
            c["right_to_left"][0][0] = -1
        elif mutation == "nonrigid":
            c["right_to_left"][0][0] = 2
        elif mutation == "bad_homogeneous":
            c["right_to_left"][3][0] = .1
        elif mutation == "zero_baseline":
            c["right_to_left"][0][3] = 0
        elif mutation == "bad_shape":
            c["right_to_left"] = [[1]]
        elif mutation == "wrong_dimensions":
            c["left"]["width"] = 3
        elif mutation == "unknown_projection":
            c["right"]["projection"] = "fisheye"
        elif mutation == "baseline_mismatch":
            c["right_to_left"][0][3] = .13
        else:
            del c["left"]["distortion"]
    rewrite_calibration(record, tmp_path, "raw", change)
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


@pytest.mark.parametrize("mutation", ["fx", "fy", "cx", "cy", "baseline", "rotation", "distortion", "dimensions"])
def test_rectified_calibration_matches_declared_geometry(record, tmp_path, mutation):
    def change(c):
        if mutation in ("fx", "fy", "cx", "cy"):
            c["right"][mutation] += .1
        elif mutation == "baseline":
            c["right_to_left"][0][3] = -.12
        elif mutation == "rotation":
            c["right_to_left"][:2] = [[0, -1, 0, .12], [1, 0, 0, 0]]
        elif mutation == "distortion":
            c["left"]["distortion"] = dict(model="brown_conrady", coefficients=[0, 0, 0, 0])
        else:
            c["left"]["height"] = 3
    rewrite_calibration(record, tmp_path, "rectified", change)
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


@pytest.mark.parametrize("model,count", [("none", 0), ("brown_conrady", 4), ("brown_conrady", 5),
                                        ("rational_polynomial", 8)])
def test_known_raw_models_admitted_structurally(record, tmp_path, model, count):
    rewrite_calibration(record, tmp_path, "raw", lambda c: c["left"].update(
        distortion=dict(model=model, coefficients=[0.]*count)))
    validate_record(record, tmp_path)


def test_raw_rigid_rotation_and_nonhorizontal_translation(record, tmp_path):
    def change(c):
        c["right_to_left"] = [[0., -1., 0., 0.], [1., 0., 0., .12], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    rewrite_calibration(record, tmp_path, "raw", change)
    validate_record(record, tmp_path)


def rounded_calibration_fixture(record, tmp_path):
    # Independently rounded float32 values from ideal t=[.12,.001,.002] and
    # float32(norm(ideal)); do not recompute the baseline from rounded t.
    record["calibration"]["geometry"]["baseline_m"] = .12002082914113998
    def raw(c):
        for row, value in enumerate([.11999999731779099, .0010000000474974513, .0020000000949949026]):
            c["right_to_left"][row][3] = value
    rewrite_calibration(record, tmp_path, "raw", raw)
    rewrite_calibration(record, tmp_path, "rectified", lambda c: c["right_to_left"][0].__setitem__(3, .12002082914113998))


def test_independently_float32_rounded_baselines_admitted(record, tmp_path):
    rounded_calibration_fixture(record, tmp_path)
    validate_record(record, tmp_path)


def test_baseline_disagreement_above_representation_bound_rejected(record, tmp_path):
    rounded_calibration_fixture(record, tmp_path)
    def change(c):
        c["right_to_left"][0][3] = .12002082914113998 + 2e-7
        c["right_to_left"][1][3] = c["right_to_left"][2][3] = 0.
    rewrite_calibration(record, tmp_path, "raw", change)
    with pytest.raises(ValueError, match="baseline disagrees"):
        validate_record(record, tmp_path)


def test_rectified_axis_alignment_remains_strict_with_baseline_tolerance(record, tmp_path):
    rounded_calibration_fixture(record, tmp_path)
    rewrite_calibration(record, tmp_path, "rectified", lambda c: c["right_to_left"][0].__setitem__(3, .12002082914113998 + 5e-10))
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


def test_numeric_geometry_equivalence_preserves_identity_and_stratum(record):
    other = deepcopy(record)
    other["implementation"]["name"] = "other"
    other["calibration"]["geometry"].update(fx=5, fy=5)
    assert input_identity(other) == record["input_identity"]
    assert input_identity(other) == independent_identity(other)
    validate_partitions([record, other])
    other["images"]["rectified_left"]["sha256"] = "d"*64
    reseal(other)
    with pytest.raises(ValueError):
        validate_partitions([record, other])


@pytest.mark.parametrize("key,value", [("kind", "svo2"), ("codec", "unknown"), ("lossless", None)])
def test_same_artifact_source_metadata_is_immutable(record, key, value):
    other = deepcopy(record)
    other["ordinal"] = 1
    other["source_id"] = 1
    other["source"][key] = value
    reseal(other)
    with pytest.raises(ValueError):
        validate_partitions([record, other])


def test_identity_float_normalization_never_rounds_distinct_integers(record):
    a, b = deepcopy(record), deepcopy(record)
    a["calibration"]["geometry"]["fx"] = 2**53
    b["calibration"]["geometry"]["fx"] = 2**53+1
    assert input_identity(a) != input_identity(b)
    b["calibration"]["geometry"]["fx"] = float(2**53)
    assert input_identity(a) == input_identity(b)
    a["calibration"]["geometry"]["cx_left"] = -0.0
    b["calibration"]["geometry"]["cx_left"] = 0
    assert input_identity(a) == input_identity(b)
    b["timestamps"]["capture"]["value_ns"] += 1
    assert input_identity(a) != input_identity(b)


def test_large_finite_nonrigid_rotation_rejected_without_overflow(record, tmp_path):
    rewrite_calibration(record, tmp_path, "raw", lambda c: c["right_to_left"][0].__setitem__(0, 1e308))
    with pytest.raises(ValueError):
        validate_record(record, tmp_path)


def test_resolved_vendor_file_stays_opaque(record, tmp_path):
    data = b"vendor calibration syntax is not normalized JSON\x00\xff"
    (tmp_path / "vendor.conf").write_bytes(data)
    record["calibration"]["resolved_file"] = available(dict(path="vendor.conf", size_bytes=len(data),
                                                             sha256=sha256(data).hexdigest()))
    reseal(record)
    validate_record(record, tmp_path)


def test_changed_calibration_bytes_keep_distinct_identity(record, tmp_path):
    before = record["input_identity"]
    rewrite_calibration(record, tmp_path, "raw", lambda c: c["left"].update(fx=5))
    assert record["input_identity"] != before  # equal numbers, deliberately different source-file bytes
    validate_record(record, tmp_path)


def test_enum_geometry_cannot_claim_right_or_range(record, tmp_path):
    for key, value in [("reference_camera", "right"), ("coordinate_axes", "unknown"),
                       ("depth_convention", "range"), ("pixel_centres", "half_integer")]:
        other = deepcopy(record)
        other["calibration"]["geometry"][key] = value
        reseal(other)
        with pytest.raises(ValueError):
            validate_record(other, tmp_path)


def test_hash_reads_are_bounded_and_duplicate_paths_hashed_once(record, tmp_path, monkeypatch):
    # All RGB descriptors share the same file; logical labels do not cause repeated I/O.
    for key in ("original_right", "rectified_left", "rectified_right"):
        record["images"][key] = deepcopy(record["images"]["original_left"])
    reseal(record)
    real_open = Path.open
    reads = []
    class Tracked:
        def __init__(self, stream, path):
            self.stream, self.path = stream, path
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def read(self, size=-1):
            reads.append((self.path.name, size))
            return self.stream.read(size)
    def tracked_open(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        return Tracked(stream, path)
    monkeypatch.setattr(Path, "open", tracked_open)
    validate_record(record, tmp_path)
    assert all(0 < size <= 16*1024*1024+1 for _, size in reads)
    # One read of data and one EOF read, despite four references.
    assert len([size for name, size in reads if name == "original_left"]) == 2


def test_checked_in_policy_stays_provisional():
    path = Path(__file__).resolve().parents[1] / "configs/benchmarks/zed_quality_policy.json"
    loaded = load_policy(path)
    assert loaded["status"] == "unqualified"
    assert loaded["margins"] == dict(completeness_loss_pp=2., relative_mae_ratio=1.1, false_boundary_excess_pp=0.)
    assert loaded["requested_range_m"] == [.3, 8.]
    assert all(value is None for value in loaded["freeze_parameters"].values())
