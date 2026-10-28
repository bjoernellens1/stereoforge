"""Version-one portable benchmark records and policy admission.

Structural validity never establishes quality, calibration accuracy or freshness.
Raw little-endian depth bytes, including invalid NaN payloads, are read only.
Provenance follows stereoforge.types.Provenance: 0 INVALID, 1 STEREO,
2 REPROJECTED, 3 FUSED, 4 NEURAL, 5 STEREO_LEFT_AGGREGATED. Codes describe
origin, not freshness: age/source IDs/settings remain separate, and unknown SDK
ancestry stays unavailable. Approximate metric depth must remain invalid.

Both calibration FileDescs contain normalized strict JSON with exactly:
{schema: stereoforge.zed_comparison.calibration.v1, space: raw|rectified,
 left: Camera, right: Camera, right_to_left: rigid 4x4, units: m,
 pixel_centres: integer_centres, coordinate_axes: x_right_y_down_z_forward}.
Camera has width/height (positive integers), fx/fy (positive), cx/cy (finite),
projection: perspective, distortion: {model, coefficients}. Supported distortion
models are none (0 coefficients), brown_conrady (4 or 5: k1,k2,p1,p2[,k3]),
rational_polynomial (8: k1,k2,p1,p2,k3,k4,k5,k6), matching OpenCV conventions.
Transforms map right camera coordinates to left, in metres. Rectified models
have no distortion, equal fy/cy, shared fx, identity rotation and +X baseline.
Raw camera dimensions match original RGB; rectified intrinsics/dimensions and
baseline agree with record geometry. Proper rotation tolerance is absolute 1e-6,
homogeneous/rectified alignment tolerance 1e-12. Baseline comparison admits
independent float32 representations at relative 1e-6 and absolute 1e-9 metres;
this representation admission tolerance is not a scientific or physical accuracy
waiver. These are structural checks, not physical qualification.
Support masks are binary declarations bound to known geometry, not mathematical
proof of support; Task 3 must validate map generation. Opaque vendor metadata
can be retained as implementation artifacts; resolved_file stays opaque bytes.
Different calibration bytes retain different content hashes. Only the five
declared record-geometry float members normalize int/float spellings for identity.

Freeze parameters (all required when frozen):
* correctness: {mode: disparity_pixels, max_error_pixels: positive}, or
  {mode: depth_tolerance, absolute_m: positive, relative: nonnegative, combine: max}.
* region_definitions: required region names -> objects with nonempty description;
  additional finite JSON details encode the actual predeclared mask procedure.
* zero_reference_error: {method: absolute_margin, absolute_mae_margin_m: positive}.
* uncertainty: {method: label, confidence_level: (0,1), registration_max_m:
  nonnegative, timestamp_max_ns: nonnegative integer}.
* statistics: {method: label, unit: scene_sequence, min_scenes: positive integer,
  min_sequences: positive integer}.
* partition_digest: lowercase SHA256 of the frozen partition manifest.
Method labels are metadata, not proof that a scientific procedure was executed.
"""
from contextlib import ExitStack
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat

import numpy as np

FRAME_SCHEMA = "stereoforge.zed_comparison.frame.v1"
POLICY_SCHEMA = "stereoforge.zed_comparison.policy.v1"
CALIBRATION_SCHEMA = "stereoforge.zed_comparison.calibration.v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
INT64_MAX = 2**63 - 1
IMAGE_NAMES = ("original_left", "original_right", "rectified_left", "rectified_right",
               "support_left", "support_right")
FILE_KEYS = {"path", "sha256", "size_bytes"}
PLANE_KEYS = FILE_KEYS | {"width", "height", "stride_bytes", "format"}
FORMATS = {"rgb8": ("u1", 3), "u8_mask": ("u1", 1), "f32_le_m": ("<f4", 4),
           "f32_le": ("<f4", 4), "u8_provenance": ("u1", 1), "i64_le": ("<i8", 8)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, keys, where):
    _require(type(value) is dict and set(value) == set(keys), f"{where}: expected keys {sorted(keys)}")


def _text(value, where):
    _require(type(value) is str and bool(value.strip()), f"{where}: expected nonempty string")


def _integer(value, where, minimum=0, maximum=None):
    _require(type(value) is int and value >= minimum and (maximum is None or value <= maximum),
             f"{where}: integer outside allowed range")


def _number(value, where, minimum=None, positive=False):
    _require(type(value) in (int, float), f"{where}: expected number, not boolean")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite and (minimum is None or value >= minimum) and (not positive or value > 0),
             f"{where}: nonfinite or outside allowed range")


def _hex(value, length, where):
    _require(type(value) is str and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None,
             f"{where}: expected lowercase {length}-digit hex")


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _require(math.isfinite(value), "nonfinite JSON value")
    elif type(value) is list:
        for v in value:
            _json_value(v)
    elif type(value) is dict:
        for k, v in value.items():
            _require(type(k) is str, "JSON keys must be strings")
            _json_value(v)
    else:
        raise ValueError("not a JSON value")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _read_json(path):
    try:
        with Path(path).open("rb") as stream:
            data = stream.read(MAX_JSON_BYTES + 1)
        _require(len(data) <= MAX_JSON_BYTES, "JSON exceeds bounded size")
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_reject_constant)
        _json_value(value)  # also rejects numeric overflow such as 1e999
        _require(type(value) is dict, "JSON document must be an object")
        return value
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"cannot read strict JSON: {path}") from exc


def _availability(value, check, where):
    _require(type(value) is dict, f"{where}: expected availability object")
    if value.get("status") == "available":
        _keys(value, {"status", "value"}, where)
        check(value["value"])
        return True
    _keys(value, {"status", "reason"}, where)
    _require(value["status"] == "unavailable", f"{where}: unknown status")
    _text(value["reason"], where + ".reason")
    return False


def _descriptor(desc, plane=False):
    _keys(desc, PLANE_KEYS if plane else FILE_KEYS, "file descriptor")
    path = desc["path"]
    _text(path, "path")
    parts = path.split("/")
    _require(not PurePosixPath(path).is_absolute() and "\\" not in path
             and all(p not in ("", ".", "..") for p in parts), "unsafe relative POSIX path")
    _hex(desc["sha256"], 64, "sha256")
    _integer(desc["size_bytes"], "size_bytes")
    if plane:
        _integer(desc["width"], "width", 1)
        _integer(desc["height"], "height", 1)
        _integer(desc["stride_bytes"], "stride_bytes", 1)
        _require(type(desc["format"]) is str and desc["format"] in FORMATS, "unknown plane format")
        _, bpp = FORMATS[desc["format"]]
        _require(desc["stride_bytes"] >= desc["width"] * bpp, "stride smaller than pixel row")
        _require(desc["size_bytes"] == desc["stride_bytes"] * desc["height"], "wrong plane extent")


class _Files:
    """One validation call, bounded hash reads and one hash per relative path."""
    def __init__(self, root, stack):
        self.root = Path(root).resolve()
        self.stack = stack
        self.hashes = {}
        self.json = {}

    def check(self, desc, json_object=False):
        relative = desc["path"]
        path = self.root
        try:
            for part in relative.split("/"):
                path = path / part
                _require(not path.is_symlink(), "symlink component forbidden")
            _require(path.is_relative_to(self.root), "file escapes record root")
            _require(stat.S_ISREG(path.stat().st_mode), "not a regular file")
            if relative not in self.hashes:
                digest = sha256()
                size = 0
                with path.open("rb") as stream:
                    while chunk := stream.read(HASH_CHUNK_BYTES):
                        digest.update(chunk)
                        size += len(chunk)
                self.hashes[relative] = (digest.hexdigest(), size)
            _require(self.hashes[relative] == (desc["sha256"], desc["size_bytes"]),
                     f"hash/size mismatch: {relative}")
            if json_object and relative not in self.json:
                self.json[relative] = _read_json(path)
            return path
        except OSError as exc:
            raise ValueError(f"cannot validate file: {relative}") from exc

    def plane(self, desc):
        path = self.check(desc)
        dtype, bpp = FORMATS[desc["format"]]
        mapping = np.memmap(path, mode="r", dtype="u1")
        self.stack.callback(mapping._mmap.close)
        width = desc["width"] * (3 if desc["format"] == "rgb8" else 1)
        step = 1 if desc["format"] == "rgb8" else bpp
        return np.ndarray((desc["height"], width), dtype=dtype, buffer=mapping,
                          strides=(desc["stride_bytes"], step))


def _content(desc):
    return {k: v for k, v in desc.items() if k != "path"}


def _normalized_geometry(geometry):
    result = dict(geometry)
    for key in ("fx", "fy", "baseline_m", "cx_left", "cx_right"):
        _number(result[key], key)
        value = result[key]
        as_float = float(value)
        # Normalize -0.0 as well: it is the same principal-point coordinate.
        # Preserve integers not exactly representable in float64; distinct
        # finite camera parameters must not collapse to an artificial identity.
        result[key] = (as_float if as_float == value else value) if value != 0 else 0.0
    return result


def _calibration_identity(calibration):
    resolved = calibration["resolved_file"]
    if resolved["status"] == "available":
        resolved = {"status": "available", "value": _content(resolved["value"])}
    return {"raw": _content(calibration["raw"]), "rectified": _content(calibration["rectified"]),
            "resolved_file": resolved, "geometry": _normalized_geometry(calibration["geometry"])}


def _camera(camera):
    _keys(camera, {"width", "height", "fx", "fy", "cx", "cy", "projection", "distortion"}, "camera")
    for key in ("width", "height"):
        _integer(camera[key], "camera." + key, 1)
    for key in ("fx", "fy", "cx", "cy"):
        _number(camera[key], "camera." + key, positive=key in ("fx", "fy"))
    _require(camera["projection"] == "perspective", "unsupported camera projection")
    distortion = camera["distortion"]
    _keys(distortion, {"model", "coefficients"}, "distortion")
    lengths = {"none": (0,), "brown_conrady": (4, 5), "rational_polynomial": (8,)}
    model, coefficients = distortion["model"], distortion["coefficients"]
    _require(type(model) is str and model in lengths, "unknown distortion model")
    _require(type(coefficients) is list and len(coefficients) in lengths[model], "wrong distortion coefficient count")
    for coefficient in coefficients:
        _number(coefficient, "distortion coefficient")


def _calibration_snapshot(snapshot, space, geometry, images):
    _keys(snapshot, {"schema", "space", "left", "right", "right_to_left", "units", "pixel_centres", "coordinate_axes"}, "calibration snapshot")
    _require(snapshot["schema"] == CALIBRATION_SCHEMA and snapshot["space"] == space,
             "unknown calibration snapshot schema/space")
    for key, expected in dict(units="m", pixel_centres="integer_centres", coordinate_axes="x_right_y_down_z_forward").items():
        _require(snapshot[key] == expected, f"unsupported calibration {key}")
    for eye in ("left", "right"):
        camera = snapshot[eye]
        _camera(camera)
        image = images[("original_" if space == "raw" else "rectified_") + eye]
        _require((camera["width"], camera["height"]) == (image["width"], image["height"]),
                 "camera/image dimensions disagree")
    transform = snapshot["right_to_left"]
    _require(type(transform) is list and len(transform) == 4 and
             all(type(row) is list and len(row) == 4 for row in transform), "stereo transform must be 4x4")
    for row in transform:
        for value in row:
            _number(value, "transform element")
    t = np.asarray(transform, dtype=np.float64)
    _require(bool(np.allclose(t[3], [0, 0, 0, 1], rtol=0, atol=1e-12)), "invalid homogeneous transform")
    rotation = t[:3, :3]
    _require(bool(np.all(np.abs(rotation) <= 1 + 1e-6)), "rotation element outside rigid range")
    _require(bool(np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)) and
             math.isclose(float(np.linalg.det(rotation)), 1., rel_tol=0, abs_tol=1e-6), "stereo rotation must be proper rigid")
    baseline = math.hypot(*t[:3, 3])
    _require(math.isfinite(baseline) and baseline > 0, "stereo baseline must be positive finite")
    _require(math.isclose(baseline, geometry["baseline_m"], rel_tol=1e-6, abs_tol=1e-9),
             "snapshot baseline disagrees with geometry")
    if space == "rectified":
        expected = np.eye(4)
        expected[0, 3] = geometry["baseline_m"]
        _require(bool(np.allclose(t, expected, rtol=0, atol=1e-12)), "rectified stereo transform must align +X")
        left, right = snapshot["left"], snapshot["right"]
        _require(left["cy"] == right["cy"], "rectified vertical principal points disagree")
        for eye, camera in (("left", left), ("right", right)):
            _require(camera["distortion"]["model"] == "none", "rectified camera must be undistorted")
            for key in ("fx", "fy", "width", "height"):
                _require(camera[key] == geometry[key], f"rectified {key} disagrees with geometry")
            _require(camera["cx"] == geometry["cx_" + eye], "rectified horizontal principal point disagrees")


def _canonical(value):
    _json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def input_identity(record):
    """Hash canonical input content, independent of filenames, algorithm and output.

    Callers may construct fixtures before input_identity exists. Full structural
    and on-disk validation is deliberately the separate validate_record operation.
    """
    try:
        source = record["source"]
        payload = {
            "source": {**source, "artifact": _content(source["artifact"])},
            "ordinal": record["ordinal"], "source_id": record["source_id"],
            "calibration": _calibration_identity(record["calibration"]),
            "capture": record["timestamps"]["capture"],
            "images": {k: _content(v) for k, v in record["images"].items()},
        }
        return sha256(_canonical(payload).encode()).hexdigest()
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("incomplete input identity") from exc


def _timestamp(value):
    _keys(value, {"value_ns", "clock_domain"}, "timestamp")
    _integer(value["value_ns"], "value_ns")
    _text(value["clock_domain"], "clock_domain")


def _validate(record, files=None):
    _json_value(record)
    _keys(record, {"schema", "scene_id", "sequence_id", "partition", "ordinal", "source_id", "source",
                   "calibration", "timestamps", "images", "observation", "implementation", "input_identity"}, "record")
    _require(record["schema"] == FRAME_SCHEMA, "unsupported frame schema")
    for key in ("scene_id", "sequence_id"):
        _text(record[key], key)
    _require(record["partition"] in ("development", "held_out"), "unknown partition")
    _integer(record["ordinal"], "ordinal")
    _integer(record["source_id"], "source_id", maximum=INT64_MAX)

    def file(desc, json_object=False):
        _descriptor(desc)
        if files:
            files.check(desc, json_object=json_object)

    source = record["source"]
    _keys(source, {"kind", "artifact", "codec", "lossless"}, "source")
    _require(source["kind"] in ("svo2", "analytic", "physical", "middlebury"), "unknown source kind")
    file(source["artifact"])
    _text(source["codec"], "codec")
    _require(source["lossless"] is None or type(source["lossless"]) is bool, "lossless must be boolean or null")
    cal = record["calibration"]
    _keys(cal, {"raw", "rectified", "resolved_file", "geometry"}, "calibration")
    file(cal["raw"], True)
    file(cal["rectified"], True)
    _availability(cal["resolved_file"], file, "resolved_file")
    g = cal["geometry"]
    _keys(g, {"width", "height", "fx", "fy", "baseline_m", "cx_left", "cx_right", "projection",
              "depth_convention", "units", "coordinate_frame", "pixel_centres", "reference_camera",
              "coordinate_axes"}, "geometry")
    for key in ("width", "height"):
        _integer(g[key], key, 1)
    for key in ("fx", "fy", "baseline_m"):
        _number(g[key], key, positive=True)
    for key in ("cx_left", "cx_right"):
        _number(g[key], key)
    _text(g["coordinate_frame"], "coordinate_frame")
    for key, expected in dict(projection="perspective", depth_convention="axial_z", units="m",
                              pixel_centres="integer_centres", reference_camera="left",
                              coordinate_axes="x_right_y_down_z_forward").items():
        _require(g[key] == expected, f"unsupported geometry {key}")
    ts = record["timestamps"]
    _keys(ts, {"capture", "ingress", "sensors"}, "timestamps")
    _timestamp(ts["capture"])
    _availability(ts["ingress"], _timestamp, "ingress")
    _require(type(ts["sensors"]) is list, "sensors must be list")
    names = set()
    for sensor in ts["sensors"]:
        _keys(sensor, {"name", "value_ns", "clock_domain"}, "sensor")
        _text(sensor["name"], "sensor.name")
        _require(sensor["name"] not in names, "duplicate sensor name")
        names.add(sensor["name"])
        _timestamp({k: sensor[k] for k in ("value_ns", "clock_domain")})

    def plane(desc, fmt, aligned=True):
        _descriptor(desc, True)
        _require(desc["format"] == fmt, f"expected plane format {fmt}")
        if aligned:
            _require((desc["width"], desc["height"]) == (g["width"], g["height"]), "plane geometry mismatch")
        data = files.plane(desc) if files else None
        if data is not None and fmt == "u8_mask":
            _require(bool(np.all((data == 0) | (data == 1))), "mask must be binary")
        return data

    images = record["images"]
    _keys(images, IMAGE_NAMES, "images")
    for key in IMAGE_NAMES:
        plane(images[key], "u8_mask" if key.startswith("support") else "rgb8", not key.startswith("original"))
    _require(tuple(images["original_left"][k] for k in ("width", "height")) ==
             tuple(images["original_right"][k] for k in ("width", "height")), "original stereo dimensions differ")
    if files:
        for space in ("raw", "rectified"):
            _calibration_snapshot(files.json[cal[space]["path"]], space, g, images)

    def observation(obs):
        _keys(obs, {"depth", "validity", "confidence", "confidence_direction", "provenance", "source_ids", "age_ns"}, "observation")
        depth = plane(obs["depth"], "f32_le_m")
        valid = plane(obs["validity"], "u8_mask")
        if files:
            valid = valid != 0
            _require(bool(np.all(np.isfinite(depth[valid]) & (depth[valid] > 0))), "valid metric depth must be finite positive")
        arrays = {}
        for key, fmt in dict(confidence="f32_le", provenance="u8_provenance", source_ids="i64_le", age_ns="i64_le").items():
            def check(desc, key=key, fmt=fmt):
                arrays[key] = plane(desc, fmt)
            _availability(obs[key], check, key)
        has_confidence = obs["confidence"]["status"] == "available"
        _require(obs["confidence_direction"] in (("low_is_good", "high_is_good") if has_confidence else ("unavailable",)),
                 "confidence direction inconsistent with availability")
        if files:
            if "confidence" in arrays:
                _require(bool(np.all(np.isfinite(arrays["confidence"][valid]))), "valid confidence must be finite")
            if "age_ns" in arrays:
                _require(bool(np.all(arrays["age_ns"] >= 0)), "negative pixel age")
            if "source_ids" in arrays:
                _require(bool(np.all(arrays["source_ids"][valid] >= 0)), "negative valid source ID")
            if "provenance" in arrays:
                p = arrays["provenance"][valid]
                _require(bool(np.all((p >= 1) & (p <= 5))), "unknown valid provenance")
    _availability(record["observation"], observation, "observation")
    impl = record["implementation"]
    _keys(impl, {"name", "version", "commit", "device", "settings", "artifacts"}, "implementation")
    _text(impl["name"], "implementation.name")
    _text(impl["version"], "implementation.version")
    _availability(impl["commit"], lambda x: _hex(x, 40, "commit"), "commit")
    _availability(impl["device"], lambda x: _require(type(x) is dict and bool(x), "device must be nonempty object"), "device")
    _require(type(impl["settings"]) is dict, "settings must be object")
    _require(type(impl["artifacts"]) is list, "artifacts must be list")
    for desc in impl["artifacts"]:
        file(desc)
    _hex(record["input_identity"], 64, "input_identity")
    _require(record["input_identity"] == input_identity(record), "canonical input identity changed")


def validate_record(record, root):
    """Validate without mutation; hashes/content are checked against root files."""
    with ExitStack() as stack:
        _validate(record, _Files(root, stack))


def load_record(path):
    path = Path(path)
    record = _read_json(path)
    validate_record(record, path.parent)
    return record


def validate_partitions(records):
    """Check identities and leakage only; load_record checks actual files first.

    Geometry stratum is calibration content plus projection geometry, excluding
    paths. Rectified pixels may differ across strata; original capture may not.
    """
    _require(type(records) is list, "records must be list")
    partitions = {"scene": {}, "sequence": {}, "source": {}}
    seen = set()
    captures, strata, sequence_frames, source_metadata = {}, {}, {}, {}
    for record in records:
        _validate(record)
        source = record["source"]["artifact"]["sha256"]
        metadata = _canonical({k: record["source"][k] for k in ("kind", "codec", "lossless")})
        _require(source_metadata.setdefault(source, metadata) == metadata, "same source artifact has conflicting metadata")
        for kind, identity in (("scene", record["scene_id"]), ("sequence", record["sequence_id"]), ("source", source)):
            previous = partitions[kind].setdefault(identity, record["partition"])
            _require(previous == record["partition"], f"cross-partition {kind} leakage")
        impl = {k: record["implementation"][k] for k in ("name", "version", "commit", "settings", "device")}
        duplicate_key = (_canonical(impl), source, record["ordinal"], record["source_id"], record["input_identity"])
        _require(duplicate_key not in seen, "duplicate implementation/frame record")
        seen.add(duplicate_key)
        capture_key = (source, record["ordinal"])
        original = {k: _content(record["images"][k]) for k in ("original_left", "original_right")}
        capture = _canonical([record["source_id"], record["timestamps"]["capture"], original])
        _require(captures.setdefault(capture_key, capture) == capture, "canonical capture changed across strata")
        stratum = _canonical(_calibration_identity(record["calibration"]))
        rectified = _canonical({k: _content(record["images"][k]) for k in IMAGE_NAMES if not k.startswith("original")})
        _require(strata.setdefault((capture_key, stratum), rectified) == rectified,
                 "rectified pixels/support changed within calibration stratum")
        sequence_key = (record["sequence_id"], record["source_id"])
        frame = _canonical([source, record["ordinal"], record["timestamps"]["capture"]])
        _require(sequence_frames.setdefault(sequence_key, frame) == frame, "sequence source ID reused for another capture")


def _freeze_parameters(params, required_regions, frozen):
    _keys(params, {"correctness", "region_definitions", "zero_reference_error", "uncertainty", "statistics", "partition_digest"}, "freeze_parameters")
    for key, value in params.items():
        if value is None:
            _require(not frozen, f"frozen policy missing {key}")
            continue
        if key == "correctness":
            _require(type(value) is dict, "correctness must be object")
            if value.get("mode") == "disparity_pixels":
                _keys(value, {"mode", "max_error_pixels"}, key)
                _number(value["max_error_pixels"], "max_error_pixels", positive=True)
            else:
                _keys(value, {"mode", "absolute_m", "relative", "combine"}, key)
                _require(value["mode"] == "depth_tolerance" and value["combine"] == "max", "unknown depth tolerance")
                _number(value["absolute_m"], "absolute_m", positive=True)
                _number(value["relative"], "relative", minimum=0)
        elif key == "region_definitions":
            _require(type(value) is dict and set(required_regions) <= set(value), "missing region definitions")
            for region, definition in value.items():
                _text(region, "region")
                _require(type(definition) is dict, "region definition must be object")
                _text(definition.get("description"), "region description")
        elif key == "zero_reference_error":
            _keys(value, {"method", "absolute_mae_margin_m"}, key)
            _require(value["method"] == "absolute_margin", "unknown zero-reference policy")
            _number(value["absolute_mae_margin_m"], "absolute_mae_margin_m", positive=True)
        elif key == "uncertainty":
            _keys(value, {"method", "confidence_level", "registration_max_m", "timestamp_max_ns"}, key)
            _text(value["method"], "uncertainty.method")
            _number(value["confidence_level"], "confidence_level", positive=True)
            _require(value["confidence_level"] < 1, "confidence_level must be below one")
            _number(value["registration_max_m"], "registration_max_m", minimum=0)
            _integer(value["timestamp_max_ns"], "timestamp_max_ns")
        elif key == "statistics":
            _keys(value, {"method", "unit", "min_scenes", "min_sequences"}, key)
            _text(value["method"], "statistics.method")
            _require(value["unit"] == "scene_sequence", "correlated frames are not independent units")
            _integer(value["min_scenes"], "min_scenes", 1)
            _integer(value["min_sequences"], "min_sequences", 1)
        else:
            _hex(value, 64, "partition_digest")


def load_policy(path):
    """Return unqualified or policy_ready, never a quality-qualified decision."""
    path = Path(path)
    if not path.exists():
        return {"status": "unqualified", "reason": "missing_policy"}
    policy = _read_json(path)
    _keys(policy, {"schema", "primary_reference", "frozen", "margins", "required_regions", "requested_range_m", "freeze_parameters"}, "policy")
    _require(policy["schema"] == POLICY_SCHEMA, "unknown policy schema")
    _require(policy["primary_reference"] == "NEURAL_LIGHT", "unknown primary reference")
    _require(type(policy["frozen"]) is bool, "frozen must be boolean")
    margins = policy["margins"]
    _keys(margins, {"completeness_loss_pp", "relative_mae_ratio", "false_boundary_excess_pp"}, "margins")
    for key, value in margins.items():
        _number(value, key, minimum=0, positive=key == "relative_mae_ratio")
    regions = policy["required_regions"]
    _require(type(regions) is list and all(type(r) is str for r in regions), "regions must be list of strings")
    _require(len(set(regions)) == len(regions) and {"all", "thin", "boundary", "near"} <= set(regions), "required regions missing/duplicated")
    for region in regions:
        _text(region, "required region")
    limits = policy["requested_range_m"]
    _require(type(limits) is list and len(limits) == 2, "range must contain two bounds")
    for bound in limits:
        _number(bound, "range", positive=True)
    _require(limits[0] < limits[1], "range must increase")
    _freeze_parameters(policy["freeze_parameters"], regions, policy["frozen"])
    return {**policy, "status": "policy_ready" if policy["frozen"] else "unqualified"}
