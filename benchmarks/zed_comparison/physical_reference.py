"""Strict physical-reference metadata admission; never survey qualification."""
from contextlib import ExitStack
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from . import schema as s

SCHEMA = "stereoforge.zed_comparison.physical_reference.v1"
KEYS = {"schema", "kind", "independent", "artifact", "calibration_sha256", "units",
        "coordinate_axes", "sampling", "interpolated", "dense_geometry_labels",
        "labels", "registration", "timestamp_uncertainty", "survey"}


def _registration(value):
    s._keys(value, {"method", "transform_to_left", "max_error_m", "justification"}, "registration")
    s._text(value["method"], "registration.method")
    s._text(value["justification"], "registration.justification")
    s._number(value["max_error_m"], "registration.max_error_m", minimum=0)
    matrix = value["transform_to_left"]
    s._require(type(matrix) is list and len(matrix) == 4, "registration: expected rigid4x4")
    for row in matrix:
        s._require(type(row) is list and len(row) == 4, "registration: expected rigid4x4")
        for number in row:
            s._number(number, "registration.transform_to_left")
    transform = np.asarray(matrix, dtype=float)
    rotation = transform[:3, :3]
    s._require(np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
               and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
               and math.isclose(float(np.linalg.det(rotation)), 1, abs_tol=1e-6, rel_tol=0),
               "registration: expected proper rigid transform")


def _timestamp(value):
    s._keys(value, {"max_error_ns", "justification"}, "timestamp_uncertainty")
    s._integer(value["max_error_ns"], "timestamp_uncertainty.max_error_ns")
    s._text(value["justification"], "timestamp_uncertainty.justification")


def _survey(value):
    s._keys(value, {"plane_normal_left", "plane_offset_m", "axial_intercept_m", "absolute_error_m", "method"}, "survey")
    normal = value["plane_normal_left"]
    s._require(type(normal) is list and len(normal) == 3, "survey: expected three-component normal")
    for number in normal:
        s._number(number, "survey.plane_normal_left")
    s._require(math.isclose(math.hypot(*normal), 1, abs_tol=1e-6, rel_tol=0), "survey: normal must be unit")
    s._number(value["plane_offset_m"], "survey.plane_offset_m")
    s._number(value["axial_intercept_m"], "survey.axial_intercept_m", positive=True)
    s._number(value["absolute_error_m"], "survey.absolute_error_m", minimum=0)
    s._text(value["method"], "survey.method")
    s._require(normal[2] != 0 and math.isclose(-value["plane_offset_m"] / normal[2],
               value["axial_intercept_m"], abs_tol=1e-12, rel_tol=1e-9), "survey: inconsistent axial intercept")


def validate_physical_reference(document, *, root=None, files=None):
    """Validate metadata and consumed files relative to an explicit root/cache.

    Invalid declarations raise ValueError; declared unavailable evidence produces
    an unqualified result. metadata_ready establishes no physical accuracy claim.
    """
    if files is None:
        s._require(root is not None, "physical reference requires explicit root")
        with ExitStack() as stack:
            return validate_physical_reference(document, files=s._Files(root, stack))
    s._require(isinstance(files, s._Files), "files must be schema._Files")
    s._require(root is None or Path(root).resolve() == files.root, "root disagrees with file cache")
    s._keys(document, KEYS, "physical reference")
    s._require(document["schema"] == SCHEMA, "unknown physical reference schema")
    s._require(document["kind"] in ("surveyed_plane", "scanner", "lidar"), "unknown physical reference kind")
    s._require(document["independent"] is True, "physical reference must be independent")
    s._hex(document["calibration_sha256"], 64, "calibration_sha256")
    s._require(document["units"] == "m", "physical reference units must be m")
    s._require(document["coordinate_axes"] == "x_right_y_down_z_forward", "wrong coordinate axes")
    s._require(document["sampling"] in ("dense", "sparse"), "unknown sampling")
    s._require(document["interpolated"] is False, "interpolated references forbidden")
    s._require(type(document["dense_geometry_labels"]) is bool, "dense_geometry_labels must be boolean")
    s._require(not document["dense_geometry_labels"] or document["sampling"] == "dense", "sparse reference cannot claim dense geometry labels")
    checked = set()

    def descriptor(value):
        s._descriptor(value)
        files.check(value)
        checked.add(value["path"])

    descriptor(document["artifact"])
    reasons = []
    evidence = {}
    for name, check in (("labels", descriptor), ("registration", _registration),
                        ("timestamp_uncertainty", _timestamp), ("survey", _survey)):
        evidence[name] = s._availability(document[name], check, name)
    required = ["registration", "timestamp_uncertainty"]
    if document["kind"] == "surveyed_plane":
        required.append("survey")
    if document["dense_geometry_labels"]:
        required.append("labels")
    for name in required:
        if not evidence[name]:
            reasons.append(f"{name}: {document[name]['reason']}")
    # A shared cache also contains earlier frames; only this call's dependencies
    # belong to this admission boundary. The caller retains its final global check.
    for path in checked:
        files._unchanged(path, files.fingerprints[path])
    return {"status": "unqualified" if reasons else "metadata_ready", "reasons": reasons,
            "metadata": deepcopy(document), "checked_paths": sorted(checked),
            "sampling": document["sampling"],
            "dense_geometry_labels_allowed": document["dense_geometry_labels"] and evidence["labels"],
            "quality_qualified": False}
