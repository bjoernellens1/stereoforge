import copy
from contextlib import ExitStack
from hashlib import sha256

import pytest

from benchmarks.zed_comparison.physical_reference import validate_physical_reference
from benchmarks.zed_comparison.schema import _Files


def available(value):
    return {"status": "available", "value": value}


def unavailable():
    return {"status": "unavailable", "reason": "not measured"}


@pytest.fixture
def reference(tmp_path):
    data = b"independent survey"
    (tmp_path / "survey.bin").write_bytes(data)
    desc = {"path": "survey.bin", "sha256": sha256(data).hexdigest(), "size_bytes": len(data)}
    return {
        "schema": "stereoforge.zed_comparison.physical_reference.v1",
        "kind": "surveyed_plane", "independent": True, "artifact": desc,
        "calibration_sha256": "a" * 64, "units": "m",
        "coordinate_axes": "x_right_y_down_z_forward", "sampling": "dense",
        "interpolated": False, "dense_geometry_labels": True, "labels": available(desc.copy()),
        "registration": available({"method": "survey", "transform_to_left": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "max_error_m": .001, "justification": "measured bound"}),
        "timestamp_uncertainty": available({"max_error_ns": 10, "justification": "clock measurement"}),
        "survey": available({"plane_normal_left": [0,0,1], "plane_offset_m": -2, "axial_intercept_m": 2, "absolute_error_m": .001, "method": "survey"}),
    }


def test_metadata_ready(reference, tmp_path):
    result = validate_physical_reference(reference, root=tmp_path)
    assert result["status"] == "metadata_ready"
    assert result["checked_paths"] == ["survey.bin"]
    assert result["dense_geometry_labels_allowed"] is True
    assert result["quality_qualified"] is False


def test_missing_root_and_evidence(reference, tmp_path):
    with pytest.raises(ValueError, match="root"):
        validate_physical_reference(reference)
    for field in ("registration", "timestamp_uncertainty", "survey"):
        document = copy.deepcopy(reference)
        document[field] = unavailable()
        assert validate_physical_reference(document, root=tmp_path)["status"] == "unqualified"


@pytest.mark.parametrize("field,value", [("calibration_sha256", "bad"), ("interpolated", True), ("independent", False)])
def test_strict_fields(reference, tmp_path, field, value):
    reference[field] = value
    with pytest.raises(ValueError):
        validate_physical_reference(reference, root=tmp_path)


@pytest.mark.parametrize("field,value", [("plane_normal_left", [0,0,2]), ("axial_intercept_m", 3), ("absolute_error_m", -1), ("plane_offset_m", float("nan"))])
def test_bad_survey(reference, tmp_path, field, value):
    reference["survey"]["value"][field] = value
    with pytest.raises(ValueError):
        validate_physical_reference(reference, root=tmp_path)


def test_bad_pose_and_bounds(reference, tmp_path):
    reference["registration"]["value"]["transform_to_left"][0][0] = -1
    with pytest.raises(ValueError):
        validate_physical_reference(reference, root=tmp_path)
    reference["registration"]["value"]["transform_to_left"][0][0] = 1
    reference["timestamp_uncertainty"]["value"]["max_error_ns"] = -1
    with pytest.raises(ValueError):
        validate_physical_reference(reference, root=tmp_path)


def test_sparse_reference_not_dense_edges(reference, tmp_path):
    reference["sampling"] = "sparse"
    with pytest.raises(ValueError):
        validate_physical_reference(reference, root=tmp_path)
    reference["dense_geometry_labels"] = False
    assert validate_physical_reference(reference, root=tmp_path)["dense_geometry_labels_allowed"] is False


def test_unsafe_symlink_and_mutated_files(reference, tmp_path):
    reference["artifact"]["path"] = "../survey.bin"
    with pytest.raises(ValueError):
        validate_physical_reference(reference, root=tmp_path)
    reference["artifact"]["path"] = "link.bin"
    (tmp_path / "link.bin").symlink_to(tmp_path / "survey.bin")
    with pytest.raises(ValueError):
        validate_physical_reference(reference, root=tmp_path)
    reference["artifact"]["path"] = "survey.bin"
    with ExitStack() as stack:
        files = _Files(tmp_path, stack)
        validate_physical_reference(reference, files=files)
        (tmp_path / "survey.bin").write_bytes(b"changed")
        with pytest.raises(ValueError):
            validate_physical_reference(reference, files=files)


def test_missing_labels_and_scanner(reference, tmp_path):
    reference["labels"] = unavailable()
    assert validate_physical_reference(reference, root=tmp_path)["status"] == "unqualified"
    reference["dense_geometry_labels"] = False
    reference["kind"] = "scanner"
    reference["survey"] = unavailable()
    assert validate_physical_reference(reference, root=tmp_path)["status"] == "metadata_ready"


def test_hash_missing_file_and_cache_root(reference, tmp_path):
    reference["artifact"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="hash/size"):
        validate_physical_reference(reference, root=tmp_path)
    reference["artifact"]["path"] = "absent.bin"
    with pytest.raises(ValueError, match="cannot validate file"):
        validate_physical_reference(reference, root=tmp_path)
    with ExitStack() as stack:
        with pytest.raises(ValueError, match="root disagrees"):
            validate_physical_reference(reference, root=tmp_path / "other", files=_Files(tmp_path, stack))


def test_strict_keys_and_boolean_numbers(reference, tmp_path):
    reference["registration"]["value"]["max_error_m"] = True
    with pytest.raises(ValueError):
        validate_physical_reference(reference, root=tmp_path)
    reference["registration"]["value"]["max_error_m"] = 0
    reference["extra"] = "forbidden"
    with pytest.raises(ValueError, match="expected keys"):
        validate_physical_reference(reference, root=tmp_path)


def test_shared_cache_only_rechecks_consumed_reference_paths(reference, tmp_path, monkeypatch):
    unrelated = b'previous frame dependency'
    (tmp_path / 'unrelated.bin').write_bytes(unrelated)
    desc = dict(path='unrelated.bin', sha256=sha256(unrelated).hexdigest(), size_bytes=len(unrelated))
    with ExitStack() as stack:
        files = _Files(tmp_path, stack)
        files.check(desc)
        visited = []
        original = files._unchanged
        def unchanged(path, fingerprint):
            visited.append(path)
            return original(path, fingerprint)
        monkeypatch.setattr(files, '_unchanged', unchanged)
        result = validate_physical_reference(reference, files=files)
        assert result['status'] == 'metadata_ready'
        assert 'survey.bin' in visited
        assert 'unrelated.bin' not in visited
        # The whole-run boundary still protects previously admitted dependencies.
        (tmp_path / 'unrelated.bin').write_bytes(b'mutated')
        with pytest.raises(ValueError, match='changed'):
            files.recheck()
