"""Offline export admission never promotes observations to qualified results."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from benchmarks.zed_comparison import schema, svo_to_frames
from test_zed_comparison_schema import record, reseal, unavailable  # noqa: F401


def export(root, records):
    descriptors = []
    for number, item in enumerate(records):
        name = f"frame{number}.json"
        data = json.dumps(item).encode()
        (root / name).write_bytes(data)
        descriptors.append(dict(path=name, sha256=sha256(data).hexdigest(), size_bytes=len(data)))
    manifest = dict(schema=svo_to_frames.EXPORT_SCHEMA, records=descriptors)
    (root / "export.json").write_text(json.dumps(manifest))
    return manifest


def test_methods_grouped_and_unavailable_preserved(record, tmp_path):
    other = deepcopy(record)
    other["implementation"]["name"] = "sdk-unavailable"
    other["observation"] = unavailable("SDK observation unavailable")
    export(tmp_path, [other, record])
    path = svo_to_frames.index_export(tmp_path)
    data = path.read_bytes()
    index = json.loads(data)
    assert set(index) == {"schema", "export", "records", "groups", "status", "quality_status"}
    assert index["groups"] == [dict(input_identity=record["input_identity"], records=["frame0.json", "frame1.json"])]
    assert index["status"] == "indexed" and index["quality_status"] == "unqualified"
    assert json.loads((tmp_path / "frame0.json").read_bytes())["observation"] == other["observation"]
    assert index["export"]["sha256"] == sha256((tmp_path / "export.json").read_bytes()).hexdigest()
    assert svo_to_frames.index_export(tmp_path) == path
    assert path.read_bytes() == data and data.endswith(b"\n")


def test_calibration_content_separates_groups(record, tmp_path):
    other = deepcopy(record)
    data = (tmp_path / "raw.json").read_bytes() + b" "
    (tmp_path / "raw2.json").write_bytes(data)
    other["calibration"]["raw"] = dict(path="raw2.json", sha256=sha256(data).hexdigest(), size_bytes=len(data))
    reseal(other)
    export(tmp_path, [other, record])
    groups = json.loads(svo_to_frames.index_export(tmp_path).read_bytes())["groups"]
    assert [group["input_identity"] for group in groups] == sorted([record["input_identity"], other["input_identity"]])


@pytest.mark.parametrize("case", ["hash", "symlink", "traversal", "duplicate_path", "empty", "leak", "reused_frame", "missing", "manifest_symlink", "manifest_directory"])
def test_invalid_exports_leave_no_index(record, tmp_path, case):
    other = deepcopy(record)
    other["implementation"]["name"] = "other"
    if case == "leak":
        other["partition"] = "test"
    if case == "reused_frame":
        other["ordinal"] = 1
        reseal(other)
    manifest = export(tmp_path, [record, other])
    path = tmp_path / "export.json"
    if case == "hash":
        manifest["records"][0]["sha256"] = "0" * 64
    elif case == "symlink":
        (tmp_path / "frame0.json").rename(tmp_path / "real.json")
        (tmp_path / "frame0.json").symlink_to(tmp_path / "real.json")
    elif case == "traversal":
        manifest["records"][0]["path"] = "../frame0.json"
    elif case == "duplicate_path":
        manifest["records"][1] = manifest["records"][0]
    elif case == "empty":
        manifest["records"] = []
    path.write_text(json.dumps(manifest))
    if case == "missing":
        path.unlink()
    elif case == "manifest_symlink":
        path.rename(tmp_path / "real-export.json")
        path.symlink_to(tmp_path / "real-export.json")
    elif case == "manifest_directory":
        path.unlink()
        path.mkdir()
    with pytest.raises(ValueError):
        svo_to_frames.index_export(tmp_path)
    assert not (tmp_path / "index.json").exists()
    assert not list(tmp_path.glob(".index-*.tmp"))


@pytest.mark.parametrize("body", ['{"schema":1,"schema":2}', '{"schema":NaN}', '{"schema":1e999}'])
def test_strict_manifest(tmp_path, body):
    (tmp_path / "export.json").write_text(body)
    with pytest.raises(ValueError):
        svo_to_frames.index_export(tmp_path)


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_different_index_rejected(record, tmp_path, symlink):
    export(tmp_path, [record])
    path = tmp_path / "index.json"
    if symlink:
        (tmp_path / "other").write_bytes(b"existing")
        path.symlink_to(tmp_path / "other")
    else:
        path.write_bytes(b"existing")
    with pytest.raises(ValueError):
        svo_to_frames.index_export(tmp_path)
    assert path.read_bytes() == b"existing"


def test_manifest_mutation_during_loader_rejected(record, tmp_path, monkeypatch):
    export(tmp_path, [record])
    original = schema.load_records
    def load(*args, **kwargs):
        records = original(*args, **kwargs)
        path = tmp_path / "export.json"
        path.write_bytes(path.read_bytes() + b" ")
        return records
    monkeypatch.setattr(schema, "load_records", load)
    with pytest.raises(ValueError, match="manifest changed"):
        svo_to_frames.index_export(tmp_path)
    assert not (tmp_path / "index.json").exists()


@pytest.mark.parametrize("name", ["export.json", "index.json", "nested/frame.json", "frame.bin", "frame\\x.json"])
def test_record_names_are_flat_json_only(record, tmp_path, name):
    manifest = export(tmp_path, [record])
    manifest["records"][0]["path"] = name
    (tmp_path / "export.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        svo_to_frames.index_export(tmp_path)
    assert not (tmp_path / "index.json").exists()


def test_failed_atomic_publish_cleans_temporary(record, tmp_path, monkeypatch):
    export(tmp_path, [record])
    def fail(*args, **kwargs):
        raise OSError("injected publish failure")
    monkeypatch.setattr(svo_to_frames.os, "link", fail)
    with pytest.raises(OSError, match="injected"):
        svo_to_frames.index_export(tmp_path)
    assert not (tmp_path / "index.json").exists()
    assert not list(tmp_path.glob(".index-*.tmp"))
