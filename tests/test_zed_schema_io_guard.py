"""Offline I/O contracts on the actual validator; elapsed time is diagnostic only."""
import builtins
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time

import pytest

from benchmarks.zed_comparison import schema
from test_zed_comparison_schema import record, policy, reseal  # noqa: F401


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts/zed-task1/performance"
CHUNK = 1024 * 1024


class IOProbe:
    """Observe Python hash reads and NumPy file opens without copying plane data."""
    def __init__(self, monkeypatch, root):
        self.root = root.resolve()
        self.opens = Counter()
        self.reads = []
        path_open, builtin_open = Path.open, builtins.open
        probe = self

        class Stream:
            def __init__(self, stream, name):
                self.stream, self.name = stream, name
            def __getattr__(self, key):
                return getattr(self.stream, key)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def read(self, size=-1):
                # Enforce before read, so a whole-source regression never allocates it.
                assert size > 0, "unbounded file read"
                if not self.name.endswith(".json"):
                    assert size <= CHUNK, "artifact hash read exceeds 1 MiB"
                data = self.stream.read(size)
                probe.reads.append((self.name, size, len(data)))
                return data

        def wrap(path, stream):
            try:
                name = str(Path(path).resolve().relative_to(probe.root))
            except (TypeError, ValueError):
                return stream
            probe.opens[name] += 1
            return Stream(stream, name)

        def via_path(path, *args, **kwargs):
            return wrap(path, path_open(path, *args, **kwargs))

        def via_builtin(path, *args, **kwargs):
            return wrap(path, builtin_open(path, *args, **kwargs))

        monkeypatch.setattr(Path, "open", via_path)
        monkeypatch.setattr(builtins, "open", via_builtin)

    def summary(self):
        return dict(opens=dict(self.opens), read_calls=dict(Counter(n for n, _, _ in self.reads)),
                    read_bytes=dict(Counter({n: sum(b for k, _, b in self.reads if k == n)
                                             for n, _, _ in self.reads})),
                    max_requested_read=max((s for _, s, _ in self.reads), default=0),
                    max_requested_read_by_file={n: max(s for k, s, _ in self.reads if k == n)
                                                for n, _, _ in self.reads})


def source_fixture(record, tmp_path):
    source = tmp_path / "source"
    digest = sha256()
    block = b"s" * CHUNK
    with source.open("wb") as stream:
        for _ in range(32):
            stream.write(block)
            digest.update(block)
    record["source"].update(kind="svo2", codec="synthetic", lossless=True)
    record["source"]["artifact"].update(size_bytes=32 * CHUNK, sha256=digest.hexdigest())
    # Duplicate source and RGB references must share hashes within this validation.
    record["implementation"]["artifacts"] = [deepcopy(record["source"]["artifact"])]
    for name in ("original_right", "rectified_left", "rectified_right"):
        record["images"][name] = deepcopy(record["images"]["original_left"])
    return reseal(record)


def assert_one_hash(probe):
    source_reads = [(size, actual) for name, size, actual in probe.reads if name == "source"]
    assert probe.opens["source"] == 1, "source opened more than once within one validation"
    assert len(source_reads) == 33  # 32 chunks and EOF
    assert sum(actual for _, actual in source_reads) == 32 * CHUNK
    assert all(size <= CHUNK for size, _ in source_reads)
    assert sum(name == "original_left" for name, _, _ in probe.reads) == 2


def test_actual_validator_streaming_dedup_and_three_run_probe(record, tmp_path, monkeypatch):
    record = source_fixture(record, tmp_path)
    runs = []
    for _ in range(3):
        with monkeypatch.context() as patch:
            probe = IOProbe(patch, tmp_path)
            start = time.perf_counter()
            schema.validate_record(record, tmp_path)
            elapsed = time.perf_counter() - start
            assert_one_hash(probe)
            runs.append(dict(elapsed_seconds=elapsed, **probe.summary()))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report = dict(scope="offline actual validate_record; synthetic source; no GPU or SDK qualification",
                  schema_sha256=sha256(Path(schema.__file__).read_bytes()).hexdigest(),
                  source_sha256=record["source"]["artifact"]["sha256"], source_size_bytes=32 * CHUNK,
                  note="Each validate_record call rehashes the shared source; no cross-record cache.", runs=runs)
    (ARTIFACTS / "probe.json").write_text(json.dumps(report, indent=2) + "\n")


def test_metadata_apis_do_not_open_source_or_pixels(record, tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy()))
    # Ban every asset open: identity and partition checks must work from metadata.
    with monkeypatch.context() as patch:
        probe = IOProbe(patch, tmp_path)
        assert schema.input_identity(record) == record["input_identity"]
        schema.validate_partitions([record])
        assert schema.load_policy(path)["status"] == "unqualified"
        assert probe.opens == {"policy.json": 1}
        assert [name for name, _, _ in probe.reads] == ["policy.json"]


@pytest.mark.parametrize("mutation", ["whole_source", "duplicate_hash"])
def test_guard_detects_io_regressions(record, tmp_path, monkeypatch, mutation):
    record = source_fixture(record, tmp_path)
    if mutation == "whole_source":
        monkeypatch.setattr(schema, "HASH_CHUNK_BYTES", 32 * CHUNK)
    else:
        original = schema._Files.check
        def forget_source(self, desc, json_object=False):
            if desc["path"] == "source":
                self.hashes.pop("source", None)
            return original(self, desc, json_object=json_object)
        monkeypatch.setattr(schema._Files, "check", forget_source)
    probe = IOProbe(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match="exceeds 1 MiB|source opened more than once"):
        schema.validate_record(record, tmp_path)
        assert_one_hash(probe)
