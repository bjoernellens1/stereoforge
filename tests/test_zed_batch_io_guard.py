"""Offline batch I/O budgets; timings describe cached synthetic files only."""
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time
import weakref

import pytest

from benchmarks.zed_comparison import schema, svo_to_frames
from test_zed_comparison_schema import record, policy, reseal  # noqa: F401
from test_zed_schema_io_guard import CHUNK, IOProbe, source_fixture
from test_zed_svo_to_frames import export


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts/zed-task2/performance"


def eight_frame_export(record, root):
    template = source_fixture(record, root)
    records = []
    for number in range(8):
        item = deepcopy(template)
        item.update(ordinal=number, source_id=number)
        item["timestamps"]["capture"]["value_ns"] += number
        records.append(reseal(item))
    manifest = export(root, records)
    return [root / desc["path"] for desc in manifest["records"]], records


def assert_source_once(probe):
    reads = [(requested, actual) for name, requested, actual in probe.reads if name == "source"]
    assert probe.opens["source"] == 1, "shared source must be hashed once per batch"
    assert len(reads) == 33, "expected 32 source chunks plus EOF"
    assert sum(actual for _, actual in reads) == 32 * CHUNK
    assert all(0 < requested <= CHUNK for requested, _ in reads)


def test_real_batch_and_full_index_three_run_probe(record, tmp_path, monkeypatch):
    paths, expected_records = eight_frame_export(record, tmp_path)
    assert len({r["input_identity"] for r in expected_records}) == 8
    runs = {"load_records": [], "index_export": []}
    for operation in runs:
        for _ in range(3):
            # Include first publication and fsync in every full index measurement.
            (tmp_path / "index.json").unlink(missing_ok=True)
            with monkeypatch.context() as patch:
                probe = IOProbe(patch, tmp_path)
                start = time.perf_counter()
                result = (schema.load_records(paths) if operation == "load_records"
                          else svo_to_frames.index_export(tmp_path))
                elapsed = time.perf_counter() - start
                assert_source_once(probe)
                assert all(probe.opens[p.name] >= 1 for p in paths)
                runs[operation].append(dict(elapsed_seconds=elapsed, **probe.summary()))
            if operation == "load_records":
                assert result == expected_records
            else:
                index = json.loads(result.read_bytes())
                assert len(index["records"]) == len(index["groups"]) == 8
                assert index["quality_status"] == "unqualified"
    report = dict(
        scope="offline cached synthetic export; actual loader and full index publication",
        limitation="No SDK, GPU, cold-cache, image-scale, pipeline FPS or 60 FPS qualification",
        frame_count=8, source_size_bytes=32 * CHUNK,
        source_sha256=expected_records[0]["source"]["artifact"]["sha256"],
        schema_sha256=sha256(Path(schema.__file__).read_bytes()).hexdigest(),
        indexer_sha256=sha256(Path(svo_to_frames.__file__).read_bytes()).hexdigest(),
        runs=runs)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "batch-probe.json").write_text(json.dumps(report, indent=2) + "\n")


def test_arrays_and_mappings_end_before_next_record(record, tmp_path, monkeypatch):
    paths, expected_records = eight_frame_export(record, tmp_path)
    mappings, arrays = [], []
    real_memmap, real_plane, real_validate = schema.np.memmap, schema._Files.plane, schema._validate
    validated = 0

    def memmap(*args, **kwargs):
        result = real_memmap(*args, **kwargs)
        mappings.append(result._mmap)
        return result

    def plane(*args, **kwargs):
        result = real_plane(*args, **kwargs)
        arrays.append(weakref.ref(result))
        return result

    def validate(item, files=None):
        nonlocal validated
        if files is not None:
            assert all(mapping.closed for mapping in mappings), "previous frame mapping retained"
            assert all(array() is None for array in arrays), "previous frame array retained"
            validated += 1
        return real_validate(item, files)

    monkeypatch.setattr(schema.np, "memmap", memmap)
    monkeypatch.setattr(schema._Files, "plane", plane)
    monkeypatch.setattr(schema, "_validate", validate)
    result = schema.load_records(paths)
    assert result == expected_records and validated == 8
    assert mappings and all(mapping.closed for mapping in mappings)
    assert arrays and all(array() is None for array in arrays)


def test_metadata_apis_keep_asset_reads_out(record, tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy()))
    probe = IOProbe(monkeypatch, tmp_path)
    assert schema.input_identity(record) == record["input_identity"]
    schema.validate_partitions([record])
    assert schema.load_policy(path)["status"] == "unqualified"
    assert probe.opens == {"policy.json": 1}
    assert [name for name, _, _ in probe.reads] == ["policy.json"]


def test_guard_rejects_in_memory_loss_of_batch_cache(record, tmp_path, monkeypatch):
    paths, _ = eight_frame_export(record, tmp_path)
    real_init = schema._Files.__init__

    def forget_batch_cache(self, root, stack, hashes=None, fingerprints=None):
        # Model a real regression: per-frame dictionaries replace batch sharing.
        real_init(self, root, stack)

    monkeypatch.setattr(schema._Files, "__init__", forget_batch_cache)
    probe = IOProbe(monkeypatch, tmp_path)
    assert len(schema.load_records(paths)) == 8
    assert probe.opens["source"] == 8  # the mutation actually exercised all frames
    with pytest.raises(AssertionError, match="shared source must be hashed once per batch"):
        assert_source_once(probe)


def assert_offline_ast(source):
    """Catch accidental hardware/process paths and unbounded streaming hash I/O."""
    tree = ast.parse(source)
    # Keep offline dependencies explicit: importing the runtime engine can pull
    # in hardware even without naming a GPU package in this module.
    offline_modules = {"", "contextlib", "hashlib", "json", "math", "pathlib", "re", "stat",
                       "numpy", "argparse", "os", "tempfile"}
    banned_calls = {"sleep", "wait", "synchronize", "wait_idle", "waitIdle", "device_wait_idle",
                    "vkDeviceWaitIdle", "vkQueueWaitIdle", "system", "popen", "__import__", "import_module"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            assert all(name in offline_modules for name in names), "hardware/process import requires offline review"
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert name not in banned_calls, "device/process wait or dispatch"
    # Hashing is streaming in both helpers. JSON parsing and reading an existing
    # length-checked index have separate bounds and are outside this hash budget.
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if function.name not in {"_hash", "load_records", "index_export"}:
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            assert node.func.attr not in {"read_bytes", "read_text", "readall"}, "unbounded hash read"
            if node.func.attr != "read":
                continue
            assert len(node.args) == 1 and not node.keywords, "unbounded hash read"
            bound = node.args[0]
            if isinstance(bound, ast.Name) and bound.id == "HASH_CHUNK_BYTES":
                assert 0 < schema.HASH_CHUNK_BYTES <= CHUNK, "hash read exceeds 1 MiB"
            else:
                assert all(isinstance(n, (ast.Expression, ast.Constant, ast.BinOp, ast.Mult, ast.Load))
                           for n in ast.walk(ast.Expression(bound))), "unknown hash read bound"
                def integer(node):
                    if isinstance(node, ast.Constant):
                        assert type(node.value) is int, "unknown hash read bound"
                        return node.value
                    assert isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult), "unknown hash read bound"
                    return integer(node.left) * integer(node.right)
                value = integer(bound)
                assert type(value) is int and 0 < value <= CHUNK, "hash read exceeds 1 MiB"


def test_actual_datapath_ast_stays_offline_and_hashes_bounded():
    for module in (schema, svo_to_frames):
        assert_offline_ast(Path(module.__file__).read_text())


@pytest.mark.parametrize("source, message", [
    ("import pyzed.sl as sl", "hardware/process import"),
    ("from subprocess import run", "hardware/process import"),
    ("def index_export(root):\n    device.synchronize()", "device/process wait"),
    ("def _hash(path):\n    stream.read()", "unbounded hash read"),
    ("def _hash(path):\n    path.read_bytes()", "unbounded hash read"),
    ("def index_export(root):\n    stream.read(32 * 1024 * 1024)", "exceeds 1 MiB"),
])
def test_ast_guard_rejects_hardware_and_hash_regressions(source, message):
    with pytest.raises(AssertionError, match=message):
        assert_offline_ast(source)
