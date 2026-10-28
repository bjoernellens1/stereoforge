"""Batch validation deduplicates I/O without retaining mappings or trusting stale files."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from benchmarks.zed_comparison import schema
from test_zed_comparison_schema import record, reseal  # noqa: F401
from test_zed_schema_io_guard import IOProbe, source_fixture, CHUNK


def batch(record, root):
    other = deepcopy(record)
    other.update(ordinal=1, source_id=1)
    reseal(other)
    paths = [root / 'frame0.json', root / 'frame1.json']
    for path, item in zip(paths, [record, other]):
        path.write_text(json.dumps(item))
    return paths


def test_shared_source_hashed_once_and_independent_calls_rehash(record, tmp_path, monkeypatch):
    paths = batch(source_fixture(record, tmp_path), tmp_path)
    probe = IOProbe(monkeypatch, tmp_path)
    assert len(schema.load_records(paths)) == 2
    assert probe.opens['source'] == 1
    assert sum(actual for name, _, actual in probe.reads if name == 'source') == 32 * CHUNK
    assert all(0 < requested <= CHUNK for name, requested, _ in probe.reads if name == 'source')
    assert sum(name == 'original_left' for name, _, _ in probe.reads) == 2
    # One streaming hash (including EOF) and one bounded JSON parse per frame.
    assert sum(name == 'raw.json' for name, _, _ in probe.reads) == 4
    schema.load_records(paths)
    assert probe.opens['source'] == 2


def test_mappings_closed_before_next_record(record, tmp_path, monkeypatch):
    paths = batch(record, tmp_path)
    real = schema.np.memmap
    mappings = []
    original_validate = schema._validate
    count = 0
    def mapped(*args, **kwargs):
        result = real(*args, **kwargs)
        mappings.append(result._mmap)
        return result
    def validate(item, files=None):
        nonlocal count
        if files:
            if count:
                assert mappings and all(m.closed for m in mappings)
            count += 1
        return original_validate(item, files)
    monkeypatch.setattr(schema.np, 'memmap', mapped)
    monkeypatch.setattr(schema, '_validate', validate)
    schema.load_records(paths)
    assert count == 2 and all(m.closed for m in mappings)


@pytest.mark.parametrize('mutation', ['restored_mtime', 'replacement', 'symlink'])
def test_changed_cached_source_rejected(record, tmp_path, monkeypatch, mutation):
    paths = batch(record, tmp_path)
    original = schema._validate
    count = 0
    def validate(item, files=None):
        nonlocal count
        if files:
            count += 1
            if count == 2:
                source = tmp_path / 'source'
                before = source.stat()
                data = source.read_bytes()
                if mutation == 'restored_mtime':
                    source.write_bytes(b'x' * len(data))
                    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
                elif mutation == 'replacement':
                    replacement = tmp_path / 'replacement'
                    replacement.write_bytes(data)
                    replacement.replace(source)
                else:
                    replacement = tmp_path / 'replacement'
                    replacement.write_bytes(data)
                    source.unlink()
                    source.symlink_to(replacement)
        return original(item, files)
    monkeypatch.setattr(schema, '_validate', validate)
    with pytest.raises(ValueError):
        schema.load_records(paths)


@pytest.mark.parametrize('target', ['source', 'frame0.json', 'raw.json'])
def test_final_recheck_catches_change_after_last_use(record, tmp_path, monkeypatch, target):
    paths = batch(record, tmp_path)
    original = schema.validate_partitions
    def partitions(records):
        original(records)
        path = tmp_path / target
        path.write_bytes(path.read_bytes() + b' ')
    monkeypatch.setattr(schema, 'validate_partitions', partitions)
    with pytest.raises(ValueError):
        schema.load_records(paths)


@pytest.mark.parametrize('invalid', ['empty', 'duplicate', 'mixed_root', 'symlink', 'traversal'])
def test_record_paths_rejected(record, tmp_path, invalid):
    paths = batch(record, tmp_path)
    if invalid == 'empty':
        paths = []
    elif invalid == 'duplicate':
        paths = [paths[0], paths[0]]
    elif invalid == 'mixed_root':
        child = tmp_path / 'child'
        child.mkdir()
        paths[1] = child / 'frame.json'
        paths[1].write_text(json.dumps(record))
    elif invalid == 'symlink':
        link = tmp_path / 'link.json'
        link.symlink_to(paths[0])
        paths[0] = link
    else:
        paths[0] = tmp_path / '..' / tmp_path.name / 'frame0.json'
    with pytest.raises(ValueError):
        schema.load_records(paths)


@pytest.mark.parametrize('invalid', ['partition', 'duplicate', 'strict_json', 'corrupt_plane'])
def test_full_record_and_partition_validation(record, tmp_path, invalid):
    paths = batch(record, tmp_path)
    if invalid == 'partition':
        other = json.loads(paths[1].read_text())
        other['partition'] = 'held_out'
        paths[1].write_text(json.dumps(other))
    elif invalid == 'duplicate':
        paths[1].write_bytes(paths[0].read_bytes())
    elif invalid == 'strict_json':
        paths[1].write_text('{"schema":1,"schema":2}')
    else:
        (tmp_path / 'depth').write_bytes(b'x')
    with pytest.raises(ValueError):
        schema.load_records(paths)


@pytest.mark.parametrize('mutation', ['none', 'hash', 'size', 'path', 'length'])
def test_expected_record_descriptors_bind_parsed_json(record, tmp_path, mutation):
    from hashlib import sha256
    paths = batch(record, tmp_path)
    expected = [dict(path=p.name, sha256=sha256(p.read_bytes()).hexdigest(),
                     size_bytes=p.stat().st_size) for p in paths]
    if mutation == 'hash':
        expected[0]['sha256'] = '0' * 64
    elif mutation == 'size':
        expected[0]['size_bytes'] += 1
    elif mutation == 'path':
        expected[0]['path'] = paths[1].name
    elif mutation == 'length':
        expected.pop()
    if mutation == 'none':
        assert len(schema.load_records(paths, expected=expected)) == 2
    else:
        with pytest.raises(ValueError):
            schema.load_records(paths, expected=expected)


@pytest.mark.parametrize('target', ['frame0.json', 'raw.json'])
def test_json_changed_between_hash_and_parse_rejected(record, tmp_path, monkeypatch, target):
    paths = batch(record, tmp_path)
    original = schema._read_json
    def read_json(path):
        if path.name == target:
            # Still valid JSON, but bytes differ from the version just hashed.
            path.write_bytes(path.read_bytes() + b' ')
        return original(path)
    monkeypatch.setattr(schema, '_read_json', read_json)
    with pytest.raises(ValueError, match='changed during validation'):
        schema.load_records(paths)


def test_mutation_during_new_hash_rejected(record, tmp_path, monkeypatch):
    paths = batch(record, tmp_path)
    original_open = Path.open
    class Stream:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
    def opened(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.name == 'source' and args == ('rb',):
            # Mutate only once: appending on each read would never reach EOF.
            class Once(Stream):
                done = False
                def read(self, size):
                    data = self.stream.read(size)
                    if not self.done:
                        self.done = True
                        with original_open(tmp_path / 'source', 'ab') as writer:
                            writer.write(b'x')
                    return data
            return Once(stream)
        return stream
    monkeypatch.setattr(Path, 'open', opened)
    with pytest.raises(ValueError, match='changed during validation'):
        schema.load_records(paths)
