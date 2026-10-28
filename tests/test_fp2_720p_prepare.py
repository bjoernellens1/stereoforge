"""CPU-only checks for the bounded 720p preparation utilities."""
import numpy as np
import pytest

from benchmarks.fp2_720p_prepare import CHUNK_BYTES, checked_offsets, exact_file, stream_equal


def test_offsets_global_band_capacity_and_invalid():
    lo = np.array([[0, 10, 60, 0, 127]], dtype=np.int32)
    hi = np.array([[0, 11, 92, 128, 128]], dtype=np.int32)
    np.testing.assert_array_equal(checked_offsets(lo, hi, (1, 5)), [0, 0, 16, 48, 176, 192])
    for a, b in [(lo, hi + 129), (lo - 1, hi), (hi, lo), (lo.astype(float), hi)]:
        with pytest.raises(ValueError):
            checked_offsets(a, b, (1, 5))
    with pytest.raises(ValueError):
        checked_offsets(lo, hi, (5, 1))


def test_stream_comparison_detects_length_and_chunk_boundary(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.write_bytes(b'abcdefghijkl'); b.write_bytes(a.read_bytes())
    assert stream_equal(a, b, 4)
    b.write_bytes(b'abcdXfghijkl')
    assert not stream_equal(a, b, 4)
    b.write_bytes(b'abc')
    assert not stream_equal(a, b, 4)
    with pytest.raises(ValueError):
        stream_equal(a, b, CHUNK_BYTES + 1)


def test_exact_file_rejects_extra_and_truncated(tmp_path):
    path = tmp_path / 'data'
    np.array([1, 2], dtype='<u4').tofile(path)
    np.testing.assert_array_equal(exact_file(path, '<u4', 2), [1, 2])
    with pytest.raises(ValueError):
        exact_file(path, '<u4', 1)
    with pytest.raises(ValueError):
        exact_file(path, '<u4', 3)


def test_scalar_image_runner_packed_dump_small_both_eyes(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    runner = tmp_path / 'runner'
    subprocess.run(['c++', '-std=c++20', '-O2', str(root/'benchmarks/fp2_banded_runner.cpp'), '-o', str(runner)], check=True)
    h, w = 3, 6
    lo = np.zeros((h, w), dtype=np.uint32)
    hi = np.full((h, w), 4, dtype=np.uint32)
    own = np.arange(h*w, dtype=np.uint32).reshape(h, w)
    other = (own * 3).astype(np.uint32)
    support = np.ones((h, w), dtype=np.uint32)
    support[1, 2] = 0
    env = os.environ.copy(); env['FP2_DUMP_PATHS'] = '1'
    for eye in (0, 1):
        case = tmp_path / str(eye); case.mkdir()
        for name, plane in [('lo', lo), ('hi', hi), ('own', own), ('other', other), ('own_support', support), ('other_support', support)]:
            np.asarray(plane, dtype='<u4').tofile(case/name)
        subprocess.run([str(runner), str(case), str(h), str(w), '128', '8', '32', str(eye)], env=env, check=True)
        offsets = checked_offsets(lo, hi, (h, w))
        np.testing.assert_array_equal(exact_file(case/'offset', '<u8', h*w+1), offsets)
        paths = exact_file(case/'paths', '<u4', 4*int(offsets[-1])).reshape(4, -1)
        sums = exact_file(case/'sums', '<u4', int(offsets[-1]))
        for i in range(h*w):
            padding = slice(int(offsets[i])+4, int(offsets[i+1]))
            assert np.all(paths[:, padding] == np.uint32(0xffffffff))
            assert np.all(sums[padding] == np.uint32(0xffffffff))
