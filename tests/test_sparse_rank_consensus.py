import numpy as np
import pytest

from benchmarks import sparse_rank_consensus as rank
from benchmarks.sparse_stereo_reference import Match, Result


def test_rank_footprint_and_contrast():
    image = np.tile(np.arange(11, dtype=np.uint8)[:, None] * 10, (1, 3))
    support = np.ones(image.shape, bool)
    descriptor, usable, pixels, samples = rank.rank_descriptor(image, support)
    assert pixels == 9 and samples == 72
    assert usable[4, 1] and descriptor[4, 1] == 240
    support[0, 1] = False
    _, usable, pixels, samples = rank.rank_descriptor(image, support)
    assert not usable[4, 1] and pixels == 9 and samples == 72
    _, usable, _, _ = rank.rank_descriptor(np.ones_like(image), np.ones_like(support))
    assert not usable.any()


def test_search_ties_and_candidate_counts():
    ref = np.array([[0, 0, 0, 0]], np.uint8)
    other = np.array([[1, 3, 0, 7]], np.uint8)
    valid = np.ones_like(ref, bool)
    assert rank._search(ref, other, valid, 0, 3, -1, 4, 1) == (1, 4, 'ok')
    other[0, 1] = 0
    assert rank._search(ref, other, valid, 0, 3, -1, 4, 1) == (None, 4, 'ambiguous')
    valid[0, :3] = False
    assert rank._search(ref, other, valid, 0, 3, -1, 4, 1) == (None, 0, 'too_few_candidates')


def test_consensus_retains_baseline_identity_and_rejects_failed_cue(monkeypatch):
    shape = (9, 5)
    img = np.tile(np.arange(9, dtype=np.uint8)[:, None] * 10, (1, 5))
    mask = np.ones(shape, bool)
    matches = (Match('left', 4, 3, 2, 1, 0, 1, True, 'ok'),
               Match('right', 4, 2, None, None, None, None, False, 'ambiguous'))
    baseline = Result('frame', matches, 11, 7)
    searches = iter([(1, 3, 'ok'), (1, 4, 'ok')])
    monkeypatch.setattr(rank, '_search', lambda *args: next(searches))
    result = rank.sparse_stereo(img, img, mask, mask, source_id='frame', baseline=baseline)
    assert result.matches == matches
    assert (result.forward_candidates, result.reverse_candidates) == (11, 7)
    assert (result.rank_forward_candidates, result.rank_reverse_candidates, result.rank_attempted_points) == (3, 4, 1)
    searches = iter([(2, 3, 'ok'), (2, 4, 'ok')])
    result = rank.sparse_stereo(img, img, mask, mask, source_id='frame', baseline=baseline)
    assert result.matches[0].reason == 'rank_disagreement' and not result.matches[0].valid
    assert result.matches[1] == matches[1]
    searches = iter([(1, 3, 'ok'), (2, 4, 'ok')])
    result = rank.sparse_stereo(img, img, mask, mask, source_id='frame', baseline=baseline)
    assert result.matches[0].reason == 'rank_reverse_mismatch'
    with pytest.raises(ValueError):
        rank.sparse_stereo(img, img, mask, mask, source_id='another', baseline=baseline)


def test_low_contrast_and_support_fail_closed():
    shape = (9, 8)
    img = np.ones(shape, np.uint8)
    mask = np.ones(shape, bool)
    baseline = Result('frame', (Match('left', 4, 5, 4, 1, 0, 1, True, 'ok'),), 1, 1)
    out = rank.sparse_stereo(img, img, mask, mask, source_id='frame', baseline=baseline)
    assert out.matches[0].reason == 'rank_low_contrast'
    mask[0, 5] = False
    out = rank.sparse_stereo(img, img, mask, mask, source_id='frame', baseline=baseline)
    assert out.matches[0].reason == 'rank_unsupported'


def test_search_supports_near_disparity():
    ref = np.zeros((1, 230), np.uint8)
    other = np.full((1, 230), 255, np.uint8)
    other[0, 40] = 0
    valid = np.ones(ref.shape, bool)
    assert rank._search(ref, other, valid, 0, 200, -1, 224, 1) == (160, 201, 'ok')


def test_reverse_ambiguity_and_margin_are_fail_closed(monkeypatch):
    img = np.tile(np.arange(9, dtype=np.uint8)[:, None] * 10, (1, 5))
    mask = np.ones(img.shape, bool)
    baseline = Result('frame', (Match('right', 4, 1, 2, 1, 0, 1, True, 'ok'),), 2, 2)
    searches = iter([(1, 3, 'ok'), (None, 3, 'ambiguous')])
    monkeypatch.setattr(rank, '_search', lambda *args: next(searches))
    result = rank.sparse_stereo(img, img, mask, mask, source_id='frame', baseline=baseline)
    assert result.matches[0].reason == 'rank_reverse_ambiguous'
    assert result.rank_reverse_candidates == 3
    with pytest.raises(ValueError):
        rank.sparse_stereo(img, img, mask, mask, source_id='frame', baseline=baseline, min_margin=0)


def test_top_level_evaluator_import_uses_same_result_class():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = """
import numpy as np
import sparse_stereo_reference as single
import sparse_rank_consensus as rank
im = np.zeros((9, 4), np.uint8)
support = np.ones_like(im, bool)
baseline = single.Result('same', (), 0, 0)
result = rank.sparse_stereo(im, im, support, support, source_id='same', baseline=baseline)
assert result.source_id == 'same' and result.matches == ()
"""
    subprocess.run([sys.executable, '-c', script], cwd=root / 'benchmarks', check=True)
