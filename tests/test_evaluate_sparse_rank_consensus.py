import json
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import evaluate_sparse_rank_consensus as ev
from sparse_stereo_reference import Match, Result
from test_evaluate_sparse_anchors import truth


def test_thin_surface_counts_false_accepts_and_empty_denominators():
    t = truth()
    t['left']['region_thin'][0, 1] = True
    t['left']['surface_id'][0, 1] = 2
    support = np.ones((2, 1, 6), bool)
    matches = [asdict(Match('left', 0, 1, 1, 7, 0, 2, True, 'ok')),
               asdict(Match('left', 0, 2, 2, 130, 0, 2, True, 'ok'))]
    scores = ev.vertical_scores.thin_surface_scores(matches, t, support, 224)
    row = scores['left']['surfaces']['2']
    assert (row['correct'], row['accepted'], row['false_accepted_eligible']) == (1, 2, 1)
    assert not row['passes'] and not scores['left']['passes']
    assert not scores['right']['passes']
    t['right']['region_thin'][:] = False
    assert not ev.vertical_scores.thin_surface_scores(matches, t, support, 224)['right']['passes']
    assert not ev._passes(0, 0, 95, 100)
    assert ev._passes(19, 20, 95, 100)
    assert not ev._passes(18, 20, 95, 100)


def setup(tmp_path, monkeypatch, mismatch=False, drift=False):
    for name in ('coarse', 'target', 'family'):
        (tmp_path / name).write_text(name)
    out = tmp_path / 'out'
    image = np.arange(12, dtype=np.uint8).reshape(2, 1, 6)
    def context(*args):
        assert json.loads((out / 'policy.json').read_text()) == ev.POLICY
        return dict(images=image, support=np.ones_like(image, bool), source_id=7)
    monkeypatch.setattr(ev.fixture, 'load_context', context)
    def single(*args, **kwargs):
        assert kwargs['disparities'] == 224
        if drift:
            monkeypatch.setitem(ev.base.POLICY, 'correct_error_px', 42)
        return Result('7', (Match('left', 0, 2, 2, 130, 0, 2, True, 'ok'),), 7, 5)
    def rank(*args, **kwargs):
        x = 1 if mismatch else 2
        assert kwargs['baseline'].forward_candidates == 7
        return SimpleNamespace(source_id='7', matches=(Match('left', 0, x, x, 130, 0, 2, True, 'ok'),),
                               forward_candidates=7, reverse_candidates=5,
                               rank_forward_candidates=4, rank_reverse_candidates=3,
                               rank_attempted_points=1, rank_descriptor_pixels=10, rank_descriptor_samples=80)
    monkeypatch.setattr(ev.single, 'sparse_stereo', single)
    monkeypatch.setattr(ev.rank, 'sparse_stereo', rank)
    def load_truth(*args):
        decisions = json.loads((out / 'decisions.json').read_text())
        assert set(decisions['runs']) == set(ev.POLICY['selectors'])
        assert all(set(p) == {'single', 'rank'} for p in decisions['runs'].values())
        return truth()
    monkeypatch.setattr(ev.fixture, 'load_truth', load_truth)
    monkeypatch.setattr(ev, 'census', lambda *args: SimpleNamespace(support=np.ones((1, 6), bool)))
    return out


def test_policy_and_decisions_precede_truth_and_comparisons_are_counted(tmp_path, monkeypatch):
    out = setup(tmp_path, monkeypatch)
    report = ev.run(tmp_path/'coarse', tmp_path/'target', tmp_path/'family', out)
    one = report['profiles']['combined']['single']
    three = report['profiles']['combined']['rank']
    assert 'baseline once plus rank' in report['work_scope']
    assert 'rank preprocessing' in report['work_scope']
    assert one['forward_candidates'] == 7
    assert three['rank_forward_candidates'] == 4
    assert three['total_forward_candidates'] == 11
    assert three['total_reverse_candidates'] == 8
    assert three['rank_descriptor_samples'] == 80
    assert one['eyes']['left']['all']['correct'] == 1
    assert not one['eyes']['left']['paired_provisional_gates']['thin_each_eligible_surface']
    assert not one['eyes']['right']['paired_provisional_gates']['accuracy_all_accepted']
    assert report['decision_sha256'] == ev.fixture.sha256(out/'decisions.json')
    assert report['policy_sha256'] == ev.fixture.sha256(out/'policy.json')
    with pytest.raises(FileExistsError):
        ev.run(tmp_path/'coarse', tmp_path/'target', tmp_path/'family', out)


def test_different_selected_points_abort_before_truth(tmp_path, monkeypatch):
    out = setup(tmp_path, monkeypatch, mismatch=True)
    monkeypatch.setattr(ev.fixture, 'load_truth', lambda *a: pytest.fail('truth accessed'))
    with pytest.raises(ValueError, match='identical selected points'):
        ev.run(tmp_path/'coarse', tmp_path/'target', tmp_path/'family', out)
    assert not (out/'decisions.json').exists()


def test_policy_drift_aborts_before_truth(tmp_path, monkeypatch):
    out = setup(tmp_path, monkeypatch, drift=True)
    monkeypatch.setattr(ev.fixture, 'load_truth', lambda *a: pytest.fail('truth accessed'))
    with pytest.raises(ValueError, match='scoring policy'):
        ev.run(tmp_path/'coarse', tmp_path/'target', tmp_path/'family', out)
    assert (out/'decisions.json').is_file()
    assert not (out/'report.json').exists()


def test_thin_scoring_policy_drift_rejected_before_truth(tmp_path, monkeypatch):
    out = setup(tmp_path, monkeypatch)
    original = ev.rank.sparse_stereo
    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setitem(ev.vertical_scores.POLICY, 'thin_surface_min_correct', 3)
        return result
    monkeypatch.setattr(ev.rank, 'sparse_stereo', drift)
    monkeypatch.setattr(ev.fixture, 'load_truth', lambda *a: pytest.fail('truth accessed'))
    with pytest.raises(ValueError, match='thin scorer policy'):
        ev.run(tmp_path/'coarse', tmp_path/'target', tmp_path/'family', out)
    assert (out/'decisions.json').is_file()
    assert not (out/'report.json').exists()


def test_thin_false_acceptance_requires_nonempty_and_uses_all_accepts():
    assert not ev._thin_passes({'accepted_total': 0, 'correct': 0})
    assert ev._thin_passes({'accepted_total': 20, 'correct': 19})
    assert not ev._thin_passes({'accepted_total': 19, 'correct': 18})


def test_invalid_rank_work_accounting_fails_before_truth(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    original = ev.rank.sparse_stereo
    def invalid(*args, **kwargs):
        result = original(*args, **kwargs)
        result.rank_descriptor_samples = 79
        return result
    monkeypatch.setattr(ev.rank, 'sparse_stereo', invalid)
    monkeypatch.setattr(ev.fixture, 'load_truth', lambda *a: pytest.fail('truth accessed'))
    with pytest.raises(ValueError, match='invalid rank work'):
        ev.run(tmp_path/'coarse', tmp_path/'target', tmp_path/'family', tmp_path/'out')


def test_real_matcher_accepts_evaluator_baseline():
    image = np.random.default_rng(5).integers(0, 256, (20, 30), dtype=np.uint8)
    other = np.roll(image, -2, axis=1)
    support = np.ones(image.shape, bool)
    settings = dict(source_id='test', cell_size=8, disparities=12, selection='combined')
    baseline = ev.single.sparse_stereo(image, other, support, support, **settings)
    result = ev.rank.sparse_stereo(image, other, support, support, baseline=baseline, **settings)
    assert [(m.eye, m.y, m.x) for m in result.matches] == [
        (m.eye, m.y, m.x) for m in baseline.matches]
    assert result.rank_descriptor_pixels == 2 * 12 * 30
    assert result.rank_descriptor_samples == 8 * result.rank_descriptor_pixels
