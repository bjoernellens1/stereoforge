import json
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import evaluate_sparse_vertical as ev
from sparse_stereo_reference import Match, Result
from test_evaluate_sparse_anchors import truth


def test_thin_surface_counts_false_accepts_and_empty_denominators():
    t = truth()
    t['left']['region_thin'][0, 1] = True
    t['left']['surface_id'][0, 1] = 2
    support = np.ones((2, 1, 6), bool)
    matches = [asdict(Match('left', 0, 1, 1, 7, 0, 2, True, 'ok')),
               asdict(Match('left', 0, 2, 2, 130, 0, 2, True, 'ok'))]
    scores = ev.thin_surface_scores(matches, t, support, 224)
    row = scores['left']['surfaces']['2']
    assert (row['correct'], row['accepted'], row['false_accepted_eligible']) == (1, 2, 1)
    assert not row['passes'] and not scores['left']['passes']
    assert not scores['right']['passes']
    t['right']['region_thin'][:] = False
    assert not ev.thin_surface_scores(matches, t, support, 224)['right']['passes']
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
    def vertical(*args, **kwargs):
        x = 1 if mismatch else 2
        return SimpleNamespace(matches=(Match('left', 0, x, x, 130, 0, 2, True, 'ok'),),
                               forward_candidates=7, reverse_candidates=5,
                               forward_hamming_ops=21, reverse_hamming_ops=15)
    monkeypatch.setattr(ev.single, 'sparse_stereo', single)
    monkeypatch.setattr(ev.vertical, 'sparse_stereo', vertical)
    def load_truth(*args):
        decisions = json.loads((out / 'decisions.json').read_text())
        assert set(decisions['runs']) == set(ev.POLICY['selectors'])
        assert all(set(p) == {'single', 'vertical3'} for p in decisions['runs'].values())
        return truth()
    monkeypatch.setattr(ev.fixture, 'load_truth', load_truth)
    monkeypatch.setattr(ev, 'census', lambda *args: SimpleNamespace(support=np.ones((1, 6), bool)))
    return out


def test_policy_and_decisions_precede_truth_and_comparisons_are_counted(tmp_path, monkeypatch):
    out = setup(tmp_path, monkeypatch)
    report = ev.run(tmp_path/'coarse', tmp_path/'target', tmp_path/'family', out)
    one = report['profiles']['combined']['single']
    three = report['profiles']['combined']['vertical3']
    assert 'single skips Hamming' in report['work_scope']
    assert one['forward_hamming_upper_bound'] == 7
    assert three['forward_hamming_upper_bound'] == 21
    assert three['reverse_hamming_upper_bound'] == 15
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
