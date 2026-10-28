import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import evaluate_sparse_recovery as recovery
from sparse_stereo_reference import Match, Result


def test_policy_and_all_profile_decisions_exist_before_truth(tmp_path, monkeypatch):
    for name in ('coarse', 'target', 'family'):
        (tmp_path / name).write_text(name)
    out = tmp_path / 'out'
    image = np.arange(24, dtype=np.uint8).reshape(2, 2, 6)
    context = dict(images=image, support=np.ones_like(image, bool), source_id=7)
    monkeypatch.setattr(recovery.fixture, 'load_context', lambda *args: context)
    called = []
    def match(*args, **kwargs):
        assert json.loads((out / 'policy.json').read_text()) == recovery.POLICY
        assert kwargs['disparities'] == 224
        i = len(called)
        called.append(kwargs['selection'])
        matches = (Match('left', 0, i, i, i, 0, 2, True, 'ok'),
                   Match('right', 1, i, None, None, None, None, False, 'ambiguous'))
        return Result('7', matches, 20 + i, 10 + i)
    monkeypatch.setattr(recovery.matcher, 'sparse_stereo', match)
    def truth(*args):
        doc = json.loads((out / 'decisions.json').read_text())
        assert list(doc['runs']) == sorted(recovery.POLICY['selectors'])
        assert all(len(doc['runs'][k]['matches']) == 2 for k in doc['runs'])
        return 'truth'
    monkeypatch.setattr(recovery.fixture, 'load_truth', truth)
    class Features:
        support = np.ones((2, 6), bool)
    monkeypatch.setattr(recovery, 'census', lambda *args: Features())
    def evaluate(decisions, truth_data, support, count):
        assert truth_data == 'truth' and count == 224
        return {'left': {'marker': decisions[0]['disparity']}, 'right': {}}
    monkeypatch.setattr(recovery.base, 'evaluate', evaluate)
    report = recovery.run(tmp_path / 'coarse', tmp_path / 'target', tmp_path / 'family', out)
    assert called == recovery.POLICY['selectors']
    for i, name in enumerate(called):
        p = report['profiles'][name]
        assert p['eyes']['left']['marker'] == i
        assert p['selected_by_eye'] == {'left': 1, 'right': 1}
        assert p['accepted_by_eye'] == {'left': 1, 'right': 0}
        assert p['forward_candidates'] == 20 + i
    assert report['decision_sha256'] == recovery.fixture.sha256(out / 'decisions.json')
    assert report['policy_sha256'] == recovery.fixture.sha256(out / 'policy.json')
    with pytest.raises(FileExistsError):
        recovery.run(tmp_path / 'coarse', tmp_path / 'target', tmp_path / 'family', out)


def test_policy_must_match_scoring_and_fixed_subdivision(monkeypatch):
    recovery.validate_policy()
    monkeypatch.setitem(recovery.base.POLICY, 'correct_error_px', 3.0)
    with pytest.raises(ValueError, match='scoring policy'):
        recovery.validate_policy()
    monkeypatch.setitem(recovery.base.POLICY, 'correct_error_px', recovery.POLICY['correct_error_px'])
    monkeypatch.setitem(recovery.POLICY, 'edge_subdivisions', 2)
    with pytest.raises(ValueError, match='subdivision'):
        recovery.validate_policy()


def test_scoring_policy_rechecked_after_matching(tmp_path, monkeypatch):
    for name in ('coarse', 'target', 'family'):
        (tmp_path / name).write_text(name)
    out = tmp_path / 'out'
    image = np.zeros((2, 2, 6), dtype=np.uint8)
    monkeypatch.setattr(recovery.fixture, 'load_context', lambda *args: dict(
        images=image, support=np.ones_like(image, bool), source_id=7))
    def match(*args, **kwargs):
        monkeypatch.setitem(recovery.base.POLICY, 'correct_error_px', 42)
        return Result('7', (), 0, 0)
    monkeypatch.setattr(recovery.matcher, 'sparse_stereo', match)
    def load_truth(*args):
        pytest.fail('truth must not be loaded after scoring policy drift')
    monkeypatch.setattr(recovery.fixture, 'load_truth', load_truth)
    with pytest.raises(ValueError, match='scoring policy'):
        recovery.run(tmp_path / 'coarse', tmp_path / 'target', tmp_path / 'family', out)
    assert (out / 'decisions.json').is_file()
    assert not (out / 'report.json').exists()
