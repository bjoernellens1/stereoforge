"""Independent adversarial timing admission regressions; no GPU execution."""
import json
import math
import pytest
from benchmarks.zed_comparison import performance as p
from tests.test_zed_performance import qualified_abba, rewrite_source, write_json, available


@pytest.mark.parametrize('status', ['unavailable', None])
def test_review_timing_requires_available_quality_frames(tmp_path, status):
    records = qualified_abba(tmp_path)
    def change(document):
        for frame in document['frames']:
            if status is None:
                frame.pop('status', None)
            else:
                frame['status'] = status
                frame['reason'] = 'missing_observation'
    rewrite_source(tmp_path, records, 'quality', change)
    result = p.summarize_timing(records, root=tmp_path)
    assert not result['comparisons'][0]['qualified'], result['comparisons'][0]
    assert 'unqualified_quality' in result['comparisons'][0]['reasons']


def test_review_finite_samples_do_not_qualify_nonfinite_ratio(tmp_path):
    records = qualified_abba(tmp_path)
    # Repair currently missing available status so this remains independent of
    # the quality-frame status admission regression above.
    rewrite_source(tmp_path, records, 'quality',
        lambda doc: [frame.update(status='available') for frame in doc['frames']])
    for record in records:
        descriptor = record['sources']['timing']
        raw = json.loads((tmp_path / descriptor['path']).read_text())
        raw['raw_samples_ms']['host_wall_ms'] = [1e308] * 3
        record['samples']['complete_wall_ms'] = available([1e308] * 3)
        record['sources']['timing'] = write_json(tmp_path, descriptor['path'], raw)
    result = p.summarize_timing(records, root=tmp_path)
    comparison = result['comparisons'][0]
    assert comparison['ratio'] is None or math.isfinite(comparison['ratio']), comparison
    for population in result['populations']:
        assert math.isfinite(population['statistics']['samples']['complete_wall_ms']['mean_ms'])
