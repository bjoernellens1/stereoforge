#!/usr/bin/env python3
"""Rerun tiny CPU-only synthetic diagnostics; no acquisition or runtime claims."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'python')]

import numpy as np
from benchmarks.zed_comparison.epipolar import evaluate_epipolar, compare_epipolar_profiles
from tests.test_zed_comparison_schema import record, available
from tests.test_zed_evaluate_temporal import sequence, run, plane
from tests.test_zed_epipolar import envelope

SOURCES = (
    'benchmarks/zed_comparison/evaluate_temporal.py',
    'benchmarks/zed_comparison/compare_quality.py',
    'benchmarks/zed_comparison/truth.py',
    'benchmarks/zed_comparison/schema.py',
    'benchmarks/zed_comparison/epipolar.py',
    'tests/test_zed_comparison_schema.py',
    'tests/test_zed_truth.py',
    'tests/test_zed_evaluate_temporal.py',
    'tests/test_zed_epipolar.py',
)


def temporal_case(name, root):
    root.mkdir()
    fixture = record.__wrapped__(root)
    data = sequence(fixture, root, depths=(2, 2, 1) if name == 'known_motion_recovery' else (1, 1, 1),
                    motion='known' if name == 'known_motion_recovery' else 'stationary',
                    explicit=name in ('known_motion_recovery', 'stationary_dynamic_ghosts'))
    seq = data[3]['sequences'][0]
    if name == 'historical_provenance':
        for i, frame in enumerate(data[0]):
            frame['observation']['value']['provenance'] = available(plane(root, f'history-{i}', [3]*4, 'u8_provenance'))
    elif name == 'known_motion_recovery':
        seq['scene_motion'] = 'dynamic'
        for source, frame in zip(data[0], seq['frames']):
            frame['world_from_left'] = available(dict(matrix=np.eye(4).tolist(), capture=source['timestamps']['capture']))
        seq['recovery_events'] = [dict(event_id='reveal', kind='disocclusion',
            start_input_identity=data[0][0]['input_identity'], target_track_ids=[0],
            justification='synthetic controlled reveal')]
    elif name == 'stationary_dynamic_ghosts':
        seq['scene_motion'] = 'dynamic'
        data[1]['frames'][1]['depth'] = plane(root, 'changed-truth', [2]*4, 'f32_le_m')
    report = run(data, root)
    assert report['quality_qualified'] is False
    group = report['groups'][0]
    fields = ('fresh_observation_count', 'derived_current_source_count', 'historical_count',
              'unknown_freshness_count', 'valid_output_count', 'unique_capture_count',
              'output_error_jitter', 'fresh_error_jitter', 'raw_depth_std', 'recoveries', 'ghosting')
    return dict(status=report['status'], quality_qualified=False, reasons=report['reasons'],
                repeat_count=report['repeat_count'], **{key: group[key] for key in fields})


def generate():
    with TemporaryDirectory(prefix='stereoforge-synthetic-evidence-') as directory:
        root = Path(directory)
        cases = {name: temporal_case(name, root / name) for name in (
            'stationary', 'historical_provenance', 'known_motion_recovery', 'stationary_dynamic_ghosts')}
    profiles = []
    for name, points in (('factory', [[1,3,1,1]]), ('effective', [[1,1,1,1]])):
        metadata = envelope()
        metadata['profile'] = name
        profiles.append(evaluate_epipolar(np.asarray(points), metadata))
    epipolar = compare_epipolar_profiles(profiles)
    assert epipolar['deltas']['vertical']['signed_median_px'] == -2
    assert cases['known_motion_recovery']['recoveries'][0]['first_correct_geometry_ns'] == [200]
    assert cases['stationary_dynamic_ghosts']['ghosting']['ghost_compatible_count'] == 4
    return dict(schema='stereoforge.controlled_temporal.synthetic_evidence.v1',
                synthetic_only=True, native_or_sdk_execution=False, physical_qualified=False,
                quality_qualified=False, policy='provisional', temporal=cases, epipolar=epipolar,
                source_sha256={name: sha256((REPO / name).read_bytes()).hexdigest() for name in SOURCES})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(), indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
