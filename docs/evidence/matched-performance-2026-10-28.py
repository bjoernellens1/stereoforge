#!/usr/bin/env python3
"""Reproduce synthetic CPU admission cases; never vendor/physical evidence."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmarks.zed_comparison import performance
from benchmarks.zed_comparison.native_input import _strict_json
from tests.test_zed_performance import qualified_abba, record, write_json, rewrite_source


def compact(records, result):
    return {
        'run_count': len(records),
        'sample_count': sum(r['sample_count'] for r in records),
        'qualified_admission_count': sum(a['qualified'] for a in result['admissions']),
        'comparisons': result['comparisons'],
        'populations': [dict(profile_id=p['profile_id'], run_count=len(p['run_ids']),
                             statistics=p['statistics']) for p in result['populations']],
    }


def case(name):
    with TemporaryDirectory(prefix='stereoforge-timing-evidence-') as directory:
        root = Path(directory)
        records = qualified_abba(root)
        if name == 'missing_quality':
            for r in records:
                r['sources']['quality'] = dict(status='unavailable', reason='not supplied')
        elif name == 'incomplete_abba':
            records = records[:-1]
        elif name == 'cross_platform':
            b_profile = records[1]['profile_id']
            for r in records:
                if r['profile_id'] == b_profile:
                    r['platform']['host_id'] = 'other-host'
            platforms = [deepcopy(records[0]['platform']), deepcopy(records[1]['platform'])]
            contention = write_json(root, 'contention.json', dict(
                schema='stereoforge.zed_comparison.contention.v1', status='clear',
                platforms=platforms, method='declared synthetic CPU fixture'))
            def update_protocol(protocol):
                for key in ('contention_before', 'contention_after'):
                    protocol[key] = dict(status='available', value=contention)
            rewrite_source(root, records, 'protocol', update_protocol)
        result = performance.summarize_timing(records, root=root)
        comparison = result['comparisons'][0]
        if name == 'synthetic_same_device':
            assert comparison['qualified'] and comparison['ratio'] == 1
        else:
            assert not comparison['qualified'] and comparison['ratio'] is None
        if name == 'cross_platform':
            assert comparison['kind'] == 'operating_points'
            assert comparison['reasons'] == ['cross_platform_operating_points']
        return compact(records, result)


def legacy(path):
    path = path.resolve()
    report = _strict_json(path)
    fixture = record()
    context = {k: v for k, v in fixture.items()
               if k not in {'schema', 'profile_id', 'sample_count', 'samples', 'stages'}}
    result = dict(status='unqualified', report_sha256=sha256(path.read_bytes()).hexdigest(),
                  report_path=path.relative_to(REPO).as_posix(),
                  raw_sample_count=report.get('frames'),
                  source='retained legacy report; not a new measurement')
    try:
        performance.adapt_native_timing(report, context)
    except ValueError as error:
        result['reason'] = str(error)
    else:
        result['reason'] = 'adaptation alone does not qualify retained evidence'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-report', type=Path)
    args = parser.parse_args()
    paths = [Path(__file__).relative_to(REPO),
             Path('tests/test_zed_performance.py'),
             Path('benchmarks/zed_comparison/performance.py'),
             Path('benchmarks/zed_comparison/schema.py'),
             Path('benchmarks/zed_comparison/native_input.py'),
             Path('benchmarks/zed_comparison/native_observation.py'),
             Path('benchmarks/zed_comparison/evaluate_temporal.py')]
    output = dict(schema='stereoforge.matched_performance_cpu_evidence.v1',
                  synthetic_only=True, new_native_or_sdk_measurements=False,
                  physical_qualified=False, quality_parity_verified=False,
                  qualification_scope='synthetic admission contract only; fixture quality is not vendor evidence',
                  source_sha256={p.as_posix(): sha256((REPO / p).read_bytes()).hexdigest() for p in paths},
                  cases={name: case(name) for name in ('synthetic_same_device', 'missing_quality',
                                                      'incomplete_abba', 'cross_platform')})
    if args.legacy_report:
        output['legacy_report'] = legacy(args.legacy_report)
    Path(__file__).with_suffix('.json').write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
