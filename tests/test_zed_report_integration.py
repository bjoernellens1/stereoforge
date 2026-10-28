"""Synthetic artifact-backed integration: never evidence of physical qualification."""
from copy import deepcopy
from hashlib import sha256
import json
import struct

import pytest

from benchmarks.zed_comparison import schema, performance
from benchmarks.zed_comparison.compare_quality import compare_quality, _digest, _method
from benchmarks.zed_comparison.native_observation import _settings
from benchmarks.zed_comparison.report import build_report
from tests.test_zed_comparison_schema import record, available
from tests.test_zed_compare_quality import policy as quality_policy
from tests.test_zed_truth import make_truth
from tests.test_zed_native_profile import PROFILE
from tests.test_zed_performance import qualified_abba, write_json, native_report, adapt
from tests.test_zed_ros2_delivery import traces, POLICY as DELIVERY_POLICY


def descriptor(root, name, data):
    (root / name).write_bytes(data)
    return dict(path=name, sha256=sha256(data).hexdigest(), size_bytes=len(data))


def fixture(root, row):
    # Expand the schema fixture to a valid shader profile using real file contents.
    width = height = 5
    for name, desc in row['images'].items():
        rgb = not name.startswith('support')
        data = bytes([42 if rgb else 1]) * (width * height * (3 if rgb else 1))
        desc.update(descriptor(root, name, data), width=width, height=height,
                    stride_bytes=width * (3 if rgb else 1))
    for name in ('raw', 'rectified'):
        desc = row['calibration'][name]
        calibration = json.loads((root / desc['path']).read_text())
        for eye in ('left', 'right'):
            calibration[eye].update(width=width, height=height)
        desc.update(descriptor(root, desc['path'], json.dumps(calibration).encode()))
    row['calibration']['geometry'].update(width=width, height=height)
    for name, data, stride in [('depth', struct.pack('<25f', *([1.] * 25)), 20),
                               ('validity', bytes([1] * 25), 5)]:
        desc = row['observation']['value'][name]
        desc.update(descriptor(root, name, data), width=width, height=height, stride_bytes=stride)
    row['partition'] = 'held_out'
    row['input_identity'] = schema.input_identity(row)
    stratum = _digest(schema._calibration_identity(row['calibration']))
    timings = qualified_abba(root)
    compiler = json.loads((root / 'raw0.json').read_text())['shader_compiler']['compiler_sha256']
    shader = json.loads((root / 'raw0.json').read_text())['stereo_spirv_sha256']['fake.spv']
    profile = deepcopy(PROFILE)
    profile.update(width=width, height=height, compiler_sha256=compiler,
                   stereo_spirv_sha256={'daxa-stereo-census.spv': shader},
                   depth_spirv_sha256={'daxa-depth-metric.spv': shader})
    device = dict(device_uuid='1'*32, driver_uuid='2'*32, vendor_id=4098, device_id=123, driver_version=10)
    for i, timing in enumerate(timings):
        raw = json.loads((root / f'raw{i}.json').read_text())
        p = deepcopy(profile); p['subpixel'] = timing['order']['position'] in (0, 3)
        raw.update(width=width, height=height, input_identity=row['input_identity'],
                   commit='d'*40, device=device, effective_estimator=p,
                   stereo_spirv_sha256=p['stereo_spirv_sha256'], depth_spirv_sha256=p['depth_spirv_sha256'])
        impl = dict(name='StereoForge', version='canonical_native_v1', commit=available(raw['commit']),
                    device=available(device), settings=_settings(raw))
        timing.update(implementation=impl, method_key=_digest(impl), input_identities=[row['input_identity']],
                      calibration_identity=stratum)
        timing['profile_id'] = performance.profile_identity(timing)
        timing['sources']['timing'] = write_json(root, f'integration-raw{i}.json', raw)
        timings[i] = adapt(timing, raw)
    a, b = timings[:2]
    row['implementation'] = dict(a['implementation'], artifacts=[])
    other = deepcopy(row); other['implementation'] = dict(b['implementation'], artifacts=[])
    sdk = deepcopy(row); sdk['implementation']['name'] = 'ZED SDK'
    sdk['implementation']['settings']['depth_mode'] = 'NEURAL_LIGHT'
    records = [row, other, sdk]
    truth = make_truth(row, root)
    truth['frames'][0]['masks']['boundary_edges'] = available(deepcopy(truth['frames'][0]['masks']['sample']))
    for name in ('thin', 'boundary', 'near'):
        truth['frames'][0]['regions'][name] = available(deepcopy(truth['frames'][0]['masks']['sample']))
    qp = quality_policy(); qp['frozen'] = True
    digest = compare_quality(records, truth, None, root=root, truth_root=root)['partition_digest']
    qp['freeze_parameters'] = dict(correctness=dict(mode='depth_tolerance', absolute_m=.01, relative=0, combine='max'),
        region_definitions={r:dict(description='synthetic independent fixture') for r in qp['required_regions']},
        zero_reference_error=dict(method='absolute_margin', absolute_mae_margin_m=.01),
        uncertainty=dict(method='declared bounded registration', confidence_level=.95, registration_max_m=.01, timestamp_max_ns=10),
        statistics=dict(method='hierarchical_scene_sequence_bootstrap_v1', unit='scene_sequence', min_scenes=1, min_sequences=1), partition_digest=digest)
    qp['freeze_parameters']['region_definitions']['boundary'].update(absolute_jump_m=.1, relative_jump=0., tolerance_pixels=0)
    quality = compare_quality(records, truth, qp, root=root, truth_root=root)
    quality_desc = write_json(root, 'integration-quality.json', quality)
    old_freeze = json.loads((root / 'freeze.json').read_text())
    profiles = []
    for timing, old in zip((a,b),old_freeze['profiles']):
        profiles.append({**{k:timing[k] for k in ('profile_id','method_key','input_identities','calibration_identity','requested_outputs')}, 'artifacts':old['artifacts']})
    frozen = write_json(root, 'integration-freeze.json', dict(schema=old_freeze['schema'], frozen=True, profiles=profiles))
    protocol = json.loads((root / 'protocol.json').read_text()); protocol['order'] = [a['profile_id'],b['profile_id'],b['profile_id'],a['profile_id']]
    protocol_desc = write_json(root, 'integration-protocol.json', protocol)
    for timing in timings:
        timing['sources'].update(freeze=frozen, quality=available(quality_desc), protocol=available(protocol_desc))
    events = traces()
    output = dict(topic='/depth', width=width, height=height, step=width*4, encoding='32FC1', frame_id='left_optical')
    for event in events:
        if event['kind'] == 'start': event['config']['outputs']['depth'] = deepcopy(output)
        if event['kind'] == 'received': event.update(width=width,height=height,step=width*4,payload_bytes=width*height*4)
    cfg = events[0]['config']; cfg['memory']['ros_requested_bytes'] = width*height*4
    cfg['calibration_identity'] = row['calibration']['rectified']['sha256']; cfg['effective_estimator'] = deepcopy(profile)
    live = deepcopy(a['implementation']); live['version'] = 'native_ros_yuyv_v1'
    live['settings'].update(original_calibration_identity=row['calibration']['raw']['sha256'],
                            rectified_calibration_identity=row['calibration']['rectified']['sha256'])
    cfg['implementation'] = available(live)
    tracks = [dict(kind='quality',payload=dict(records=records,truth=truth,root=root,truth_root=root)),
              dict(kind='timing',payload=dict(records=timings,root=root)),
              dict(kind='delivery',payload=dict(events=events))]
    policy = dict(schema='stereoforge.zed_comparison.report_policy.v1',frozen=True,quality=qp,
                  delivery=DELIVERY_POLICY,require_temporal=False,require_view=False)
    return tracks,policy


def test_artifact_backed_real_evaluators_and_binding(record, tmp_path):
    tracks, policy = fixture(tmp_path, record)
    result = build_report(tracks, policy)
    assert result['tracks']['quality']['parity_qualified']
    assert any(c['qualified'] for c in result['tracks']['timing']['comparisons'])
    assert not result['qualified']
    assert 'processing_budget_not_verified' in result['reasons']
    assert 'camera_clock_mapping_unverified' in result['tracks']['delivery']['runs'][0]['reasons']
    assert 'quality_delivery_binding_mismatch' not in result['reasons']
    assert 'timing_delivery_binding_mismatch' not in result['reasons']
    for key in ('profile', 'raw', 'rectified'):
        mutated = deepcopy(tracks)
        config = mutated[2]['payload']['events'][0]['config']
        if key == 'profile':
            config['effective_estimator']['p2'] += 1
            config['implementation']['value']['settings']['effective_estimator']['p2'] += 1
        elif key == 'raw': config['implementation']['value']['settings']['original_calibration_identity'] = '0'*64
        else:
            config['calibration_identity'] = '0'*64
            config['implementation']['value']['settings']['rectified_calibration_identity'] = '0'*64
        assert 'quality_delivery_binding_mismatch' in build_report(mutated, policy)['reasons']
