"""Cold, hash-bound conversion of executed native two-eye metric dumps.

These checks bind declarations and detect observed changes. They do not certify
physical accuracy, establish an atomic filesystem snapshot, or run a backend.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import shutil

import numpy as np

from . import schema
from .native_input import (_canonical, _descriptors, _path, _require,
                           _snapshot, _strict_json, SAMPLING_PROFILE)

REPORT_SCHEMA = 'stereoforge.daxa_depth_rectified_benchmark.v1'
OUTPUT_PREFIX = 'native-observation'
MAX_DUMP_BYTES = 512 * 1024 * 1024
DUMPS = {
    'depth': ('depth.f32', '<f4', 'f32_le_m'),
    'validity': ('metric_validity.u32', '<u4', 'u8_mask'),
    'confidence': ('metric_confidence.f32', '<f4', 'f32_le'),
    'provenance': ('provenance.u32', '<u4', 'u8_provenance'),
    'source_ids': ('source_ids.i64', '<i8', 'i64_le'),
    'age_ns': ('age_ns.i64', '<i8', 'i64_le'),
}


def _pixel_identity(record, names):
    return sha256(_canonical({name: {key: value for key, value in record['images'][name].items()
                                    if key != 'path'} for name in names})).hexdigest()


def _dump_path(root, name):
    schema._text(name, 'native dump path')
    _require('\\' not in name and '\0' not in name, 'unsafe native dump path')
    parts = name.split('/')
    if name.startswith('/'):
        parts = parts[1:]
    _require(all(part and part not in ('.', '..') for part in parts), 'unsafe native dump component')
    candidate = Path(name)
    path = _path(candidate if candidate.is_absolute() else root/candidate)
    _require(path.is_relative_to(root), 'native dump outside explicit dump_root')
    return path


def _settings(report):
    _require(report.get('schema') == REPORT_SCHEMA, 'unsupported native report schema')
    variant = report.get('variant')
    _require(variant in ('full_sgm', 'coarse_refine'), 'unsupported native variant')
    for key, low, high in [('frames', 1, 300), ('warmup', 0, 30), ('slots', 2, 2),
                            ('radius', 2, 8), ('sgm_directions', 4, 4)]:
        schema._integer(report.get(key), key, low, high)
    _require(report['radius'] in (2, 4, 8), 'unsupported native radius')
    for key in ('subpixel', 'sgm_subpixel'):
        _require(type(report.get(key)) is bool, 'executed boolean setting required: ' + key)
    _require(report.get('sgm_disparities') == (128 if variant == 'full_sgm' else 64) and
             type(report.get('sgm_disparities')) is int, 'unsupported executed SGM disparities')
    _require(report.get('right_reference_mode') == 'independent_sgm' and
             report.get('lr_dependency') == 'independent_references' and
             report.get('sampling_profile') == SAMPLING_PROFILE and
             report.get('scope') == 'static_repeated_canonical_record; no_live_quality_or_pipeline_FPS_claim',
             'unsupported native profile/dependency/scope')
    for key in ('validation_errors_after_teardown', 'validation_warnings_after_teardown'):
        schema._integer(report.get(key), key, 0, 0)
    samples = report.get('raw_samples_ms')
    _require(type(samples) is dict and type(samples.get('host_wall_ms')) is list,
             'executed frame samples required')
    for values in samples.values():
        if values is None:
            continue
        _require(type(values) is list and len(values) == report['frames'], 'native sample frame count mismatch')
        for value in values:
            schema._number(value, 'native sample', minimum=0)
    fields = ['variant', 'sampling_profile', 'slots', 'sgm_disparities', 'sgm_directions',
              'right_reference_mode', 'lr_dependency', 'subpixel', 'sgm_subpixel', 'final_estimator']
    if variant == 'full_sgm':
        _require(report.get('final_estimator') == 'independent_sgm' and
                 report['subpixel'] == report['sgm_subpixel'], 'inconsistent full SGM profile')
    else:
        _require(report.get('final_estimator') == 'independent-local-census' and
                 report.get('final_right_reference_mode') == 'independent-local-census',
                 'unsupported refinement estimator')
        schema._integer(report.get('refinement_disparities'), 'refinement_disparities', 128, 128)
        _require(type(report.get('refinement_subpixel')) is bool and
                 report['refinement_subpixel'] == report['subpixel'], 'missing/inconsistent refinement subpixel')
        _require(report.get('recovery') is False, 'fresh refinement requires recovery disabled')
        for key in ('prior_confidence_floor', 'max_coarse_span', 'lr_tolerance'):
            schema._number(report.get(key), key, minimum=0)
        schema._integer(report.get('min_margin'), 'min_margin', 0)
        fields += ['radius', 'refinement_subpixel', 'refinement_disparities', 'prior_confidence_floor',
                   'max_coarse_span', 'min_margin', 'lr_tolerance', 'recovery', 'final_right_reference_mode']
    settings = {key: deepcopy(report[key]) for key in fields}
    settings['search_bounds'] = dict(min_disparity=0, max_disparity_exclusive=128)
    for key in ('shader_compiler', 'refinement_shader_compiler', 'producer_shader_identity',
                'depth_shader_identity', 'refinement_shader_identity', 'producer_spirv_sha256',
                'stereo_spirv_sha256', 'depth_spirv_sha256', 'refinement_spirv_sha256', 'shader_hash_scope'):
        if key in report:
            _require(type(report[key]) in (dict, str) and bool(report[key]), 'invalid compiler/shader identity')
            settings[key] = deepcopy(report[key])
    return settings


def _bindings(report, record, record_descriptor):
    capture = record['timestamps']['capture']
    expected = dict(record_sha256=record_descriptor['sha256'], record_ordinal=record['ordinal'],
                    source_id=record['source_id'], camera_timestamp_ns=capture['value_ns'],
                    camera_clock_domain=capture['clock_domain'], input_identity=record['input_identity'],
                    original_calibration_identity=record['calibration']['raw']['sha256'],
                    rectified_calibration_identity=record['calibration']['rectified']['sha256'],
                    width=record['calibration']['geometry']['width'], height=record['calibration']['geometry']['height'],
                    original_pixel_identity=_pixel_identity(record, ('original_left', 'original_right')),
                    rectified_pixel_identity=_pixel_identity(record, ('rectified_left', 'rectified_right', 'support_left', 'support_right')))
    ingress = record['timestamps']['ingress']
    if ingress['status'] == 'available':
        ingress = ingress['value']
        expected.update(host_ingress_timestamp_ns=ingress['value_ns'], host_clock_domain=ingress['clock_domain'])
    else:
        expected.update(host_ingress_timestamp_ns=0, host_clock_domain='unknown')
    for key, value in expected.items():
        _require(type(report.get(key)) is type(value) and report[key] == value, 'native/source binding mismatch: ' + key)
    if report['variant'] == 'coarse_refine':
        _require(type(report.get('coarse_width')) is int and type(report.get('coarse_height')) is int and
                 report['coarse_width']*2 == expected['width'] and report['coarse_height']*2 == expected['height'],
                 'native coarse grid mismatch')


def export_native_observation(report_path, source_record_path, dump_root, output_dir):
    """Export one exact left observation, returning its new portable record path.

    Relative dump paths are interpreted only against explicit dump_root. Missing
    commit metadata stays unavailable. A failed call removes only its own fresh
    output directory and never publishes a successful partial record.
    """
    report_path, source_path = _path(report_path), _path(source_record_path)
    root = _path(dump_root, regular=False)
    _require(root.is_dir(), 'explicit dump_root directory required')
    output = _path(output_dir, regular=False)
    _require(output.parent.is_dir(), 'output parent directory required')
    snapshots = {path: _snapshot(path) for path in (source_path, report_path)}
    report, candidate = _strict_json(report_path), _strict_json(source_path)
    settings = _settings(report)
    record = schema.load_records([source_path])[0]
    _require(record == candidate, 'source changed during portable admission')
    _require(record['source']['kind'] == 'analytic' and
             record['calibration']['geometry']['coordinate_frame'] == 'left_optical',
             'canonical native reports require analytic left optical inputs')
    _bindings(report, record, snapshots[source_path][0])
    dependencies = {}
    for desc in _descriptors(record):
        name = desc['path']; content = {key: desc[key] for key in ('sha256', 'size_bytes')}
        _require(name not in dependencies or dependencies[name] == content, 'conflicting source descriptors')
        dependencies[name] = content
    names = [*dependencies, source_path.name]
    for name in names:
        _require(name != OUTPUT_PREFIX and not name.startswith(OUTPUT_PREFIX+'/'), 'source conflicts with native observation paths')
        _require(not any(other.startswith(name+'/') for other in names if other != name), 'source file/directory collision')
    _require(source_path.name not in dependencies, 'source record conflicts with dependency path')
    for name, expected in dependencies.items():
        path = _path(source_path.parent/name); actual = _snapshot(path)
        _require(actual[0] == expected, 'source descriptor hash/size mismatch')
        snapshots[path] = actual
    width, height = report['width'], report['height']; count = width*height
    _require(0 < count and count*64 <= MAX_DUMP_BYTES, 'native metric dump byte ceiling exceeded')
    _require(type(report.get('files')) is dict, 'native report has no dump descriptors')
    mappings = {}; left_bytes = {}
    try:
        for field, (name, dtype, _) in DUMPS.items():
            descriptor = report['files'].get(name)
            schema._keys(descriptor, {'path', 'sha256', 'elements'}, 'native dump descriptor')
            schema._integer(descriptor['elements'], 'native dump elements', 2*count, 2*count)
            schema._hex(descriptor['sha256'], 64, 'native dump hash')
            path = _dump_path(root, descriptor['path'])
            _require(path.stat().st_size == 2*count*np.dtype(dtype).itemsize, 'native dump byte extent mismatch')
            actual = _snapshot(path)
            _require(actual[0] == dict(sha256=descriptor['sha256'], size_bytes=2*count*np.dtype(dtype).itemsize),
                     'native dump hash/extent mismatch')
            _require(path not in snapshots or snapshots[path] == actual, 'conflicting native dump snapshot')
            snapshots[path] = actual
            mappings[field] = np.memmap(path, dtype=dtype, mode='r', shape=(2*count,))
            with path.open('rb') as stream:
                left_bytes[field] = stream.read(count*np.dtype(dtype).itemsize)
            _require(_snapshot(path) == actual, 'native dump changed during materialization')
        validity = mappings['validity']; provenance = mappings['provenance']
        _require(bool(np.all((validity == 0) | (validity == 1))), 'native validity must be binary U32')
        _require(bool(np.all((provenance == 0) | (provenance == 1))), 'fresh native provenance must be NONE/STEREO')
        valid = validity != 0
        _require(bool(np.all(provenance[valid] == 1)), 'valid native provenance must be STEREO')
        _require(bool(np.all(mappings['source_ids'][valid] == record['source_id'])), 'valid native source ID mismatch')
        _require(bool(np.all(mappings['age_ns'] == 0)), 'fresh native observations require zero age')
        _require(bool(np.all(np.isfinite(mappings['depth'][valid]) & (mappings['depth'][valid] > 0))), 'native valid depth must be finite positive')
        _require(bool(np.all(np.isfinite(mappings['confidence'][valid]))), 'native valid confidence must be finite')
        for field in ('validity', 'provenance'):
            left_bytes[field] = mappings[field][:count].astype(np.uint8).tobytes()
    finally:
        for mapping in mappings.values():
            mapping._mmap.close()
    unavailable = lambda reason: dict(status='unavailable', reason=reason)
    implementation = dict(name='StereoForge', version='canonical_native_v1',
                          commit=unavailable('native report did not record a StereoForge commit'),
                          device=unavailable('native report did not record a device'), settings=settings, artifacts=[])
    if 'commit' in report:
        schema._hex(report['commit'], 40, 'native StereoForge commit')
        implementation['commit'] = dict(status='available', value=report['commit'])
    if 'device' in report:
        _require(type(report['device']) is dict and bool(report['device']), 'invalid native device metadata')
        implementation['device'] = dict(status='available', value=deepcopy(report['device']))
    created = False
    try:
        output.mkdir(exist_ok=False); created = True
        for name, expected in dependencies.items():
            destination = output/name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path.parent/name, destination)
            _require(_snapshot(destination)[0] == expected, 'copied source dependency changed')
        target = output/OUTPUT_PREFIX; target.mkdir()
        for origin, name in ((source_path, 'source-record.json'), (report_path, 'native-report.json')):
            destination = target/name; shutil.copyfile(origin, destination)
            descriptor = dict(path=OUTPUT_PREFIX+'/'+name, **snapshots[origin][0])
            _require(_snapshot(destination)[0] == snapshots[origin][0], 'copied source/report changed')
            implementation['artifacts'].append(descriptor)
        observation = dict(confidence_direction='high_is_good')
        for field, (_, _, fmt) in DUMPS.items():
            data = left_bytes[field]; name = OUTPUT_PREFIX+'/'+field+'.bin'; (output/name).write_bytes(data)
            descriptor = dict(path=name, sha256=sha256(data).hexdigest(), size_bytes=len(data),
                              width=width, height=height, stride_bytes=len(data)//height, format=fmt)
            observation[field] = descriptor if field in ('depth', 'validity') else dict(status='available', value=descriptor)
        converted = deepcopy(record); converted['observation'] = dict(status='available', value=observation)
        converted['implementation'] = implementation
        _require(schema.input_identity(converted) == record['input_identity'], 'observation conversion changed input identity')
        schema.validate_record(converted, output)
        for path, expected in snapshots.items():
            _require(_snapshot(path) == expected, 'native observation input changed during export')
        record_path = output/source_path.name; record_path.write_bytes(_canonical(converted))
        _require(schema.load_records([record_path])[0] == converted, 'published portable observation changed')
        return record_path
    except BaseException:
        if created:
            shutil.rmtree(output)
        raise
