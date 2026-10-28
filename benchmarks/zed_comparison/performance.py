"""Offline timing admission. Declared supplied evidence is not hardware attestation."""
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import math

from . import schema as s

TIMING_SCHEMA = 'stereoforge.zed_comparison.timing.v1'
METRICS = ('cold_prepare_ms', 'gpu_total_ms', 'gpu_upload_ms', 'gpu_readback_ms',
           'cpu_submit_ms', 'cpu_wait_ms', 'cpu_query_ms', 'cpu_control_ms',
           'complete_wall_ms', 'opaque_api_ms', 'file_io_ms', 'publication_ms')
BOUNDARIES = dict(gpu_resident='final_submit_wait_and_query', host_output='requested_cpu_output_ready',
                  opaque_api='opaque_api_return')
NATIVE = {'stereoforge.daxa_depth.v1', 'stereoforge.daxa_refinement.v1',
          'stereoforge.daxa_depth_rectified_benchmark.v1'}
HEADERS = {'schema', 'session_id', 'run_id', 'profile_id', 'method_key', 'implementation', 'platform',
           'input_identities', 'calibration_identity', 'phase', 'warmup_count', 'lane', 'completion_boundary',
           'scope', 'requested_outputs', 'isolation', 'cache', 'sources', 'order', 'sample_count', 'samples', 'stages'}


def _digest(value):
    return sha256(s._canonical(value).encode()).hexdigest()


def _unavailable(reason='not measured'):
    return dict(status='unavailable', reason=reason)


def _available(value):
    return dict(status='available', value=value)


def _value(value):
    return value['value'] if value['status'] == 'available' else None


def _bounded(value):
    try:
        s._require(len(s._canonical(value).encode()) <= s.MAX_JSON_BYTES, 'timing JSON exceeds bounded size')
    except RecursionError as exc:
        raise ValueError('timing JSON nesting exceeds supported depth') from exc


def _labels(value, where):
    s._require(type(value) is list and bool(value), where + ': expected nonempty list')
    for item in value:
        s._text(item, where)
    s._require(len(set(value)) == len(value), where + ': duplicates')


def _platform(value):
    s._keys(value, {'host_id', 'device_uuid', 'vendor', 'driver', 'os'}, 'platform')
    for key in ('host_id', 'vendor', 'driver', 'os'):
        s._text(value[key], key)
    s._availability(value['device_uuid'], lambda x: s._text(x, 'UUID'), 'device_uuid')


def profile_identity(record):
    """Canonical profile identity excludes platform and measured timing values."""
    return _digest({k: sorted(record[k]) if k == 'requested_outputs' else record[k]
                    for k in ('method_key', 'input_identities', 'calibration_identity', 'requested_outputs')})


def validate_timing_lane(record):
    """Strict structural validation; this function never qualifies evidence."""
    _bounded(record)
    s._keys(record, HEADERS, 'timing record')
    s._require(record['schema'] == TIMING_SCHEMA, 'unsupported timing schema')
    for key in ('session_id', 'run_id', 'scope'):
        s._text(record[key], key)
    impl = record['implementation']
    s._keys(impl, {'name', 'version', 'commit', 'device', 'settings'}, 'implementation')
    for key in ('name', 'version'):
        s._text(impl[key], key)
    s._availability(impl['commit'], lambda x: s._hex(x, 40, 'commit'), 'commit')
    s._availability(impl['device'], lambda x: s._require(type(x) is dict and bool(x), 'nonempty device required'), 'device')
    s._require(type(impl['settings']) is dict, 'settings must be object')
    s._hex(record['method_key'], 64, 'method_key')
    s._require(record['method_key'] == _digest(impl), 'method identity mismatch')
    _platform(record['platform'])
    inputs = record['input_identities']
    _labels(inputs, 'input_identities')
    for identity in inputs:
        s._hex(identity, 64, 'input_identity')
    s._require(inputs == sorted(inputs), 'input identities must be sorted')
    s._hex(record['calibration_identity'], 64, 'calibration_identity')
    _labels(record['requested_outputs'], 'requested_outputs')
    s._hex(record['profile_id'], 64, 'profile_id')
    s._require(record['profile_id'] == profile_identity(record), 'profile identity mismatch')
    s._require(record['phase'] in ('cold', 'warm', 'unwarmed'), 'unknown phase')
    s._integer(record['warmup_count'], 'warmup_count')
    s._require(record['phase'] != 'warm' or record['warmup_count'] > 0, 'warm requires excluded warmup')
    s._require(record['phase'] != 'cold' or record['warmup_count'] == 0, 'cold forbids warmup')
    s._require(record['phase'] != 'unwarmed' or record['warmup_count'] == 0, 'unwarmed forbids warmup')
    s._require(type(record['lane']) is str and record['lane'] in BOUNDARIES, 'unknown lane')
    s._require(record['completion_boundary'] == BOUNDARIES[record['lane']], 'completion boundary mismatch')
    s._keys(record['isolation'], {'file_io_in_span', 'diagnostic_retrieval_in_span'}, 'isolation')
    s._require(all(type(v) is bool for v in record['isolation'].values()), 'isolation requires booleans')
    s._keys(record['cache'], {'temperature', 'allocation_activity', 'host_output_properties'}, 'cache')
    for key, value in record['cache'].items():
        s._availability(value, lambda x: s._text(x, key), key)
    s._keys(record['sources'], {'timing', 'freeze', 'quality', 'protocol'}, 'sources')
    for key in ('timing', 'freeze'):
        s._descriptor(record['sources'][key])
    for key in ('quality', 'protocol'):
        s._availability(record['sources'][key], s._descriptor, key)
    s._keys(record['order'], {'block', 'position'}, 'order')
    s._integer(record['order']['block'], 'block')
    s._integer(record['order']['position'], 'position', maximum=3)
    count = record['sample_count']
    s._integer(count, 'sample_count', 1, 100000)

    def array(value):
        s._require(type(value) is list and len(value) == count, 'raw sample count mismatch')
        for sample in value:
            s._number(sample, 'sample ms', minimum=0)
    s._keys(record['samples'], METRICS, 'samples')
    s._require(type(record['stages']) is dict, 'stages must be object')
    for key, value in {**record['samples'], **record['stages']}.items():
        s._text(key, 'metric label')
        s._availability(value, array, key)
    s._require(not (set(record['stages']) & set(METRICS)), 'stage duplicates metric')
    s._require((_value(record['samples']['cold_prepare_ms']) is not None) == (record['phase'] == 'cold'),
               'cold_prepare samples require cold phase')


def adapt_native_timing(report, context):
    """Adapt actual chronological columns; cold construction remains setup metadata."""
    _bounded(report)
    s._require(report.get('schema') in NATIVE, 'unsupported native timing schema')
    metadata = report.get('timing_metadata')
    s._require(type(metadata) is dict and metadata.get('schema') == 'stereoforge.native_timing_metadata.v1',
               'missing timing_metadata')
    s._keys(metadata, {'schema','units','sample_population','sample_order','measured_count','excluded_warmup_count','phase',
        'source_reuse','completion_boundary','readback_in_span','file_io_in_span','diagnostic_retrieval_in_span',
        'legacy_summary_policy','columns','cold_prepare','cold_prepare_scope','cache_temperature','allocation_activity',
        'host_output_properties'}, 'timing_metadata')
    s._text(metadata['legacy_summary_policy'], 'legacy_summary_policy')
    s._text(metadata['cold_prepare_scope'], 'cold_prepare_scope')
    s._availability(metadata['cold_prepare'], lambda x: s._number(x, 'cold_prepare', minimum=0), 'cold_prepare')
    expected = dict(units='ms', sample_population='measured_iterations_only', sample_order='chronological',
                    source_reuse='static_retained_record', completion_boundary='final_submit_wait_and_query',
                    readback_in_span=False, file_io_in_span=False, diagnostic_retrieval_in_span=False)
    for key, value in expected.items():
        s._require(metadata.get(key) == value and type(metadata[key]) is type(value), 'invalid timing_metadata ' + key)
    count, warmup = report.get('frames'), report.get('warmup')
    s._integer(count, 'frames', 1, 100000); s._integer(warmup, 'warmup')
    phase = 'warm' if warmup else 'unwarmed'
    s._integer(metadata.get('measured_count'), 'metadata measured_count', 1, 100000)
    s._integer(metadata.get('excluded_warmup_count'), 'metadata excluded_warmup_count')
    for key, value in dict(measured_count=count, excluded_warmup_count=warmup, phase=phase).items():
        s._require(metadata.get(key) == value, 'native metadata count/phase mismatch')
    s._require(type(context) is dict, 'context must be object')
    result = dict(context)
    for key, value in dict(phase=phase, warmup_count=warmup, lane='gpu_resident',
                           completion_boundary=expected['completion_boundary'],
                           isolation=dict(file_io_in_span=False, diagnostic_retrieval_in_span=False)).items():
        s._require(key not in context or context[key] == value, 'context contradicts native ' + key)
        result[key] = value
    cache = {k: metadata.get(source) for k, source in dict(temperature='cache_temperature',
             allocation_activity='allocation_activity', host_output_properties='host_output_properties').items()}
    s._require('cache' not in context or context['cache'] == cache, 'context contradicts native cache')
    result['cache'] = cache
    raw, columns = report.get('raw_samples_ms'), metadata.get('columns')
    s._require(type(raw) is dict, 'raw columns must be object')
    raw = dict(raw)
    for group in ('gpu_paired_stereo_phases_raw_samples_ms', 'stereo_stage_raw_samples_ms'):
        if group in report:
            s._require(type(report[group]) is dict, 'raw stage group must be object')
            raw.update({group+'.'+key: value for key,value in report[group].items()})
    if 'gpu_paired_stereo_raw_samples_ms' in report:
        raw['gpu_paired_stereo_raw_samples_ms'] = report['gpu_paired_stereo_raw_samples_ms']
    s._require(type(raw) is dict and type(columns) is dict and set(raw) == set(columns), 'raw columns mismatch')
    mapping = dict(enqueue_cpu_ms='cpu_submit_ms', wait_cpu_ms='cpu_wait_ms', query_cpu_ms='cpu_query_ms',
                   timestamp_query_cpu_ms='cpu_query_ms', host_wall_ms='complete_wall_ms',
                   gpu_upload_ms='gpu_upload_ms', gpu_combined_queue_span_ms='gpu_total_ms')
    samples = {key: _unavailable() for key in METRICS}; stages = {}
    for key, values in raw.items():
        present = s._availability(columns[key], lambda x: s._text(x, key), key)
        if present:
            s._require(type(values) is list and len(values) == count, 'bad native raw dimensions')
            for value in values:
                s._number(value, key, minimum=0)
        else:
            s._require(values is None or (type(values) is list and len(values) == count), 'bad unavailable raw dimensions')
        target = samples if key in mapping else stages
        target[mapping.get(key, key)] = _available(values) if present else _unavailable(columns[key]['reason'])
    result.update(schema=TIMING_SCHEMA, sample_count=count, samples=samples, stages=stages)
    result['profile_id'] = profile_identity(result)
    validate_timing_lane(result)
    return result


def _stats(values):
    ordered = sorted(values); count = len(values)
    return dict(count=count, mean_ms=math.fsum(value/count for value in values), tail_guarantee=False,
                **{f'p{p}_ms': ordered[math.ceil(p/100*count)-1] for p in (50,95,99)}, max_ms=ordered[-1])


def _load(files, desc):
    s._descriptor(desc); files.check(desc, json_object=True)
    return files.json[desc['path']]


def _freeze(document, record, files):
    s._keys(document, {'schema', 'frozen', 'profiles'}, 'freeze')
    s._require(document['schema'] == 'stereoforge.zed_comparison.timing_freeze.v1', 'unknown freeze schema')
    s._require(type(document['frozen']) is bool and type(document['profiles']) is list, 'invalid freeze')
    seen = set(); bound = False
    keys = {'profile_id', 'method_key', 'input_identities', 'calibration_identity', 'requested_outputs', 'artifacts'}
    for profile in document['profiles']:
        s._keys(profile, keys, 'frozen profile')
        s._hex(profile['profile_id'], 64, 'frozen profile')
        s._hex(profile['method_key'], 64, 'frozen method')
        s._hex(profile['calibration_identity'], 64, 'frozen calibration')
        _labels(profile['input_identities'], 'frozen inputs')
        for identity in profile['input_identities']: s._hex(identity, 64, 'frozen input')
        s._require(profile['input_identities'] == sorted(profile['input_identities']), 'frozen input order')
        _labels(profile['requested_outputs'], 'frozen outputs')
        s._require(profile_identity(profile) == profile['profile_id'], 'frozen profile identity mismatch')
        s._require(profile['profile_id'] not in seen, 'duplicate frozen profile'); seen.add(profile['profile_id'])
        s._require(type(profile['artifacts']) is list and bool(profile['artifacts']), 'freeze requires artifacts')
        for desc in profile['artifacts']:
            s._descriptor(desc); files.check(desc)
        if profile['profile_id'] == record['profile_id']:
            for key in keys - {'artifacts'}:
                s._require(profile[key] == record[key], 'freeze binding mismatch')
            bound = True
    return document['frozen'] and bound


def _freeze_coverage(document, raw, record):
    """Match retained content, never reopen absolute paths from a native report."""
    if raw.get('schema') != 'stereoforge.daxa_depth_rectified_benchmark.v1':
        return False
    profiles = [profile for profile in document['profiles'] if profile['profile_id'] == record['profile_id']]
    if len(profiles) != 1:
        return False
    artifacts = profiles[0]['artifacts']
    retained = {desc['sha256'] for desc in artifacts}
    extents = {(desc['sha256'],desc['size_bytes']) for desc in artifacts}
    admitted = raw.get('admitted_files')
    if type(admitted) is not list or not admitted:
        return False
    for desc in admitted:
        s._require(type(desc) is dict, 'invalid native admitted file')
        s._hex(desc.get('sha256'), 64, 'native admitted SHA')
        s._integer(desc.get('size_bytes'), 'native admitted bytes')
        if (desc['sha256'],desc['size_bytes']) not in extents:
            return False
    compiler_keys = ['shader_compiler']
    spirv_keys = ['producer_spirv_sha256','stereo_spirv_sha256','depth_spirv_sha256']
    if raw.get('variant') == 'coarse_refine':
        compiler_keys.append('refinement_shader_compiler')
        spirv_keys.append('refinement_spirv_sha256')
    for key in compiler_keys:
        compiler = raw.get(key)
        if type(compiler) is not dict or 'compiler_sha256' not in compiler:
            return False
        s._hex(compiler['compiler_sha256'], 64, 'compiler SHA')
        if compiler['compiler_sha256'] not in retained:
            return False
    for key in spirv_keys:
        mapping = raw.get(key)
        if type(mapping) is not dict or not mapping:
            return False
        for digest in mapping.values():
            s._hex(digest, 64, 'SPIR-V SHA')
            if digest not in retained:
                return False
    return True


def _quality(document, record):
    required = {'schema','status','parity_qualified','reasons','methods','frames','strata','intersections',
                'comparisons','requested_range_source','partition_digest','partition_digest_algorithm'}
    s._keys(document, required, 'quality report')
    s._require(document['schema'] == 'stereoforge.zed_comparison.quality.v1', 'unknown quality schema')
    s._hex(document['partition_digest'], 64, 'partition_digest')
    for key in ('methods','frames','comparisons','reasons','strata','intersections'):
        s._require(type(document[key]) is list, 'invalid quality ' + key)
    for key in ('methods','frames','comparisons'):
        s._require(all(type(row) is dict for row in document[key]), 'quality rows must be objects')
    for row in document['frames']:
        for key in ('method_key','input_identity','stratum_key'): s._hex(row.get(key), 64, 'quality '+key)
        s._require(row.get('partition') in ('development','held_out'), 'invalid quality partition')
    method = record['method_key']
    methods = [m for m in document['methods'] if m.get('method_key') == method]
    s._require(len(methods) == 1 and methods[0].get('implementation') == record['implementation'], 'quality method binding mismatch')
    frames = [f for f in document['frames'] if f.get('method_key') == method]
    bound = (bool(frames) and all(f.get('status') == 'available' and f.get('partition') == 'held_out' and f.get('stratum_key') == record['calibration_identity'] for f in frames)
             and sorted({f.get('input_identity') for f in frames}) == record['input_identities'])
    comparison = any(c.get('parity_qualified') is True and c.get('reasons') == [] and
                     method in (c.get('candidate_method'), c.get('reference_method')) for c in document['comparisons'])
    return document['status'] == 'qualified' and document['parity_qualified'] is True and document['reasons'] == [] and bound and comparison


def _protocol(document, record, files):
    s._keys(document, {'schema','frozen','session_id','order','min_blocks','min_samples_per_run','contention_before','contention_after'}, 'protocol')
    s._require(document['schema'] == 'stereoforge.zed_comparison.timing_protocol.v1', 'unknown protocol schema')
    s._require(type(document['frozen']) is bool, 'protocol frozen must be boolean')
    s._text(document['session_id'], 'protocol session')
    s._require(document['session_id'] == record['session_id'], 'protocol session mismatch')
    order = document['order']
    s._require(type(order) is list and len(order) == 4, 'protocol requires ABBA')
    for profile in order:
        s._hex(profile, 64, 'protocol profile')
    s._require(order[0] == order[3] and order[1] == order[2] and order[0] != order[1], 'protocol requires ABBA')
    s._integer(document['min_blocks'], 'min_blocks', 2); s._integer(document['min_samples_per_run'], 'min_samples_per_run', 1)
    clear = True
    for key in ('contention_before','contention_after'):
        value = document[key]
        s._availability(value, s._descriptor, key)
        if _value(value) is None:
            clear = False; continue
        contention = _load(files, value['value'])
        s._keys(contention, {'schema','status','platforms','method'}, 'contention')
        s._require(contention['schema'] == 'stereoforge.zed_comparison.contention.v1', 'unknown contention schema')
        s._require(contention['status'] in ('clear','contended','unknown'), 'invalid contention status')
        s._text(contention['method'], 'contention method')
        s._require(type(contention['platforms']) is list and bool(contention['platforms']), 'missing contention platforms')
        for platform in contention['platforms']:
            _platform(platform)
        clear &= contention['status'] == 'clear' and record['platform'] in contention['platforms']
    return clear


def _raw_binding(document, record):
    if document.get('schema') in NATIVE:
        context = {k:v for k,v in record.items() if k not in {'schema','profile_id','sample_count','samples','stages'}}
        adapted = adapt_native_timing(document, context)
        s._require(adapted == record, 'raw native samples/metadata mismatch')
        identity = document.get('input_identity')
        if identity is not None:
            s._hex(identity, 64, 'raw input_identity')
            s._require(record['input_identities'] == [identity], 'raw input identity mismatch')
        if document['schema'] != 'stereoforge.daxa_depth_rectified_benchmark.v1':
            return ['raw_input_identity_not_bound']
        from .native_observation import _settings
        from .evaluate_temporal import temporal_settings
        settings = _settings(document)
        if 'temporal' in document:
            settings['temporal'] = document['temporal']
        expected = dict(name='StereoForge', version='canonical_native_v1',
            commit=_available(document['commit']) if 'commit' in document else _unavailable('native report did not record a StereoForge commit'),
            device=_available(document['device']) if 'device' in document else _unavailable('native report did not record a device'), settings=settings)
        s._require(record['implementation'] == expected, 'raw implementation/settings binding mismatch')
        state = temporal_settings({'implementation':expected})['state']
        return ([] if identity is not None else ['raw_input_identity_not_bound']) + ([] if state == 'temporal_off' else ['raw_temporal_settings_not_bound'])
    s._require(document.get('schema') == 'stereoforge.zed_comparison.sdk_timing.v1', 'unrecognized raw timing schema')
    s._require(record['lane'] == 'opaque_api' and document.get('completion_boundary') == 'opaque_api_return', 'SDK timing is opaque only')
    # SDK report format is checked against the actual retained raw API-return span.
    for key, expected in dict(operation='benchmark',lane='opaque_api',timing_lane='opaque-api',units='ms',
            sample_order='chronological',sample_population='successful_measured_grabs_only',
            file_io_in_span=False,diagnostic_retrieval_in_span=False,playback_pacing=False).items():
        s._require(type(document.get(key)) is type(expected) and document[key] == expected, 'SDK timing metadata mismatch: '+key)
    s._require(document.get('sample_count') == record['sample_count'] and document.get('phase') == record['phase'] and
        document.get('excluded_warmup_count') == record['warmup_count'] and document.get('requested_outputs') == record['requested_outputs'], 'SDK count/phase/output mismatch')
    values = document.get('opaque_grab_api_ms')
    s._require(values == document.get('complete_wall_ms'), 'SDK raw wall mismatch')
    s._require(values == _value(record['samples']['opaque_api_ms']) and values == _value(record['samples']['complete_wall_ms']), 'SDK raw samples mismatch')
    s._require(all(_value(v) is None for k,v in record['samples'].items() if k not in ('opaque_api_ms','complete_wall_ms')), 'SDK internal timings unavailable')
    s._require(not record['stages'], 'SDK internal stages unavailable')
    return ['raw_input_identity_not_bound']


def summarize_timing(records, *, root=None):
    """Return raw populations and fail-closed ABBA comparisons; never infer FPS."""
    s._require(type(records) is list and bool(records), 'records must be nonempty list')
    s._require(len(records) <= 100000, 'too many timing records')
    runs, slots, blocks, populations, admissions = set(), set(), {}, {}, []
    total = 0
    roots = ([None]*len(records) if root is None else root if type(root) is list else [root]*len(records))
    s._require(len(roots) == len(records) and (root is None or all(r is not None for r in roots)), 'root list must match records')
    file_sets, evidence_cache = {}, {}
    with ExitStack() as stack:
        for record, directory in zip(records, roots):
            validate_timing_lane(record)
            total += sum(len(v['value']) for collection in ('samples','stages') for v in record[collection].values() if _value(v) is not None)
            s._require(total <= 1000000, 'total raw sample bound exceeded')
            session, run = record['session_id'], record['run_id']; order = record['order']
            s._require((session,run) not in runs, 'repeated run'); runs.add((session,run))
            slot = (session,order['block'],order['position'])
            s._require(slot not in slots, 'duplicate interleaving slot'); slots.add(slot)
            block = blocks.setdefault((session,order['block']), [])
            block.append(record)
            signature = lambda r: s._canonical([r[k] for k in ('phase','lane','completion_boundary','scope')] + [sorted(r['requested_outputs'])])
            s._require(all(signature(r) == signature(record) for r in block), 'mixed completion/lane/scope/outputs/phase in block')
            key = s._canonical([record[k] for k in ('phase','lane','completion_boundary','platform','profile_id','scope')])
            population = populations.setdefault(key, dict(phase=record['phase'], lane=record['lane'], completion_boundary=record['completion_boundary'],
                platform=record['platform'], profile_id=record['profile_id'], scope=record['scope'], run_ids=[], samples={}, stages={}, cold_prepare_setup=[]))
            population['run_ids'].append(run)
            for collection in ('samples','stages'):
                for metric, value in record[collection].items():
                    if _value(value) is not None:
                        population[collection].setdefault(metric, []).extend(value['value'])
            reasons = []; protocol = None
            if directory is None:
                reasons.extend(['missing_root', 'freeze_not_verified'])
                for kind in ('quality','protocol'):
                    if _value(record['sources'][kind]) is None: reasons.append('missing_' + kind)
            else:
                files = file_sets.setdefault(Path(directory).resolve(), s._Files(directory, stack))
                raw = _load(files, record['sources']['timing'])
                reasons.extend(_raw_binding(raw, record))
                if raw.get('schema') in NATIVE:
                    population['cold_prepare_setup'].append(dict(run_id=run, value=raw['timing_metadata']['cold_prepare'], scope=raw['timing_metadata']['cold_prepare_scope']))
                freeze = _load(files, record['sources']['freeze'])
                freeze_key = ('freeze', files.root, record['sources']['freeze']['sha256'], record['profile_id'])
                if freeze_key not in evidence_cache:
                    evidence_cache[freeze_key] = _freeze(freeze, record, files)
                if not evidence_cache[freeze_key]: reasons.append('missing_or_unfrozen_profile')
                if not _freeze_coverage(freeze, raw, record): reasons.append('freeze_artifact_coverage_missing')
                for kind in ('quality','protocol'):
                    desc = _value(record['sources'][kind])
                    if desc is None:
                        reasons.append('missing_' + kind); continue
                    document = _load(files, desc)
                    if kind == 'quality':
                        quality_key = ('quality', files.root, desc['sha256'], record['profile_id'])
                        if quality_key not in evidence_cache:
                            evidence_cache[quality_key] = _quality(document, record)
                        if not evidence_cache[quality_key]: reasons.append('unqualified_quality')
                    else:
                        protocol = document
                        protocol_key = ('protocol', files.root, desc['sha256'], record['session_id'], s._canonical(record['platform']))
                        if protocol_key not in evidence_cache:
                            evidence_cache[protocol_key] = _protocol(document, record, files)
                        if not evidence_cache[protocol_key]: reasons.append('unavailable_or_contended_platform')
                        if not document['frozen']: reasons.append('unfrozen_protocol')
            if record['phase'] != 'warm': reasons.append('requires_warm_phase')
            if record['lane'] == 'opaque_api': reasons.append('opaque_api_not_compute_complete')
            if any(record['isolation'].values()): reasons.append('io_or_diagnostics_in_span')
            if _value(record['samples']['complete_wall_ms']) is None: reasons.append('missing_complete_wall')
            if record['lane'] == 'gpu_resident' and _value(record['samples']['gpu_total_ms']) is None: reasons.append('missing_gpu_total')
            if record['lane'] == 'host_output' and _value(record['cache']['host_output_properties']) is None: reasons.append('missing_host_output_properties')
            admissions.append(dict(run_id=run, qualified=not reasons, reasons=reasons, protocol=protocol))
        for files in file_sets.values(): files.recheck()
    comparisons = []
    phase_blocks = {}
    for (session,_),block in blocks.items():
        phase_blocks.setdefault((session,block[0]['phase']), []).append(block)
    sessions = {}
    for i,record in enumerate(records):
        sessions.setdefault((record['session_id'],record['phase']), []).append(i)
    for (session,phase),indices in sorted(sessions.items()):
        members = [records[i] for i in indices]; reasons = {reason for i in indices for reason in admissions[i]['reasons']}
        profiles = sorted({r['profile_id'] for r in members})
        protocols = [admissions[i]['protocol'] for i in indices]
        protocol = protocols[0]
        if protocol is None or any(p != protocol for p in protocols): reasons.add('missing_or_changed_protocol')
        freezes = {r['sources']['freeze']['sha256'] for r in members}
        if len(freezes) != 1: reasons.add('changed_freeze')
        session_blocks = phase_blocks[(session,phase)]
        if any([r['order']['position'] for r in b] != [0,1,2,3] for b in session_blocks): reasons.add('incomplete_interleaving')
        if protocol:
            if any([r['profile_id'] for r in b] != protocol['order'] for b in session_blocks): reasons.add('invalid_abba_order')
            if len(session_blocks) < protocol['min_blocks']: reasons.add('insufficient_blocks')
            if any(r['sample_count'] < protocol['min_samples_per_run'] for r in members): reasons.add('insufficient_run_samples')
        else: reasons.add('incomplete_interleaving')
        if len(profiles) != 2: reasons.add('requires_two_profiles')
        bindings = {s._canonical([r['input_identities'],r['calibration_identity'],r['lane'],r['completion_boundary'],r['scope'],sorted(r['requested_outputs'])]) for r in members}
        if len(bindings) != 1: reasons.add('incomparable_profiles')
        same_platform = len({s._canonical(r['platform']) for r in members}) == 1 and all(_value(r['platform']['device_uuid']) is not None for r in members)
        if not same_platform: reasons.add('cross_platform_operating_points')
        ratio = None
        if not reasons:
            order_profiles = protocol['order'][:2]
            means = []
            for profile in order_profiles:
                count = sum(r['sample_count'] for r in members if r['profile_id'] == profile)
                means.append(math.fsum(x/count for r in members if r['profile_id'] == profile for x in r['samples']['complete_wall_ms']['value']))
            if means[1] > 0:
                ratio = means[0]/means[1]
                if not math.isfinite(ratio):
                    ratio = None; reasons.add('ratio_outside_finite_range')
            else: reasons.add('zero_ratio_denominator')
        comparisons.append(dict(session_id=session, phase=phase, profiles=profiles, qualified=not reasons,
                                kind='same_device_runtime' if same_platform else 'operating_points', ratio=ratio,
                                ratio_scope='A_complete_wall_mean_over_B_complete_wall_mean', reasons=sorted(reasons)))
    for population in populations.values():
        population['statistics'] = {collection:{metric:_stats(values) for metric,values in population[collection].items()} for collection in ('samples','stages')}
    for admission in admissions: admission.pop('protocol')
    return dict(schema='stereoforge.zed_comparison.timing_summary.v1', populations=list(populations.values()),
                comparisons=comparisons, admissions=admissions)
