"""Offline accounting of bounded ROS publication and subscriber traces.

A delivery result does not establish stereo quality or SDK parity. Trace data
are assertions from the producing applications, not independent attestation.
"""
import math
import re
from collections import defaultdict
from copy import deepcopy

from . import schema as s

EVENT_SCHEMA = 'stereoforge.ros_delivery_event.v1'
POLICY_SCHEMA = 'stereoforge.ros_delivery_policy.v1'
SUMMARY_SCHEMA = 'stereoforge.ros_delivery_summary.v1'
BASE = {'schema', 'kind', 'run_id', 'role', 'at_ns'}
COUNTERS = {'read', 'admitted', 'ring_drops', 'identity_rejects', 'published_depth',
            'published_rgb', 'received_depth', 'received_rgb', 'timeouts'}
CAPTURE = {'source_records', 'source_frames', 'delivered_frames', 'queue_drops',
           'no_slot_drops', 'source_frame_gaps', 'repeated_frames', 'source_timeouts',
           'invalid_frames', 'backpressure_waits', 'backpressure_wait_ns', 'queued_frames',
           'queue_high_watermark', 'pool_capacity', 'packed_bytes_per_slot', 'running'}
MEMORY = {'producer_requested_bytes', 'export_requested_bytes', 'capture_requested_bytes',
          'trace_capacity', 'trace_record_bytes', 'ros_requested_bytes',
          'export_memory_type_index', 'export_memory_property_flags', 'export_host_cached',
          'storage_accounting_complete'}
PUBLISHED = {'output', 'source_id', 'capture_ns', 'capture_clock', 'host_ingress_ns',
             'mapped_capture_ns', 'mapping_error_ns', 'producer_ticket', 'copy_ticket',
             'producer_complete_observed_ns', 'copy_complete_observed_ns', 'copy_begin_ns',
             'copy_end_ns', 'publish_begin_ns', 'publish_end_ns', 'capture_queue', 'pending', 'freshness'}
RECEIVED = {'output', 'mapped_capture_ns', 'arrival_system_ns', 'width', 'height', 'step',
            'encoding', 'frame_id', 'payload_bytes'}
STOP = {'window_end_ns', 'reason', 'counters', 'capture_counters', 'queued', 'pending',
        'trace_drops', 'drain_ns'}
NS_MAX = (1 << 63) - 1
ROS_STAMP_MAX = (1 << 31) * 1_000_000_000 - 1


def _int(value, name, low=0, high=NS_MAX):
    s._integer(value, name, low, high)


def _text(value, name):
    s._text(value, name)
    s._require(len(value.encode('utf-8')) <= 4096 and '\0' not in value, name + ' too long or contains NUL')


def _bounded_metadata(value):
    stack=[(value,0)]; count=0; text_bytes=0
    while stack:
        item,depth=stack.pop(); count+=1
        s._require(count<=10000 and depth<=32, 'metadata exceeds item or nesting bound')
        if item is None or type(item) is bool: continue
        if type(item) is int:
            s._require(-(1 << 63) <= item < (1 << 64), 'metadata integer outside bounds'); continue
        if type(item) is float:
            s._require(math.isfinite(item), 'nonfinite metadata'); continue
        if type(item) is str:
            text_bytes+=len(item.encode('utf-8'))
            s._require(len(item.encode('utf-8'))<=4096 and text_bytes<=1_048_576 and '\0' not in item, 'metadata text exceeds bounds'); continue
        if type(item) is list:
            s._require(len(item)<=10000-count, 'metadata list exceeds bounds')
            stack.extend((v,depth+1) for v in item); continue
        if type(item) is dict:
            s._require(len(item)<=10000-count, 'metadata object exceeds bounds')
            for key,val in item.items():
                s._require(type(key) is str, 'metadata keys must be strings')
                stack.append((key,depth+1));stack.append((val,depth+1))
            continue
        raise ValueError('metadata is not JSON')


def _capture(value):
    s._keys(value, CAPTURE, 'capture counters')
    for key in CAPTURE - {'running'}:
        _int(value[key], key)
    s._require(type(value['running']) is bool, 'capture running must be boolean')
    s._require(value['queued_frames'] <= value['queue_high_watermark'], 'invalid capture high watermark')


def _outputs(value):
    s._require(type(value) is dict and value and set(value) <= {'depth', 'rgb'}, 'invalid outputs')
    topics = set()
    for name, spec in value.items():
        s._keys(spec, {'topic', 'width', 'height', 'step', 'encoding', 'frame_id'}, 'output')
        _text(spec['topic'], 'topic'); _text(spec['frame_id'], 'frame_id')
        s._require(spec['frame_id'] != 'unknown' and spec['topic'] not in topics, 'invalid frame or duplicate topic')
        topics.add(spec['topic'])
        _int(spec['width'], 'width', 1, 8192); _int(spec['height'], 'height', 1, 8192)
        s._require(spec['step'] == spec['width'] * 4 and spec['encoding'] == ('32FC1' if name == 'depth' else 'rgba8'), 'invalid image layout')


def _qos(value):
    s._keys(value, {'history', 'depth', 'reliability', 'durability'}, 'qos')
    s._require(value['history'] == 'keep_last' and type(value['depth']) is int and value['depth'] == 1 and
               value['reliability'] in ('reliable', 'best_effort') and value['durability'] == 'volatile', 'invalid qos')


def validate_delivery_policy(policy):
    keys = {'schema', 'frozen', 'min_duration_s', 'target_hz', 'min_delivery_hz',
            'max_drop_fraction', 'max_gap_ms', 'max_frame_age_ms', 'max_clock_error_ns',
            'max_queue_growth', 'max_requested_bytes'}
    s._keys(policy, keys, 'delivery policy')
    s._require(policy['schema'] == POLICY_SCHEMA and type(policy['frozen']) is bool, 'invalid delivery policy')
    for key in ('min_duration_s', 'target_hz', 'min_delivery_hz', 'max_drop_fraction', 'max_gap_ms', 'max_frame_age_ms'):
        s._number(policy[key], key, minimum=0)
    s._require(policy['min_duration_s'] >= 600 and policy['target_hz'] == 60 and
               0 < policy['min_delivery_hz'] <= policy['target_hz'] and policy['max_drop_fraction'] <= 1,
               'invalid delivery policy limits')
    for key in ('max_clock_error_ns', 'max_queue_growth', 'max_requested_bytes'):
        _int(policy[key], key)
    return dict(policy)


def _start(event):
    s._keys(event, BASE | {'config'}, 'start')
    c = event['config']; role = event['role']
    common = {'clock_id', 'nominal_hz', 'outputs', 'qos'}
    publisher = {'source_mode', 'slots', 'capture_queue_capacity', 'implementation', 'calibration_identity',
                 'effective_estimator', 'view_profile', 'clock_mapping', 'capture_counters', 'memory'}
    optional_cpu = {'thread_cpu_timing'} if role == 'publisher' and 'thread_cpu_timing' in c else set()
    s._keys(c, common | (publisher if role == 'publisher' else set()) | optional_cpu, 'start config')
    if optional_cpu: s._require(type(c['thread_cpu_timing']) is bool, 'thread_cpu_timing must be boolean')
    _text(c['clock_id'], 'clock_id')
    match=re.fullmatch(r'[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}:monotonic:([1-9][0-9]{0,19})', c['clock_id'])
    s._require(match is not None and int(match.group(1)) <= (1 << 64)-1,
               'clock_id must name a Linux boot UUID, monotonic clock and time namespace')
    s._number(c['nominal_hz'], 'nominal_hz', positive=True)
    s._require(c['nominal_hz'] <= 120, 'nominal_hz too large')
    _outputs(c['outputs']); _qos(c['qos'])
    if role == 'subscriber': return
    s._require(c['source_mode'] in ('camera', 'replay', 'synthetic') and type(c['slots']) is int and c['slots'] == 2 and
               type(c['capture_queue_capacity']) is int and c['capture_queue_capacity'] == 2, 'invalid source/ring')
    s._hex(c['calibration_identity'], 64, 'calibration_identity')
    s._require(type(c['effective_estimator']) is dict, 'invalid estimator')
    s._availability(c['view_profile'], lambda x: _text(x, 'view profile'), 'view profile')
    s._availability(c['implementation'], lambda x: s._require(type(x) is dict and bool(x), 'invalid implementation'), 'implementation')
    for key in ('effective_estimator','view_profile','implementation'): _bounded_metadata(c[key])
    m = c['clock_mapping']; s._keys(m, {'source_clock','offset_ns','valid_from_ns','valid_until_ns','max_error_ns'}, 'clock mapping')
    _text(m['source_clock'], 'source clock'); s._require(m['source_clock'] != 'unknown', 'unknown source clock')
    s._require(type(m['offset_ns']) is int and -(1 << 63) <= m['offset_ns'] <= NS_MAX, 'invalid clock offset')
    _int(m['valid_from_ns'], 'valid_from_ns', 1); _int(m['valid_until_ns'], 'valid_until_ns', m['valid_from_ns']); _int(m['max_error_ns'], 'max_error_ns')
    _capture(c['capture_counters']); mem=c['memory']
    optional_reason = {'producer_requested_bytes_reason'} if isinstance(mem, dict) and 'producer_requested_bytes_reason' in mem else set()
    s._keys(mem, MEMORY | optional_reason, 'memory')
    for key in MEMORY - {'producer_requested_bytes', 'export_host_cached', 'storage_accounting_complete'}: _int(mem[key], key)
    producer_bytes=mem['producer_requested_bytes']
    if producer_bytes is None:
        s._require(bool(optional_reason), 'unavailable producer bytes require reason')
        _text(mem['producer_requested_bytes_reason'], 'producer_requested_bytes_reason')
        s._require(mem['producer_requested_bytes_reason'] != 'available' and mem['storage_accounting_complete'] is False,
                   'unavailable producer bytes cannot claim complete accounting')
    else:
        _int(producer_bytes, 'producer_requested_bytes')
        if optional_reason: s._require(mem['producer_requested_bytes_reason'] == 'available', 'numeric producer bytes require available reason')
    flags=mem['export_memory_property_flags']
    s._require(type(mem['storage_accounting_complete']) is bool and type(mem['export_host_cached']) is bool and bool(flags & 8) == mem['export_host_cached'] and
               (flags & 6) == 6 and mem['export_memory_type_index'] < 32, 'invalid coherent export memory properties')


def _stop(event, start):
    s._keys(event, BASE | STOP, 'stop')
    _int(event['window_end_ns'], 'window_end', start['at_ns'], event['at_ns'])
    _text(event['reason'], 'reason')
    s._require(event['reason'] in ('limit', 'eof', 'shutdown', 'error', 'trace_full'), 'invalid stop reason')
    s._keys(event['counters'], COUNTERS, 'counters')
    for key in COUNTERS: _int(event['counters'][key], key)
    unused = ({'received_depth', 'received_rgb'} if event['role'] == 'publisher' else
              {'read', 'admitted', 'ring_drops', 'identity_rejects', 'published_depth', 'published_rgb'})
    s._require(all(event['counters'][key] == 0 for key in unused), 'nonapplicable counters must be zero')
    for key in ('queued', 'pending', 'trace_drops', 'drain_ns'): _int(event[key], key)
    if event['role'] == 'publisher':
        _capture(event['capture_counters'])
        baseline = start['config']['capture_counters']; final = event['capture_counters']
        for key in CAPTURE - {'running', 'queued_frames', 'pool_capacity', 'packed_bytes_per_slot'}:
            s._require(final[key] >= baseline[key], 'capture counter went backwards: '+key)
        s._require(event['queued'] <= 2 and event['pending'] <= 2, 'queue exceeds bounded capacity')
    else:
        s._require(event['capture_counters'] is None and event['queued'] == 0 and event['pending'] == 0, 'invalid subscriber stop')


ENQUEUE_TIMING = {'producer_enqueue_begin_ns', 'producer_enqueue_end_ns', 'export_enqueue_end_ns'}


def _frame(event, start):
    role = event['role']; config = start['config']
    timing = ENQUEUE_TIMING & event.keys() if role == 'publisher' else set()
    s._require(not timing or timing == ENQUEUE_TIMING, 'incomplete enqueue timing')
    cpu_enabled = role == 'publisher' and config.get('thread_cpu_timing', False)
    cpu_field = {'thread_cpu_timing'} if role == 'publisher' and 'thread_cpu_timing' in event else set()
    s._require(bool(cpu_field) == cpu_enabled, 'missing or undeclared thread cpu timing')
    s._keys(event, BASE | (PUBLISHED | timing | cpu_field if role == 'publisher' else RECEIVED), 'frame')
    if cpu_enabled:
        cpu = event['thread_cpu_timing']
        s._require(type(cpu) is dict and cpu.get('status') in ('available', 'unavailable'), 'invalid thread cpu timing')
        if cpu['status'] == 'available':
            fields = ('copy_begin_ns','copy_end_ns','publish_begin_ns','publish_end_ns')
            s._keys(cpu, {'status', *fields}, 'thread cpu timing')
            for field in fields: _int(cpu[field], field)
            values = [cpu[field] for field in fields]
            s._require(values == sorted(values), 'thread cpu timing went backwards')
        else:
            s._keys(cpu, {'status','reason'}, 'thread cpu timing')
            s._require(cpu['reason'] in ('clock_read_failed','nonmonotonic'), 'invalid thread cpu failure')
    name = event['output']; s._require(type(name) is str and name in config['outputs'], 'unknown output')
    if role == 'subscriber':
        spec = config['outputs'][name]
        for field in ('width','height','step','encoding','frame_id'):
            s._require(type(event[field]) is type(spec[field]) and event[field] == spec[field], 'received layout mismatch')
        _int(event['mapped_capture_ns'], 'mapped capture', 1, ROS_STAMP_MAX); _int(event['arrival_system_ns'], 'arrival system', 1)
        s._require(type(event['payload_bytes']) is int and event['payload_bytes'] == spec['step'] * spec['height'], 'received extent mismatch')
        return
    for key in PUBLISHED - {'output','capture_clock','freshness'}: _int(event[key], key)
    s._require(event['capture_ns'] > 0 and event['host_ingress_ns'] > 0 and event['mapped_capture_ns'] > 0 and event['producer_ticket'] > 0 and
               event['copy_ticket'] > event['producer_ticket'], 'invalid source or ticket')
    m = config['clock_mapping']; _text(event['capture_clock'], 'capture clock')
    s._require(event['capture_clock'] == m['source_clock'] and m['valid_from_ns'] <= event['capture_ns'] <= m['valid_until_ns'] and
               event['mapped_capture_ns'] <= ROS_STAMP_MAX and event['mapped_capture_ns'] == event['capture_ns'] + m['offset_ns'] and
               event['mapping_error_ns'] == m['max_error_ns'], 'clock mapping mismatch')
    order = [event[k] for k in ('producer_complete_observed_ns','copy_complete_observed_ns','copy_begin_ns',
                                 'copy_end_ns','publish_begin_ns','publish_end_ns')]
    s._require(order == sorted(order) and (config['source_mode'] != 'camera' or event['host_ingress_ns'] <= order[0]) and event['at_ns'] == order[-1], 'invalid completion order')
    if timing:
        enqueue = [event[k] for k in ('producer_enqueue_begin_ns', 'producer_enqueue_end_ns', 'export_enqueue_end_ns')]
        for key in ENQUEUE_TIMING: _int(event[key], key, 1)
        s._require(enqueue == sorted(enqueue) and enqueue[-1] <= order[0] and
                   (config['source_mode'] != 'camera' or event['host_ingress_ns'] <= enqueue[0]), 'invalid enqueue order')
    s._require(event['capture_queue'] <= 2 and event['pending'] <= 2 and
               event['freshness'] in ('measured_single_source','historical','unknown'), 'invalid queue or freshness')


def _percentiles(values):
    if not values: return None
    values = sorted(values)
    def nearest(q): return values[max(0, math.ceil(len(values)*q)-1)]
    return {'p50': nearest(.50), 'p95': nearest(.95), 'p99': nearest(.99), 'max': values[-1]}


def summarize_delivery(events, *, policy=None):
    """Validate and summarize each run; unverified/missing evidence is unqualified."""
    s._require(type(events) is list and len(events) <= 2_000_000, 'invalid event list')
    if policy is not None: policy = validate_delivery_policy(policy)
    groups = defaultdict(lambda: defaultdict(list))
    for event in events:
        s._require(type(event) is dict and event.get('schema') == EVENT_SCHEMA, 'invalid event schema')
        _text(event.get('run_id'), 'run_id'); _int(event.get('at_ns'), 'at_ns')
        role = event.get('role'); kind = event.get('kind')
        s._require(role in ('publisher','subscriber') and kind in ('start','stop','published','received'), 'invalid event kind/role')
        s._require(kind in ('start','stop') or (kind == 'published') == (role == 'publisher'), 'kind/role mismatch')
        rows = groups[event['run_id']][role]
        s._require(not rows or event['at_ns'] >= rows[-1]['at_ns'], 'event time went backwards')
        if kind == 'start':
            s._require(not rows, 'duplicate start'); _start(event)
        else:
            s._require(rows and rows[0]['kind'] == 'start' and rows[-1]['kind'] != 'stop', 'event outside run')
            if kind == 'stop': _stop(event, rows[0])
            else: _frame(event, rows[0])
        rows.append(event)
    runs=[]
    for run_id, roles in sorted(groups.items()):
        reasons=[]; outputs={}
        if policy is None: reasons.append('missing_policy')
        elif not policy['frozen']: reasons.append('provisional_policy')
        if set(roles) != {'publisher','subscriber'}: reasons.append('missing_role')
        if any(not rows or rows[-1]['kind'] != 'stop' for rows in roles.values()): reasons.append('missing_stop')
        if reasons and (set(roles) != {'publisher','subscriber'} or 'missing_stop' in reasons):
            runs.append(dict(run_id=run_id,qualified=False,reasons=reasons,outputs=outputs)); continue
        pub, sub = roles['publisher'], roles['subscriber']; pc, sc = pub[0]['config'], sub[0]['config']
        if pc['clock_id'] != sc['clock_id']: reasons.append('incompatible_host_clocks')
        if pc['outputs'] != sc['outputs'] or pc['qos'] != sc['qos'] or pc['nominal_hz'] != sc['nominal_hz']: reasons.append('profile_mismatch')
        if pc['source_mode'] != 'camera': reasons.append('not_live_camera')
        else: reasons.append('camera_clock_mapping_unverified')
        if any(x[-1]['trace_drops'] or x[-1]['reason'] in ('error','trace_full') for x in (pub,sub)): reasons.append('incomplete_trace')
        if len(pub) > pc['memory']['trace_capacity']: reasons.append('trace_capacity_exceeded')
        for role_rows, name, counter_prefix in ((pub, 'publisher', 'published'), (sub, 'subscriber', 'received')):
            counters = role_rows[-1]['counters']
            in_window = [e for e in role_rows[1:-1] if e['at_ns'] <= role_rows[-1]['window_end_ns']]
            for output in ('depth', 'rgb'):
                actual = sum(e['kind'] == counter_prefix and e['output'] == output for e in in_window)
                if counters[counter_prefix + '_' + output] != actual:
                    reasons.append(name + '_counter_mismatch')
            if name == 'publisher':
                if counters['admitted'] > counters['read'] or counters['ring_drops'] + counters['identity_rejects'] > counters['read']:
                    reasons.append('publisher_counter_mismatch')
                delivered_delta = role_rows[-1]['capture_counters']['delivered_frames'] - pc['capture_counters']['delivered_frames']
                if counters['read'] != delivered_delta: reasons.append('capture_read_mismatch')
        same_clock=pc['clock_id']==sc['clock_id']
        if same_clock:
            start=max(pub[0]['at_ns'],sub[0]['at_ns']); end=min(pub[-1]['window_end_ns'],sub[-1]['window_end_ns'])
            s._require(end >= start, 'non-overlapping windows'); duration=(end-start)/1e9
        else:
            start=end=duration=None
        if policy and ((duration is not None and duration < policy['min_duration_s']) or pc['nominal_hz'] != policy['target_hz']): reasons.append('duration_or_nominal_rate')
        mem=pc['memory']
        if mem['producer_requested_bytes'] is None:
            requested_bytes=None
            reasons.append('producer_requested_bytes_unavailable')
        else:
            requested_bytes = sum(mem[k] for k in ('producer_requested_bytes','export_requested_bytes','capture_requested_bytes','ros_requested_bytes')) + mem['trace_capacity']*mem['trace_record_bytes']
        if not mem['storage_accounting_complete']: reasons.append('requested_storage_accounting_incomplete')
        if policy and requested_bytes is not None and requested_bytes > policy['max_requested_bytes']: reasons.append('requested_memory_exceeds_policy')
        first_pending=next((e['pending'] for e in pub[1:-1] if e['kind']=='published' and e['at_ns'] <= pub[-1]['window_end_ns']), 0)
        if policy and (pub[-1]['queued'] - pc['capture_counters']['queued_frames'] > policy['max_queue_growth'] or
                       pub[-1]['pending'] - first_pending > policy['max_queue_growth']): reasons.append('queue_growth')
        pstart, pend=(start,end) if same_clock else (pub[0]['at_ns'],pub[-1]['window_end_ns'])
        sstart, send=(start,end) if same_clock else (sub[0]['at_ns'],sub[-1]['window_end_ns'])
        pframes=[e for e in pub[1:-1] if e['kind']=='published' and pstart <= e['at_ns'] <= pend]
        rframes=[e for e in sub[1:-1] if e['kind']=='received' and sstart <= e['at_ns'] <= send]
        for output in pc['outputs']:
            publications={}; source_seen=set(); duplicate=0; previous_id=-1; previous_capture=0
            output_publications=[x for x in pframes if x['output']==output]
            for e in output_publications:
                stamp=e['mapped_capture_ns']
                if stamp in publications:
                    duplicate+=1
                    if publications[stamp]['source_id'] != e['source_id']:
                        reasons.append('conflicting_publication_stamp')
                    continue
                publications[stamp]=e
                if e['source_id'] in source_seen: duplicate+=1
                if e['source_id'] <= previous_id or e['capture_ns'] <= previous_capture:
                    reasons.append('source_identity_not_increasing')
                previous_id, previous_capture = e['source_id'], e['capture_ns']
                source_seen.add(e['source_id'])
            received=[]; stamps=set(); ids=set(); unmatched=0; repeated=0
            for e in (x for x in rframes if x['output']==output):
                stamp=e['mapped_capture_ns']; p=publications.get(stamp)
                if p is None: unmatched+=1; continue
                if stamp in stamps or p['source_id'] in ids: repeated+=1; continue
                stamps.add(stamp); ids.add(p['source_id'])
                if p['freshness'] == 'measured_single_source': received.append((e,p))
            gaps=[(b[0]['at_ns']-a[0]['at_ns'])/1e6 for a,b in zip(received,received[1:])] if same_clock else []
            delays=[(e['at_ns']-p['publish_begin_ns'])/1e6 for e,p in received] if pc['clock_id']==sc['clock_id'] else []
            valid_delays = all(x >= 0 for x in delays)
            if not valid_delays: reasons.append('arrival_before_publish')
            # Host ingress is the monotonic timestamp after capture's host copy,
            # not an exposure timestamp. Replay ingress is not live capture age.
            ingress_delays=[]
            if pc['source_mode'] != 'camera': ingress_status='not_live_camera'
            elif not same_clock: ingress_status='incompatible_host_clocks'
            elif duplicate or repeated: ingress_status='ambiguous_source_identity'
            else:
                ingress_delays=[(e['at_ns']-p['host_ingress_ns'])/1e6 for e,p in received]
                if any(x < 0 for x in ingress_delays):
                    ingress_status='negative_interval'; reasons.append('arrival_before_host_ingress')
                elif not ingress_delays: ingress_status='no_matched_fresh_observations'
                else: ingress_status='available'
            rate=(len(received)/duration if duration > 0 else 0) if same_clock else None
            captured=pub[-1]['capture_counters']['source_frames']-pc['capture_counters']['source_frames']
            gaps_in_source=pub[-1]['capture_counters']['source_frame_gaps']-pc['capture_counters']['source_frame_gaps']
            offered=captured+gaps_in_source
            drop_fraction=(offered-len(received))/offered if same_clock and offered else None
            if same_clock: s._require(offered >= len(received), 'received exceeds source frames')
            if same_clock and unmatched: reasons.append('unmatched_subscriber_messages')
            if duplicate or (same_clock and repeated): reasons.append('repeated_identity')
            if policy and same_clock and (rate < policy['min_delivery_hz'] or drop_fraction is None or drop_fraction > policy['max_drop_fraction'] or
                           not gaps or max(gaps) > policy['max_gap_ms']): reasons.append('delivery_threshold')
            if pc.get('thread_cpu_timing', False):
                cpu_samples = [e['thread_cpu_timing'] for e in output_publications if e['thread_cpu_timing']['status'] == 'available']
                cpu_timing = dict(status='enabled', samples=len(cpu_samples), failures=len(output_publications)-len(cpu_samples),
                    copy_ms=_percentiles([(x['copy_end_ns']-x['copy_begin_ns'])/1e6 for x in cpu_samples]),
                    publish_ms=_percentiles([(x['publish_end_ns']-x['publish_begin_ns'])/1e6 for x in cpu_samples]))
            else: cpu_timing = dict(status='not_enabled', samples=0, failures=0, copy_ms=None, publish_ms=None)
            outputs[output]=dict(thread_cpu_timing=cpu_timing, published=len(output_publications),unique_publication_stamps=len(publications),unique_received=len(received) if same_clock else None,duplicate_sources=duplicate,
                repeated_arrivals=repeated if same_clock else None,unmatched=unmatched if same_clock else None,captured_source_frames=captured,
                missing_source_frames=gaps_in_source,offered_source_frames=offered,drop_fraction=drop_fraction,
                delivery_hz=rate,arrival_gap_ms=_percentiles(gaps),observation_age_ms=None,
                observation_age_status='unverified_clock_mapping',
                host_ingress_to_arrival_status=ingress_status,
                host_ingress_to_arrival_ms=_percentiles(ingress_delays) if ingress_status=='available' else None,
                cpu_producer_enqueue_ms=_percentiles([(e['producer_enqueue_end_ns']-e['producer_enqueue_begin_ns'])/1e6 for e in publications.values()]) if all(ENQUEUE_TIMING <= e.keys() for e in publications.values()) else None,
                cpu_export_enqueue_ms=_percentiles([(e['export_enqueue_end_ns']-e['producer_enqueue_end_ns'])/1e6 for e in publications.values()]) if all(ENQUEUE_TIMING <= e.keys() for e in publications.values()) else None,
                cpu_copy_ms=_percentiles([(e['copy_end_ns']-e['copy_begin_ns'])/1e6 for e in publications.values()]),
                cpu_publish_call_ms=_percentiles([(e['publish_end_ns']-e['publish_begin_ns'])/1e6 for e in publications.values()]),
                publish_call_start_to_arrival_ms=_percentiles(delays) if valid_delays else None)
        runs.append(dict(run_id=run_id,qualified=not reasons,reasons=sorted(set(reasons)),duration_s=duration,
            window_start_ns=start,window_end_ns=end,source_mode=pc['source_mode'],clock_id=pc['clock_id'],
            nominal_hz=pc['nominal_hz'],calibration_identity=pc['calibration_identity'],
            implementation=deepcopy(pc['implementation']),effective_estimator=deepcopy(pc['effective_estimator']),view_profile=deepcopy(pc['view_profile']),
            qos=deepcopy(pc['qos']),output_profiles=deepcopy(pc['outputs']),memory=deepcopy(pc['memory']),requested_bytes=requested_bytes,
            queue_at_window_end=pub[-1]['queued'],pending_at_window_end=pub[-1]['pending'],
            capture_queue_high_watermark_since_capture_start=pub[-1]['capture_counters']['queue_high_watermark'],
            observed_capture_queue_peak=max((e['capture_queue'] for e in pframes),default=0),
            observed_pending_peak=max((e['pending'] for e in pframes),default=0),
            capture_counter_deltas={key:pub[-1]['capture_counters'][key]-pc['capture_counters'][key]
                for key in CAPTURE-{'running','queued_frames','queue_high_watermark','pool_capacity','packed_bytes_per_slot'}},
            publisher_counters=deepcopy(pub[-1]['counters']),subscriber_counters=deepcopy(sub[-1]['counters']),
            publisher_drain_ns=pub[-1]['drain_ns'],subscriber_drain_ns=sub[-1]['drain_ns'],outputs=outputs))
    return dict(schema=SUMMARY_SCHEMA,qualified=bool(runs) and all(x['qualified'] for x in runs),runs=runs)
