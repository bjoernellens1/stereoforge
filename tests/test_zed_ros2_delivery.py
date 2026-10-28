import copy
import pytest
from benchmarks.zed_comparison.ros2_delivery import summarize_delivery

SCHEMA = 'stereoforge.ros_delivery_event.v1'
COUNTERS = dict(read=0, admitted=0, ring_drops=0, identity_rejects=0,
                published_depth=0, published_rgb=0, received_depth=0,
                received_rgb=0, timeouts=0)
CAPTURE = dict(source_records=0, source_frames=0, delivered_frames=0,
               queue_drops=0, no_slot_drops=0, source_frame_gaps=0,
               repeated_frames=0, source_timeouts=0, invalid_frames=0,
               backpressure_waits=0, backpressure_wait_ns=0, queued_frames=0,
               queue_high_watermark=0, pool_capacity=6, packed_bytes_per_slot=32,
               running=True)
OUTPUTS = {'depth': dict(topic='/depth', width=2, height=1, step=8,
                         encoding='32FC1', frame_id='left_optical')}
QOS = dict(history='keep_last', depth=1, reliability='best_effort', durability='volatile')
POLICY = dict(schema='stereoforge.ros_delivery_policy.v1', frozen=True,
              min_duration_s=600, target_hz=60, min_delivery_hz=59.9,
              max_drop_fraction=1, max_gap_ms=1000000, max_frame_age_ms=1000000,
              max_clock_error_ns=1000000, max_queue_growth=0, max_requested_bytes=10000)


def ev(kind, role, at, **fields):
    return dict(schema=SCHEMA, kind=kind, run_id='run', role=role, at_ns=at, **fields)


def traces(seconds=600, repeated=False):
    start = 1_000_000_000
    end = start + seconds * 1_000_000_000
    mapping = dict(source_clock='cam', offset_ns=1_000_000_000,
                   valid_from_ns=1, valid_until_ns=2_000_000_000_000, max_error_ns=10)
    cfg = dict(clock_id='12345678-1234-1234-1234-123456789abc:monotonic:4026531834', source_mode='camera', nominal_hz=60, slots=2,
               capture_queue_capacity=2, outputs=copy.deepcopy(OUTPUTS), qos=copy.deepcopy(QOS),
               implementation={'status':'unavailable','reason':'not measured'},
               calibration_identity='a'*64, effective_estimator={},
               view_profile={'status':'unavailable','reason':'not measured'},
               clock_mapping=mapping, capture_counters=copy.deepcopy(CAPTURE),
               memory=dict(producer_requested_bytes=100, export_requested_bytes=100,
                           capture_requested_bytes=100, trace_capacity=20,
                           trace_record_bytes=16, ros_requested_bytes=8, export_memory_type_index=5,
                           export_memory_property_flags=14, export_host_cached=True, storage_accounting_complete=True))
    subcfg = dict(clock_id='12345678-1234-1234-1234-123456789abc:monotonic:4026531834', nominal_hz=60, outputs=copy.deepcopy(OUTPUTS), qos=copy.deepcopy(QOS))
    pub = [ev('start','publisher',start,config=cfg)]
    sub = [ev('start','subscriber',start,config=subcfg)]
    for i in range(2):
        at = start+100+i*100
        source = 1 if repeated else i+1
        capture = i+1
        pub.append(ev('published','publisher',at+5,output='depth',source_id=source,
                      capture_ns=capture,capture_clock='cam',host_ingress_ns=at,
                      mapped_capture_ns=capture+1_000_000_000,mapping_error_ns=10,
                      producer_ticket=i*2+1,copy_ticket=i*2+2,
                      producer_complete_observed_ns=at,copy_complete_observed_ns=at,
                      copy_begin_ns=at,copy_end_ns=at+1,publish_begin_ns=at+2,
                      publish_end_ns=at+5,capture_queue=0,pending=1,
                      freshness='measured_single_source'))
        sub.append(ev('received','subscriber',at+6,output='depth',
                      mapped_capture_ns=capture+1_000_000_000,arrival_system_ns=at+6,
                      width=2,height=1,step=8,encoding='32FC1',frame_id='left_optical',payload_bytes=8))
    pc = dict(COUNTERS, read=2, admitted=2, published_depth=2)
    sc = dict(COUNTERS, received_depth=2)
    cap = dict(CAPTURE,source_records=2,source_frames=2,delivered_frames=2)
    pub.append(ev('stop','publisher',end+10,window_end_ns=end,reason='limit',
                  counters=pc,capture_counters=cap,queued=0,pending=0,trace_drops=0,drain_ns=10))
    sub.append(ev('stop','subscriber',end+10,window_end_ns=end,reason='limit',
                  counters=sc,capture_counters=None,queued=0,pending=0,trace_drops=0,drain_ns=10))
    return pub+sub


def test_short_run_and_missing_receiver_unqualified():
    assert not summarize_delivery(traces(599),policy=POLICY)['qualified']
    assert not summarize_delivery(traces()[:4],policy=POLICY)['qualified']


def test_repeated_depth_not_unique():
    result=summarize_delivery(traces(repeated=True),policy=POLICY)
    assert not result['qualified']
    assert result['runs'][0]['outputs']['depth']['unique_received'] == 1


def test_bad_payload_and_queue_overflow_rejected():
    records=traces(); records[-2]['payload_bytes']=7
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)
    records=traces(); records[1]['pending']=3
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)


def test_unknown_or_unfrozen_policy_unqualified():
    assert not summarize_delivery(traces())['qualified']
    policy=copy.deepcopy(POLICY); policy['frozen']=False
    assert not summarize_delivery(traces(),policy=policy)['qualified']


def test_unmatched_and_forged_counters_do_not_qualify():
    records=traces(); records[-2]['mapped_capture_ns'] += 5
    result=summarize_delivery(records,policy=POLICY)
    assert 'unmatched_subscriber_messages' in result['runs'][0]['reasons']
    records=traces(); records[3]['counters']['published_depth']=99
    result=summarize_delivery(records,policy=POLICY)
    assert 'publisher_counter_mismatch' in result['runs'][0]['reasons']


def test_clock_mapping_and_unverified_age_cannot_promote():
    records=traces(); records[1]['mapped_capture_ns'] += 1
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)
    records=traces(); records[-2]['arrival_system_ns']=1
    result=summarize_delivery(records,policy=POLICY)
    assert 'camera_clock_mapping_unverified' in result['runs'][0]['reasons']
    assert result['runs'][0]['outputs']['depth']['observation_age_ms'] is None


def test_role_clock_and_replay_cannot_qualify():
    records=traces(); records[4]['config']['clock_id']='abcdefab-1234-1234-1234-123456789abc:monotonic:4026531834'
    assert 'incompatible_host_clocks' in summarize_delivery(records,policy=POLICY)['runs'][0]['reasons']
    records=traces(); records[0]['config']['source_mode']='replay'
    assert 'not_live_camera' in summarize_delivery(records,policy=POLICY)['runs'][0]['reasons']


def test_full_synthetic_delivery_cannot_verify_claimed_camera_clock():
    base=traces()
    pub=[base[0]]; sub=[base[4]]
    pub[0]['config']['memory']['trace_capacity']=36002
    first=1_000_000_000
    for i in range(36000):
        at=first+100+i*1_000_000_000//60
        capture=at-1_000_000_000-10
        pub.append(ev('published','publisher',at+5,output='depth',source_id=i,capture_ns=capture,
            capture_clock='cam',host_ingress_ns=at,mapped_capture_ns=capture+1_000_000_000,
            mapping_error_ns=10,producer_ticket=i*2+1,copy_ticket=i*2+2,
            producer_complete_observed_ns=at,copy_complete_observed_ns=at,copy_begin_ns=at,
            copy_end_ns=at+1,publish_begin_ns=at+2,publish_end_ns=at+5,
            capture_queue=0,pending=1,freshness='measured_single_source'))
        sub.append(ev('received','subscriber',at+6,output='depth',mapped_capture_ns=capture+1_000_000_000,
            arrival_system_ns=at+6,width=2,height=1,step=8,encoding='32FC1',frame_id='left_optical',payload_bytes=8))
    pubstop=copy.deepcopy(base[3]); substop=copy.deepcopy(base[7])
    pubstop['counters']=dict(COUNTERS,read=36000,admitted=36000,published_depth=36000)
    pubstop['capture_counters']=dict(CAPTURE,source_records=36000,source_frames=36000,delivered_frames=36000)
    substop['counters']=dict(COUNTERS,received_depth=36000)
    pub.append(pubstop); sub.append(substop)
    policy=dict(POLICY,max_requested_bytes=1_000_000)
    result=summarize_delivery(pub+sub,policy=policy)
    assert not result['qualified']
    assert result['runs'][0]['reasons']==['camera_clock_mapping_unverified']
    assert result['runs'][0]['outputs']['depth']['unique_received']==36000
    assert result['runs'][0]['outputs']['depth']['delivery_hz']==60
    assert result['runs'][0]['outputs']['depth']['observation_age_ms'] is None
    assert result['runs'][0]['outputs']['depth']['observation_age_status']=='unverified_clock_mapping'
    assert 'quality_qualified' not in result


def test_zero_error_clock_claim_is_not_calibration_evidence():
    records=traces()
    records[0]['config']['clock_mapping']['max_error_ns']=0
    for record in records:
        if record['kind']=='published': record['mapping_error_ns']=0
    result=summarize_delivery(records,policy=POLICY)
    assert not result['qualified']
    assert 'camera_clock_mapping_unverified' in result['runs'][0]['reasons']
    assert result['runs'][0]['outputs']['depth']['observation_age_ms'] is None
    assert result['runs'][0]['outputs']['depth']['publish_call_start_to_arrival_ms'] is not None


def test_source_gaps_remain_in_denominator_and_role_counters_are_checked():
    records=traces(); records[3]['capture_counters']['source_frame_gaps']=2
    row=summarize_delivery(records,policy=POLICY)['runs'][0]['outputs']['depth']
    assert row['captured_source_frames']==2 and row['missing_source_frames']==2
    assert row['offered_source_frames']==4 and row['drop_fraction']==0.5
    records=traces(); records[-1]['counters']['read']=1
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)


def test_unknown_boot_clock_and_ros_stamp_overflow_rejected():
    records=traces(); records[0]['config']['clock_id']='unknown'
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)
    records=traces(); records[-2]['mapped_capture_ns']=2_147_483_648_000_000_000
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)


def test_queue_growth_rejects_sustained():
    records=traces(); records[3]['pending']=2
    result=summarize_delivery(records,policy=POLICY)
    assert 'queue_growth' in result['runs'][0]['reasons']


def test_export_memory_flags_must_report_real_coherent_layout():
    records=traces(); records[0]['config']['memory']['export_memory_property_flags']=2
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)
    records=traces(); records[0]['config']['memory']['export_host_cached']=False
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)


def test_incomplete_requested_storage_accounting_is_unqualified():
    records=traces(); records[0]['config']['memory']['storage_accounting_complete']=False
    assert 'requested_storage_accounting_incomplete' in summarize_delivery(records,policy=POLICY)['runs'][0]['reasons']


def test_exact_duplicate_capture_stamp_is_counted_once_and_unqualified():
    records=traces()
    records[2]['source_id']=records[1]['source_id']
    records[2]['capture_ns']=records[1]['capture_ns']
    records[2]['mapped_capture_ns']=records[1]['mapped_capture_ns']
    records[-2]['mapped_capture_ns']=records[-3]['mapped_capture_ns']
    records[3]['capture_counters']['source_frames']=1
    records[3]['capture_counters']['repeated_frames']=1
    result=summarize_delivery(records,policy=POLICY)
    row=result['runs'][0]['outputs']['depth']
    assert not result['qualified'] and row['published']==2
    assert row['unique_received']==1 and row['duplicate_sources']==1 and row['repeated_arrivals']==1


def test_ros_units_qos_and_frame_identity():
    records=traces(); records[4]['config']['qos']=dict(QOS,reliability='reliable')
    assert 'profile_mismatch' in summarize_delivery(records,policy=POLICY)['runs'][0]['reasons']
    records=traces(); records[-2]['frame_id']='wrong_frame'
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)
    records=traces(); records[0]['config']['outputs']['depth']['encoding']='16UC1'
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)


def test_metadata_is_bounded_and_result_does_not_alias_input():
    records=traces(); result=summarize_delivery(records,policy=POLICY)
    records[0]['config']['memory']['producer_requested_bytes']=999
    records[3]['counters']['read']=999
    assert result['runs'][0]['memory']['producer_requested_bytes']==100
    assert result['runs'][0]['publisher_counters']['read']==2
    records=traces(); records[0]['config']['effective_estimator']={'bad':float('nan')}
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)
    records=traces(); records[0]['config']['implementation']={'status':'available','value':{'x':'a'*4097}}
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)


def test_enqueue_timing_is_reported_as_host_cpu_cost_and_old_traces_remain_unknown():
    old = summarize_delivery(traces(), policy=POLICY)['runs'][0]['outputs']['depth']
    assert old['cpu_producer_enqueue_ms'] is None and old['cpu_export_enqueue_ms'] is None
    records = traces()
    for event in records:
        if event['kind'] == 'published':
            start = event['host_ingress_ns']
            event.update(producer_enqueue_begin_ns=start, producer_enqueue_end_ns=start + 2,
                         export_enqueue_end_ns=start + 5,
                         producer_complete_observed_ns=start + 5, copy_complete_observed_ns=start + 5,
                         copy_begin_ns=start + 5, copy_end_ns=start + 6,
                         publish_begin_ns=start + 7, publish_end_ns=start + 10, at_ns=start + 10)
    for event in records:
        if event['kind'] == 'received': event['at_ns'] += 5
    output = summarize_delivery(records, policy=POLICY)['runs'][0]['outputs']['depth']
    assert output['cpu_producer_enqueue_ms']['p99'] == pytest.approx(2e-6)
    assert output['cpu_export_enqueue_ms']['p99'] == pytest.approx(3e-6)
    del records[1]['producer_enqueue_begin_ns']
    with pytest.raises(ValueError, match='incomplete enqueue timing'):
        summarize_delivery(records, policy=POLICY)


@pytest.mark.parametrize('field, offset', [('producer_enqueue_begin_ns', -1),
    ('producer_enqueue_end_ns', 3), ('export_enqueue_end_ns', 6)])
def test_invalid_enqueue_order_rejected(field, offset):
    records = traces(); event = records[1]; start = event['host_ingress_ns']
    event.update(producer_enqueue_begin_ns=start, producer_enqueue_end_ns=start + 1,
                 export_enqueue_end_ns=start + 2, producer_complete_observed_ns=start + 2,
                 copy_complete_observed_ns=start + 2, copy_begin_ns=start + 2,
                 copy_end_ns=start + 3, publish_begin_ns=start + 4,
                 publish_end_ns=start + 5, at_ns=start + 5)
    event[field] = start + offset
    with pytest.raises(ValueError, match='invalid enqueue order'):
        summarize_delivery(records, policy=POLICY)


def test_partial_run_enqueue_measurements_are_not_summarized():
    records = traces(); event = records[1]; start = event['host_ingress_ns']
    event.update(producer_enqueue_begin_ns=start, producer_enqueue_end_ns=start,
                 export_enqueue_end_ns=start)
    output = summarize_delivery(records, policy=POLICY)['runs'][0]['outputs']['depth']
    assert output['cpu_producer_enqueue_ms'] is None and output['cpu_export_enqueue_ms'] is None


def test_optional_thread_cpu_timing_and_failed_samples():
    records=traces(); records[0]['config']['thread_cpu_timing']=True
    records[1]['thread_cpu_timing']=dict(status='available', copy_begin_ns=10, copy_end_ns=20,
        publish_begin_ns=21, publish_end_ns=41)
    records[2]['thread_cpu_timing']=dict(status='unavailable', reason='clock_read_failed')
    output=summarize_delivery(records,policy=POLICY)['runs'][0]['outputs']['depth']
    assert output['thread_cpu_timing']['samples']==1
    assert output['thread_cpu_timing']['failures']==1
    assert output['thread_cpu_timing']['copy_ms']['p50']==pytest.approx(10/1e6)
    assert output['thread_cpu_timing']['publish_ms']['p50']==pytest.approx(20/1e6)
    assert summarize_delivery(traces(),policy=POLICY)['runs'][0]['outputs']['depth']['thread_cpu_timing']['status']=='not_enabled'


def test_thread_cpu_timing_rejects_partial_undeclared_and_bad_order():
    records=traces(); records[0]['config']['thread_cpu_timing']=True
    records[1]['thread_cpu_timing']=dict(status='available',copy_begin_ns=3,copy_end_ns=2,publish_begin_ns=4,publish_end_ns=5)
    records[2]['thread_cpu_timing']=dict(status='unavailable',reason='clock_read_failed')
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)
    records[1]['thread_cpu_timing']['copy_end_ns']=4
    records[1]['thread_cpu_timing']['publish_begin_ns']=5
    records[1]['thread_cpu_timing']['publish_end_ns']=6
    del records[2]['thread_cpu_timing']
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)
    records[0]['config']['thread_cpu_timing']=False
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)


@pytest.mark.parametrize('invalid', [
    {'status':'available','copy_begin_ns':1,'copy_end_ns':2,'publish_begin_ns':3},
    {'status':'available','copy_begin_ns':1,'copy_end_ns':2,'publish_begin_ns':3,'publish_end_ns':True},
    {'status':'unavailable','reason':'unknown'},
    {'status':'unavailable','reason':'clock_read_failed','copy_begin_ns':1},
    {'status':'other'}, None,
])
def test_malformed_thread_cpu_samples_rejected(invalid):
    records=traces(); records[0]['config']['thread_cpu_timing']=True
    records[1]['thread_cpu_timing']=invalid
    records[2]['thread_cpu_timing']={'status':'unavailable','reason':'clock_read_failed'}
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)


def test_thread_cpu_declaration_is_boolean():
    records=traces(); records[0]['config']['thread_cpu_timing']=1
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)


def test_host_ingress_to_arrival_only_for_live_fresh_matching_sources():
    records = traces()
    records[2]['freshness'] = 'historical'
    run = summarize_delivery(records, policy=POLICY)['runs'][0]
    row = run['outputs']['depth']
    assert row['host_ingress_to_arrival_status'] == 'available'
    assert row['host_ingress_to_arrival_ms']['p99'] == pytest.approx(6e-6)
    assert row['unique_received'] == 1
    assert row['observation_age_ms'] is None
    assert 'camera_clock_mapping_unverified' in run['reasons']


def test_host_ingress_latency_unavailable_for_replay_or_clock_mismatch():
    records = traces(); records[0]['config']['source_mode'] = 'replay'
    row = summarize_delivery(records, policy=POLICY)['runs'][0]['outputs']['depth']
    assert row['host_ingress_to_arrival_ms'] is None
    assert row['host_ingress_to_arrival_status'] == 'not_live_camera'
    records = traces(); records[4]['config']['clock_id'] = 'abcdefab-1234-1234-1234-123456789abc:monotonic:4026531834'
    row = summarize_delivery(records, policy=POLICY)['runs'][0]['outputs']['depth']
    assert row['host_ingress_to_arrival_ms'] is None
    assert row['host_ingress_to_arrival_status'] == 'incompatible_host_clocks'


def test_negative_host_ingress_to_arrival_is_detected():
    records = traces(); records[5]['at_ns'] = records[1]['host_ingress_ns'] - 1
    run = summarize_delivery(records, policy=POLICY)['runs'][0]
    row = run['outputs']['depth']
    assert 'arrival_before_host_ingress' in run['reasons']
    assert row['host_ingress_to_arrival_ms'] is None
    assert row['host_ingress_to_arrival_status'] == 'negative_interval'


def test_host_ingress_latency_requires_fresh_matched_observation():
    records = traces()
    for event in records:
        if event['kind'] == 'published': event['freshness'] = 'unknown'
    row = summarize_delivery(records, policy=POLICY)['runs'][0]['outputs']['depth']
    assert row['host_ingress_to_arrival_ms'] is None
    assert row['host_ingress_to_arrival_status'] == 'no_matched_fresh_observations'


def test_incompatible_clocks_do_not_compare_windows_or_report_rates():
    records = traces()
    offset = 2_000_000_000_000
    for event in records[4:]:
        event['at_ns'] += offset
        if event['kind'] == 'start':
            event['config']['clock_id'] = 'abcdefab-1234-1234-1234-123456789abc:monotonic:4026531834'
        if event['kind'] == 'stop': event['window_end_ns'] += offset
    run = summarize_delivery(records, policy=POLICY)['runs'][0]
    assert not run['qualified'] and 'incompatible_host_clocks' in run['reasons']
    assert run['duration_s'] is None and run['window_start_ns'] is None and run['window_end_ns'] is None
    row = run['outputs']['depth']
    assert row['delivery_hz'] is None and row['arrival_gap_ms'] is None
    assert row['publish_call_start_to_arrival_ms'] is None
    assert row['host_ingress_to_arrival_ms'] is None
    assert row['host_ingress_to_arrival_status'] == 'incompatible_host_clocks'


def test_host_ingress_latency_rejects_conflicting_or_repeated_source_identity():
    records = traces()
    records[2]['capture_ns'] = records[1]['capture_ns']
    records[2]['mapped_capture_ns'] = records[1]['mapped_capture_ns']
    run = summarize_delivery(records, policy=POLICY)['runs'][0]
    assert 'conflicting_publication_stamp' in run['reasons']
    row = run['outputs']['depth']
    assert row['host_ingress_to_arrival_ms'] is None
    assert row['host_ingress_to_arrival_status'] == 'ambiguous_source_identity'
    records = traces(repeated=True)
    row = summarize_delivery(records, policy=POLICY)['runs'][0]['outputs']['depth']
    assert row['host_ingress_to_arrival_ms'] is None
    assert row['host_ingress_to_arrival_status'] == 'ambiguous_source_identity'
