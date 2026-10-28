"""Unknown producer storage must never become a measured zero or qualify."""
import pytest
from benchmarks.zed_comparison.ros2_delivery import summarize_delivery
from test_zed_ros2_delivery import traces, POLICY


def test_unavailable_producer_storage_is_explicit_and_unqualified():
    records=traces(); mem=records[0]['config']['memory']
    mem.update(producer_requested_bytes=None, producer_requested_bytes_reason='dual_producer_requested_bytes_unavailable', storage_accounting_complete=False)
    run=summarize_delivery(records,policy=POLICY)['runs'][0]
    assert run['requested_bytes'] is None
    assert run['memory']['producer_requested_bytes'] is None
    assert run['memory']['producer_requested_bytes_reason']=='dual_producer_requested_bytes_unavailable'
    assert 'requested_storage_accounting_incomplete' in run['reasons']
    assert 'producer_requested_bytes_unavailable' in run['reasons']
    assert not run['qualified']


@pytest.mark.parametrize('value,reason,complete', [
    (None, None, False), (None, '', False), (None, 'available', False),
    (None, 'unavailable', True), (None, 1, False), (True, 'available', False),
    (-1, 'available', False), (1.0, 'available', False), (100, 'unavailable', False),
    (100, '', False),
])
def test_bad_memory_availability_is_rejected(value,reason,complete):
    records=traces();mem=records[0]['config']['memory'];mem['producer_requested_bytes']=value;mem['storage_accounting_complete']=complete
    if reason is not None: mem['producer_requested_bytes_reason']=reason
    with pytest.raises(ValueError): summarize_delivery(records,policy=POLICY)


def test_numeric_old_and_new_traces_keep_exact_totals():
    records=traces(); run=summarize_delivery(records,policy=POLICY)['runs'][0]
    assert run['requested_bytes']==628
    records[0]['config']['memory']['producer_requested_bytes_reason']='available'
    run=summarize_delivery(records,policy=POLICY)['runs'][0]
    assert run['requested_bytes']==628
    records[0]['config']['memory']['producer_requested_bytes']=99999
    run=summarize_delivery(records,policy=POLICY)['runs'][0]
    assert 'requested_memory_exceeds_policy' in run['reasons']
