from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_trace_uses_fixed_bounded_storage_without_per_event_allocations(tmp_path):
    exe=tmp_path/'trace'
    subprocess.run(['c++','-std=c++20','-O2','-DNDEBUG','-Wall','-Wextra','-Werror',
                    '-I',str(ROOT/'include'),str(ROOT/'tests/ros_delivery_trace.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_json_serialization_preserves_actual_received_values(tmp_path):
    import json
    import os
    exe=tmp_path/'json-trace'
    include=os.environ.get('STEREOFORGE_JSON_INCLUDE',str(ROOT/'build/daxa/_deps/sf_native_json-src'))
    subprocess.run(['c++','-std=c++20','-O2','-DNDEBUG','-Wall','-Wextra','-Werror',
                    '-I',str(ROOT/'include'),'-I',include,str(ROOT/'tests/ros_delivery_json.cpp'),'-o',str(exe)],check=True)
    event,published=json.loads(subprocess.check_output([str(exe)],text=True))
    assert event['mapped_capture_ns']==123 and event['arrival_system_ns']==300
    assert event['width']==2 and event['height']==1 and event['payload_bytes']==8
    from benchmarks.zed_comparison.ros2_delivery import BASE, PUBLISHED, RECEIVED, ENQUEUE_TIMING
    assert set(event)==BASE | RECEIVED
    assert set(published)==BASE | PUBLISHED | ENQUEUE_TIMING
    assert published['producer_enqueue_begin_ns']==124 and published['producer_enqueue_end_ns']==125 and published['export_enqueue_end_ns']==126
    assert published['source_id']==42 and published['freshness']=='measured_single_source'
