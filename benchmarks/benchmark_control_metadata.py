"""Bounded host-only cached control metadata benchmark and hot-path guards.

Run: PYTHONPATH=python .venv/bin/python benchmarks/benchmark_control_metadata.py
No physical camera, image processing, disk I/O, or GPU execution is performed.
"""
import ast
from copy import deepcopy
import inspect
import json
from statistics import median
from time import perf_counter_ns
import textwrap
from stereoforge.controls import CameraControl
from stereoforge.capture import ZedCamera
from stereoforge.recording import _json


class NoHardware:
    def __init__(self):self.read_count=0
    def get_controls(self):
        self.read_count+=1
        return dict(exposure_left=100,exposure_right=100,gain_left=10,gain_right=10,
                    auto_exposure_gain=False,white_balance=4500,auto_white_balance=False,
                    roi_left=[0,0,1280,720],roi_right=[0,0,1280,720])


def static_guard():
    """Fail if cached per-frame metadata access gains register I/O or GPU waits."""
    forbidden={'get_controls','set_controls','wait_for_idle','wait_for_submit','synchronize','read_controls'}
    for method in (ZedCamera.read,ZedCamera.control_snapshot.fget,CameraControl.snapshot.fget):
        tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
        called={node.func.attr for node in ast.walk(tree)
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        assert not called & forbidden,(method.__qualname__,called & forbidden)
        assert not any(isinstance(node,(ast.For,ast.While,ast.ListComp,ast.DictComp)) for node in ast.walk(tree)),method.__qualname__


def measure(call,repetitions=1000,batches=7):
    for _ in range(20):call()
    samples=[]
    for _ in range(batches):
        start=perf_counter_ns()
        for _ in range(repetitions):call()
        samples.append((perf_counter_ns()-start)/repetitions/1000)
    return dict(median_us=median(samples),min_us=min(samples),max_us=max(samples),
                samples_us=samples,repetitions_per_batch=repetitions)


def main():
    static_guard()
    native=NoHardware();control=CameraControl(native,1280,720);control.read()
    snapshot=control.snapshot
    results=dict(metadata_json_bytes=len(_json(snapshot).encode()),
        cached_snapshot=measure(lambda:control.snapshot),
        rectifier_metadata_copy=measure(lambda:deepcopy(snapshot)),
        recorder_json_snapshot=measure(lambda:json.loads(_json(snapshot))))
    assert native.read_count==1,'cached snapshots polled camera registers'
    results.update(native_control_reads=native.read_count,static_guard='passed',
        scope='host cached metadata only; excludes USB, image conversion, rectification, recording I/O and GPU')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
