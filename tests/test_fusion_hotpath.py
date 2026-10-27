"""Host-only guards for the Vulkan compositor submission and metadata caches."""
import ast
from pathlib import Path
import numpy as np
from stereoforge.gpu_geometry import VulkanViewSynthesizer
from stereoforge.types import CameraModel, Calibration
from stereoforge.fusion import VirtualCamera

ROOT=Path(__file__).parents[1]/'python'/'stereoforge'


def renderer():
    obj=VulkanViewSynthesizer.__new__(VulkanViewSynthesizer)
    obj._preview_cache={};obj._ray_cache={}
    return obj


def calibration():
    c=CameraModel(16,12,10.,10.,7.5,5.5)
    pose=np.eye(4);pose[0,3]=.12
    return Calibration(c,c,pose)


def test_preview_cache_bounded_and_translation_independent():
    obj=renderer();cal=calibration();a=VirtualCamera(16,12,90.)
    key,mapping=obj._preview_map(cal,a)
    assert obj._preview_map(cal,a)[1] is mapping
    pose=np.eye(4);pose[0,3]=1.
    translated=VirtualCamera(16,12,90.,camera_to_rig=pose)
    assert obj._preview_map(cal,translated)[0]==key
    assert obj._preview_map(cal,translated)[1] is mapping
    for fov in (100.,110.,120.):obj._preview_map(cal,VirtualCamera(16,12,fov))
    assert len(obj._preview_cache)==2
    assert mapping.dtype==np.uint32 and mapping.shape==(2*16*12,)


def test_compositor_has_one_submission_and_post_wait_readback():
    tree=ast.parse((ROOT/'gpu_geometry.py').read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='render')
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    submits=[n for n in calls if n.func.attr=='submit_command_buffer']
    waits=[n for n in calls if n.func.attr=='wait_for_submit']
    assert len(submits)==len(waits)==1
    assert all(n.lineno>waits[0].lineno for n in calls if n.func.attr=='to_numpy')
    assert all(n.lineno<submits[0].lineno for n in calls if n.func.attr=='dispatch')
    for node in ast.walk(fn):
        if isinstance(node,ast.For):
            assert isinstance(node.iter,(ast.Tuple,ast.List)) or (
                isinstance(node.iter,ast.Name) and node.iter.id=='outputs') or (
                isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Attribute)
                and isinstance(node.iter.func.value,ast.Name) and node.iter.func.value.id=='values'
                and node.iter.func.attr=='items')


def test_preview_upload_guard_and_fixed_histogram_allocation():
    tree=ast.parse((ROOT/'gpu_geometry.py').read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='render')
    parents={child:node for node in ast.walk(fn) for child in ast.iter_child_nodes(node)}
    uploads=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and n.func.attr=='copy_from_numpy' and any(isinstance(a,ast.Name) and a.id=='mapping' for a in n.args)]
    assert len(uploads)==1
    parent=parents[uploads[0]]
    while not isinstance(parent,ast.If):parent=parents[parent]
    assert any(isinstance(n,ast.Attribute) and n.attr=='_uploaded_preview' for n in ast.walk(parent.test))
    allocation=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                    and n.func.id=='buffer' and n.args and isinstance(n.args[0],ast.Constant)
                    and n.args[0].value=='ratio_histogram')
    count=allocation.args[1]
    assert isinstance(count,ast.BinOp) and isinstance(count.op,ast.Mult)
    assert ast.literal_eval(count.left)*ast.literal_eval(count.right)==12288


def test_optional_timestamp_queries_are_fixed_and_after_existing_wait():
    tree=ast.parse((ROOT/'gpu_geometry.py').read_text())
    constructor=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    profile=constructor.args.args.index(next(a for a in constructor.args.args if a.arg=='profile'))
    assert constructor.args.defaults[profile-(len(constructor.args.args)-len(constructor.args.defaults))].value is False
    pool=next(n for n in ast.walk(constructor) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
              and n.func.attr=='create_query_pool')
    query_count=ast.literal_eval(pool.args[1])
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='render')
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    wait=next(n for n in calls if n.func.attr=='wait_for_submit')
    reads=[n for n in calls if n.func.attr=='get_results']
    resets=[n for n in calls if n.func.attr=='reset']
    assert len(reads)==len(resets)==1
    dispatches=[n for n in calls if n.func.attr=='dispatch']
    assert query_count==2*len(dispatches) and query_count<=32
    assert reads[0].lineno>wait.lineno and [ast.literal_eval(a) for a in reads[0].args]==[0,query_count]
    assert resets[0].lineno<wait.lineno


def _cpp_body(source,signature):
    start=source.index('{',source.index(signature));depth=1;end=start+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start+1:end-1]


def test_native_project_encodes_once_without_midstage_host_completion():
    source=(ROOT.parents[1]/'src'/'geometry.cpp').read_text()
    project=_cpp_body(source,'GeometryFrame NativeGeometry::project(')
    assert project.count('createCommandEncoder(')==1 and project.count('e.submit(')==1
    assert not any(call in project for call in ('waitForFences(','waitOnHost(','readBuffer(','readback('))
    stages=['clear','clear_layers','splat_layer_depth','splat_layer_owner',
            'histogram_ratios','reduce_ratios','resolve_layers']
    import re
    assert re.findall(r'e\.dispatch\(encoder,slot,"project\.([^"]+)"',project)==stages
    assert 'e.buffer(slot,"ratio_histogram",3*4096)' in project
    assert 'std::array<Slot,2> slots' in source
    allocator=_cpp_body(source,'rhi::IBuffer* buffer(')
    assert allocator.index('if(!slot.buffers.contains(name))')<allocator.index('createBuffer(')
    # Persistent project-only buffer payload per leased slot (allocator overhead,
    # upload staging, shader programs and any temporal-operation buffers excluded).
    def bytes_per_slot(n,m):return 88*n+48*max(n,m)+24*m+3*4096*4+3*4
    assert bytes_per_slot(320*180,320*180)==9265164
    assert bytes_per_slot(1280*720,1280*720)==147505164
