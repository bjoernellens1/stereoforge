"""Structural guards against accidental Python pixel iteration or enqueue readback."""
import ast
from pathlib import Path

ROOT=Path(__file__).parents[1]/'python'/'stereoforge'


def function(module,name):
    tree=ast.parse((ROOT/module).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)


def fixed_iterator(node):
    return (isinstance(node,(ast.Tuple,ast.List)) or
            isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='range'
            and all(isinstance(a,ast.Constant) and isinstance(a.value,int) and a.value<=8 for a in node.args))


def test_no_python_pixel_loops():
    targets={'calibration.py':['depth_to_raw'],
             'fusion.py':['_source_cloud','_raw_projection','_pixels','_winner_indices','_right_observations','_project','_render_layer','_infinity_layer','synthesize'],
             'temporal.py':['reproject_depth','fuse','adaptive_bounds']}
    for module,names in targets.items():
        for name in names:
            for node in ast.walk(function(module,name)):
                if isinstance(node,(ast.For,ast.ListComp,ast.SetComp,ast.DictComp,ast.GeneratorExp)):
                    iterators=[node.iter] if isinstance(node,ast.For) else [g.iter for g in node.generators]
                    assert all(fixed_iterator(i) for i in iterators), f'{module}:{name} has variable iteration'


def test_enqueue_has_no_readback_or_unconditional_wait():
    fn=function('gpu.py','enqueue')
    parents={child:node for node in ast.walk(fn) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(fn):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute): continue
        assert node.func.attr not in ('wait_for_submit','to_numpy'), 'enqueue must remain asynchronous'
        if node.func.attr=='wait_for_idle':
            parent=parents[node]
            while parent is not fn and not isinstance(parent,ast.If): parent=parents[parent]
            assert isinstance(parent,ast.If) and ast.unparse(parent.test)=='self.shape != (h, w)'


def test_gpu_geometry_and_temporal_have_no_pixel_iteration():
    targets={'gpu_geometry.py':['_rays','render'],
             'gpu_temporal.py':['reproject','fuse']}
    for module,names in targets.items():
        for name in names:
            for node in ast.walk(function(module,name)):
                if not isinstance(node,ast.For): continue
                iterator=node.iter
                # The renderer iterates a fixed field dictionary / output names;
                # temporal fusion iterates fixed prefixes and fields.
                metadata=(isinstance(iterator,ast.Name) and iterator.id=='outputs') or (
                    isinstance(iterator,ast.Call) and isinstance(iterator.func,ast.Attribute)
                    and isinstance(iterator.func.value,ast.Name) and iterator.func.value.id=='values'
                    and iterator.func.attr=='items')
                assert fixed_iterator(iterator) or metadata, f'{module}:{name} has variable iteration'


def test_gpu_idle_waits_are_shape_conditional():
    for module,name in [('gpu_geometry.py','render'),('gpu_temporal.py','_prepare')]:
        fn=function(module,name)
        parents={child:node for node in ast.walk(fn) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(fn):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='wait_for_idle': continue
            parent=parents[node]
            while parent is not fn and not isinstance(parent,ast.If): parent=parents[parent]
            assert isinstance(parent,ast.If)
            assert any(isinstance(n,ast.Attribute) and n.attr=='shape' for n in ast.walk(parent.test))


def test_renderer_ray_cache_is_bounded_and_keeps_stereo_pair():
    from stereoforge.gpu_geometry import VulkanViewSynthesizer
    from stereoforge.types import CameraModel
    renderer=VulkanViewSynthesizer.__new__(VulkanViewSynthesizer)
    renderer._ray_cache={}
    cameras=[CameraModel(8,6,float(10+i),10.,3.5,2.5) for i in range(3)]
    left=renderer._rays(cameras[0]); right=renderer._rays(cameras[1])
    assert renderer._rays(cameras[0]) is left
    assert renderer._rays(cameras[1]) is right
    renderer._rays(cameras[2])
    assert len(renderer._ray_cache)==2
