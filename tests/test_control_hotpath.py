"""Static guard: frame acquisition attaches cached metadata without register I/O."""
import ast
import inspect
import textwrap
from stereoforge.capture import ZedCamera
from stereoforge.controls import CameraControl


def test_cached_frame_metadata_has_no_control_polling_or_gpu_wait():
    forbidden={'get_controls','set_controls','read_controls','wait_for_idle','wait_for_submit','synchronize'}
    methods=(ZedCamera.read,ZedCamera.control_snapshot.fget,CameraControl.snapshot.fget)
    for method in methods:
        tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):continue
            assert node.func.attr not in forbidden,(method.__qualname__,node.func.attr)
            if node.func.attr=='read':
                receiver=ast.unparse(node.func.value)
                assert 'camera_control' not in receiver,(method.__qualname__,receiver)
