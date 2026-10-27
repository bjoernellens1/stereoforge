"""Optional portable inference qualification status.

No network is selected or silently substituted. A future model adapter must prove
operator placement and conversion parity before it becomes a usable backend.
"""
import importlib.util

def capabilities():
    return {'required_backend':'vulkan','qualified_models':[],
            'ncnn_installed':importlib.util.find_spec('ncnn') is not None,
            'iree_installed':importlib.util.find_spec('iree') is not None,
            'available':False,'reason':'No portable neural model has been qualified; use classical stereo.'}
