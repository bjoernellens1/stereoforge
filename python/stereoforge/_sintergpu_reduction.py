"""Compose the opt-in SinterGPU reduction at the existing shader boundary."""
from hashlib import sha256
import os
from pathlib import Path

_VERSION = '0.2.11'
_HELPER_SHA256 = 'b963fc6f1cad8365068c60579f57b55c73c1ad3c3a696a1d3573b0cf627d9388'


def reduction_source(path):
    original = Path(path).read_bytes()
    enabled = os.environ.get('SINTERGPU_SHARED_REDUCTION') == '1'
    if not enabled:
        return original.decode('utf-8'), {'source_sha256': sha256(original).hexdigest(), 'shared_reduction': False}
    import sintergpu
    if sintergpu.__version__ != _VERSION:
        raise RuntimeError(f'SinterGPU {_VERSION} required for shared reduction')
    helper = sintergpu.module_path('common.block_reduce').read_bytes()
    if sha256(helper).hexdigest() != _HELPER_SHA256:
        raise RuntimeError('SinterGPU block reduction source hash mismatch')
    source = b'#define SINTERGPU_USE_SHARED_REDUCTION 1\n' + helper + b'\n' + original
    return source.decode('utf-8'), {'source_sha256': sha256(source).hexdigest(), 'shared_reduction': True,
                                    'shared_reduction_sha256': _HELPER_SHA256}
