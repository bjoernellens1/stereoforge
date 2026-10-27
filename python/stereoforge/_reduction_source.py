"""Experimental source composition; no GPU or optional dependency at import time."""
from hashlib import sha256
import os
from pathlib import Path


def reduction_source(path):
    """Freeze the actual compiled source and dependency identity at setup time."""
    original = Path(path).read_bytes()
    enabled = os.environ.get("SINTERGPU_SHARED_REDUCTION") == "1"
    helper = b""
    if enabled:
        from sintergpu import module_path
        helper = module_path("common.block_reduce").read_bytes()
    source = (b"#define SINTERGPU_USE_SHARED_REDUCTION 1\n" + helper + b"\n" + original) if enabled else original
    identity = dict(source_sha256=sha256(source).hexdigest(),
                    original_source_sha256=sha256(original).hexdigest(),
                    shared_reduction=enabled,
                    shared_reduction_sha256=sha256(helper).hexdigest() if enabled else None)
    return source, helper, identity
