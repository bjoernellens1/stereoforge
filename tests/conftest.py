"""Real C++ header discovery for CPU-only source probes."""
import os
from pathlib import Path

import pytest


def resolve_json_include(root, explicit=None, system=Path('/usr/include/nlohmann')):
    """Return the directory containing json.hpp, never a mock implementation."""
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        candidates = [Path(root) / 'build/daxa/_deps/sf_native_json-src', Path(system)]
    for candidate in candidates:
        if (candidate / 'json.hpp').is_file():
            return candidate
    raise RuntimeError('Missing nlohmann json.hpp: install nlohmann-json3-dev or set '
                       'STEREOFORGE_JSON_INCLUDE to its header directory; searched: '
                       + ', '.join(map(str, candidates)))


@pytest.fixture
def json_include():
    return resolve_json_include(Path(__file__).resolve().parents[1],
                                os.environ.get('STEREOFORGE_JSON_INCLUDE'))
