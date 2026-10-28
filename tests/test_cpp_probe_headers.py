"""Header discovery is independent of a locally populated native build."""
import shutil
import subprocess

import pytest
from conftest import resolve_json_include


def test_explicit_real_header_compiles_without_vendor_cache(tmp_path, json_include):
    include = tmp_path / 'explicit'
    shutil.copytree(json_include, include)
    resolved = resolve_json_include(tmp_path / 'clean-checkout', include)
    assert resolved == include
    code = tmp_path / 'probe.cpp'
    code.write_text('#include <json.hpp>\nint main(){nlohmann::json j={{"x",42}};'
                    'return j.at("x").get<int>()==42?0:1;}\n')
    binary = tmp_path / 'probe'
    subprocess.run(['c++', '-std=c++20', '-I', str(resolved), str(code), '-o',
                    str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True)


def test_system_fallback_without_vendor_and_explicit_precedence(tmp_path):
    system = tmp_path / 'system'
    system.mkdir()
    (system / 'json.hpp').touch()
    assert resolve_json_include(tmp_path / 'clean-checkout', system=system) == system
    with pytest.raises(RuntimeError, match='STEREOFORGE_JSON_INCLUDE'):
        resolve_json_include(tmp_path, tmp_path / 'missing-explicit', system)
    with pytest.raises(RuntimeError, match='Missing nlohmann'):
        resolve_json_include(tmp_path, system=tmp_path / 'missing-system')
