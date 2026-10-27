"""Daxa external-resource names are a setup correctness contract, CPU-only."""
from pathlib import Path
import re
import pytest

ROOT=Path(__file__).resolve().parents[1]


def check_external_names(source):
    table=re.search(r'metadata_resource_names\s*\{([^}]+)\}',source)
    assert table, 'external resource name table required'
    names=re.findall(r'"([^"]+)"',table.group(1))
    assert len(names)==12 and len(set(names))==12, 'all twelve external names must be unique'
    assert all(0<len(name)<64 for name in names), 'Daxa names must not truncate into collisions'
    indices=re.findall(r'daxa::ExternalTaskBuffer\(\{[^}]*\.name=metadata_resource_names\[([^]]+)\]',source)
    assert set(indices)=={'i','i+4','i+8','10','11'}, 'each semantic resource group needs disjoint name indices'


def test_metadata_external_names_are_unique_before_graph_registration():
    check_external_names((ROOT/'apps/daxa_variant_benchmark.cpp').read_text())


def test_name_guard_detects_colliding_setup_names():
    source=(ROOT/'apps/daxa_variant_benchmark.cpp').read_text()
    changed=source.replace('"metadata source validity"','"metadata source owner"')
    assert changed!=source
    with pytest.raises(AssertionError,match='unique'):
        check_external_names(changed)
