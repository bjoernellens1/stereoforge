"""Offline benchmark fixtures must be deterministic and pin actual inputs."""
import importlib.util
from pathlib import Path
import numpy as np


def test_fixture_identity_and_statistics():
    path=Path(__file__).resolve().parents[1]/'benchmarks/benchmark_view_fusion.py'
    spec=importlib.util.spec_from_file_location('view_benchmark',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    first=module.fixture(32,18);second=module.fixture(32,18)
    assert module.input_identity(*first)==module.input_identity(*second)
    assert first[0].left.shape==(18,32,3)
    first[0].left[0,0,0]^=1
    assert module.input_identity(*first)!=module.input_identity(*second)
    stats=module.statistics([1.,2.,3.])
    assert stats['median']==2 and stats['minimum']==1 and stats['maximum']==3
