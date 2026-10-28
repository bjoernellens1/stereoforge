"""Static contracts for the opt-in live refinement integration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_opt_in_refinement_is_explicit_and_wired():
    source = (ROOT / 'apps/ros2_depth_publisher.cpp').read_text()
    assert 'pipeline_mode="full-sgm"' in source
    assert '--coarse-manifest' in source and '--pipeline' in source
    assert 'experimental-refinement' in source
    assert 'refinement_config.borrow_full_frame=true' in source
    assert 'refinement->process(index->depth,index->full)' in source
    assert 'index->depth=pipeline->enqueue(f)' in source
    assert 'index->full=full_preprocess->enqueue(f)' in source
    assert 'experimental_refinement_not_qualified' in source
    assert 'validate_daxa_refinement_geometry(coarse_maps,maps)' in source
    assert 'stereoforge_daxa_refinement' in (ROOT / 'cmake/ROS2.cmake').read_text()
