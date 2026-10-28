"""Contracts for the opt-in shared dual-resolution ROS producer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dual_producer_is_explicit_and_bound_to_coarse_prototype():
    source = (ROOT / 'apps/ros2_depth_publisher.cpp').read_text()
    assert 'experimental-dual-refinement' in source
    assert 'std::make_unique<DaxaDualPreprocess>(maps,coarse_maps,2,context)' in source
    assert 'std::make_unique<DaxaDepthPipeline>(depth_config,context,pair.coarse)' in source
    assert 'pipeline->enqueue_preprocessed(pair.coarse)' in source
    assert 'dual_preprocess->enqueue(f)' in source
    assert 'dual_producer_requested_bytes_unavailable' in source
    assert 'stereoforge_daxa_dual_preprocess' in (ROOT / 'cmake/ROS2.cmake').read_text()
