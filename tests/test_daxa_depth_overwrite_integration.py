from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_overwrite_is_explicit_and_sequential_independent_only():
    header = (ROOT / 'include/stereoforge/daxa_depth.hpp').read_text()
    source = (ROOT / 'src/daxa_depth.cpp').read_text()
    assert 'DaxaSgmVariant stereo_variant=DaxaSgmVariant::PackedSpecialized;' in header
    assert 'stereo_variant!=DaxaSgmVariant::PackedSpecializedOverwrite||(stereo_schedule==DaxaDepthStereoSchedule::Sequential&&right_reference_mode==DaxaRightReferenceMode::IndependentSGM)' in source
    assert source.count('new DaxaStereo(c.stereo,pre,c.stereo_variant,c.slots') == 2


def test_canonical_benchmark_exposes_and_records_actual_stereo_variant():
    app = (ROOT / 'apps/daxa_depth_rectified_benchmark.cpp').read_text()
    assert 'key=="--stereo-variant"' in app
    assert 'depth_config.stereo_variant=stereo_variant==' in app
    assert '{"stereo_variant",DaxaStereo::variant_name(depth_config.stereo_variant)}' in app


def test_experimental_variant_cannot_claim_canonical_estimator_identity():
    app = (ROOT / 'apps/daxa_depth_rectified_benchmark.cpp').read_text()
    assert 'if(!refinement&&depth_config.stereo_variant==DaxaSgmVariant::PackedSpecialized)' in app
    assert 'result["effective_estimator"]=nullptr' in app
    assert 'experimental_variant_not_qualified_for_canonical_estimator_identity' in app
