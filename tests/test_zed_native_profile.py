from copy import deepcopy
import pytest
from benchmarks.zed_comparison.native_profile import validate_full_sgm_profile

TEMPORAL={'stabilization':{'status':'available','value':0},'tracking':{'status':'available','value':False},'filling':{'status':'available','value':False}}
PROFILE=dict(descriptor_mode='census',stereo_schedule='sequential',algorithm='independent_sgm',width=1280,height=720,disparities=128,paths=4,p1=8,p2=32,subpixel=True,lr_tolerance=1.0,
             right_reference_mode='independent_sgm',depth_units='axial_Z_metres_rectified_rays',temporal=TEMPORAL,compiler_sha256='a'*64,
             stereo_spirv_sha256={'daxa-stereo-census.spv':'b'*64},depth_spirv_sha256={'daxa-depth-metric.spv':'c'*64})


def test_profile_is_checked_and_copied():
    result=validate_full_sgm_profile(PROFILE)
    assert result==PROFILE and result is not PROFILE
    result['temporal']['tracking']['value']=True
    assert PROFILE['temporal']['tracking']['value'] is False


@pytest.mark.parametrize('key,value',[('width',4),('disparities',257),('paths',3),('subpixel',1),('p1',40),
    ('lr_tolerance',float('nan')),('algorithm','derived'),('depth_units','range'),('compiler_sha256','0')])
def test_bad_profiles_rejected(key,value):
    data=deepcopy(PROFILE);data[key]=value
    with pytest.raises(ValueError): validate_full_sgm_profile(data)


def test_historical_or_modified_shader_profiles_do_not_pass():
    data=deepcopy(PROFILE);data['temporal']['tracking']={'status':'unavailable','reason':'unknown'}
    with pytest.raises(ValueError): validate_full_sgm_profile(data)
    data=deepcopy(PROFILE);data['stereo_spirv_sha256']={'../daxa-stereo-census.spv':'b'*64}
    with pytest.raises(ValueError): validate_full_sgm_profile(data)


@pytest.mark.parametrize('field', ['descriptor_mode', 'stereo_schedule'])
def test_historical_incomplete_identity_is_not_inferred(field):
    profile = deepcopy(PROFILE)
    del profile[field]
    with pytest.raises(ValueError):
        validate_full_sgm_profile(profile)


@pytest.mark.parametrize('field,value', [('descriptor_mode','unknown'), ('stereo_schedule','unknown')])
def test_unknown_cost_or_schedule_rejected(field, value):
    profile = deepcopy(PROFILE)
    profile[field] = value
    with pytest.raises(ValueError):
        validate_full_sgm_profile(profile)


@pytest.mark.parametrize('schedule', ['sequential','paired-per-path','layered-per-path',
                                    'shared-census-sequential','hierarchical-1to8-fallback'])
@pytest.mark.parametrize('descriptor', ['census','ortho-hybrid'])
def test_explicit_cost_schedule_identity_preserved(schedule, descriptor):
    profile = deepcopy(PROFILE)
    profile.update(descriptor_mode=descriptor,stereo_schedule=schedule,disparities=64)
    assert validate_full_sgm_profile(profile) == profile


@pytest.mark.parametrize('key,value', [('width',7),('height',9),('disparities',15),('disparities',65)])
def test_hierarchical_identity_requires_native_domain(key, value):
    profile=deepcopy(PROFILE)
    profile.update(stereo_schedule='hierarchical-1to8-fallback',disparities=64)
    profile[key]=value
    with pytest.raises(ValueError):
        validate_full_sgm_profile(profile)


@pytest.mark.parametrize('schedule',['sequential','shared-census-sequential'])
def test_native_d288_profile_is_explicitly_supported(schedule):
    profile=deepcopy(PROFILE)
    profile.update(width=1920,height=1080,disparities=288,stereo_schedule=schedule)
    assert validate_full_sgm_profile(profile)==profile


@pytest.mark.parametrize('schedule',['paired-per-path','layered-per-path','hierarchical-1to8-fallback'])
def test_native_d288_profile_rejects_unsupported_schedule(schedule):
    profile=deepcopy(PROFILE)
    profile.update(disparities=288,stereo_schedule=schedule)
    with pytest.raises(ValueError):validate_full_sgm_profile(profile)


@pytest.mark.parametrize('disparities',[257,287,289])
def test_native_wide_profile_rejects_uncompiled_disparity_count(disparities):
    profile=deepcopy(PROFILE);profile['disparities']=disparities
    with pytest.raises(ValueError):validate_full_sgm_profile(profile)
