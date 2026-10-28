from copy import deepcopy
import pytest
from benchmarks.zed_comparison.native_profile import validate_full_sgm_profile

TEMPORAL={'stabilization':{'status':'available','value':0},'tracking':{'status':'available','value':False},'filling':{'status':'available','value':False}}
PROFILE=dict(algorithm='independent_sgm',width=1280,height=720,disparities=128,paths=4,p1=8,p2=32,subpixel=True,lr_tolerance=1.0,
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
