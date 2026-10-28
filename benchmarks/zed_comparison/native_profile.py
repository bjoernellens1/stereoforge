"""Strict shared identity for an executed independent full-resolution SGM path."""
from copy import deepcopy
from . import schema as s
from .evaluate_temporal import temporal_settings

FIELDS={'algorithm','width','height','disparities','paths','p1','p2','subpixel','lr_tolerance',
        'right_reference_mode','depth_units','temporal','compiler_sha256',
        'stereo_spirv_sha256','depth_spirv_sha256'}


def validate_full_sgm_profile(profile):
    s._keys(profile,FIELDS,'native estimator profile')
    s._require(profile['algorithm']=='independent_sgm' and profile['right_reference_mode']=='independent_sgm' and
               profile['depth_units']=='axial_Z_metres_rectified_rays', 'unsupported estimator semantics')
    for key in ('width','height'): s._integer(profile[key],key,5,8192)
    s._integer(profile['disparities'],'disparities',1,256)
    s._require(profile['width']*profile['height']*profile['disparities'] <= 0xffffffff,'stereo volume overflow')
    s._integer(profile['paths'],'paths'); s._require(profile['paths'] in (4,8),'invalid path count')
    s._integer(profile['p1'],'p1',0,10000);s._integer(profile['p2'],'p2',profile['p1'],10000)
    s._require(type(profile['subpixel']) is bool,'subpixel must be boolean')
    s._number(profile['lr_tolerance'],'lr_tolerance',minimum=0)
    s._require(temporal_settings({'implementation':{'settings':{'temporal':profile['temporal']}}})['state']=='temporal_off',
               'profile requires explicit instantaneous settings')
    s._hex(profile['compiler_sha256'],64,'compiler_sha256')
    for field,prefix in (('stereo_spirv_sha256','daxa-stereo-'),('depth_spirv_sha256','daxa-depth-')):
        hashes=profile[field]
        s._require(type(hashes) is dict and 0<len(hashes)<=128,'missing shader family')
        for name,value in hashes.items():
            s._require(type(name) is str and name.startswith(prefix) and name.endswith('.spv') and
                       '/' not in name and '\\' not in name and len(name)<=256,'invalid shader file name')
            s._hex(value,64,'shader digest')
    return deepcopy(profile)
