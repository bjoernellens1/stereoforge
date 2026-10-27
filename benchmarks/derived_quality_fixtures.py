"""Independent integer-forward-warp development geometry, never matcher GT.

These fixtures are algorithm probes, not physical optical renders or held-out data.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def census_validity(width,height):
    y,x=np.indices((height,width));return ((x>=2)&(x+2<width)&(y>=2)&(y+2<height)).astype('<u4').tobytes()


def array(raw,dtype,count):
    if len(raw)!=count*np.dtype(dtype).itemsize:raise ValueError('plane byte extent mismatch')
    return np.frombuffer(raw,dtype)


def forward_warp_fixture(left,left_disparity):
    """Independent integer rectified geometry; max disparity is nearest Z.

    Z-buffer winners define visibility. Unseen Right pixels carry no GT;
    off-image and occluded Left samples are explicitly separate masks.
    """
    left=np.asarray(left);d=np.asarray(left_disparity)
    if left.ndim!=2 or d.shape!=left.shape or left.dtype!=np.uint8 or d.dtype.kind not in 'ui' or np.any(d>0x7fffffff):raise ValueError('uint8 Left and bounded integer disparity surface required')
    h,w=left.shape;right=np.zeros_like(left);rd=np.zeros((h,w),'<u4');source=np.full((h,w),-1,'<i4');visible=np.zeros((h,w),bool);off=np.zeros_like(visible)
    for y in range(h):
        for x in range(w):
            xr=x-int(d[y,x])
            if xr<0 or xr>=w:off[y,x]=True;continue
            if source[y,xr]<0 or d[y,x]>rd[y,xr]:
                source[y,xr]=x;rd[y,xr]=d[y,x];right[y,xr]=left[y,x]
    rv=source>=0
    for y,xr in zip(*np.nonzero(rv)):visible[y,source[y,xr]]=True
    return dict(left=left.copy(),right=right,left_disparity=d.astype('<u4'),right_disparity=rd,right_source_x=source,left_visible=visible,right_visible=rv,left_occluded=~visible&~off,left_off_image=off,right_unseen=~rv)


def development_fixture(kind,width=96,height=32,seed=42):
    """Deterministic development cases, never held-out qualification data.

    Integer forward warps make GT independent of any Census or SGM result.
    Slope is a staircase integer disparity surface, not a smooth optical render.
    """
    if kind not in ('plane','slope','thin-occluder','photometric') or width<24 or height<8:raise ValueError('unknown case or small dimensions')
    # Explicit xorshift generator is stable across NumPy versions.
    state=seed;left=np.empty((height,width),np.uint8)
    for i in range(left.size):
        state=(state^(state<<13))&0xffffffff;state=(state^(state>>17))&0xffffffff;state=(state^(state<<5))&0xffffffff;left.flat[i]=state&255
    d=np.full(left.shape,4,'<u4')
    if kind=='slope':d[:]=2+np.arange(width,dtype=np.uint32)[None,:]*5//width
    if kind=='thin-occluder':
        d[:,width//2:]=2;d[height//4:3*height//4,width//3:2*width//3]=7;d[:,width//2:width//2+2]=9
    case=forward_warp_fixture(left,d)
    if kind=='photometric':case['right']=np.clip(case['right'].astype(np.int16)+17,0,255).astype(np.uint8)
    return case


def synthetic_quality(raw,case,eye='right'):
    """Report coverage/error only where independent geometry supplies GT."""
    if eye not in ('left','right'):raise ValueError('unknown eye')
    h,w=case[eye].shape;n=w*h;gt=case[eye+'_disparity'].ravel();visible=case[eye+'_visible'].ravel();d=array(raw['disparity.f32'],'<f4',n);valid=array(raw['validity.u32'],'<u4',n)!=0
    census=array(census_validity(w,h),'<u4',n)!=0;domain=visible&census;usable=domain&valid&np.isfinite(d)
    error=np.abs(d[usable].astype(float)-gt[usable].astype(float))
    return dict(scope='deterministic_integer_forward_warp_development_GT; no_heldout_or_physical_qualification',promoted=False,visible_census_domain=int(domain.sum()),valid_in_domain=int(usable.sum()),invalid_in_domain=int((domain&~valid).sum()),valid_outside_visible_geometry=int((valid&~visible).sum()),mean_absolute_disparity_error=float(error.mean()) if error.size else None,max_absolute_disparity_error=float(error.max()) if error.size else None)


def regions(case,eye='left'):
    """Masks in the requested eye's pixel coordinates, from independent GT.

    Right surface labels use visible z-buffer winners. Surface boundaries
    include transitions into unseen Right geometry (the stored disparity is0).
    Visibility masks from the other eye are not applied in this coordinate frame.
    """
    if eye not in ('left','right'):raise ValueError('unknown eye')
    h,w=case[eye].shape;y,x=np.indices((h,w));surface=case[eye+'_disparity'];edge=np.zeros((h,w),bool)
    jumps=surface[:,1:]!=surface[:,:-1]
    for offset in (-2,-1,0,1,2):
        sy,sx=np.nonzero(jumps);sx=sx+offset;inside=(sx>=0)&(sx<w);edge[sy[inside],sx[inside]]=True
    result=dict(all=np.ones((h,w),bool),census_border=(x<2)|(x+2>=w)|(y<2)|(y+2>=h),thin_foreground=(surface==9)&(case['right_visible'] if eye=='right' else True),surface_boundary=edge)
    names=('left_visible','left_occluded','left_off_image') if eye=='left' else ('right_visible','right_unseen')
    result.update({name:case[name] for name in names})
    return result


def export_fixture(directory,kind,width=96,height=32,seed=42):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    case=development_fixture(kind,width,height,seed);planes={}
    arrays=dict(case,**{'region_'+k:v for k,v in regions(case,'left').items()})
    for eye in ('left','right'):
        arrays.update({'region_'+eye+'_'+key:value for key,value in regions(case,eye).items()})
    for key,value in arrays.items():
        # Every image/mask/GT uses uint32, except signed source identity.
        dtype='<i4' if key=='right_source_x' else '<u4'
        data=value.astype(dtype).tobytes();filename=key+('.i32' if dtype=='<i4' else '.u32');path=directory/filename;path.write_bytes(data);path.chmod(0o444)
        planes[key]=dict(file=filename,shape=list(value.shape),dtype=dtype,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    manifest=dict(schema='stereoforge.derived_quality_fixture.v1',kind=kind,width=width,height=height,seed=seed,scope='controlled_integer_forward_warp_development; not_physical_render_or_heldout',geometry='xR=xL-d; largest_disparity_wins; equal_depth_first_source_x_wins',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),planes=planes,promoted=False)
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n');return manifest


def paired_region_scores(raw,case,lr_tolerance=1.):
    """LR acceptance versus known visibility and correspondence, by region.

    'false' means the accepted rounded correspondence disagrees with geometric
    winner identity or lies outside visible geometry. No quality pass threshold.
    Report disparity error separately without declaring a tolerance a pass gate.
    """
    h,w=case['left'].shape;n=w*h;y,x=np.indices((h,w));scores={}
    d=[array(row['disparity.f32'],'<f4',n).reshape(h,w) for row in raw]
    valid=[array(row['validity.u32'],'<u4',n).reshape(h,w)!=0 for row in raw]
    for eye in (0,1):
        name='right' if eye else 'left';own=d[eye];other=d[1-eye];finite=np.isfinite(own)
        rounded=np.floor(np.where(finite,x+(own.astype(float) if eye else -own.astype(float))+.5,-1)).astype(np.int64)
        inside=finite&(rounded>=0)&(rounded<w);safe=np.where(inside,rounded,0)
        lr=valid[eye]&inside&valid[1-eye][y,safe]&np.isfinite(other[y,safe])&(np.abs(own-other[y,safe])<=np.float32(lr_tolerance))
        visible=case[name+'_visible'];gt=case[name+'_disparity'];geometric_other=x+(gt.astype(np.int64) if eye else -gt.astype(np.int64))
        # For Left, also require the actual z-buffer winner to be this source.
        consistent=visible&(rounded==geometric_other)
        if eye==0:consistent &= case['right_source_x'][y,safe]==x
        false=lr&~consistent
        eye_scores={}
        for region,mask in regions(case,name).items():
            domain=mask&visible;usable=domain&valid[eye]&finite;err=np.abs(own[usable].astype(float)-gt[usable].astype(float));accepted=lr&mask
            eye_scores[region]=dict(pixels=int(mask.sum()),visible_geometry=int(domain.sum()),valid_visible=int(usable.sum()),lr_accepted=int(accepted.sum()),lr_false_geometric_correspondence=int((false&mask).sum()),mean_absolute_disparity_error=float(err.mean()) if err.size else None,max_absolute_disparity_error=float(err.max()) if err.size else None)
        scores[name]=eye_scores
    return dict(scope='algorithm_reference_development_geometry; no_heldout_or_physical_qualification',promoted=False,lr_tolerance=lr_tolerance,regions=scores)
