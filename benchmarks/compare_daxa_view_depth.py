#!/usr/bin/env python3
"""Independent calibrated known-point diagnostic oracle; no physical qualification.

Synthetic observation intake qualifies geometry/readers/compositor only. Actual
stereo, resident performance and physical calibration have separate gates.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import numpy as np
import compare_daxa_depth as depth
import compare_daxa_preprocess as preprocess
import compare_daxa_stereo as base
import compare_daxa_view as legacy
from compare_native_resident import PLANES
ROOT=Path(__file__).resolve().parents[1]
UINT_MAX=0xffffffff
ASSOCIATION_PRODUCER='independent_calibrated_pinhole_distortion_geometry'
ASSOCIATION_SAMPLING='OpenCV_INTER_LINEAR_5bit_nearest_even'


def geometry_fixture(profile='default',virtual_pose='identity'):
    rotation=[[1.,0.,0.],[0.,.8,-.6],[0.,.6,.8]]
    pose=np.eye(4);pose[0,3]=.625
    camera=dict(width=16,height=12,fx=4.,fy=4.,cx=7.5,cy=5.5,distortion=[.01,0.,0.,0.,0.])
    rect=[dict(width=7,height=5,fx=4.,fy=4.,cx=3.,cy=2.,distortion=[]),dict(width=7,height=5,fx=4.,fy=4.,cx=4.,cy=2.,distortion=[])]
    if profile not in ('default','shared-raw-cell') or virtual_pose not in ('identity','rotate-z90'):raise ValueError('unknown rational geometry profile')
    if profile=='shared-raw-cell':
        for c in rect:c['fx']=8.
    virtual=np.eye(4) if virtual_pose=='identity' else np.array([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    return dict(schema='stereoforge.view_depth.geometry.v1',raw_size=[16,12],rect_size=[7,5],original=[dict(camera),dict(camera)],rectified=rect,R1=rotation,R2=rotation,right_to_left=pose.tolist(),view=dict(width=13,height=11,fx=8. if profile=='shared-raw-cell' else 4.,fy=4.,cx=6.,cy=4.,hfov_radians=float(np.float32(math.pi/2)),projection='perspective',virtual_from_original_left=virtual.tolist()),witnesses=[dict(original_left_point=[0.,.75,1.]),dict(original_left_point=[-.3125,.75,1.]),dict(original_left_point=[.3125,.75,1.])])


def rectified_observation(fixture,world,eye):
    point=np.r_[world,1.]
    if eye:point=np.linalg.solve(np.asarray(fixture['right_to_left']),point)
    point=np.asarray(fixture['R2' if eye else 'R1'])@point[:3];c=fixture['rectified'][eye]
    return (c['fx']*point[0]/point[2]+c['cx'],c['fy']*point[1]/point[2]+c['cy']),float(point[2])


def reconstruct_world(fixture,pixel,z,eye):
    c=fixture['rectified'][eye];q=np.array([(pixel[0]-c['cx'])/c['fx'],(pixel[1]-c['cy'])/c['fy'],1.])
    point=np.asarray(fixture['R2' if eye else 'R1']).T@(q*z)
    return (np.asarray(fixture['right_to_left'])@np.r_[point,1.])[:3] if eye else point


def distorted_project(point,camera):
    x,y=point[0]/point[2],point[1]/point[2];coeff=list(camera['distortion'])+[0.]*8;k1,k2,p1,p2,k3,k4,k5,k6=coeff[:8];r2=x*x+y*y;radial=(1+k1*r2+k2*r2*r2+k3*r2**3)/(1+k4*r2+k5*r2*r2+k6*r2**3)
    return camera['fx']*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+camera['cx'],camera['fy']*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+camera['cy']


def geometry_maps(fixture):
    w,h=fixture['rect_size'];maps={}
    for eye,name in enumerate(('left','right')):
        c=fixture['rectified'][eye];rotation=np.asarray(fixture['R1' if eye==0 else 'R2']);mx=np.empty((h,w),'<f4');my=np.empty_like(mx)
        for y in range(h):
            for x in range(w):
                ray=np.array([(x-c['cx'])/c['fx'],(y-c['cy'])/c['fy'],1.]);point=np.linalg.solve(rotation,ray);mx[y,x],my[y,x]=distorted_project(point,fixture['original'][eye])
        maps[name+'_x']=mx;maps[name+'_y']=my
    return maps


def validate_map_association(fixture,maps):
    expected=geometry_maps(fixture)
    if set(maps)!=set(expected):raise ValueError('map/model association plane mismatch')
    for name,value in expected.items():
        actual=np.asarray(maps[name])
        if actual.shape!=value.shape or actual.dtype!=np.float32 or not np.isfinite(actual).all() or not np.array_equal(np.rint(actual.astype(float)*32),np.rint(value.astype(float)*32)):raise ValueError('map/model quantized sampling association differs: '+name)
    return dict(producer=ASSOCIATION_PRODUCER,quantized_sampling=ASSOCIATION_SAMPLING,physical_calibration_truth=False,source_geometry_sha256=hashlib.sha256(preprocess.canonical(fixture)).hexdigest(),maps_checked_against_models=True)


def project_known_point(point,view):
    if not math.isfinite(view['hfov_radians']) or not 0<view['hfov_radians']<2*math.pi:raise ValueError('cylindrical HFOV must be strictly between0and2pi; full-seam360unsupported')
    if view['projection'] not in ('perspective','cylindrical'):raise ValueError('unknown projection')
    local=np.asarray(view['virtual_from_original_left'])@np.r_[point,1.];x,y,z=local[:3]
    if view['projection']=='perspective':
        if z<=0:return None
        u=view['fx']*x/z+view['cx'];v=view['fy']*y/z+view['cy'];output_depth=z
    else:
        radius=math.hypot(x,z)
        if radius<=0:return None
        u=(math.atan2(x,z)/view['hfov_radians']+.5)*view['width']-.5;v=view['fy']*y/radius+view['cy'];output_depth=math.sqrt(x*x+y*y+z*z)
    if not all(math.isfinite(float(a)) for a in (u,v,output_depth)) or not(-.5<=u<view['width']-.5 and -.5<=v<view['height']-.5):return None
    return math.floor(u+.5),math.floor(v+.5),float(output_depth)


def empty_output(count):
    out={name:np.zeros(count,'<i8' if name.endswith('i64') else '<f4' if name.endswith('f32') else '<u4') for name in PLANES}
    out['depth.f32'].view('<u4')[:]=0x7fc00000;out['owner.u32'][:]=UINT_MAX;out['source_ids.i64'][:]=-1
    return out


def assign_preview(out,index,rgb,eye):
    if eye not in (0,1):raise ValueError('invalid preview eye')
    out['rgb.u32'][index]=rgb;out['rgb_validity.u32'][index]=1;out['approximate.u32'][index]=1;out['source.u32'][index]=eye+1


def pack_rgb(image):
    rgb=np.asarray(image,np.uint32);return rgb[...,0]|(rgb[...,1]<<8)|(rgb[...,2]<<16)


def original_preview_indices(fixture,view):
    """Independent virtual-ray to original-eye distorted projection; no translation."""
    count=view['width']*view['height'];rw,rh=fixture['raw_size'];result=np.full((2,count),UINT_MAX,'<u4');V=np.asarray(view['virtual_from_original_left'])[:3,:3]
    for y in range(view['height']):
        for x in range(view['width']):
            if view['projection']=='perspective':direction=np.array([(x-view['cx'])/view['fx'],(y-view['cy'])/view['fy'],1.])
            else:
                angle=((x+.5)/view['width']-.5)*view['hfov_radians'];direction=np.array([math.sin(angle),(y-view['cy'])/view['fy'],math.cos(angle)])
            original_left=np.linalg.solve(V,direction)
            for eye in (0,1):
                direction_eye=np.asarray(fixture['right_to_left'])[:3,:3].T@original_left if eye else original_left
                if direction_eye[2]<=0:continue
                u,v=distorted_project(direction_eye,fixture['original'][eye]);ix,iy=math.floor(u+.5),math.floor(v+.5)
                if 0<=ix<rw and 0<=iy<rh:result[eye,y*view['width']+x]=eye*rw*rh+iy*rw+ix
    return result


def known_observations(fixture,case='tie',source_id=7):
    """Sparse 9-plane typed input and explicit independent physical witnesses."""
    w,h=fixture['rect_size'];n=w*h;planes={p:np.zeros(2*n,'<i8' if p.endswith('i64') else '<f4' if p.endswith('f32') else '<u4') for p in depth.PLANES}
    for p in ('lr_disparity.f32','depth.f32'):planes[p].view('<u4')[:]=0x7fc00000
    planes['source_ids.i64'][:]=-1;witnesses=[]
    if case not in ('tie','conflict','missing','unsupported','preview','shared-raw-cell'):raise ValueError('unknown known-point case')
    if case not in ('missing','preview'):
        for eye in (0,1):
            world=[0.,.75,1.] if not(case=='conflict' and eye==1) else [0.,1.5,2.]
            pixel,z=rectified_observation(fixture,world,eye)
            if any(abs(v-round(v))>1e-12 for v in pixel):
                # Conflict can use different rectified coordinates; only exact
                # integer-grid witnesses qualify this scalar byte profile.
                raise ValueError('known point does not lie on exact rectified pixel')
            x,y=map(round,pixel)
            if not(0<=x<w and 0<=y<h):raise ValueError('known point outside rectified grid')
            i=eye*n+y*w+x
            for p in ('lr_validity.u32','metric_validity.u32'):planes[p][i]=1
            planes['lr_disparity.f32'][i]=fixture['rectified'][0]['fx']*fixture['right_to_left'][0][3]/z+fixture['rectified'][0]['cx']-fixture['rectified'][1]['cx'];planes['lr_confidence.f32'][i]=.5;planes['depth.f32'][i]=z;planes['metric_confidence.f32'][i]=.5;planes['provenance.u32'][i]=1;planes['source_ids.i64'][i]=source_id
            witnesses.append(dict(eye=eye,owner=i,pixel=[x,y],rectified_z=z,original_left_point=world))
    if case=='shared-raw-cell':
        world=[.15625,.75,1.];pixel,z=rectified_observation(fixture,world,0);x,y=map(round,pixel);i=y*w+x
        for name in ('lr_validity.u32','metric_validity.u32'):planes[name][i]=1
        planes['lr_disparity.f32'][i]=fixture['rectified'][0]['fx']*fixture['right_to_left'][0][3]/z+fixture['rectified'][0]['cx']-fixture['rectified'][1]['cx'];planes['lr_confidence.f32'][i]=.5;planes['depth.f32'][i]=z;planes['metric_confidence.f32'][i]=.5;planes['provenance.u32'][i]=1;planes['source_ids.i64'][i]=source_id
        witnesses.append(dict(eye=0,owner=i,pixel=[x,y],rectified_z=z,original_left_point=world))
    if case=='unsupported':
        for name in ('lr_disparity.f32','depth.f32'):planes[name].view('<u4')[:]=0x7fc00000
        for name in ('lr_validity.u32','lr_confidence.f32','metric_validity.u32','metric_confidence.f32','provenance.u32'):planes[name][:]=0
        planes['source_ids.i64'][:]=-1
        maps=geometry_maps(fixture);rw,rh=fixture['raw_size'];support=preprocess.support_mask(maps['left_x'],maps['left_y'],rw,rh);bad=np.flatnonzero(support.ravel()==0)
        if not len(bad):raise ValueError('unsupported fixture requires a real unsupported remap sample')
        i=int(bad[0]);planes['lr_validity.u32'][i]=1;planes['lr_disparity.f32'][i]=1.;planes['lr_confidence.f32'][i]=.5;witnesses=[]
    return planes,witnesses


def expected_known_output(fixture,view,planes,witnesses,rect_rgb,original_rgb,mode='strict',exposure_gain=(1.,1.,1.)):
    """Exact diagnostic profile for rational witnesses; no generic float tolerance.

    Exposure gains are supplied only as mathematically forced clamp endpoints
    or unity. General histogram log/exp float bytes require frozen GPU controls.
    """
    count=view['width']*view['height'];out=empty_output(count);layers=[{},{}];w,h=fixture['rect_size'];n=w*h
    for witness in witnesses:
        owner=witness['owner']
        if not planes['metric_validity.u32'][owner]:continue
        projected=project_known_point(witness['original_left_point'],view)
        if projected is None:continue
        x,y,z=projected;t=y*view['width']+x;candidate=(z,owner,witness)
        if t not in layers[witness['eye']] or candidate[:2]<layers[witness['eye']][t][:2]:layers[witness['eye']][t]=candidate
    previews=original_preview_indices(fixture,view);rgb_rect=[pack_rgb(a).ravel() for a in rect_rgb];rgb_original=np.concatenate([pack_rgb(a).ravel() for a in original_rgb])
    def color(index):
        packed=int(rgb_rect[index//n][index%n]);return np.array([packed&255,(packed>>8)&255,(packed>>16)&255],float)
    for t in range(count):
        left=layers[0].get(t);right=layers[1].get(t)
        if left is None and right is None:
            if mode=='preview':
                for eye in (0,1):
                    index=int(previews[eye,t])
                    if index!=UINT_MAX:assign_preview(out,t,rgb_original[index],eye);break
            continue
        compatible=left is not None and right is not None and abs(left[0]-right[0])<=.01+.01*min(left[0],right[0]);lc=planes['metric_confidence.f32'][left[1]] if left else 0.;rc=planes['metric_confidence.f32'][right[1]] if right else 0.
        choose_right=right is not None and (left is None or (not compatible and right[0]<left[0]) or (compatible and rc>lc));chosen=right if choose_right else left;z,owner,_=chosen;rgb=color(owner)*(exposure_gain if choose_right else np.ones(3));source=2 if choose_right else 1
        if compatible and abs(float(lc)-float(rc))<=.1:
            alpha=float(lc)/(float(lc)+float(rc)) if lc+rc>0 else .5;rgb=color(left[1])*alpha+color(right[1])*exposure_gain*(1-alpha);source=3
        packed=np.rint(np.clip(rgb,0,255)).astype(np.uint32);out['rgb.u32'][t]=packed[0]|packed[1]<<8|packed[2]<<16
        out['depth.f32'][t]=z;out['confidence.f32'][t]=planes['metric_confidence.f32'][owner];out['validity.u32'][t]=out['rgb_validity.u32'][t]=1;out['owner.u32'][t]=owner;out['source.u32'][t]=source;out['provenance.u32'][t]=planes['provenance.u32'][owner];out['source_ids.i64'][t]=planes['source_ids.i64'][owner];out['age_ns.i64'][t]=planes['age_ns.i64'][owner]
    return out


def export_fixture(directory,case='tie',projection='perspective',rgb_pattern='constant',virtual_pose='identity'):
    import cv2
    from stereoforge.types import Calibration
    if cv2.__version__!='4.14.0':raise ValueError('fixture requires pinned OpenCV4.14.0')
    directory=Path(directory);base.create_output(directory);fixture=geometry_fixture('shared-raw-cell' if case=='shared-raw-cell' else 'default',virtual_pose);fixture['view']['projection']=projection
    if case=='shared-raw-cell' and projection!='perspective':raise ValueError('shared-cell scalar profile qualified for perspective only')
    if projection not in ('perspective','cylindrical') or rgb_pattern not in ('constant','periphery'):raise ValueError('unknown fixture profile')
    original=dict(left=fixture['original'][0],right=fixture['original'][1],right_to_left=fixture['right_to_left'],serial='synthetic-independent-view-geometry')
    rectified=dict(left=fixture['rectified'][0],right=fixture['rectified'][1],right_to_left=fixture['right_to_left'],serial='synthetic-independent-view-geometry-rectified')
    original=Calibration.from_dict(original);rectified=Calibration.from_dict(rectified)
    def save(name,data):
        path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return dict(file=name,bytes=len(data),sha256=base.sha256(path))
    original_entry=save('original-calibration.json',preprocess.canonical(original.to_dict()));rect_entry=save('rectified-calibration.json',preprocess.canonical(rectified.to_dict()))
    rw,rh=fixture['raw_size'];w,h=fixture['rect_size'];packed=np.empty((rh,2*rw,2),np.uint8);packed[...,1]=128;packed[:,:rw,0]=200;packed[:,rw:,0]=80
    if rgb_pattern=='periphery':
        yy,xx=np.indices((rh,rw));packed[:,:rw,0]=(32+(xx*7+yy*3)%160).astype(np.uint8);packed[:,rw:,0]=(40+(xx*5+yy*11)%150).astype(np.uint8)
    def string(value):
        raw=value.encode();return struct.pack('<I',len(raw))+raw
    record=struct.pack('<IIIQQQ',rw*2,rh,rw*4,7,100,0)
    for text in ('synthetic_camera','synthetic_host',original.identity,'synthetic-known-geometry','{"synthetic":true}','synthetic'):record+=string(text)
    record+=struct.pack('<Q',0)+string('synthetic_controls')
    for text in ('synthetic_no_imu','synthetic_imu','synthetic_no_pose_sensor'):record+=string(text)
    payload=packed.tobytes();record+=struct.pack('<QI',0,0)+struct.pack('<Q',len(payload))+payload;checksum=14695981039346656037
    for byte in record:checksum=((checksum^byte)*1099511628211)&0xffffffffffffffff
    capture=save('capture/capture.sfp',b'SFYUYV1\0'+struct.pack('<IQ',1,1)+record+struct.pack('<Q',checksum));capture.pop('bytes');capture['record_ordinal']=0
    maps=geometry_maps(fixture);association=validate_map_association(fixture,maps);decoded=cv2.cvtColor(packed,cv2.COLOR_YUV2RGB_YUY2);expected={};map_entries={};images=[];original_images=[]
    for eye,name in enumerate(('left','right')):
        c=fixture['original'][eye];r=fixture['rectified'][eye];K=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1]],float);P=np.array([[r['fx'],0,r['cx']],[0,r['fy'],r['cy']],[0,0,1]],float)
        cvmx,cvmy=cv2.initUndistortRectifyMap(K,np.asarray(c['distortion']),np.asarray(fixture['R1' if eye==0 else 'R2']),P,(w,h),cv2.CV_32FC1)
        for component,cvmap in (('x',cvmx),('y',cvmy)):
            key=name+'_'+component
            if not np.array_equal(np.rint(maps[key].astype(float)*32),np.rint(cvmap.astype(float)*32)):raise ValueError('independent geometry/OpenCV effective map disagreement')
            map_entries[key]=save('maps/'+key+'.f32',maps[key].tobytes())
        raw=decoded[:,eye*rw:(eye+1)*rw].copy();rect=cv2.remap(raw,maps[name+'_x'],maps[name+'_y'],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);support=preprocess.support_mask(maps[name+'_x'],maps[name+'_y'],rw,rh)
        original_images.append(raw);images.append(rect)
        for key,data in (('original_'+name+'_rgb',raw.tobytes()),('rect_'+name+'_rgb',rect.tobytes()),(name+'_luma',cv2.cvtColor(rect,cv2.COLOR_RGB2GRAY).astype('<u4').tobytes()),(name+'_support',support.tobytes())):
            expected[key]=save('expected/'+preprocess.PLANES[key],data)
    cal=dict(original_identity=original.identity,rectified_identity=rectified.identity,original_json=original_entry,rectified_json=rect_entry,R1=fixture['R1'],R2=fixture['R2'])
    for eye,name in enumerate(('P1','P2')):
        c=fixture['rectified'][eye];cal[name]=[[c['fx'],0,c['cx'],-c['fx']*.625 if eye else 0],[0,c['fy'],c['cy'],0],[0,0,1,0]]
    cv_binary=next(Path(cv2.__file__).parent.glob('*.so'))
    manifest=dict(schema=preprocess.SCHEMA,profile=preprocess.PROFILE,source=dict(eye_width=rw,height=rh,stride_bytes=rw*4),rectified=dict(width=w,height=h),capture=capture,calibration=cal,maps=map_entries,expected=expected,provenance=dict(image_convention=preprocess.IMAGE_CONVENTION,cv2_version=cv2.__version__,cv2_binary_sha256=base.sha256(cv_binary),opencv_source_commit=preprocess.SOURCE_COMMIT,exporter_sha256=base.sha256(__file__),synthetic=True))
    (directory/'manifest.json').write_bytes(preprocess.canonical(manifest));preprocess.validate_fixture(directory/'manifest.json')
    planes,witnesses=known_observations(fixture,case)
    if case=='unsupported':
        for index in range(2*w*h):
            if not planes['metric_validity.u32'][index]:
                planes['depth.f32'].view('<u4')[index]=0x7fc00000;planes['metric_confidence.f32'][index]=0;planes['provenance.u32'][index]=0;planes['source_ids.i64'][index]=-1
    observations={name:save('observations.'+name,data.tobytes()) for name,data in planes.items()}
    geometry=dict(schema='stereoforge.view_depth.fixture.v1',geometry=fixture,association=association,case=case,rgb_pattern=rgb_pattern,virtual_pose=virtual_pose,witnesses=witnesses,observations=observations,metric_source='synthetic_known_geometry',physical_calibration_truth=False,performance_qualified=False,manifest_sha256=base.sha256(directory/'manifest.json'))
    (directory/'geometry.json').write_bytes(preprocess.canonical(geometry));return geometry


def load_fixture(directory):
    directory=Path(directory).resolve();manifest,metadata=preprocess.validate_fixture(directory/'manifest.json')
    if (directory/'geometry.json').is_symlink():raise ValueError('unsafe geometry fixture file')
    geometry=base.strict_json((directory/'geometry.json').read_text());fixture=geometry['geometry'];w,h=fixture['rect_size'];rw,rh=fixture['raw_size']
    if geometry.get('schema')!='stereoforge.view_depth.fixture.v1' or geometry.get('metric_source')!='synthetic_known_geometry' or geometry.get('physical_calibration_truth') is not False or geometry.get('performance_qualified') is not False:raise ValueError('unqualified diagnostic geometry fixture')
    if geometry['manifest_sha256']!=base.sha256(directory/'manifest.json'):raise ValueError('fixture manifest identity changed')
    if (rw,rh)!=(manifest['source']['eye_width'],manifest['source']['height']) or (w,h)!=(manifest['rectified']['width'],manifest['rectified']['height']):raise ValueError('geometry/source dimensions differ')
    for eye,name in enumerate(('left','right')):
        for space,models in (('original',fixture['original']),('rectified',fixture['rectified'])):
            cal=base.strict_json(preprocess.fixture_file(directory,manifest['calibration'][space+'_json']).read_text())
            if preprocess.canonical(cal[name])!=preprocess.canonical(models[eye]) or preprocess.canonical(cal['right_to_left'])!=preprocess.canonical(fixture['right_to_left']):raise ValueError('geometry/calibration model association differs')
    for name in ('R1','R2'):
        if preprocess.canonical(manifest['calibration'][name])!=preprocess.canonical(fixture[name]):raise ValueError('geometry rotation association differs')
    maps={name:np.frombuffer(preprocess.fixture_file(directory,entry).read_bytes(),'<f4').reshape(h,w) for name,entry in manifest['maps'].items()};association=validate_map_association(fixture,maps)
    if geometry['association']!=association:raise ValueError('geometry producer association evidence differs')
    # Verify actual capture->OpenCV decode/remap RGB/luma/support, not merely
    # self-consistent hashes of nominal expected images.
    depth.fixture_planes(directory/'manifest.json')
    planes={name:np.frombuffer(preprocess.fixture_file(directory,entry).read_bytes(),'<i8' if name.endswith('i64') else '<f4' if name.endswith('f32') else '<u4').copy() for name,entry in geometry['observations'].items()}
    if set(planes)!=set(depth.PLANES) or any(len(v)!=2*w*h for v in planes.values()):raise ValueError('known observation plane extent mismatch')
    supports=[np.frombuffer(preprocess.fixture_file(directory,manifest['expected'][name+'_support']).read_bytes(),'<u4') for name in ('left','right')]
    for i in range(2*w*h):
        valid=int(planes['metric_validity.u32'][i]);lr=int(planes['lr_validity.u32'][i])
        if valid not in (0,1) or lr not in (0,1):raise ValueError('nonbinary known observation mask')
        if valid:
            eye=i//(w*h);index=i%(w*h)
            if not lr or not supports[eye][index]:raise ValueError('unsupported geometry observation')
            disparity=float(planes['lr_disparity.f32'][i])
            if not math.isfinite(disparity):raise ValueError('invalid metric correspondence support')
            other_x=math.floor(index%w+(disparity if eye else -disparity)+.5)
            if not 0<=other_x<w or not supports[1-eye][index//w*w+other_x]:raise ValueError('unsupported metric correspondence support')
            if not math.isfinite(float(planes['depth.f32'][i])) or planes['depth.f32'][i]<=0 or not 0<=planes['metric_confidence.f32'][i]<=1 or not 1<=planes['provenance.u32'][i]<=5 or planes['source_ids.i64'][i]<0 or planes['age_ns.i64'][i]<0:raise ValueError('invalid known metric observation')
        elif planes['depth.f32'].view('<u4')[i]!=0x7fc00000 or planes['metric_confidence.f32'][i]!=0 or planes['provenance.u32'][i]!=0 or planes['source_ids.i64'][i]!=-1 or planes['age_ns.i64'][i]!=0:raise ValueError('invalid metric defaults')
        if lr:
            if not math.isfinite(float(planes['lr_disparity.f32'][i])) or not 0<=planes['lr_confidence.f32'][i]<=1:raise ValueError('invalid known LR observation')
        elif planes['lr_disparity.f32'].view('<u4')[i]!=0x7fc00000 or planes['lr_confidence.f32'][i]!=0:raise ValueError('invalid LR defaults')
    owners=set()
    for witness in geometry['witnesses']:
        owner=witness['owner'];eye=witness['eye']
        if type(eye) is not int or eye not in (0,1) or type(owner) is not int or not 0<=owner<2*w*h or len(witness['pixel'])!=2 or any(type(v) is not int for v in witness['pixel']) or not(0<=witness['pixel'][0]<w and 0<=witness['pixel'][1]<h) or np.asarray(witness['original_left_point']).shape!=(3,) or not np.isfinite(witness['original_left_point']).all():raise ValueError('known3D witness domain invalid')
        pixel,z=rectified_observation(fixture,witness['original_left_point'],eye)
        if owner in owners or owner!=eye*w*h+witness['pixel'][1]*w+witness['pixel'][0] or not planes['metric_validity.u32'][owner] or np.max(np.abs(np.asarray(pixel)-witness['pixel']))>1e-12 or z!=witness['rectified_z'] or np.float32(z)!=planes['depth.f32'][owner]:raise ValueError('known3D witness association differs')
        owners.add(owner)
    if owners!=set(np.flatnonzero(planes['metric_validity.u32'])):raise ValueError('every metric observation requires independent known3D witness')
    if 'synthetic' not in metadata['device_identity'] or 'synthetic' not in metadata['camera_clock_domain']:raise ValueError('explicit synthetic provenance required')
    images={}
    for eye,name in enumerate(('left','right')):
        images['rect_'+name]=np.frombuffer(preprocess.fixture_file(directory,manifest['expected']['rect_'+name+'_rgb']).read_bytes(),np.uint8).reshape(h,w,3)
        images['original_'+name]=np.frombuffer(preprocess.fixture_file(directory,manifest['expected']['original_'+name+'_rgb']).read_bytes(),np.uint8).reshape(rh,rw,3)
    return manifest,metadata,geometry,planes,images


def export_rhi_control(directory,geometry,planes,images):
    """Pack unique metric entries plus preview-only RGB; no grid-neighbor ops."""
    directory=Path(directory);base.create_output(directory);fixture=geometry['geometry'];view=fixture['view'];w,h=fixture['rect_size'];rw,rh=fixture['raw_size'];n=w*h;raw_n=rw*rh;m=n+raw_n;target_n=view['width']*view['height']
    arrays={'source_depth.f32':np.full(2*m,np.nan,'<f4'),'confidence.f32':np.zeros(2*m,'<f4'),'rgb.u32':np.zeros(2*m,'<u4'),'provenance.u32':np.zeros(2*m,'<u4'),'rays.f32':np.tile(np.array([0,0,1],'<f4'),(2*m,1)),'source_ids.i64':np.full(2*m,-1,'<i8'),'age_ns.i64':np.zeros(2*m,'<i8')}
    for eye,name in enumerate(('left','right')):
        arrays['rgb.u32'][eye*m:eye*m+n]=pack_rgb(images['rect_'+name]).ravel();arrays['rgb.u32'][eye*m+n:(eye+1)*m]=pack_rgb(images['original_'+name]).ravel()
    for witness in geometry['witnesses']:
        owner=witness['owner'];i=witness['eye']*m+owner%n;point=witness['original_left_point'];z=point[2]
        if z<=0:raise ValueError('frozen RHI unitZ control requires forward original-space witness')
        arrays['source_depth.f32'][i]=z;arrays['rays.f32'][i]=[point[0]/z,point[1]/z,1]
        for filename,input_plane in (('confidence.f32','metric_confidence.f32'),('provenance.u32','provenance.u32'),('source_ids.i64','source_ids.i64'),('age_ns.i64','age_ns.i64')):arrays[filename][i]=planes[input_plane][owner]
    preview=original_preview_indices(fixture,view);packed=preview.copy()
    for eye in (0,1):
        valid=packed[eye]!=UINT_MAX;packed[eye,valid]=eye*m+n+(packed[eye,valid]-eye*raw_n)
    arrays['preview_indices.u32']=packed.ravel()
    files={}
    for name,data in arrays.items():(directory/name).write_bytes(data.tobytes());files[name]=dict(file=name,bytes=data.nbytes,sha256=base.sha256(directory/name))
    pose=' '.join(repr(float(v)) for v in np.asarray(view['virtual_from_original_left']).ravel());header=f"{m} 1 {view['width']} {view['height']}\n"+' '.join(repr(view[k]) for k in ('fx','fy','cx','cy'))+'\n'+repr(view['hfov_radians'])+'\n'+pose+'\n'+pose+'\n100\nsynthetic-independent-known3D-control\n';(directory/'header.txt').write_text(header);files['header.txt']=dict(file='header.txt',bytes=len(header.encode()),sha256=base.sha256(directory/'header.txt'))
    manifest=dict(schema_version=1,files=files,target_width=view['width'],target_height=view['height'],source_width=m,source_height=1,metric_owner_namespace='eye*(rectN+rawN)+rectIndex',preview_namespace='eye*(rectN+rawN)+rectN+rawIndex',topology_audit='source dimensions used only for count/eye offsets; no row-neighbor or spatial weights',projection='perspective',scope='frozen_compositor_control; perspective_only; unique_observation_packing_not_raw_depth_canvas')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def normalize_control_owners(raw,geometry):
    fixture=geometry['geometry'];n=math.prod(fixture['rect_size']);m=n+math.prod(fixture['raw_size']);owners=np.frombuffer(raw['owner.u32'],'<u4').copy();valid=np.frombuffer(raw['validity.u32'],'<u4')!=0
    if np.any(valid&((owners//m>1)|(owners%m>=n))):raise ValueError('control metric owner outside unique metric namespace')
    approximate=np.frombuffer(raw['approximate.u32'],'<u4')!=0;rgb_valid=np.frombuffer(raw['rgb_validity.u32'],'<u4')!=0;preview=(~valid)&approximate&rgb_valid
    if np.any((~valid)&(~preview)&(owners!=UINT_MAX)):raise ValueError('control strict hole has corrupt owner')
    if np.any(preview&((owners//m>1)|(owners%m<n))):raise ValueError('control preview owner outside independent raw RGB namespace')
    owners[valid]=(owners[valid]//m)*n+owners[valid]%m;owners[preview]=UINT_MAX
    return dict(raw,**{'owner.u32':owners.tobytes()})


VIEW_ENTRIES=('clear','clear_layers','splat_layer_depth','splat_layer_owner','histogram_ratios','reduce_ratios','resolve_layers','gather_project_metadata','inspect_outputs')
SOURCE_FILES=('src/daxa_observation.cpp','include/stereoforge/daxa_observation.hpp','benchmarks/compare_daxa_view_depth.py','apps/daxa_view_depth_fixture.cpp','apps/daxa_view_depth_common.hpp','apps/daxa_view_depth_benchmark.cpp','src/daxa_view.cpp','src/daxa_view_calibration.cpp','include/stereoforge/daxa_view.hpp','include/stereoforge/daxa_view_calibration.hpp','include/stereoforge/daxa_view_parameters.hpp','apps/native_view_benchmark.cpp','src/geometry.cpp','python/stereoforge/shaders/daxa/view_depth.slang','python/stereoforge/shaders/daxa/view.slang','python/stereoforge/shaders/ops/projection.slang','python/stereoforge/shaders/fusion/project.slang','CMakeLists.txt')


def native_identities(rhi,native,compiler,manifest):
    identity=depth.native_identities(ROOT/'build/standalone/native_stereo',native,compiler,manifest)
    identity['view_rhi_binary_sha256']=base.sha256(rhi);identity['view_rhi_dependencies']=legacy.dependencies(rhi,False)['sha256']
    for name in SOURCE_FILES:identity['source_sha256'][name]=base.sha256(ROOT/name)
    directory=Path(identity['offline_shader_directory'])
    for prefix in ('daxa-view-','daxa-view-depth-'):
        for entry in VIEW_ENTRIES:identity['offline_spirv_sha256'][prefix+entry+'.spv']=base.sha256(directory/(prefix+entry+'.spv'))
    # Preserve every additional compiled control in this actual offline directory.
    for path in sorted(directory.glob('*.spv')):identity['offline_spirv_sha256'][path.name]=base.sha256(path)
    identity['compiler_input_sha256']['view_daxa']=depth.compiler_header_identities(directory,(('stereoforge_daxa_foundation','src/daxa_observation.cpp'),('daxa_view_depth_fixture','apps/daxa_view_depth_fixture.cpp'),('stereoforge_daxa_view','src/daxa_view.cpp'),('stereoforge_daxa_view','src/daxa_view_calibration.cpp')))
    identity['compiler_input_sha256']['view_rhi']=depth.compiler_header_identities(Path(rhi).resolve().parent,(('native_view_benchmark','apps/native_view_benchmark.cpp'),('stereoforge_runtime','src/geometry.cpp')))
    groups=dict(view_depth_shader_source_sha256_pair=('python/stereoforge/shaders/daxa/view_depth.slang','python/stereoforge/shaders/ops/projection.slang'),depth_shader_source_sha256_pair=('python/stereoforge/shaders/daxa/depth.slang','python/stereoforge/shaders/ops/stereo_observation.slang'),preprocess_shader_source_sha256_pair=('python/stereoforge/shaders/daxa/preprocess.slang','python/stereoforge/shaders/ops/preprocess.slang'))
    identity['view_shader_source_identities']={key:':'.join(base.sha256(ROOT/name) for name in names) for key,names in groups.items()}
    child=subprocess.run([str(native),'--shader-source-identities'],capture_output=True,text=True,errors='replace');base.require_child_clean(child.returncode,child.stdout,child.stderr)
    if base.strict_json(child.stdout)!=identity['view_shader_source_identities']:raise ValueError('actual compiled view/depth/preprocess shader identities differ')
    return identity


def validate_report(report,manifest,metadata,geometry,identity,mode,validation):
    fixture=geometry['geometry'];view=fixture['view'];w,h=fixture['rect_size'];rw,rh=fixture['raw_size'];original_preview=original_preview_indices(fixture,view)
    expected=dict(schema='stereoforge.daxa_view_depth.v1',scope='synthetic_known_observation_typed_compositor_fixture_no_stereo_accuracy_or_performance',observations_overridden_for_fixture=True,measured_stereo_output=False,metric_source='synthetic_known_geometry',performance_qualified=False,rectified_width=w,rectified_height=h,original_width=rw,original_height=rh,target_width=view['width'],target_height=view['height'],projection=view['projection'],depth_convention='virtual_axial_Z' if view['projection']=='perspective' else 'virtual_ray_range',mode=mode,map_geometry_association='opencv4.14-effective-quantized-map-model-validation',map_geometry_truth_scope='effective_OpenCV4.14_sample_association_of_supplied_models_not_physical_calibration_truth',preview_valid_count=int((original_preview!=UINT_MAX).sum()),metric_owner_count=2*w*h,preview_namespace_count=2*rw*rh,preview_only_owner=UINT_MAX,fixture_manifest_sha256=identity['manifest_sha256'],capture_sha256=manifest['capture']['sha256'],original_identity=manifest['calibration']['original_identity'],rectified_identity=manifest['calibration']['rectified_identity'],source_id=metadata['frame_id'],shader_source_identities=identity['view_shader_source_identities'],compound_cpu_float_reference_checked=False,physical_geometry_qualified=False,live_60hz_qualified=False,synthetic_capture=True,replacement_transfer_completion_waited=True,validation_requested=validation,diagnostic_observation_sha256=[geometry['observations'][name]['sha256'] for name in depth.PLANES],source_metadata={key:metadata[key] for key in ('frame_id','camera_timestamp_ns','host_ingress_timestamp_ns','camera_clock_domain','host_clock_domain','calibration_identity','device_identity')})
    requested_fov=float(np.float32(view['hfov_radians'])) if view['projection']=='cylindrical' else math.atan((view['width']-.5-view['cx'])/view['fx'])-math.atan((-.5-view['cx'])/view['fx'])
    expected.update(virtual_camera_pose_policy='explicit_virtual_from_original_left',virtual_from_original_left=np.asarray(view['virtual_from_original_left']).astype('<f4').ravel().astype(float).tolist(),virtual_intrinsics=[float(np.float32(view[k])) for k in ('fx','fy','cx','cy')],hfov_radians=float(np.float32(view['hfov_radians'])),requested_horizontal_fov_radians=requested_fov,virtual_camera_canvas_policy='calibration_left_canvas_unless_explicit_intrinsics_and_dimensions',coverage_scope='explicit_posttiming_export_target_canvas_pixel_fractions_not_observed_FOV_or_physical_truth')
    for key,value in expected.items():
        if key not in report or preprocess.canonical(report[key])!=preprocess.canonical(value):raise ValueError('changed/missing typed view diagnostic contract: '+key)
    for key in ('rgb_valid_coverage','metric_valid_coverage'):
        if type(report.get(key)) not in (int,float) or not math.isfinite(report[key]) or not 0<=report[key]<=1:raise ValueError('invalid explicit exported coverage: '+key)
    for eye,name in enumerate(('effective_source_to_virtual_left','effective_source_to_virtual_right')):
        R=np.eye(4);R[:3,:3]=np.asarray(fixture['R2' if eye else 'R1']).T;V=np.asarray(view['virtual_from_original_left']);expected_pose=V@(np.asarray(fixture['right_to_left']) if eye else np.eye(4))@R
        actual=np.asarray(report.get(name))
        if actual.shape!=(16,) or not np.isfinite(actual).all() or not np.array_equal(actual.astype('<f4').view('<u4'),expected_pose.astype('<f4').ravel().view('<u4')):raise ValueError('effective pose/rotation-once contract differs: '+name)
        # Require exact float32 JSON numeric values, not nearby values that round alike.
        if not np.array_equal(actual,expected_pose.astype('<f4').ravel().astype(float)):raise ValueError('effective pose report not exact float32 values')
    device=report.get('device',{})
    if device.get('slangc_sha256')!=identity['compiler_sha256']:raise ValueError('actual compiler identity differs')
    for row,keys in ((report,('validation_errors_after_teardown','validation_warnings_after_teardown')),(device,('validation_errors','validation_warnings'))):
        for key in keys:
            if type(row.get(key)) is not int or row[key]!=0:raise ValueError('view validation/teardown diagnostics')


def load_output(prefix,count):
    raw={}
    for name in PLANES:
        data=Path(str(prefix)+'.'+name).read_bytes()
        if len(data)!=count*(8 if name.endswith('i64') else 4):raise ValueError('final plane extent mismatch: '+name)
        raw[name]=data
    return raw


def forced_gain(geometry,planes,images):
    """Only prove unity/no overlap or clamp endpoints; no log/exp approximation."""
    fixture=geometry['geometry'];view=fixture['view'];w,h=fixture['rect_size'];n=w*h;layers=[{},{}]
    for witness in geometry['witnesses']:
        p=project_known_point(witness['original_left_point'],view)
        if p:layers[witness['eye']][p[:2]]=(p[2],witness)
    gains=[]
    for channel in range(3):
        ratios=[]
        for target,left in layers[0].items():
            right=layers[1].get(target)
            if not right or abs(left[0]-right[0])>.01+.01*min(left[0],right[0]):continue
            lv=images['rect_left'].reshape(-1,3)[left[1]['owner']%n,channel];rv=images['rect_right'].reshape(-1,3)[right[1]['owner']%n,channel]
            if 8<lv<247 and 8<rv<247:ratios.append(float(lv)/float(rv))
        if not ratios:gains.append(1.)
        elif min(ratios)>2.01:gains.append(2.)
        elif max(ratios)<.49:gains.append(.5)
        else:raise ValueError('general exposure compound-float profile requires frozen GPU control; no scalar byte claim')
    return np.asarray(gains)


def qualify_exact_profile(geometry,planes):
    """Fail closed outside the independently reasoned rational scalar profile."""
    expected=geometry_fixture('shared-raw-cell' if geometry['case']=='shared-raw-cell' else 'default',geometry.get('virtual_pose','identity'));actual=geometry['geometry'];expected['view']['projection']=actual['view']['projection']
    if preprocess.canonical(expected)!=preprocess.canonical(actual):raise ValueError('geometry outside qualified exact known-point arithmetic profile')
    wanted,witnesses=known_observations(actual,geometry['case'])
    if geometry['case']=='unsupported':
        for i in range(len(wanted['depth.f32'])):
            wanted['depth.f32'].view('<u4')[i]=0x7fc00000;wanted['metric_confidence.f32'][i]=0;wanted['provenance.u32'][i]=0;wanted['source_ids.i64'][i]=-1
    if witnesses!=geometry['witnesses'] or any(wanted[name].tobytes()!=planes[name].tobytes() for name in wanted):raise ValueError('observations outside exact independent known3D fixture profile')


def run_comparison(fixture_directory,native,rhi,compiler,output,validation=False,native_only_validation=False):
    # --validation retains its both-child meaning. The split is explicit, never a retry.
    if native_only_validation and not validation:raise ValueError('native-only validation requires --validation')
    fixture_directory=Path(fixture_directory).resolve();manifest,metadata,geometry,planes,images=load_fixture(fixture_directory);qualify_exact_profile(geometry,planes)
    output=Path(output).absolute();base.create_output(output);view=geometry['geometry']['view'];count=view['width']*view['height'];control=output/'frozen-control';export_rhi_control(control,geometry,planes,images)
    fixture_hashes={str(p.resolve()):base.sha256(p) for p in fixture_directory.rglob('*') if p.is_file()};control_hashes={str(p.resolve()):base.sha256(p) for p in control.rglob('*') if p.is_file()}
    evidence=dict(schema='stereoforge.daxa_view_depth.comparison.v1',passed=False,promoted=False,scope='synthetic_known_geometry_typed_view_intake_calibration_compositor_only; no_stereo_accuracy_physical_truth_performance_or_live60qualification',frozen_compositor_control_available=view['projection']=='perspective',frozen_compositor_control_scope='unchanged_perspective_RHI_compositor_only; legacy_binary_cannot_represent_cylindrical; cylinder_uses_independent_knownpoint_exact_profile_only',metric_source='synthetic_known_geometry',performance_qualified=False,cpu_opencv_capture_preprocessing_checked=True,compound_cpu_float_reference_checked=False,scalar_profile='rational_known3D_witness_positions; canonical_invalids; exact_copied_metadata; confidence_half; unity_or_forced_exposure_clamp_endpoints; no_general_float_tolerance',association=geometry['association'],fixture_manifest_sha256=base.sha256(fixture_directory/'manifest.json'),geometry_sha256=base.sha256(fixture_directory/'geometry.json'),children={},comparisons={})
    reports={};dumps={};env=os.environ.copy();env.update(PYTHONHOME='/nonexistent',PYTHONPATH='')
    if validation:env['SF_DAXA_VALIDATION']='1';env['VK_INSTANCE_LAYERS']='VK_LAYER_KHRONOS_validation'
    control_env=env.copy()
    removed={}
    if native_only_validation:
        for name in legacy.VALIDATION_ENVIRONMENT:
            if name in control_env:removed[name]=control_env.pop(name)
    evidence['report_sha256']=reports;evidence['dump_sha256']=dumps
    evidence['validation']=dict(explicit_validation=bool(validation),native_only_validation=bool(native_only_validation),native_activation_requested=legacy.validation_requested(validation,env),control_activation_requested=legacy.validation_requested(False,control_env),inherited_activation_environment={name:os.environ[name] for name in legacy.VALIDATION_ENVIRONMENT if name in os.environ},control_removed_activation_environment=removed,native_modes_report_checked=[],native_required_layer_report_checked=False,control_required_layer_checked=False,overall_validation_qualified=False,control_validation_status='not_requested_explicit_native_only_split' if native_only_validation else 'activation_requested_required_layer_guard_unavailable' if legacy.validation_requested(False,control_env) else 'not_requested_runtime_activation_unverified',scope='native_required_layer_and_report_counters_if_requested; frozen_RHI_has_no_required_layer_guard; clean_diagnostics_are_not_control_validation_qualification')
    try:
        identity=native_identities(rhi,native,compiler,fixture_directory/'manifest.json');evidence['identities']=identity
        def verify_inputs():
            for path,digest in {**fixture_hashes,**control_hashes}.items():
                if base.sha256(path)!=digest:raise ValueError('bound geometry/observation/control fixture bytes changed')
        def child(name,command,environment):
            verify_inputs();result=subprocess.run(command,env=environment,capture_output=True,text=True,errors='replace')
            for suffix,text in (('.stdout',result.stdout),('.stderr',result.stderr)):
                path=output/(name+suffix);path.write_text(text);reports[path.name]=base.sha256(path)
            evidence['children'][name]=dict(command=command,returncode=result.returncode,validation_activation_environment={key:environment[key] for key in legacy.VALIDATION_ENVIRONMENT if key in environment},validation_activation_requested=legacy.validation_requested(False,environment),diagnostics=dict(stdout=legacy.parse_diagnostics(result.stdout),stderr=legacy.parse_diagnostics(result.stderr)),diagnostic_scan_clean=False)
            base.require_child_clean(result.returncode,result.stdout,result.stderr);evidence['children'][name]['diagnostic_scan_clean']=True;verify_inputs()
        args=[]
        for option,key in (('target-width','width'),('target-height','height'),('fx','fx'),('fy','fy'),('cx','cx'),('cy','cy'),('hfov-radians','hfov_radians'),('projection','projection')):args+=['--'+option,str(view[key])]
        args+=['--virtual-from-original-left',','.join(repr(float(v)) for v in np.asarray(view['virtual_from_original_left']).ravel())]
        gain=forced_gain(geometry,planes,images)
        for mode in ('strict','preview'):
            prefix=output/(mode+'-daxa');native_report=output/(mode+'-daxa.json');cmd=[str(native),'--manifest',str(fixture_directory/'manifest.json'),'--observations-prefix',str(fixture_directory/'observations'),'--output',str(native_report),'--dump-prefix',str(prefix),'--disparities','16','--paths','4','--slots','2','--wave-size','32','--mode',mode,*args]
            if validation:cmd+=['--validation']
            child(mode+'-daxa',cmd,env);report=base.strict_json(native_report.read_text());reports[native_report.name]=base.sha256(native_report);validate_report(report,manifest,metadata,geometry,identity,mode,validation);actual=load_output(prefix,count)
            if validation:evidence['validation']['native_modes_report_checked'].append(mode)
            evidence['validation']['native_required_layer_report_checked']=len(evidence['validation']['native_modes_report_checked'])==2
            for key,plane in (('rgb_valid_coverage','rgb_validity.u32'),('metric_valid_coverage','validity.u32')):
                if report[key]!=int((np.frombuffer(actual[plane],'<u4')==1).sum())/count:raise ValueError('reported coverage differs from actual exported masks')
            expected=expected_known_output(geometry['geometry'],view,planes,geometry['witnesses'],[images['rect_left'],images['rect_right']],[images['original_left'],images['original_right']],mode,gain);expected={name:data.tobytes() for name,data in expected.items()};rows={}
            control_raw=None
            if view['projection']=='perspective':
                rhi_prefix=output/(mode+'-rhi');rhi_report=output/(mode+'-rhi.json');child(mode+'-rhi',[str(rhi),'--fixture',str(control),'--mode',mode,'--input','resident','--readback','gpu','--frames','1','--warmup','0','--output',str(rhi_report),'--dump-prefix',str(rhi_prefix)],control_env)
                r=base.strict_json(rhi_report.read_text());reports[rhi_report.name]=base.sha256(rhi_report)
                # The frozen binary cannot attest required-layer activation. Never trust
                # added report claims to turn an output comparison into validation proof.
                if 'validation_requested' in r and (type(r['validation_requested']) is not bool or r['validation_requested']!=evidence['validation']['control_activation_requested']):raise ValueError('frozen RHI control validation request claim differs from actual environment')
                if any(key in r and r[key] is not False for key in ('validation_checked','validation_qualified','required_validation_layer_checked')):raise ValueError('frozen RHI control validation qualification unavailable')
                required=dict(mode=mode,input='resident',readback='gpu',frames=1,warmup=0,source_width=math.prod(geometry['geometry']['rect_size'])+math.prod(geometry['geometry']['raw_size']),source_height=1,target_width=view['width'],target_height=view['height'],python_interpreter_used=False,timestamp_ns=100,calibration_id='synthetic-independent-known3D-control')
                if any(key not in r or preprocess.canonical(r[key])!=preprocess.canonical(value) for key,value in required.items()) or r.get('device')!=report['device'].get('device'):raise ValueError('frozen RHI control report profile/device changed')
                control_raw=load_output(rhi_prefix,count)
                for name in PLANES:dumps[str(Path(str(rhi_prefix)+'.'+name))]=base.sha256(Path(str(rhi_prefix)+'.'+name))
                control_raw=normalize_control_owners(control_raw,geometry)
            for name,data in actual.items():
                file=Path(str(prefix)+'.'+name);dumps[str(file)]=base.sha256(file);expected_file=output/(mode+'-expected.'+name);expected_file.write_bytes(expected[name]);dumps[str(expected_file)]=base.sha256(expected_file)
                rows[name]=dict(bytes=len(data),exact_known_point_oracle=data==expected[name],actual_sha256=dumps[str(file)],expected_sha256=dumps[str(expected_file)],exact_frozen_compositor=None if control_raw is None else data==control_raw[name],frozen_compositor_matches_known_oracle=None if control_raw is None else control_raw[name]==expected[name])
            evidence['comparisons'][mode]=rows
        verify_inputs()
        if identity!=native_identities(rhi,native,compiler,fixture_directory/'manifest.json'):raise ValueError('source/compiler/binary/SPIRV/dependency identity changed')
        if any(base.sha256(output/name)!=digest for name,digest in reports.items()) or any(base.sha256(path)!=digest for path,digest in dumps.items()):raise ValueError('bound report/dump bytes changed')
        evidence.update(passed=all(row['exact_known_point_oracle'] and row['exact_frozen_compositor'] is not False and row['frozen_compositor_matches_known_oracle'] is not False for rows in evidence['comparisons'].values() for row in rows.values()),fixture_file_sha256=fixture_hashes,control_file_sha256=control_hashes,report_sha256=reports,dump_sha256=dumps,frozen_compositor_control_checked=view['projection']=='perspective',owner_normalization='ONLY frozencontrol eye*M+rectIndex→eye*rectN+rectIndex; invalid/preview→UINT_MAX; allother10planes unchanged',rhi_validation_boundary='separately_recorded_activation_environment_and_unconditional_stdout_stderr_scan; frozen_RHI_binary_has_no_required_layer_guard; no_control_or_overall_validation_qualification',rhi_spirv_identity='not_exposed_by_frozen_RHI_runtime; bound_source_compiler_binary_dependencies_headers_configuration_and_actual_output_bytes')
    except (ValueError,OSError,KeyError,TypeError,OverflowError,struct.error,subprocess.SubprocessError) as error:evidence['error']=str(error)
    (output/'comparison.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n');return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    export=sub.add_parser('export');export.add_argument('--output',type=Path,required=True);export.add_argument('--case',choices=['tie','conflict','missing','unsupported','preview','shared-raw-cell'],default='tie');export.add_argument('--projection',choices=['perspective','cylindrical'],default='perspective');export.add_argument('--rgb-pattern',choices=['constant','periphery'],default='constant')
    export.add_argument('--virtual-pose',choices=['identity','rotate-z90'],default='identity')
    run=sub.add_parser('compare');run.add_argument('--fixture',type=Path,required=True);run.add_argument('--output',type=Path,required=True);run.add_argument('--compiler',type=Path,required=True);run.add_argument('--native',type=Path,default=ROOT/'build/daxa/daxa_view_depth_fixture');run.add_argument('--rhi',type=Path,default=ROOT/'build/standalone/native_view_benchmark');run.add_argument('--validation',action='store_true',help='Request validation for both native and frozen control (no automatic fallback)');run.add_argument('--native-only-validation',action='store_true',help='Requires --validation; explicitly omit validation activation for frozen control only; no control validation qualification')
    args=parser.parse_args()
    if args.action=='export':export_fixture(args.output,args.case,args.projection,args.rgb_pattern,args.virtual_pose);print(json.dumps(dict(exported=str(args.output))))
    else:
        result=run_comparison(args.fixture,args.native,args.rhi,args.compiler,args.output,args.validation,args.native_only_validation);print(json.dumps(dict(passed=result['passed'],output=str(args.output))))
        if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
