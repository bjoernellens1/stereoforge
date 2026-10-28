"""Offline analytic bundle admission and byte-preserving portable export."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import struct

import pytest

from benchmarks.zed_comparison import schema
from benchmarks.zed_comparison.native_input import write_native_input
from test_zed_comparison_schema import record as base_record, reseal


def put(root, desc, data):
    path = root / desc['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    desc.update(sha256=sha256(data).hexdigest(), size_bytes=len(data))


@pytest.fixture
def native_record(tmp_path):
    record = base_record.__wrapped__(tmp_path)
    for name, desc in record['images'].items():
        rgb = desc['format'] == 'rgb8'
        stride = 15 if rgb else 5
        data = bytearray([0xe7] * (stride * 4))
        for y in range(4):
            for x in range(4):
                if rgb:
                    data[y*stride+x*3:y*stride+x*3+3] = bytes((x+y*4, 19+x, 223-y))
                else:
                    data[y*stride+x] = int((x, y) != (1, 2))
        desc.update(width=4, height=4, stride_bytes=stride)
        put(tmp_path, desc, bytes(data))
    for which in ('raw', 'rectified'):
        desc = record['calibration'][which]
        value = json.loads((tmp_path / desc['path']).read_text())
        for eye in ('left', 'right'):
            value[eye].update(width=4, height=4, fx=8., fy=8., cx=1.5 if eye == 'left' else 1.6, cy=1.5)
        put(tmp_path, desc, json.dumps(value).encode())
    geometry = record['calibration']['geometry']
    geometry.update(width=4, height=4, fx=8., fy=8., cx_left=1.5, cx_right=1.6)
    observation = record['observation']['value']
    for name, data, stride in (('depth', struct.pack('<16f', *([2.] * 16)), 16), ('validity', bytes([1]*16), 4)):
        observation[name].update(width=4, height=4, stride_bytes=stride)
        put(tmp_path, observation[name], data)
    artifact = dict(path='implementation/diagnostic.bin', sha256='', size_bytes=0)
    put(tmp_path, artifact, b'opaque implementation bytes\0\xff')
    record['implementation']['artifacts'].append(artifact)
    reseal(record)
    path = tmp_path / 'record.json'
    path.write_text(json.dumps(record, indent=2))
    supplement = dict(schema='stereoforge.canonical_geometry.v1', input_identity=record['input_identity'],
                      record_sha256=sha256(path.read_bytes()).hexdigest(),
                      raw_calibration_sha256=record['calibration']['raw']['sha256'],
                      rectified_calibration_sha256=record['calibration']['rectified']['sha256'],
                      R1=[[1,0,0],[0,1,0],[0,0,1]], R2=[[1,0,0],[0,1,0],[0,0,1]],
                      P1=[[8,0,1.5,0],[0,8,1.5,0],[0,0,1,0]],
                      P2=[[8,0,1.6,-.96],[0,8,1.5,0],[0,0,1,0]],
                      support_qualification='analytic_known_geometry', producer='independent analytic fixture')
    supplement_path = tmp_path / 'geometry.json'
    supplement_path.write_text(json.dumps(supplement, indent=2))
    return record, path, supplement, supplement_path


def test_bundle_preserves_complete_portable_record(native_record, tmp_path):
    record, record_path, supplement, supplement_path = native_record
    output = tmp_path / 'bundle'
    manifest_path = write_native_input(record_path, supplement_path, output)
    manifest = json.loads(manifest_path.read_text())
    assert manifest_path == output / 'native-input.json'
    assert schema.load_records([output / 'record.json']) == [record]
    assert (output / 'record.json').read_bytes() == record_path.read_bytes()
    assert (output / 'native-geometry.json').read_bytes() == supplement_path.read_bytes()
    for path in [*record['images'].values(), record['calibration']['raw'], record['calibration']['rectified'],
                 record['source']['artifact'], record['observation']['value']['depth'],
                 record['observation']['value']['validity'], record['implementation']['artifacts'][0]]:
        assert (output / path['path']).read_bytes() == (tmp_path / path['path']).read_bytes()
    assert manifest['sampling_profile'] == 'canonical_rgb8_box2x2_halfup_gray15_support_and4_halfpixel_v1'
    for key, names in [('original_pixel_identity', ('original_left', 'original_right')),
                       ('rectified_pixel_identity', ('rectified_left', 'rectified_right', 'support_left', 'support_right'))]:
        payload = {name: {k:v for k,v in record['images'][name].items() if k != 'path'} for name in names}
        expected = sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        assert manifest[key] == expected
    coarse = json.loads((output / 'native-coarse-calibration.json').read_text())
    full = json.loads((tmp_path / 'rect.json').read_text())
    for eye in ('left', 'right'):
        assert coarse[eye]['width'] == coarse[eye]['height'] == 2
        assert coarse[eye]['fx'] == coarse[eye]['fy'] == 4
        assert coarse[eye]['cx'] == (full[eye]['cx']+.5)/2-.5
        assert coarse[eye]['cy'] == .5
    assert coarse['right_to_left'] == full['right_to_left']
    assert manifest['coarse_calibration']['sha256'] == sha256((output/'native-coarse-calibration.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('change', ['missing_R1','bad_shape','boolean','reflection','inconsistent_rotation','P2_sign','P1_translation','record_hash','raw_hash','input_identity','unknown_key','qualification'])
def test_reject_bad_supplement(native_record, tmp_path, change):
    _, path, supplement, supplement_path = native_record
    if change == 'missing_R1': del supplement['R1']
    elif change == 'bad_shape': supplement['R1'] = [1,0,0,0,1,0,0,0,1]
    elif change == 'boolean': supplement['P1'][0][0] = True
    elif change == 'reflection': supplement['R1'][0][0] = -1
    elif change == 'inconsistent_rotation': supplement['R2'] = [[0,-1,0],[1,0,0],[0,0,1]]
    elif change == 'P2_sign': supplement['P2'][0][3] *= -1
    elif change == 'P1_translation': supplement['P1'][0][3] = 1e-10
    elif change == 'record_hash': supplement['record_sha256'] = '0'*64
    elif change == 'raw_hash': supplement['raw_calibration_sha256'] = '0'*64
    elif change == 'input_identity': supplement['input_identity'] = '0'*64
    elif change == 'unknown_key': supplement['coarse_identity'] = '0'*64
    elif change == 'qualification': supplement['support_qualification'] = 'sdk_qualified'
    supplement_path.write_text(json.dumps(supplement))
    with pytest.raises(ValueError): write_native_input(path, supplement_path, tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('kind', ['svo2', 'physical', 'middlebury'])
def test_reject_nonanalytic(native_record, tmp_path, kind):
    record, path, supplement, supplement_path = native_record
    record['source']['kind'] = kind
    reseal(record);path.write_text(json.dumps(record))
    supplement.update(input_identity=record['input_identity'], record_sha256=sha256(path.read_bytes()).hexdigest())
    supplement_path.write_text(json.dumps(supplement))
    with pytest.raises(ValueError): write_native_input(path,supplement_path,tmp_path/'bundle')


@pytest.mark.parametrize('failure', ['support','reserved','nested_reserved','unsafe','symlink','duplicate_json','output_exists','input_mutation'])
def test_reject_filesystem_and_content(native_record, tmp_path, monkeypatch, failure):
    record, path, supplement, supplement_path = native_record
    output = tmp_path/'bundle'
    if failure == 'support':
        desc=record['images']['support_left'];data=bytearray((tmp_path/desc['path']).read_bytes());data[0]=2;put(tmp_path,desc,bytes(data))
    elif failure in ('reserved','nested_reserved','unsafe'):
        desc=record['implementation']['artifacts'][0]
        desc['path']={'reserved':'native-input.json','nested_reserved':'native-input.json/nested','unsafe':'../outside'}[failure]
        if failure != 'unsafe': put(tmp_path,desc,b'artifact')
    elif failure == 'symlink':
        desc=record['source']['artifact'];target=tmp_path/'real-source';(tmp_path/desc['path']).rename(target);(tmp_path/desc['path']).symlink_to(target)
    elif failure == 'duplicate_json': supplement_path.write_text('{"schema":1,"schema":2}')
    elif failure == 'output_exists': output.mkdir();(output/'preserve').write_text('existing')
    elif failure == 'input_mutation':
        import benchmarks.zed_comparison.native_input as module
        original=module.shutil.copyfile
        def mutate(source,destination,*args,**kwargs):
            result=original(source,destination,*args,**kwargs)
            if Path(source)==path: path.write_bytes(path.read_bytes()+b'\n')
            return result
        monkeypatch.setattr(module.shutil,'copyfile',mutate)
    if failure in ('support','reserved','nested_reserved','unsafe'):
        reseal(record);path.write_text(json.dumps(record));supplement.update(input_identity=record['input_identity'],record_sha256=sha256(path.read_bytes()).hexdigest());supplement_path.write_text(json.dumps(supplement))
    with pytest.raises((ValueError, FileExistsError)): write_native_input(path,supplement_path,output)
    if failure == 'output_exists': assert (output/'preserve').read_text() == 'existing'
    else: assert not output.exists()


def test_known_pixel_and_coarse_digest_vectors(native_record, tmp_path):
    _, path, _, supplement = native_record
    manifest=json.loads(write_native_input(path,supplement,tmp_path/'bundle').read_text())
    assert manifest['original_pixel_identity'] == '380d8cc1becb902b098e771cf33a81708ec22a58b73430ec220588c1f3a118b9'
    assert manifest['rectified_pixel_identity'] == '94d9b419d83ed2df3744ce72069908d78da88a13c07d300a18a47d93ce088090'
    assert manifest['coarse_calibration']['sha256'] == '065cee7edbc3e2543b936e0253fe18c20bdb4675b3c3358d0e54babf481ae928'
    assert manifest['coarse_calibration']['sha256'] != native_record[0]['calibration']['rectified']['sha256']
    assert set(manifest) == {'schema','source_record','geometry_supplement','coarse_calibration','sampling_profile','original_pixel_identity','rectified_pixel_identity'}


def republish(record,path,supplement,supplement_path):
    reseal(record);path.write_text(json.dumps(record))
    supplement.update(input_identity=record['input_identity'],record_sha256=sha256(path.read_bytes()).hexdigest(),
                      raw_calibration_sha256=record['calibration']['raw']['sha256'],
                      rectified_calibration_sha256=record['calibration']['rectified']['sha256'])
    supplement_path.write_text(json.dumps(supplement))


def test_nested_resolved_and_implementation_dependencies(native_record,tmp_path):
    record,path,supplement,supplement_path=native_record
    desc=dict(path='calibration/opaque/resolved.bin',sha256='',size_bytes=0)
    put(tmp_path,desc,b'not JSON: opaque vendor bytes are not a camera model\0')
    record['calibration']['resolved_file']={'status':'available','value':desc}
    extra=dict(path='implementation/nested/extra.bin',sha256='',size_bytes=0)
    put(tmp_path,extra,b'exact nested content')
    record['implementation']['settings']['nested_reference']=extra
    republish(record,path,supplement,supplement_path)
    output=tmp_path/'bundle';write_native_input(path,supplement_path,output)
    for file in (desc,extra): assert (output/file['path']).read_bytes()==(tmp_path/file['path']).read_bytes()
    assert schema.load_records([output/path.name]) == [record]


@pytest.mark.parametrize('which',['record_parent','supplement','output_parent'])
def test_reject_symlink_components(native_record,tmp_path,which):
    _,path,_,supplement=native_record
    output=tmp_path/'bundle'
    link=tmp_path/'link'
    if which == 'record_parent':
        link.symlink_to(tmp_path,target_is_directory=True);path=link/path.name
    elif which == 'supplement':
        link.symlink_to(supplement);supplement=link
    else:
        link.symlink_to(tmp_path,target_is_directory=True);output=link/'bundle'
    with pytest.raises(ValueError):write_native_input(path,supplement,output)
    assert not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('change',['unknown_clock','zero_timestamp','odd_grid','nan','nesting','json_size'])
def test_native_bounds_and_strict_json(native_record,tmp_path,change):
    record,path,supplement,supplement_path=native_record
    if change in ('unknown_clock','zero_timestamp'):
        record['timestamps']['capture'].update(clock_domain='unknown' if change=='unknown_clock' else 'camera',value_ns=0 if change=='zero_timestamp' else 100)
        republish(record,path,supplement,supplement_path)
    elif change == 'odd_grid':
        for desc in record['images'].values():desc['width']=3
        for desc in (record['observation']['value']['depth'],record['observation']['value']['validity']):desc['width']=3
        for which in ('raw','rectified'):
            desc=record['calibration'][which];value=json.loads((tmp_path/desc['path']).read_text())
            for eye in ('left','right'):value[eye]['width']=3
            put(tmp_path,desc,json.dumps(value).encode())
        record['calibration']['geometry']['width']=3
        republish(record,path,supplement,supplement_path)
        schema.load_records([path])  # Valid portable grid, outside exact2x native admission.
    elif change == 'nan':
        supplement['P1'][0][0]=float('nan');supplement_path.write_text(json.dumps(supplement))
    elif change == 'nesting':
        nested=0
        for _ in range(66):nested=[nested]
        supplement['producer']=nested;supplement_path.write_text(json.dumps(supplement))
    else:supplement_path.write_bytes(supplement_path.read_bytes()+b' '*(16*1024*1024))
    with pytest.raises(ValueError):write_native_input(path,supplement_path,tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()


def test_explicit_nonidentity_raw_to_rectified_rotation(native_record,tmp_path):
    record,path,supplement,supplement_path=native_record
    desc=record['calibration']['raw'];raw=json.loads((tmp_path/desc['path']).read_text())
    rotation=[[0,-1,0],[1,0,0],[0,0,1]]
    for row in range(3):raw['right_to_left'][row][:3]=rotation[row]
    put(tmp_path,desc,json.dumps(raw).encode())
    supplement['R2']=rotation
    republish(record,path,supplement,supplement_path)
    manifest=write_native_input(path,supplement_path,tmp_path/'bundle')
    assert json.loads((manifest.parent/'native-geometry.json').read_text())['R2']==rotation


def write_analytic_record(directory, width=128, height=64, disparity=8):
    """Write independent frontoparallel-plane truth; return record/supplement paths.

    The procedural texture is sampled at integer coordinates. Both raw cameras
    have principal points one pixel larger than the canonical cameras, so exact
    rectification samples raw(x+1,y+1), with the last row/column unsupported.
    Right texture samples world(x+disparity,y); positive left disparity and
    z=fx*baseline/disparity therefore follow the calibrated pinhole equations.
    This is an analytic fixture, not measured camera or SDK output.
    """
    import numpy as np
    if any(type(v) is not int for v in (width,height,disparity)) or width<4 or height<4 or width%2 or height%2 or not 0<disparity<width-2:
        raise ValueError('expected even dimensions>=4 and positive interior integer disparity')
    root=Path(directory);root.mkdir(exist_ok=False)
    fx=width*.75;baseline=.12;cx=(width-1)/2;cy=(height-1)/2
    def file(name,data):
        desc=dict(path=name,sha256='',size_bytes=0);put(root,desc,data);return desc
    def plane(name,values,fmt,stride):
        values=np.ascontiguousarray(values)
        rows=values.view(np.uint8).reshape(height,-1)
        padded=np.full((height,stride),0xa7,dtype=np.uint8);padded[:,:rows.shape[1]]=rows
        return dict(**file(name,padded.tobytes()),width=width,height=height,stride_bytes=stride,format=fmt)
    def texture(x,y):
        # Explicit modulo-2^32 integer texture; no stochastic ground truth.
        x=np.asarray(x,dtype=np.int64).astype(np.uint32);y=np.asarray(y,dtype=np.int64).astype(np.uint32)
        channels=[]
        for channel in range(3):
            v=(x+np.uint32(1))*np.uint32(0x9e3779b9)^(y+np.uint32(3))*np.uint32(0x85ebca6b)^np.uint32(((channel+7)*0xc2b2ae35)&0xffffffff)
            v^=v>>np.uint32(16);v*=np.uint32(0x7feb352d);v^=v>>np.uint32(15);channels.append((v>>np.uint32(24)).astype(np.uint8))
        return np.stack(np.broadcast_arrays(*channels),axis=-1)
    y,x=np.indices((height,width),dtype=np.int64)
    support=((x+1<width)&(y+1<height)).astype(np.uint8)
    images={}
    for eye,shift in (('left',0),('right',disparity)):
        original=texture(x-1+shift,y-1)
        rectified=texture(x+shift,y);rectified[support==0]=0
        images['original_'+eye]=plane('images/original_'+eye+'.rgb',original,'rgb8',width*3+5)
        images['rectified_'+eye]=plane('images/rectified_'+eye+'.rgb',rectified,'rgb8',width*3+5)
        images['support_'+eye]=plane('images/support_'+eye+'.u8',support,'u8_mask',width+3)
    pose=[[1.,0.,0.,baseline],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
    def calibration(space):
        offset=1 if space=='raw' else 0
        camera=dict(width=width,height=height,fx=fx,fy=fx,cx=cx+offset,cy=cy+offset,
                    projection='perspective',distortion=dict(model='none',coefficients=[]))
        return dict(schema=schema.CALIBRATION_SCHEMA,space=space,left=deepcopy(camera),right=deepcopy(camera),right_to_left=deepcopy(pose),units='m',pixel_centres='integer_centres',coordinate_axes='x_right_y_down_z_forward')
    unavailable=dict(status='unavailable',reason='not a measured camera field')
    valid=((support!=0)&(x>=disparity)).astype(np.uint8)
    depth=np.full((height,width),np.nan,dtype='<f4');depth[valid!=0]=fx*baseline/disparity
    spec=dict(schema='stereoforge.analytic_plane_fixture.v1',family='frontoparallel_point_texture',width=width,height=height,disparity_pixels=disparity,depth_m=fx*baseline/disparity,rectified_to_raw_pixel_offset=[1,1],texture='uint32_modular_hash_v1')
    record=dict(schema=schema.FRAME_SCHEMA,scene_id='analytic-plane',sequence_id=f'analytic-{width}x{height}-d{disparity}',partition='development',ordinal=0,source_id=1,
                source=dict(kind='analytic',artifact=file('source/analytic-plane.json',json.dumps(spec,sort_keys=True).encode()),codec='analytic_point_texture',lossless=True),
                calibration=dict(raw=file('calibration/raw.json',json.dumps(calibration('raw'),sort_keys=True).encode()),rectified=file('calibration/rectified.json',json.dumps(calibration('rectified'),sort_keys=True).encode()),resolved_file=deepcopy(unavailable),
                                 geometry=dict(width=width,height=height,fx=fx,fy=fx,baseline_m=baseline,cx_left=cx,cx_right=cx,projection='perspective',depth_convention='axial_z',units='m',coordinate_frame='left_optical',pixel_centres='integer_centres',reference_camera='left',coordinate_axes='x_right_y_down_z_forward')),
                timestamps=dict(capture=dict(value_ns=1_000_000,clock_domain='analytic_fixture_ticks'),ingress=deepcopy(unavailable),sensors=[]),images=images,
                observation=dict(status='available',value=dict(depth=plane('observation/depth.f32',depth,'f32_le_m',width*4+4),validity=plane('observation/validity.u8',valid,'u8_mask',width+3),confidence=deepcopy(unavailable),confidence_direction='unavailable',provenance=deepcopy(unavailable),source_ids=deepcopy(unavailable),age_ns=deepcopy(unavailable))),
                implementation=dict(name='analytic_plane_fixture',version='1',commit=deepcopy(unavailable),device=deepcopy(unavailable),settings=spec,artifacts=[]),input_identity='pending')
    record['input_identity']=schema.input_identity(record)
    path=root/'record.json';path.write_text(json.dumps(record,sort_keys=True,indent=2))
    supplement=dict(schema='stereoforge.canonical_geometry.v1',input_identity=record['input_identity'],record_sha256=sha256(path.read_bytes()).hexdigest(),raw_calibration_sha256=record['calibration']['raw']['sha256'],rectified_calibration_sha256=record['calibration']['rectified']['sha256'],R1=[[1,0,0],[0,1,0],[0,0,1]],R2=[[1,0,0],[0,1,0],[0,0,1]],P1=[[fx,0,cx,0],[0,fx,cy,0],[0,0,1,0]],P2=[[fx,0,cx,-fx*baseline],[0,fx,cy,0],[0,0,1,0]],support_qualification='analytic_known_geometry',producer='analytic_frontoparallel_plane_point_texture_raw_shift_1_v1')
    supplement_path=root/'geometry.json';supplement_path.write_text(json.dumps(supplement,sort_keys=True,indent=2))
    schema.load_records([path])
    return path,supplement_path


def test_independent_analytic_fixture_geometry_and_shift(tmp_path):
    import numpy as np
    path,supplement=write_analytic_record(tmp_path/'analytic',width=16,height=8,disparity=4)
    record=schema.load_records([path])[0]
    def image(name,channels=3):
        desc=record['images'][name];rows=np.frombuffer((path.parent/desc['path']).read_bytes(),dtype=np.uint8).reshape(8,desc['stride_bytes'])
        return rows[:,:16*channels].reshape(8,16,channels)
    original=image('original_left');left=image('rectified_left');right=image('rectified_right');support=image('support_left',1)[:,:,0]
    assert np.array_equal(left[:-1,:-1],original[1:,1:])
    assert np.array_equal(right[:-1,:-5],left[:-1,4:-1])
    assert np.all(support[:-1,:-1]==1) and np.all(support[-1]==0) and np.all(support[:,-1]==0)
    manifest=write_native_input(path,supplement,tmp_path/'bundle')
    assert schema.load_records([manifest.parent/'record.json']) == [record]


def test_byte_ceiling_rejects_before_portable_file_scans(native_record,tmp_path,monkeypatch):
    record,path,_,supplement=native_record
    record['images']['original_left']['size_bytes']=512*1024*1024+1
    path.write_text(json.dumps(record))
    def forbidden(*args,**kwargs):raise AssertionError('unbounded portable file scan reached')
    monkeypatch.setattr(schema,'load_records',forbidden)
    with pytest.raises(ValueError,match='byte ceiling'):write_native_input(path,supplement,tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()


def test_native_bridge_requires_left_optical_metric_frame(native_record, tmp_path):
    record, path, supplement, supplement_path = native_record
    record['calibration']['geometry']['coordinate_frame'] = 'another_named_optical_frame'
    reseal(record); path.write_text(json.dumps(record))
    supplement.update(input_identity=record['input_identity'], record_sha256=sha256(path.read_bytes()).hexdigest())
    supplement_path.write_text(json.dumps(supplement))
    schema.load_records([path])  # Valid portable label; native metric convention is narrower.
    with pytest.raises(ValueError, match='left_optical'):
        write_native_input(path, supplement_path, tmp_path/'bundle')
