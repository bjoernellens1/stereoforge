#!/usr/bin/env python3
"""CPU fixture exporter and bounded native preprocessing comparison.

Fixture creation uses OpenCV only offline. Native execution is explicit, preserves
recorded metadata and compares all requested GPU dumps exactly after timing.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
import shutil
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from compare_daxa_stereo import sha256,strict_json,create_output,require_child_clean

SCHEMA='stereoforge.daxa_preprocess_fixture.v1'
PROFILE='opencv414_yuyv20_remap32_rte_rgb_halfup_gray15'
PLANES={'original_left_rgb':'original_left.rgb8','original_right_rgb':'original_right.rgb8',
        'rect_left_rgb':'rect_left.rgb8','rect_right_rgb':'rect_right.rgb8',
        'left_luma':'left_luma.u32','right_luma':'right_luma.u32',
        'left_support':'left_support.u32','right_support':'right_support.u32'}
IMAGE_CONVENTION='x_right_y_down_z_forward; maps_rectified_to_original; right_to_left'
SOURCE_COMMIT='0654a42e19215ef25b1d367d822f3c630447e7c7'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def read_capture_record(path,ordinal):
    """Read existing SFPv1 through one exact ordinal; preserve opaque float bits.

    Whole-file SHA is bound separately. Existing records through the selected one
    are checksum-validated. This is not an alternate recording format or live path.
    """
    if type(ordinal) is not int or ordinal<0:raise ValueError('invalid record ordinal')
    with Path(path).open('rb') as source:
        def read(n):
            data=source.read(n)
            if len(data)!=n:raise ValueError('truncated packed capture')
            return data
        magic,version,count=struct.unpack('<8sIQ',read(20))
        if magic!=b'SFYUYV1\0' or version!=1 or count>10000000 or ordinal>=count:raise ValueError('invalid/nonfinalized packed capture or ordinal')
        for index in range(ordinal+1):
            checksum=14695981039346656037
            def data(n):
                nonlocal checksum
                raw=read(n)
                for byte in raw:checksum=((checksum^byte)*1099511628211)&0xffffffffffffffff
                return raw
            def number(fmt):return struct.unpack('<'+fmt,data(struct.calcsize('<'+fmt)))
            def text():
                length=number('I')[0]
                if length>65536:raise ValueError('packed metadata exceeds bound')
                try:return data(length).decode('utf-8')
                except UnicodeDecodeError as error:raise ValueError('packed text is not UTF8') from error
            values=number('IIIQQQ');metadata=dict(zip(('source_width','height','stride_bytes','frame_id','camera_timestamp_ns','host_ingress_timestamp_ns'),values))
            for key in ('camera_clock_domain','host_clock_domain','calibration_identity','device_identity','controls_json','control_status'):metadata[key]=text()
            metadata['controls_timestamp_ns']=number('Q')[0];metadata['controls_clock_domain']=text()
            for key in ('sensor_status','imu_clock_domain','imu_association'):metadata[key]=text()
            metadata['sensor_dropped_count']=number('Q')[0];samples=number('I')[0]
            if samples>8192:raise ValueError('packed IMU count exceeds bound')
            metadata['imu_samples']=[]
            for _ in range(samples):
                stamp,*values,sync=number('Q7IB')
                if sync not in (0,1):raise ValueError('invalid packed IMU sync')
                metadata['imu_samples'].append(dict(timestamp_ns=stamp,acceleration_bits=values[:3],angular_velocity_bits=values[3:6],temperature_bits=values[6],sync=bool(sync)))
            length=number('Q')[0];width,height,stride=metadata['source_width'],metadata['height'],metadata['stride_bytes']
            if not width or width%4 or width>8192 or not 0<height<=8192 or stride!=width*2 or length!=stride*height or length>256*1024*1024 or not metadata['camera_timestamp_ns'] or not metadata['camera_clock_domain'] or not metadata['host_clock_domain']:
                raise ValueError('invalid packed dimensions/stride/timestamp/domain')
            payload=data(length)
            if struct.unpack('<Q',read(8))[0]!=checksum:raise ValueError('packed capture checksum mismatch')
            if index==ordinal:
                if index+1==count and source.read(1):raise ValueError('trailing packed capture bytes')
                return metadata,payload


def support_mask(mx,my,width,height):
    import numpy as np
    if mx.dtype!=np.float32 or my.dtype!=np.float32 or mx.shape!=my.shape or not np.isfinite(mx).all() or not np.isfinite(my).all() or np.max(np.abs(mx))>32768 or np.max(np.abs(my))>32768:
        raise ValueError('finite CV32FC1 map coordinates within +/-32768 required')
    ix=np.rint(mx*32).astype(np.int64);iy=np.rint(my*32).astype(np.int64);x=ix>>5;y=iy>>5;fx=ix&31;fy=iy&31
    result=np.ones(mx.shape,bool)
    for dx,dy,weight in ((0,0,(32-fx)*(32-fy)),(1,0,fx*(32-fy)),(0,1,(32-fx)*fy),(1,1,fx*fy)):
        result&=(weight==0)|((x+dx>=0)&(x+dx<width)&(y+dy>=0)&(y+dy<height))
    return result.astype('<u4')


def export_fixture(capture,ordinal,calibration,output):
    import cv2
    import numpy as np
    from stereoforge.calibration import Rectifier
    if cv2.__version__!='4.14.0':raise ValueError('frozen fixture exporter requires cv24.14.0')
    source_hash=sha256(capture);metadata,payload=read_capture_record(capture,ordinal)
    width,height=metadata['source_width']//2,metadata['height']
    if not metadata['calibration_identity'] or metadata['calibration_identity']!=calibration.identity:raise ValueError('missing/mismatched original calibration identity')
    if any((c.width,c.height)!=(width,height) for c in (calibration.left,calibration.right)):raise ValueError('calibration dimensions differ from capture')
    if not 0<width<32767 or not 0<height<32767 or width%2:raise ValueError('invalid frozen preprocessing dimensions')
    rectifier=Rectifier(calibration)
    packed=np.frombuffer(payload,np.uint8).reshape(height,width*2,2)
    decoded=cv2.cvtColor(packed,cv2.COLOR_YUV2RGB_YUY2)
    images={};maps={}
    for eye,index in (('left',0),('right',1)):
        original=decoded[:,index*width:(index+1)*width].copy();mx,my=rectifier._maps[index]
        support=support_mask(mx,my,width,height);rect=cv2.remap(original,mx,my,cv2.INTER_LINEAR)
        images['original_'+eye+'_rgb']=original;images['rect_'+eye+'_rgb']=rect
        images[eye+'_luma']=cv2.cvtColor(rect,cv2.COLOR_RGB2GRAY).astype('<u4');images[eye+'_support']=support
        maps[eye+'_x']=mx.astype('<f4');maps[eye+'_y']=my.astype('<f4')
    output=Path(output);create_output(output)
    def save(name,raw):
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as file:file.write(raw)
        return dict(file=name,sha256=sha256(path),bytes=len(raw))
    (output/'capture').mkdir();shutil.copyfile(capture,output/'capture/capture.sfp')
    if sha256(capture)!=source_hash or sha256(output/'capture/capture.sfp')!=source_hash:raise ValueError('capture bytes changed while exporting')
    original=save('original-calibration.json',canonical(calibration.to_dict()));rectified=save('rectified-calibration.json',canonical(rectifier.rectified_calibration.to_dict()))
    cv_binary=next(Path(cv2.__file__).parent.glob('*.so'))
    manifest=dict(schema=SCHEMA,profile=PROFILE,source=dict(eye_width=width,height=height,stride_bytes=metadata['stride_bytes']),rectified=dict(width=width,height=height),
        capture=dict(file='capture/capture.sfp',sha256=source_hash,record_ordinal=ordinal),
        calibration=dict(original_identity=calibration.identity,rectified_identity=rectifier.rectified_calibration.identity,original_json=original,rectified_json=rectified,R1=rectifier.R1.tolist(),R2=rectifier.R2.tolist(),P1=rectifier.P1.tolist(),P2=rectifier.P2.tolist()),
        maps={key:save('maps/'+key+'.f32',array.tobytes()) for key,array in maps.items()},
        expected={key:save('expected/'+suffix,images[key].tobytes()) for key,suffix in PLANES.items()},
        provenance=dict(image_convention=IMAGE_CONVENTION,cv2_version=cv2.__version__,cv2_binary_sha256=sha256(cv_binary),opencv_source_commit=SOURCE_COMMIT,exporter_sha256=sha256(__file__),
            stage_order='decode RGB8; split lenses; remap RGB8 linear constant0; gray15; widen u32',support_semantics='all nonzero-weight quantized bilinear taps inbounds; diagnostic, does not mask RGB/luma',
            source_sha256={name:sha256(ROOT/name) for name in ('python/stereoforge/capture.py','python/stereoforge/calibration.py','python/stereoforge/pipeline.py','python/stereoforge/types.py','src/capture.cpp')}))
    (output/'manifest.json').write_bytes(canonical(manifest))
    validate_fixture(output/'manifest.json')
    return manifest


def fixture_file(root,entry):
    name=entry['file']
    if not isinstance(name,str):raise ValueError('invalid fixture path')
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:raise ValueError('unsafe fixture path')
    path=root/relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):raise ValueError('missing/unsafe fixture file')
    if sha256(path)!=entry['sha256'] or ('bytes' in entry and (type(entry['bytes']) is not int or path.stat().st_size!=entry['bytes'])):raise ValueError('fixture hash/size mismatch')
    return path


def validate_fixture(path):
    import numpy as np
    from stereoforge.types import Calibration
    path=Path(path);root=path.parent;manifest=strict_json(path.read_text())
    if manifest.get('schema')!=SCHEMA or manifest.get('profile')!=PROFILE:raise ValueError('unknown preprocessing schema/profile')
    source,rect=manifest['source'],manifest['rectified'];w,h=source['eye_width'],source['height'];rw,rh=rect['width'],rect['height']
    if any(type(x) is not int or not 0<x<32767 for x in (w,h,rw,rh)) or w%2 or type(source['stride_bytes']) is not int or source['stride_bytes']!=w*4:raise ValueError('invalid preprocessing dimensions/stride')
    capture=fixture_file(root,manifest['capture']);metadata,_=read_capture_record(capture,manifest['capture']['record_ordinal'])
    if (metadata['source_width'],metadata['height'],metadata['stride_bytes'])!=(w*2,h,source['stride_bytes']):raise ValueError('capture shape mismatch')
    cal=manifest['calibration']
    for side in ('original','rectified'):
        entry=cal[side+'_json'];file=fixture_file(root,entry);model=Calibration.from_dict(strict_json(file.read_text()))
        if file.read_bytes()!=canonical(model.to_dict()) or model.identity!=cal[side+'_identity'] or entry['sha256']!=model.identity:raise ValueError('calibration hash/identity mismatch')
        if any((c.width,c.height)!=((w,h) if side=='original' else (rw,rh)) for c in (model.left,model.right)):raise ValueError('calibration shape mismatch')
    if not metadata['calibration_identity'] or metadata['calibration_identity']!=cal['original_identity']:raise ValueError('capture calibration mismatch')
    for key,shape in (('R1',(3,3)),('R2',(3,3)),('P1',(3,4)),('P2',(3,4))):
        a=np.asarray(cal[key]);
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('invalid rectification matrix')
    if set(manifest['maps'])!={'left_x','left_y','right_x','right_y'} or set(manifest['expected'])!=set(PLANES):raise ValueError('missing map/output plane')
    for key,entry in manifest['maps'].items():
        file=fixture_file(root,entry)
        if entry['bytes']!=rw*rh*4:raise ValueError('map byte layout mismatch')
        a=np.frombuffer(file.read_bytes(),'<f4')
        if not np.isfinite(a).all() or (np.abs(a)>32768).any():raise ValueError('invalid frozen map coordinates')
    for key,entry in manifest['expected'].items():
        fixture_file(root,entry);pixels=w*h if key.startswith('original') else rw*rh
        if entry['bytes']!=pixels*(3 if key.endswith('_rgb') else 4):raise ValueError('expected output byte layout mismatch')
    provenance=manifest.get('provenance',{})
    if provenance.get('image_convention')!=IMAGE_CONVENTION:raise ValueError('missing/unknown image convention')
    if provenance.get('cv2_version')!='4.14.0' or provenance.get('opencv_source_commit')!=SOURCE_COMMIT:raise ValueError('missing/unknown OpenCV profile provenance')
    for key in ('cv2_binary_sha256','exporter_sha256'):
        value=provenance.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('missing source-profile digest')
    return manifest,metadata


TIMINGS=('upload_cpu_ms','enqueue_cpu_ms','wait_cpu_ms','timestamp_query_cpu_ms','host_wall_ms')
GPU_STAGES=('upload','decode','rectify','total')


def validate_report(report,manifest,metadata,identity,frames,warmup,slots,validation,stereo,resident=False):
    expected=dict(schema='stereoforge.daxa_preprocess.v1',profile=PROFILE,source=manifest['source'],rectified=manifest['rectified'],
        scope='repeated_retained_record_staged_two_pass_no_lr',source_metadata=metadata,calibration={key:manifest['calibration'][key] for key in ('original_identity','rectified_identity')},frames=frames,warmup=warmup,slots=slots,validation_requested=validation,
        fixture_manifest_sha256=identity['manifest_sha256'],expected_planes_checked=True,
        shader_source_sha256_pair=identity['shader_source_sha256_pair'])
    for key,value in expected.items():
        if key not in report or canonical(report[key])!=canonical(value):raise ValueError('changed/missing native preprocessing contract: '+key)
    def timing(row,key):
        if not isinstance(row,dict) or set(row)!={'p50','p95','mean'} or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in row.values()):raise ValueError('invalid native preprocessing timing: '+key)
    if stereo:
        actual=report.get('stereo')
        flags=dict(integer_reference_checked=not resident,packed_penalties_reference_checked=not resident,compound_float_reference_checked=False)
        if not isinstance(actual,dict) or set(actual)!=set(stereo)|{'gpu_ms','reference_scope'}|set(flags) or canonical({key:actual[key] for key in stereo})!=canonical(stereo):raise ValueError('changed stereo bridge profile')
        if actual['reference_scope']!='integer_and_packed_checks_require_check_reference_and_dump_prefix':raise ValueError('changed stereo reference scope')
        for key,value in flags.items():
            if type(actual[key]) is not bool or actual[key] is not value:raise ValueError('changed stereo reference check: '+key)
        timing(actual['gpu_ms'],'stereo.gpu_ms')
    elif report.get('stereo','missing') is not None:raise ValueError('unexpected/missing stereo bridge profile')
    for key in TIMINGS:timing(report.get(key),key)
    gpu=report.get('preprocess_gpu_ms')
    if not isinstance(gpu,dict) or set(gpu)!=set(GPU_STAGES):raise ValueError('missing preprocessing GPU stage timings')
    for key in GPU_STAGES:timing(gpu[key],key)
    count=report.get('completed_input_retirement_count')
    if type(count) is not int or count!=frames+warmup:raise ValueError('missing/changed completed input retirement')
    for key in ('validation_errors_after_teardown','validation_warnings_after_teardown'):
        if type(report.get(key)) is not int or report[key]!=0:raise ValueError('native preprocessing teardown diagnostics')
    device=report.get('device',{})
    if device.get('slangc_sha256')!=identity['compiler_sha256']:raise ValueError('actual preprocessing compiler identity mismatch')
    for key in ('validation_errors','validation_warnings'):
        if type(device.get(key)) is not int or device[key]!=0:raise ValueError('native preprocessing validation diagnostics')


JSON_HEADER_SHA256='9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6'


def optional_dependency_identity(directory):
    dependency=subprocess.run(['ninja','-C',str(directory),'-t','deps','CMakeFiles/stereoforge_daxa_preprocess.dir/src/daxa_preprocess.cpp.o'],capture_output=True,text=True,errors='replace')
    require_child_clean(dependency.returncode,dependency.stdout,dependency.stderr)
    paths=[Path(line.strip()) for line in dependency.stdout.splitlines() if line.startswith('    ')]
    headers=[path for path in paths if path.name=='json.hpp' or 'openssl' in path.parts]
    json_headers=[path for path in headers if path.name=='json.hpp']
    if len(json_headers)!=1 or sha256(json_headers[0])!=JSON_HEADER_SHA256:raise ValueError('actual optional JSON compiler header missing or differs from pin')
    if not any(path.name=='evp.h' for path in headers):raise ValueError('actual OpenSSL compiler headers missing')
    return {str(path.resolve()):sha256(path) for path in headers}


def native_identities(binary,compiler,manifest_path,stereo=False):
    from compare_daxa_view import dependencies
    manifest,metadata=validate_fixture(manifest_path);root=Path(manifest_path).parent
    getter=subprocess.run([str(binary),'--shader-directory'],capture_output=True,text=True,errors='replace')
    require_child_clean(getter.returncode,getter.stdout,getter.stderr)
    directory=Path(getter.stdout.strip()).resolve(strict=True)
    wrapper=ROOT/'python/stereoforge/shaders/daxa/preprocess.slang';ops=ROOT/'python/stereoforge/shaders/ops/preprocess.slang'
    paths=[Path(manifest_path)]+[fixture_file(root,manifest['capture'])]
    for key in ('original_json','rectified_json'):paths.append(fixture_file(root,manifest['calibration'][key]))
    for section in ('maps','expected'):
        paths.extend(fixture_file(root,entry) for entry in manifest[section].values())
    entries=['daxa-preprocess-decode_yuyv.spv','daxa-preprocess-rectify_gray.spv']
    if stereo:
        entries+=['daxa-stereo-'+entry+'.spv' for entry in ('census','cost','aggregate_baseline','aggregate_wave','select_output','depth_convert','inspect','clear_packed',
            'aggregate_packed8','aggregate_packed16','select_packed','aggregate_specialized8','aggregate_specialized16','select_specialized8','select_specialized16',
            'aggregate_d1288','aggregate_d12816','select_quad8','select_quad16','aggregate_shuffle8','aggregate_shuffle16')]
    shader_paths=[wrapper,ops]
    if stereo:shader_paths.extend([ROOT/'python/stereoforge/shaders/daxa/stereo.slang',ROOT/'python/stereoforge/shaders/ops/sgm.slang'])
    sources=('src/daxa_preprocess.cpp','include/stereoforge/daxa_preprocess.hpp','apps/daxa_preprocess_benchmark.cpp','cmake/DaxaPreprocess.cmake',
        'src/capture.cpp','include/stereoforge/capture.hpp','src/daxa_stereo.cpp','include/stereoforge/daxa_stereo.hpp',
        'src/daxa_frame.cpp','include/stereoforge/daxa_frame.hpp','src/daxa_foundation.cpp','include/stereoforge/daxa_foundation.hpp',
        'include/stereoforge/contracts.hpp','CMakeLists.txt','cmake/Daxa.cmake','cmake/DaxaStereo.cmake')
    return dict(optional_dependency_header_sha256=optional_dependency_identity(directory),binary_sha256=sha256(binary),compiler_sha256=sha256(compiler),dependency_sha256=dependencies(binary,True)['sha256'],
        manifest_sha256=sha256(manifest_path),shader_source_sha256_pair=sha256(wrapper)+':'+sha256(ops),
        shader_source_sha256={str(path.relative_to(ROOT)):sha256(path) for path in shader_paths},
        source_sha256={name:sha256(ROOT/name) for name in sources},harness_sha256=sha256(__file__),
        fixture_file_sha256={str(path.resolve()):sha256(path) for path in paths},offline_shader_directory=str(directory),
        offline_spirv_sha256={name:sha256(directory/name) for name in entries})


def commands(binary,manifest,output,frames,warmup,slots,validation,stereo,resident):
    result=[str(binary),'--manifest',str(manifest),'--output',str(output/'native.json'),'--frames',str(frames),'--warmup',str(warmup),'--slots',str(slots),'--check-reference']
    if not resident:result+=['--dump-prefix',str(output/'gpu')]
    if validation:result.append('--validation')
    if stereo:
        result.append('--stereo')
        for key in ('disparities','paths','p1','p2','wave-size'):result+=['--'+key,str(stereo['required_wave_size'] if key=='wave-size' else stereo[key])]
    return result


def compare_outputs(output,manifest,fixture_root):
    rows={}
    for key,suffix in PLANES.items():
        entry=manifest['expected'][key];actual=Path(output)/('gpu.'+suffix);expected=fixture_file(fixture_root,entry)
        if not actual.is_file() or actual.stat().st_size!=entry['bytes']:raise ValueError('missing/incomplete GPU output: '+key)
        first,second=sha256(actual),sha256(expected);rows[key]=dict(bytes=entry['bytes'],gpu_sha256=first,expected_sha256=second,exact=first==second)
    return rows


def run_comparison(manifest_path,binary,compiler,output,frames=30,warmup=3,slots=1,validation=False,stereo=None,resident=False,validation_only=False):
    if type(frames) is not int or type(warmup) is not int or not 1<=frames<=300 or not 0<=warmup<=30 or slots not in (1,2) or (not validation_only and (frames<30 or warmup<3)):raise ValueError('bounded counts require30..300/3..30, smaller onlyvalidation')
    manifest_path=Path(manifest_path).resolve();manifest,metadata=validate_fixture(manifest_path)
    if stereo:
        expected_keys={'variant','disparities','paths','p1','p2','required_wave_size','scope'}
        if set(stereo)!=expected_keys or stereo['variant']!='packed-specialized' or stereo['scope']!='single_reference_no_lr' or stereo['disparities'] not in (16,32,64,128) or stereo['paths'] not in (4,8) or not 0<=stereo['p1']<=stereo['p2']<=10000 or stereo['required_wave_size'] not in (32,64):raise ValueError('invalid bridge stereo profile')
    output=Path(output).absolute();create_output(output)
    evidence=dict(schema=1,passed=False,selected=None,frames=frames,warmup=warmup,slots=slots,performance_qualified_counts=not validation_only,
        scope='Repeated exact recorded fixture, GPU preprocessing and optional single-reference stereo; no fresh-camera/LR/full-pipeline qualification',
        timing_contract='Upload/enqueue/wait and GPU stage scopes explicit; final native8-plane CPU oracle and optional dumps occur after all measured samples',
        resident_performance=resident,stereo=stereo)
    try:
        identity=native_identities(binary,compiler,manifest_path,bool(stereo));evidence['identities']=identity
        command=commands(binary,manifest_path,output,frames,warmup,slots,validation,stereo,resident);evidence['command']=command
        child=subprocess.run(command,capture_output=True,text=True,errors='replace')
        (output/'native.stdout').write_text(child.stdout);(output/'native.stderr').write_text(child.stderr);evidence['returncode']=child.returncode
        require_child_clean(child.returncode,child.stdout,child.stderr)
        report_hash=sha256(output/'native.json');report=strict_json((output/'native.json').read_text());evidence['report']=report
        validate_report(report,manifest,metadata,identity,frames,warmup,slots,validation,stereo,resident)
        if not resident:
            evidence['planes']=compare_outputs(output,manifest,manifest_path.parent)
            if not all(row['exact'] for row in evidence['planes'].values()):raise ValueError('GPU preprocessing output bytes differ from independent cv2 fixture')
        if identity!=native_identities(binary,compiler,manifest_path,bool(stereo)):raise ValueError('preprocessing source/compiler/binary/SPIRV/input/report identity changed')
        if sha256(output/'native.json')!=report_hash:raise ValueError('native preprocessing report bytes changed during validation')
        evidence['native_report_sha256']=report_hash;evidence['passed']=True
    except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as error:evidence['error']=str(error)
    (output/'comparison.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path);parser.add_argument('--record-ordinal',type=int,default=0);parser.add_argument('--calibration',type=Path)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--compiler',type=Path);parser.add_argument('--binary',type=Path,default=ROOT/'build/daxa/daxa_preprocess_benchmark')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--frames',type=int);parser.add_argument('--warmup',type=int);parser.add_argument('--slots',type=int,choices=[1,2],default=1)
    for name in ('validation','validation-only','resident-performance','stereo'):parser.add_argument('--'+name,action='store_true')
    for name,default in (('disparities',128),('paths',4),('p1',8),('p2',32),('wave-size',32)):parser.add_argument('--'+name,type=int,default=default)
    args=parser.parse_args()
    if args.export:
        if args.manifest or args.calibration is None:parser.error('export requires --calibration and no --manifest')
        from stereoforge.types import Calibration
        export_fixture(args.export,args.record_ordinal,Calibration.from_dict(strict_json(args.calibration.read_text())),args.output);return
    if args.manifest is None or args.compiler is None:parser.error('comparison requires --manifest and --compiler')
    stereo=dict(variant='packed-specialized',disparities=args.disparities,paths=args.paths,p1=args.p1,p2=args.p2,required_wave_size=args.wave_size,scope='single_reference_no_lr') if args.stereo else None
    result=run_comparison(args.manifest,args.binary.resolve(),args.compiler.resolve(),args.output,args.frames if args.frames is not None else 1 if args.validation_only else 30,
        args.warmup if args.warmup is not None else 0 if args.validation_only else 3,args.slots,args.validation,stereo,args.resident_performance,args.validation_only)
    print(json.dumps(dict(passed=result['passed'],output=str(args.output))))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
