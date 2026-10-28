#!/usr/bin/env python3
"""Independent diagnostic sequence protocol; native execution is a separate gate.

Uses reviewed scalar semantics, never production temporal helpers. Exact rational
float32 export and broader float differences are separate; no generic epsilon can
qualify projection, masks, lineage or unsupported floating-point domains.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np
import temporal_known_motion_oracle as oracle
import compare_daxa_stereo as base
import compare_daxa_depth as depth
import compare_daxa_view as legacy
ROOT=Path(__file__).resolve().parents[1]
SCHEMA='stereoforge.temporal.sequence.v1'
PLANES={'depth.f32':('depth','<f4',1),'confidence.f32':('confidence','<f4',1),'validity.u32':('valid','<u4',1),'provenance.u32':('provenance','<u4',1),'source_ids.i64':('source_ids','<i8',1),'age_ns.i64':('age_ns','<i8',1),'contributor_ids.i64':('ancestry_ids','<i8',4),'contributor_count.u32':('count','<u4',1),'oldest_age_ns.i64':('oldest_age_ns','<i8',1),'reuse.u32':('reuse','<u4',1),'origin_mask.u32':('origin_mask','<u4',1)}
FRESH_NAMES=('depth','confidence','validity','provenance','source_ids','age_ns')
NATIVE_SOURCE_FILES=('apps/daxa_temporal_fixture.cpp','include/stereoforge/daxa_temporal.hpp','src/daxa_temporal.cpp','include/stereoforge/daxa_observation.hpp','src/daxa_observation.cpp','include/stereoforge/daxa_calibration.hpp','src/daxa_calibration.cpp','src/daxa_view_calibration.cpp','src/daxa_view.cpp','python/stereoforge/shaders/daxa/temporal.slang','python/stereoforge/shaders/ops/temporal.slang','CMakeLists.txt')
SOURCE_FILES=NATIVE_SOURCE_FILES+('cmake/DaxaTemporal.cmake','benchmarks/compare_daxa_temporal.py','benchmarks/temporal_known_motion_oracle.py')
WORK_KEYS=dict(history_scanned='source_pixels_scanned',history_eligible='eligible_history_candidates',history_projected='projected_history_candidates',history_winners='visible_history_pixels',filled='history_fill_pixels',fused='fused_pixels')


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def layout():
    return {name.split('.')[0]:dict(dtype=dtype,endian='little',layout='eye_y_x_contributor' if multiple==4 else 'eye_y_x',elements_per_pixel=multiple) for name,(_,dtype,multiple) in PLANES.items()}


def safe_file(directory,name):
    if not isinstance(name,str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe relative file')
    path=(Path(directory)/name).resolve()
    if not path.is_relative_to(Path(directory).resolve()):raise ValueError('file escapes bound directory')
    return path


def normalized_config(raw):
    if not isinstance(raw,dict) or set(raw)!=set(dataclasses.asdict(oracle.Config())):raise ValueError('configuration keys differ')
    out=dict(raw)
    for name in ('relative_z_tolerance','min_confidence'):
        v=out[name]
        if isinstance(v,(bool,str)) or not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>np.finfo(np.float32).max:raise ValueError('invalid finite float32 configuration')
        out[name]=float(np.float32(v))
    cfg=oracle.Config(**out)
    if cfg.max_reuse<1:raise ValueError('native reuse cap must be positive')
    return cfg


def integer_array(value,shape,name,low,high):
    a=np.asarray(value)
    if a.shape!=shape or a.dtype.kind not in 'iu' or any(not low<=int(v)<=high for v in a.flat):raise ValueError('invalid integer plane '+name)
    return a


def fresh_frame(row,shape):
    raw=row['fresh']
    if not isinstance(raw,dict) or set(raw)!=set(FRESH_NAMES):raise ValueError('fresh six planes missing/extra')
    valid=integer_array(raw['validity'],shape,'validity',0,1).astype(bool)
    z=np.asarray(raw['depth'],dtype=object)
    if z.shape!=shape:raise ValueError('depth shape differs')
    decoded=np.full(shape,np.nan,np.float64)
    for i in np.ndindex(shape):
        value=z[i]
        if valid[i]:
            if isinstance(value,(bool,str)) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<=np.finfo(np.float32).max:raise ValueError('invalid float32 metric depth')
            decoded[i]=float(np.float32(value))
            if decoded[i]<=0:raise ValueError('depth underflows float32')
        elif value is not None:raise ValueError('invalid depth must be JSON null')
    confidence=np.asarray(raw['confidence'])
    if confidence.shape!=shape or confidence.dtype.kind not in 'ifu' or not np.isfinite(confidence).all() or np.any((confidence<0)|(confidence>1)):raise ValueError('invalid confidence')
    provenance=integer_array(raw['provenance'],shape,'provenance',0,5)
    ids=integer_array(raw['source_ids'],shape,'source IDs',-1,oracle.I64_MAX)
    age=integer_array(raw['age_ns'],shape,'age',0,oracle.I64_MAX)
    if np.any(~valid&((confidence!=0)|(provenance!=0)|(ids!=-1)|(age!=0))):raise ValueError('noncanonical invalid six-plane metadata')
    if np.any(valid&(~np.isin(provenance,[1,5])|(ids<0))):raise ValueError('raw fresh provenance/identity invalid')
    return oracle.fresh_frame(decoded,confidence=confidence.astype('<f4').astype(float),provenance=provenance,source_ids=ids,age_ns=age,timestamp_ns=row['host_ingress_timestamp_ns'],capture_id=row['capture_id'],calibration_id=row['calibration_id'],device_id=row['device_id'],clock_domain=row['host_clock_domain'])


def parse_pose(row):
    raw=row['pose']
    if raw is None:return None
    if set(raw)!= {'world_from_original_left','timestamp_ns','clock_domain'}:raise ValueError('pose fields differ')
    a=np.asarray(raw['world_from_original_left'],dtype=float)
    if a.shape!=(16,):raise ValueError('pose must be row-major16')
    oracle.rigid(a.reshape(4,4))
    return oracle.TimedPose(a.reshape(4,4),raw['timestamp_ns'],raw['clock_domain'])


def load_sequence(path):
    path=Path(path);data=base.strict_json(path.read_text())
    if data.get('schema')!=SCHEMA or data.get('float_profile') not in ('exact_rational','general_float') or data.get('plane_layout')!=layout():raise ValueError('sequence schema/float/dtype layout differs')
    cfg=normalized_config(data['config']);g=data['geometry'];width=oracle.integer(g['width'],'width');height=oracle.integer(g['height'],'height');shape=(2,height,width)
    support=integer_array(g['support'],shape,'support',0,1).astype(bool)
    oracle.Geometry(g['intrinsics'],g['eye_to_rig'],support)
    if g.get('profile') not in ('native_depth_compatible','diagnostic_only'):raise ValueError('geometry profile required')
    if g['profile']=='native_depth_compatible':
        k=np.asarray(g['intrinsics'])
        if not np.array_equal(k[0,[0,1,3]],k[1,[0,1,3]]):raise ValueError('native rectified fx/fy/cy incompatible')
    producer=data.get('producer_manifest')
    if producer is not None:
        p=safe_file(path.parent,producer['file'])
        if sha256(p)!=producer['sha256']:raise ValueError('producer manifest hash differs')
    if not isinstance(data['frames'],list) or not data['frames']:raise ValueError('nonempty sequence required')
    for row in data['frames']:
        if row.get('qualification_scope') not in ('native','native_host_rejection','diagnostic_reference_only') or type(row.get('reset')) is not bool:raise ValueError('frame scope/reset invalid')
        for name in ('capture_id','host_ingress_timestamp_ns'):oracle.integer(row[name],name)
        try:fresh_frame(row,shape);parse_pose(row)
        except (ValueError,TypeError,KeyError) as error:
            if row['qualification_scope'] not in ('diagnostic_reference_only','native_host_rejection'):raise ValueError('invalid sequence frame: '+str(error)) from error
            # Malformed fresh planes can only be a diagnostic precheck, never
            # evidence that a closed native Depth producer scans them at runtime.
        if row['qualification_scope']=='native_host_rejection':fresh_frame(row,shape)
    return data


def domain_counts():return dict(fresh_domain_rejected=0,history_input_domain_rejected=0,projected_domain_rejected=0,arithmetic_faults=0)


def normal_history(history,geometry,motions,timestamp_ns,config):
    """Disclosed native-normal-domain adapter; frozen scalar oracle is unchanged.

    Domain filtering precedes global visibility so a rejected tiny candidate
    cannot hide a supported normal candidate. All counts are per source pixel.
    Geometry remains float64 reference math, not an emulation of GPU float32.
    """
    out=history.copy();counts=domain_counts();tiny=float(np.finfo(np.float32).tiny);elapsed=timestamp_ns-history.timestamp_ns
    for index in np.ndindex(history.depth.shape):
        if not history.valid[index]:continue
        z=float(history.depth[index])
        if z<tiny:
            oracle.clear_pixel(out,index);counts['history_input_domain_rejected']+=1;continue
        age=int(history.age_ns[index])+elapsed;oldest=int(history.oldest_age_ns[index])+elapsed;reuse=int(history.reuse[index])+1
        if elapsed<0 or not geometry.support[index] or age>oracle.I64_MAX or oldest>oracle.I64_MAX or oldest>config.max_age_ns or reuse>config.max_reuse or history.count[index]>config.max_contributors or history.confidence[index]<config.min_confidence:continue
        eye,y,x=index;fx,fy,cx,cy=geometry.intrinsics[eye];q=motions[eye]@np.array([(x-cx)*z/fx,(y-cy)*z/fy,z,1.])
        if np.isfinite(q).all() and 0<q[2]<tiny:
            oracle.clear_pixel(out,index);counts['projected_domain_rejected']+=1
    return out,counts


def normal_process(stage,current,pose,reset):
    # expected_sequence applies the explicit reset command before validating
    # row arguments. This flag preserves the scalar reset reason only.
    current=current.copy();counts=domain_counts();tiny=float(np.finfo(np.float32).tiny)
    for index in np.ndindex(current.depth.shape):
        if not stage.geometry.support[index]:oracle.clear_pixel(current,index);continue
        if current.valid[index] and current.depth[index]<tiny:oracle.clear_pixel(current,index);counts['fresh_domain_rejected']+=1
    original=stage.history
    eligible=original is not None and stage.pose is not None and pose is not None and not reset and current.calibration_id==original.calibration_id and current.device_id==original.device_id and current.clock_domain==original.clock_domain and current.timestamp_ns>original.timestamp_ns and current.capture_id>original.capture_id and current.timestamp_ns-original.timestamp_ns<=stage.config.max_age_ns
    if eligible:
        motions=oracle.eye_motion(stage.geometry,stage.pose.world_from_rig,pose.world_from_rig)
        stage.history,history_counts=normal_history(original,stage.geometry,motions,current.timestamp_ns,stage.config)
        for key,value in history_counts.items():counts[key]+=value
    try:out,work=stage.process(current,pose,reset=reset)
    except Exception:
        stage.history=original
        raise
    return out,work,counts


def expected_sequence(sequence):
    g=sequence['geometry'];support=np.asarray(g['support'],bool);geometry=oracle.Geometry(g['intrinsics'],g['eye_to_rig'],support);cfg=normalized_config(sequence['config']);stage=oracle.TemporalOracle(geometry,cfg);rows=[]
    for index,row in enumerate(sequence['frames']):
        result=dict(index=index,qualification_scope=row['qualification_scope'],accepted=False)
        saved_history,saved_pose,saved_calibration=stage.history,stage.pose,stage.calibration_id
        # The actual fixture skips diagnostic rows entirely; their reset flags
        # never touch native history. For real rows reset is a separate command
        # completed BEFORE fresh/pose validation, even if admission then fails.
        reset_command=row['reset'] and row['qualification_scope']!='diagnostic_reference_only'
        if reset_command:stage.history=None;stage.pose=None
        try:
            frame=fresh_frame(row,support.shape);pose=parse_pose(row)
            out,work,counters=normal_process(stage,frame,pose,reset_command);result.update(accepted=True,work=work,domain_counters=counters,planes={name:getattr(out,attr).astype(dtype).tobytes() for name,(attr,dtype,_) in PLANES.items()})
        except (ValueError,TypeError,KeyError,OverflowError) as error:
            if row['qualification_scope']=='native':raise ValueError('unexpected scalar rejection: '+str(error)) from error
            result['reference_rejection']=str(error)
        if row['qualification_scope']=='diagnostic_reference_only':stage.history,stage.pose,stage.calibration_id=saved_history,saved_pose,saved_calibration
        if row['qualification_scope']=='native_host_rejection' and result['accepted']:raise ValueError('declared native host rejection accepted by scalar')
        rows.append(result)
    return dict(schema='stereoforge.temporal.expected.v1',native_executed=False,general_float_qualified=False,frames=rows)


def export_sequence(directory,case='identity',float_profile='exact_rational'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    if case not in ('identity','translation','rotation','missing-pose','limits','provenance5','ancestry-cap','shared-id','host-rejection','boundaries','age-overflow','unsupported','ties','visibility','explicit-reset','unsupported-subnormal'):raise ValueError('unknown fixture case')
    shape=(2,5,5);A=np.array([np.eye(4)]*2);A[1,0,3]=.25
    config=dataclasses.asdict(oracle.Config());config.update(relative_z_tolerance=.25,min_confidence=.5)
    data=dict(schema=SCHEMA,float_profile=float_profile,case=case,scope='synthetic_literal_geometry_not_physical_truth',producer_manifest=None,plane_layout=layout(),config=config,geometry=dict(width=5,height=5,intrinsics=[[1.,1.,2.,2.],[1.,1.,2.,2.]],eye_to_rig=A.tolist(),support=np.ones(shape,np.uint32).tolist(),profile='native_depth_compatible'),frames=[])
    if case=='boundaries':config['max_age_ns']=20
    if case=='age-overflow':config['max_age_ns']=oracle.I64_MAX
    if case=='unsupported-subnormal':data['geometry']['support'][0][2][2]=0
    if case=='unsupported':data['geometry']['support'][0][2][3]=0;data['geometry']['support'][1][2][3]=0
    count=7 if case in ('limits','ancestry-cap','missing-pose') else 3
    for i in range(count):
        z=np.full(shape,np.nan);z[:,2,2]=2.;valid=np.isfinite(z);p=np.eye(4)
        if case=='translation':p[0,3]=-float(i)
        if case=='rotation':
            z[:]=np.nan
            if i==0:z[:,2,3]=2.
            else:p[:2,:2]=[[0.,-1.],[1.,0.]]
            valid=np.isfinite(z)
        if case=='limits' and i:z[:]=np.nan;valid=np.isfinite(z)
        if case in ('unsupported','ties','visibility','boundaries','age-overflow') and i:z[:]=np.nan;valid=np.isfinite(z)
        if case=='unsupported' and i:p[0,3]=-1.
        if case in ('ties','visibility'):
            if i==0:z[:,2,3]=2. if case=='ties' else 4.;z[:,2,2]=2. if case=='ties' else 1.;valid=np.isfinite(z)
            elif case=='ties':p[2,3]=-4.
            else:p[0,3]=-1.
        source=1 if case=='shared-id' else i+1
        row=dict(capture_id=i+1,host_ingress_timestamp_ns=i*10,host_clock_domain='host',device_id='synthetic-device',calibration_id='synthetic-calibration',reset=False,qualification_scope='native',pose=dict(world_from_original_left=p.ravel().tolist(),timestamp_ns=i*10,clock_domain='host'),fresh=dict(depth=np.where(valid,z,None).tolist(),confidence=np.where(valid,.5,0.).tolist(),validity=valid.astype(np.uint32).tolist(),provenance=np.where(valid,5 if case=='provenance5' else 1,0).tolist(),source_ids=np.where(valid,source,-1).tolist(),age_ns=np.zeros(shape,np.int64).tolist()))
        if case=='boundaries':row['host_ingress_timestamp_ns']=[0,20,21][i];row['pose']['timestamp_ns']=[0,20,21][i]
        if case=='age-overflow' and i==0:row['fresh']['age_ns'][0][2][2]=oracle.I64_MAX
        if case in ('ties','visibility') and i==0:
            row['fresh']['confidence'][0][2][3]=1.;row['fresh']['confidence'][1][2][3]=1.
        if case=='unsupported-subnormal' and i==0:
            for eye in (0,1):row['fresh']['depth'][eye][2][2]=float(np.nextafter(np.float32(0),np.float32(1)))
        if case=='explicit-reset':
            row['capture_id']=[10,1,2][i];row['host_ingress_timestamp_ns']=[100,0,10][i];row['pose']['timestamp_ns']=[100,0,10][i];row['reset']=i==1
            for eye in (0,1):row['fresh']['source_ids'][eye][2][2]=row['capture_id']
            if i==2:
                for eye in (0,1):
                    row['fresh']['depth'][eye][2][2]=None
                    for name in ('confidence','validity','provenance','age_ns'):row['fresh'][name][eye][2][2]=0
                    row['fresh']['source_ids'][eye][2][2]=-1
        if case=='missing-pose' and i==1:row['pose']=None
        if case=='host-rejection' and i==1:row['pose']['timestamp_ns']=0;row['qualification_scope']='native_host_rejection'
        data['frames'].append(row)
    path=directory/'sequence.json';path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');load_sequence(path);return path


def collect_identities(native,compiler):
    native=Path(native).resolve();compiler=Path(compiler).resolve()
    if not native.is_file() or not compiler.is_file():raise ValueError('native fixture/compiler unavailable; no native qualification')
    names=set(SOURCE_FILES)|{str(p.relative_to(ROOT)) for directory,extension in (('src','*.cpp'),('include','*.hpp'),('python/stereoforge/shaders','*.slang')) for p in (ROOT/directory).rglob(extension)}
    source={name:sha256(ROOT/name) for name in sorted(names)}
    spvs={str(p.resolve()):sha256(p) for p in native.parent.rglob('*.spv')}
    if not any('temporal' in Path(p).name.lower() for p in spvs):raise ValueError('compiled temporal SPIRV unavailable')
    # ldd load addresses vary under ASLR. Stable identity retains the complete
    # resolved library path->content hash map, never the address transcript.
    return dict(source_sha256=source,binary_sha256=sha256(native),compiler_sha256=sha256(compiler),dependency_sha256=legacy.dependencies(native,True)['sha256'],build_configuration_sha256={str(p):sha256(p) for p in (native.parent/'CMakeCache.txt',native.parent/'build.ninja')},spirv_sha256=spvs,compiler_input_sha256=depth.compiler_header_identities(native.parent,(('daxa_temporal_fixture','apps/daxa_temporal_fixture.cpp'),('stereoforge_daxa_temporal','src/daxa_temporal.cpp'),('stereoforge_daxa_foundation','src/daxa_observation.cpp'),('stereoforge_daxa_calibration','src/daxa_calibration.cpp'),('stereoforge_daxa_view','src/daxa_view.cpp'),('stereoforge_daxa_view','src/daxa_view_calibration.cpp'),('stereoforge_daxa_depth','src/daxa_depth.cpp'))))


def run_comparison(sequence_path,native,compiler,output,validation=False):
    path=Path(sequence_path).resolve();sequence=load_sequence(path);expected=expected_sequence(sequence);output=Path(output).resolve()
    if output.is_relative_to(path.parent):raise ValueError('output must be outside bound input directory')
    base.create_output(output)
    evidence=dict(schema='stereoforge.daxa_temporal.comparison.v1',passed=False,native_executed=False,physical_qualified=False,performance_qualified=False,general_float_qualified=False,exact_float_profile_checked=False,sequence_sha256=sha256(path),float_profile=sequence['float_profile'],comparisons=[],diagnostic_reference_only_frames=sum(r['qualification_scope']=='diagnostic_reference_only' for r in expected['frames']))
    output_hashes={}
    inputs={str(p.resolve()):sha256(p) for p in path.parent.rglob('*') if p.is_file()};evidence['input_sha256']=inputs
    def unchanged():
        if any(sha256(p)!=digest for p,digest in inputs.items()):raise ValueError('bound sequence/producer input changed')
    try:
        if all(row['qualification_scope']=='diagnostic_reference_only' for row in expected['frames']):raise ValueError('diagnostic-only sequence is not native qualification')
        identity=collect_identities(native,compiler);evidence['identities']=identity;unchanged();report_path=output/'native.json';dump=output/'planes';command=[str(native),'--sequence',str(path),'--output',str(report_path),'--dump-directory',str(dump)]
        if validation:command+=['--validation']
        env=os.environ.copy()
        if validation:env.update(SF_DAXA_VALIDATION='1',VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation')
        result=subprocess.run(command,env=env,capture_output=True,text=True,errors='replace');evidence['native_executed']=True;evidence['command']=command
        evidence['child_logs']={}
        for name,value in (('stdout',result.stdout),('stderr',result.stderr)):
            p=output/('native.'+name);p.write_text(value);output_hashes[str(p)]=sha256(p);evidence['child_logs'][name]=dict(sha256=sha256(p),diagnostics=legacy.parse_diagnostics(value))
        base.require_child_clean(result.returncode,result.stdout,result.stderr);unchanged();report=base.strict_json(report_path.read_text());evidence['report_sha256']=sha256(report_path);output_hashes[str(report_path)]=sha256(report_path)
        required=dict(schema='stereoforge.daxa_temporal.fixture.v1',sequence_sha256=evidence['sequence_sha256'],observations_overridden_for_fixture=True,measured_stereo_output=False,performance_qualified=False,validation_requested=bool(validation),validation_errors=0,validation_warnings=0)
        if any(key not in report or json.dumps(report[key])!=json.dumps(value) for key,value in required.items()):raise ValueError('native diagnostic report contract changed')
        reported=report.get('identities',{})
        if reported.get('source_sha256')!={name:identity['source_sha256'][name] for name in NATIVE_SOURCE_FILES} or any(reported.get(key)!=identity[key] for key in ('binary_sha256','compiler_sha256','spirv_sha256')):raise ValueError('actual native source/compiler/binary/SPIRV identity differs')
        binding=report.get('producer_binding',{})
        if binding!=dict(kind='literal_controlled_planes_after_actual_Depth',geometry_models_checked=True,calibration_map_association_checked=True):raise ValueError('actual controlled Depth/model association evidence missing')
        rows=report['frames']
        if len(rows)!=len(expected['frames']):raise ValueError('frame report count differs')
        for actual,wanted in zip(rows,expected['frames']):
            if wanted['qualification_scope']=='diagnostic_reference_only':
                if actual.get('accepted') is not False or actual.get('native_executed') is not False or actual.get('skip_reason')!='diagnostic_reference_only' or actual.get('qualification_scope')!='diagnostic_reference_only' or actual.get('index')!=wanted['index'] or actual.get('planes'):raise ValueError('diagnostic precheck claimed native qualification')
                continue
            for key in ('index','accepted','qualification_scope'):
                if type(actual.get(key)) is not type(wanted[key]) or actual[key]!=wanted[key]:raise ValueError('native frame status/scope differs')
            if actual.get('native_executed') is not True:raise ValueError('native frame admission not executed')
            if not wanted['accepted']:
                if actual.get('rejection_stage')!='actual_Temporal_host_admission' or actual.get('rejected_before_submission') is not True:raise ValueError('host rejection not from actual native admission')
                continue
            counters=actual.get('domain_counters')
            if counters!=wanted['domain_counters'] or any(type(v) is not int for v in counters.values()) or counters['arithmetic_faults']!=0:raise ValueError('native domain/arithmetic counters differ')
            work={key:wanted['work'][value] for key,value in WORK_KEYS.items()}
            work['history_eligible']+=wanted['domain_counters']['projected_domain_rejected']
            if actual.get('work_counters')!=work or any(type(v) is not int for v in actual['work_counters'].values()):raise ValueError('native work counters differ')
            if set(actual['planes'])!=set(PLANES):raise ValueError('missing/extra output plane')
            for name,(_,dtype,multiple) in PLANES.items():
                entry=actual['planes'][name];filename=f'frame-{wanted["index"]:04d}.{name}'
                if entry['file']!=filename:raise ValueError('unsafe/noncanonical output filename')
                p=safe_file(dump,entry['file']);raw=p.read_bytes();output_hashes[str(p)]=sha256(p);wanted_raw=wanted['planes'][name]
                if sha256(p)!=entry['sha256'] or len(raw)!=len(wanted_raw):raise ValueError('output hash/extent differs')
                row=dict(index=wanted['index'],plane=name,sha256=sha256(p),exact=raw==wanted_raw,float_word_mismatches=0)
                if dtype=='<f4':
                    a=np.frombuffer(raw,'<f4');b=np.frombuffer(wanted_raw,'<f4');a_bits=a.view('<u4');b_bits=b.view('<u4');row['float_word_mismatches']=int(np.count_nonzero(a_bits!=b_bits));finite=np.isfinite(a)&np.isfinite(b);row['max_positive_float_ulp_difference']=int(np.max(np.abs(a_bits[finite].astype(np.int64)-b_bits[finite].astype(np.int64)))) if finite.any() else 0
                    if np.any(np.isinf(a)):raise ValueError('nonfinite native metric output')
                evidence['comparisons'].append(row)
        unchanged()
        if identity!=collect_identities(native,compiler):raise ValueError('source/compiler/binary/SPIRV identity changed')
        if any(sha256(p)!=digest for p,digest in output_hashes.items()):raise ValueError('bound report/output/log changed')
        evidence['output_sha256']=output_hashes
        evidence['scope']='synthetic_temporal_numerical_protocol_only; no_measured_stereo_physical_or_performance_qualification'
        evidence['producer_manifest_bound']=sequence.get('producer_manifest') is not None
        evidence['native_depth_integration_qualified']=False  # Requires actual producer/calibration association evidence beyond literal overrides.
        evidence['exact_float_profile_checked']=sequence['float_profile']=='exact_rational';evidence['passed']=all(row['exact'] for row in evidence['comparisons'])
    except (ValueError,OSError,KeyError,TypeError,OverflowError,subprocess.SubprocessError) as error:evidence['error']=str(error)
    (output/'comparison.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n');return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='action',required=True)
    export=subs.add_parser('export');export.add_argument('--output',type=Path,required=True);export.add_argument('--case',default='identity')
    compare=subs.add_parser('compare');compare.add_argument('--sequence',type=Path,required=True);compare.add_argument('--native',type=Path,required=True);compare.add_argument('--compiler',type=Path,required=True);compare.add_argument('--output',type=Path,required=True);compare.add_argument('--validation',action='store_true')
    args=parser.parse_args()
    if args.action=='export':print(export_sequence(args.output,args.case))
    else:
        result=run_comparison(args.sequence,args.native,args.compiler,args.output,args.validation);print(json.dumps(dict(passed=result['passed'],native_executed=result['native_executed'])))
        if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
