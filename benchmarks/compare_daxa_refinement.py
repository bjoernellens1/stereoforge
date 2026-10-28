#!/usr/bin/env python3
"""Explicit guided-only native refinement diagnostic comparison, never a GPU import.

prepare creates a CPU-only identity freeze; compare requires its separately
supplied SHA256 before executing the native fixture. Matching integer stages,
branching predictions, masks and identities are exact. Continuous output checks
are conditional on the frozen precise SPIR-V profile and R0 operation bounds;
this harness never promotes global arithmetic, physical or live qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np

import compare_daxa_depth as depth
import compare_daxa_preprocess as fixture
from compare_daxa_stereo import sha256, strict_json
from stereoforge import refinement as oracle

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ('rectify', 'census', 'match', 'lr', 'metric')
PLANES = ('disparity.f32','confidence.f32','validity.u32','depth.f32','metric_confidence.f32','metric_validity.u32','provenance.u32','source_ids.i64','age_ns.i64','prediction.f32','prediction_validity.u32','lower.u32','upper.u32','reason.u32','method.u32','winner.u32','best_cost.u32','second_cost.u32','supported_candidates.u32','evaluated_candidates.u32','subpixel_applied.u32','raw_disparity.f32','raw_confidence.f32','raw_validity.u32','census.u32','census_validity.u32','luma.u32','support.u32','rectified_left.u32','rectified_right.u32')
COARSE_PLANES = ('disparity.f32','confidence.f32','validity.u32')
DEVICE_KEYS = ('device','device_uuid','vendor_id','device_id','driver_uuid','driver_version','driver_name','driver_info','daxa_commit','slang_version','slangc_sha256','compiler_route')
REFINEMENT_VERSION = '2026.19'
REFINEMENT_COMMIT = '0fb6b7587a94fa33d88141ec9930766eedc6f59a'
REFINEMENT_PACKAGE = 'bd6cfc47b7353b2cffa36a866b1584749f76f4b2dbebf6dda26bc3fb40dc3c0e'


def _same(a, b, label):
    if fixture.canonical(a) != fixture.canonical(b):
        raise ValueError('changed/malformed '+label)


def _safe(path, root=None):
    p = Path(os.path.abspath(path))
    if any(parent.is_symlink() for parent in (p, *p.parents)):
        raise ValueError('symlink input/output path: '+str(p))
    if root is not None and not p.is_relative_to(Path(root).absolute()):
        raise ValueError('dump path escapes fresh output')
    return p


def _file(root, entry):
    p = fixture.fixture_file(Path(root), entry)
    _safe(p)
    return p


def load_context(coarse_manifest, target_manifest):
    """Actually decode/remap both fixtures using the frozen preprocessing helper."""
    coarse_path = _safe(coarse_manifest); target_path = _safe(target_manifest)
    coarse, cm, _, _, _, _ = depth.fixture_planes(coarse_path)
    target, tm, luma, support, metric, _ = depth.fixture_planes(target_path)
    for key in ('capture', 'source'):
        if key == 'capture':
            _same([coarse[key]['sha256'],coarse[key]['record_ordinal']], [target[key]['sha256'],target[key]['record_ordinal']], key)
        else:
            _same(coarse[key], target[key], key)
    _same(coarse['calibration']['original_identity'], target['calibration']['original_identity'], 'original identity')
    for key in ('R1','R2'):
        _same(coarse['calibration'][key], target['calibration'][key], key)
    cw, ch = coarse['rectified']['width'], coarse['rectified']['height']
    w, h = target['rectified']['width'], target['rectified']['height']
    if (w,h) != (cw*2,ch*2):
        raise ValueError('refinement requires exact2x target grid')
    a = np.array([[2,0,.5],[0,2,.5],[0,0,1]],np.float64)
    for key in ('P1','P2'):
        if not np.array_equal(a@np.asarray(coarse['calibration'][key]), np.asarray(target['calibration'][key])):
            raise ValueError('half-pixel projection relation differs')
    if type(tm['frame_id']) is not int or not 0 <= tm['frame_id'] < 1 << 63:
        raise ValueError('source ID does not fit signed64')
    _same(cm, tm, 'source record metadata')
    images = np.stack([np.frombuffer(raw,'<u4').reshape(h,w) for raw in luma])
    supports = np.stack([np.frombuffer(raw,'<u4').reshape(h,w) for raw in support])
    rgb=[]
    for eye in ('left','right'):
        raw=np.frombuffer(_file(target_path.parent,target['expected']['rect_'+eye+'_rgb']).read_bytes(),np.uint8).reshape(h,w,3).astype(np.uint32)
        rgb.append(raw[...,0]|(raw[...,1]<<8)|(raw[...,2]<<16)|np.uint32(0xff000000))
    return dict(coarse_path=coarse_path,target_path=target_path,coarse=coarse,target=target,
                metadata=tm,source_id=tm['frame_id'],images=images,support=supports,rgb=rgb,
                geometry=oracle.Geometry(**{k:metric[k] for k in ('fx','baseline_m','cx_offset')}))


def expected_report(context, settings, validation):
    if settings.disparities != 128 or settings.recovery:
        raise ValueError('native comparator profile is guidedD128 only')
    t,c=context['target'],context['coarse']; g=context['geometry']
    return dict(schema='stereoforge.daxa_refinement.v1',percentile_policy='nearest_rank_ceil_p_times_N_minus1',
        scope='static_repeated_original_record; no_live_or_quality_claim',width=t['rectified']['width'],height=t['rectified']['height'],
        coarse_width=c['rectified']['width'],coarse_height=c['rectified']['height'],coarse_disparities=64,disparities=128,
        radius=settings.radius,subpixel=settings.subpixel,recovery=False,prior_confidence_floor=float(np.float32(settings.prior_confidence_floor)),
        max_coarse_span=float(np.float32(settings.prior_span)),min_margin=settings.min_margin,lr_tolerance=float(np.float32(settings.lr_tolerance)),
        estimator='independent-local-census',right_reference_mode='independent-local-census',lr_dependency='independent-references',
        coarse_manifest_sha256=sha256(context['coarse_path']),target_manifest_sha256=sha256(context['target_path']),
        capture_sha256=c['capture']['sha256'],record_ordinal=c['capture']['record_ordinal'],source_id=context['source_id'],
        target_rectified_identity=t['calibration']['rectified_identity'],coarse_rectified_identity=c['calibration']['rectified_identity'],
        float_qualification='unqualified; exact masks and operation-specific error bounds required',
        metric_calibration={k:float(getattr(g,k)) for k in ('fx','baseline_m','cx_offset')},actual_coarse_pipeline=True,diagnostic_injection=False,
        validation_requested=validation,validation_errors_after_teardown=0,validation_warnings_after_teardown=0)


def oracle_planes(result, context):
    """Nominal diagnostic layout. Signed -1 oracle sentinels map to UINT_MAX."""
    matches=result.matches
    def stack(name): return np.stack([getattr(m,name) for m in matches])
    values=dict(zip(('disparity.f32','confidence.f32','validity.u32','depth.f32','metric_confidence.f32','metric_validity.u32','provenance.u32','source_ids.i64','age_ns.i64'),
                    (result.lr_disparity,result.lr_confidence,result.lr_validity,result.depth,result.metric_confidence,result.metric_validity,result.provenance,result.source_ids,result.age_ns)))
    values.update({'prediction.f32':np.stack([p.disparity for p in result.predictions]),'prediction_validity.u32':np.stack([p.valid for p in result.predictions])})
    for wire, field in [('lower','lower'),('upper','upper'),('reason','reasons'),('method','method'),('winner','integer_winner'),('best_cost','best'),('second_cost','second'),('supported_candidates','candidates'),('evaluated_candidates','candidates'),('subpixel_applied','subpixel_applied')]:
        values[wire+'.u32']=stack(field).astype('<u4')
    for wire, field in [('raw_disparity','disparity'),('raw_confidence','confidence'),('raw_validity','validity')]: values[wire+('.u32' if field=='validity' else '.f32')]=getattr(result,field)
    census=[oracle.census(context['images'][e],context['support'][e]) for e in (0,1)]
    values.update({'census.u32':np.stack([c.descriptor for c in census]),'census_validity.u32':np.stack([c.support for c in census]),'luma.u32':context['images'],'support.u32':context['support'],
                   'rectified_left.u32':context['rgb'][0],'rectified_right.u32':context['rgb'][1]})
    return values


def _read_planes(entries, names, shape, dump_root, coarse=False):
    if not isinstance(entries,dict) or set(entries)!=set(names): raise ValueError('missing/extra native planes')
    result={}
    for name in names:
        entry=entries[name]
        if not isinstance(entry,dict) or set(entry)!=({'path','sha256'} if coarse else {'path','sha256','elements'}): raise ValueError('malformed dump descriptor')
        p=_safe(entry['path'],dump_root)
        dims=shape[1:] if name.startswith('rectified_') else shape
        count=int(np.prod(dims));dtype='<i8' if name.endswith('.i64') else '<f4' if name.endswith('.f32') else '<u4'
        if not p.is_file() or p.stat().st_size!=count*np.dtype(dtype).itemsize or sha256(p)!=entry['sha256']: raise ValueError('native plane hash/extent differs: '+name)
        if not coarse and (type(entry['elements']) is not int or entry['elements']!=count): raise ValueError('native element count differs')
        result[name]=np.frombuffer(p.read_bytes(),dtype).reshape(dims)
    return result


def _exact(name, actual, expected):
    dtype=actual.dtype
    if actual.shape!=expected.shape or actual.tobytes()!=np.asarray(expected,dtype=dtype).tobytes(): raise ValueError('exact plane differs: '+name)


def _invalid(name, actual, valid, confidence=False):
    bits=actual.view('<u4')
    if np.any(bits[~valid] != (0 if confidence else 0x7fc00000)):
        raise ValueError('noncanonical invalid '+name)
    if not np.isfinite(actual[valid]).all(): raise ValueError('nonfinite valid '+name)


def _unique_bounds(function, columns):
    keys=np.stack(columns,axis=-1);unique,inverse=np.unique(keys,axis=0,return_inverse=True)
    bounds=function(*[unique[:,i] for i in range(unique.shape[1])])
    return oracle.FloatBounds(bounds.lower[inverse],bounds.upper[inverse])


def _float_check(name, actual, nominal, mask, bounds):
    _invalid(name,actual,mask,confidence='confidence' in name)
    a=actual[mask].astype(np.float64)
    if np.any((a<bounds.lower)|(a>bounds.upper)): raise ValueError('operation-specific bound exceeded: '+name)
    return dict(different_nominal_bits=int(np.count_nonzero(actual[mask].view('<u4')!=nominal[mask].astype('<f4').view('<u4'))),
                max_nominal_abs_residual=float(np.max(np.abs(a-nominal[mask].astype(np.float64)),initial=0)))


def compare_report(report, context, frozen, dump_root, settings=oracle.Settings(radius=4,subpixel=False), validation=False, truth=None):
    expected=expected_report(context,settings,validation)
    for key,value in expected.items():
        if key not in report: raise ValueError('missing report field: '+key)
        _same(report[key],value,key)
    for key,value in frozen['report_identity'].items(): _same(report.get(key),value,key)
    for key,value in frozen['device_identity'].items(): _same(report.get('device',{}).get(key),value,'device.'+key)
    if type(report.get('requested_refinement_buffer_bytes')) is not int or report['requested_refinement_buffer_bytes']<=0: raise ValueError('invalid resource byte count')
    h,w=context['images'].shape[1:]; shape=(2,h,w)
    actual=_read_planes(report['files'],PLANES,shape,dump_root)
    coarse=_read_planes(report['coarse_files'],COARSE_PLANES,(2,h//2,w//2),dump_root,True)
    cv=coarse['validity.u32']
    if np.any(cv>1): raise ValueError('nonbinary coarse validity')
    _invalid('coarse disparity',coarse['disparity.f32'],cv!=0)
    _invalid('coarse confidence',coarse['confidence.f32'],cv!=0,True)
    result=oracle.refine_pair(context['images'],context['support'],coarse['disparity.f32'],coarse['confidence.f32'],cv,context['geometry'],context['source_id'],settings)
    nominal=oracle_planes(result,context)
    continuous={'disparity.f32','confidence.f32','depth.f32','metric_confidence.f32','raw_disparity.f32','raw_confidence.f32'}
    for name in PLANES:
        if name not in continuous: _exact(name,actual[name],nominal[name])
    raw=result.validity; lr=result.lr_validity; metric=result.metric_validity
    floats={}; n=int(raw.sum())
    if n:
        best=nominal['best_cost.u32'][raw].astype(np.int32); second=nominal['second_cost.u32'][raw].astype(np.int32)
        confidence=_unique_bounds(oracle.confidence_bounds,[best,second])
        floats['raw_confidence.f32']=_float_check('raw_confidence.f32',actual['raw_confidence.f32'],result.confidence,raw,confidence)
        lo=result.disparity[raw].astype(np.float64); hi=lo.copy()
        applied=nominal['subpixel_applied.u32'][raw]!=0
        if applied.any():
            eye,y,x=np.nonzero(raw);winner=nominal['winner.u32'][raw].astype(np.int64);desc=nominal['census.u32'];minus=[];plus=[]
            for delta,output in [(-1,minus),(1,plus)]:
                d=winner[applied]+delta;target=x[applied]+np.where(eye[applied]!=0,d,-d)
                output.extend(np.bitwise_count(desc[eye[applied],y[applied],x[applied]]^desc[1-eye[applied],y[applied],target]).tolist())
            bound=_unique_bounds(oracle.subpixel_bounds,[np.asarray(minus,np.int32),best[applied],np.asarray(plus,np.int32),winner[applied].astype(np.int32)])
            lo[applied]=bound.lower;hi[applied]=bound.upper
        disparity_bound=oracle.FloatBounds(lo,hi)
        floats['raw_disparity.f32']=_float_check('raw_disparity.f32',actual['raw_disparity.f32'],result.disparity,raw,disparity_bound)
    else:
        _invalid('raw_disparity',actual['raw_disparity.f32'],raw);_invalid('raw_confidence',actual['raw_confidence.f32'],raw,True)
    for name,raw_name,mask in [('disparity.f32','raw_disparity.f32',lr),('confidence.f32','raw_confidence.f32',lr),('metric_confidence.f32','raw_confidence.f32',metric)]:
        _invalid(name,actual[name],mask,confidence='confidence' in name)
        _exact(name,actual[name][mask],actual[raw_name][mask])
    if metric.any():
        in_metric=metric[raw]
        # Cache repeated interval endpoints before exact-rational diagnostic math.
        db=_unique_bounds(lambda low,high:oracle.metric_bounds(oracle.FloatBounds(low,high),context['geometry']),[lo[in_metric],hi[in_metric]])
        floats['depth.f32']=_float_check('depth.f32',actual['depth.f32'],result.depth,metric,db)
    else: _invalid('depth',actual['depth.f32'],metric)
    summary=dict(integer_masks_identity_exact=True,operation_bounds_checked=True,gpu_global_arithmetic_qualified=False,
        arithmetic_gate='conditional precise-operation bounds; exact nominal predicates required; global boundary profile unqualified',
        physical_qualified=False,live_qualified=False,float_residuals=floats,raw_valid=int(raw.sum()),lr_valid=int(lr.sum()),metric_valid=int(metric.sum()),
        supported_candidate_evaluations=int(nominal['evaluated_candidates.u32'].sum(dtype=np.uint64)))
    if truth is not None: summary['world_quality']=evaluate_quality(actual,truth,nominal['census_validity.u32']!=0)
    return summary


def load_truth(family_path,context):
    family_path=_safe(family_path);family=strict_json(family_path.read_text());root=family_path.parent
    if family.get('schema')!='stereoforge.refinement_quality_family.v1' or family.get('development_only') is not True: raise ValueError('unknown truth family')
    if family.get('generator_sha256')!=sha256(ROOT/'benchmarks/refinement_quality_fixtures.py'): raise ValueError('truth generator differs from reviewed source')
    for key in ('scene','original_capture','original_calibration'): _file(root,family[key])
    if family['original_capture']['sha256']!=context['target']['capture']['sha256'] or family['original_calibration']['sha256']!=context['target']['calibration']['original_identity']: raise ValueError('truth original input differs')
    profiles=family.get('profiles',{})
    selected=[v for v in profiles.values() if v['native_manifest']['sha256']==sha256(context['target_path'])]
    if len(selected)!=1: raise ValueError('truth target binding missing/ambiguous')
    profile=selected[0];_file(root,profile['native_manifest']);tp=_file(root,profile['truth_manifest']);truth=strict_json(tp.read_text())
    model=dict(context['target']['calibration'],**context['target']['rectified'])
    for key,value in dict(schema='stereoforge.analytic_stereo_truth.v1',generator_sha256=family['generator_sha256'],scene_sha256=family['scene']['sha256'],original_capture_sha256=family['original_capture']['sha256'],original_calibration_sha256=family['original_calibration']['sha256'],native_manifest_sha256=sha256(context['target_path']),projection_sha256=hashlib.sha256(fixture.canonical(model)).hexdigest(),rectified=context['target']['rectified'],disparities=128,world_from_left=family['world_from_left']).items(): _same(truth.get(key),value,'truth.'+key)
    output={};shape=context['images'].shape[1:]
    required={'axial_z_m','disparity','cross_u','cross_v','surface_id','valid','visible','occluded','off_image','out_of_range','mixed_bilinear','remap_support','region_near','region_far','region_thin','region_slant','region_repeated','region_lowtexture','region_edges'}
    if set(truth['eyes'])!={'left','right'}: raise ValueError('missing truth eye')
    for eye,entries in truth['eyes'].items():
        if set(entries)!=required: raise ValueError('missing/extra truth planes')
        arrays={}
        for name,e in entries.items():
            dtype='<f8' if name in ('axial_z_m','disparity','cross_u','cross_v') else '<i4' if name=='surface_id' else '|u1'
            _same(e.get('shape'),list(shape),'truth shape');_same(e.get('dtype'),dtype,'truth dtype')
            p=_file(tp.parent,e)
            if p.stat().st_size!=int(np.prod(shape))*np.dtype(dtype).itemsize: raise ValueError('truth byte extent differs')
            a=np.frombuffer(p.read_bytes(),dtype).reshape(shape)
            if dtype=='|u1' and np.any(a>1): raise ValueError('nonbinary truth mask')
            arrays[name]=a
        _exact('truth remap support',arrays['remap_support'].astype(np.uint32),context['support'][0 if eye=='left' else 1])
        output[eye]=arrays
    return output


def evaluate_quality(actual,truth,census_support):
    """Truth denominators include invalid estimates; never use SGM as truth."""
    result=dict(scope='analytic synthetic development; not physical or heldout',correct_disparity_threshold_pixels=1.,
                denominator='visible in-range centre-ray truth with full Census/remap support and no mixed raw footprint')
    for eye,name in enumerate(('left','right')):
        t=truth[name];visible=t['valid'].astype(bool)&t['visible'].astype(bool)
        domain=visible&~t['out_of_range'].astype(bool)&~t['off_image'].astype(bool)&t['remap_support'].astype(bool)&census_support[eye]&~t['mixed_bilinear'].astype(bool)
        raw=actual['raw_validity.u32'][eye]!=0;metric=actual['metric_validity.u32'][eye]!=0;lr=actual['validity.u32'][eye]!=0
        with np.errstate(invalid='ignore'): error=np.abs(actual['raw_disparity.f32'][eye].astype(np.float64)-t['disparity'])
        rows={}
        regions={'all':np.ones(domain.shape,bool),'mixed':t['mixed_bilinear'].astype(bool)}
        regions.update({k.removeprefix('region_'):v.astype(bool) for k,v in t.items() if k.startswith('region_')})
        for region,mask in regions.items():
            eligible=domain&mask;n=int(eligible.sum());rv=eligible&raw;mv=eligible&metric
            err=error[rv];zerr=np.abs(actual['depth.f32'][eye][mv].astype(np.float64)-t['axial_z_m'][mv])
            correct=rv&(error<=1)
            rows[region]=dict(region_pixels=int(mask.sum()),measurement_domain='ineligible_mixed_footprint_diagnostic' if region=='mixed' else 'clean_support_visible_in_range_truth',
                eligible=n,raw_accepted=int(rv.sum()),metric_accepted=int(mv.sum()),raw_completeness=int(rv.sum())/n if n else None,
                metric_completeness=int(mv.sum())/n if n else None,correct_raw_completeness=int(correct.sum())/n if n else None,
                correct_metric_completeness=int((mv&(error<=1)).sum())/n if n else None,
                lr_accepted_total=int((lr&mask).sum()),lr_accepted_outside_clean_domain=int((lr&mask&~domain).sum()),
                lr_accepted_out_of_range=int((lr&mask&t['out_of_range'].astype(bool)).sum()),
                lr_accepted_unsupported=int((lr&mask&(~t['remap_support'].astype(bool)|~census_support[eye])).sum()),
                lr_accepted_mixed=int((lr&mask&t['mixed_bilinear'].astype(bool)).sum()),
                lr_false_accepted=int((lr&mask&(~visible|(error>1))).sum()),raw_disparity_mae=float(err.mean()) if err.size else None,
                raw_disparity_max=float(err.max()) if err.size else None,metric_depth_mae_m=float(zerr.mean()) if zerr.size else None)
        result[name]=rows
    return result


def audit_spirv(path):
    """Inspect actual SPIR-V words, not a disassembly filename or report claim.

    Refinement uses scalar/vector FNegate/FAdd/FSub/FMul/FDiv/FRem/FMod. Reject
    grouped decorations, RelaxedPrecision, FPFastMathMode and GLSL Fma; every
    arithmetic result must carry NoContraction. This audit is a conditional
    operation-profile check, not a proof of all floating boundary predicates.
    """
    data=Path(path).read_bytes()
    if len(data)%4 or len(data)<20: raise ValueError('malformed SPIR-V extent')
    words=struct.unpack('<'+'I'*(len(data)//4),data)
    if words[0]!=0x07230203: raise ValueError('wrong SPIR-V magic')
    at=5;arithmetic=set();precise=set()
    while at<len(words):
        count=words[at]>>16;op=words[at]&65535
        if count<1 or at+count>len(words): raise ValueError('malformed SPIR-V instruction')
        args=words[at+1:at+count]
        if op==71:
            if len(args)<2: raise ValueError('malformed SPIR-V decoration')
            if args[1] in (0,40): raise ValueError('relaxed/fast-math SPIR-V forbidden')
            if args[1]==42: precise.add(args[0])
        if op in (74,75,332): raise ValueError('unsupported indirect SPIR-V decoration')
        if op in (127,129,131,133,136,140,141):
            if len(args)<2: raise ValueError('malformed floating instruction')
            arithmetic.add(args[1])
        if op==12 and len(args)>=4 and args[3]==50: raise ValueError('Fma SPIR-V forbidden')
        at+=count
    if arithmetic-precise: raise ValueError('floating operation lacks NoContraction')
    return dict(sha256=sha256(path),floating_arithmetic=len(arithmetic),nocontraction=len(precise),
                missing_nocontraction=[],relaxed_fastmath_fma_absent=True)


def _fixture_paths(path,manifest):
    paths=[Path(path),_file(Path(path).parent,manifest['capture'])]
    for key in ('original_json','rectified_json'): paths.append(_file(Path(path).parent,manifest['calibration'][key]))
    for section in ('maps','expected'):
        paths.extend(_file(Path(path).parent,entry) for entry in manifest[section].values())
    return paths


def _truth_paths(path):
    paths=[Path(path)];family=strict_json(Path(path).read_text());root=Path(path).parent
    for key in ('scene','original_capture','original_calibration'): paths.append(_file(root,family[key]))
    for profile in family['profiles'].values():
        paths.append(_file(root,profile['native_manifest']));tp=_file(root,profile['truth_manifest']);paths.append(tp)
        for entries in strict_json(tp.read_text())['eyes'].values(): paths.extend(_file(tp.parent,e) for e in entries.values())
    return paths


def _source_paths():
    paths=[]
    for directory in ('src','include/stereoforge','cmake'):
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.cpp','.hpp','.h','.cmake'))
    paths.extend((ROOT/'python/stereoforge/shaders').rglob('*.slang'))
    paths.extend(ROOT/name for name in ('CMakeLists.txt','python/stereoforge/refinement.py','python/stereoforge/calibration.py','python/stereoforge/types.py',
        'benchmarks/compare_daxa_refinement.py','benchmarks/compare_daxa_depth.py','benchmarks/compare_daxa_preprocess.py','benchmarks/compare_daxa_stereo.py',
        'benchmarks/compare_daxa_view.py','benchmarks/compare_native_resident.py','apps/daxa_refinement_fixture.cpp','apps/daxa_refinement_common.hpp',
        'tests/test_compare_daxa_refinement.py'))
    return paths


def _cold(command):
    result=subprocess.run([str(v) for v in command],capture_output=True,text=True,errors='replace',check=False)
    if result.returncode!=0: raise ValueError('cold identity command failed: '+result.stderr)
    return result.stdout.strip(),result.stderr.strip()


def runtime_dependencies(executable):
    """Bind loader and linked DSOs for each native/compiler executable.

    Header scripts are supported for CPU protocol fixtures by binding their real
    interpreter and its ELF closure. A changed SONAME/RPATH resolution changes
    this logical-path map even when the replacement happens to have equal bytes.
    """
    executable=Path(executable)
    with executable.open('rb') as stream: prefix=stream.read(4096)
    selected=executable;paths=[executable]
    if prefix.startswith(b'#!'):
        parts=prefix.splitlines()[0][2:].decode().split()
        if not parts or not Path(parts[0]).is_absolute(): raise ValueError('unbound executable interpreter')
        selected=Path(parts[0]);paths.append(selected)
    elif not prefix.startswith(b'\x7fELF'):
        raise ValueError('unsupported executable format for runtime dependency closure')
    out,err=_cold(['ldd',selected])
    if 'not found' in out or 'not found' in err: raise ValueError('unresolved runtime dependency')
    for line in out.splitlines():
        for token in line.split():
            if token.startswith('/'):
                p=Path(token)
                if not p.is_file(): raise ValueError('missing runtime dependency')
                paths.append(p)
    return {str(p.absolute()):sha256(p) for p in paths}


def prepare_freeze(native,coarse_manifest,target_manifest,coarse_compiler,refinement_compiler,audit,checkpoint,device_report,output,truth=None):
    """Cold getters/ldd/Ninja dependency reads only. Never initializes Vulkan."""
    context=load_context(coarse_manifest,target_manifest)
    native=_safe(native);coarse_compiler=_safe(coarse_compiler);refinement_compiler=_safe(refinement_compiler)
    for path,version in ((coarse_compiler,'2026.12'),(refinement_compiler,REFINEMENT_VERSION)):
        out,err=_cold([path,'-version'])
        if '\n'.join(v for v in (out,err) if v)!=version: raise ValueError('unexpected actual compiler version')
    out,err=_cold([native,'--shader-directory'])
    if err or not out or '\n' in out: raise ValueError('malformed cold shader-directory getter')
    directory=_safe(out)
    actual_audit={f'daxa-refinement-{e}.spv':audit_spirv(directory/f'daxa-refinement-{e}.spv') for e in ENTRIES}
    audit_path=_safe(audit);audited=strict_json(audit_path.read_text())
    _same(audited.get('modules'),actual_audit,'actual SPIR-V audit')
    checkpoint_path=_safe(checkpoint);checkpoint_values=strict_json(checkpoint_path.read_text())
    for name,digest in checkpoint_values.items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        if not p.is_file() or sha256(p)!=digest: raise ValueError('native checkpoint changed: '+name)
    device_path=_safe(device_report);device=strict_json(device_path.read_text()).get('device')
    if not isinstance(device,dict) or any(k not in device for k in DEVICE_KEYS): raise ValueError('missing frozen device identity')
    compiler=dict(version=REFINEMENT_VERSION,commit=REFINEMENT_COMMIT,package_sha256=REFINEMENT_PACKAGE,
        compiler_sha256=sha256(refinement_compiler),floating_point_mode='precise',spirv_sha256={e:actual_audit[f'daxa-refinement-{e}.spv']['sha256'] for e in ENTRIES})
    report_identity=dict(refinement_shader_compiler=compiler,coarse_shader_compiler=dict(version='2026.12',compiler_sha256=sha256(coarse_compiler)),
        shader_identity=sha256(ROOT/'python/stereoforge/shaders/daxa/refinement.slang')+':'+sha256(ROOT/'python/stereoforge/shaders/ops/refinement.slang'))
    paths=[native,coarse_compiler,refinement_compiler,audit_path,checkpoint_path,device_path]
    paths+=_source_paths()+_fixture_paths(context['coarse_path'],context['coarse'])+_fixture_paths(context['target_path'],context['target'])
    paths+=list(directory.glob('*.spv'))
    paths += [Path(name) if Path(name).is_absolute() else ROOT/name for name in checkpoint_values]
    from compare_daxa_view import dependencies
    paths += [Path(p) for p in dependencies(native,True)['sha256']]
    runtime={name:runtime_dependencies(path) for name,path in [('native',native),('coarse_compiler',coarse_compiler),('refinement_compiler',refinement_compiler)]}
    paths += [Path(p) for closure in runtime.values() for p in closure]
    # Bind actual recorded compiler headers plus build configuration; no build.
    deps,_=_cold(['ninja','-C',directory,'-t','deps'])
    for line in deps.splitlines():
        if line.startswith('    '):
            p=Path(line.strip());p=p if p.is_absolute() else directory/p
            if not p.is_file(): raise ValueError('missing recorded compiler input')
            paths.append(p)
    paths += [directory/'build.ninja',directory/'CMakeCache.txt',directory/'CMakeFiles/rules.ninja']
    import cv2
    paths.append(next(Path(cv2.__file__).parent.glob('*.so')))
    if truth is not None:
        load_truth(truth,context);paths+=_truth_paths(truth)+[ROOT/'benchmarks/refinement_quality_fixtures.py']
    files={};aliases={}
    for path in paths:
        logical=Path(os.path.abspath(path));actual=logical.resolve(strict=True)
        if not actual.is_file(): raise ValueError('frozen leaf is not a file')
        files[str(actual)]=sha256(actual)
        if logical!=actual: aliases[str(logical)]=str(actual)
    descriptor=lambda p:dict(file=str(p.resolve()),sha256=sha256(p))
    frozen=dict(schema='stereoforge.refinement_freeze.v1',native=descriptor(native),coarse_compiler=descriptor(coarse_compiler),refinement_compiler=descriptor(refinement_compiler),
        audit=descriptor(audit_path),checkpoint=descriptor(checkpoint_path),shader_directory=str(directory),file_sha256=dict(sorted(files.items())),runtime_alias_targets=dict(sorted(aliases.items())),
        coarse_manifest=descriptor(context['coarse_path']),target_manifest=descriptor(context['target_path']),truth_family=descriptor(Path(truth)) if truth else None,
        report_identity=report_identity,device_identity={k:device[k] for k in DEVICE_KEYS},runtime_dependency_sha256=runtime)
    output=_safe(output)
    if output.exists(): raise ValueError('freeze output must be fresh')
    with output.open('xb') as stream: stream.write(fixture.canonical(frozen))
    return frozen


def verify_freeze(path,expected_hash,native,context,truth=None):
    path=_safe(path)
    if type(expected_hash) is not str or len(expected_hash)!=64 or sha256(path)!=expected_hash: raise ValueError('explicit freeze SHA256 differs')
    frozen=strict_json(path.read_text())
    keys={'schema','native','coarse_compiler','refinement_compiler','audit','checkpoint','shader_directory','file_sha256','runtime_alias_targets','coarse_manifest','target_manifest','truth_family','report_identity','device_identity','runtime_dependency_sha256'}
    if set(frozen)!=keys or frozen['schema']!='stereoforge.refinement_freeze.v1': raise ValueError('unsupported freeze schema')
    inventory=frozen['file_sha256'];aliases=frozen['runtime_alias_targets']
    if not isinstance(inventory,dict) or not inventory or not isinstance(aliases,dict): raise ValueError('missing freeze closure')
    for name,digest in inventory.items():
        p=Path(name)
        if not p.is_absolute() or p.resolve()!=p or not p.is_file() or sha256(p)!=digest: raise ValueError('frozen file changed: '+name)
    for logical,target in aliases.items():
        if not Path(logical).is_absolute() or not Path(target).is_absolute() or str(Path(logical).resolve(strict=True))!=target or target not in inventory:
            raise ValueError('runtime alias retargeted')
    def bound(p):
        p=Path(os.path.abspath(p));actual=p.resolve(strict=True)
        if p!=actual and aliases.get(str(p))!=str(actual): raise ValueError('unbound alias')
        if inventory.get(str(actual))!=sha256(actual): raise ValueError('required file absent from frozen closure: '+str(p))
    for key,actual in [('native',native),('coarse_manifest',context['coarse_path']),('target_manifest',context['target_path'])]:
        descriptor=frozen[key]
        if set(descriptor)!={'file','sha256'} or str(Path(actual).resolve())!=descriptor['file'] or sha256(actual)!=descriptor['sha256']: raise ValueError('frozen '+key+' identity differs')
    for key in ('native','coarse_compiler','refinement_compiler','audit','checkpoint','coarse_manifest','target_manifest'):
        desc=frozen[key]
        if set(desc)!={'file','sha256'} or inventory.get(desc['file'])!=desc['sha256']: raise ValueError('descriptor absent from frozen inventory')
    for label in ('native','coarse_compiler','refinement_compiler'):
        current=runtime_dependencies(frozen[label]['file'])
        _same(frozen['runtime_dependency_sha256'].get(label),current,'runtime dependency resolution '+label)
        for leaf in current: bound(leaf)
    checkpoint_values=strict_json(Path(frozen['checkpoint']['file']).read_text())
    if not checkpoint_values: raise ValueError('empty native checkpoint')
    for name,digest in checkpoint_values.items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        bound(p)
        if sha256(p)!=digest: raise ValueError('checkpoint source identity differs')
    if set(frozen['device_identity'])!=set(DEVICE_KEYS): raise ValueError('incomplete device binding')
    identity=frozen['report_identity'];compiler=identity['refinement_shader_compiler']
    for key,value in dict(version=REFINEMENT_VERSION,commit=REFINEMENT_COMMIT,package_sha256=REFINEMENT_PACKAGE,compiler_sha256=frozen['refinement_compiler']['sha256'],floating_point_mode='precise').items(): _same(compiler.get(key),value,'refinement compiler.'+key)
    _same(identity['coarse_shader_compiler'],dict(version='2026.12',compiler_sha256=frozen['coarse_compiler']['sha256']),'coarse compiler')
    _same(identity['shader_identity'],sha256(ROOT/'python/stereoforge/shaders/daxa/refinement.slang')+':'+sha256(ROOT/'python/stereoforge/shaders/ops/refinement.slang'),'shader source identity')
    actual_audit={}
    for entry in ENTRIES:
        p=Path(frozen['shader_directory'])/f'daxa-refinement-{entry}.spv';bound(p);actual_audit[p.name]=audit_spirv(p)
        _same(compiler['spirv_sha256'].get(entry),sha256(p),'SPIR-V compiler binding')
    _same(strict_json(Path(frozen['audit']['file']).read_text()).get('modules'),actual_audit,'frozen actual opcode audit')
    for p in _source_paths()+_fixture_paths(context['coarse_path'],context['coarse'])+_fixture_paths(context['target_path'],context['target']): bound(p)
    if truth is None:
        if frozen['truth_family'] is not None: raise ValueError('truth freeze requires explicit truth selection')
    else:
        if frozen['truth_family']!={'file':str(Path(truth).resolve()),'sha256':sha256(truth)}: raise ValueError('truth selection differs from freeze')
        load_truth(truth,context)
        for p in _truth_paths(truth)+[ROOT/'benchmarks/refinement_quality_fixtures.py']: bound(p)
    return frozen


def native_command(native,context,output,settings,validation):
    # The fixture accepts no recovery/ROI/injection option. Coarse SGM is actual.
    if settings.recovery or settings.disparities!=128 or settings.prior_confidence_floor!=.05 or settings.prior_span!=2 or settings.min_margin!=1 or settings.lr_tolerance!=1:
        raise ValueError('unsupported fixed native guided profile')
    command=[str(Path(native).resolve()),'--coarse-manifest',str(context['coarse_path']),'--target-manifest',str(context['target_path']),
        '--output',str(Path(output)/'native.json'),'--dump-prefix',str(Path(output)/'planes'/'frame'),
        '--radius',str(settings.radius),'--subpixel',str(int(settings.subpixel)),'--slots','2','--wave-size','32']
    if validation: command.append('--validation')
    return command


def run_comparison(coarse_manifest,target_manifest,native,freeze,freeze_sha256,output,*,radius=4,subpixel=False,validation=False,truth=None):
    settings=oracle.Settings(radius=radius,subpixel=subpixel,recovery=False)
    context=load_context(coarse_manifest,target_manifest)
    frozen=verify_freeze(freeze,freeze_sha256,native,context,truth)
    output=_safe(output)
    if output.exists(): raise ValueError('comparison output must be fresh')
    output.mkdir(parents=True,exist_ok=False);(output/'planes').mkdir()
    command=native_command(native,context,output,settings,validation)
    child=subprocess.run(command,capture_output=True,text=True,errors='replace',check=False)
    (output/'native.stdout').write_text(child.stdout);(output/'native.stderr').write_text(child.stderr)
    from compare_daxa_view import DIAGNOSTIC_PATTERNS
    if child.returncode!=0 or any(p.search(child.stdout+'\n'+child.stderr) for p in DIAGNOSTIC_PATTERNS.values()):
        raise ValueError('native failed/diagnosed; inspect '+str(output))
    report_path=_safe(output/'native.json',output)
    report=strict_json(report_path.read_text())
    result=compare_report(report,context,frozen,output,settings,validation,load_truth(truth,context) if truth else None)
    if verify_freeze(freeze,freeze_sha256,native,context,truth)!=frozen: raise ValueError('frozen inputs changed during dispatch')
    result.update(schema='stereoforge.refinement_comparison.v1',status='conditional_implementation_agreement',freeze_sha256=freeze_sha256,
        command=command,native_report_sha256=sha256(report_path),scope='static synthetic diagnostic; not physical/live/default promotion')
    with (output/'comparison.json').open('xb') as stream: stream.write(fixture.canonical(result))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='action',required=True)
    prepare=commands.add_parser('prepare');compare=commands.add_parser('compare')
    for p in (prepare,compare):
        for option in ('native','coarse-manifest','target-manifest','output'): p.add_argument('--'+option,type=Path,required=True)
        p.add_argument('--truth',type=Path)
    for option in ('coarse-compiler','refinement-compiler','audit','checkpoint','device-report'): prepare.add_argument('--'+option,type=Path,required=True)
    compare.add_argument('--freeze',type=Path,required=True);compare.add_argument('--freeze-sha256',required=True)
    compare.add_argument('--radius',type=int,choices=(2,4,8),default=4);compare.add_argument('--subpixel',type=int,choices=(0,1),default=0);compare.add_argument('--validation',action='store_true')
    args=parser.parse_args(argv)
    if args.action=='prepare':
        frozen=prepare_freeze(args.native,args.coarse_manifest,args.target_manifest,args.coarse_compiler,args.refinement_compiler,args.audit,args.checkpoint,args.device_report,args.output,args.truth)
        print(json.dumps(dict(freeze=str(args.output),sha256=sha256(args.output),files=len(frozen['file_sha256']))))
    else:
        result=run_comparison(args.coarse_manifest,args.target_manifest,args.native,args.freeze,args.freeze_sha256,args.output,radius=args.radius,subpixel=bool(args.subpixel),validation=args.validation,truth=args.truth)
        print(json.dumps(result,allow_nan=False))
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError,OverflowError) as error:
        import sys
        print('compare_daxa_refinement: '+str(error),file=sys.stderr);raise SystemExit(1)
