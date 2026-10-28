"""Independent metric truth admission and one-frame normalization."""
from copy import deepcopy
from hashlib import sha256
import json
import numpy as np
import pytest
from tests.test_zed_comparison_schema import record, available, unavailable
from benchmarks.zed_comparison import schema


def make_truth(record, root, *, depth=None, sample=None, depth_format='f64_le_m', stride=None):
    """Write a tiny independent truth bundle beside a portable fixture."""
    g = record['calibration']['geometry']
    shape = (g['height'], g['width'])
    def file(name, data):
        (root / name).write_bytes(data)
        return dict(path=name, sha256=sha256(data).hexdigest(), size_bytes=len(data))
    def plane(name, values, fmt, row_stride=None):
        dtype = {'f64_le_m':'<f8', 'f32_le_m':'<f4', 'u8_mask':'u1'}[fmt]
        a = np.asarray(values, dtype=dtype).reshape(shape)
        row_bytes = a.shape[1] * a.dtype.itemsize
        row_stride = row_bytes if row_stride is None else row_stride
        payload = b''.join(row.tobytes() + bytes(row_stride-row_bytes) for row in a)
        return dict(**file(name, payload), width=shape[1], height=shape[0], stride_bytes=row_stride, format=fmt)
    masks = {k:unavailable() for k in ('geometric_support','known_negative','visible','occluded','mixed','boundary_edges')}
    masks['sample'] = plane('truth-sample', np.ones(shape) if sample is None else sample, 'u8_mask')
    pixel = {k:schema._content(record['images'][k]) for k in ('rectified_left','rectified_right','support_left','support_right')}
    frame = dict(input_identity=record['input_identity'], source_id=record['source_id'], capture=deepcopy(record['timestamps']['capture']), partition=record['partition'], calibration=deepcopy(record['calibration']['rectified']), rectified_pixel_identity=sha256(schema._canonical(pixel).encode()).hexdigest(), geometry=deepcopy(g), depth_convention='axial_z', depth=plane('truth-depth', np.ones(shape) if depth is None else depth, depth_format, stride), masks=masks, regions={}, registration=available(dict(method='analytic_same_grid', transform_to_left=np.eye(4).tolist(), max_error_m=0, justification='same analytic grid')), timestamp_uncertainty=available(dict(max_error_ns=0, justification='same analytic instant')))
    return dict(schema='stereoforge.zed_comparison.truth.v1', independent=True, provenance=dict(kind='analytic', artifact=file('truth-artifact', b'independent analytic declaration'), generator=available(file('truth-generator', b'analytic generator'))), frames=[frame])


def test_missing_truth_api():
    from benchmarks.zed_comparison.truth import load_truth
    assert callable(load_truth)


def load(document, records, root):
    from benchmarks.zed_comparison.truth import load_truth
    return load_truth(document, records, root=root)


@pytest.mark.parametrize('fmt,stride', [('f64_le_m',19),('f32_le_m',11)])
def test_off_axis_range_to_axial_z(record,tmp_path,fmt,stride):
    truth=make_truth(record,tmp_path,depth=[[2,np.nan],[np.inf,4]],sample=[[1,0],[0,1]],depth_format=fmt,stride=stride)
    truth['frames'][0]['depth_convention']='ray_range'
    before=deepcopy(truth)
    bundle=load(truth,[record],tmp_path)
    with bundle.frame(record['input_identity']) as f:
        expected=np.array([2,4])/np.hypot(1,np.hypot(.5/5,.5/5))
        np.testing.assert_allclose(f['depth_m'][f['sample']],expected)
        assert f['sample'].dtype==bool
        assert f['known_negative'] is None
        assert not f['depth_m'].flags.writeable
        assert not f['sample'].flags.writeable
        with pytest.raises(ValueError): f['depth_m'][0,0]=9
    assert truth==before


@pytest.mark.parametrize('field', ['input_identity','source_id','capture','partition','calibration','geometry','rectified_pixel_identity'])
def test_binding_rejected(record,tmp_path,field):
    t=make_truth(record,tmp_path); f=t['frames'][0]
    if field in ('input_identity','rectified_pixel_identity'): f[field]='0'*64
    elif field=='source_id': f[field]=1
    elif field=='capture': f[field]['value_ns']+=1
    elif field=='partition': f[field]='held_out'
    elif field=='calibration': f[field]['sha256']='0'*64
    else: f[field]['fx']+=1
    with pytest.raises(ValueError): load(t,[record],tmp_path)


@pytest.mark.parametrize('case', ['duplicate','binary','negative','nonfinite','overlap','bound','transform','symlink','hash','unsafe','budget','independent','extra'])
def test_invalid_truth(record,tmp_path,case):
    t=make_truth(record,tmp_path); f=t['frames'][0]
    if case=='duplicate': t['frames'].append(deepcopy(f))
    elif case in ('binary','negative','nonfinite'):
        t=make_truth(record,tmp_path,depth=[[np.nan if case=='nonfinite' else -1,1],[1,1]]) if case!='binary' else make_truth(record,tmp_path,sample=[[2,0],[0,1]])
    elif case=='overlap': f['masks']['known_negative']=available(deepcopy(f['masks']['sample']))
    elif case=='bound': f['timestamp_uncertainty']['value']['max_error_ns']=-1
    elif case=='transform': f['registration']['value']['transform_to_left'][0][0]=2
    elif case=='symlink':
        (tmp_path/'alias').symlink_to(tmp_path/'truth-depth'); f['depth']['path']='alias'
    elif case=='hash': (tmp_path/'truth-depth').write_bytes(b'x'*32)
    elif case=='unsafe': f['depth']['path']='../truth-depth'
    elif case=='budget': f['depth'].update(stride_bytes=512*1024*1024,size_bytes=1024*1024*1024)
    elif case=='independent': t['independent']=False
    else: f['extra']=1
    with pytest.raises(ValueError): load(t,[record],tmp_path)


def test_empty_sparse_unknown_and_unavailable_bounds(record,tmp_path):
    t=make_truth(record,tmp_path,depth=np.full((2,2),np.nan),sample=np.zeros((2,2)))
    f=t['frames'][0]; f['registration']=unavailable('not registered'); f['timestamp_uncertainty']=unavailable('not bounded')
    f['masks']['known_negative']=available(deepcopy(f['masks']['sample']))
    f['regions']['thin']=unavailable('no definition')
    with load(t,[record,deepcopy(record)],tmp_path).frame(record['input_identity']) as out:
        assert not out['sample'].any()
        assert out['regions']['thin'] is None
        assert out['registration']==f['registration']


def test_mutation_after_admission(record,tmp_path):
    t=make_truth(record,tmp_path); b=load(t,[record],tmp_path)
    (tmp_path/'truth-depth').write_bytes(b'x'*32)
    with pytest.raises(ValueError):
        with b.frame(record['input_identity']): pass


def test_mutation_during_materialization(record,tmp_path):
    t=make_truth(record,tmp_path); b=load(t,[record],tmp_path)
    with pytest.raises(ValueError):
        with b.frame(record['input_identity']): (tmp_path/'truth-depth').write_bytes(b'x'*32)


def test_strict_json_and_missing_generator(record,tmp_path):
    t=make_truth(record,tmp_path); t['provenance']['generator']=unavailable()
    with pytest.raises(ValueError): load(t,[record],tmp_path)
    path=tmp_path/'truth.json'; path.write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError): load(path,[record],tmp_path)


def test_nonbinary_optional_mask_and_region(record,tmp_path):
    for target in ('visible','region'):
        t=make_truth(record,tmp_path,sample=[[1,0],[0,0]])
        d=deepcopy(t['frames'][0]['masks']['sample'])
        raw=bytes([0,2,0,0]); (tmp_path/'bad-mask').write_bytes(raw)
        d.update(path='bad-mask',sha256=sha256(raw).hexdigest())
        if target=='visible': t['frames'][0]['masks']['visible']=available(d)
        else: t['frames'][0]['regions']['thin']=available(d)
        with pytest.raises(ValueError): load(t,[record],tmp_path)


def test_measured_bounds_and_generator_availability(record,tmp_path):
    t=make_truth(record,tmp_path)
    t['provenance']['kind']='measured'; t['provenance']['generator']=unavailable('surveyed independently')
    t['frames'][0]['registration']['value']['max_error_m']=.001
    t['frames'][0]['timestamp_uncertainty']['value']['max_error_ns']=10
    with load(t,[record],tmp_path).frame(record['input_identity']) as f:
        assert f['registration']['value']['max_error_m']==.001


def test_conflicting_duplicate_record_partition(record,tmp_path):
    t=make_truth(record,tmp_path); other=deepcopy(record); other['partition']='held_out'
    with pytest.raises(ValueError): load(t,[record,other],tmp_path)


def test_snapshot_content_validation(record,tmp_path):
    t=make_truth(record,tmp_path)
    cal=json.loads((tmp_path/'rect.json').read_text()); cal['left']['cy']=1
    data=json.dumps(cal).encode(); (tmp_path/'rect.json').write_bytes(data)
    desc=dict(path='rect.json',sha256=sha256(data).hexdigest(),size_bytes=len(data))
    record['calibration']['rectified']=desc; record['input_identity']=schema.input_identity(record)
    f=t['frames'][0]; f['input_identity']=record['input_identity']; f['calibration']=desc
    with pytest.raises(ValueError): load(t,[record],tmp_path)


def test_json_size_bound(record,tmp_path,monkeypatch):
    t=make_truth(record,tmp_path)
    monkeypatch.setattr(schema,'MAX_JSON_BYTES',16)
    with pytest.raises(ValueError): load(t,[record],tmp_path)


def test_frame_rechecks_only_its_dependencies(record,tmp_path,monkeypatch):
    t=make_truth(record,tmp_path); other=deepcopy(record)
    other['ordinal']=1; other['input_identity']=schema.input_identity(other)
    f=deepcopy(t['frames'][0]); f['input_identity']=other['input_identity']
    data=(tmp_path/'truth-depth').read_bytes(); (tmp_path/'other-depth').write_bytes(data)
    f['depth']['path']='other-depth'; t['frames'].append(f)
    bundle=load(t,[record,other],tmp_path)
    checked=[]; original=schema._Files._unchanged
    def track(self,relative,fingerprint):
        checked.append(relative)
        return original(self,relative,fingerprint)
    monkeypatch.setattr(schema._Files,'_unchanged',track)
    with bundle.frame(record['input_identity']): pass
    assert 'truth-depth' in checked
    assert 'other-depth' not in checked


def test_bundle_final_recheck_catches_completed_frame_mutation(record,tmp_path):
    t=make_truth(record,tmp_path);b=load(t,[record],tmp_path)
    with b.frame(record['input_identity']):pass
    (tmp_path/'truth-depth').write_bytes((tmp_path/'truth-depth').read_bytes())
    with pytest.raises(ValueError):b.recheck()
