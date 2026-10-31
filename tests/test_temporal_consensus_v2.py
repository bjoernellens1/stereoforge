from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH))
from temporal_consensus_v2 import (
    ConsensusConfig, TemporalConsensus, corrected_disparity, raw_frame, valid_depth,
    RAW_CURRENT, CONFIRMED_CURRENT, HELD_HISTORY)
from replay_temporal_v2 import Metrics, sequence_info, summarize_existing, write_json

FXB = 60.0


def rgba(shape=(12,16)):
    rng = np.random.default_rng(80)
    x = rng.integers(20, 220, (*shape, 4), dtype=np.uint8)
    x[:,:,3] = 255
    return x


def model(shape=(12,16), **kw):
    return TemporalConsensus(shape, FXB, ConsensusConfig(global_guard=False, **kw))


def run_frame(m, i, q=20., rgb=None, dt=50_000_000):
    d = np.full(m.shape, FXB / q if np.isfinite(q) else np.nan, np.float32)
    return m.update(d, rgba(m.shape) if rgb is None else rgb,
                    receipt_ns=1_000_000_000 + i*dt,
                    source_stamp_ns=5_000_000_000 + i*dt)


def test_two_observations_do_not_confirm_three_vote_policy():
    m = model()
    for i in range(2):
        assert not valid_depth(run_frame(m,i).depth).any()
    out = run_frame(m,2)
    assert np.all(out.provenance == CONFIRMED_CURRENT)
    assert np.all(out.support_hits == 3)
    assert np.allclose(out.depth,3.)


def test_all_invalid_and_one_impulse_never_create_surface():
    for impulse in (False,True):
        m = model()
        for i in range(12):
            out = run_frame(m,i,20. if impulse and i == 2 else np.nan)
            assert not valid_depth(out.depth).any()
            assert not (out.provenance == HELD_HISTORY).any()


def test_hold_has_bounded_real_elapsed_age_and_no_recursive_votes():
    m = model(hold_ms=100.)
    for i in range(3):
        run_frame(m,i)
    a = run_frame(m,3,np.nan)
    b = run_frame(m,4,np.nan)
    c = run_frame(m,5,np.nan)
    assert np.all(a.provenance == HELD_HISTORY)
    assert np.all(a.age_ms == 50.)
    assert np.all(b.age_ms == 100.)
    assert np.all(a.support_hits == 3) and np.all(b.support_hits == 3)
    assert not valid_depth(c.depth).any()


def test_confirm_profile_cannot_fill_invalid_pixels():
    m = model(hold_ms=0.)
    for i in range(3): run_frame(m,i)
    assert not valid_depth(run_frame(m,3,np.nan).depth).any()


def test_valid_contradiction_never_emits_old_surface():
    m = model()
    for i in range(3): run_frame(m,i,20.)
    changed = run_frame(m,3,40.)
    assert not valid_depth(changed.depth).any()
    assert changed.diagnostics['contradiction_reset_pixels'] == 192
    assert not valid_depth(run_frame(m,4,np.nan).depth).any()
    run_frame(m,5,40.)
    out = run_frame(m,6,40.)
    assert np.allclose(out.depth,1.5)


def test_sparse_invalid_neighbors_do_not_erase_thin_structure():
    m = model()
    depth = np.full(m.shape,np.nan,np.float32);depth[:,7] = 1.
    for i in range(3):
        out = m.update(depth,rgba(),receipt_ns=1_000_000_000+i*50_000_000,source_stamp_ns=i+1)
    assert np.allclose(out.depth[:,7],1.)
    assert valid_depth(out.depth).sum() == m.shape[0]


def test_no_blending_between_disparate_modes():
    m = model(agreement_px=.5)
    for i,q in enumerate([20.,40.,20.,40.,20.,40.,20.,40.,20.]):
        out=run_frame(m,i,q)
        if valid_depth(out.depth).any():
            disp=FXB/out.depth[valid_depth(out.depth)]
            assert np.all((np.abs(disp-20)<.01)|(np.abs(disp-40)<.01))


def test_photometric_change_resets_history_and_uses_current_raw():
    m=model();image=rgba()
    for i in range(3): run_frame(m,i,rgb=image)
    changed=image.copy();changed[5:7,5:7,:3]=255
    out=run_frame(m,3,q=40.,rgb=changed)
    assert out.provenance[5,5] == RAW_CURRENT
    assert out.depth[5,5] == 1.5
    assert out.age_ms[5,5] == 0
    next_out=run_frame(m,4,q=np.nan,rgb=changed)
    assert not np.isfinite(next_out.depth[5,5])


def test_global_camera_motion_falls_back_to_fresh_data():
    shape=(80,120);image=rgba(shape)
    m=TemporalConsensus(shape,FXB,ConsensusConfig(global_shift_px=.5))
    for i in range(3): run_frame(m,i,rgb=image)
    moved=np.roll(image,5,axis=1)
    out=run_frame(m,3,rgb=moved)
    assert out.diagnostics['global_motion_guard'] == 'reset'
    assert np.all(out.provenance == RAW_CURRENT)


def test_global_unobservable_rgb_is_not_claimed_static():
    m=TemporalConsensus((12,16),FXB,ConsensusConfig())
    image=np.full((12,16,4),100,np.uint8)
    run_frame(m,0,rgb=image)
    out=run_frame(m,1,rgb=image)
    assert out.diagnostics['global_motion_guard'] == 'unobservable_texture'
    assert np.all(out.provenance == RAW_CURRENT)


def test_long_gap_resets_not_fills_stale_depth():
    m=model()
    for i in range(3): run_frame(m,i)
    out=m.update(np.full(m.shape,np.nan,np.float32),rgba(),
                 receipt_ns=3_000_000_000,source_stamp_ns=8_000_000_000)
    assert out.diagnostics['gap_reset']
    assert not valid_depth(out.depth).any()


def test_nonmonotonic_identity_rejected_before_mutation():
    m=model();run_frame(m,0)
    snapshot=m.history.copy()
    with pytest.raises(ValueError,match='strictly increase'):
        run_frame(m,0)
    np.testing.assert_equal(snapshot,m.history)


def test_window_expiry_is_time_based_not_only_frame_count():
    m=model(evidence_ms=100.,hold_ms=100.)
    for i in range(3):run_frame(m,i)
    out=run_frame(m,4,np.nan)
    assert not valid_depth(out.depth).any() # Only one of 3 supports still in window.


def test_no_future_observation_influence():
    a=model();b=model()
    for i in range(7):
        x=run_frame(a,i,20. if i%3 else np.nan)
        y=run_frame(b,i,20. if i%3 else np.nan)
        np.testing.assert_equal(x.depth,y.depth)
    # Later contradictory data in one model cannot revise the earlier saved output.
    before=x.depth.copy()
    run_frame(a,7,80.)
    np.testing.assert_equal(x.depth,before)


def test_invalid_payload_and_geometry():
    m=model()
    with pytest.raises(ValueError,match='geometry'):
        m.update(np.ones((1,1)),rgba(),receipt_ns=1,source_stamp_ns=1)
    with pytest.raises(ValueError): TemporalConsensus((0,16),FXB,ConsensusConfig())
    with pytest.raises(ValueError): TemporalConsensus((12,16),float('nan'),ConsensusConfig())


@pytest.mark.parametrize('kwargs',[
    {'window_frames':1},{'min_hits':1},{'min_hits':10},{'hold_ms':-1},
    {'hold_ms':600},{'agreement_px':0},{'dominance':.5},{'dominance':1.1},
    {'rgb_dilate':-1},{'evidence_ms':float('nan')},
])
def test_bad_config(kwargs):
    with pytest.raises(ValueError): ConsensusConfig(**kwargs).validate()


def test_raw_depth_and_corrected_disparity_encoding():
    z=np.array([[3.,0.,np.nan,np.inf,-1.]],np.float32)
    out=raw_frame(z)
    assert out.provenance.tolist() == [[RAW_CURRENT,0,0,0,0]]
    assert np.isfinite(out.depth).sum()==1
    q=corrected_disparity(z,60.)
    assert q[0,0]==20 and np.isnan(q[0,1:]).all()


def test_shared_fresh_support_metrics_do_not_reward_hold_as_fresh():
    m=Metrics((1,2),60.)
    a=raw_frame(np.array([[3.,np.nan]],np.float32))
    m.update(a,a.depth,1)
    b=raw_frame(np.array([[3.,3.]],np.float32));b.provenance[0,0]=HELD_HISTORY;b.age_ms[0,0]=50
    m.update(b,np.array([[np.nan,3.]],np.float32),50_000_001)
    r=m.finish()
    assert r['fresh_common_pixel_transitions']==0
    assert r['raw_same_fresh_support_p90_px_median'] is None
    assert r['held_fraction_mean']==.25


def make_sequence(root,n=12):
    root.mkdir();shape=(12,16);image=rgba(shape);rows=[]
    for i in range(n):
        d=f'frame-{i:05}.depth.f32';c=f'frame-{i:05}.rgb.rgba8'
        np.full(shape,3.,'<f4').tofile(root/d);image.tofile(root/c)
        rows.append({'index':i,'stamp_ns':1_000_000_000+i*50_000_000,
                     'depth_received_ns':1_000_000_000+i*50_000_000,
                     'depth_file':d,'rgba_file':c,'frame_id':'test'})
    info={'schema':'stereoforge.ros_depth_sequence.v1','width':16,'height':12,'paired_rgb':True,'frames':rows}
    write_json(root/'manifest.json',info)
    return info


def test_preflight_catches_missing_sequence_and_symlinks(tmp_path):
    root=tmp_path/'raw';info=make_sequence(root)
    (root/info['frames'][0]['depth_file']).unlink()
    with pytest.raises(ValueError,match='raw payloads missing'):sequence_info(root)
    assert sequence_info(root,complete=False)['width']==16
    (root/info['frames'][0]['depth_file']).symlink_to(root/info['frames'][1]['depth_file'])
    with pytest.raises(ValueError,match='symlinks'):sequence_info(root,complete=False)


def test_cli_replay_and_legacy_manifest_export(tmp_path):
    root=tmp_path/'raw';make_sequence(root)
    output=tmp_path/'out'
    p=subprocess.run([sys.executable,str(BENCH/'replay_temporal_v2.py'),'replay',str(root),
                      '--output-dir',str(output),'--fxb','60','--assume-static',
                      '--no-global-guard','--save-sequence','hold100','--preview'],
                     text=True,capture_output=True)
    assert p.returncode==0,p.stdout+p.stderr
    r=json.loads((output/'summary.json').read_text())
    assert r['comparison_warmup_frames_excluded']==8
    assert r['profiles']['hold100']['valid_fraction_mean']==1
    assert r['profiles']['hold100']['held_fraction_mean']==0
    assert (output/'comparison.mp4').stat().st_size>0
    assert sequence_info(output/'hold100')['frames'][0]['provenance_file'].endswith('.u8')
    p2=subprocess.run([sys.executable,str(BENCH/'replay_temporal_v2.py'),'replay',str(root),
                       '--output-dir',str(output),'--fxb','60','--assume-static'],text=True,capture_output=True)
    assert p2.returncode!=0


def test_diagnostic_conditioning_counts():
    f=[{'valid_pixels':20,'stamp_ns':i*50_000_000+1,'depth_received_ns':i*50_000_000+1,
        'at_high_disparity_boundary_fraction_of_valid':.01} for i in range(2)]
    p=[{'from_frame':0,'to_frame':1,'common_valid_pixels':12,'disparity_delta_p50_px':.2,'depth_delta_p50_m':.01}]
    r={'frame_metrics':f,'pair_metrics':p,'width':10,'height':10,'valid_fraction_mean':.2,
       'pixel_ever_valid_fraction':.28,'pixel_valid_80pct_frames_fraction':.12,'pair_disparity_delta_p90_px_median':2.}
    audit=summarize_existing(r)
    assert audit['conditional_dropout_probability']==.4
    assert audit['flip_fraction_among_pair_union']==pytest.approx(16/28)
    assert audit['receipt_fps_first_to_last']==20.

from stereo_stage_audit_v2 import classify_lr


def test_lr_stage_exact_mapping_and_exclusive_partition():
    left=np.full((3,8),2.,np.float32);right=left.copy();v=np.ones((3,8),bool)
    accepted,cats,_=classify_lr(left,v,right,v,1.)
    assert not accepted[:,:2].any() and accepted[:,2:].all()
    assert cats['correspondence_out_of_bounds'].sum()==6
    accepted_right,_,_=classify_lr(right,v,left,v,1.,right=True)
    assert accepted_right[:,:-2].all() and not accepted_right[:,-2:].any()


def test_lr_audit_does_not_confuse_invalid_right_with_disagreement():
    l=np.full((1,8),2.,np.float32);r=np.full_like(l,4.)
    lv=np.ones((1,8),bool);rv=lv.copy();rv[0,0]=False
    a,c,_=classify_lr(l,lv,r,rv,1.)
    assert c['correspondence_other_invalid'].sum()==1
    assert c['lr_disagreement'].sum()==5
    a2,c2,_=classify_lr(l,lv,r,rv,2.)
    assert a2.sum()==5
    assert c2['correspondence_other_invalid'].sum()==1


def test_lr_audit_handles_nonfinite_disparity_without_cast_corruption():
    d=np.array([[np.nan,np.inf,1e30,2,2,2]],np.float32);v=np.ones(d.shape,bool)
    a,c,_=classify_lr(d,v,np.full_like(d,2),v,1.)
    assert c['selection_valid_nonfinite_disparity'].sum()==2
    assert c['correspondence_out_of_bounds'].sum()==1
    assert a.sum()==3

from stereo_stage_audit_v2 import audit_one


def test_native_dump_reader_end_to_end(tmp_path):
    w,h=8,3;n=w*h
    p=tmp_path/'run-1'
    report={'schema':'stereoforge.daxa_depth.v1','width':w,'height':h,'layers':'left_then_right',
            'lr_tolerance':1.,'disparities':32,'p1':8,'p2':32,'paths':4,'source_id':99,
            'scope':'repeated_test','capture_sha256':'a'*64}
    write_json(p.with_suffix('.json'),report)
    lr=[]
    for right,eye in enumerate(('left','right')):
        d=np.full((h,w),2.,'<f4');valid=np.ones((h,w),'<u4')
        d.tofile(str(p)+f'.{eye}.disparity.f32')
        np.full((h,w),.5,'<f4').tofile(str(p)+f'.{eye}.confidence.f32')
        valid.tofile(str(p)+f'.{eye}.validity.u32')
        good,_,_=classify_lr(d,valid,d,valid,1.,right=bool(right));lr.append(good)
    lr=np.asarray(lr)
    lr.astype('<u4').tofile(str(p)+'.lr_validity.u32')
    lr.astype('<u4').tofile(str(p)+'.metric_validity.u32')
    np.where(lr,30.,np.nan).astype('<f4').tofile(str(p)+'.depth.f32')
    r,identity=audit_one(p.with_suffix('.json'))
    assert r['eyes']['left']['native_lr_valid_fraction']==.75
    assert r['eyes']['right']['cpu_lr_mask_disagrees_with_native_pixels']==0
    assert identity['capture_sha256']=='a'*64
    assert len(r['plane_sha256'])==7
