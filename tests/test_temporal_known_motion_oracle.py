"""Independent CPU Task7b contract; no GPU tolerance or timing qualification."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import temporal_known_motion_oracle as oracle


def geometry(width=3,height=1,support=None):
    return oracle.Geometry(np.array([[1.,1.,0.,0.]]*2),np.array([np.eye(4),np.eye(4)]),np.ones((2,height,width),bool) if support is None else support)


def sample(z=(1.,2.,3.),t=0,capture=1,source=None,confidence=.5,provenance=1,age=0,clock='host',calibration='cal',device='device'):
    depth=np.array([z,z],np.float64).reshape(2,1,-1)
    return oracle.fresh_frame(depth,confidence=confidence,provenance=provenance,source_ids=capture if source is None else source,age_ns=age,timestamp_ns=t,capture_id=capture,clock_domain=clock,calibration_id=calibration,device_id=device)


def pose(t=0,clock='host',x=0.,z=0.):
    world=np.eye(4);world[0,3]=x;world[2,3]=z
    return oracle.TimedPose(world,t,clock)


def test_initial_exact_six_planes_and_complete_ancestry():
    f=sample((1.,np.nan,3.),source=2**40+1,provenance=5)
    stage=oracle.TemporalOracle(geometry());out,work=stage.process(f,pose())
    assert out.depth[0,0,0]==1 and out.provenance[0,0,0]==5
    assert out.ancestry_ids[0,0,0].tolist()==[2**40+1,-1,-1,-1]
    assert out.origin_mask[0,0,0]==1<<5 and out.count[0,0,0]==1
    assert out.reuse.max()==0 and out.oldest_age_ns.max()==0
    assert out.depth.view(np.uint64)[0,0,1]==0x7ff8000000000000
    assert out.source_ids[0,0,1]==-1 and out.count[0,0,1]==0
    assert np.all(out.ancestry_ids[0,0,1]==-1)
    assert work['source_pixels_scanned']==0
    out.depth[:]=99
    assert stage.history.depth[0,0,0]==1 and f.depth[0,0,0]==1


def test_known_world_translation_nearest_visibility_tie_and_work():
    stage=oracle.TemporalOracle(geometry());stage.process(sample((1.,4.,np.nan)),pose())
    # Current rig moves left: old world point x=0 projects right to pixel1.
    out,work=stage.process(sample((np.nan,)*3,t=10,capture=2),pose(10,x=-1))
    assert out.valid.tolist()==[[[False,True,False]],[[False,True,False]]]
    assert out.depth[0,0,1]==1 and out.source_ids[0,0,1]==1
    assert out.age_ns[0,0,1]==10 and out.oldest_age_ns[0,0,1]==10
    assert out.provenance[0,0,1]==2 and out.reuse[0,0,1]==1
    assert work['source_pixels_scanned']==6 and work['projected_history_candidates']==4
    assert work['visible_history_pixels']==2 and work['history_fill_pixels']==2
    tie=sample((1.,1.,np.nan));tie.source_ids[:,0,1]=9;tie.ancestry_ids[:,0,1,0]=9
    motion=np.eye(4);motion[2,3]=9
    target=oracle.warp(tie,geometry(),np.array([motion]*2),10,oracle.Config())[0]
    # Proper +Z translation makes both source points land in pixel0 at identical Z.
    assert target.source_ids[0,0,0]==1


def test_per_eye_pose_includes_right_lever_arm_and_independent_rotation():
    g=geometry();g.eye_to_rig[1,0,3]=2
    r=np.array([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    g.eye_to_rig[1]=g.eye_to_rig[1]@r
    previous=np.eye(4);current=r
    motion=oracle.eye_motion(g,previous,current)
    point=np.array([1.,0.,2.,1.])
    expected=np.linalg.solve(g.eye_to_rig[1],np.linalg.solve(current,previous@g.eye_to_rig[1]@point))
    np.testing.assert_array_equal(motion[1]@point,expected)
    assert not np.array_equal(motion[0],motion[1])


def test_disjoint_fusion_selected_age_oldest_age_and_origin5():
    current=sample((1.02,)*3,t=10,capture=2,age=1,provenance=5)
    prior=sample((1.,)*3,t=0,age=3)
    prior,_=oracle.warp(prior,geometry(),np.array([np.eye(4)]*2),10,oracle.Config())
    out,work=oracle.fuse(current,prior,geometry().support,oracle.Config())
    assert out.depth[0,0,0]==1.01 and out.confidence[0,0,0]==.5
    assert out.source_ids[0,0,0]==2 and out.age_ns[0,0,0]==1
    assert out.oldest_age_ns[0,0,0]==13 and out.reuse[0,0,0]==1
    assert out.ancestry_ids[0,0,0].tolist()==[1,2,-1,-1]
    assert out.count[0,0,0]==2 and out.origin_mask[0,0,0]==(1<<1)|(1<<5)
    assert out.provenance[0,0,0]==3 and work['fused_pixels']==6


def test_shared_ancestor_never_averages_or_increases_confidence():
    a=sample((1.,)*3,t=1,capture=1);b=sample((1.02,)*3,t=1,capture=2)
    fused,_=oracle.fuse(a,b,geometry().support,oracle.Config())
    again,work=oracle.fuse(a,fused,geometry().support,oracle.Config())
    np.testing.assert_array_equal(again.depth,a.depth)
    assert again.count.max()==1 and again.confidence.max()==.5 and work['fused_pixels']==0


def test_count_cap_never_truncates_but_allows_history_fill():
    g=geometry();config=oracle.Config();f=sample((1.,)*3)
    for i in range(2,5):f,_=oracle.fuse(sample((1.,)*3,capture=i),f,g.support,config)
    assert f.count.max()==4 and f.ancestry_ids[0,0,0].tolist()==[1,2,3,4]
    fresh=sample((1.01,)*3,capture=5)
    out,_=oracle.fuse(fresh,f,g.support,config)
    np.testing.assert_array_equal(out.depth,fresh.depth);assert out.count.max()==1
    out,_=oracle.fuse(sample((np.nan,)*3),f,g.support,config)
    assert out.count.max()==4 and out.provenance[0,0,0]==3


@pytest.mark.parametrize('origin,expected',[(1,2),(2,2),(3,3),(5,5)])
def test_history_fill_keeps_ancestry_origin_tag(origin,expected):
    f=sample((1.,)*3,provenance=1 if origin==3 else origin)
    if origin==3:f,_=oracle.fuse(f,sample((1.,)*3,capture=2),geometry().support,oracle.Config())
    out,_=oracle.fuse(sample((np.nan,)*3),f,geometry().support,oracle.Config())
    assert out.provenance[0,0,0]==expected
    assert out.origin_mask[0,0,0]==(1<<1 if origin==3 else 1<<origin)


@pytest.mark.parametrize('confidence,combine',[(.049999,False),(.05,True),(.5,True)])
def test_confidence_gate_is_inclusive_without_rejecting_current(confidence,combine):
    current=sample((1.02,)*3,capture=2,confidence=confidence);previous=sample((1.,)*3)
    out,work=oracle.fuse(current,previous,geometry().support,oracle.Config())
    assert (out.count[0,0,0]==2)==combine and out.valid.all()
    assert out.confidence.max()==(.5 if combine else confidence)


def test_current_disagreement_wins_and_unsupported_hole_never_fills():
    g=geometry();g.support[:,0,1]=False
    out,_=oracle.fuse(sample((2.,np.nan,np.nan)),sample((1.,)*3),g.support,oracle.Config())
    assert out.depth[0,0,0]==2 and not out.valid[0,0,1] and out.valid[0,0,2]


def test_reuse_four_allowed_fifth_rejected_and_ages_do_not_refresh():
    stage=oracle.TemporalOracle(geometry());stage.process(sample((1.,)*3),pose())
    for i in range(1,6):
        out,_=stage.process(sample((np.nan,)*3,t=i,capture=i+1),pose(i))
        assert bool(out.valid.any())==(i<=4)
        if i<=4:assert out.reuse.max()==i and out.age_ns.max()==i


@pytest.mark.parametrize('elapsed,valid',[(200000000,True),(200000001,False)])
def test_oldest_age_limit_inclusive(elapsed,valid):
    f=sample((1.,)*3)
    out,_=oracle.warp(f,geometry(),np.array([np.eye(4)]*2),elapsed,oracle.Config())
    assert bool(out.valid.any())==valid


def test_history_age_overflow_invalidates_only_offending_pixel():
    f=sample((1.,)*3);f.age_ns[:,0,0]=np.iinfo(np.int64).max;f.oldest_age_ns[:,0,0]=np.iinfo(np.int64).max
    out,_=oracle.warp(f,geometry(),np.array([np.eye(4)]*2),1,oracle.Config())
    assert not out.valid[:,0,0].any() and out.valid[:,0,1:].all()


def test_missing_pose_recovery_requires_new_compatible_pair():
    stage=oracle.TemporalOracle(geometry());stage.process(sample((1.,)*3),pose())
    out,w=stage.process(sample((1.,)*3,t=1,capture=2),None);assert w['reset_reason']=='missing_motion'
    out,w=stage.process(sample((np.nan,)*3,t=2,capture=3),pose(2));assert not out.valid.any()
    stage.process(sample((1.,)*3,t=3,capture=4),pose(3))
    out,w=stage.process(sample((np.nan,)*3,t=4,capture=5),pose(4));assert out.valid.all()


@pytest.mark.parametrize('mutation',['late','duplicate_id','stale_pose','wrong_clock','unknown_clock','bad_pose','bad_metadata','wrong_calibration'])
def test_rejected_input_preserves_prior_transaction(mutation):
    stage=oracle.TemporalOracle(geometry());stage.process(sample(t=10,capture=10),pose(10));before=stage.history.to_json()
    f=sample(t=11,capture=11);p=pose(11)
    if mutation=='late':f.timestamp_ns=10;p=pose(10)
    if mutation=='duplicate_id':f.capture_id=10
    if mutation=='stale_pose':p=pose(10)
    if mutation=='wrong_clock':p=pose(11,clock='other')
    if mutation=='unknown_clock':f.clock_domain='unknown';p=pose(11,clock='unknown')
    if mutation=='bad_pose':p.world_from_rig[0,0]=2
    if mutation=='bad_metadata':f.source_ids[0,0,0]=-1
    if mutation=='wrong_calibration':f.calibration_id='other'
    with pytest.raises(ValueError):stage.process(f,p)
    assert stage.history.to_json()==before


def test_clock_device_epoch_restart_and_explicit_reconfigure():
    stage=oracle.TemporalOracle(geometry());stage.process(sample(t=10,capture=10),pose(10))
    out,w=stage.process(sample(t=0,capture=1,clock='new-clock',device='new-device'),pose(0,'new-clock'))
    assert w['reset_reason']=='epoch_change' and out.count.max()==1
    stage.reconfigure(geometry(),calibration_id='other')
    out,w=stage.process(sample(t=0,capture=1,calibration='other'),pose())
    assert out.count.max()==1 and w['source_pixels_scanned']==0


def test_gap_and_explicit_reset_do_not_bridge_prior():
    stage=oracle.TemporalOracle(geometry());stage.process(sample(),pose())
    out,w=stage.process(sample((np.nan,)*3,t=200000001,capture=2),pose(200000001))
    assert not out.valid.any() and w['reset_reason']=='gap'
    stage.process(sample(t=200000002,capture=3),pose(200000002))
    out,w=stage.process(sample((np.nan,)*3,t=200000003,capture=4),pose(200000003),reset=True)
    assert not out.valid.any() and w['reset_reason']=='explicit_reset'


@pytest.mark.parametrize('kwargs',[{'max_age_ns':float('nan')},{'max_reuse':True},{'max_contributors':5},{'min_confidence':float('inf')},{'relative_z_tolerance':-.1}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):oracle.Config(**kwargs)


@pytest.mark.parametrize('prior_z,combines',[(1.25,True),(float(np.nextafter(1.25,np.inf)),False)])
def test_exact_quarter_relative_depth_boundary_never_waives_mask(prior_z,combines):
    out,work=oracle.fuse(sample((1.,)*3,capture=2),sample((prior_z,)*3),geometry().support,oracle.Config(relative_z_tolerance=.25))
    assert bool(work['fused_pixels'])==combines


@pytest.mark.parametrize('u,inside',[(.5,True),(-.5,True),(2.5,False),(-.5000000000000001,False)])
def test_nearest_pixel_half_open_projection_domain(u,inside):
    f=sample((1.,np.nan,np.nan));m=np.eye(4);m[0,3]=u
    out,_=oracle.warp(f,geometry(),np.array([m]*2),1,oracle.Config())
    assert bool(out.valid.any())==inside
    if inside:assert out.valid[0,0,int(np.floor(u+.5))]


@pytest.mark.parametrize('mutation',['duplicate_ancestor','truncated_ancestor','selected_not_ancestor','bad_count','bad_oldest','unknown_origin_bit','fused_without_lineage','noncanonical_invalid','shape'])
def test_complete_metadata_rejection_is_transactional(mutation):
    stage=oracle.TemporalOracle(geometry());stage.process(sample(),pose());before=stage.history.to_json()
    f=sample(t=1,capture=2)
    if mutation=='duplicate_ancestor':f.count[0,0,0]=2;f.ancestry_ids[0,0,0,:2]=[2,2]
    if mutation=='truncated_ancestor':f.ancestry_ids[0,0,0,1]=3
    if mutation=='selected_not_ancestor':f.source_ids[0,0,0]=9
    if mutation=='bad_count':f.count[0,0,0]=5
    if mutation=='bad_oldest':f.age_ns[0,0,0]=1
    if mutation=='unknown_origin_bit':f.origin_mask[0,0,0]=1
    if mutation=='fused_without_lineage':f.provenance[0,0,0]=3
    if mutation=='noncanonical_invalid':f.valid[0,0,0]=False
    if mutation=='shape':f=sample((1.,)*4,t=1,capture=2)
    with pytest.raises(ValueError):stage.process(f,pose(1))
    assert stage.history.to_json()==before


def test_independent_world_sequence_known_collision_and_two_eye_rotation():
    g=geometry();g.eye_to_rig[1,0,3]=.25
    fixtures=oracle.world_point_sequence(g,[(101,[0.,0.,1.]),(202,[4.,0.,4.])],[pose(0),pose(10,x=-1)])
    assert fixtures[0]['world_point_ids'][0,0].tolist()==[101,202,-1]
    assert fixtures[1]['world_point_ids'][0,0].tolist()==[-1,101,-1]
    stage=oracle.TemporalOracle(g);stage.process(fixtures[0]['frame'],fixtures[0]['pose'])
    warped,_=stage.process(sample((np.nan,)*3,t=10,capture=2),fixtures[1]['pose'])
    np.testing.assert_array_equal(warped.valid,fixtures[1]['frame'].valid)
    np.testing.assert_array_equal(warped.depth[warped.valid],fixtures[1]['frame'].depth[warped.valid])
    support=np.ones((2,3,5),bool);A=np.array([np.eye(4),np.eye(4)]);A[1]=np.array([[0.,-1.,0.,1.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    g=oracle.Geometry(np.array([[2.,2.,2.,1.],[1.,2.,2.,1.]]),A,support)
    fixture=oracle.world_point_sequence(g,[(7,[0.,0.,2.])],[pose()])[0]
    assert fixture['world_point_ids'][0,1,2]==7 and fixture['world_point_ids'][1,2,2]==7
    assert fixture['frame'].depth[0,1,2]==2 and fixture['frame'].depth[1,2,2]==2


@pytest.mark.parametrize('mutation',['fused_current','reused_current','forged_origin','oldest_not_selected','above_configured_cap','reprojected_current','neural_current'])
def test_process_accepts_only_fresh_raw_measured_current_transactionally(mutation):
    config=oracle.Config(max_contributors=1 if mutation=='above_configured_cap' else 4)
    stage=oracle.TemporalOracle(geometry(),config);stage.process(sample(),pose())
    history=stage.history.to_json();old_pose=stage.pose.world_from_rig.copy()
    current=sample(t=1,capture=2)
    if mutation in ('fused_current','above_configured_cap'):
        current,_=oracle.fuse(current,sample(t=1,capture=3),geometry().support,oracle.Config())
    if mutation in ('reprojected_current','neural_current'):
        value=2 if mutation=='reprojected_current' else 4
        current.provenance[:]=value;current.origin_mask[:]=1<<value
    if mutation=='reused_current':current.reuse[:]=1
    if mutation=='forged_origin':current.origin_mask[:]=(1<<1)|(1<<5)
    if mutation=='oldest_not_selected':current.oldest_age_ns[:]=1
    # Every altered frame is valid accumulated-history data, but not fresh intake.
    oracle.validate_frame(current)
    with pytest.raises(ValueError,match='fresh measured'):
        stage.process(current,pose(1))
    assert stage.history.to_json()==history
    np.testing.assert_array_equal(stage.pose.world_from_rig,old_pose)
    assert stage.calibration_id=='cal'


@pytest.mark.parametrize('depth,confidence',[(1e308,1.),(float(np.nextafter(0.,1.)),.05),(float(np.finfo(np.float64).max),1.)])
def test_weighted_fusion_preserves_equal_finite_extreme_and_smallest_subnormal(depth,confidence):
    a=sample((depth,)*3,confidence=confidence,capture=1)
    b=sample((depth,)*3,confidence=confidence,capture=2)
    out,work=oracle.fuse(a,b,geometry().support,oracle.Config())
    assert np.isfinite(out.depth).all() and (out.depth>0).all()
    np.testing.assert_array_equal(out.depth,a.depth)
    assert out.count.max()==2 and work['fused_pixels']==6
    oracle.validate_frame(out)


def test_weighted_fusion_unequal_extremes_is_bounded_without_product_overflow():
    a=sample((1e308,)*3,confidence=1.,capture=1)
    b=sample((1.1e308,)*3,confidence=1.,capture=2)
    out,_=oracle.fuse(a,b,geometry().support,oracle.Config(relative_z_tolerance=.25))
    assert np.isfinite(out.depth).all() and np.all((out.depth>=a.depth)&(out.depth<=b.depth))
    assert out.depth[0,0,0]==1e308+(1.1e308-1e308)*.5


def test_moving_absolute_world_rotation_has_literal_pixels_depth_and_aged_ids():
    # A world witness (2,0,2) rotates from camera x=2 to y=-2 under a
    # +90degree world-from-rig turn. Neither expected pixel uses eye_motion.
    g=oracle.Geometry(np.array([[1.,1.,2.,2.]]*2),np.array([np.eye(4)]*2),np.ones((2,5,5),bool))
    rotated=np.array([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    sequence=oracle.world_point_sequence(g,[(71,[2.,0.,2.])],[pose(),oracle.TimedPose(rotated,10,'host')])
    for eye in (0,1):
        assert sequence[0]['world_point_ids'][eye,2,3]==71
        assert sequence[1]['world_point_ids'][eye,1,2]==71
        assert sequence[0]['frame'].depth[eye,2,3]==2
        assert sequence[1]['frame'].depth[eye,1,2]==2
    stage=oracle.TemporalOracle(g);stage.process(sequence[0]['frame'],sequence[0]['pose'])
    missing=oracle.fresh_frame(np.full((2,5,5),np.nan),timestamp_ns=10,capture_id=2)
    out,work=stage.process(missing,sequence[1]['pose'])
    expected=np.zeros((2,5,5),bool);expected[:,1,2]=True
    np.testing.assert_array_equal(out.valid,expected)
    assert out.depth[:,1,2].tolist()==[2.,2.]
    assert out.source_ids[:,1,2].tolist()==[1,1]
    assert out.age_ns[:,1,2].tolist()==[10,10] and out.reuse[:,1,2].tolist()==[1,1]
    assert work['history_fill_pixels']==2


def test_unrepresentable_fusion_rejects_transaction_without_partial_commit(monkeypatch):
    stage=oracle.TemporalOracle(geometry());stage.process(sample(),pose())
    before=stage.history.to_json();old_pose=stage.pose.world_from_rig.copy()
    original=oracle.bounded_weighted_depth
    def unrepresentable(*args):return original(float('inf'),1.,1.,1.)
    monkeypatch.setattr(oracle,'bounded_weighted_depth',unrepresentable)
    with pytest.raises(ValueError,match='unrepresentable fused'):
        stage.process(sample(t=1,capture=2),pose(1))
    assert stage.history.to_json()==before
    np.testing.assert_array_equal(stage.pose.world_from_rig,old_pose)
