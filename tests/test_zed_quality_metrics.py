import copy
import numpy as np
import pytest
from benchmarks.zed_comparison.quality_metrics import evaluate_frame, paired_decision


def fixture(z=(1., 2., 0., 0.)):
    z = np.asarray([z], dtype=float)
    return dict(depth_m=z, sample=z > 0, geometry=dict(fx=100., baseline_m=.1, cx_left=2., cx_right=0.),
                geometric_support=np.ones_like(z, dtype=bool), known_negative=np.asarray([[False, False, True, False]]),
                regions={}, boundary_edges=None)


def policy():
    return dict(frozen=False, requested_range_m=[.3, 8.], required_regions=['all'],
                margins=dict(completeness_loss_pp=2., relative_mae_ratio=1.1, false_boundary_excess_pp=0.),
                freeze_parameters=dict(correctness=dict(mode='depth_tolerance', absolute_m=.1, relative=0., combine='max'),
                                       region_definitions=None, zero_reference_error=None, uncertainty=None, statistics=None, partition_digest=None))


def run(z, valid=None, conf=None, direction='higher_is_better', truth=None, pol=None, bounds=None):
    z = np.asarray([z], dtype=float)
    return evaluate_frame(z, np.ones(z.shape, bool) if valid is None else np.asarray([valid]),
                          None if conf is None else np.asarray([conf]), direction, truth or fixture(), pol or policy(), search_bounds=bounds)


def test_invalid_sparse_truth_denominators():
    out = run([1., np.nan, 7., 4.])
    p = out['lanes']['requested']
    assert (p['truth_count'], p['accepted_count'], p['correct_count']) == (2, 1, 1)
    assert p['completeness'] == .5
    assert out['unknown_accept_count'] == 1
    assert out['known_negative_accept_count'] == 1
    assert p['accepted_incorrect_count'] == 0


def test_confidence_direction_coverage_curve():
    out = run([1., 3., 7., 4.], conf=[.1,.9,.9,.9], direction='lower_is_better')
    assert out['confidence_curve'][0]['threshold'] == .1
    assert out['confidence_curve'][0]['accepted_count'] == 1
    assert out['confidence_curve'][-1]['unknown_accept_count'] == 1


def test_tied_confidence_does_not_split_group():
    out = run([1., 3., 7., 4.], conf=[1.,1.,1.,1.])
    assert len(out['confidence_curve']) == 1
    assert out['confidence_curve'][0]['accepted_count'] == 2


def test_d128_range_strata():
    t = fixture((.05,.3,8.,9.))
    t['known_negative'][:] = False
    out = run([.05,.3,8.,9.], truth=t, bounds=dict(min_disparity=0,max_disparity_exclusive=128))
    assert out['lanes']['full_sampled']['truth_count'] == 4
    assert out['lanes']['requested']['truth_count'] == 2
    assert out['lanes']['search_supported']['truth_count'] == 2
    assert out['distance_bins']['<0.3']['truth_count'] == 1
    assert out['distance_bins']['>8']['truth_count'] == 1


def test_missing_threshold_empty_regions_and_boundary_definitions():
    p = policy(); p['freeze_parameters']['correctness'] = None
    t = fixture(); t['regions']['thin'] = np.zeros((1,4),bool)
    out = run([1.,2.,3.,4.], truth=t, pol=p)
    assert out['lanes']['requested']['correct_count'] is None
    assert out['regions']['thin']['requested']['completeness'] is None
    assert out['boundary_f1']['value'] is None
    assert out['surface_geometry']['value'] is None


def test_nearest_rank_and_overflow_are_unavailable_not_nan():
    t = fixture((1e-300,1.,0.,0.))
    out = run([1e308,2.,0.,0.], truth=t)
    assert out['lanes']['full_sampled']['relative_error_sum'] is None
    assert out['lanes']['full_sampled']['squared_error_sum_m2'] is None
    assert out['lanes']['requested']['p95_abs_error_m'] == 1.


def test_disparity_tolerance_and_calibration_offset():
    p=policy(); p['freeze_parameters']['correctness']=dict(mode='disparity_pixels',max_error_pixels=.6)
    out=run([1.05,2.2,0.,0.],pol=p,bounds=dict(min_disparity=7,max_disparity_exclusive=13))
    assert out['lanes']['requested']['correct_count']==2
    assert out['lanes']['search_supported']['truth_count']==2


def frozen():
    p=policy(); p['frozen']=True
    p['freeze_parameters'].update(statistics=dict(method='hierarchical_scene_sequence_bootstrap_v1',unit='scene_sequence',min_scenes=2,min_sequences=2),
        uncertainty=dict(confidence_level=.95), zero_reference_error=dict(method='absolute_margin',absolute_mae_margin_m=.01),
        region_definitions={'all':{'description':'all'}},partition_digest='a'*64)
    return p


def units(c=.005,r=0.):
    return [dict(scene_id=str(i),sequence_id=str(i),region='all',candidate=dict(mae_m=c,completeness=1.,boundary_false_accept_rate=0.),reference=dict(mae_m=r,completeness=1.,boundary_false_accept_rate=0.)) for i in range(2)]


def test_zero_reference_error_policy():
    out=paired_decision(units(),frozen())
    assert out['statistics_pass'] is True
    assert out['gates']['all']['mae']['upper_bound'] == pytest.approx(-.005)
    assert out['parity_qualified'] is False
    p=frozen();p['freeze_parameters']['zero_reference_error']=None
    assert paired_decision(units(),p)['statistics_pass'] is False


def test_bootstrap_is_deterministic_and_unknown_method_unqualified():
    p=frozen(); assert paired_decision(units(),p)==paired_decision(units(),p)
    p['freeze_parameters']['statistics']['method']='unknown'
    assert not paired_decision(units(),p)['statistics_pass']
    assert not paired_decision(units()[:1],frozen())['statistics_pass']


def test_nonfinite_summary_rejected_and_duplicate_units_not_independent():
    u=units();u[0]['candidate']['mae_m']=float('nan')
    with pytest.raises(ValueError): paired_decision(u,frozen())
    u=units();u.append(copy.deepcopy(u[0]))
    with pytest.raises(ValueError): paired_decision(u,frozen())


def test_fixed_truth_false_accept_rate_and_canonical_confidence_direction():
    out=run([1.,3.,7.,4.],conf=[.8,.9,.1,.2],direction='high_is_good')
    p=out['lanes']['requested']
    assert p['false_accept_rate']==.5
    assert p['conditional_false_accept_rate']==.5
    assert out['all_prediction_accept_count']==4
    assert out['confidence_curve'][0]['threshold']==.9
    t=fixture();t['regions']['boundary']=np.asarray([[True,True,False,False]])
    out=run([1.,3.,7.,4.],truth=t)
    assert out['boundary_false_accept_rate']==.5
    assert out['region_acceptance']['boundary']['known_negative_false_accept_rate'] is None


def test_missing_geometric_and_negative_truth_are_unavailable():
    t=fixture();t['geometric_support']=None;t['known_negative']=None
    out=run([1.,2.,7.,4.],truth=t)
    assert out['lanes']['common_support'] is None
    assert out['lanes']['search_supported'] is None
    assert out['known_negative_count'] is None
    assert out['unknown_accept_count']==2


def test_boundary_f1_requires_independent_sampled_neighbors():
    t=fixture((1.,1.,2.,2.));t['known_negative'][:]=False
    t['regions']['boundary']=np.ones((1,4),bool)
    t['boundary_edges']=np.asarray([[False,True,True,False]])
    p=policy();p['freeze_parameters']['region_definitions']={'boundary':dict(description='jump endpoints',absolute_jump_m=.2,relative_jump=0.,tolerance_pixels=0)}
    out=run([1.,1.,2.,2.],truth=t,pol=p)
    assert out['boundary_f1']['value']==1.
    t['sample'][:]=False
    assert run([1.,1.,2.,2.],truth=t,pol=p)['boundary_f1']['value'] is None


def test_confidence_curve_caps_thresholds_and_preserves_final_population():
    z=np.ones((1,1000)); t=dict(depth_m=z,sample=np.ones_like(z,bool),geometry={},regions={})
    out=evaluate_frame(z,np.ones_like(z,bool),np.arange(1000).reshape(1,-1),'low_is_good',t,policy())
    assert len(out['confidence_curve'])==101
    assert out['confidence_curve'][-1]['accepted_count']==1000


def test_scene_equal_weight_not_sequence_count_weight():
    p=frozen();p['margins']['relative_mae_ratio']=1.
    u=units(.01,.01)
    u[0]['candidate']['mae_m']=.03
    # Scene 0 has ten correlated sequence units; its mean remains .02 loss.
    u.extend(dict(scene_id='0',sequence_id=str(i+10),region='all',candidate=copy.deepcopy(u[0]['candidate']),reference=copy.deepcopy(u[0]['reference'])) for i in range(9))
    out=paired_decision(u,p)
    assert out['gates']['all']['mae']['estimate']==pytest.approx(.01)
    assert not out['statistics_pass']


def test_thin_recall_requires_declared_recipe():
    t=fixture();t['regions']['thin']=np.asarray([[True,False,False,False]])
    out=run([1.,2.,0.,0.],truth=t)
    assert out['thin_pixel_recall'] is None
    p=policy();p['freeze_parameters']['region_definitions']={'thin':{'description':'independent thin structure pixels'}}
    assert run([1.,2.,0.,0.],truth=t,pol=p)['thin_pixel_recall']==1.


def test_nonsampled_truth_and_empty_predictions_do_not_fabricate_scores():
    t=fixture((0.,0.,0.,0.));t['known_negative'][:]=False
    out=run([float('nan'),float('inf'),-1.,0.],truth=t)
    p=out['lanes']['requested']
    assert p['truth_count']==p['accepted_count']==0
    assert p['mae_m'] is None and p['completeness'] is None
    assert out['all_prediction_accept_count']==0


def test_bootstrap_rejects_impossible_summary_rates():
    u=units();u[0]['candidate']['completeness']=2.
    with pytest.raises(ValueError):paired_decision(u,frozen())
    u=units();u[0]['reference']['mae_m']=-1.
    with pytest.raises(ValueError):paired_decision(u,frozen())


def test_finite_inputs_cannot_overflow_paired_comparison_into_passing_gate():
    p=frozen();p['margins']['relative_mae_ratio']=1e308
    u=units(c=1e308,r=1e308)
    with pytest.raises(ValueError):paired_decision(u,p)


def test_declared_all_region_cannot_shrink_implicit_all_support():
    t=fixture();t['regions']['all']=np.zeros((1,4),bool)
    with pytest.raises(ValueError):run([1.,2.,0.,0.],truth=t)


def test_disjoint_nonempty_boundaries_have_zero_f1():
    t = fixture((1.,1.,2.,2.,2.,2.,2.,2.))
    t['known_negative'] = np.zeros((1,8),bool)
    t['regions']['boundary'] = np.ones((1,8),bool)
    t['boundary_edges'] = np.array([[0,1,1,0,0,0,0,0]],bool)
    p = policy()
    p['freeze_parameters']['region_definitions'] = {'boundary': {
        'description':'independent exact edge pixels', 'absolute_jump_m':.2,
        'relative_jump':0., 'tolerance_pixels':0}}
    out = run([1.,1.,1.,1.,1.,2.,2.,2.], truth=t, pol=p)['boundary_f1']
    assert out['predicted_count'] == out['truth_count'] == 2
    assert out['precision'] == out['recall'] == out['value'] == 0.
    assert out['reason'] is None
