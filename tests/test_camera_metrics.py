import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('qualify_camera',Path(__file__).parents[1]/'benchmarks'/'qualify_camera.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
epipolar_residuals=module.epipolar_residuals


def test_metrics_report_vertical_error_without_epipolar_filtering():
    left=np.array([[10.,10.],[20.,20.],[30.,30.],[40.,40.]])
    right=left-np.array([[5.,0.],[5.,1.],[5.,2.],[5.,20.]])
    result=epipolar_residuals(left,right)
    assert result['matches']==4
    assert result['vertical_abs_px_p50']==pytest.approx(1.5)
    assert result['vertical_abs_px_p95']==pytest.approx(17.3)
    assert result['horizontal_positive_fraction']==1
    assert result['horizontal_px_p50']==5


def test_metrics_expose_reversed_disparity_sign():
    left=np.array([[10.,2.],[20.,3.],[30.,4.]])
    right=left+np.array([3.,0.])
    result=epipolar_residuals(left,right)
    assert result['horizontal_positive_fraction']==0
    assert result['horizontal_px_p50']==-3
    assert result['vertical_abs_px_p95']==0


@pytest.mark.parametrize('left,right',[(np.empty((0,2)),np.empty((0,2))), (np.zeros((2,2)),np.zeros((3,2))), (np.array([[np.nan,0]]),np.zeros((1,2)))])
def test_bad_correspondences_never_report_passing_metrics(left,right):
    with pytest.raises(ValueError): epipolar_residuals(left,right)
