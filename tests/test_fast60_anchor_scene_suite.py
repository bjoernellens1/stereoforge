import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from fast60_anchor_scene_suite import make_scene, run


def test_scenes_and_visibility():
    for name in ('thin','occlusion','low_texture','repetitive','slanted'):
        s=make_scene(name)
        assert s['left'].shape==(180,320)
        assert s['right'].dtype==np.uint8
        assert s['risk_mask'].any()
        assert np.isfinite(s['truth_disparity']).any()
        assert np.isnan(s['truth_disparity']).any()
        np.testing.assert_array_equal(s['left'],make_scene(name)['left'])
        d=s['truth_disparity']
        yy,xx=np.where(np.isfinite(d))
        xr=xx-d[yy,xx].astype(int)
        assert (xr>=0).all()
        np.testing.assert_array_equal(s['right'][yy,xr],s['left'][yy,xx])


def test_output(tmp_path):
    results=run(tmp_path,width=128,height=72,disparities=32)
    assert len(results)==5
    assert all((tmp_path/f'{r["scene"]}.npz').exists() for r in results)
    assert (tmp_path/'summary.json').exists()
    assert all(r['incorrect']>=0 for r in results)
