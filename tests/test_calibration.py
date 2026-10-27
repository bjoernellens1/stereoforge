import json
import cv2
import numpy as np
import pytest
from stereoforge.types import CameraModel, Calibration, StereoFrame, DepthFrame
from stereoforge.calibration import load_calibration, parse_zed_conf, download_calibration, Rectifier


def cal(rot=0.):
    c=CameraModel(16,12,20.,20.,7.5,5.5)
    t=np.eye(4); t[:3,:3]=cv2.Rodrigues(np.array([0.,rot,0.]))[0]; t[0,3]=.12
    return Calibration(c,c,t)


def test_load(tmp_path):
    p=tmp_path/'cal.json'; p.write_text(json.dumps(cal().to_dict()))
    assert load_calibration(p).identity==cal().identity
    with pytest.raises(ValueError): load_calibration(tmp_path/'missing')


def test_zed_units_rotation(tmp_path):
    p=tmp_path/'SN.conf'
    camera='fx=700\nfy=701\ncx=640\ncy=360\nk1=.1\nk2=.2\np1=.01\np2=.02\nk3=.3\n'
    p.write_text('[left_cam_HD]\n'+camera+'[RIGHT_CAM_hd]\n'+camera+
                 '[stereo]\nBaseline=120\nTY_HD=2\nTZ_HD=3\nRX_HD=.01\nCV_HD=.02\nRZ_HD=.03\n')
    c=parse_zed_conf(p,1280,720,'42')
    np.testing.assert_allclose(c.right_to_left[:3,3],[.12,.002,.003])
    np.testing.assert_allclose(c.right_to_left[:3,:3],cv2.Rodrigues(np.array([.01,.02,.03]))[0].T)
    assert c.left.distortion==(.1,.2,.01,.02,.3)
    with pytest.raises(ValueError): parse_zed_conf(p,640,480,'42')
    p.write_text('[STEREO]\nBaseline=120')
    with pytest.raises(ValueError): parse_zed_conf(p,1280,720,'42')


def test_cache_no_network(tmp_path):
    p=tmp_path/'SN42.conf'; p.write_text('cached')
    assert download_calibration('42',tmp_path)==p
    with pytest.raises(ValueError): download_calibration('../42',tmp_path)


def test_identity_rectification():
    c=cal(); rect=Rectifier(c)
    rgb=np.arange(12*16*3,dtype=np.uint8).reshape(12,16,3)
    original=rgb.copy(); f=StereoFrame(rgb,rgb.copy(),c,100,2,sensor_samples=[{'imu':1}])
    out=rect.rectify(f)
    np.testing.assert_array_equal(rgb,original)
    np.testing.assert_array_equal(out.left,rgb)
    assert out.frame_id==2 and out.sensor_samples==f.sensor_samples
    assert out.calibration.identity==rect.rectified_calibration.identity
    with pytest.raises(ValueError): Rectifier(cal(.01)).rectify(f)


@pytest.mark.parametrize('rotation',[0.,.05])
def test_depth_to_raw(rotation):
    c=cal(rotation); rect=Rectifier(c); shape=(12,16)
    z=np.full(shape,2.,np.float32); valid=np.ones(shape,bool)
    d=DepthFrame(np.zeros(shape),z,np.ones(shape),valid,np.ones(shape,np.uint8),
                 np.full(shape,7),np.full(shape,9),100,rect.rectified_calibration.identity)
    out=rect.depth_to_raw(d)
    assert out.calibration_id==c.identity and out.valid.any()
    assert np.isnan(out.disparity).all()
    assert (out.source_ids[out.valid]==7).all() and (out.age_ns[out.valid]==9).all()
    # Raw points transformed back must have rectified axial depth 2 m.
    yy,xx=np.where(out.valid)
    rays=cv2.undistortPoints(np.stack([xx,yy],axis=1).astype(float)[:,None],c.left.K,np.zeros(5))[:,0]
    points=np.column_stack([rays,np.ones(len(rays))])*out.depth_m[out.valid,None]
    recovered=points@rect.R1.T
    np.testing.assert_allclose(recovered[:,2],2.,atol=.004)
    d.calibration_id='wrong'
    with pytest.raises(ValueError): rect.depth_to_raw(d)


def test_depth_to_raw_collision_and_invalid():
    c=cal(); rect=Rectifier(c); shape=(12,16)
    # Collapse raw projected x onto one column to force equal-depth collisions.
    raw=CameraModel(16,12,.001,20.,7.5,5.5)
    rect.calibration=Calibration(raw,c.right,c.right_to_left)
    valid=np.zeros(shape,bool); valid[5,3:6]=True
    z=np.full(shape,np.nan,np.float32); z[5,3:6]=[2.,1.,1.]
    ids=np.full(shape,-1,np.int64); ids[5,3:6]=[30,40,50]
    d=DepthFrame(np.zeros(shape),z,np.ones(shape),valid,np.ones(shape,np.uint8),ids,
                 np.full(shape,9),100,rect.rectified_calibration.identity)
    out=rect.depth_to_raw(d)
    assert out.valid.sum()==1
    assert out.source_ids[out.valid].item()==40
    assert out.depth_m[out.valid].item()==1.
    # Invalid source data must never project or win a target pixel.
    d.valid[:]=False
    out=rect.depth_to_raw(d)
    assert not out.valid.any() and np.isnan(out.depth_m).all()


def test_zed_rotation_rectifies_independent_optical_correspondences(tmp_path):
    p=tmp_path/'SN.conf'
    camera='fx=700\nfy=701\ncx=640\ncy=360\nk1=0\nk2=0\np1=0\np2=0\nk3=0\n'
    p.write_text('[LEFT_CAM_HD]\n'+camera+'[RIGHT_CAM_HD]\n'+camera+
                 '[STEREO]\nBaseline=120\nRX_HD=.01\nCV_HD=.02\nRZ_HD=.03\n')
    calibration=parse_zed_conf(p,1280,720,'42')
    rectifier=Rectifier(calibration)
    # Independently form left-to-right optical rotation using Rodrigues' formula.
    vector=np.array([.01,.02,.03]); theta=np.linalg.norm(vector); axis=vector/theta
    x,y,z=axis; cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    optical=np.eye(3)+np.sin(theta)*cross+(1-np.cos(theta))*(cross@cross)
    left=np.array([[-.4,-.2,1.], [.1,.3,2.], [.8,-.1,4.], [-1.,.5,6.]])
    right=(left-np.array([.12,0.,0.]))@optical.T
    def pixel(points):
        return points[:,:2]/points[:,2,None]*[700.,701.]+[640.,360.]
    l=cv2.undistortPoints(pixel(left)[:,None],calibration.left.K,np.zeros(5),
                          R=rectifier.R1,P=rectifier.P1)[:,0]
    r=cv2.undistortPoints(pixel(right)[:,None],calibration.right.K,np.zeros(5),
                          R=rectifier.R2,P=rectifier.P2)[:,0]
    np.testing.assert_allclose(l[:,1],r[:,1],atol=1e-8)
    assert np.all(l[:,0]-r[:,0]>0)
