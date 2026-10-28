"""Offline profile geometry and immutable native fixture contracts; no GPU."""
import copy
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import compare_daxa_preprocess as frozen
import multiresolution_fixture as profile
import test_daxa_preprocess_comparison as donor_helper
from stereoforge.types import Calibration, CameraModel


def tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def donor(tmp_path, monkeypatch):
    # The donor deliberately has distorted unequal original intrinsics, unequal
    # rectified cx, fx!=fy, and physically associated nonidentity unequal R1/R2.
    angle = .02
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    pose[0, 3] = .2
    cal = Calibration(CameraModel(16, 12, 20., 18., 7.2, 5.1, (.01, -.002, .001, -.001, 0.)),
                      CameraModel(16, 12, 21., 19., 7.8, 5.4, (-.01, .001, -.001, .002, 0.)), pose)
    with monkeypatch.context() as patch:
        patch.setattr(donor_helper, 'calibration', lambda *_: cal)
        capture, _ = donor_helper.record_file(tmp_path, records=1, width=16, height=12)
    directory = tmp_path / 'donor'
    m = frozen.export_fixture(capture, 0, cal, directory)
    rect = frozen.strict_json((directory / 'rectified-calibration.json').read_text())
    rect['left']['fy'] *= .8
    rect['right']['fy'] = rect['left']['fy']
    rect['right']['cx'] += .75
    model = Calibration.from_dict(rect)

    def replace(entry, raw):
        p = directory / entry['file']
        p.write_bytes(raw)
        entry.update(sha256=frozen.sha256(p), bytes=len(raw))

    replace(m['calibration']['rectified_json'], frozen.canonical(model.to_dict()))
    m['calibration']['rectified_identity'] = model.identity
    for eye, name, camera, rect_camera in zip(('left', 'right'), ('P1', 'P2'), (cal.left, cal.right), (model.left, model.right)):
        p = np.asarray(m['calibration'][name], float)
        p[1, 1], p[0, 2] = rect_camera.fy, rect_camera.cx
        m['calibration'][name] = p.tolist()
        r = np.asarray(m['calibration']['R1' if eye == 'left' else 'R2'])
        mx, my = cv2.initUndistortRectifyMap(camera.K, np.asarray(camera.distortion), r, p, (16, 12), cv2.CV_32FC1)
        replace(m['maps'][eye + '_x'], mx.astype('<f4').tobytes())
        replace(m['maps'][eye + '_y'], my.astype('<f4').tobytes())
        rgb = np.frombuffer(frozen.fixture_file(directory, m['expected']['original_' + eye + '_rgb']).read_bytes(), np.uint8).reshape(12, 16, 3)
        image = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR)
        replace(m['expected']['rect_' + eye + '_rgb'], image.tobytes())
        replace(m['expected'][eye + '_luma'], cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype('<u4').tobytes())
        replace(m['expected'][eye + '_support'], frozen.support_mask(mx, my, 16, 12).tobytes())
    path = directory / 'manifest.json'
    path.write_bytes(frozen.canonical(m))
    frozen.validate_fixture(path)
    return path


@pytest.mark.parametrize('size', [(16, 12), (12, 9), (8, 6)])
def test_direct_profiles_preserve_geometry_original_bytes_and_world_points(donor, tmp_path, monkeypatch, size):
    before = tree_bytes(donor.parent)
    source, metadata = frozen.validate_fixture(donor)
    # Export must consume the donor's geometry, never estimate a replacement.
    monkeypatch.setattr(cv2, 'stereoRectify', lambda *a, **k: pytest.fail('donor rotations/projections must be preserved'))
    monkeypatch.setattr(cv2, 'resize', lambda *a, **k: pytest.fail('maps are made directly from original models'))
    result = profile.export_profile(donor, tmp_path / 'profile', *size)
    target, same_metadata = frozen.validate_fixture(result)
    assert same_metadata == metadata and target['source'] == source['source']
    assert before == tree_bytes(donor.parent)
    assert target['calibration']['R1'] == source['calibration']['R1']
    assert target['calibration']['R2'] == source['calibration']['R2']
    assert not np.array_equal(source['calibration']['R1'], source['calibration']['R2'])
    assert not np.array_equal(source['calibration']['R1'], np.eye(3))
    for section, key in [('capture', None), ('calibration', 'original_json'), ('expected', 'original_left_rgb'), ('expected', 'original_right_rgb')]:
        a, b = (source[section], target[section]) if key is None else (source[section][key], target[section][key])
        assert frozen.fixture_file(donor.parent, a).read_bytes() == frozen.fixture_file(result.parent, b).read_bytes()
    original = Calibration.from_dict(frozen.strict_json(frozen.fixture_file(donor.parent, source['calibration']['original_json']).read_text()))
    old_rect = Calibration.from_dict(frozen.strict_json(frozen.fixture_file(donor.parent, source['calibration']['rectified_json']).read_text()))
    new_rect = Calibration.from_dict(frozen.strict_json(frozen.fixture_file(result.parent, target['calibration']['rectified_json']).read_text()))
    assert np.array_equal(old_rect.right_to_left, new_rect.right_to_left)
    s = size[0] / 16
    a = np.array([[s, 0, (s - 1) / 2], [0, s, (s - 1) / 2], [0, 0, 1]])
    for key in ('P1', 'P2'):
        assert np.array_equal(target['calibration'][key], a @ np.asarray(source['calibration'][key]))
    for i, (old, new) in enumerate(zip((old_rect.left, old_rect.right), (new_rect.left, new_rect.right))):
        assert old.fx != old.fy and new.cx == s * (old.cx + .5) - .5
        point = np.array([.1, .05, 2., 1.])
        if i:
            point = np.linalg.solve(original.right_to_left, point)
        r = np.asarray(source['calibration']['R1' if i == 0 else 'R2'])
        q = r @ point[:3]
        old_uv = np.array([old.fx*q[0]/q[2]+old.cx, old.fy*q[1]/q[2]+old.cy])
        new_uv = np.array([new.fx*q[0]/q[2]+new.cx, new.fy*q[1]/q[2]+new.cy])
        np.testing.assert_allclose(new_uv + .5, s*(old_uv + .5), atol=2e-15, rtol=0)
        reconstructed = np.linalg.solve(r, [((new_uv[0]-new.cx)/new.fx)*q[2], ((new_uv[1]-new.cy)/new.fy)*q[2], q[2]])
        np.testing.assert_allclose(reconstructed, point[:3], atol=1e-15, rtol=0)
        # Independent ray/distortion arithmetic must select the exact same
        # nearest-even 1/32-pixel sampling phases as the newly exported maps.
        raw_camera = original.left if i == 0 else original.right
        yy, xx = np.indices((size[1],size[0]),dtype=float)
        rays = np.stack([(xx-new.cx)/new.fx,(yy-new.cy)/new.fy,np.ones_like(xx)],axis=-1)
        raw = rays @ np.linalg.inv(r).T
        x, y = raw[...,0]/raw[...,2], raw[...,1]/raw[...,2]
        k1,k2,p1,p2,k3 = raw_camera.distortion
        r2 = x*x+y*y
        radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        expected_x = (raw_camera.fx*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+raw_camera.cx).astype(np.float32)
        expected_y = (raw_camera.fy*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+raw_camera.cy).astype(np.float32)
        eye = 'left' if i == 0 else 'right'
        for axis, expected in [('x',expected_x),('y',expected_y)]:
            actual = np.fromfile(frozen.fixture_file(result.parent,target['maps'][eye+'_'+axis]),'<f4').reshape(size[1],size[0])
            assert np.array_equal(np.rint(actual.astype(float)*32),np.rint(expected.astype(float)*32))

    # Canvas boundaries retain the donor field of view under the declared convention.
    assert ((size[0]-.5)+.5)/s-.5 == 16-.5
    assert ((size[1]-.5)+.5)/s-.5 == 12-.5
    assert (-.5+.5)/s-.5 == -.5
    if s == 1:
        assert target['maps'] == source['maps'] and target['expected'] == source['expected']


def test_reproducible_direct_maps_support_and_provenance(donor, tmp_path):
    first = profile.export_profile(donor, tmp_path/'a', 8, 6)
    second = profile.export_profile(donor, tmp_path/'b', 8, 6)
    assert tree_bytes(first.parent) == tree_bytes(second.parent)
    m, _ = frozen.validate_fixture(first)
    p = m['provenance']['multiresolution_profile']
    assert p['scale'] == {'numerator': 1, 'denominator': 2}
    assert p['exporter_sha256'] == frozen.sha256(profile.__file__)
    assert p['donor_manifest_sha256_before'] == p['donor_manifest_sha256_after'] == frozen.sha256(donor)
    assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in p['donor_files_sha256'])
    assert len(p['donor_files_sha256']) == 16  # manifest+capture+2calibrations+4maps+8expected
    for eye in ('left', 'right'):
        mx = np.fromfile(frozen.fixture_file(first.parent, m['maps'][eye+'_x']), '<f4').reshape(6, 8)
        my = np.fromfile(frozen.fixture_file(first.parent, m['maps'][eye+'_y']), '<f4').reshape(6, 8)
        support = np.fromfile(frozen.fixture_file(first.parent, m['expected'][eye+'_support']), '<u4').reshape(6, 8)
        assert np.any((mx >= 8) & (support != 0)), 'support must use original16-wide bounds, not rectified8-wide bounds'
        independent = np.ones((6, 8), bool)
        for y in range(6):
            for x in range(8):
                ix, iy = round(float(mx[y,x])*32), round(float(my[y,x])*32)
                bx, by, fx, fy = ix//32, iy//32, ix%32, iy%32
                for dx,dy,weight in [(0,0,(32-fx)*(32-fy)),(1,0,fx*(32-fy)),(0,1,(32-fx)*fy),(1,1,fx*fy)]:
                    if weight and not (0 <= bx+dx < 16 and 0 <= by+dy < 12):
                        independent[y,x] = False
        assert np.array_equal(support, independent)
    assert frozen.support_mask(np.array([[-1/64,15+1/64,15.5]],np.float32), np.zeros((1,3),np.float32), 16,12).tolist() == [[1,1,0]]


@pytest.mark.parametrize('size', [(0,6),(8,0),(17,12),(12,12),(8,7),(True,1),(-8,6),(8.,6)])
def test_bad_target_shapes_fail_without_output(donor, tmp_path, size):
    with pytest.raises(ValueError):
        profile.export_profile(donor,tmp_path/'bad',*size)
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('mutation', ['missing_manifest','manifest_symlink','unsafe_path','missing_plane','tampered_plane','leaf_symlink','parent_symlink'])
def test_donor_rejects_unsafe_missing_or_changed_inputs(donor,tmp_path,mutation):
    m = frozen.strict_json(donor.read_text())
    source = donor
    if mutation == 'missing_manifest':
        source = tmp_path/'missing.json'
    elif mutation == 'manifest_symlink':
        source = tmp_path/'link.json';source.symlink_to(donor)
    elif mutation == 'unsafe_path':
        m['maps']['left_x']['file'] = '../outside.f32';donor.write_bytes(frozen.canonical(m))
    elif mutation in ('missing_plane','tampered_plane','leaf_symlink'):
        p = donor.parent/m['maps']['left_x']['file']
        raw = p.read_bytes();p.unlink()
        if mutation == 'tampered_plane': p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        if mutation == 'leaf_symlink':
            other = donor.parent/'other.f32';other.write_bytes(raw);p.symlink_to(other)
    else:
        (donor.parent/'maps').rename(donor.parent/'real-maps');(donor.parent/'maps').symlink_to(donor.parent/'real-maps',target_is_directory=True)
    with pytest.raises((ValueError,OSError)):
        profile.export_profile(source,tmp_path/'bad',8,6)
    assert not (tmp_path/'bad').exists()


def test_existing_output_rejected_and_source_mutation_detected(donor,tmp_path,monkeypatch):
    out = tmp_path/'out';out.mkdir();(out/'sentinel').write_bytes(b'preserve')
    before = tree_bytes(donor.parent)
    with pytest.raises(ValueError):profile.export_profile(donor,out,8,6)
    assert tree_bytes(out) == {'sentinel': b'preserve'} and before == tree_bytes(donor.parent)
    original = cv2.initUndistortRectifyMap
    changed = False
    def mutate(*args,**kwargs):
        nonlocal changed
        if not changed:
            changed = True;donor.write_bytes(donor.read_bytes()+b' ')
        return original(*args,**kwargs)
    monkeypatch.setattr(cv2,'initUndistortRectifyMap',mutate)
    with pytest.raises(ValueError,match='donor changed'):profile.export_profile(donor,tmp_path/'changed',8,6)
    assert not (tmp_path/'changed').exists()


def test_scaled_disparity_preserves_depth_but_integer_ranges_are_only_approximate():
    fx,b,offset,z = 800.,.12,1.75,2.5
    disparity = fx*b/z+offset
    for s,d in [(1.,128),(.75,96),(.5,64)]:
        assert (s*fx*b)/(s*disparity-s*offset) == pytest.approx(z,abs=1e-15)
        near = profile.nominal_near_depth(s*fx,b,s*offset,d)
        assert near == s*fx*b/((d-1)-s*offset)
    assert profile.nominal_near_depth(.5*fx,b,.5*offset,64) != profile.nominal_near_depth(fx,b,offset,128)
    assert profile.nominal_near_depth(fx,b,127,128) is None
    assert profile.nominal_near_depth(fx,b,128,128) is None


def test_export_cannot_create_outputs_inside_immutable_donor(donor):
    before = tree_bytes(donor.parent)
    with pytest.raises(ValueError,match='donor'):
        profile.export_profile(donor,donor.parent/'new-profile',8,6)
    assert tree_bytes(donor.parent) == before
    lexical_donor = donor.parent/'..'/donor.parent.name/donor.name
    with pytest.raises(ValueError,match='donor'):
        profile.export_profile(lexical_donor,donor.parent/'other-profile',8,6)
    assert tree_bytes(donor.parent) == before
