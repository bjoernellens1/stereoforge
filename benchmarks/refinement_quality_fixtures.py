#!/usr/bin/env python3
"""Offline analytic development truth for independent two-camera refinement.

Metres, +X right/+Y down/+Z forward. World is the original Left optical
frame unless world_from_left is supplied. Every image pixel samples its centre
ray once (no optical antialiasing or area integration). Surfaces are bounded
world-XY patches z=slope_x*x+slope_y*y+intercept_m. Original cameras are
independently ray traced; neither a forward warp nor a matcher produces truth.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import struct

import numpy as np

import compare_daxa_preprocess as native
import multiresolution_fixture as profiles
from stereoforge.types import Calibration, CameraModel, validate_pose

SCENE_SCHEMA = 'stereoforge.analytic_stereo_scene.v1'
SCHEMA = 'stereoforge.refinement_quality_family.v1'
REGIONS = ('near', 'far', 'thin', 'slant', 'repeated', 'lowtexture')
TEXTURES = ('textured', 'repeated', 'lowtexture')


def synthetic_calibration(width=1280, height=720):
    """Public illustrative calibration, never ZED factory/physical calibration."""
    if type(width) is not int or type(height) is not int or min(width, height) < 8 or width % 4 or height % 2:
        raise ValueError('even half-size grids and width divisible by four required')
    pose = np.eye(4); pose[0, 3] = .12
    model = CameraModel(width, height, 528. * width / 1280, 528. * height / 720,
                        (width - 1) / 2., (height - 1) / 2.)
    return Calibration(model, model, pose, 'synthetic-public-refinement-v1')


def default_scene(calibration, seed=42, *, world_from_left=None):
    """Development hard cases with controlled full-rectified-Left thin phases.

    Patches' world bounds and phase are frozen in scene.json, not redefined for
    each grid. Lower grids can therefore miss a thin patch entirely. Thin widths
    are projected extents on their world-Y=0 centreline; with rotated cameras a
    world-XY rectangle need not have the same image width at other rows.
    """
    model = rectified_models(calibration)
    world_pose = _world_pose(world_from_left)
    z = .7
    # Homography from the fixed world foreground plane (X,Y,z) to the actual
    # full rectified Left grid. Its inverse maps the snapped image centre and
    # endpoints back to the plane. This handles nonidentity rectification/pose;
    # an arbitrary world offset of phase*z/fx would not establish a pixel phase.
    plane_to_world = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., z], [0., 0., 1.]])
    k = np.asarray(model['P1'])[:, :3]
    h = k @ np.asarray(model['R1']) @ np.linalg.inv(world_pose)[:3, :] @ plane_to_world
    def project(point, eye):
        raw = np.linalg.solve(_eye_pose(calibration, eye, world_pose), np.r_[point, 1.])[:3]
        q = np.asarray(model['R1' if eye == 'left' else 'R2']) @ raw
        p = np.asarray(model['P1' if eye == 'left' else 'P2'])
        if q[2] <= 0 or not np.isfinite(q).all():
            raise ValueError('thin design point must be in front of both cameras')
        return np.array([p[0, 0]*q[0]/q[2]+p[0, 2], p[1, 1]*q[1]/q[2]+p[1, 2]])
    def plane_x(u):
        # Inverse homography restricted to the explicit world-Y=0 centreline.
        denominator = u*h[2, 0]-h[0, 0]
        if denominator == 0:
            raise ValueError('thin centreline is parallel to the requested ray')
        x = (h[0, 2]-u*h[2, 2])/denominator
        if not np.isfinite(x):
            raise ValueError('invalid thin centreline intersection')
        return float(x)
    def patch(i, z, bounds, region, texture='textured', slope=0.):
        return dict(id=i, slope_x=slope, slope_y=0., intercept_m=z,
                    bounds_xy_m=bounds, region=region, texture=texture)
    surfaces = [patch(1, 3., [-100., 100., -100., 100.], 'far'),
                patch(2, 1.4, [-.9, -.35, -.4, .4], 'slant', slope=.4),
                patch(3, .75, [-.12, .06, -.2, .18], 'near'),
                patch(4, 1.8, [.3, .75, -.5, -.05], 'repeated', 'repeated'),
                patch(5, 2., [.25, .8, .08, .5], 'lowtexture', 'lowtexture'),
                # Deliberate stress outside the 720p D128 near-depth domain.
                # Geometry remains truth; the separate out_of_range tag prevents
                # confusing domain exclusion with estimator incompleteness.
                patch(9, .35, [-.23, -.18, -.1, .1], 'near')]
    for i, (pixels, phase) in enumerate(((1, .0), (2, .25), (4, .5))):
        desired_u = project([.15+i*.07, 0., z], 'left')[0]
        snapped_u = float(np.floor(desired_u)+phase)
        centre = [plane_x(snapped_u), 0., z]
        lo, hi = plane_x(snapped_u-pixels/2), plane_x(snapped_u+pixels/2)
        if lo >= hi:
            raise ValueError('thin foreground centreline must increase with Left image x')
        actual = {}
        for eye in ('left', 'right'):
            u, v = project(centre, eye)
            width = project([hi, 0., z], eye)[0]-project([lo, 0., z], eye)[0]
            actual[eye] = dict(centre_u=float(u), centre_v=float(v), phase_pixels=float(u % 1.),
                               centreline_width_pixels=float(width))
        design = dict(requested_left_phase_pixels=phase, requested_left_centreline_width_pixels=pixels,
                      world_centre_m=centre, actual_projection=actual,
                      phase_convention='u-floor(u), independently projected per eye in float64',
                      width_scope='world-Y=0 centreline only; no claim of constant width at other rows')
        surfaces.append(dict(patch(6+i, z, [lo, hi, -.3, .3], 'thin'), full_grid_design=design))
    return dict(schema=SCENE_SCHEMA, seed=seed, surfaces=surfaces,
                world_units='metres; original Left optical frame',
                sample_footprint='one geometric centre ray per pixel; no optical antialiasing',
                texture_cell_m=.007, development_only=True,
                thin_design_projection=dict(model=model, world_from_left=world_pose.tolist()))


def _validate_scene(scene):
    if not isinstance(scene, dict) or scene.get('schema') != SCENE_SCHEMA:
        raise ValueError('unknown analytic scene schema')
    if type(scene.get('seed')) is not int or not 0 <= scene['seed'] < 2**32:
        raise ValueError('scene seed must be uint32')
    surfaces = scene.get('surfaces')
    if not isinstance(surfaces, list) or not 1 <= len(surfaces) <= 64:
        raise ValueError('one to 64 bounded analytic patches required')
    ids = set()
    for p in surfaces:
        values = [p.get(k) for k in ('slope_x', 'slope_y', 'intercept_m')]
        bounds = np.asarray(p.get('bounds_xy_m'), dtype=float)
        if (any(type(v) not in (int, float) for v in values) or not np.isfinite(values).all()
                or values[2] <= 0 or bounds.shape != (4,) or not np.isfinite(bounds).all()
                or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]
                or type(p.get('id')) is not int or not 0 < p['id'] < 2**31 or p['id'] in ids
                or p.get('texture') not in TEXTURES or p.get('region') not in REGIONS):
            raise ValueError('invalid analytic patch geometry/identity/texture/region')
        ids.add(p['id'])
    if not np.isfinite(scene.get('texture_cell_m', .007)) or scene.get('texture_cell_m', .007) <= 0:
        raise ValueError('positive finite world texture cell required')


def render_rays(scene, origins, directions):
    """Nearest positive analytic intersection, vectorized over arbitrary ray shape.

    Unnormalized directions are allowed. Equal-distance ties choose the smaller
    surface ID. Memory is O(number of rays), not O(rays*patches).
    """
    _validate_scene(scene)
    origin = np.asarray(origins, np.float64)
    direction = np.asarray(directions, np.float64)
    if (direction.ndim < 1 or direction.shape[-1] != 3 or not np.isfinite(direction).all()
            or not np.isfinite(origin).all() or np.any(np.linalg.norm(direction, axis=-1) == 0)):
        raise ValueError('finite nonzero 3D rays required')
    try:
        origin = np.broadcast_to(origin, direction.shape)
    except ValueError as e:
        raise ValueError('ray origins cannot broadcast to directions') from e
    distance = np.full(direction.shape[:-1], np.inf)
    surface = np.zeros(distance.shape, np.int32)
    for p in sorted(scene['surfaces'], key=lambda s: s['id']):
        a, b, c = (p[k] for k in ('slope_x', 'slope_y', 'intercept_m'))
        denominator = direction[..., 2] - a * direction[..., 0] - b * direction[..., 1]
        numerator = c + a * origin[..., 0] + b * origin[..., 1] - origin[..., 2]
        t = np.full_like(distance, np.inf)
        np.divide(numerator, denominator, out=t, where=denominator != 0)
        # Avoid infinity arithmetic in missed/parallel rays, including Werror.
        finite = np.isfinite(t) & (t > 0)
        safe_t = np.where(finite, t, 0.)
        x = origin[..., 0] + safe_t * direction[..., 0]
        y = origin[..., 1] + safe_t * direction[..., 1]
        lo_x, hi_x, lo_y, hi_y = p['bounds_xy_m']
        use = finite & (t < distance) & (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
        distance[use], surface[use] = t[use], p['id']
    valid = surface != 0
    points = origin + np.where(valid, distance, 0.)[..., None] * direction
    points[~valid] = np.nan
    return dict(points_world=points, ray_parameter=np.where(valid, distance, np.nan), surface_id=surface, valid=valid)


def _world_pose(world_from_left):
    pose = np.eye(4) if world_from_left is None else np.asarray(world_from_left, np.float64)
    validate_pose(pose)
    return pose


def _eye_pose(calibration, eye, world_from_left):
    if eye not in ('left', 'right'):
        raise ValueError('eye must be left or right')
    return _world_pose(world_from_left) @ (calibration.right_to_left if eye == 'right' else np.eye(4))


def render_original(calibration, scene, eye, *, world_from_left=None):
    """Invert Brown/rational camera distortion with OpenCV, then ray trace world."""
    import cv2
    c = getattr(calibration, eye) if eye in ('left', 'right') else None
    if c is None or len(c.distortion) not in (0, 4, 5, 8):
        raise ValueError('eye and Brown/rational 0/4/5/8 distortion required')
    y, x = np.indices((c.height, c.width), dtype=np.float64)
    pixels = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
    # Iterated inverse, with a strict independent forward residual gate.
    distortion = np.asarray(c.distortion if c.distortion else (0.,)*5)
    normalized = cv2.undistortPointsIter(pixels, c.K, distortion, None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-13)).reshape(c.height, c.width, 2)
    ray = np.concatenate((normalized, np.ones((c.height, c.width, 1))), axis=-1)
    reprojection = cv2.projectPoints(ray.reshape(-1, 3), np.zeros(3), np.zeros(3), c.K, distortion)[0].reshape(c.height, c.width, 2)
    if not np.isfinite(ray).all() or np.max(np.abs(reprojection - np.stack((x, y), axis=-1))) > 1e-7:
        raise ValueError('distorted ray inverse did not converge')
    pose = _eye_pose(calibration, eye, world_from_left)
    hit = render_rays(scene, pose[:3, 3], ray @ pose[:3, :3].T)
    hit['axial_z_m'] = hit['ray_parameter'].copy()
    hit['luma'] = _texture(scene, hit)
    return hit


def _texture(scene, hit):
    points = np.where(hit['valid'][..., None], hit['points_world'], 0.)
    cell = float(scene.get('texture_cell_m', .007))
    x = np.floor(points[..., 0] / cell).astype(np.int64)
    y = np.floor(points[..., 1] / cell).astype(np.int64)
    # Fixed uint32 spatial hash, independent of image grid and eye.
    with np.errstate(over='ignore'):
        bits = x.astype(np.uint32) * np.uint32(73856093) ^ y.astype(np.uint32) * np.uint32(19349663)
        bits ^= hit['surface_id'].astype(np.uint32) * np.uint32(83492791) ^ np.uint32(scene['seed'])
        bits ^= bits >> np.uint32(16); bits *= np.uint32(0x7feb352d)
        bits ^= bits >> np.uint32(15); bits *= np.uint32(0x846ca68b); bits ^= bits >> np.uint32(16)
    luma = (32 + bits % 192).astype(np.uint8)
    for p in scene['surfaces']:
        mask = hit['surface_id'] == p['id']
        if p['texture'] == 'repeated':
            luma[mask] = np.where(((x[mask] // 3) % 2) != 0, 200, 55)
        elif p['texture'] == 'lowtexture':
            luma[mask] = 126
    luma[~hit['valid']] = 16
    return luma


def rectified_models(calibration):
    from stereoforge.calibration import Rectifier
    r = Rectifier(calibration)
    return dict(width=calibration.left.width, height=calibration.left.height,
                R1=r.R1.tolist(), R2=r.R2.tolist(), P1=r.P1.tolist(), P2=r.P2.tolist())


def scaled_models(model, width, height):
    if (type(width) is not int or type(height) is not int or min(width, height) <= 0
            or width > model['width'] or width * model['height'] != height * model['width']):
        raise ValueError('equal-axis downscale required')
    s = width / model['width']
    a = np.array([[s, 0., (s-1)/2], [0., s, (s-1)/2], [0., 0., 1.]])
    result = copy.deepcopy(model); result.update(width=width, height=height)
    for key in ('P1', 'P2'):
        result[key] = (a @ np.asarray(model[key])).tolist()
    return result


def _model_matrices(model, eye):
    w, h = model['width'], model['height']
    r = np.asarray(model['R1' if eye == 'left' else 'R2'], np.float64)
    p = np.asarray(model['P1' if eye == 'left' else 'P2'], np.float64)
    if (type(w) is not int or type(h) is not int or min(w, h) <= 0 or r.shape != (3, 3)
            or p.shape != (3, 4) or not np.isfinite(r).all() or not np.isfinite(p).all()
            or not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-10)
            or not np.isclose(np.linalg.det(r), 1., rtol=0, atol=1e-10)
            or p[0, 0] <= 0 or p[1, 1] <= 0 or not np.array_equal(p[2], [0., 0., 1., 0.])
            or p[0, 1] != 0 or p[1, 0] != 0):
        raise ValueError('invalid rectified model')
    return w, h, r, p[:, :3]


def render_grid_truth(calibration, model, scene, eye, *, disparities=128, world_from_left=None):
    """Fresh centre-ray truth in actual rectified R/P geometry, not resized depth.

    disparity is x_left-x_right in both eyes; cross_u/v is a continuous opposite
    rectified coordinate. Visibility is an independent nearest intersection along
    the exact opposite-camera ray to the same world point, never a rounded lookup.
    Off-image tests use pixel-area boundaries [-.5,width-.5), separately from
    native/Census/bilinear support. Invalid intersections use NaN/ID0/masks0.
    """
    if eye not in ('left', 'right') or type(disparities) is not int or not 1 <= disparities <= 256:
        raise ValueError('invalid eye/disparity domain')
    w, h, r, k = _model_matrices(model, eye)
    y, x = np.indices((h, w), dtype=float)
    q = np.stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1], np.ones((h, w))), axis=-1)
    pose = _eye_pose(calibration, eye, world_from_left)
    hit = render_rays(scene, pose[:3, 3], (q @ r) @ pose[:3, :3].T)
    other = 'right' if eye == 'left' else 'left'
    _, _, other_r, other_k = _model_matrices(model, other)
    other_pose = _eye_pose(calibration, other, world_from_left)
    safe_points = np.where(hit['valid'][..., None], hit['points_world'], pose[:3, 3])
    other_raw = (safe_points - other_pose[:3, 3]) @ other_pose[:3, :3]
    other_rect = other_raw @ other_r.T
    front = hit['valid'] & (other_rect[..., 2] > 0)
    u = np.full((h, w), np.nan); v = u.copy()
    np.divide(other_rect[..., 0] * other_k[0, 0], other_rect[..., 2], out=u, where=front)
    np.divide(other_rect[..., 1] * other_k[1, 1], other_rect[..., 2], out=v, where=front)
    u += other_k[0, 2]; v += other_k[1, 2]
    off = hit['valid'] & (~front | (u < -.5) | (u >= w-.5) | (v < -.5) | (v >= h-.5))
    # All valid surface points have a nonzero ray from the other camera; use a
    # harmless forward ray for missing points solely to avoid invalid arithmetic.
    direction = safe_points - other_pose[:3, 3]
    direction[~hit['valid']] = [0., 0., 1.]
    check = render_rays(scene, other_pose[:3, 3], direction)
    same = check['valid'] & (check['surface_id'] == hit['surface_id'])
    same &= np.abs(check['ray_parameter'] - 1.) <= 1e-10
    visible = hit['valid'] & ~off & same
    occluded = hit['valid'] & ~off & ~same
    disparity = (x-u) if eye == 'left' else (u-x)
    result = dict(axial_z_m=hit['ray_parameter'], disparity=disparity, cross_u=u, cross_v=v,
                  surface_id=hit['surface_id'], valid=hit['valid'], visible=visible,
                  occluded=occluded, off_image=off,
                  out_of_range=hit['valid'] & ((disparity < 0) | (disparity > disparities-1)))
    for region in REGIONS:
        ids = [p['id'] for p in scene['surfaces'] if p['region'] == region]
        result['region_'+region] = np.isin(hit['surface_id'], ids) & hit['valid']
    edge = np.zeros((h, w), bool)
    horizontal = hit['surface_id'][:, 1:] != hit['surface_id'][:, :-1]
    vertical = hit['surface_id'][1:, :] != hit['surface_id'][:-1, :]
    edge[:, 1:] |= horizontal; edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical; edge[:-1, :] |= vertical
    result['region_edges'] = edge
    return result


def _mixed_footprints(mx, my, original_ids, centre_ids):
    """Exact frozen 1/32 bilinear tap convention; geometry truth stays separate."""
    h, w = original_ids.shape
    ix = np.rint(mx.astype(float)*32).astype(np.int64)
    iy = np.rint(my.astype(float)*32).astype(np.int64)
    x, y, fx, fy = ix >> 5, iy >> 5, ix & 31, iy & 31
    mixed = np.zeros(mx.shape, bool)
    for dx, dy, weight in ((0, 0, (32-fx)*(32-fy)), (1, 0, fx*(32-fy)),
                           (0, 1, (32-fx)*fy), (1, 1, fx*fy)):
        active = weight != 0
        inside = (x+dx >= 0) & (x+dx < w) & (y+dy >= 0) & (y+dy < h)
        sampled = original_ids[np.clip(y+dy, 0, h-1), np.clip(x+dx, 0, w-1)]
        mixed |= active & inside & (sampled != centre_ids)
    return mixed


def _capture_bytes(calibration, left, right):
    h, w = left.shape
    packed = np.empty((h, 2*w, 2), np.uint8)
    packed[..., 1] = 128
    packed[:, :w, 0], packed[:, w:, 0] = left, right
    def string(value):
        raw = value.encode(); return struct.pack('<I', len(raw)) + raw
    record = struct.pack('<IIIQQQ', 2*w, h, 4*w, 7, 1000000000, 1000000001)
    for value in ('synthetic_camera', 'synthetic_host', calibration.identity,
                  'synthetic-analytic-world-v1', '{"synthetic":true}', 'synthetic'):
        record += string(value)
    record += struct.pack('<Q', 0) + string('synthetic_controls')
    for value in ('synthetic_no_imu', 'synthetic_imu', 'synthetic_no_pose_sensor'):
        record += string(value)
    payload = packed.tobytes()
    record += struct.pack('<QI', 0, 0) + struct.pack('<Q', len(payload)) + payload
    checksum = 14695981039346656037
    for byte in record:
        checksum = ((checksum ^ byte) * 1099511628211) & 0xffffffffffffffff
    return b'SFYUYV1\0' + struct.pack('<IQ', 1, 1) + record + struct.pack('<Q', checksum)


def export_scene_family(output, *, width=1280, height=720, calibration=None, scene=None,
                        seed=42, photometric=None, world_from_left=None):
    """Export fresh full/half native fixtures and independent analytic truth.

    Returns family.json. No physical recordings or private calibration caches are
    consumed. photometric supplies per-eye gain/offset; clipping is documented.
    All saved references are relative and content hashed; paths don't affect bytes.
    """
    output = profiles._no_symlinks(output)
    if output.exists():
        raise ValueError('output must be a fresh nonexisting directory')
    default = synthetic_calibration(width, height)
    calibration = default if calibration is None else calibration
    if any((c.width, c.height) != (width, height) for c in (calibration.left, calibration.right)):
        raise ValueError('calibration dimensions differ from family')
    pose = _world_pose(world_from_left)
    scene = default_scene(calibration, seed, world_from_left=pose) if scene is None else copy.deepcopy(scene)
    _validate_scene(scene)
    settings = dict(left=dict(gain=1., offset=0.), right=dict(gain=1., offset=0.))
    if photometric is not None:
        if not isinstance(photometric, dict) or set(photometric) - set(settings):
            raise ValueError('unknown photometric eye')
        for eye, values in photometric.items():
            if not isinstance(values, dict) or set(values) != {'gain', 'offset'}:
                raise ValueError('photometric gain/offset required')
            if not np.isfinite(list(values.values())).all() or not 0 < values['gain'] <= 4 or abs(values['offset']) > 128:
                raise ValueError('photometric settings outside bounded domain')
            settings[eye] = copy.deepcopy(values)
    originals = {eye: render_original(calibration, scene, eye, world_from_left=pose) for eye in ('left', 'right')}
    lumas = {}
    clipped = {}
    for eye, rendered in originals.items():
        transformed = rendered['luma'].astype(float) * settings[eye]['gain'] + settings[eye]['offset']
        clipped[eye] = int(((transformed < 0) | (transformed > 255)).sum())
        lumas[eye] = np.floor(np.clip(transformed, 0, 255) + .5).astype(np.uint8)
    # Inputs and ray domains validate before the fresh output is created.
    output.mkdir(parents=True)
    def save(name, raw, *, dtype=None, shape=None):
        path = output / name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        entry = dict(file=name, bytes=len(raw), sha256=native.sha256(path))
        if dtype is not None:
            entry.update(dtype=dtype, shape=list(shape))
        return entry
    scene_entry = save('scene.json', native.canonical(scene))
    capture_entry = save('original/capture.sfp', _capture_bytes(calibration, lumas['left'], lumas['right']))
    calibration_entry = save('original/calibration.json', native.canonical(calibration.to_dict()))
    native.export_fixture(output / capture_entry['file'], 0, calibration, output / 'full')
    profiles.export_profile(output / 'full/manifest.json', output / 'half', width//2, height//2)
    family = dict(schema=SCHEMA, development_only=True, metric_truth_scope='analytic synthetic world only; no physical qualification',
                  generator_sha256=native.sha256(__file__), scene=scene_entry, original_capture=capture_entry,
                  original_calibration=calibration_entry, world_from_left=pose.tolist(),
                  photometric=dict(settings=settings, clipped_pixels=clipped, rounding='floor(clamp(Y*gain+offset,0,255)+.5); neutral U/V128'),
                  profiles={})
    for name in ('full', 'half'):
        path = output / name / 'manifest.json'
        manifest, _ = native.validate_fixture(path)
        model = dict(manifest['calibration'], **manifest['rectified'])
        gt = dict(schema='stereoforge.analytic_stereo_truth.v1', generator_sha256=family['generator_sha256'],
                  scene_sha256=scene_entry['sha256'], original_capture_sha256=capture_entry['sha256'],
                  original_calibration_sha256=calibration_entry['sha256'], native_manifest_sha256=native.sha256(path),
                  projection_sha256=hashlib.sha256(native.canonical(model)).hexdigest(),
                  rectified=manifest['rectified'], world_from_left=pose.tolist(), disparities=128 if name == 'full' else 64,
                  conventions=dict(depth='positive axial Z in own rectified camera, metres; invalid NaN',
                     disparity='continuous x_left-x_right, target pixels, including principal-point offset',
                     visibility='exact opposite ray nearest intersection; off-image separate; no rounded lookup',
                     footprint='centre-ray truth; mixed_bilinear marks differing nonzero inbounds raw surface taps; support separate',
                     missing_surface='surface_id0, valid0, floatNaN', sample_boundary='pixel area [-.5,width-.5) / [-.5,height-.5)'), eyes={})
        for eye in ('left', 'right'):
            planes = render_grid_truth(calibration, model, scene, eye, disparities=gt['disparities'], world_from_left=pose)
            shape = (manifest['rectified']['height'], manifest['rectified']['width'])
            mx = np.frombuffer(native.fixture_file(path.parent, manifest['maps'][eye+'_x']).read_bytes(), '<f4').reshape(shape)
            my = np.frombuffer(native.fixture_file(path.parent, manifest['maps'][eye+'_y']).read_bytes(), '<f4').reshape(shape)
            planes['mixed_bilinear'] = _mixed_footprints(mx, my, originals[eye]['surface_id'], planes['surface_id'])
            planes['remap_support'] = native.support_mask(mx, my, width, height).astype(bool)
            gt['eyes'][eye] = {}
            for key, plane in planes.items():
                dtype = '<f8' if plane.dtype.kind == 'f' else ('<i4' if key == 'surface_id' else '|u1')
                entry = save(f'{name}/truth/{eye}/{key}.bin', plane.astype(dtype).tobytes(), dtype=dtype, shape=plane.shape)
                # Truth-manifest references are relative to its own directory.
                entry['file'] = str(Path(entry['file']).relative_to(name))
                gt['eyes'][eye][key] = entry
        truth_entry = save(name+'/truth.json', native.canonical(gt))
        family['profiles'][name] = dict(native_manifest=dict(file=name+'/manifest.json', sha256=native.sha256(path)), truth_manifest=truth_entry)
    family_entry = output / 'family.json'
    family_entry.write_bytes(native.canonical(family))
    return family_entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        print(export_scene_family(args.output, width=args.width, height=args.height, seed=args.seed))
    except (ValueError, OSError) as e:
        parser.error(str(e))


if __name__ == '__main__':
    main()
