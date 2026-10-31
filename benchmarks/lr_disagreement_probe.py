#!/usr/bin/env python3
"""Analyze StereoForge native Left/Right disparity dumps without modifying matching.

The physically correct convention is d_L(x,y) ~= d_R(x-d_L,y). Other
coordinate probes are *diagnostic controls*, never alternative geometry.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def read_dump(prefix: Path, width: int, height: int):
    n = width * height
    def load(suffix: str, dtype):
        path = Path(str(prefix) + suffix)
        arr = np.fromfile(path, dtype=dtype)
        if arr.size != n:
            raise ValueError(f"{path}: expected {n} elements, got {arr.size}")
        return arr.reshape(height, width)
    return (load('.left.disparity.f32', '<f4'),
            load('.right.disparity.f32', '<f4'),
            load('.left.validity.u32', '<u4') != 0,
            load('.right.validity.u32', '<u4') != 0)


def q(values, p: float):
    return float(np.quantile(values, p)) if values.size else None


def sampled(left, lvalid, stride):
    h, w = left.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    yy, xx = yy.ravel(), xx.ravel()
    m = lvalid[yy, xx] & np.isfinite(left[yy, xx]) & (left[yy, xx] > 0)
    return yy[m], xx[m], left[yy[m], xx[m]]


def compare(left, right, lvalid, rvalid, *, stride=2, offset_x=0, offset_y=0,
            direction='physical', tolerance=1.0, preselected=None):
    """No interpolation; exactly the native nearest-integer x matching rule."""
    if left.ndim != 2 or left.shape != right.shape or lvalid.shape != left.shape or rvalid.shape != left.shape:
        raise ValueError('Left, Right and masks must share the same 2D shape')
    if not (isinstance(stride, int) and stride >= 1 and np.isfinite(tolerance) and tolerance >= 0):
        raise ValueError('Invalid stride or tolerance')
    if direction not in ('physical', 'opposite', 'same-x'):
        raise ValueError('Unknown coordinate control')
    if preselected is None: preselected = sampled(left, lvalid, stride)
    yy, xx, dd = preselected
    h, w = left.shape
    if direction == 'same-x': target_x = xx + offset_x
    else:
        sign = -1 if direction == 'physical' else 1
        target_x = np.floor(xx.astype(np.float64) + sign * dd.astype(np.float64) + 0.5).astype(np.int64) + offset_x
    target_y = yy + offset_y
    inside = (target_x >= 0) & (target_x < w) & (target_y >= 0) & (target_y < h)
    indices = np.flatnonzero(inside)
    valid = np.zeros(dd.size, dtype=bool)
    residual = np.full(dd.size, np.nan, dtype=np.float32)
    if indices.size:
        tx, ty = target_x[indices], target_y[indices]
        rv = rvalid[ty, tx] & np.isfinite(right[ty, tx]) & (right[ty, tx] > 0)
        valid[indices] = rv
        selected = indices[rv]
        residual[selected] = dd[selected] - right[target_y[selected], target_x[selected]]
    absr = np.abs(residual[valid])
    match = valid & (np.abs(residual) <= tolerance)
    count = int(dd.size)
    result = {
        'samples_left_selected': count,
        'opposite_in_bounds': int(inside.sum()),
        'opposite_selected': int(valid.sum()),
        'accepted': int(match.sum()),
        'opposite_selected_fraction': float(valid.mean()) if count else None,
        'accepted_fraction_of_left_selected': float(match.mean()) if count else None,
        'accepted_fraction_of_both_selected': float(match.sum()/valid.sum()) if valid.any() else None,
        'signed_disparity_residual_median_px': q(residual[valid], .5),
        'signed_disparity_residual_p10_px': q(residual[valid], .1),
        'signed_disparity_residual_p90_px': q(residual[valid], .9),
        'absolute_residual_p50_px': q(absr, .5),
        'absolute_residual_p90_px': q(absr, .9),
        'acceptance_by_tolerance_px': {str(t): float((absr <= t).mean()) if absr.size else None
                                       for t in (.25, .5, 1., 2., 4., 8.)},
    }
    return result


def analyze(left, right, lvalid, rvalid, *, stride=2, radius=4, vertical_radius=2, tolerance=1.):
    preselected = sampled(left, lvalid, stride)
    call = lambda **kw: compare(left, right, lvalid, rvalid, stride=stride, tolerance=tolerance,
                                 preselected=preselected, **kw)
    baseline = call(direction='physical')
    controls = {'same_x': call(direction='same-x'), 'wrong_direction_x_plus_d': call(direction='opposite')}
    # Sweep one axis at a time: a uniform offset is not a replacement for remapping.
    horizontal = {str(dx): call(direction='physical', offset_x=dx) for dx in range(-radius, radius+1)}
    vertical = {str(dy): call(direction='physical', offset_y=dy) for dy in range(-vertical_radius, vertical_radius+1)}
    yy, xx, dd = preselected
    bins = [0, 4, 8, 16, 32, 64, 128, 256]
    disparity_bins = {}
    for low, high in zip(bins[:-1], bins[1:]):
        m = (dd >= low) & (dd < high)
        if m.sum() >= 16:
            disparity_bins[f'{low}:{high}'] = compare(left, right, lvalid, rvalid,
                stride=stride, tolerance=tolerance, direction='physical', preselected=(yy[m],xx[m],dd[m]))
    def rank(spectrum):
        ordered = sorted(spectrum, key=lambda k: (
            spectrum[k]['accepted_fraction_of_left_selected'] if spectrum[k]['accepted_fraction_of_left_selected'] is not None else -1,
            spectrum[k]['accepted']), reverse=True)
        return [{'offset_px':int(k), 'accepted_fraction_of_left_selected': spectrum[k]['accepted_fraction_of_left_selected'],
                 'accepted_fraction_of_both_selected': spectrum[k]['accepted_fraction_of_both_selected'],
                 'both_selected':spectrum[k]['opposite_selected']} for k in ordered[:3]]
    return {'schema':'stereoforge.lr_disagreement_probe.v1',
            'geometry':'dL(x,y) compared to dR(round(x-dL), y); no subpixel coordinate interpolation',
            'purpose':'diagnostic; nonphysical controls and shift sweeps do not establish calibration corrections',
            'width':int(left.shape[1]),'height':int(left.shape[0]), 'sample_stride':stride,
            'total_left_valid_fraction':float(np.mean(lvalid & np.isfinite(left) & (left>0))),
            'total_right_valid_fraction':float(np.mean(rvalid & np.isfinite(right) & (right>0))),
            'baseline':baseline,'controls':controls,'horizontal_offsets':horizontal,
            'vertical_offsets':vertical,'top_horizontal':rank(horizontal),'top_vertical':rank(vertical),
            'disparity_bins':disparity_bins}


def diagnostic_image(left,right,lvalid,rvalid,path,tolerance):
    import cv2
    h,w=left.shape
    sampled_full=sampled(left,lvalid,1)
    yy,xx,dd=sampled_full
    xx2=np.floor(xx.astype(np.float64)-dd.astype(np.float64)+.5).astype(np.int64)
    inside=(xx2>=0)&(xx2<w)
    possible=np.zeros(xx.size,bool)
    ids=np.flatnonzero(inside)
    possible[ids]=rvalid[yy[ids],xx2[ids]] & np.isfinite(right[yy[ids],xx2[ids]]) & (right[yy[ids],xx2[ids]]>0)
    match=np.zeros(xx.size,bool)
    ids=np.flatnonzero(possible)
    match[ids]=np.abs(dd[ids]-right[yy[ids],xx2[ids]])<=tolerance
    # Status BGR: black invalid, orange opposite invalid, red disagreed, green accepted.
    image=np.zeros((h,w,3),np.uint8)
    image[yy,xx]=(0,150,255)
    image[yy[possible],xx[possible]]=(0,0,255)
    image[yy[match],xx[match]]=(0,210,0)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image): raise RuntimeError(f'Cannot write {path}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True,help='native daxa_depth_benchmark JSON')
    p.add_argument('--prefix',type=Path,help='--dump-prefix used with benchmark; default derives report path stem')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--status-image',type=Path)
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--horizontal-radius',type=int,default=4)
    p.add_argument('--vertical-radius',type=int,default=2)
    p.add_argument('--tolerance',type=float,default=1.)
    a=p.parse_args()
    if not (1<=a.stride<=64 and 0<=a.horizontal_radius<=32 and 0<=a.vertical_radius<=16):
        p.error('invalid stride or search radius')
    if a.output.exists(): p.error('output already exists')
    report=json.loads(a.report.read_text())
    width,height=report['width'],report['height']
    prefix=a.prefix or a.report.with_suffix('')
    left,right,lv,rv=read_dump(prefix,width,height)
    result=analyze(left,right,lv,rv,stride=a.stride,radius=a.horizontal_radius,
                   vertical_radius=a.vertical_radius,tolerance=a.tolerance)
    result['input_report']=str(a.report)
    result['input_prefix']=str(prefix)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    if a.status_image is not None: diagnostic_image(left,right,lv,rv,a.status_image,a.tolerance)
    print('Left stereo selected: {:.1%} | Right stereo selected: {:.1%}'.format(result['total_left_valid_fraction'],result['total_right_valid_fraction']))
    b=result['baseline']
    print('Physically matched (among both selected): {:.1%} | abs residual p50/p90: {} / {} px'.format(b['accepted_fraction_of_both_selected'] or 0,b['absolute_residual_p50_px'],b['absolute_residual_p90_px']))
    print('Diagnostic controls: same-x={:.1%} | wrong x+d={:.1%}'.format(*(result['controls'][k]['accepted_fraction_of_both_selected'] or 0 for k in ('same_x','wrong_direction_x_plus_d'))))
    print('Best horizontal shifts:',result['top_horizontal'])
    print('Best vertical shifts:',result['top_vertical'])
    print('JSON:',a.output)

if __name__=='__main__':main()
