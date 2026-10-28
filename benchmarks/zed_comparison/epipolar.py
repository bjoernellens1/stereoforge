"""Pure rectified correspondence diagnostics; never metric ground truth."""
import copy
import hashlib
import math

import numpy as np

from .schema import _availability, _hex, _integer, _keys, _number, _require, _text


def _envelope(value):
    _keys(value, {'schema', 'profile', 'space', 'geometry', 'calibration_sha256',
                  'original_pixel_identity', 'correspondence_provenance'}, 'epipolar')
    _require(value['schema'] == 'stereoforge.zed_comparison.epipolar.v1', 'unsupported epipolar schema')
    _require(value['profile'] in ('factory', 'effective', 'analytic'), 'unsupported profile')
    _require(value['space'] == 'rectified', 'vertical residual requires rectified space')
    for key in ('calibration_sha256', 'original_pixel_identity'):
        _hex(value[key], 64, key)
    g = value['geometry']
    _keys(g, {'width', 'height', 'fx', 'fy', 'baseline_m', 'cx_left', 'cx_right',
              'projection', 'depth_convention', 'units', 'coordinate_frame', 'pixel_centres',
              'reference_camera', 'coordinate_axes'}, 'geometry')
    for key in ('width', 'height'):
        _integer(g[key], key, 1)
    for key in ('fx', 'fy', 'baseline_m'):
        _number(g[key], key, positive=True)
    for key in ('cx_left', 'cx_right'):
        _number(g[key], key)
    _text(g['coordinate_frame'], 'coordinate_frame')
    for key, expected in dict(projection='perspective', depth_convention='axial_z', units='m',
                              pixel_centres='integer_centres', reference_camera='left',
                              coordinate_axes='x_right_y_down_z_forward').items():
        _require(g[key] == expected, f'unsupported geometry {key}')
    p = value['correspondence_provenance']
    _keys(p, {'method', 'selection', 'population_identity'}, 'correspondence_provenance')
    for key in ('method', 'selection'):
        _text(p[key], key)
    _availability(p['population_identity'], lambda x: _hex(x, 64, 'population_identity'), 'population_identity')


def _summary(values):
    absolute = np.abs(values)
    return dict(count=int(values.size), signed_median_px=float(np.median(values)),
                absolute_median_px=float(np.median(absolute)),
                absolute_p95_px=float(np.partition(absolute, math.ceil(.95 * len(values))-1)[math.ceil(.95 * len(values))-1]),
                absolute_max_px=float(np.max(absolute)),
                positive_count=int(np.count_nonzero(values > 0)),
                negative_count=int(np.count_nonzero(values < 0)),
                zero_count=int(np.count_nonzero(values == 0)))


def evaluate_epipolar(matches: np.ndarray, geometry: dict) -> dict:
    """Summarize left-minus-right vertical residual and horizontal disparity."""
    _envelope(geometry)
    _require(isinstance(matches, np.ndarray) and matches.ndim == 2 and matches.shape[1] == 4,
             'matches must be an Nx4 ndarray')
    _require(matches.dtype.kind in 'iuf', 'matches must be real nonbool numbers')
    with np.errstate(over='ignore', invalid='ignore'):
        points = np.array(matches, dtype='<f8', order='C', copy=True)
    _require(bool(np.all(np.isfinite(points))), 'matches must be finite float64 coordinates')
    g = geometry['geometry']
    for columns, extent in (((0, 2), g['width']), ((1, 3), g['height'])):
        _require(bool(np.all((points[:, columns] >= -.5) & (points[:, columns] < extent-.5))),
                 'matches outside image footprint')
    points[points == 0] = 0.0  # canonicalize signed zero along with dtype and byte order
    result = copy.deepcopy(geometry)
    result.update(status='available' if len(points) else 'unavailable', diagnostic=True,
                  metric_truth=False, quality_qualified=False, count=len(points),
                  matches_sha256=hashlib.sha256(points.tobytes(order='C')).hexdigest(),
                  residual_convention='left_minus_right', p95_policy='nearest_rank')
    if len(points):
        result.update(vertical=_summary(points[:, 1]-points[:, 3]),
                      horizontal=_summary(points[:, 0]-points[:, 2]))
    else:
        result.update(reason='empty correspondence population', vertical=None, horizontal=None)
    return result


def compare_epipolar_profiles(results: list[dict]) -> dict:
    """Retain diagnostics and compare only declared identical correspondence populations."""
    _require(type(results) is list, 'results must be a list')
    envelope_keys = {'schema', 'profile', 'space', 'geometry', 'calibration_sha256',
                     'original_pixel_identity', 'correspondence_provenance'}
    report_keys = envelope_keys | {'status', 'diagnostic', 'metric_truth', 'quality_qualified',
        'count', 'matches_sha256', 'residual_convention', 'p95_policy', 'vertical', 'horizontal'}
    metric_keys = {'signed_median_px', 'absolute_median_px', 'absolute_p95_px', 'absolute_max_px'}
    sign_keys = {'positive_count', 'negative_count', 'zero_count'}
    for result in results:
        _require(type(result) is dict, 'result must be an object')
        status = result.get('status')
        _require(status in ('available', 'unavailable'), 'unsupported result status')
        _keys(result, report_keys | ({'reason'} if status == 'unavailable' else set()), 'result')
        _envelope({k: result[k] for k in envelope_keys})
        _require(result['diagnostic'] is True and result['metric_truth'] is False
                 and result['quality_qualified'] is False, 'result must remain an unqualified diagnostic')
        _require(result['residual_convention'] == 'left_minus_right'
                 and result['p95_policy'] == 'nearest_rank', 'unsupported diagnostic conventions')
        _hex(result['matches_sha256'], 64, 'matches_sha256')
        _integer(result['count'], 'count', 1 if status == 'available' else 0)
        if status == 'unavailable':
            _require(result['count'] == 0 and result['vertical'] is None
                     and result['horizontal'] is None, 'unavailable result must have no samples or summaries')
            _text(result['reason'], 'reason')
            continue
        for axis in ('vertical', 'horizontal'):
            summary = result[axis]
            _keys(summary, metric_keys | sign_keys | {'count'}, axis)
            _integer(summary['count'], axis + '.count', 1)
            _require(summary['count'] == result['count'], 'summary count mismatch')
            for key in metric_keys:
                _number(summary[key], axis + '.' + key,
                        minimum=None if key == 'signed_median_px' else 0)
            for key in sign_keys:
                _integer(summary[key], axis + '.' + key)
            _require(sum(summary[key] for key in sign_keys) == result['count'], 'sign counts mismatch')
            _require(abs(summary['signed_median_px']) <= summary['absolute_max_px']
                     and summary['absolute_median_px'] <= summary['absolute_p95_px']
                     <= summary['absolute_max_px'], 'inconsistent summary bounds')
    report = dict(profiles=copy.deepcopy(results), diagnostic=True, metric_truth=False, quality_qualified=False,
                  deltas=dict(status='unavailable', reason='unmatched populations'))
    if len(results) != 2:
        return report
    a, b = results
    pa, pb = a['correspondence_provenance'], b['correspondence_provenance']
    matched = (a.get('status') == b.get('status') == 'available'
               and a['original_pixel_identity'] == b['original_pixel_identity']
               and pa == pb and pa['population_identity']['status'] == 'available'
               and a.get('count') == b.get('count'))
    if matched:
        report['deltas'] = dict(status='available', convention='second_minus_first',
            vertical={key: b['vertical'][key]-a['vertical'][key] for key in
                      ('signed_median_px', 'absolute_median_px', 'absolute_p95_px', 'absolute_max_px')},
            horizontal={key: b['horizontal'][key]-a['horizontal'][key] for key in
                        ('signed_median_px', 'absolute_median_px', 'absolute_p95_px', 'absolute_max_px')})
        for axis in ('vertical', 'horizontal'):
            for value in report['deltas'][axis].values():
                _number(value, 'profile delta')
    return report
