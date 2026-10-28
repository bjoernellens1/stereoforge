"""Offline three-row Census cost ablation for independent sparse stereo.

Each supported candidate uses three complete 5x5 Census footprints, one at
(y-1, x), (y, x), and (y+1, x). The selector remains identical to the single
Census reference. This allocating CPU oracle makes no depth or speed claim.
"""
from dataclasses import dataclass

import numpy as np

from stereoforge.refinement import census
from benchmarks.sparse_stereo_reference import Match, select_points


@dataclass(frozen=True)
class Result:
    source_id: str
    matches: tuple[Match, ...]
    forward_candidates: int
    reverse_candidates: int
    forward_hamming_ops: int
    reverse_hamming_ops: int


def _search(ref, other, other_support, y, x, sign, disparities, min_margin):
    d = np.arange(disparities, dtype=np.int32)
    ox = x + sign * d
    inside = (ox >= 0) & (ox < other.shape[1])
    d, ox = d[inside], ox[inside]
    valid = other_support[y-1, ox] & other_support[y, ox] & other_support[y+1, ox]
    d, ox = d[valid], ox[valid]
    count = int(d.size)
    costs = sum(np.bitwise_count(other[y+dy, ox] ^ ref[y+dy, x]).astype(np.uint16)
                for dy in (-1, 0, 1))
    if count < 2:
        return None, None, None, count, 'too_few_candidates'
    order = np.argsort(costs, kind='stable')
    best, second = int(costs[order[0]]), int(costs[order[1]])
    if second - best < min_margin:
        return None, best, second, count, 'ambiguous'
    return int(d[order[0]]), best, second, count, 'ok'


def match_descriptors(left, right, left_support, right_support, points, eye,
                      disparities=224, min_margin=1, source_id=''):
    left, right = np.asarray(left), np.asarray(right)
    left_support, right_support = np.asarray(left_support), np.asarray(right_support)
    if (left.ndim != 2 or left.shape != right.shape or left.dtype != np.uint32 or
        right.dtype != np.uint32 or left_support.shape != left.shape or
        right_support.shape != left.shape or left_support.dtype != np.bool_ or
        right_support.dtype != np.bool_):
        raise ValueError('expected matching uint32 descriptors and bool support')
    if eye not in ('left', 'right') or type(disparities) is not int or not 1 <= disparities <= 256:
        raise ValueError('invalid eye or disparity count')
    if type(min_margin) is not int or min_margin < 1:
        raise ValueError('min_margin must be a positive integer')
    ref, other = (left, right) if eye == 'left' else (right, left)
    ref_support, other_support = ((left_support, right_support) if eye == 'left'
                                  else (right_support, left_support))
    sign = -1 if eye == 'left' else 1
    matches = []
    forward = reverse = 0
    for y, x in points:
        if not (isinstance(y, (int, np.integer)) and isinstance(x, (int, np.integer)) and
                0 <= y < ref.shape[0] and 0 <= x < ref.shape[1]):
            raise ValueError('point outside reference image')
        y, x = int(y), int(x)
        if y == 0 or y + 1 >= ref.shape[0] or not ref_support[y-1:y+2, x].all():
            matches.append(Match(eye, y, x, None, None, None, None, False, 'unsupported'))
            continue
        d, best, second, count, reason = _search(ref, other, other_support, y, x, sign, disparities, min_margin)
        forward += count
        if d is None:
            matches.append(Match(eye, y, x, None, None, best, second, False, reason))
            continue
        other_x = x + sign * d
        reverse_d, _, _, count, reverse_reason = _search(other, ref, ref_support, y, other_x, -sign, disparities, min_margin)
        reverse += count
        if reverse_d is None:
            reason = 'reverse_' + reverse_reason
        elif reverse_d != d:
            reason = 'reverse_mismatch'
        else:
            reason = 'ok'
        matches.append(Match(eye, y, x, other_x, d, best, second, reason == 'ok', reason))
    return Result(source_id, tuple(matches), forward, reverse, 3 * forward, 3 * reverse)


def sparse_stereo(left, right, left_support, right_support, *, source_id,
                  cell_size=16, disparities=224, min_margin=1, selection='combined'):
    if not isinstance(source_id, str) or not source_id:
        raise ValueError('a non-empty frame source_id is required')
    if selection not in ('gradient', 'descriptor', 'combined', 'combined_edge'):
        raise ValueError('unknown sparse selection')
    left, right = np.asarray(left), np.asarray(right)
    left_support, right_support = np.asarray(left_support), np.asarray(right_support)
    if left.shape != right.shape:
        raise ValueError('stereo image dimensions must match')
    lc, rc = census(left, left_support), census(right, right_support)
    results = []
    for eye, image, features in (('left', left, lc), ('right', right, rc)):
        points = select_points(image, features.support, features.descriptor, cell_size, selection=selection)
        results.append(match_descriptors(lc.descriptor, rc.descriptor, lc.support, rc.support,
                                         points, eye, disparities, min_margin, source_id))
    return Result(source_id, results[0].matches + results[1].matches,
                  sum(r.forward_candidates for r in results), sum(r.reverse_candidates for r in results),
                  sum(r.forward_hamming_ops for r in results), sum(r.reverse_hamming_ops for r in results))
