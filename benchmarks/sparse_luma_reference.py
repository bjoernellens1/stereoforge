"""Offline sparse Census matcher with a lexicographic luma tie-break.

Scores encode (Hamming distance, absolute uint8 luma difference) as 256*h+l.
This relies on brightness agreement; it is not exposure invariant or independent
of Census evidence. Integer image correspondences are not metric depth and do
not resolve subpixel or mixed-footprint boundary geometry. CPU research only.
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
    forward_hamming_upper_bound: int
    reverse_hamming_upper_bound: int
    forward_luma_upper_bound: int
    reverse_luma_upper_bound: int


def _search(ref, other, ref_luma, other_luma, support, y, x, sign, disparities, min_margin):
    d = np.arange(disparities, dtype=np.int32)
    ox = x + sign * d
    inside = (ox >= 0) & (ox < other.shape[1])
    d, ox = d[inside], ox[inside]
    usable = support[y, ox]
    d, ox = d[usable], ox[usable]
    count = int(d.size)
    if count < 2:
        return None, None, None, count, 'too_few_candidates'
    hamming = np.bitwise_count(other[y, ox] ^ ref[y, x]).astype(np.int32)
    luma = np.abs(other_luma[y, ox].astype(np.int32) - int(ref_luma[y, x]))
    costs = hamming * 256 + luma
    order = np.argsort(costs, kind='stable')
    best, second = int(costs[order[0]]), int(costs[order[1]])
    if second - best < min_margin:
        return None, best, second, count, 'ambiguous'
    return int(d[order[0]]), best, second, count, 'ok'


def match_descriptors(left, right, left_luma, right_luma, left_support, right_support,
                      points, eye, disparities=224, min_margin=1, source_id=''):
    """Run a full fresh reverse search; counters bound actual comparison work."""
    left, right = np.asarray(left), np.asarray(right)
    left_luma, right_luma = np.asarray(left_luma), np.asarray(right_luma)
    left_support, right_support = np.asarray(left_support), np.asarray(right_support)
    if (left.ndim != 2 or left.shape != right.shape or left.dtype != np.uint32 or
        right.dtype != np.uint32 or left_luma.shape != left.shape or
        right_luma.shape != left.shape or left_luma.dtype != np.uint8 or
        right_luma.dtype != np.uint8 or left_support.shape != left.shape or
        right_support.shape != left.shape or left_support.dtype != np.bool_ or
        right_support.dtype != np.bool_):
        raise ValueError('expected matching uint32 descriptors, uint8 luma and bool support')
    if eye not in ('left', 'right') or type(disparities) is not int or not 1 <= disparities <= 256:
        raise ValueError('invalid eye or disparity count')
    if type(min_margin) is not int or min_margin < 1:
        raise ValueError('min_margin must be a positive integer encoded-score margin')
    ref, other = (left, right) if eye == 'left' else (right, left)
    ref_luma, other_luma = (left_luma, right_luma) if eye == 'left' else (right_luma, left_luma)
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
        if not ref_support[y, x]:
            matches.append(Match(eye, y, x, None, None, None, None, False, 'unsupported'))
            continue
        d, best, second, count, reason = _search(ref, other, ref_luma, other_luma,
                                                  other_support, y, x, sign, disparities, min_margin)
        forward += count
        if d is None:
            matches.append(Match(eye, y, x, None, None, best, second, False, reason))
            continue
        other_x = x + sign * d
        rd, _, _, count, reverse_reason = _search(other, ref, other_luma, ref_luma,
                                                   ref_support, y, other_x, -sign, disparities, min_margin)
        reverse += count
        reason = ('reverse_' + reverse_reason if rd is None else
                  'reverse_mismatch' if rd != d else 'ok')
        matches.append(Match(eye, y, x, other_x, d, best, second, reason == 'ok', reason))
    return Result(source_id, tuple(matches), forward, reverse, forward, reverse, forward, reverse)


def sparse_stereo(left, right, left_support, right_support, *, source_id,
                  cell_size=16, disparities=224, min_margin=1, selection='gradient'):
    if not isinstance(source_id, str) or not source_id:
        raise ValueError('a non-empty frame source_id is required')
    left, right = np.asarray(left), np.asarray(right)
    left_support, right_support = np.asarray(left_support), np.asarray(right_support)
    if left.shape != right.shape:
        raise ValueError('stereo image dimensions must match')
    lc, rc = census(left, left_support), census(right, right_support)
    results = []
    for eye, image, features in (('left', left, lc), ('right', right, rc)):
        points = select_points(image, features.support, features.descriptor, cell_size, selection=selection)
        results.append(match_descriptors(lc.descriptor, rc.descriptor, left, right,
                                         lc.support, rc.support, points, eye, disparities,
                                         min_margin, source_id))
    f = sum(r.forward_candidates for r in results)
    r = sum(r.reverse_candidates for r in results)
    return Result(source_id, results[0].matches + results[1].matches, f, r, f, r, f, r)
