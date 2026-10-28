"""Deterministic CPU research reference for sparse rectified stereo anchors.

The selector uses image intensity only; no ground truth, coarse prior or temporal
state enters matching. Census descriptors use the project's 5x5 full-footprint
support contract. Each accepted forward match is checked by a fresh reverse
search over the same disparity range, rather than reusing a forward cost or
assuming that another selected point supplies independent evidence.

Disparities are integer x_left - x_right, including zero; zero is a valid image
correspondence but is not automatically a valid metric depth. This is an
allocating CPU oracle, not a real-time implementation or uncertainty model.
"""
from dataclasses import dataclass

import numpy as np

from stereoforge.refinement import census


@dataclass(frozen=True)
class Match:
    eye: str
    y: int
    x: int
    other_x: int | None
    disparity: int | None
    best_cost: int | None
    second_cost: int | None
    valid: bool
    reason: str


@dataclass(frozen=True)
class Result:
    source_id: str
    matches: tuple[Match, ...]
    forward_candidates: int
    reverse_candidates: int


def select_grid(image, support, cell_size=16):
    """Pick the strongest central-difference response per cell, ties row-major.

    A zero-response cell yields no anchor. Selection uses only the reference
    image; unsupported samples are excluded from the gradient footprint.
    """
    image = np.asarray(image)
    support = np.asarray(support)
    if image.ndim != 2 or image.dtype != np.uint8 or support.shape != image.shape or support.dtype != np.bool_:
        raise ValueError('expected uint8 image and matching bool support')
    if type(cell_size) is not int or cell_size < 1:
        raise ValueError('cell_size must be a positive integer')
    h, w = image.shape
    response = np.zeros((h, w), np.int16)
    eligible = np.zeros((h, w), bool)
    if h >= 3 and w >= 3:
        a = image.astype(np.int16)
        response[1:-1, 1:-1] = (np.abs(a[1:-1, 2:] - a[1:-1, :-2]) +
                                 np.abs(a[2:, 1:-1] - a[:-2, 1:-1]))
        eligible[1:-1, 1:-1] = (support[1:-1, 1:-1] & support[1:-1, 2:] &
                                support[1:-1, :-2] & support[2:, 1:-1] & support[:-2, 1:-1])
    response[~eligible] = 0
    points = []
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            cell = response[y:y + cell_size, x:x + cell_size]
            index = int(np.argmax(cell))
            if cell.flat[index] > 0:
                dy, dx = np.unravel_index(index, cell.shape)
                points.append((y + int(dy), x + int(dx)))
    return points


def _search(reference, other, other_support, y, x, sign, disparities, min_margin):
    d = np.arange(disparities, dtype=np.int32)
    ox = x + sign * d
    inside = (ox >= 0) & (ox < other.shape[1])
    d, ox = d[inside], ox[inside]
    usable = other_support[y, ox]
    d, ox = d[usable], ox[usable]
    count = int(d.size)
    if count < 2:
        return None, None, None, count, 'too_few_candidates'
    costs = np.bitwise_count(other[y, ox] ^ reference[y, x])
    order = np.argsort(costs, kind='stable')
    best, second = int(costs[order[0]]), int(costs[order[1]])
    if second - best < min_margin:
        return None, best, second, count, 'ambiguous'
    return int(d[order[0]]), best, second, count, 'ok'


def match_descriptors(left, right, left_support, right_support, points, eye,
                      disparities=224, min_margin=1, source_id=''):
    """Match specified points in one eye and verify each with a reverse search.

    Both searches count evaluated supported descriptor pairs. Invalid points
    remain in the result with a reason; no output is filled from a prior.
    """
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
    forward_count = reverse_count = 0
    for y, x in points:
        if not (isinstance(y, (int, np.integer)) and isinstance(x, (int, np.integer)) and
                0 <= y < ref.shape[0] and 0 <= x < ref.shape[1]):
            raise ValueError('point outside reference image')
        y, x = int(y), int(x)
        if not ref_support[y, x]:
            matches.append(Match(eye, y, x, None, None, None, None, False, 'unsupported'))
            continue
        d, best, second, count, reason = _search(ref, other, other_support, y, x, sign, disparities, min_margin)
        forward_count += count
        if d is None:
            matches.append(Match(eye, y, x, None, None, best, second, False, reason))
            continue
        other_x = x + sign * d
        reverse_d, _, _, count, reverse_reason = _search(other, ref, ref_support, y, other_x, -sign, disparities, min_margin)
        reverse_count += count
        if reverse_d is None:
            reason = 'reverse_' + reverse_reason
        elif reverse_d != d:
            reason = 'reverse_mismatch'
        else:
            reason = 'ok'
        matches.append(Match(eye, y, x, other_x, d, best, second, reason == 'ok', reason))
    return Result(source_id, tuple(matches), forward_count, reverse_count)


def sparse_stereo(left, right, left_support, right_support, *, source_id,
                  cell_size=16, disparities=224, min_margin=1):
    """Compute selected anchors independently for both eyes of one frame."""
    if not isinstance(source_id, str) or not source_id:
        raise ValueError('a non-empty frame source_id is required')
    left, right = np.asarray(left), np.asarray(right)
    left_support, right_support = np.asarray(left_support), np.asarray(right_support)
    if left.shape != right.shape:
        raise ValueError('stereo image dimensions must match')
    lc, rc = census(left, left_support), census(right, right_support)
    results = []
    for eye, image, supported in (('left', left, lc.support), ('right', right, rc.support)):
        points = select_grid(image, supported, cell_size)
        results.append(match_descriptors(lc.descriptor, rc.descriptor, lc.support, rc.support,
                                         points, eye, disparities, min_margin, source_id))
    return Result(source_id, results[0].matches + results[1].matches,
                  sum(r.forward_candidates for r in results),
                  sum(r.reverse_candidates for r in results))
