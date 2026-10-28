"""Offline stereo ablation: require independent vertical rank agreement.

Both cues originate in the same imagery and are correlated. This is not a
confidence calibration, a dense estimator, or a production implementation.
"""
from dataclasses import dataclass, replace

import numpy as np

if __package__:
    from . import sparse_stereo_reference as single
else:
    import sparse_stereo_reference as single


@dataclass(frozen=True)
class Result:
    source_id: str
    matches: tuple[single.Match, ...]
    forward_candidates: int
    reverse_candidates: int
    rank_forward_candidates: int
    rank_reverse_candidates: int
    rank_attempted_points: int
    rank_descriptor_pixels: int
    rank_descriptor_samples: int


def rank_descriptor(image, support):
    """Compute eight comparisons per interior pixel; support and contrast gate use."""
    image, support = np.asarray(image), np.asarray(support)
    if image.ndim != 2 or image.dtype != np.uint8 or support.shape != image.shape or support.dtype != np.bool_:
        raise ValueError('expected uint8 image and matching bool support')
    h, w = image.shape
    desc = np.zeros((h, w), np.uint8)
    footprint = np.zeros((h, w), bool)
    contrast = np.zeros((h, w), bool)
    if h >= 9:
        center = image[4:h-4]
        valid = support[4:h-4].copy()
        lo, hi = center.copy(), center.copy()
        for bit, dy in enumerate((-4, -3, -2, -1, 1, 2, 3, 4)):
            sample = image[4+dy:h-4+dy]
            valid &= support[4+dy:h-4+dy]
            desc[4:h-4] |= ((sample > center).astype(np.uint8) << bit)
            lo = np.minimum(lo, sample)
            hi = np.maximum(hi, sample)
        footprint[4:h-4] = valid
        contrast[4:h-4] = (hi.astype(np.int16) - lo.astype(np.int16)) >= 8
    pixels = max(h - 8, 0) * w
    return desc, footprint & contrast, pixels, pixels * 8


def _search(reference, other, other_support, y, x, sign, disparities, min_margin):
    d = np.arange(disparities, dtype=np.int32)
    ox = x + sign * d
    inside = (ox >= 0) & (ox < other.shape[1])
    d, ox = d[inside], ox[inside]
    usable = other_support[y, ox]
    d, ox = d[usable], ox[usable]
    if d.size < 2:
        return None, 0, 'too_few_candidates'
    costs = np.bitwise_count(other[y, ox] ^ reference[y, x])
    order = np.argsort(costs, kind='stable')
    best, second = int(costs[order[0]]), int(costs[order[1]])
    if second - best < min_margin:
        return None, int(d.size), 'ambiguous'
    return int(d[order[0]]), int(d.size), 'ok'


def sparse_stereo(left, right, left_support, right_support, *, source_id,
                  cell_size=16, disparities=224, min_margin=1, selection='gradient', baseline=None):
    """Run the cue against the baseline.

    An injected baseline is trusted experimental input: the caller must tie it
    to the same images, support, calibration and selector. A source ID alone
    does not authenticate that association; this API is not a runtime boundary.
    """
    if not isinstance(source_id, str) or not source_id:
        raise ValueError('a non-empty frame source_id is required')
    if type(disparities) is not int or not 1 <= disparities <= 256:
        raise ValueError('invalid disparity count')
    if type(min_margin) is not int or min_margin < 1:
        raise ValueError('min_margin must be a positive integer')
    left, right = np.asarray(left), np.asarray(right)
    left_support, right_support = np.asarray(left_support), np.asarray(right_support)
    if left.shape != right.shape:
        raise ValueError('stereo image dimensions must match')
    ld, ls, lp, lo = rank_descriptor(left, left_support)
    rd, rs, rp, ro = rank_descriptor(right, right_support)
    if baseline is None:
        baseline = single.sparse_stereo(left, right, left_support, right_support, source_id=source_id,
                                        cell_size=cell_size, disparities=disparities,
                                        min_margin=min_margin, selection=selection)
    if not isinstance(baseline, single.Result) or baseline.source_id != source_id:
        raise ValueError('baseline source identity mismatch')
    matches = []
    forward = reverse = attempted = 0
    h, w = left.shape
    for match in baseline.matches:
        if match.eye not in ('left', 'right') or not (0 <= match.y < h and 0 <= match.x < w):
            raise ValueError('baseline point outside stereo image')
        if not match.valid:
            matches.append(match)
            continue
        attempted += 1
        ref, other, ref_ok, other_ok, raw = ((ld, rd, ls, rs, left_support) if match.eye == 'left'
                                            else (rd, ld, rs, ls, right_support))
        sign = -1 if match.eye == 'left' else 1
        y, x = match.y, match.x
        if not ref_ok[y, x]:
            footprint = y >= 4 and y + 4 < h and bool(raw[y-4:y+5, x].all())
            matches.append(replace(match, valid=False, reason='rank_low_contrast' if footprint else 'rank_unsupported'))
            continue
        d, n, why = _search(ref, other, other_ok, y, x, sign, disparities, min_margin)
        forward += n
        if d is None:
            matches.append(replace(match, valid=False, reason='rank_' + why))
            continue
        other_x = x + sign * d
        back, n, why = _search(other, ref, ref_ok, y, other_x, -sign, disparities, min_margin)
        reverse += n
        if back is None:
            reason = 'rank_reverse_' + why
        elif back != d:
            reason = 'rank_reverse_mismatch'
        elif d != match.disparity:
            reason = 'rank_disagreement'
        else:
            reason = 'ok'
        matches.append(match if reason == 'ok' else replace(match, valid=False, reason=reason))
    return Result(source_id, tuple(matches), baseline.forward_candidates, baseline.reverse_candidates,
                  forward, reverse, attempted, lp + rp, lo + ro)
