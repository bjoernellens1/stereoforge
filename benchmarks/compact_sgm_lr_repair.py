"""Offline opposite-reference Census competition; not independent SGM evidence.

Each request originates from a valid original winner with an invalid opposite
winner. A target is searched once over the complete configured disparity range.
This is correlated stereo evidence: the initiating winner selects the target,
and both eyes use the same image pair. Never use as a live depth producer.
"""
from dataclasses import dataclass
import numpy as np
from stereoforge.refinement import lr_consistency


@dataclass
class Repair:
    raw: np.ndarray
    valid: np.ndarray
    lr: np.ndarray
    repaired: np.ndarray
    requests: int
    targets: int
    candidates: int
    ties: int
    unsupported: int
    conflicts: int
    zero_winners: int


def repair(raw, valid, descriptors, support, disparities=128):
    raw = np.asarray(raw)
    valid = np.asarray(valid, dtype=bool)
    descriptors = np.asarray(descriptors)
    support = np.asarray(support, dtype=bool)
    if (raw.dtype != np.float32 or raw.ndim != 3 or raw.shape[0] != 2
            or any(a.shape != raw.shape for a in (valid, descriptors, support))
            or descriptors.dtype != np.uint32 or not 2 <= disparities <= 128
            or np.any(valid & (~np.isfinite(raw) | (raw <= 0)))):
        raise ValueError('invalid stereo planes or disparity range')
    _, height, width = raw.shape
    requests = {}
    request_count = 0
    for eye, y, x in np.argwhere(valid):
        # Match lr_consistency's binary32, half-up image-coordinate rule.
        offset = raw[eye, y, x] if eye else -raw[eye, y, x]
        target = int(np.floor(np.float32(np.float32(x) + offset) + np.float32(.5)))
        if 0 <= target < width and not valid[1-eye, y, target]:
            request_count += 1
            key = (1-int(eye), int(y), target)
            requests[key] = requests.get(key, 0) + 1
    result = raw.copy()
    final_valid = valid.copy()
    repaired = np.zeros(raw.shape, bool)
    candidates = ties = unsupported = conflicts = zero_winners = 0
    for eye, y, x in sorted(requests):
        if requests[(eye, y, x)] > 1:
            conflicts += 1
            continue
        if not support[eye, y, x]:
            unsupported += 1
            continue
        costs = []
        for d in range(disparities):
            other_x = x+d if eye else x-d
            if 0 <= other_x < width and support[1-eye, y, other_x]:
                cost = int(descriptors[eye, y, x] ^ descriptors[1-eye, y, other_x]).bit_count()
                costs.append((cost, d))
        candidates += len(costs)
        if len(costs) < 2:
            unsupported += 1
            continue
        costs.sort()
        if costs[0][0] == costs[1][0]:
            ties += 1
            continue
        if costs[0][1] == 0:
            zero_winners += 1
            continue
        result[eye, y, x] = np.float32(costs[0][1])
        final_valid[eye, y, x] = repaired[eye, y, x] = True
    return Repair(result, final_valid, lr_consistency(result, final_valid, 1.), repaired,
                  request_count, len(requests), candidates, ties, unsupported, conflicts, zero_winners)
