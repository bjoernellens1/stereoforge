"""CPU validation of ordered queue timestamp boundaries for the FP2 experiment."""
from __future__ import annotations

import math


def queue_intervals(ticks, available, bits, period_ns, wall_ns):
    """Return consecutive queue intervals in milliseconds, rejecting ambiguous wraps."""
    if not isinstance(bits, int) or isinstance(bits, bool) or not 1 <= bits <= 64:
        raise ValueError('invalid timestamp width')
    if not math.isfinite(period_ns) or period_ns <= 0 or not math.isfinite(wall_ns) or wall_ns < 0:
        raise ValueError('invalid clock parameters')
    if len(ticks) < 2 or len(ticks) != len(available) or not all(x == 1 for x in available):
        raise ValueError('timestamp availability')
    modulus = 1 << bits
    if wall_ns >= modulus * period_ns:
        raise ValueError('timestamp wrap ambiguous')
    if any(not isinstance(t, int) or isinstance(t, bool) or not 0 <= t < modulus for t in ticks):
        raise ValueError('timestamp out of range')
    increments = [(b - a) % modulus for a, b in zip(ticks, ticks[1:])]
    if sum(increments) * period_ns > wall_ns:
        raise ValueError('timestamp order or elapsed time invalid')
    return [d * period_ns / 1_000_000 for d in increments]
