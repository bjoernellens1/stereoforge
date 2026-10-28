"""Independent CPU specification of experimental two-eye local Census refinement.

This module is an allocating NumPy oracle, not a runtime fallback. It imports no
GPU backend and neither copies coarse Z nor treats a prediction as evidence.
Arrays use [eye,y,x], costs use [disparity,y,x]; intervals are inclusive.

Nominal arithmetic is explicit binary32 without contraction. The optional bound
helpers propagate each operation under the Vulkan single-precision contract:
https://docs.vulkan.org/spec/latest/appendices/spirvenv.html
They require normal finite intermediate ranges (zero is allowed for add/mul).
They do not waive exact masks or qualify a future shader's contraction/rounding
policy. In particular interval-crossing LR/rounding predicates remain open gates.
"""
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from fractions import Fraction
import math

import numpy as np


CANONICAL_NAN = np.array(0x7fc00000, dtype=np.uint32).view(np.float32)[()]
ARITHMETIC_QUALIFICATION = 'GPU unqualified: operation policy and boundary predicates require independent review'


class Reason(IntFlag):
    PRIOR_MISSING = 1
    PRIOR_LOW_CONFIDENCE = 2
    PRIOR_DISCONTINUITY = 4
    PRIOR_NONFINITE = 8
    PRIOR_OUTSIDE = 16
    UNSUPPORTED = 32
    TOO_FEW = 64
    AMBIGUOUS = 128
    RESTRICTED_ENDPOINT = 256
    GLOBAL_ENDPOINT = 512
    MISSING_SUBPIXEL_NEIGHBOUR = 1024
    LR_FAILED = 2048
    EXCLUDED = 4096
    EXPLICIT_RECOVERY = 8192


class Method(IntEnum):
    NONE = 0
    GUIDED = 1
    RECOVERED = 2


def _number(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError('invalid '+name)
    if not math.isfinite(float(value)) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError('invalid '+name)


@dataclass(frozen=True)
class Settings:
    disparities: int = 128
    radius: int = 2
    subpixel: bool = True
    prior_confidence_floor: float = .05
    prior_span: float = 2.
    min_margin: int = 1
    lr_tolerance: float = 1.
    recovery: bool = False

    def __post_init__(self):
        if type(self.disparities) is not int or not 2 <= self.disparities <= 128:
            raise ValueError('disparities must be 2..128')
        if type(self.radius) is not int or self.radius not in (2, 4, 8):
            raise ValueError('target-pixel radius must be 2,4,8')
        if type(self.min_margin) is not int or not 1 <= self.min_margin <= 24:
            raise ValueError('positive integer Census uniqueness margin required')
        if type(self.subpixel) is not bool or type(self.recovery) is not bool:
            raise ValueError('boolean profiles required')
        _number(self.prior_confidence_floor, 'prior confidence', 0, 1)
        _number(self.prior_span, 'prior span', 0, 127)
        _number(self.lr_tolerance, 'LR tolerance', 0, 128)


@dataclass(frozen=True)
class Geometry:
    fx: float
    baseline_m: float
    cx_offset: float = 0.

    def __post_init__(self):
        for name in ('fx', 'baseline_m', 'cx_offset'):
            value = getattr(self, name)
            _number(value, name)
            if abs(float(value)) > np.finfo(np.float32).max:
                raise ValueError('calibration must fit binary32')
            q = np.float32(value)
            if (q != 0 and abs(q) < np.finfo(np.float32).tiny) or (name != 'cx_offset' and q <= 0):
                raise ValueError('positive normal binary32 focal length/baseline required')
            object.__setattr__(self, name, q)


@dataclass(frozen=True)
class Census:
    descriptor: np.ndarray
    support: np.ndarray


@dataclass(frozen=True)
class Prediction:
    disparity: np.ndarray
    valid: np.ndarray
    reasons: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class Match:
    disparity: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    integer_winner: np.ndarray
    best: np.ndarray
    second: np.ndarray
    subpixel_applied: np.ndarray
    reasons: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    requested: np.ndarray
    candidates: np.ndarray
    guided_candidates: np.ndarray
    recovery_candidates: np.ndarray
    method: np.ndarray


@dataclass(frozen=True)
class Result:
    predictions: tuple
    matches: tuple
    disparity: np.ndarray
    confidence: np.ndarray
    validity: np.ndarray
    lr_disparity: np.ndarray
    lr_confidence: np.ndarray
    lr_validity: np.ndarray
    depth: np.ndarray
    metric_confidence: np.ndarray
    metric_validity: np.ndarray
    provenance: np.ndarray
    source_ids: np.ndarray
    age_ns: np.ndarray
    contributor_ids: np.ndarray


def _image(value, name, shape=None):
    a = np.asarray(value)
    if a.ndim != 2 or min(a.shape) <= 0 or (shape is not None and a.shape != shape):
        raise ValueError(name+' must be nonempty HxW with matching shape')
    return a


def _mask(value, name, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'bu' or np.any(a > 1):
        raise ValueError(name+' must be a matching binary mask')
    return a.astype(bool)


def _floats(value, name, shape=None):
    a = _image(value, name, shape)
    if a.dtype != np.float32:
        raise ValueError(name+' must be binary32')
    return a


def predict_prior(disparity, confidence, valid, target_shape, settings=Settings()):
    """Exact2x half-pixel bilinear prediction; four nonzero taps required.

    The 2x phase gives weights1/16,3/16,9/16. Accumulation is sequential
    binary32 in dy,dx order, followed by binary32 multiplication by2.
    No border clamp: a nonzero-weight off-image neighbour invalidates the prior.
    """
    d = _floats(disparity, 'coarse disparity')
    c = _floats(confidence, 'coarse confidence', d.shape)
    v = _mask(valid, 'coarse validity', d.shape)
    if len(target_shape) != 2 or any(type(x) is not int for x in target_shape) or tuple(target_shape) != (2*d.shape[0], 2*d.shape[1]):
        raise ValueError('target must be an exact2x grid')
    if not np.isfinite(c).all() or np.any((c < 0) | (c > 1)):
        raise ValueError('coarse confidence must be finite0..1')
    h, w = target_shape; ch, cw = d.shape
    yy, xx = np.indices((h, w))
    # Exact quarter pixel arithmetic, including -1/4 at the outer border.
    x0 = (2*xx-1)//4; y0 = (2*yy-1)//4
    wx = ((2*xx-1) % 4).astype(np.float32)*np.float32(.25)
    wy = ((2*yy-1) % 4).astype(np.float32)*np.float32(.25)
    reasons = np.zeros((h, w), np.uint32)
    total = np.zeros((h, w), np.float32)
    minimum = np.full((h, w), np.inf, np.float32)
    maximum = np.full((h, w), -np.inf, np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            weight = (wx if dx else np.float32(1)-wx)*(wy if dy else np.float32(1)-wy)
            used = weight != 0
            x = x0+dx; y = y0+dy
            inside = (x >= 0) & (x < cw) & (y >= 0) & (y < ch)
            sx = np.clip(x, 0, cw-1); sy = np.clip(y, 0, ch-1)
            value = d[sy, sx]; finite = np.isfinite(value)
            reasons[used & ~inside] |= np.uint32(Reason.PRIOR_OUTSIDE)
            reasons[used & inside & ~v[sy, sx]] |= np.uint32(Reason.PRIOR_MISSING)
            reasons[used & inside & ~finite] |= np.uint32(Reason.PRIOR_NONFINITE)
            reasons[used & inside & (c[sy, sx] < np.float32(settings.prior_confidence_floor))] |= np.uint32(Reason.PRIOR_LOW_CONFIDENCE)
            reasons[used & inside & finite & ((value < 0) | (value > (settings.disparities-1)/2))] |= np.uint32(Reason.PRIOR_OUTSIDE)
            in_domain = finite & (value >= 0) & (value <= (settings.disparities-1)/2)
            safe = np.where(in_domain, value, np.float32(0))
            total = total + weight*safe
            minimum = np.minimum(minimum, np.where(used & inside & in_domain, value, np.inf))
            maximum = np.maximum(maximum, np.where(used & inside & in_domain, value, -np.inf))
    span = maximum-minimum
    reasons[span > np.float32(settings.prior_span)] |= np.uint32(Reason.PRIOR_DISCONTINUITY)
    predicted = np.float32(2)*total
    ok = reasons == 0
    centre = np.floor(np.where(ok, predicted, np.float32(0))+np.float32(.5)).astype(np.int32)
    lower = np.where(ok, np.maximum(0, centre-settings.radius), 0).astype(np.int32)
    upper = np.where(ok, np.minimum(settings.disparities-1, centre+settings.radius), -1).astype(np.int32)
    return Prediction(np.where(ok, predicted, CANONICAL_NAN).astype(np.float32), ok, reasons, lower, upper)


def census(image, support):
    """5x5 row-major comparison, centre excluded, lowest bit upper-left."""
    image = _image(image, 'luma')
    if image.dtype.kind not in 'ui' or np.any(image < 0) or np.any(image > 255):
        raise ValueError('luma must be integer0..255')
    support = _mask(support, 'rectification support', image.shape)
    h, w = image.shape
    descriptor = np.zeros((h, w), np.uint32); valid = np.zeros((h, w), bool)
    if h < 5 or w < 5:
        return Census(descriptor, valid)
    inner = descriptor[2:-2, 2:-2]
    footprint = np.ones(inner.shape, bool)
    centre = image[2:-2, 2:-2]; bit = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            footprint &= support[2+dy:h-2+dy, 2+dx:w-2+dx]
            if dx == 0 and dy == 0:
                continue
            inner |= (image[2+dy:h-2+dy, 2+dx:w-2+dx] < centre).astype(np.uint32) << bit
            bit += 1
    valid[2:-2, 2:-2] = footprint
    descriptor[~valid] = 0
    return Census(descriptor, valid)


def select_costs(costs, supported, lower, upper, settings=Settings()):
    """Select only explicitly supported integer costs; no sentinel arithmetic.

    Best/second are -1 if unavailable. Ties include every other supported
    hypothesis. A restricted-boundary winner is diagnostically retained but its
    raw floating output is canonical invalid. Global endpoints remain raw-valid.
    """
    costs = np.asarray(costs)
    if costs.ndim != 3 or costs.shape[0] != settings.disparities or min(costs.shape[1:]) <= 0 or costs.dtype.kind not in 'ui' or np.any(costs < 0) or np.any(costs > 24):
        raise ValueError('costs must be [D,H,W] bounded integer Census Hamming costs')
    supported = _mask(supported, 'cost support', costs.shape)
    shape = costs.shape[1:]
    lower = _image(lower, 'lower', shape); upper = _image(upper, 'upper', shape)
    if lower.dtype.kind not in 'iu' or upper.dtype.kind not in 'iu' or np.any(lower < 0) or np.any(lower >= settings.disparities) or np.any(upper < -1) or np.any(upper >= settings.disparities):
        raise ValueError('invalid inclusive search bounds')
    ds = np.arange(settings.disparities, dtype=np.int32)[:, None, None]
    allowed = supported & (ds >= lower) & (ds <= upper)
    count = allowed.sum(axis=0, dtype=np.uint32)
    #25 is exclusively an ordering key; never enters a subpixel/confidence formula.
    ordering = np.where(allowed, costs, 25)
    winner = ordering.argmin(axis=0).astype(np.int32)
    best = np.take_along_axis(ordering, winner[None], axis=0)[0].astype(np.int32)
    second = np.min(np.where(allowed & (ds != winner), costs, 25), axis=0).astype(np.int32)
    reasons = np.zeros(shape, np.uint32)
    reasons[count < 2] |= np.uint32(Reason.TOO_FEW)
    unique = (count >= 2) & (second-best >= settings.min_margin)
    reasons[(count >= 2) & ~unique] |= np.uint32(Reason.AMBIGUOUS)
    restricted = (count > 0) & (((winner == lower) & (lower > 0)) | ((winner == upper) & (upper < settings.disparities-1)))
    global_end = (count > 0) & ((winner == 0) | (winner == settings.disparities-1))
    reasons[restricted] |= np.uint32(Reason.RESTRICTED_ENDPOINT)
    reasons[global_end] |= np.uint32(Reason.GLOBAL_ENDPOINT)
    valid = unique & ~restricted
    confidence = np.zeros(shape, np.float32)
    np.divide((second-best).astype(np.float32), np.maximum(second, 1).astype(np.float32), out=confidence, where=valid)
    confidence = np.clip(confidence, np.float32(0), np.float32(1))
    disparity = winner.astype(np.float32)
    applied = np.zeros(shape, bool)
    if settings.subpixel:
        lo = np.clip(winner-1, 0, settings.disparities-1); hi = np.clip(winner+1, 0, settings.disparities-1)
        neighbours = (winner > 0) & (winner+1 < settings.disparities)
        neighbours &= np.take_along_axis(allowed, lo[None], axis=0)[0] & np.take_along_axis(allowed, hi[None], axis=0)[0]
        applied = valid & neighbours
        reasons[valid & ~neighbours] |= np.uint32(Reason.MISSING_SUBPIXEL_NEIGHBOUR)
        minus = np.take_along_axis(costs, lo[None], axis=0)[0].astype(np.float32)
        plus = np.take_along_axis(costs, hi[None], axis=0)[0].astype(np.float32)
        denominator = (minus-np.float32(2)*best.astype(np.float32))+plus
        offset = np.zeros(shape, np.float32)
        np.divide(np.float32(.5)*(minus-plus), denominator, out=offset, where=applied & (denominator != 0))
        disparity = disparity+np.clip(offset, np.float32(-1), np.float32(1))
    disparity = np.where(valid, disparity, CANONICAL_NAN).astype(np.float32)
    requested = np.maximum(upper-lower+1, 0).astype(np.uint32)
    return Match(disparity, confidence, valid, np.where(count > 0, winner, -1).astype(np.int32),
                 np.where(count > 0, best, -1).astype(np.int32), np.where(count >= 2, second, -1).astype(np.int32),
                 applied, reasons, lower.astype(np.int32), upper.astype(np.int32), requested, count,
                 count.copy(), np.zeros(shape, np.uint32), np.where(requested > 0, Method.GUIDED, Method.NONE).astype(np.uint32))


def match_eye(own, other, prediction, eye, settings=Settings(), region=None, unrestricted=False):
    """Independent Hamming matching; Left(x-d), Right(x+d), no SGM costs."""
    if type(eye) is not int or eye not in (0, 1):
        raise ValueError('eye must be0or1')
    shape = own.descriptor.shape
    for c in (own, other):
        if c.descriptor.shape != shape or c.descriptor.dtype != np.uint32:
            raise ValueError('matching Census descriptors required')
        _mask(c.support, 'Census support', shape)
    active = np.ones(shape, bool) if region is None else _mask(region, 'region', shape)
    if unrestricted:
        lower = np.zeros(shape, np.int32); upper = np.full(shape, settings.disparities-1, np.int32)
    else:
        if prediction.disparity.shape != shape:
            raise ValueError('matching prediction shape required')
        active &= prediction.valid
        lower, upper = prediction.lower.copy(), prediction.upper.copy()
    upper[~active] = -1; lower[~active] = 0
    costs = np.zeros((settings.disparities, *shape), np.uint32)
    supported = np.zeros(costs.shape, bool)
    width = shape[1]
    for d in range(min(settings.disparities, width)):
        a = slice(0, width-d) if eye else slice(d, width)
        b = slice(d, width) if eye else slice(0, width-d)
        take = active[:, a] & own.support[:, a] & other.support[:, b] & (d >= lower[:, a]) & (d <= upper[:, a])
        # Work is counted only for supported requested comparisons. Vectorize
        # across pixels; the sole data-dependent iteration is bounded disparity.
        xor = own.descriptor[:, a][take] ^ other.descriptor[:, b][take]
        costs[d, :, a][take] = np.bitwise_count(xor).astype(np.uint32)
        supported[d, :, a] = take
    result = select_costs(costs, supported, lower, upper, settings)
    if not unrestricted:
        result.reasons[:] |= prediction.reasons
    result.reasons[active & ~own.support] |= np.uint32(Reason.UNSUPPORTED)
    if region is not None:
        result.reasons[~_mask(region, 'region', shape)] |= np.uint32(Reason.EXCLUDED)
    return result


def lr_consistency(disparity, valid, tolerance=1.):
    """Noncontracted t=f32(f32(x)+/-d); u=f32(t+.5); floor(u)."""
    d = np.asarray(disparity)
    if d.ndim != 3 or d.shape[0] != 2 or d.dtype != np.float32 or min(d.shape[1:]) <= 0:
        raise ValueError('two binary32 disparity layers required')
    v = _mask(valid, 'raw validity', d.shape)
    _number(tolerance, 'LR tolerance', 0, 128)
    _, h, w = d.shape; yy, xx = np.indices((h, w)); result = np.zeros(d.shape, bool)
    for eye in (0, 1):
        finite = np.isfinite(d[eye])
        translated = xx.astype(np.float32)+(d[eye] if eye else -d[eye])
        coordinate = np.floor(translated+np.float32(.5))
        inside = finite & (coordinate >= 0) & (coordinate < w)
        target = np.where(inside, coordinate, 0).astype(np.int64)
        other = d[1-eye][yy, target]
        with np.errstate(invalid='ignore', over='ignore'):
            difference = np.abs(d[eye]-other)
        result[eye] = v[eye] & inside & v[1-eye][yy, target] & np.isfinite(other) & (difference <= np.float32(tolerance))
    return result


def _recover_match(initial, recovered, take, reasons):
    fields = {}
    for name in Match.__dataclass_fields__:
        fields[name] = np.where(take, getattr(recovered, name), getattr(initial, name))
    fields['reasons'] = initial.reasons | reasons | np.where(take, recovered.reasons, 0).astype(np.uint32)
    fields['guided_candidates'] = initial.candidates.copy()
    fields['recovery_candidates'] = np.where(take, recovered.candidates, 0).astype(np.uint32)
    fields['candidates'] = initial.candidates+fields['recovery_candidates']
    fields['requested'] = initial.requested+np.where(take, recovered.requested, 0).astype(np.uint32)
    fields['method'] = np.where(take, Method.RECOVERED, initial.method).astype(np.uint32)
    return Match(**fields)


def refine_pair(images, support, coarse_disparity, coarse_confidence, coarse_validity,
                geometry, source_id, settings=Settings(), regions=None, recovery_regions=None):
    """Full guided pass, optional one full-D recovery, then final LR/metric.

    `regions` excludes matching entirely; recovery cannot populate excluded pixels.
    `recovery_regions` explicitly requests recovery and is intersected with regions.
    Provisional LR failures are selected after *both* initial independent matches;
    no iteration or cascading recovery decisions based on half-updated results.
    """
    if not isinstance(settings, Settings) or not isinstance(geometry, Geometry):
        raise ValueError('validated Settings and Geometry required')
    if type(source_id) is not int or not 0 <= source_id < 1 << 63:
        raise ValueError('current source ID must fit nonnegative int64')
    image = np.asarray(images)
    if image.ndim != 3 or image.shape[0] != 2 or min(image.shape[1:]) <= 0:
        raise ValueError('two target image layers required')
    sup = _mask(support, 'target support', image.shape)
    region = np.ones(image.shape, bool) if regions is None else _mask(regions, 'selected regions', image.shape)
    explicit = np.zeros(image.shape, bool) if recovery_regions is None else _mask(recovery_regions, 'explicit recovery regions', image.shape)
    for a in (coarse_disparity, coarse_confidence, coarse_validity):
        if np.asarray(a).shape != (2, image.shape[1]//2, image.shape[2]//2):
            raise ValueError('two exact2x coarse prior layers required')
    prediction = tuple(predict_prior(coarse_disparity[e], coarse_confidence[e], coarse_validity[e], image.shape[1:], settings) for e in (0, 1))
    descriptors = tuple(census(image[e], sup[e]) for e in (0, 1))
    matches = tuple(match_eye(descriptors[e], descriptors[1-e], prediction[e], e, settings, region[e]) for e in (0, 1))
    provisional = lr_consistency(np.stack([m.disparity for m in matches]), np.stack([m.valid for m in matches]), settings.lr_tolerance)
    if settings.recovery:
        final = []
        for eye in (0, 1):
            initial = matches[eye]
            take = region[eye] & (~initial.valid | ~provisional[eye] | explicit[eye])
            reasons = np.zeros(take.shape, np.uint32)
            reasons[initial.valid & ~provisional[eye]] |= np.uint32(Reason.LR_FAILED)
            reasons[explicit[eye] & region[eye]] |= np.uint32(Reason.EXPLICIT_RECOVERY)
            recovered = match_eye(descriptors[eye], descriptors[1-eye], prediction[eye], eye, settings, take, unrestricted=True)
            final.append(_recover_match(initial, recovered, take, reasons))
        matches = tuple(final)
    disparity = np.stack([m.disparity for m in matches]); confidence = np.stack([m.confidence for m in matches]); validity = np.stack([m.valid for m in matches])
    lr = lr_consistency(disparity, validity, settings.lr_tolerance)
    for eye in (0, 1):
        matches[eye].reasons[validity[eye] & ~lr[eye]] |= np.uint32(Reason.LR_FAILED)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):
        denominator = disparity-geometry.cx_offset
        numerator = np.float32(geometry.fx*geometry.baseline_m)
        z = numerator/denominator
    metric = lr & sup & np.isfinite(z) & (z > 0) & (denominator > 0) & (disparity > 0)
    # Descriptor validity already establishes every paired support sample. LR also
    # explicitly gates the selected counterpart; no unsupported prediction leaks.
    provenance = metric.astype(np.uint32)
    ids = np.where(metric, source_id, -1).astype(np.int64)
    contributors = np.full((*image.shape, 4), -1, np.int64); contributors[..., 0] = ids
    return Result(prediction, matches, disparity, confidence, validity,
                  np.where(lr, disparity, CANONICAL_NAN).astype(np.float32), np.where(lr, confidence, 0).astype(np.float32), lr,
                  np.where(metric, z, CANONICAL_NAN).astype(np.float32), np.where(metric, confidence, 0).astype(np.float32), metric,
                  provenance, ids, np.zeros(image.shape, np.int64), contributors)


@dataclass(frozen=True)
class FloatBounds:
    """Closed binary64 enclosure of allowed binary32 operation results.

    A bound is conditional on the documented noncontracted operation graph and
    preserved normal arithmetic; it is not an assertion of shader qualification.
    """
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        low = np.asarray(self.lower, np.float64); high = np.asarray(self.upper, np.float64)
        if low.shape != high.shape or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
            raise ValueError('finite ordered arithmetic bounds required')
        object.__setattr__(self, 'lower', low)
        object.__setattr__(self, 'upper', high)


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    return Fraction.from_float(float(value))


def _exact(value):
    """Vectorized exact rational endpoints, preserving tiny nonzero terms.

    Used only by diagnostic arithmetic bounds, never by image matching. Binary32
    and binary64 inputs are exact binary rationals; no double arithmetic may
    erase a term before the outward binary32 rounding enclosure is chosen.
    """
    return np.asarray(np.frompyfunc(_fraction, 1, 1)(np.asarray(value)), object)


def _normal(values, zero=False):
    a = _exact(values)
    tiny = Fraction.from_float(float(np.finfo(np.float32).tiny))
    maximum = Fraction.from_float(float(np.finfo(np.float32).max))
    if np.any(np.abs(a) > maximum) or np.any((np.abs(a) < tiny) & ((a != 0) if zero else True)):
        raise ValueError('arithmetic bound is outside finite normal binary32 domain')


def _round_enclosure(low, high):
    """Bracket *exact* real endpoints by adjacent binary32 numbers.

    Converting the rational to binary64 is only an initial guess. Exact rational
    comparison subsequently corrects its direction, including a tiny lost term
    at1, exact zero and binade boundaries. Either permitted rounding direction is
    enclosed; no numerical epsilon is added.
    """
    low = _exact(low); high = _exact(high)
    _normal(low, zero=True); _normal(high, zero=True)
    lo = low.astype(np.float32); hi = high.astype(np.float32)
    lo = np.where(_exact(lo) > low, np.nextafter(lo, np.float32(-np.inf)), lo)
    hi = np.where(_exact(hi) < high, np.nextafter(hi, np.float32(np.inf)), hi)
    _normal(lo, zero=True); _normal(hi, zero=True)
    return FloatBounds(lo.astype(np.float64), hi.astype(np.float64))


def _ulp(value):
    """Vulkan's minimum enclosing-adjacent spacing for a positive rational."""
    value = abs(value)
    exponent = value.numerator.bit_length()-value.denominator.bit_length()
    power = Fraction(2)**exponent
    if value < power:
        exponent -= 1
        power /= 2
    # At a representable power of2 the adjacent gap below is the smaller one.
    spacing = Fraction(2)**(exponent-24 if value == power else exponent-23)
    return max(spacing, Fraction(2)**-149)


def _float_outward(exact, lower):
    """Publish rational endpoints as binary64 without narrowing the interval."""
    result = np.asarray(exact, object).astype(np.float64)
    wrong_side = (_exact(result) > exact) if lower else (_exact(result) < exact)
    return np.where(wrong_side, np.nextafter(result, -np.inf if lower else np.inf), result)


def _division(low, high):
    low = _exact(low); high = _exact(high)
    _normal(low); _normal(high)
    if np.any((low <= 0) & (high >= 0)):
        raise ValueError('zero-crossing division requires a separate arithmetic case')
    # ULP is monotone in magnitude except for the smaller gap at exact powers2.
    # The maximum absolute endpoint therefore supplies a conservative spacing
    # throughout this interval (also when the endpoint itself is a power2).
    maximum = np.maximum(np.abs(low), np.abs(high))
    ulp = np.asarray(np.frompyfunc(_ulp, 1, 1)(maximum), object)
    lo = low-Fraction(5, 2)*ulp
    hi = high+Fraction(5, 2)*ulp
    _normal(lo); _normal(hi)
    return FloatBounds(_float_outward(lo, True), _float_outward(hi, False))


def confidence_bounds(best, second):
    best = np.asarray(best); second = np.asarray(second)
    if best.shape != second.shape or best.dtype.kind not in 'iu' or second.dtype.kind not in 'iu' or np.any(best < 0) or np.any(second > 24) or np.any(second <= best):
        raise ValueError('unique supported integer Census costs required')
    exact = (_exact(second)-_exact(best))/_exact(np.maximum(second, 1))
    b = _division(exact, exact)
    return FloatBounds(np.clip(b.lower, 0, 1), np.clip(b.upper, 0, 1))


def subpixel_bounds(minus, best, plus, winner):
    arrays = [np.asarray(a) for a in (minus, best, plus, winner)]
    if any(a.shape != arrays[0].shape or a.dtype.kind not in 'iu' for a in arrays) or any(np.any((a < 0) | (a > 24)) for a in arrays[:3]) or np.any((arrays[3] < 0) | (arrays[3] > 127)):
        raise ValueError('bounded integer costs/winner required')
    m, b, p, w = [_exact(a) for a in arrays]
    den = (m-2*b)+p
    safe = np.where(den != 0, den, Fraction(1))
    exact = np.where(den != 0, Fraction(1, 2)*(m-p)/safe, Fraction(0))
    lo = exact.copy(); hi = exact.copy(); nonzero = exact != 0
    if np.any(nonzero):
        quotient = _division(exact[nonzero], exact[nonzero])
        lo[nonzero] = _exact(quotient.lower); hi[nonzero] = _exact(quotient.upper)
    return _round_enclosure(w+np.clip(lo, -1, 1), w+np.clip(hi, -1, 1))


def metric_bounds(disparity, geometry):
    if not isinstance(disparity, FloatBounds) or not isinstance(geometry, Geometry):
        raise ValueError('validated disparity bounds and calibration required')
    product = _exact(geometry.fx)*_exact(geometry.baseline_m)
    numerator = _round_enclosure(product, product)
    offset = _exact(geometry.cx_offset)
    denominator = _round_enclosure(_exact(disparity.lower)-offset, _exact(disparity.upper)-offset)
    if np.any(denominator.lower <= 0) or numerator.lower <= 0:
        raise ValueError('metric validity predicate is not stable positive')
    if np.any(denominator.upper > 2.**126):
        raise ValueError('denominator outside Vulkan FDiv guaranteed range')
    return _division(_exact(numerator.lower)/_exact(denominator.upper), _exact(numerator.upper)/_exact(denominator.lower))
