#!/usr/bin/env python3
"""Causal, static-camera disparity consensus reference (NumPy; not a GPU path).

Only direct input measurements vote. A held estimate never votes for itself.
Outputs distinguish raw current, corroborated current, and historical fill.
RGB guards reject visible motion; they do not establish geometric alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np

INVALID, RAW_CURRENT, CONFIRMED_CURRENT, HELD_HISTORY = 0, 1, 2, 3


@dataclass(frozen=True)
class ConsensusConfig:
    window_frames: int = 9
    min_hits: int = 3
    agreement_px: float = 1.0
    dominance: float = 0.6
    evidence_ms: float = 500.0
    hold_ms: float = 100.0
    reset_gap_ms: float = 500.0
    rgb_delta: float = 12.0
    rgb_dilate: int = 1
    global_shift_px: float = 0.5
    phase_response_min: float = 0.1
    global_guard: bool = True

    def validate(self) -> None:
        if type(self.window_frames) is not int or not 2 <= self.window_frames <= 32:
            raise ValueError("window_frames must be an integer in [2,32]")
        if type(self.min_hits) is not int or not 2 <= self.min_hits <= self.window_frames:
            raise ValueError("min_hits must be in [2,window_frames]")
        for key in ("agreement_px", "evidence_ms", "reset_gap_ms", "rgb_delta", "global_shift_px"):
            v = getattr(self, key)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(self.hold_ms) or not 0 <= self.hold_ms <= self.evidence_ms:
            raise ValueError("hold_ms must be finite, nonnegative, and <= evidence_ms")
        if not math.isfinite(self.dominance) or not 0.5 < self.dominance <= 1:
            raise ValueError("dominance must be in (0.5,1]")
        if type(self.rgb_dilate) is not int or not 0 <= self.rgb_dilate <= 8:
            raise ValueError("rgb_dilate must be an integer in [0,8]")
        if not math.isfinite(self.phase_response_min) or not 0 <= self.phase_response_min <= 1:
            raise ValueError("phase_response_min must be in [0,1]")


@dataclass
class ConsensusFrame:
    depth: np.ndarray
    provenance: np.ndarray
    age_ms: np.ndarray
    support_hits: np.ndarray
    diagnostics: dict


def valid_depth(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0)


def corrected_disparity(depth: np.ndarray, fxb: float) -> np.ndarray:
    """Return fB/Z = d - (cx_left-cx_right), not uncorrected disparity."""
    z = np.asarray(depth, dtype=np.float32)
    out = np.full(z.shape, np.nan, dtype=np.float32)
    np.divide(fxb, z, out=out, where=valid_depth(z))
    out[~np.isfinite(out)] = np.nan
    return out


def gray_image(rgba: np.ndarray) -> np.ndarray:
    import cv2
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError("RGB input must be uint8 HxWx3 or HxWx4")
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY if rgba.shape[2] == 4 else cv2.COLOR_RGB2GRAY)


def raw_frame(depth: np.ndarray) -> ConsensusFrame:
    out = np.asarray(depth, dtype=np.float32).copy()
    good = valid_depth(out)
    out[~good] = np.nan
    provenance = good.astype(np.uint8) * RAW_CURRENT
    age = np.where(good, 0.0, np.nan).astype(np.float32)
    return ConsensusFrame(out, provenance, age, good.astype(np.uint8), {})


class TemporalConsensus:
    """One history per fixed pixel; never use for arbitrary moving-camera fusion.

    receipt_ns is a monotonically increasing host receipt clock. Ages refer to
    last corroborated input receipt, NOT exposure time or GPU measurement time.
    source_stamp_ns is checked for duplicate/backward frame identities only.
    """
    def __init__(self, shape: tuple[int, int], fxb: float, config: ConsensusConfig):
        config.validate()
        if (len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape)
                or math.prod(shape) > 16_777_216):
            raise ValueError("invalid image shape")
        if not math.isfinite(fxb) or fxb <= 0:
            raise ValueError("fxb must be finite and positive")
        self.shape, self.fxb, self.config = shape, float(fxb), config
        self.history = np.full((config.window_frames, *shape), np.nan, np.float32)
        self.history_ns = np.zeros(config.window_frames, np.int64)
        self.estimate = np.full(shape, np.nan, np.float32)
        self.last_support_ns = np.zeros(shape, np.int64)
        self.support_gray = np.zeros(shape, np.uint8)
        self.previous_gray = None
        self.motion_reference = None
        self.phase_window = None
        self.last_receipt_ns = self.last_source_stamp_ns = 0
        self.slot = 0

    def _clear(self, mask: np.ndarray | None = None) -> None:
        if mask is None:
            self.history.fill(np.nan)
            self.history_ns.fill(0)
            self.estimate.fill(np.nan)
            self.last_support_ns.fill(0)
        else:
            self.history[:, mask] = np.nan
            self.estimate[mask] = np.nan
            self.last_support_ns[mask] = 0

    def _motion(self, gray: np.ndarray) -> tuple[bool, dict]:
        if not self.config.global_guard:
            return False, {"global_motion_guard": "disabled"}
        import cv2
        h, w = gray.shape
        scale = min(1.0, 320.0 / w)
        size = (max(2, round(w * scale)), max(2, round(h * scale)))
        small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.motion_reference is None:
            self.motion_reference = small.copy()
            self.phase_window = cv2.createHanningWindow(size, cv2.CV_32F)
            return False, {"global_motion_guard": "initial_reference"}
        # Uniform RGB does not prove an unmoving camera: fail closed to raw.
        if float(small.std()) < 1.0 or float(self.motion_reference.std()) < 1.0:
            self.motion_reference = small.copy()
            return True, {"global_motion_guard": "unobservable_texture"}
        shift, response = cv2.phaseCorrelate(
            self.motion_reference.copy(), small.copy(), self.phase_window)
        shift_px = math.hypot(*shift) / scale
        reset = (not math.isfinite(shift_px) or not math.isfinite(response)
                 or response < self.config.phase_response_min
                 or shift_px > self.config.global_shift_px)
        if reset:
            self.motion_reference = small.copy()
        return reset, {"global_motion_guard": "reset" if reset else "accepted",
                       "reference_translation_px": shift_px if math.isfinite(shift_px) else None,
                       "phase_response": response if math.isfinite(response) else None}

    def update(self, depth: np.ndarray, rgba: np.ndarray, *, receipt_ns: int,
               source_stamp_ns: int) -> ConsensusFrame:
        started = time.perf_counter_ns()
        z = np.asarray(depth, dtype=np.float32)
        if z.shape != self.shape or rgba.shape[:2] != self.shape:
            raise ValueError("frame geometry changed")
        for name, value, previous in (("receipt", receipt_ns, self.last_receipt_ns),
                                      ("source stamp", source_stamp_ns, self.last_source_stamp_ns)):
            if not isinstance(value, (int, np.integer)) or not 0 < value <= np.iinfo(np.int64).max:
                raise ValueError(f"{name} must be a positive int64 timestamp")
            if value <= previous:
                raise ValueError(f"{name} must strictly increase; duplicate/backward frame")
        gray = gray_image(rgba)
        c = self.config
        motion_reset, motion_info = self._motion(gray)
        gap_reset = bool(self.last_receipt_ns and receipt_ns - self.last_receipt_ns > c.reset_gap_ms * 1e6)
        reset_all = motion_reset or gap_reset
        photo_changed = np.zeros(self.shape, bool)
        if self.previous_gray is not None:
            photo_changed = np.abs(gray.astype(np.int16) - self.previous_gray.astype(np.int16)) > c.rgb_delta
        cached = np.isfinite(self.estimate)
        photo_changed |= cached & (np.abs(gray.astype(np.int16) - self.support_gray.astype(np.int16)) > c.rgb_delta)
        if c.rgb_dilate and photo_changed.any():
            import cv2
            k = 2 * c.rgb_dilate + 1
            photo_changed = cv2.dilate(photo_changed.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        if reset_all:
            self._clear()
            photo_changed[:] = True  # fall back to current raw, not stale history
        elif photo_changed.any():
            self._clear(photo_changed)
        for k in range(c.window_frames):
            if self.history_ns[k] and receipt_ns - int(self.history_ns[k]) > c.evidence_ms * 1e6:
                self.history[k].fill(np.nan)
                self.history_ns[k] = 0
        q = corrected_disparity(z, self.fxb)
        good = np.isfinite(q) & (q > 0)
        # A direct contradictory observation invalidates historical continuity.
        # It is not hidden by clinging to a previously stable surface.
        contradiction = good & np.isfinite(self.estimate) & (np.abs(q - self.estimate) > c.agreement_px)
        if contradiction.any():
            self._clear(contradiction)
        self.history[self.slot] = q
        self.history_ns[self.slot] = receipt_ns
        self.slot = (self.slot + 1) % c.window_frames
        anchor = np.where(good, q, self.estimate)
        hits = np.zeros(self.shape, np.uint8)
        available = np.zeros(self.shape, np.uint8)
        total = np.zeros(self.shape, np.float32)
        for plane in self.history:
            observed = np.isfinite(plane)
            agree = observed & np.isfinite(anchor) & (np.abs(plane - anchor) <= c.agreement_px)
            available += observed
            hits += agree
            total += np.where(agree, plane, 0.0)
        consistent = (hits >= c.min_hits) & (hits >= c.dominance * available)
        estimate = np.full(self.shape, np.nan, np.float32)
        np.divide(total, hits, out=estimate, where=consistent)
        current = good & consistent & ~photo_changed
        self.estimate[current] = estimate[current]
        self.last_support_ns[current] = receipt_ns
        self.support_gray[current] = gray[current]
        age = (receipt_ns - self.last_support_ns).astype(np.float64) / 1e6
        held = (~good & consistent & np.isfinite(self.estimate) & ~photo_changed
                & (self.last_support_ns > 0) & (age <= c.hold_ms) & (c.hold_ms > 0))
        out = np.full(self.shape, np.nan, np.float32)
        accepted = current | held
        np.divide(self.fxb, estimate, out=out, where=accepted)
        prov = np.zeros(self.shape, np.uint8)
        prov[current], prov[held] = CONFIRMED_CURRENT, HELD_HISTORY
        raw_fallback = photo_changed & good
        out[raw_fallback] = z[raw_fallback]
        prov[raw_fallback] = RAW_CURRENT
        age_out = np.full(self.shape, np.nan, np.float32)
        age_out[current | raw_fallback] = 0
        age_out[held] = age[held]
        self.previous_gray = gray.copy()
        self.last_receipt_ns, self.last_source_stamp_ns = int(receipt_ns), int(source_stamp_ns)
        diagnostics = {**motion_info, "gap_reset": gap_reset,
                       "rgb_reset_pixels": int(photo_changed.sum()),
                       "contradiction_reset_pixels": int(contradiction.sum()),
                       "cpu_reference_ms": (time.perf_counter_ns() - started) / 1e6}
        return ConsensusFrame(out, prov, age_out, hits, diagnostics)
