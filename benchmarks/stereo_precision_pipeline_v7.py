#!/usr/bin/env python3
"""StereoForge v7 — offline evaluation of 32-bit deadband multi-scale Census,
interpolated subpixel LR consistency, disparity-domain 8-connected speckle
filtering, and instantaneous vs. 2-frame motion-guarded temporal corroboration.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np


def read_fixture_luma(folder: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = json.loads((folder / "manifest.json").read_text())
    shape = meta.get("rectified") or meta.get("rectification")
    if not isinstance(shape, dict):
        raise ValueError("rectified dimensions missing")
    h, w = int(shape["height"]), int(shape["width"])
    out = []
    for eye in ("left", "right"):
        entry = meta["expected"][eye + "_luma"]
        rel = Path(entry["file"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("unsafe fixture path")
        b = (folder / rel).read_bytes()
        if (
            len(b) != h * w * 4
            or len(b) != entry["bytes"]
            or hashlib.sha256(b).hexdigest() != entry["sha256"]
        ):
            raise ValueError("luma fixture hash/size mismatch")
        out.append(np.frombuffer(b, dtype="<u4").reshape(h, w).copy())
    return out[0], out[1], meta


def census_24bit_legacy(luma: np.ndarray) -> np.ndarray:
    """Legacy 24-bit strict-< 5x5 Census."""
    if luma.ndim != 2 or luma.shape[0] < 5 or luma.shape[1] < 5:
        raise ValueError("luma too small")
    h, w = luma.shape
    out = np.zeros((h, w), dtype=np.uint32)
    center = luma[2 : h - 2, 2 : w - 2]
    bit = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            out[2 : h - 2, 2 : w - 2] |= (
                luma[2 + dy : h - 2 + dy, 2 + dx : w - 2 + dx] < center
            ).astype(np.uint32) << bit
            bit += 1
    return out


# Symmetric pairs for bits 24..31: (dy1, dx1, dy2, dx2) within 5x5 footprint
SYMMETRIC_PAIRS_8: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 0, -1),
    (0, -1, 0, 1),
    (1, 0, -1, 0),
    (-1, 0, 1, 0),
    (0, 2, 0, -2),
    (0, -2, 0, 2),
    (2, 0, -2, 0),
    (-2, 0, 2, 0),
)


def census_32bit_multiscale(luma: np.ndarray, epsilon: int = 0) -> np.ndarray:
    """32-bit multi-scale center-symmetric + center-referenced Census within the 5x5 support footprint.

    - Bits 0..23: 5x5 neighbor + epsilon < center.
    - Bits 24..31: 8 center-symmetric ternary horizontal/vertical gradient comparisons at
      radii 1 and 2 (with optional deadband epsilon).
    """
    if luma.ndim != 2 or luma.shape[0] < 5 or luma.shape[1] < 5:
        raise ValueError("luma too small")
    if epsilon < 0:
        raise ValueError("negative epsilon")
    h, w = luma.shape
    out = np.zeros((h, w), dtype=np.uint32)
    center = luma[2 : h - 2, 2 : w - 2].astype(np.int64)
    eps = int(epsilon)
    bit = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            nb = luma[2 + dy : h - 2 + dy, 2 + dx : w - 2 + dx].astype(np.int64)
            out[2 : h - 2, 2 : w - 2] |= ((nb + eps) < center).astype(np.uint32) << bit
            bit += 1
    for dy1, dx1, dy2, dx2 in SYMMETRIC_PAIRS_8:
        a = luma[2 + dy1 : h - 2 + dy1, 2 + dx1 : w - 2 + dx1].astype(np.int64)
        b = luma[2 + dy2 : h - 2 + dy2, 2 + dx2 : w - 2 + dx2].astype(np.int64)
        out[2 : h - 2, 2 : w - 2] |= ((a + eps) < b).astype(np.uint32) << bit
        bit += 1
    return out


def census_winner_audit(
    ref: np.ndarray,
    match: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    D: int,
    eye: str = "left",
    max_bits: int = 32,
    box_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate local Census winner, best cost, second-best cost, and uniqueness."""
    if ref.shape != match.shape or ref.ndim != 2:
        raise ValueError("Census shape mismatch")
    h, w = ref.shape
    ys = np.asarray(ys, dtype=np.int32)
    xs = np.asarray(xs, dtype=np.int32)
    n = ys.size
    best = np.full(n, 32767, np.int32)
    second = np.full(n, 32767, np.int32)
    best_d = np.zeros(n, np.int16)
    margin = 2 + box_radius
    for d in range(1, D):
        target = xs + (-d if eye == "left" else d)
        ok = (target >= margin) & (target < w - margin) & (ys >= margin) & (ys < h - margin)
        cost = np.full(n, 32767, dtype=np.int32)
        idx = np.flatnonzero(ok)
        if idx.size:
            acc = np.zeros(idx.size, dtype=np.int32)
            for by in range(-box_radius, box_radius + 1):
                for bx in range(-box_radius, box_radius + 1):
                    xor = np.bitwise_xor(
                        ref[ys[idx] + by, xs[idx] + bx],
                        match[ys[idx] + by, target[idx] + bx],
                    )
                    acc += np.bitwise_count(xor).astype(np.int32)
            cost[idx] = acc
        improved = cost < best
        second = np.where(improved, best, np.minimum(second, cost))
        best_d = np.where(improved, d, best_d)
        best = np.minimum(cost, best)
    max_valid_cost = max_bits * ((2 * box_radius + 1) ** 2)
    valid = (best <= max_valid_cost) & (second > best) & (best_d > 0) & (best_d + 1 < D)
    return best_d, best, second, valid


def lr_consistency_interpolated(
    left: np.ndarray,
    right: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    tolerance: float = 1.0,
    interpolate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare continuous subpixel LR consistency vs. nearest-integer rounded LR check."""
    if left.shape != right.shape or left.shape != left_valid.shape or left.shape != right_valid.shape:
        raise ValueError("LR input shape mismatch")
    h, w = left.shape
    yy, xx = np.indices((h, w), dtype=np.int32)
    lv = (left_valid != 0) & np.isfinite(left) & (left > 0)
    rv = (right_valid != 0) & np.isfinite(right) & (right > 0)
    if not interpolate:
        dest = np.floor(xx.astype(np.float64) - np.where(lv, left, 0.0) + 0.5).astype(np.int32)
        inside = lv & (dest >= 0) & (dest < w)
        t = np.clip(dest, 0, w - 1)
        both = inside & rv[yy, t]
        residual = np.where(both, np.abs(left - right[yy, t]), np.nan)
        accepted = both & (residual <= tolerance)
        return accepted, residual

    xr = xx.astype(np.float64) - np.where(lv, left.astype(np.float64), 0.0)
    x0 = np.floor(xr).astype(np.int32)
    x1 = x0 + 1
    alpha = xr - x0.astype(np.float64)
    in0 = lv & (x0 >= 0) & (x0 < w)
    in1 = lv & (x1 >= 0) & (x1 < w)
    c0 = np.clip(x0, 0, w - 1)
    c1 = np.clip(x1, 0, w - 1)
    v0 = in0 & rv[yy, c0]
    v1 = in1 & rv[yy, c1]
    d0 = right[yy, c0].astype(np.float64)
    d1 = right[yy, c1].astype(np.float64)
    both_valid_and_smooth = v0 & v1 & (np.abs(d0 - d1) <= 1.5)
    # Fallback to nearest valid integer sample at occlusion edges
    nearest = np.floor(xr + 0.5).astype(np.int32)
    in_near = lv & (nearest >= 0) & (nearest < w)
    cn = np.clip(nearest, 0, w - 1)
    vn = in_near & rv[yy, cn]
    dn = right[yy, cn].astype(np.float64)
    interp_d = np.where(both_valid_and_smooth, (1.0 - alpha) * d0 + alpha * d1, dn)
    comparable = both_valid_and_smooth | vn
    residual = np.where(comparable, np.abs(left.astype(np.float64) - interp_d), np.nan)
    accepted = comparable & (residual <= tolerance)
    return accepted, residual.astype(np.float32)


def local_disparity_support_filter(
    depth: np.ndarray,
    fx_baseline: float,
    max_disp_step_px: float = 1.25,
    min_neighbors: int = 2,
    smooth_agree_px: float = 1.0,
) -> np.ndarray:
    """3x3 disparity-domain local support & edge-preserving subpixel smoothing."""
    if fx_baseline <= 0:
        raise ValueError("fx_baseline must be positive")
    valid = np.isfinite(depth) & (depth > 0)
    disp = np.where(valid, fx_baseline / np.where(valid, depth, 1.0), np.nan).astype(np.float32)
    h, w = disp.shape
    padded_d = np.pad(disp, 1, mode="constant", constant_values=np.nan)
    padded_v = np.pad(valid, 1, mode="constant", constant_values=False)
    agree_count = np.zeros((h, w), dtype=np.int32)
    smooth_sum = np.where(valid, disp, 0.0).astype(np.float64)
    smooth_weight = np.where(valid, 1.0, 0.0).astype(np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb_v = padded_v[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
            nb_d = padded_d[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
            diff = np.abs(disp - nb_d)
            ok_support = valid & nb_v & (diff <= max_disp_step_px)
            agree_count += ok_support.astype(np.int32)
            ok_smooth = valid & nb_v & (diff <= smooth_agree_px)
            w_tap = np.where(ok_smooth, 0.5, 0.0)
            smooth_sum += np.where(ok_smooth, nb_d.astype(np.float64) * w_tap, 0.0)
            smooth_weight += w_tap
    keep = valid & (agree_count >= min_neighbors)
    refined_disp = np.where(keep, smooth_sum / np.maximum(smooth_weight, 1e-6), np.nan)
    return np.where(keep & (refined_disp > 0), fx_baseline / refined_disp, np.nan).astype(np.float32)


def speckle_filter_depth(
    depth: np.ndarray,
    min_component_pixels: int = 50,
    max_neighbor_delta_m: float = 0.05,
    fx_baseline_px_m: float = 0.0,
    max_neighbor_disparity_px: float = 0.0,
    eight_connected: bool = True,
) -> tuple[np.ndarray, dict]:
    """Reference implementation of C++ DepthSpeckleFilter with disparity-domain & 8-connectivity."""
    if depth.ndim != 2:
        raise ValueError("depth must be 2D")
    h, w = depth.shape
    out = depth.astype(np.float32).copy()
    valid = np.isfinite(out) & (out > 0.0)
    use_disp = fx_baseline_px_m > 0.0 and max_neighbor_disparity_px > 0.0
    domain = np.where(valid, fx_baseline_px_m / np.where(valid, out, 1.0), np.nan) if use_disp else out
    threshold = max_neighbor_disparity_px if use_disp else max_neighbor_delta_m

    seen = np.zeros((h, w), dtype=np.uint8)
    offsets = (
        ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        if eight_connected
        else ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    input_valid = int(np.count_nonzero(valid))
    invalidated = 0
    removed_components = 0
    retained_components = 0

    ys, xs = np.nonzero(valid)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        seen[sy, sx] = 1
        comp = [(sy, sx)]
        q = deque([(sy, sx)])
        while q:
            cy, cx = q.popleft()
            v_at = float(domain[cy, cx])
            for dy, dx in offsets:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and valid[ny, nx]:
                    if abs(float(domain[ny, nx]) - v_at) <= threshold:
                        seen[ny, nx] = 1
                        comp.append((ny, nx))
                        q.append((ny, nx))
        if len(comp) >= min_component_pixels:
            retained_components += 1
        else:
            removed_components += 1
            invalidated += len(comp)
            for ry, rx in comp:
                out[ry, rx] = np.nan
    return out, {
        "input_valid_pixels": input_valid,
        "output_valid_pixels": input_valid - invalidated,
        "invalidated_pixels": invalidated,
        "removed_components": removed_components,
        "retained_components": retained_components,
    }


def evaluate_temporal_corroboration_sequence(
    seq_dir: Path,
    fx_baseline: float,
    max_frames: int = 30,
    disparity_tol_px: float = 1.0,
    rgb_motion_threshold: float = 12.0,
) -> dict:
    """Compare (A) Instantaneous Disparity-Domain Speckle Filter vs.
    (B) 2-Frame Motion-Guarded Temporal Corroboration (no stale holding).
    """
    depth_files = sorted(seq_dir.glob("frame-*.depth.f32"))[:max_frames]
    rgb_files = sorted(seq_dir.glob("frame-*.rgb.rgba8"))[:max_frames]
    if len(depth_files) < 3 or len(depth_files) != len(rgb_files):
        raise ValueError(f"Expected >=3 matched depth/rgb frames in {seq_dir}")

    h, w = 720, 1280
    raw_coverages, inst_coverages, temp_coverages = [], [], []
    raw_flicker, inst_flicker, temp_flicker = [], [], []
    raw_speckle_pixels, inst_speckle_pixels, temp_speckle_pixels = [], [], []

    prev_disp = None
    prev_luma = None
    prev_raw_valid = None
    prev_inst_valid = None
    prev_temp_valid = None

    for df, rf in zip(depth_files, rgb_files):
        depth = np.fromfile(df, dtype="<f4").reshape(h, w)
        rgba = np.fromfile(rf, dtype=np.uint8).reshape(h, w, 4)
        luma = (
            0.299 * rgba[:, :, 0].astype(np.float32)
            + 0.587 * rgba[:, :, 1].astype(np.float32)
            + 0.114 * rgba[:, :, 2].astype(np.float32)
        )

        # (0) Raw input
        raw_valid = np.isfinite(depth) & (depth > 0)
        raw_disp = np.where(raw_valid, fx_baseline / np.where(raw_valid, depth, 1.0), np.nan)
        _, raw_sp = speckle_filter_depth(
            depth, min_component_pixels=40, fx_baseline_px_m=fx_baseline,
            max_neighbor_disparity_px=1.25, eight_connected=True
        )
        raw_speckle_pixels.append(raw_sp["invalidated_pixels"])

        # (A) Instantaneous disparity-domain 8-connected + 3x3 local support
        inst_depth = local_disparity_support_filter(
            depth, fx_baseline=fx_baseline, max_disp_step_px=1.25, min_neighbors=2
        )
        inst_depth, _ = speckle_filter_depth(
            inst_depth, min_component_pixels=40, fx_baseline_px_m=fx_baseline,
            max_neighbor_disparity_px=1.25, eight_connected=True
        )
        inst_valid = np.isfinite(inst_depth) & (inst_depth > 0)
        _, inst_sp = speckle_filter_depth(
            inst_depth, min_component_pixels=40, fx_baseline_px_m=fx_baseline,
            max_neighbor_disparity_px=1.25, eight_connected=True
        )
        inst_speckle_pixels.append(inst_sp["invalidated_pixels"])

        # (B) 2-frame motion-guarded temporal corroboration (no stale holding)
        inst_disp = np.where(inst_valid, fx_baseline / np.where(inst_valid, inst_depth, 1.0), np.nan)
        if prev_disp is None:
            temp_depth = inst_depth.copy()
        else:
            motion = np.abs(luma - prev_luma) > rgb_motion_threshold
            prev_ok = np.isfinite(prev_disp) & (np.abs(inst_disp - prev_disp) <= disparity_tol_px)
            # Check strong 3x3 spatial support (at least 5 neighbors) to allow newly visible static surfaces
            padded_v = np.pad(inst_valid, 1, mode="constant", constant_values=False)
            padded_d = np.pad(inst_disp, 1, mode="constant", constant_values=np.nan)
            strong_nb = np.zeros((h, w), dtype=np.int32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nb_v = padded_v[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
                    nb_d = padded_d[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
                    strong_nb += (inst_valid & nb_v & (np.abs(inst_disp - nb_d) <= 1.0)).astype(np.int32)
            keep_temp = inst_valid & (motion | prev_ok | (strong_nb >= 6))
            temp_disp = np.where(
                keep_temp & prev_ok & (~motion),
                0.5 * (inst_disp + prev_disp),
                np.where(keep_temp, inst_disp, np.nan),
            )
            temp_depth = np.where(np.isfinite(temp_disp) & (temp_disp > 0), fx_baseline / temp_disp, np.nan).astype(np.float32)

        temp_valid = np.isfinite(temp_depth) & (temp_depth > 0)
        _, temp_sp = speckle_filter_depth(
            temp_depth, min_component_pixels=40, fx_baseline_px_m=fx_baseline,
            max_neighbor_disparity_px=1.25, eight_connected=True
        )
        temp_speckle_pixels.append(temp_sp["invalidated_pixels"])

        raw_coverages.append(float(raw_valid.mean()))
        inst_coverages.append(float(inst_valid.mean()))
        temp_coverages.append(float(temp_valid.mean()))

        if prev_raw_valid is not None:
            raw_flicker.append(float(np.mean(raw_valid != prev_raw_valid)))
            inst_flicker.append(float(np.mean(inst_valid != prev_inst_valid)))
            temp_flicker.append(float(np.mean(temp_valid != prev_temp_valid)))

        prev_disp = raw_disp
        prev_luma = luma
        prev_raw_valid = raw_valid
        prev_inst_valid = inst_valid
        prev_temp_valid = temp_valid

    return {
        "frames_evaluated": len(depth_files),
        "raw": {
            "mean_coverage": float(np.mean(raw_coverages)),
            "mean_validity_flicker_rate": float(np.mean(raw_flicker)),
            "mean_isolated_speckle_pixels_lt40": float(np.mean(raw_speckle_pixels)),
        },
        "instantaneous_disparity_speckle": {
            "mean_coverage": float(np.mean(inst_coverages)),
            "mean_validity_flicker_rate": float(np.mean(inst_flicker)),
            "mean_isolated_speckle_pixels_lt40": float(np.mean(inst_speckle_pixels)),
        },
        "motion_guarded_temporal_corroboration": {
            "mean_coverage": float(np.mean(temp_coverages)),
            "mean_validity_flicker_rate": float(np.mean(temp_flicker)),
            "mean_isolated_speckle_pixels_lt40": float(np.mean(temp_speckle_pixels)),
        },
    }


def run_audit(fixture_dir: Path, sequence_dir: Path | None, stride: int = 8, disparities: int = 128) -> dict:
    left, right, meta = read_fixture_luma(fixture_dir)
    h, w = left.shape
    yy, xx = np.mgrid[4 : h - 4 : stride, 4 : w - 4 : stride]
    ys = yy.reshape(-1)
    xs = xx.reshape(-1)

    c24_l = census_24bit_legacy(left)
    c24_r = census_24bit_legacy(right)
    c32_l = census_32bit_multiscale(left, epsilon=0)
    c32_r = census_32bit_multiscale(right, epsilon=0)
    c32_eps1_l = census_32bit_multiscale(left, epsilon=1)
    c32_eps1_r = census_32bit_multiscale(right, epsilon=1)

    def eval_census_variant(dl, dr, max_bits: int, box_radius: int):
        ld, lb, ls, lv = census_winner_audit(dl, dr, ys, xs, disparities, "left", max_bits=max_bits, box_radius=box_radius)
        dest = xs - ld
        inbounds = lv & (dest >= 2 + box_radius) & (dest < w - 2 - box_radius)
        rhs = ys[inbounds]
        rxs = dest[inbounds].astype(np.int32)
        rt_d, _, _, rt_v = census_winner_audit(dr, dl, rhs, rxs, disparities, "right", max_bits=max_bits, box_radius=box_radius)
        agree = (np.abs(ld[inbounds] - rt_d) <= 1) & rt_v
        max_c = max_bits * ((2 * box_radius + 1) ** 2)
        return {
            "samples": int(ys.size),
            "unique_fraction": float(lv.mean()),
            "tied_fraction": float(((lb <= max_c) & (ls == lb)).mean()),
            "lr_confirmed_among_unique_left": float(agree.sum() / lv.sum()) if lv.any() else 0.0,
            "lr_confirmed_all_samples": float(agree.sum() / ys.size),
        }

    # Slanted-plane subpixel LR interpolation test
    sy, sx = np.indices((64, 160), dtype=np.float32)
    true_disp_left = 12.35 + 0.08 * (sx - 80.0) + 0.03 * (sy - 32.0)
    # Right eye disparity at xr satisfies d_R(x - d_L(x)) = d_L(x)
    # For a linear plane d_L(x) = a + b*x, x_R = x - (a + b*x) = (1-b)*x - a => x = (x_R + a)/(1 - b)
    x_from_xr = (sx + (12.35 - 0.08 * 80.0 + 0.03 * (sy - 32.0))) / (1.0 - 0.08)
    true_disp_right = 12.35 + 0.08 * (x_from_xr - 80.0) + 0.03 * (sy - 32.0)
    valid_mask = np.ones((64, 160), dtype=np.uint32)
    acc_round, res_round = lr_consistency_interpolated(
        true_disp_left, true_disp_right, valid_mask, valid_mask, tolerance=0.05, interpolate=False
    )
    acc_interp, res_interp = lr_consistency_interpolated(
        true_disp_left, true_disp_right, valid_mask, valid_mask, tolerance=0.05, interpolate=True
    )

    result = {
        "schema": "stereoforge.precision_pipeline_audit.v7",
        "fixture": str(fixture_dir),
        "descriptor_comparison": {
            "legacy_24bit_single_pixel": eval_census_variant(c24_l, c24_r, max_bits=24, box_radius=0),
            "multiscale_32bit_eps0_single_pixel": eval_census_variant(c32_l, c32_r, max_bits=32, box_radius=0),
            "multiscale_32bit_eps0_box3x3": eval_census_variant(c32_l, c32_r, max_bits=32, box_radius=1),
            "multiscale_32bit_eps1_single_pixel": eval_census_variant(c32_eps1_l, c32_eps1_r, max_bits=32, box_radius=0),
            "multiscale_32bit_eps1_box3x3": eval_census_variant(c32_eps1_l, c32_eps1_r, max_bits=32, box_radius=1),
        },
        "slanted_plane_lr_interpolation": {
            "nearest_integer_p90_residual_px": float(np.nanpercentile(res_round, 90)),
            "interpolated_p90_residual_px": float(np.nanpercentile(res_interp, 90)),
            "nearest_integer_accept_at_0p05px": float(acc_round.mean()),
            "interpolated_accept_at_0p05px": float(acc_interp.mean()),
        },
    }

    if sequence_dir is not None and sequence_dir.exists():
        # fx * baseline from ZED 2 HD fixture
        fx_b = 527.790283203125 * 0.11944399774074554
        result["sequence_evaluation"] = evaluate_temporal_corroboration_sequence(
            sequence_dir, fx_baseline=fx_b, max_frames=30
        )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--sequence", type=Path, default=None)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--disparities", type=int, default=128)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Output already exists: {args.output}")
    report = run_audit(args.fixture, args.sequence, stride=args.stride, disparities=args.disparities)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
