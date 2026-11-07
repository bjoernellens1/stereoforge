"""Resolve launch defaults without forwarding controls a pipeline cannot execute."""

REFINEMENT_DEFAULTS = {
    "fast_mode": "1",
    "ultra_fast": "1",
    "refinement_subpixel": "1",
    "continuous_subpixel": "0",
    "adaptive_tiles": "1",
    "min_disparity_bound": "0",
    "max_disparity_bound": "0",
    "selective_repair": "1",
    "planar_recovery": "1",
    "guided_fill_px": "0",
}
EXPERIMENTAL_STEREO_DEFAULTS = {
    "descriptor_mode": "ortho-hybrid",
    "coarse_schedule": "shared-census-sequential",
    "coarse_right_reference_mode": "left-aggregated-selection",
}


CUSTOM_DEFAULTS = {
    "disparities": "128", "stereo_variant": "packed-specialized",
    "capture_fps": "60", "depth_rate": "60",
    "depth_speckle_min_pixels": "0", "depth_speckle_delta_m": "0.05",
    "depth_speckle_delta_px": "1.25",
}
NAMED_CONTROL_KEYS = (set(REFINEMENT_DEFAULTS) | set(EXPERIMENTAL_STEREO_DEFAULTS) |
    set(CUSTOM_DEFAULTS) | {"coarse_manifest", "gpu_duty_budget_pct", "min_depth_fps",
    "max_depth_fps", "consumer_min_depth_fps", "max_depth_age_ms", "downshift_ms"})


def resolve_pipeline_options(values):
    """Return a copy; empty launch values select defaults for the chosen pipeline."""
    resolved = dict(values)
    profile = resolved.get("profile", "")
    if profile:
        if profile not in ("quality-reference", "live-quality"):
            raise RuntimeError("unknown named profile: " + profile)
        supplied = [key for key in sorted(NAMED_CONTROL_KEYS) if resolved.get(key, "") != ""]
        if supplied or resolved.get("pipeline", "") not in ("", "full-sgm") or resolved.get("mode") == "rgb":
            raise RuntimeError("named profile fixes estimator and scheduling controls; remove overrides or RGB-only mode")
        resolved["pipeline"] = "full-sgm"
        return resolved
    if not resolved.get("pipeline"):
        resolved["pipeline"] = "experimental-dual-refinement"
    for key, default in CUSTOM_DEFAULTS.items():
        if key in resolved and not resolved[key]:
            resolved[key] = default
    pipeline = resolved["pipeline"]
    if pipeline == "full-sgm":
        supplied = [name for name in REFINEMENT_DEFAULTS if resolved.get(name, "") != ""]
        if supplied:
            raise RuntimeError(
                "refinement-only options require an experimental refinement pipeline: "
                + ", ".join(supplied)
            )
        return resolved
    if pipeline not in ("experimental-refinement", "experimental-dual-refinement"):
        raise RuntimeError("unknown publisher pipeline: " + pipeline)
    for defaults in (EXPERIMENTAL_STEREO_DEFAULTS, REFINEMENT_DEFAULTS):
        for name, default in defaults.items():
            if name in resolved and not resolved[name]:
                resolved[name] = default
    if int(resolved.get("guided_fill_px", "0")) > 0 and (
        resolved["mode"] != "depth-rgb" or int(resolved["planar_recovery"]) != 1
    ):
        raise RuntimeError("guided fill requires depth-rgb and planar recovery")
    return resolved
