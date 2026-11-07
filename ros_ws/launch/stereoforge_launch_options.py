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


def resolve_pipeline_options(values):
    """Return a copy; empty launch values select defaults for the chosen pipeline."""
    resolved = dict(values)
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
