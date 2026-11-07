# ROS pipeline option binding

The native publisher validates the final option set before creating a trace,
initializing ROS, or opening capture. Legacy refinement `--profile` presets require
`experimental-refinement` or `experimental-dual-refinement` and a compatible
coarse manifest.

These individual controls also require an experimental refinement pipeline,
including when explicitly set to zero. A later `--profile quality` label does
not erase an earlier refinement request on a full-SGM command:


- `--fast-mode`, `--ultra-fast`, `--refinement-subpixel`, `--continuous-subpixel`
- `--adaptive-tiles`, `--min-disparity-bound`, `--max-disparity-bound`
- `--selective-repair`, `--planar-recovery`, `--guided-fill-px`

Enabled guided fill additionally requires `--mode depth-rgb` and
`--planar-recovery 1`, because it consumes both exported RGB and completed depth.
It remains a separate spatial completion layer.

Descriptor mode, SGM schedule and right-reference mode configure the depth
pipeline in both full-SGM and experimental routes. They are not refinement-only
switches. This option-admission contract does not qualify any matching variant
for accuracy or latency promotion.

Both ROS launch files resolve empty values by pipeline. Their default
experimental configuration remains unchanged: ortho-hybrid descriptors, shared
Census scheduling, derived coarse-right selection, enabled fast/ultra/adaptive
refinement, subpixel refinement, selective repair and planar recovery. Continuous
subpixel and disparity bounds default to zero. The publisher launch retains the
native disabled guided-fill default; the standard launch forwards its explicit
zero default.

For `pipeline:=full-sgm`, unset controls remain unset so the native Census,
sequential-schedule, independent-right defaults apply. Explicit common stereo
controls are preserved. Explicit refinement-only launch arguments are rejected
before camera wake or trace cleanup, rather than silently discarded. Empty launch
argument defaults therefore mean “resolve for the selected pipeline”; they do
not change the default experimental recipe.

Keep `stereoforge_launch_options.py` beside both launch files when copying the
source launch entry points. Existing depth/RGB topics and separate completed
output topics are unchanged. Named development recipes are described below;
qualified accuracy/frontier/live promotion remains unfinished.

## Effective estimator identity

Independent SGM identities require explicit `descriptor_mode` (`census` or
`ortho-hybrid`) and `stereo_schedule`. All five native schedules have distinct
identities, including hierarchical fallback. Recording a configuration identity
does not qualify its accuracy or latency. ROS metadata and canonical benchmark
reports obtain these fields from the effective native configuration. Canonical
observation admission checks them against the executed report.

Historical profiles missing either field are rejected for estimator matching;
no cost or schedule is inferred from shader bundle hashes. Original evidence
remains unchanged. Re-export fresh reports or independently audit their
source/configuration before making a new comparison.

Native raw-depth benchmark reports now also include the executed
`stereo_variant`. Both standalone byte-comparison gates require explicit Census
and packed-specialized settings; their fixed oracle must not admit another
matcher or infer missing settings.

Canonical observation exports retain an explicitly reported variant. Old
reports without a variant remain unknown. An experimental overwrite/quad report
with a null canonical identity and its exact unavailable reason can be exported
for quality scoring. That identity stays unavailable: it cannot qualify a
quality/live estimator match. Canonical matching additionally requires explicit
packed-specialized settings on both executed implementations. This preserves
the current whitelist and grants no quality or latency promotion.

## Named operating profiles

Select `--profile quality-reference` or `--profile live-quality` on the native
publisher, or `profile:=quality-reference` / `profile:=live-quality` on the
publisher, standard or camera launch entry point. Empty launch `profile`
preserves the previous custom/experimental defaults. Depth/RGB topics and
full-resolution geometry stay unchanged.

| Recipe | Native grid | Disparities | Matcher | Depth/pending slots | Capture and offered depth rate |
|---|---|---:|---|---:|---:|
| quality-reference | 1280×720 | 256 | Census/4, shared-Census independent references | 2 | 60 Hz |
| quality-reference | 1920×1080 | 288 | Census/4, shared-Census independent references | 1 | 30 Hz |
| live-quality | 1280×720 | 128 | Census/4, sequential independent overwrite-quad | 2 | 60 Hz |

Both use P1=8/P2=32, subpixel stereo, LR tolerance1pixel and current-frame
measured output. Host export and optional cloud retain their existing minimum
2bounded slots, even for the single depth/pending slot at1080p; producer leases
retire before that slot is reused. Metadata distinguishes each capacity.

Names identify **unpromoted development recipes**. Wide search is a reference
baseline, not a claim of improved completeness. Retained D256 near-field and
8-path thin/edge regressions remain rejected; no threshold or default matcher
was promoted. The live recipe has unchanged-output development parity and
historical uncontrolled rate evidence, not controlled live/physical acceptance.
Its fixed60Hz admission request and30Hz minimum consumer policy do not guarantee
30fresh outputs/s. The configured freshness bound is100ms.

Profiles require matching original/rectified native shape and the manifest's
resolution-specific calibration. They never resize inputs or outputs. Geometry
admission checks calibrated hypotheses throughout axial0.5–3m; reference mode
also requires16pixels of near search margin. Unsupported shapes/calibration
ranges reject before capture/GPU construction. This verifies configuration
arithmetic, not factory calibration provenance or physical accuracy.

Named recipes reject explicit estimator, filtering and scheduling overrides,
even matching/disabled values, so their meaning cannot depend on argv order.
Use the legacy/custom route when changing these controls. Explicit
`--pipeline full-sgm` is allowed; refinement routes, repeated/mixed profile
selections and RGB-only mode reject. Output, cloud/debug, topic, run and
validation controls remain available. Launch defaults stay empty until the
shared resolver chooses a custom route, preventing hidden overrides; the camera
wrapper forwards the same unset state.

`implementation.settings.operating_profile` records requested name, recipe
revision, selection status, effective parameters/capacities/rates, original and
rectified grids and calibration identities. Qualification is explicitly
unavailable. Experimental live canonical identity remains unavailable; source
age, rig/clock and uncertainty qualifications remain separate. D288 canonical
configuration identity is accepted only for its supported sequential/shared
schedules. Neither configuration identity nor these names grant promotion.
