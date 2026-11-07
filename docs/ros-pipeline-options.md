# ROS pipeline option binding

The native publisher validates the final option set before creating a trace,
initializing ROS, or opening capture. Non-default `--profile` presets require
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
output topics are unchanged. The requested qualified `quality-reference` and
`live-quality` profiles remain a separate unfinished requirement.

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
