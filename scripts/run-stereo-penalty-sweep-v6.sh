#!/usr/bin/env bash
# Offline native Daxa penalty sweep, exactly one retained fixture per configuration.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
fixture="${1:-configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json}"
output="${2:-artifacts/depth-video/penalty-v6-$(date +%Y%m%d-%H%M%S-%N)}"
if [[ -e "$output" ]]; then echo "Output exists: $output" >&2; exit 1; fi
mkdir -p "$output"
if [[ ! -f "$fixture" ]]; then echo "Missing fixture: $fixture" >&2; exit 2; fi
binary="build/ros-daxa-native/daxa_depth_benchmark"
if [[ ! -x "$binary" ]]; then
  echo "Missing $binary: build with pixi run -m ros_ws cmake --build build/ros-daxa-native --target daxa_depth_benchmark -j 8" >&2
  exit 2
fi
pixi_lib="$(realpath ros_ws/.pixi/envs/default/lib)"
for pair in 4:16 8:32 12:48 16:56; do
  p1="${pair%:*}"; p2="${pair#*:}"; label="p${p1}_${p2}"
  echo "=== Native P1=$p1 P2=$p2 ==="
  env LD_LIBRARY_PATH="$pixi_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    LD_PRELOAD=/usr/lib64/libdrm.so.2:/usr/lib64/libdrm_amdgpu.so.1 \
    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    "$binary" --manifest "$fixture" \
    --output "$output/$label.json" --dump-prefix "$output/$label" \
    --frames 3 --warmup 1 --slots 2 --wave-size 32 \
    --disparities 128 --paths 4 --p1 "$p1" --p2 "$p2" --lr-tolerance 1
  uv run python benchmarks/lr_confidence_audit_v6.py \
    --fixture "$(dirname "$fixture")" \
    --native-prefix "$output/$label" \
    --disparities 128 --stride 8 \
    --output "$output/$label-confidence.json"
done
printf 'Results: %s\n' "$output"
