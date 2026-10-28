#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/cleanup.sh"
# Inside a ROS Jazzy container with rmw_zenoh_cpp installed, source ROS before running.
: "${OUT:?set OUT to an empty writable output directory}"
mkdir -p "$OUT"
seconds=${SECONDS_PER_ARM:-60}; warmup=${WARMUP_SECONDS:-3}
export ROS_DOMAIN_ID=218
index=0
for arm in ${ARM_ORDER:-fast zenoh zenoh-shm}; do
  case "$arm" in fast|zenoh|zenoh-shm) ;; *) echo "invalid arm: $arm" >&2; exit 2;; esac
  index=$((index+1)); tag=$(printf '%03d-%s' "$index" "$arm")
  printf '%s,%s\n' "$tag" "$arm" >> "$OUT/arm-order.csv"
  router=''
  unset ZENOH_ROUTER_CONFIG_URI ZENOH_SESSION_CONFIG_URI || true
  if [[ $arm == fast ]]; then export RMW_IMPLEMENTATION=rmw_fastrtps_cpp; else
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp
    export ZENOH_ROUTER_CONFIG_URI="$OUT/router-$tag.json5" ZENOH_SESSION_CONFIG_URI="$OUT/session-$tag.json5"
    /usr/bin/python3 - "$OUT" "$arm" "$tag" <<'PY'
import sys
from pathlib import Path
out, arm, tag=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
for name in ('ROUTER','SESSION'):
 s=Path('/opt/ros/jazzy/share/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_'+name+'_CONFIG.json5').read_text()
 if name=='ROUTER': s=s.replace('"tcp/[::]:7447"','"tcp/127.0.0.1:7447"')
 if arm=='zenoh-shm':
  assert '/// ROS setting: disabled by default until fully tested\n      enabled: false,' in s
  s=s.replace('/// ROS setting: disabled by default until fully tested\n      enabled: false,','enabled: true,').replace('pool_size: 50331648','pool_size: 6291456')
 (out/f'{name.lower()}-{tag}.json5').write_text(s)
PY
    if [[ $arm == zenoh-shm && ${SHM_TRACE_DIAGNOSTIC:-0} == 1 ]]; then export RUST_LOG=info,zenoh_transport::shm=trace,zenoh_shm=trace; else export RUST_LOG=off; fi
    setsid /usr/bin/time -v -o "$OUT/$tag-router-time.txt" bash -c 'echo $$ > "$1"; exec /opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd' _ "$OUT/$tag-router.pid" >"$OUT/$tag-router.log" 2>&1 & router_wait=$!; router_group=$router_wait
    for _ in {1..100}; do [[ -s "$OUT/$tag-router.pid" ]] && break; sleep 0.01; done
    router=$(cat "$OUT/$tag-router.pid")
    sleep 2
  fi
  setsid /usr/bin/time -v -o "$OUT/$tag-sub-time.txt" "$OUT/transport" sub "$OUT/$tag-sub.csv" "$seconds" "$warmup" >"$OUT/$tag-sub.log" 2>&1 & sub=$!; sub_group=$sub
  sleep 2
  setsid /usr/bin/time -v -o "$OUT/$tag-pub-time.txt" "$OUT/transport" pub "$OUT/$tag-pub.csv" "$seconds" "$warmup" >"$OUT/$tag-pub.log" 2>&1 & pub_group=$!
  wait "$pub_group"
  pub_group=''
  wait "$sub"
  sub_group=''
  if [[ -n "$router" ]]; then kill "$router"; wait "$router_wait" 2>/dev/null || true; router_group=''; fi
  echo "$tag complete"
done
