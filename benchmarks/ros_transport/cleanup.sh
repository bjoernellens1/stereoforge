# Source from a Bash process; benchmark children are started in distinct sessions.
sub_group='' router_group='' pub_group=''
cleanup_transport_children() {
  local group
  for group in "$pub_group" "$sub_group" "$router_group"; do
    if [[ $group =~ ^[1-9][0-9]*$ ]]; then kill -TERM -- "-$group" 2>/dev/null || true; fi
  done
  # Bound cleanup even if a middleware process ignores termination.
  if [[ -n $pub_group || -n $sub_group || -n $router_group ]]; then sleep 0.2; fi
  for group in "$pub_group" "$sub_group" "$router_group"; do
    if [[ $group =~ ^[1-9][0-9]*$ ]]; then
      kill -KILL -- "-$group" 2>/dev/null || true
      wait "$group" 2>/dev/null || true
    fi
  done
}
trap cleanup_transport_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
