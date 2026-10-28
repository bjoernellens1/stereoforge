"""Summarize separate-process synthetic transport traces, without claiming live-camera FPS."""
import argparse
import csv
import json
import math
from pathlib import Path


def percentile(values, p):
    if not values:
        return None
    v = sorted(values)
    return v[math.ceil((len(v) - 1) * p)]


def summarize(publisher, subscriber):
    with Path(publisher).open(newline='') as f:
        pub = list(csv.DictReader(f))
    with Path(subscriber).open(newline='') as f:
        sub = list(csv.DictReader(f))
    if not pub:
        raise ValueError('empty publisher trace')
    ids = [int(r['id']) for r in pub]
    if ids != list(range(1, len(ids)+1)):
        raise ValueError('publisher IDs must be contiguous and ordered')
    offered = {int(r['id']): r for r in pub if int(r['begin_ns']) > 0}
    if any(r['kind'] not in ('d', 'r') for r in sub):
        raise ValueError('unknown subscriber output')
    result = {'scheduled': len(pub), 'offered': len(offered), 'pacing_skips': len(pub)-len(offered), 'outputs': {}}
    starts = [int(r['begin_ns']) for r in pub if int(r['begin_ns']) > 0]
    result['offered_mean_interval_ms'] = (starts[-1]-starts[0])/1e6/(len(starts)-1) if len(starts)>1 else None
    for name, kind in [('depth','d'),('rgb','r')]:
        rows = [r for r in sub if r['kind'] == kind]
        received = [int(r['id']) for r in rows]
        unmatched = [i for i in received if i not in offered]
        if unmatched:
            raise ValueError('subscriber has IDs not offered by publisher')
        unique = set(received)
        call = [int(r[name+'_wall_ns'])/1e6 for r in offered.values()]
        thread_cpu = [int(r[name+'_cpu_ns'])/1e6 for r in offered.values()]
        latencies = [(int(r['arrival_ns']) - int(offered[int(r['id'])]['begin_ns']))/1e6 for r in rows]
        if any(x < 0 for x in latencies):
            raise ValueError('arrival precedes publisher start; check clock domains')
        result['outputs'][name] = {'unique_received': len(unique), 'duplicates': len(received)-len(unique),
            'offered_not_received': len(set(offered)-unique), 'publish_wall_p50_ms': percentile(call,.5),
            'publish_wall_p99_ms': percentile(call,.99), 'publish_thread_cpu_p50_ms': percentile(thread_cpu,.5),
            'publish_thread_cpu_p99_ms': percentile(thread_cpu,.99), 'arrival_from_pair_begin_p99_ms': percentile(latencies,.99)}
    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('publisher');parser.add_argument('subscriber');args=parser.parse_args()
    print(json.dumps(summarize(args.publisher,args.subscriber),indent=2,sort_keys=True))
