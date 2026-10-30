#!/usr/bin/env python3
"""Compare stereo latency and competing GPU throughput from measured JSONL samples.

No invented FPS, GPU utilization or pipeline quality claims. Each record is an
individual measurement with mode, frame_identity, latency_ms, and optional
background_work_units/background_elapsed_ms. Inputs are produced by separate
instrumented workers; this tool only validates and summarizes them.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered)-1)*q
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    return ordered[lo]*(hi-k)+ordered[hi]*(k-lo) if lo != hi else ordered[lo]


def summarize(records):
    by_mode = defaultdict(list)
    for row in records:
        if not isinstance(row, dict) or type(row.get("mode")) is not str or not row["mode"]:
            raise ValueError("each sample requires a nonempty mode")
        frame = row.get("frame_identity")
        if not isinstance(frame, (int, str)) or isinstance(frame, bool) or frame == "":
            raise ValueError("frame_identity required")
        latency = row.get("latency_ms")
        if type(latency) not in (float, int) or not math.isfinite(latency) or latency < 0:
            raise ValueError("finite nonnegative latency_ms required")
        work = row.get("background_work_units")
        elapsed = row.get("background_elapsed_ms")
        if (work is None) != (elapsed is None):
            raise ValueError("background work and elapsed must be specified together")
        if work is not None and (type(work) not in (int,float) or type(elapsed) not in (int,float)
            or not math.isfinite(work) or not math.isfinite(elapsed) or work<0 or elapsed<=0):
            raise ValueError("invalid background sample")
        by_mode[row["mode"]].append(row)
    result = {}
    for mode, rows in sorted(by_mode.items()):
        ids = [str(r["frame_identity"]) for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate frame identity in {mode}")
        t = [r["latency_ms"] for r in rows]
        background=[r for r in rows if "background_work_units" in r]
        total_work=sum(r["background_work_units"] for r in background)
        total_seconds=sum(r["background_elapsed_ms"] for r in background)/1000
        result[mode] = {
            "samples":len(rows),"latency_p50_ms":quantile(t,.5),
            "latency_p95_ms":quantile(t,.95),"latency_p99_ms":quantile(t,.99),
            "latency_max_ms":max(t),
            "background_samples":len(background),
            "background_work_units_per_s":total_work/total_seconds if total_seconds else None,
        }
    return result


def compare(summary, baseline="background_only"):
    baseline_rate=summary.get(baseline,{}).get("background_work_units_per_s")
    out={}
    for name, mode in summary.items():
        rate=mode["background_work_units_per_s"]
        out[name]={**mode,"background_retention":(
            rate/baseline_rate if rate is not None and baseline_rate else None),
            "background_baseline":baseline}
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--background-baseline",default="background_only")
    args=parser.parse_args()
    rows=[]
    with args.samples.open(encoding="utf-8") as handle:
        for i,line in enumerate(handle,1):
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError as error: raise ValueError(f"invalid JSON line {i}") from error
    if not rows: raise ValueError("no measurement samples")
    result={"schema":"stereoforge.gpu_sharing_summary.v1","scope":"sampled host-reported observations; no inferred hardware GPU utilization",
            "modes":compare(summarize(rows),args.background_baseline)}
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
