#!/usr/bin/env python3
"""Deterministic temporal-component experiments with known synthetic truth.

These are NOT rendered stereo images or measured ZED 2 results. They test
post-estimator temporal behavior, dropout, outliers, change, and stale history.
"""
from __future__ import annotations
import argparse
from dataclasses import replace, asdict
import json
from pathlib import Path
import numpy as np
from temporal_consensus_v2 import ConsensusConfig, TemporalConsensus, corrected_disparity, raw_frame, valid_depth, HELD_HISTORY
from replay_temporal_v2 import Metrics, print_summary, write_json


def synthetic_case(name, *, frames=90, shape=(48,80), seed=20261030):
    rng=np.random.default_rng(seed)
    h,w=shape; yy,xx=np.indices(shape)
    base=(20+4*xx/w).astype(np.float32)
    base[:,w//3:w//3+2]=60. # Thin foreground; spatial mixing is unnecessary.
    texture=rng.integers(30,190,(*shape,3),dtype=np.uint8)
    for i in range(frames):
        truth=base.copy(); rgb=texture.copy()
        keep=.25
        if name=='moving_patch':
            keep=.45
            start=(i*2)%(w-14)
            truth[h//3:2*h//3,start:start+12]=45
            rgb[h//3:2*h//3,start:start+12]=240
        if name=='depth_step_same_rgb' and i>=frames//2:
            truth[:,w//2:]+=12
        observed=truth+rng.normal(0,.22,shape).astype(np.float32)
        if name=='stable_wrong':
            observed=truth+8. # Deliberately stable but geometrically wrong.
            keep=.7
        valid=rng.random(shape)<keep
        if name=='outlier_speckles':
            outlier=valid & (rng.random(shape)<.15)
            observed[outlier]=rng.uniform(2,110,int(outlier.sum()))
        z=np.full(shape,np.nan,np.float32);z[valid]=60./observed[valid]
        rgba=np.concatenate((rgb,np.full((*shape,1),255,np.uint8)),axis=2)
        yield z,rgba,truth


def run(output_dir, frames=90):
    output_dir.mkdir(parents=True,exist_ok=False)
    report={'scope':'Synthetic post-estimator temporal test only; NOT real ZED2 stereo accuracy or speed',
            'seed':20261030,'cases':{}}
    for case in ('static_sparse','outlier_speckles','moving_patch','depth_step_same_rgb','stable_wrong'):
        cfg=ConsensusConfig(global_guard=True)
        filters={k:TemporalConsensus((48,80),60.,replace(cfg,hold_ms=ms))
                 for k,ms in [('confirm',0),('hold100',100),('hold200',200)]}
        metrics={k:Metrics((48,80),60.) for k in ('raw',*filters)}
        truth_stats={k:{'output':0,'sum_error':0.,'bad_gt1px':0,'held':0,'held_bad_gt1px':0} for k in metrics}
        for i,(z,rgb,truth) in enumerate(synthetic_case(case,frames=frames)):
            t=1_000_000_000+i*50_000_000
            results={'raw':raw_frame(z)}
            results.update({k:m.update(z,rgb,receipt_ns=t,source_stamp_ns=t+1_000_000_000) for k,m in filters.items()})
            if i<cfg.window_frames-1:continue
            for k,out in results.items():
                metrics[k].update(out,z,t)
                good=valid_depth(out.depth);q=corrected_disparity(out.depth,60.)
                error=np.abs(q-truth);held=out.provenance==HELD_HISTORY
                s=truth_stats[k];s['output']+=int(good.sum());s['sum_error']+=float(error[good].sum())
                s['bad_gt1px']+=int((good & (error>1)).sum());s['held']+=int(held.sum())
                s['held_bad_gt1px']+=int((held & (error>1)).sum())
        summaries={k:m.finish() for k,m in metrics.items()}
        for k,s in truth_stats.items():
            summaries[k]['synthetic_gt_mae_px']=s['sum_error']/s['output'] if s['output'] else None
            summaries[k]['synthetic_gt_bad_gt1px_fraction']=s['bad_gt1px']/s['output'] if s['output'] else None
            summaries[k]['synthetic_held_gt_bad_gt1px_fraction']=s['held_bad_gt1px']/s['held'] if s['held'] else None
            # Keep the committed synthetic evidence small; full real replay retains pair/frame details.
            summaries[k].pop('frame_metrics');summaries[k].pop('pair_metrics')
        report['cases'][case]={'profiles':summaries}
        print('\nSYNTHETIC:',case)
        print_summary({'profiles':summaries})
        for k,s in summaries.items():
            print(k,'truth_bad>1px=',s['synthetic_gt_bad_gt1px_fraction'], 'truth_MAE=',s['synthetic_gt_mae_px'])
    write_json(output_dir/'synthetic_summary.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--frames',type=int,default=90)
    a=p.parse_args()
    if a.frames<12:p.error('need at least 12 frames')
    run(a.output_dir,a.frames)


if __name__=='__main__':main()
