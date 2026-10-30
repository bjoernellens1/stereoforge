#!/usr/bin/env python3
"""Deterministic occlusion-aware rectified stereo scenes and anchor evaluation.

Left-view disparity is ground truth only when the corresponding surface wins the
right-view z-buffer; disoccluded right pixels receive unrelated background.
No truth is supplied to the matcher.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from fast60_anchor_evaluate import evaluate
from fast60_anchor_hypotheses import AnchorConfig


def make_scene(kind, *, height=180, width=320, seed=17):
    if kind not in ('thin', 'occlusion', 'low_texture', 'repetitive', 'slanted'):
        raise ValueError('unknown scene')
    if width < 96 or height < 64:
        raise ValueError('scene too small')
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width))
    d = np.full((height, width), 8, dtype=np.int16)
    risk = np.zeros((height, width), dtype=bool)
    if kind == 'thin':
        # 2--5 px objects deliberately vanish under aggressive downsampling.
        for x, thickness in [(width//3, 2), (width//2, 3), (2*width//3, 5)]:
            region = (xx >= x) & (xx < x+thickness) & (yy >= height//6) & (yy < 5*height//6)
            d[region] = 26
            risk |= region
    elif kind == 'occlusion':
        region = ((xx-width//2)**2 / (width/6)**2 + (yy-height//2)**2 / (height/3)**2) <= 1
        d[region] = 25
        risk |= region
    elif kind == 'slanted':
        d = (6 + (xx * 20 // width)).astype(np.int16)
        risk |= (xx > width//4) & (xx < 3*width//4)
    elif kind == 'low_texture':
        risk |= (xx > width//4) & (xx < 3*width//4)
    elif kind == 'repetitive':
        risk |= (xx > width//4) & (xx < 3*width//4)

    if kind == 'low_texture':
        img = np.clip(105 + yy/12 + xx/32, 0, 255).astype(np.uint8)
    elif kind == 'repetitive':
        img = (55 + 175 * ((xx//8) % 2)).astype(np.uint8)
    else:
        img = rng.integers(20, 236, size=(height, width), dtype=np.uint8)
        if kind in ('thin', 'occlusion'):
            img[d > 8] = rng.integers(30, 226, size=int((d > 8).sum()), dtype=np.uint8)

    # Forward project each left pixel into the right view. Higher disparity is
    # closer (constant calibrated f*b), therefore wins right-view visibility.
    right = rng.integers(35, 220, size=(height, width), dtype=np.uint8)
    winner = np.full((height, width), -1, dtype=np.int32)
    right_depth = np.full((height, width), -1, dtype=np.int16)
    for y in range(height):
        for x in range(width):
            xr = x - int(d[y, x])
            if xr < 0:
                continue
            if d[y, x] > right_depth[y, xr]:
                right_depth[y, xr] = d[y, x]
                winner[y, xr] = x
                right[y, xr] = img[y, x]
    truth = np.full((height, width), np.nan, dtype=np.float32)
    for y in range(height):
        for xr in range(width):
            x = winner[y, xr]
            if x >= 0:
                truth[y, x] = d[y, x]

    coarse = truth.copy()
    if kind in ('thin', 'occlusion'):
        # Simulate a coarse estimator missing all foreground support.
        coarse[risk] = 8
    elif kind == 'slanted':
        coarse[risk] = np.round(coarse[risk]/8)*8
    elif kind in ('low_texture', 'repetitive'):
        coarse[risk] = np.nan
    return dict(left=img, right=right, truth_disparity=truth,
                coarse_disparity=coarse, risk_mask=risk)


def run(out, *, height=180, width=320, seed=17, disparities=32, stride=4):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name in ('thin','occlusion','low_texture','repetitive','slanted'):
        scene = make_scene(name, height=height, width=width, seed=seed)
        np.savez_compressed(out / f'{name}.npz', **scene)
        result = evaluate(scene['left'],scene['right'],scene['truth_disparity'],
                          coarse=scene['coarse_disparity'],risk=scene['risk_mask'],
                          config=AnchorConfig(disparities=disparities,stride=stride))
        (out/f'{name}.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        results.append(dict(scene=name, selected=result['selected_anchor_count'],
            correct=result['correct_anchor_count'],incorrect=result['incorrect_anchor_count'],
            recovered=result['coarse_comparison']['anchors_correct_where_coarse_incorrect_or_missing'],
            risk_coverage=result['risk']['correct_risk_anchor_coverage'],
            evaluations=result['evaluated_candidate_count']))
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('scene          selected correct wrong recovered risk_recall candidates')
    for r in results:
        v=r['risk_coverage']
        print(f"{r['scene']:<14} {r['selected']:>7} {r['correct']:>7} {r['incorrect']:>5} {r['recovered']:>9} {('n/a' if v is None else f'{v:.3f}'):>11} {r['evaluations']:>10}")
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/anchor-scene-suite'))
    p.add_argument('--height',type=int,default=180)
    p.add_argument('--width',type=int,default=320)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--disparities',type=int,default=32)
    p.add_argument('--stride',type=int,default=4)
    a=p.parse_args()
    run(a.output_dir,height=a.height,width=a.width,seed=a.seed,
        disparities=a.disparities,stride=a.stride)


if __name__=='__main__':
    main()
