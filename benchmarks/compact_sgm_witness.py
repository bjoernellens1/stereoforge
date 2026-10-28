"""Offline bounded scanline reconstruction for compact SGM failure witnesses.

No ground truth is used by the recurrence. It reconstructs a single target
pixel from four complete predecessor lines and uses O(D) working memory.
"""
from pathlib import Path
import numpy as np
from benchmarks.compact_sgm_reference import DIRECTIONS


def provenance_dependencies():
    root = Path(__file__).resolve().parents[1]
    return (root / 'benchmarks/compact_sgm_reference.py',)


def trace_witness(shape, lo, hi, y, x, cost_at, p1=8, p2=32):
    h, w, disparities = shape
    lo, hi = np.asarray(lo), np.asarray(hi)
    if (min(shape) <= 0 or lo.shape != (h, w) or hi.shape != (h, w)
            or not np.issubdtype(lo.dtype, np.integer) or not np.issubdtype(hi.dtype, np.integer)
            or np.any(lo < 0) or np.any(hi > disparities) or np.any(lo > hi)
            or not 0 <= y < h or not 0 <= x < w
            or not np.isfinite([p1, p2]).all() or not 0 <= p1 <= p2 <= 10000
            or p1 != int(p1) or p2 != int(p2)):
        raise ValueError('invalid witness geometry, intervals or penalties')
    paths = np.full((4, disparities), np.inf)
    for k, (dx, dy) in enumerate(DIRECTIONS):
        sx, sy = x, y
        while 0 <= sx - dx < w and 0 <= sy - dy < h:
            sx, sy = sx - dx, sy - dy
        previous = {}
        while True:
            minimum = min(previous.values(), default=None)
            current = {}
            for d in range(int(lo[sy, sx]), int(hi[sy, sx])):
                c = cost_at(sy, sx, d)
                if c == np.inf:
                    continue
                if not np.isfinite(c) or c < 0 or c > 10000 or c != int(c):
                    raise ValueError('cost must be an integer in [0, 10000] or +inf')
                if minimum is None:
                    value = c
                else:
                    value = c + min(previous.get(d, np.inf), previous.get(d - 1, np.inf) + p1,
                                    previous.get(d + 1, np.inf) + p1, minimum + p2) - minimum
                current[d] = value
            if (sy, sx) == (y, x):
                for d, value in current.items():
                    paths[k, d] = value
                break
            previous = current
            sx, sy = sx + dx, sy + dy
    sums = paths.sum(axis=0)
    return {'paths': paths, 'sums': sums,
            'winner': int(np.argmin(sums)) if np.isfinite(sums).any() else -1}


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import compare_daxa_refinement as cmp
    from stereoforge.refinement import census

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family-root', required=True)
    parser.add_argument('--frozen', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, frozen_path, selection_path = map(Path, (args.family_root, args.frozen, args.selection))
    paths = [root / 'half/manifest.json', root / 'full/manifest.json', root / 'family.json',
             frozen_path, selection_path, Path(args.result), Path(__file__),
             Path('benchmarks/compare_daxa_refinement.py'), Path('python/stereoforge/refinement.py'),
             *provenance_dependencies()]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(p): sha(p) for p in paths}
    report = json.loads(Path(args.result).read_text())
    if (report['hashes']['frozen_intervals'] != hashes[str(frozen_path)]
            or report['hashes']['frozen_selection'] != hashes[str(selection_path)]):
        raise ValueError('frozen files differ from experiment result')
    ctx = cmp.load_context(paths[0], paths[1])
    with np.load(frozen_path, allow_pickle=False) as f:
        lo, hi = f['lo'].copy(), f['hi'].copy()
        config = json.loads(str(f['config']))
        recorded = json.loads(str(f['input_hashes']))
    if (recorded['coarse'] != hashes[str(paths[0])] or recorded['target'] != hashes[str(paths[1])]
            or recorded['family'] != hashes[str(paths[2])] or config['disparities'] != 128):
        raise ValueError('frozen inputs/configuration mismatch')
    with np.load(selection_path, allow_pickle=False) as f:
        selected = {key: f[key].copy() for key in f.files}
    descriptions = [census(ctx['images'][i], ctx['support'][i]) for i in (0, 1)]
    truth = cmp.load_truth(paths[2], ctx)
    results = []
    h, w = ctx['images'].shape[1:]
    for eye, name in enumerate(('left', 'right')):
        t = truth[name]
        disparity = t['disparity']
        valid = np.asarray(selected['raw_validity.u32'][eye], bool)
        error = np.abs(selected['raw_disparity.f32'][eye] - disparity)
        mask = (np.asarray(t['valid'], bool) & np.asarray(t['visible'], bool)
                & ~np.asarray(t['out_of_range'], bool) & ~np.asarray(t['off_image'], bool)
                & np.asarray(t['remap_support'], bool) & descriptions[eye].support
                & ~np.asarray(t['mixed_bilinear'], bool) & np.asarray(t['region_thin'], bool)
                & valid & (error > 1) & np.isfinite(disparity))
        # Pick deterministic first witnesses with at least one supported candidate within 1px of truth.
        positions = np.argwhere(mask)
        own, other = descriptions[eye], descriptions[1-eye]
        def cost(y, x, d):
            ox = x + d if eye else x - d
            if not own.support[y, x] or not 0 <= ox < w or not other.support[y, ox]:
                return np.inf
            return (int(own.descriptor[y, x]) ^ int(other.descriptor[y, ox])).bit_count()
        found = 0
        for y, x in positions:
            truth_candidates = [d for d in range(int(lo[eye,y,x]), int(hi[eye,y,x]))
                                if abs(d - float(disparity[y,x])) <= 1 and np.isfinite(cost(y,x,d))]
            if not truth_candidates:
                continue
            result = trace_witness((h,w,128), lo[eye], hi[eye], int(y), int(x), cost,
                                   config['p1'], config['p2'])
            winner = result['winner']
            if winner != int(selected['raw_disparity.f32'][eye,y,x]):
                raise ValueError('reconstructed winner disagrees with frozen selection')
            truth_d = min(truth_candidates, key=lambda d: (abs(d - float(disparity[y,x])), d))
            def candidate(d):
                return {'disparity': d, 'local_census': int(cost(y,x,d)),
                        'directional_scores': [int(v) for v in result['paths'][:,d]],
                        'aggregate': int(result['sums'][d])}
            results.append({'eye': name, 'x': int(x), 'y': int(y), 'truth_disparity': float(disparity[y,x]),
                            'interval': [int(lo[eye,y,x]), int(hi[eye,y,x])],
                            'winner': candidate(winner), 'truth_nearest_supported': candidate(truth_d),
                            'all_truth_near_supported': [candidate(d) for d in truth_candidates],
                            'lr_accepted': bool(selected['validity.u32'][eye,y,x])})
            found += 1
            if found == 2:
                break
        if found != 2:
            raise ValueError('insufficient eligible witnesses')
    if any(sha(p) != hashes[str(p)] for p in paths):
        raise ValueError('input changed during analysis')
    with open(args.output, 'x') as f:
        json.dump({'scope': 'post-hoc selected synthetic development witnesses; no performance or generalization claim',
                   'hashes': hashes, 'configuration': config, 'witnesses': results}, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
