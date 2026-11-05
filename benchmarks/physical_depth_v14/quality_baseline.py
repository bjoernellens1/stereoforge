#!/usr/bin/env python3
"""Reproducible development-only coverage diagnosis, without cross-session rig reuse.

python -m benchmarks.physical_depth_v14.quality_baseline --corpus ~/stereoforge_corpus \
    --scenes S1_target_1.0m S2_white_wall --indices 0 74 149 --out /tmp/quality-baseline
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from benchmarks.physical_depth_v14 import reference_sgm as R
from benchmarks.physical_depth_v14.quality_contract import TARGETS, failure_accounting, require_equivalent
from stereoforge.epipolar_audit import audit_epipolar

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / 'configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json'
REASON_BITS = {'no_prior': 1, 'low_prior_confidence': 2, 'prior_span': 4,
               'nonfinite_prior': 8, 'outside_prior': 16, 'unsupported': 32,
               'too_few_candidates': 64, 'ambiguous': 128, 'restricted_minimum': 256,
               'global_boundary': 512, 'missing_subpixel': 1024, 'lr_failed': 2048, 'full_search': 4096,
               'search_unconfirmed': 8192, 'photometric_rejected': 16384}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_scene(scene):
    metadata = json.loads((scene / 'scene.json').read_text())
    required = {'native_depth': 'rs_native_depth.npy', 'factory_geometry': 'rs_factory.json',
                'device_timestamps': 'rs_timing.json', 'ir': 'rs_ir_left.npy',
                'survey': 'survey.json', 'clock_qualification': 'clock_qualification.json'}
    missing = [key for key, filename in required.items() if not (scene / filename).is_file()]
    if not metadata.get('rig_session'):
        missing.append('rig_session')
    # Presence is inventory, never validation of independently known support.
    return dict(rig_session=metadata.get('rig_session'), split=metadata.get('split', 'development'),
                absolute_reference_qualified=False, dynamic_reference_available=False,
                missing_reference_evidence=missing, tape_m_unqualified=metadata.get('tape_m'),
                legacy_aligned_depth=(scene / 'rs_depth.npy').exists(),
                reference_status='unqualified; independent geometry and clock validation required')


def load_geometry(manifest):
    document = json.loads(manifest.read_text())
    width, height = document['rectified']['width'], document['rectified']['height']
    identities = {str(manifest): sha256(manifest)}
    for key in ('original_json', 'rectified_json'):
        desc = document['calibration'][key]
        path = manifest.parent / desc['file']
        actual = sha256(path)
        if actual != desc['sha256'] or actual != document['calibration'][key.replace('_json', '_identity')]:
            raise ValueError('calibration file/identity mismatch')
        identities[str(path)] = actual

    maps = []
    for key in ('left_x', 'left_y', 'right_x', 'right_y'):
        item = document['maps'][key]
        path = manifest.parent / item['file']
        actual = sha256(path)
        if 'sha256' in item and item['sha256'] != actual:
            raise ValueError(f'calibration map hash mismatch: {path}')
        identities[str(path)] = actual
        maps.append(np.fromfile(path, '<f4').reshape(height, width))
    p1 = np.asarray(document['calibration']['P1']).reshape(3, 4)
    p2 = np.asarray(document['calibration']['P2']).reshape(3, 4)
    metric = dict(fx=float(p1[0, 0]), baseline=float(-p2[0, 3] / p2[0, 0]),
                  principal_offset=float(p1[0, 2] - p2[0, 2]))
    if not np.isfinite(list(metric.values())).all() or metric['fx'] <= 0 or metric['baseline'] <= 0:
        raise ValueError('invalid metric calibration')
    return document, maps, identities, metric


def validate_coarse_geometry(target, coarse):
    if target['source'] != coarse['source'] or target['calibration']['original_identity'] != coarse['calibration']['original_identity']:
        raise ValueError('coarse and target original calibration differ')
    if any(target['rectified'][key] != 4 * coarse['rectified'][key] for key in ('width', 'height')):
        raise ValueError('CPU baseline requires exact 4x coarse geometry')
    a = np.array([[.25, 0, -.375], [0, .25, -.375], [0, 0, 1.]])
    for key in ('R1', 'R2'):
        if not np.array_equal(target['calibration'][key], coarse['calibration'][key]):
            raise ValueError('coarse and target rectification rotations differ')
    for key in ('P1', 'P2'):
        if not np.allclose(a @ np.asarray(target['calibration'][key]), coarse['calibration'][key], rtol=1e-10, atol=1e-10):
            raise ValueError('coarse projection violates half-pixel calibration association')


def metric_depth(disparity, metric):
    denominator = disparity - metric['principal_offset']
    valid = np.isfinite(denominator) & (denominator > 0)
    depth = np.full(disparity.shape, np.nan, np.float32)
    np.divide(metric['fx'] * metric['baseline'], denominator, out=depth, where=valid)
    valid &= np.isfinite(depth) & (depth > 0)
    depth[~valid] = np.nan
    return depth, valid


def rectified_pair(raw, maps):
    height, width = maps[0].shape
    if raw.shape != (height, width * 2, 2) or raw.dtype != np.uint8:
        raise ValueError('recording must use native resolution matching factory calibration')
    rgb = cv2.cvtColor(raw, cv2.COLOR_YUV2RGB_YUYV)
    images = [cv2.remap(rgb[:, eye * width:(eye + 1) * width], *maps[2 * eye:2 * eye + 2],
                        cv2.INTER_LINEAR) for eye in (0, 1)]
    return tuple(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images)


def run_observed_cpu(backend, **configuration):
    """Time the actual path; admit reason writes only after byte-exact equality."""
    if configuration.get('borrow_workspace', False):
        raise ValueError('equivalence requires owned output snapshots, not borrowed workspace')
    configuration['diagnostic_writes'] = False
    start = time.perf_counter()
    actual = backend.run_full_camera_to_depth_pipeline(**configuration)
    elapsed = time.perf_counter() - start
    if not configuration.get('fast_mode', True):
        return actual, elapsed, 0., False
    configuration['diagnostic_writes'] = True
    start = time.perf_counter()
    diagnostic = backend.run_full_camera_to_depth_pipeline(**configuration)
    diagnostic_elapsed = time.perf_counter() - start
    require_equivalent(actual, {key: diagnostic[key] for key in actual})
    return diagnostic, elapsed, diagnostic_elapsed, True


def run(args):
    document, maps, identities, metric = load_geometry(args.manifest)
    h, w = maps[0].shape
    if (document['source']['eye_width'], document['source']['height']) != (w, h):
        raise ValueError('reference baseline requires native-resolution rectification')
    source_files = subprocess.check_output(['git', 'ls-files', '*.py', '*.slang', '*.hpp', '*.cpp'], cwd=ROOT, text=True).splitlines()
    # Bind dirty source bytes as well as HEAD; HEAD alone is insufficient.
    source_files.extend(str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob('quality_*.py') if str(p.relative_to(ROOT)) not in source_files)
    sources = {name: sha256(ROOT / name) for name in sorted(source_files)}
    variants = list(product(args.disparities, getattr(args, 'reference_costs', ['normalized']),
                            getattr(args, 'reference_paths', [8]), getattr(args, 'reference_subpixel', ['equiangular'])))
    lr_configuration = dict(lr_sampling=getattr(args, 'reference_lr_sampling', 'nearest'),
                            lr_tol=getattr(args, 'reference_lr_tolerance', 1.),
                            lr_max_jump=getattr(args, 'reference_lr_max_jump', 1.))
    report = dict(schema='stereoforge.quality_baseline.v1', targets=TARGETS,
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  source_files_sha256=sources, calibration_sha256=identities,
                  runtime=dict(numpy=np.__version__, opencv=cv2.__version__),
                  status='incomplete', requested_scenes=args.scenes, requested_indices=args.indices,
                  disparity_ranges=args.disparities, cpu_requested=args.cpu,
                  reference_variants=[dict(disparities=d, matching_cost=c, paths=n, subpixel=s) for d, c, n, s in variants],
                  reference_lr_configuration=lr_configuration,
                  scope='recorded development data; CPU execution; no absolute or live performance qualification',
                  promotion='withheld: independently surveyed support, session and clock qualification absent',
                  absolute_correct_completeness=None, native_fast_diagnostic_equivalence='not_executed_cpu_baseline_only', scenes={})
    backend = None
    if args.cpu:
        from stereoforge.slang_cpu_backend import SlangCpuRefinementBackend
        backend = SlangCpuRefinementBackend()
        report['cpu_library_sha256'] = sha256(backend.so_path)
        coarse_doc, coarse_maps, coarse_ids, _ = load_geometry(args.coarse_manifest)
        validate_coarse_geometry(document, coarse_doc)
        identities.update(coarse_ids)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name in args.scenes:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('scene must be a single directory name')
        scene = args.corpus / name
        raw_path = scene / 'zed_raw.npy'
        frames = np.load(raw_path, mmap_mode='r', allow_pickle=False)
        if not args.indices or any(i < 0 or i >= len(frames) for i in args.indices):
            raise ValueError(f'frame indices out of range for {name}')
        metadata = json.loads((scene / 'scene.json').read_text())
        if metadata.get('split', 'development') != 'development':
            raise ValueError('diagnosis/tuning runner accepts development recordings only')
        captured_calibration = metadata.get('zed_calibration_sha256')
        if captured_calibration and captured_calibration != document['calibration']['original_identity']:
            raise ValueError('recording factory calibration does not match experiment')
        entry = dict(audit=audit_scene(scene), recording_calibration_association='hash_verified' if captured_calibration else 'unverified_legacy',
                     recording_sha256=sha256(raw_path),
                     scene_metadata_sha256=sha256(scene / 'scene.json'), frames=[])
        for index in args.indices:
            raw = np.ascontiguousarray(frames[index])
            pair = rectified_pair(raw, maps)
            epipolar = audit_epipolar(*pair)
            row = dict(index=index, raw_frame_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                       rectification=dict(matches=epipolar.matches, median_dy_px=float(epipolar.median_dy_px) if np.isfinite(epipolar.median_dy_px) else None,
                                          robust_rms_dy_px=float(epipolar.robust_rms_dy_px) if np.isfinite(epipolar.robust_rms_dy_px) else None), modes={})
            for disparities, cost, paths, fit in variants:
                params = R.Params(disparities=disparities, matching_cost=cost, paths=paths, subpixel=fit, **lr_configuration)
                start = time.perf_counter()
                result = R.estimate(*pair, params)
                depth, metric_valid = metric_depth(result['disparity'], metric)
                valid = result['valid'] & metric_valid
                masks = dict(support_failure=~result['support'], out_of_image=result['out_of_image'], range_boundary=result['range_boundary'],
                             ambiguous=result['ambiguous'], lr_discontinuity=result['lr_discontinuity'],
                             lr_support_failure=result['lr_support_failure'], lr_failed=result['lr_diff'] > params.lr_tol,
                             metric_rejected=result['valid'] & ~metric_valid)
                key = f'independent_reference_d{disparities}'
                if (cost, paths, fit) != ('normalized', 8, 'equiangular'):
                    key += f'_{cost}_{paths}path_{fit}'
                if (params.lr_sampling, params.lr_tol, params.lr_max_jump) != ('nearest', 1., 1.):
                    key += f'_lr{params.lr_sampling}_t{params.lr_tol:g}_j{params.lr_max_jump:g}'
                summary, bits, first = failure_accounting(masks)
                summary.update(params=asdict(params), measured_pixels=int(valid.sum()),
                               measured_coverage_pct=float(valid.mean() * 100), cpu_wall_s=time.perf_counter() - start,
                               correct_completeness=None, unobserved_reasons=['rectification_support', 'absolute_correctness'])
                artifact = args.out / f'{name}-{index}-{key}.npz'
                np.savez_compressed(artifact, depth=np.where(valid, depth, np.nan), disparity=result['disparity'],
                                    right_disparity=result['right_disparity'], right_validity=result['right_validity'],
                                    lr_discontinuity=result['lr_discontinuity'], validity=valid,
                                    reason_mask=bits, first_failure=first)
                summary['artifact'] = dict(file=artifact.name, sha256=sha256(artifact))
                row['modes'][key] = summary
            if backend is not None:
                for key, mode in [('live_cpu', dict(adaptive_tiles=True, fast_mode=True, ultra_fast=True,
                                                   continuous_subpixel=False, planar_recovery=True)),
                                  ('full_cpu', dict(adaptive_tiles=False, fast_mode=False, ultra_fast=False,
                                                   continuous_subpixel=True, planar_recovery=False))]:
                    r, cpu_wall_s, diagnostic_wall_s, equivalence_checked = run_observed_cpu(backend,
                        raw_yuyv=raw.ravel(), target_maps=tuple(a.ravel() for a in maps),
                        coarse_maps=tuple(a.ravel() for a in coarse_maps), eye_width=w, source_height=h,
                        stride_bytes=w * 4, width=w, height=h, coarse_width=w // 4, coarse_height=h // 4,
                        **metric, disparities=128, radius=2, subpixel=True, ortho_hybrid=True, selective_repair=True, **mode)
                    plane = lambda field: r[field][:w*h].reshape(h, w)
                    measured = (plane('metric_validity') != 0) & (plane('provenance') == 1) & np.isfinite(plane('depth'))
                    masks = dict(support_failure=plane('census_validity') == 0)
                    masks.update(raw_match_rejected=plane('raw_validity') == 0,
                                 lr_rejected=(plane('raw_validity') != 0) & (plane('validity') == 0),
                                 metric_rejected=(plane('validity') != 0) & ~measured)
                    summary, bits, first = failure_accounting(masks)
                    # Shader reason flags include recovered prior failures and
                    # subpixel warnings even on accepted pixels. They are not
                    # a causal first-failure partition of the final output.
                    if 'reason' in r:
                        summary['shader_reason_counts'] = {reason: int(np.count_nonzero(plane('reason') & bit))
                                                           for reason, bit in REASON_BITS.items()}
                        summary['shader_reason_bits'] = REASON_BITS
                    summary.update(measured_pixels=int(measured.sum()), measured_coverage_pct=float(measured.mean() * 100),
                                   cpu_wall_s=cpu_wall_s, diagnostic_wall_s=diagnostic_wall_s,
                                   diagnostic_equivalence_checked=equivalence_checked,
                                   configuration=mode, correct_completeness=None,
                                   spatial_pixels=int(np.count_nonzero(plane('completed_validity') & (plane('completed_provenance') == 6))),
                                   detailed_reasons_available='reason' in r,
                                   unobserved_reasons=['correct_disparity_excluded', 'absolute_correctness', 'post_filter'])
                    artifact = args.out / f'{name}-{index}-{key}.npz'
                    np.savez_compressed(artifact, depth=plane('depth'), disparity=plane('disparity'),
                                        validity=measured, reason_mask=bits, first_failure=first,
                                        provenance=plane('provenance'), completed_depth=plane('completed_depth'),
                                        completed_provenance=plane('completed_provenance'),
                                        shader_reasons=plane('reason') if 'reason' in r else np.empty(0, np.uint32))
                    summary['artifact'] = dict(file=artifact.name, sha256=sha256(artifact))
                    row['modes'][key] = summary
            entry['frames'].append(row)
            print(f'{name} frame {index}: ' + ', '.join(f"{k} {v['measured_coverage_pct']:.2f}%" for k, v in row['modes'].items()), flush=True)
        report['scenes'][name] = entry
        (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for path, expected in {**identities, **{str(ROOT / k): v for k, v in sources.items()}}.items():
        if sha256(path) != expected:
            raise ValueError(f'input changed during experiment: {path}')
    for name, entry in report['scenes'].items():
        if sha256(args.corpus / name / 'zed_raw.npy') != entry['recording_sha256'] or sha256(args.corpus / name / 'scene.json') != entry['scene_metadata_sha256']:
            raise ValueError(f'recording changed during experiment: {name}')
    report['status'] = 'complete_development_diagnosis'
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--scenes', nargs='+', required=True)
    ap.add_argument('--indices', nargs='+', type=int, default=[0])
    ap.add_argument('--disparities', nargs='*', type=int, default=[128, 256],
                    help='Search sizes; derive larger native-profile ranges from calibrated fx*baseline/nearest_Z plus margin')
    ap.add_argument('--reference-costs', nargs='+', choices=['normalized', 'census'], default=['normalized'])
    ap.add_argument('--reference-paths', nargs='+', type=int, choices=[4, 8], default=[8])
    ap.add_argument('--reference-subpixel', nargs='+', choices=['equiangular', 'parabolic', 'none'], default=['equiangular'])
    ap.add_argument('--reference-lr-sampling', choices=['nearest', 'linear'], default='nearest')
    ap.add_argument('--reference-lr-tolerance', type=float, default=1.)
    ap.add_argument('--reference-lr-max-jump', type=float, default=1.,
                    help='Maximum disparity jump across contributing linear LR taps; native pixels')
    ap.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument('--coarse-manifest', type=Path, default=ROOT / 'artifacts/depth-video/v8-review-milestone/fixtures/zed2_320x180/manifest.json')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
