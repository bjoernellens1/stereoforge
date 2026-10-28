#!/usr/bin/env python3
"""Explicit, bounded independent-SGM static matrix; never a live-FPS claim.

Only ``compare`` dispatches native work. Profiles JSON is a list of three
{name, manifest, disparities} rows; manifest paths are relative to that JSON.
All three manifests must be F0 exports (including the identity-size export).

Freeze v1 minimal shape (hashes are full SHA256 strings)::

  {"schema":"stereoforge.multiresolution_freeze.v1",
   "native":{"file":"/absolute/daxa_depth_benchmark","sha256":"..."},
   "compiler":{"file":"/absolute/slangc","sha256":"..."},
   "file_sha256":{"/absolute/canonical-source-or-fixture-or-dependency":"..."},
   "runtime_alias_targets":{"/logical/library.so":"/canonical/library.so.3"},
   "report_identity":{"full":{...},"medium":{...},"half":{...}},
   "device_identity":{...stable device/UUID/driver/compiler report fields...}}

Each report_identity is the complete identity accepted by
compare_daxa_depth.validate_report, collected by its native_identities helper.
file_sha256 binds every identity source/header/dependency/build/SPIR-V input,
all fixture files, this driver and the profiles JSON. Root supplies the reviewed
freeze; this driver never creates or updates an expected hash from current bytes.
Raw vectors remain chronological. Additive nearest-rank statistics do not alter
the native report's historical mean/upper-index p50/p95 policy.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import compare_daxa_depth as depth
import multiresolution_fixture as fixture_exporter
from multiresolution_fixture import PIXEL_CENTER_CONVENTION

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(paths=4, p1=8, p2=32, subpixel=True, slots=2, wave_size=32,
                lr_tolerance=1, stereo_schedule='sequential')
SHAPES = ((1280, 720, 128), (960, 540, 96), (640, 360, 64))
DEVICE_KEYS = ('device', 'device_uuid', 'vendor_id', 'device_id', 'driver_name', 'driver_info',
               'driver_uuid', 'driver_version', 'slang_version', 'slangc_sha256', 'daxa_commit', 'compiler_route')
HASH = re.compile(r'[0-9a-f]{64}\Z')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_profile_family(profiles):
    """Validate already fixture-validated manifest objects; no I/O or dispatch."""
    try:
        if not isinstance(profiles, list) or len(profiles) != 3:
            raise ValueError('exactly three ordered profiles required')
        names = [r['name'] for r in profiles]
        if any(not isinstance(n, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', n) for n in names) or len(set(names)) != 3:
            raise ValueError('three distinct safe profile names required')
        first = profiles[0]['manifest']
        fmeta = profiles[0]['metadata']
        donor = first['provenance']['multiresolution_profile']
        for row, (w, h, d) in zip(profiles, SHAPES):
            m = row['manifest']; p = m['provenance']['multiresolution_profile']; s = w / 1280
            if (m['rectified'] != dict(width=w, height=h) or type(row['disparities']) is not int or row['disparities'] != d):
                raise ValueError('unsupported shape/disparity profile')
            for key, value in SETTINGS.items():
                if key in row and (type(row[key]) is not type(value) or row[key] != value):
                    raise ValueError('changed matched stereo settings: ' + key)
            if row['metadata'] != fmeta or m['source'] != first['source'] or m['capture'] != first['capture']:
                raise ValueError('different original capture/source/metadata')
            for key in ('original_identity', 'original_json', 'R1', 'R2'):
                if m['calibration'][key] != first['calibration'][key]:
                    raise ValueError('different calibration/pose')
            for key in ('original_left_rgb', 'original_right_rgb'):
                if m['expected'][key]['sha256'] != first['expected'][key]['sha256']:
                    raise ValueError('different original RGB')
            for key in ('schema', 'donor_manifest', 'donor_files_sha256', 'donor_manifest_sha256_before',
                        'donor_manifest_sha256_after', 'donor_rectified', 'pixel_center_convention', 'exporter_sha256'):
                if p[key] != donor[key]:
                    raise ValueError('different F0 donor or export convention')
            if (p['schema'] != 'stereoforge.multiresolution_profile.v1' or
                    p['pixel_center_convention'] != PIXEL_CENTER_CONVENTION or
                    p['donor_rectified'] != dict(width=1280, height=720) or p['target_rectified'] != m['rectified'] or
                    p['donor_manifest_sha256_before'] != p['donor_manifest_sha256_after'] or
                    p['donor_manifest']['sha256'] != p['donor_manifest_sha256_before']):
                raise ValueError('invalid donor provenance')
            ratio = p['scale']
            if (type(ratio['numerator']) is not int or type(ratio['denominator']) is not int or
                    ratio['denominator'] <= 0 or ratio['numerator'] / ratio['denominator'] != s):
                raise ValueError('invalid profile scale')
            a = np.array([[s, 0, (s-1)/2], [0, s, (s-1)/2], [0, 0, 1.]])
            for key in ('P1', 'P2'):
                actual = np.asarray(m['calibration'][key], dtype=float)
                reference = np.asarray(first['calibration'][key], dtype=float)
                if (actual.shape != (3, 4) or reference.shape != (3, 4) or
                        not np.isfinite(actual).all() or not np.array_equal(actual, a @ reference)):
                    raise ValueError('projection does not preserve pixel-centre metric geometry')
        return profiles
    except (KeyError, TypeError, ZeroDivisionError, OverflowError) as error:
        raise ValueError('malformed profile family') from error


def interleaved_schedule(names, repeats=3):
    if (not isinstance(names, (list, tuple)) or len(names) != 3 or len(set(names)) != 3 or
            type(repeats) is not int or not 3 <= repeats <= 9 or repeats % 3):
        raise ValueError('three names and 3/6/9 balanced repetitions required')
    return [(r, names[(i+r) % 3]) for r in range(repeats) for i in range(3)]


def summarize_samples(samples):
    """Nearest rank ceil(p*N)-1; never sorts/mutates chronological input."""
    if (not isinstance(samples, (list, tuple)) or not samples or
            any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in samples)):
        raise ValueError('nonempty finite nonnegative chronological samples required')
    ordered = sorted(samples); n = len(ordered)
    try:
        mean = math.fsum(samples)/n
    except OverflowError as error:
        raise ValueError('timing sample sum exceeds finite domain') from error
    return dict(count=n, mean=mean, p50=ordered[math.ceil(.5*n)-1],
                p95=ordered[math.ceil(.95*n)-1], p99=ordered[math.ceil(.99*n)-1], max=ordered[-1],
                percentile_policy='nearest_rank_ceil_pN_minus_1')


def evaluate_quality(predicted, validity, truth=None, truth_validity=None, support=None,
                     *, truth_declaration=None, error_threshold_m=.05):
    """Score a declared independent truth on its own native pixel grid.

    Unsupported pixels are excluded explicitly; missed valid truth stays in the
    completeness denominator. Unknown truth is never treated as a false accept.
    This function does not resample a full-SGM result into reference truth.
    """
    if truth is None:
        return dict(status='unknown_no_independent_truth', quality_qualified=False)
    if (type(error_threshold_m) not in (int, float) or not math.isfinite(error_threshold_m) or error_threshold_m <= 0):
        raise ValueError('positive finite frozen metric error threshold required')
    p = np.asarray(predicted, dtype=float); t = np.asarray(truth, dtype=float)
    declaration = truth_declaration or {}
    if (p.ndim != 2 or p.shape != t.shape or declaration.get('independent') is not True or
            declaration.get('grid_shape') != list(p.shape) or not isinstance(declaration.get('source'), str) or
            not declaration['source'].strip()):
        raise ValueError('independent truth with declared matching native grid required')
    def mask(values):
        a = np.asarray(values)
        if a.shape != p.shape or not np.isin(a, [0, 1]).all():
            raise ValueError('binary native-grid validity/support required')
        return a.astype(bool)
    pv, tv = mask(validity), mask(truth_validity)
    supported = np.ones(p.shape, bool) if support is None else mask(support)
    if not np.isfinite(p[pv]).all() or np.any(p[pv] <= 0) or not np.isfinite(t[tv]).all() or np.any(t[tv] <= 0):
        raise ValueError('valid axial depths must be finite positive metres')
    target = tv & supported; compared = target & pv
    error = np.abs(p[compared] - t[compared]); n = int(target.sum()); c = len(error)
    correct = int((error <= error_threshold_m).sum())
    scale = float(error.max()) if c else 0.
    normalized = error / scale if scale else error
    return dict(status='independent_native_grid', quality_qualified=False, truth_declaration=declaration,
                error_threshold_m=error_threshold_m, truth_valid_supported=n, compared=c,
                missed_valid_truth=n-c, completeness=c/n if n else None,
                correct_completeness=correct/n if n else None, false_accepts=c-correct,
                false_accept_fraction=(c-correct)/c if c else None,
                unscored_accepts=int((pv & supported & ~tv).sum()), unsupported_accepts=int((pv & ~supported).sum()),
                mae_m=float(normalized.mean()*scale) if c else None, rmse_m=float(np.sqrt(np.mean(normalized**2))*scale) if c else None,
                max_error_m=float(error.max()) if c else None)


def _path(value, directory):
    p = Path(value)
    if not p.is_absolute():
        p = directory / p
    if any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('symlink inputs are not allowed')
    return p.resolve(strict=True)


def verify_freeze(freeze_path, expected_hash, native, profiles_path, rows):
    """Verify reviewed known hashes, including every transitive identity file."""
    if not isinstance(expected_hash, str) or not HASH.fullmatch(expected_hash) or sha256(freeze_path) != expected_hash:
        raise ValueError('freeze SHA256 differs from explicitly supplied hash')
    frozen = depth.base.strict_json(Path(freeze_path).read_text())
    if frozen.get('schema') != 'stereoforge.multiresolution_freeze.v1':
        raise ValueError('unsupported freeze schema')
    device = frozen['device_identity']
    if not isinstance(device, dict) or any(k not in device for k in DEVICE_KEYS):
        raise ValueError('frozen device/driver/compiler identity required')
    files = frozen['file_sha256']
    if not isinstance(files, dict) or not files:
        raise ValueError('nonempty frozen file inventory required')
    resolved = {}
    for name, digest in files.items():
        p = _path(name, ROOT)
        if not isinstance(digest, str) or not HASH.fullmatch(digest) or sha256(p) != digest:
            raise ValueError('frozen input changed: ' + str(name))
        resolved[str(p)] = digest
    aliases = frozen.get('runtime_alias_targets', {})
    if not isinstance(aliases, dict):
        raise ValueError('runtime alias targets must be a mapping')
    for logical, target in aliases.items():
        if not Path(logical).is_absolute() or not Path(target).is_absolute():
            raise ValueError('runtime alias targets require absolute paths')
        canonical = _path(target, ROOT)
        if str(canonical) != target or Path(logical).resolve(strict=True) != canonical or target not in resolved:
            raise ValueError('runtime alias target changed: ' + logical)
    def bound(path, digest=None):
        logical = Path(path)
        logical = Path(os.path.abspath(logical if logical.is_absolute() else ROOT/logical))
        p = logical.resolve(strict=True)
        if p != logical and aliases.get(str(logical)) != str(p):
            raise ValueError('unfrozen runtime alias target: ' + str(logical))
        if str(p) not in resolved or (digest is not None and resolved[str(p)] != digest):
            raise ValueError('input absent from reviewed freeze: ' + str(p))
    for module in (depth, depth.preprocess, depth.base, depth.right, fixture_exporter):
        bound(module.__file__)
    bound(__file__); bound(profiles_path)
    for key, given in (('native', native), ('compiler', None)):
        entry = frozen[key]; actual = Path(entry['file']).resolve(strict=True)
        if given is not None and actual != Path(given).resolve(strict=True):
            raise ValueError('native path differs from freeze')
        bound(entry['file'], entry['sha256'])
    for row in rows:
        identity = frozen['report_identity'][row['name']]
        if identity['binary_sha256'] != frozen['native']['sha256'] or identity['compiler_sha256'] != frozen['compiler']['sha256']:
            raise ValueError('report binary/compiler identity differs from freeze')
        for section in ('source_sha256', 'shader_source_sha256', 'dependency_sha256', 'build_configuration_sha256', 'compiler_input_sha256', 'fixture_file_sha256', 'optional_dependency_header_sha256'):
            mapping = identity[section]
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError('missing identity section: ' + section)
            def visit(tree):
                for path, digest in tree.items():
                    if isinstance(digest, dict):
                        visit(digest)
                    else:
                        bound(path, digest)
            visit(mapping)
        spvs = identity['offline_spirv_sha256']
        if not isinstance(spvs, dict) or not spvs:
            raise ValueError('missing SPIR-V inventory')
        for name, digest in spvs.items():
            bound(Path(identity['offline_shader_directory']) / name, digest)
        for path, digest in row['inventory'].items():
            bound(path, digest)
        if identity['manifest_sha256'] != sha256(row['manifest_path']):
            raise ValueError('manifest report identity differs')
    return frozen


def load_profiles(path):
    path = _path(path, ROOT)
    # Reuse the strict recursive duplicate/nonfinite parser with an object
    # envelope: its public report API deliberately rejects top-level lists.
    wrapped = depth.base.strict_json('{"profiles":' + path.read_text() + '}')
    if set(wrapped) != {'profiles'}:
        raise ValueError('invalid profiles JSON trailing members')
    entries = wrapped['profiles']
    if not isinstance(entries, list):
        raise ValueError('profiles JSON must be a list')
    rows = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'name', 'manifest', 'disparities'}:
            raise ValueError('profile requires only name, manifest, disparities')
        manifest_path = _path(entry['manifest'], path.parent)
        m, metadata, _, _, metric, cv_identity = depth.fixture_planes(manifest_path)
        import cv2
        cv_binary = next(Path(cv2.__file__).parent.glob('*.so')).resolve(strict=True)
        inventory = {str(manifest_path): sha256(manifest_path), str(cv_binary): cv_identity['binary_sha256']}
        for value in [m['capture'], m['calibration']['original_json'], m['calibration']['rectified_json'], *m['maps'].values(), *m['expected'].values()]:
            p = _path(value['file'], manifest_path.parent)
            inventory[str(p)] = value['sha256']
        rows.append(dict(entry, manifest=m, metadata=metadata, metric=metric,
                         manifest_path=manifest_path, inventory=inventory))
    return validate_profile_family(rows)


def _validate_samples(report, frames):
    raw = report.get('raw_samples_ms')
    if not isinstance(raw, dict) or set(raw) != set(depth.TIMINGS):
        raise ValueError('all thirteen chronological native timing vectors required')
    result = {}
    for key in depth.TIMINGS:
        values = raw[key]
        stats = summarize_samples(values)
        if len(values) != frames:
            raise ValueError('native raw sample count differs: ' + key)
        ordered = sorted(values)
        total = 0.
        for value in ordered:
            total += value  # Native summary sorts its copy before std::accumulate.
        old = dict(mean=total/frames, p50=ordered[frames//2], p95=ordered[min(frames-1,frames*95//100)])
        # Preserve and check the historical C++ accumulation/index policy.
        if report[key] != old:
            raise ValueError('native historical summary differs from raw samples: ' + key)
        result[key] = stats
    return result


def run_comparison(profiles_path, native, freeze_path, freeze_sha256, output, *, frames=100, warmup=20, repeats=3):
    if type(frames) is not int or not 30 <= frames <= 300 or type(warmup) is not int or not 3 <= warmup <= 30:
        raise ValueError('bounded performance samples require frames30..300/warmup3..30')
    schedule = interleaved_schedule(['A','B','C'], repeats)
    rows = load_profiles(profiles_path)
    frozen = verify_freeze(freeze_path, freeze_sha256, native, profiles_path, rows)
    output = Path(output).absolute()
    depth.base.create_output(output)
    output_hashes = {}
    evidence = dict(schema='stereoforge.multiresolution_comparison.v1', passed=False, runs=[],
                    freeze_sha256=freeze_sha256, frames=frames, warmup=warmup, repeats=repeats,
                    profile_order=[r['name'] for r in rows], settings=SETTINGS,
                    scope='static repeated original capture; independent two-eye SGM; RGB/Temporal disabled',
                    quality=dict(status='unknown_no_independent_truth', quality_qualified=False),
                    unique_throughput='unmeasured_static_fixture', frame_age='unmeasured_no_clock_mapping',
                    drops='unmeasured_static_fixture', live60_qualified=False,
                    inherited_validation_environment={k:os.environ[k] for k in ('VK_INSTANCE_LAYERS','VK_LAYER_VALIDATE_SYNC','VK_LAYER_ENABLES','VK_LOADER_DEBUG','SF_DAXA_VALIDATION') if k in os.environ})
    def unchanged():
        if verify_freeze(freeze_path, freeze_sha256, native, profiles_path, rows) != frozen:
            raise ValueError('reviewed freeze changed')
        if any(sha256(path) != digest for path, digest in output_hashes.items()):
            raise ValueError('previous output/log changed')
    try:
        if evidence['inherited_validation_environment']:
            raise ValueError('performance matrix requires validation environment unset')
        for repetition, letter in schedule:
            row = rows[ord(letter)-ord('A')]; name = f"r{repetition+1}-{row['name']}"
            directory = output/name; directory.mkdir()
            report_path = directory/'native.json'
            command = [str(Path(native).resolve()), '--manifest', str(row['manifest_path']), '--output', str(report_path),
                       '--check-reference', '--frames', str(frames), '--warmup', str(warmup), '--slots', '2',
                       '--wave-size', '32', '--disparities', str(row['disparities']), '--paths', '4',
                       '--p1', '8', '--p2', '32', '--lr-tolerance', '1', '--stereo-schedule', 'sequential',
                       '--right-reference-mode', 'independent-sgm']
            run = dict(repetition=repetition, profile=row['name'], command=command, passed=False)
            evidence['runs'].append(run)
            unchanged()
            child = subprocess.run(command, capture_output=True, text=True, errors='replace')
            for suffix, value in (('stdout',child.stdout),('stderr',child.stderr)):
                (directory/suffix).write_text(value)
                output_hashes[str(directory/suffix)] = sha256(directory/suffix)
            run['returncode'] = child.returncode
            # Always inspect inputs after a child, even if the child failed.
            unchanged()
            depth.base.require_child_clean(child.returncode, child.stdout, child.stderr)
            raw_report = report_path.read_bytes(); report_hash = sha256(report_path)
            report = depth.base.strict_json(raw_report.decode())
            output_hashes[str(report_path)] = report_hash
            profile = dict(width=row['manifest']['rectified']['width'],height=row['manifest']['rectified']['height'],
                           disparities=row['disparities'],paths=4,p1=8,p2=32,subpixel=True)
            depth.validate_report(report,row['manifest'],row['metadata'],row['metric'],frozen['report_identity'][row['name']],
                                  profile,frames,warmup,2,32,False,stereo_schedule='sequential')
            if any(report['device'].get(k) != v for k, v in frozen['device_identity'].items()):
                raise ValueError('actual device/driver/compiler identity differs')
            run.update(statistics_ms=_validate_samples(report,frames),report_sha256=report_hash,
                       candidate_hypotheses=2*profile['width']*profile['height']*profile['disparities'],
                       requested_buffer_bytes=report['requested_combined_buffer_bytes'],
                       diagnostics_after_timing_ms=report['diagnostic_after_timing_ms'],
                       initialization_ms=report['initialization_ms'],
                       nominal_near_depth_m=row['manifest']['provenance']['multiresolution_profile']['nominal_near_depth_m_by_disparities'][str(row['disparities'])])
            if report_path.read_bytes() != raw_report:
                raise ValueError('native report changed during validation')
            unchanged(); run['passed'] = True
        evidence['passed'] = True
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        evidence['error'] = str(error)
    finally:
        try:
            unchanged(); evidence['inputs_unchanged_after'] = True
        except (ValueError, OSError, KeyError, TypeError) as error:
            evidence['passed'] = False; evidence['inputs_unchanged_after'] = False
            evidence['final_input_error'] = str(error)
        evidence['output_sha256'] = output_hashes
        (output/'comparison.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command',required=True)
    compare = commands.add_parser('compare')
    for option in ('profiles','native','freeze','output'):
        compare.add_argument('--'+option,type=Path,required=True)
    compare.add_argument('--freeze-sha256',required=True)
    compare.add_argument('--frames',type=int,default=100)
    compare.add_argument('--warmup',type=int,default=20)
    compare.add_argument('--repeats',type=int,default=3)
    args = parser.parse_args()
    try:
        result = run_comparison(args.profiles,args.native,args.freeze,args.freeze_sha256,args.output,
                                frames=args.frames,warmup=args.warmup,repeats=args.repeats)
    except (ValueError,OSError,KeyError,TypeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(passed=result['passed'],output=str(args.output))))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
