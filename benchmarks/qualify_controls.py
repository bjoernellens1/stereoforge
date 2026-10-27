"""Bounded native camera-control API smoke; no images or image-quality claim.

Default is API readback only. --exercise writes the baseline's existing values,
checks validation atomicity, and restores all baseline fields in finally. Opening
a camera retains the existing capture initialization behavior. Automatic modes
can subsequently select different exposure/gain/WB, so these are not restoration
failures when their automatic mode was originally enabled.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import monotonic_ns, sleep
import math

CONTROL_KEYS = (
    'auto_exposure_gain', 'auto_white_balance', 'exposure_left', 'exposure_right',
    'gain_left', 'gain_right', 'white_balance', 'brightness', 'contrast',
    'saturation', 'hue', 'sharpness', 'gamma', 'led', 'roi_left', 'roi_right',
)
AUTOMATIC_KEYS = ('auto_exposure_gain', 'auto_white_balance')
# The upstream LED getter changes GPIO direction; qualification neither reads
# it through that unsafe getter nor writes a value that cannot be restored.
RESTORABLE_KEYS = tuple(key for key in CONTROL_KEYS if key != 'led')


def _stable_keys(baseline):
    keys = set(RESTORABLE_KEYS)
    if baseline['auto_exposure_gain']:
        keys.difference_update(('exposure_left', 'exposure_right', 'gain_left', 'gain_right'))
    if baseline['auto_white_balance']:
        keys.discard('white_balance')
    return sorted(keys)


def _mismatches(expected, observed, keys):
    return {key: {'expected': expected[key], 'observed': observed.get(key)}
            for key in keys if expected[key] != observed.get(key)}


def _sample(camera):
    start = monotonic_ns()
    result = camera.get_controls()
    return {'sample_start_monotonic_ns': start, 'sample_end_monotonic_ns': monotonic_ns(),
            'values': result}


def qualify_controls(camera, exercise=False):
    """Operate an already-open native capture; does not take its ownership."""
    before = _sample(camera)
    baseline = before['values']
    report = {'mode': 'exercise_baseline_values' if exercise else 'read_only', 'before': before,
              'measurement': 'camera API readback only',
              'limitations': ['Automatic bool getters cannot distinguish upstream I/O failure.',
                             'Void upstream setters cannot report write success; readback is compared.',
                             'Exposure/gain values are normalized controls, not calibrated exposure duration.',
                             'LED omitted from exercise/restoration because safe LED readback is unavailable.',
                             'No image-quality, physical illumination, or sensor timing qualification.']}
    if not exercise:
        report['status'] = 'partial_readback' if baseline.get('read_errors') else 'readback_observed'
        return report
    missing = [key for key in RESTORABLE_KEYS if baseline.get(key) is None]
    if missing or baseline.get('read_errors'):
        report.update(status='exercise_not_started', reason='Complete baseline required for restoration',
                      missing_controls=missing)
        return report
    stable = _stable_keys(baseline)
    report['stable_comparison_keys'] = stable
    manual = {key: baseline[key] for key in RESTORABLE_KEYS if key not in AUTOMATIC_KEYS}
    manual.update(auto_exposure_gain=False, auto_white_balance=False)
    restoration_errors = []
    try:
        # Even the validation probe belongs inside finally: restore if a broken
        # adapter were ever to apply part of this intentionally invalid request.
        invalid_rejected = False
        try:
            camera.set_controls({'exposure_left': baseline['exposure_left'], 'brightness': 999})
        except (TypeError, ValueError, OverflowError):
            invalid_rejected = True
        after_invalid = _sample(camera)
        report['invalid_request'] = {'rejected': invalid_rejected, 'after': after_invalid,
            'stable_mismatches': _mismatches(baseline, after_invalid['values'], stable)}
        if not invalid_rejected or report['invalid_request']['stable_mismatches']:
            report['status'] = 'invalid_request_failed_atomicity'
        else:
            report['request'] = manual
            observed = camera.set_controls(manual)
            report['applied_readback'] = observed
            report['applied_mismatches'] = _mismatches(manual, observed, RESTORABLE_KEYS)
            report['status'] = ('pass_api_readback' if not report['applied_mismatches'] and
                                not observed.get('read_errors') and not observed.get('write_errors')
                                else 'api_readback_mismatch')
    except Exception as error:
        report['status'] = 'exercise_error'
        report['exercise_error'] = f'{type(error).__name__}: {error}'
    finally:
        # Manual setters disable automatic modes upstream. Restore the complete
        # snapshot first, then restore the explicit automatic flags separately.
        try:
            restored_manual = camera.set_controls({key: baseline[key] for key in RESTORABLE_KEYS
                                                   if key not in AUTOMATIC_KEYS})
            restoration_errors.extend(restored_manual.get('write_errors', []))
        except Exception as error:
            restoration_errors.append(f'manual restoration: {type(error).__name__}: {error}')
        try:
            restored_flags = camera.set_controls({key: baseline[key] for key in AUTOMATIC_KEYS})
            restoration_errors.extend(restored_flags.get('write_errors', []))
        except Exception as error:
            restoration_errors.append(f'automatic restoration: {type(error).__name__}: {error}')
        try:
            restored = _sample(camera)
            report['restored'] = restored
            report['restore_mismatches'] = _mismatches(baseline, restored['values'], stable)
            restoration_errors.extend(restored['values'].get('read_errors', []))
        except Exception as error:
            restoration_errors.append(f'restoration readback: {type(error).__name__}: {error}')
        report['restore_errors'] = restoration_errors
        if restoration_errors or report.get('restore_mismatches'):
            report['status'] = 'restoration_not_verified'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exercise', action='store_true', help='Write current baseline values, then restore all baseline controls')
    parser.add_argument('--warmup-seconds',type=float,default=0.,help='Read-only stability screen before qualification (0 to 5 seconds)')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--device-id', type=int, default=-1)
    parser.add_argument('--output', type=Path, default=Path('artifacts/controls-smoke.json'))
    args = parser.parse_args()
    if not math.isfinite(args.warmup_seconds) or not 0<=args.warmup_seconds<=5:
        parser.error("warmup-seconds must be finite and between 0 and 5")
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'images_saved': False}
    try:
        from stereoforge.capture import _native_module
        native = _native_module()
        camera = native.ZedCapture(args.width, args.height, args.fps, args.device_id)
        try:
            report['serial'] = camera.serial
            warmup=[]
            deadline=monotonic_ns()+int(args.warmup_seconds*1e9)
            while monotonic_ns()<deadline:
                warmup.append(_sample(camera))
                remaining=(deadline-monotonic_ns())/1e9
                if remaining>0:sleep(min(.25,remaining))
            if warmup:
                keys=_stable_keys(warmup[-1]['values'])
                screened=warmup[-3:]
                stable=all(not _mismatches(screened[0]['values'],sample['values'],keys) and
                           not sample['values'].get('read_errors') for sample in screened)
                report.update(warmup_samples=warmup,warmup_stable=stable,warmup_comparison_keys=keys)
            if args.exercise and warmup and not report['warmup_stable']:
                report.update(status='exercise_not_started',reason='Read-only warmup API snapshots were unstable')
            else:
                report.update(qualify_controls(camera, args.exercise))
        finally:
            camera.close()
        try:
            camera.get_controls()
            report['closed_guard'] = False
            report['status'] = 'closed_guard_failed'
        except RuntimeError:
            report['closed_guard'] = True
    except Exception as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'artifact': str(args.output), 'status': report['status'], 'images_saved': False}))
    return 0 if report['status'] in ('readback_observed', 'pass_api_readback') else 1


if __name__ == '__main__':
    raise SystemExit(main())
