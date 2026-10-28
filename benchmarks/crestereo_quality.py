#!/usr/bin/env python3
"""Offline CREStereo development diagnostic: freeze prediction before scoring truth."""
import argparse
import hashlib
import json
import time
import platform
from pathlib import Path

import cv2
import numpy as np

try:
    from . import crestereo_cpu_smoke as adapter
    from .crestereo_cpu_smoke import prepare_inputs, verify_model
except ImportError:
    import crestereo_cpu_smoke as adapter
    from crestereo_cpu_smoke import prepare_inputs, verify_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shifted_pair(image, shift):
    if image.ndim not in (2, 3) or not isinstance(shift, int) or abs(shift) >= image.shape[1]:
        raise ValueError('invalid image or shift')
    right = np.zeros_like(image)
    support = np.zeros(image.shape[:2], bool)
    w = image.shape[1]
    lo, hi = max(0, shift), min(w, w + shift)
    right[:, lo-shift:hi-shift] = image[:, lo:hi]
    support[:, lo:hi] = True
    return image.copy(), right, support


def restore_disparity(raw, width, height):
    if raw.shape != (240, 320) or min(width, height) <= 0:
        raise ValueError('expected model grid and positive dimensions')
    return cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR) * (width / 320.)


def depth_from_disparity(disparity, focal, baseline, principal_offset):
    if not np.isfinite([focal, baseline, principal_offset]).all() or focal <= 0 or baseline <= 0:
        raise ValueError('positive finite focal and baseline required')
    effective = disparity - principal_offset
    valid = np.isfinite(disparity) & (effective > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float64)
    np.divide(focal * baseline, effective, out=depth, where=valid)
    return depth, valid


def score(prediction, truth, support, threshold=1., principal_offset=0.):
    if prediction.shape != truth.shape or support.shape != truth.shape or not np.isfinite([threshold, principal_offset]).all() or threshold < 0:
        raise ValueError('matching grids and nonnegative threshold required')
    domain = np.asarray(support, bool) & np.isfinite(truth)
    positive = domain & np.isfinite(prediction) & (prediction > 0)
    valid = domain & np.isfinite(prediction) & (prediction - principal_offset > 0)
    error = np.abs(prediction - truth)
    correct = valid & (error <= threshold)
    eligible, count = int(domain.sum()), int(valid.sum())
    return dict(eligible=eligible, numerical_positive=int(positive.sum()), numerical_valid=count, correct=int(correct.sum()),
                false_accepted=int((valid & ~correct).sum()),
                valid_coverage=count/eligible if eligible else None,
                correct_completeness=int(correct.sum())/eligible if eligible else None,
                mean_abs_disparity_error=float(error[valid].mean()) if count else None)


def read_entry(root, entry):
    path = root / entry['file']
    data = path.read_bytes()
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('fixture hash mismatch')
    return data


def load_model(model):
    import onnxruntime as ort
    verify_model(model)
    opt = ort.SessionOptions(); opt.intra_op_num_threads = 4; opt.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model / 'crestereo.onnx'), sess_options=opt, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('CPU provider required')
    return session


def predict(session, left, right):
    output = session.run(['disparity'], prepare_inputs(left, right))[0]
    if output.shape != (1, 1, 240, 320):
        raise ValueError('unexpected output shape')
    return output[0, 0]


def witness(model, output):
    if output.exists(): raise FileExistsError(output)
    cv2.setNumThreads(1)
    session = load_model(model)
    rng = np.random.default_rng(20261028)
    image = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    results = []
    for shift in (0, 8, -8, 16, -16):
        left, right, support = shifted_pair(image, shift)
        # Ignore padding and 32 pixels at the border; still only a behavioral witness.
        support[:, :48] = False; support[:, -48:] = False
        support[:32, :] = False; support[-32:, :] = False
        pred = predict(session, left, right)
        finite = support & np.isfinite(pred)
        results.append(dict(input_disparity_pixels=shift, finite_count=int(finite.sum()),
                            median_raw_output=float(np.median(pred[finite])) if finite.any() else None))
    left, right, support = shifted_pair(image, 8)
    pred = predict(session, right, left)
    interior = np.zeros(support.shape, bool); interior[32:-32, 48:-48] = True
    finite = interior & np.isfinite(pred)
    results.append(dict(input_disparity_pixels=-8, mode='swapped_positive_pair', finite_count=int(finite.sum()),
                        median_raw_output=float(np.median(pred[finite])) if finite.any() else None))
    report = dict(scope='synthetic shifts, same 320x240 grid; behavior witness only, not proof of unit, accuracy or calibration',
                  model_hashes=verify_model(model), results=results)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


def prefilter(image, sigma):
    if sigma not in (0, 2):
        raise ValueError("supported diagnostic sigma values are 0 and 2")
    return image if sigma == 0 else cv2.GaussianBlur(image, (0, 0), sigmaX=2, sigmaY=2)


def freeze(model, fixture, output, sigma=0):
    if output.exists(): raise FileExistsError(output)
    manifest_path = fixture / 'manifest.json'; manifest = json.loads(manifest_path.read_bytes())
    w, h = manifest['rectified']['width'], manifest['rectified']['height']
    images = [np.frombuffer(read_entry(fixture, manifest['expected'][name]), np.uint8).reshape(h, w, 3)
              for name in ('rect_left_rgb', 'rect_right_rgb')]
    import onnxruntime as ort
    model_hashes = verify_model(model); manifest_hash = sha(manifest_path)
    source_hashes = {str(Path(__file__).name): sha(__file__), 'crestereo_cpu_smoke.py': sha(adapter.__file__)}
    cv2.setNumThreads(1); session = load_model(model)
    begin = time.perf_counter_ns()
    images = [prefilter(image, sigma) for image in images]
    prefilter_ms = (time.perf_counter_ns() - begin) / 1e6
    pred = predict(session, *images)
    if (verify_model(model) != model_hashes or sha(manifest_path) != manifest_hash
            or sha(__file__) != source_hashes[Path(__file__).name] or sha(adapter.__file__) != source_hashes['crestereo_cpu_smoke.py']):
        raise ValueError('model or input manifest changed during prediction')
    for name in ('rect_left_rgb', 'rect_right_rgb'):
        read_entry(fixture, manifest['expected'][name])
    output.mkdir(parents=True)
    raw = output / 'prediction.f32'; raw.write_bytes(pred.astype('<f4').tobytes())
    report = dict(scope='frozen raw model prediction; no truth read', manifest_sha256=sha(manifest_path),
                  image_sha256={name:manifest['expected'][name]['sha256'] for name in ('rect_left_rgb','rect_right_rgb')},
                  model_hashes=verify_model(model), source_hashes=source_hashes, runtime=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__, onnxruntime=ort.__version__, provider='CPUExecutionProvider', intra_op_threads=4, inter_op_threads=1, opencv_threads=cv2.getNumThreads()), prediction_sha256=sha(raw), width=w, height=h, prefilter_sigma=sigma, prefilter_ms=prefilter_ms,
                  preprocessing='OpenCV bilinear RGB to 320x240 BGR, float32 0..1; half model input aligned corners',
                  resize_geometry='pixel centers: x_model=(x_source+0.5)*320/width-0.5; analogous y with 240/height')
    (output / 'freeze.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


def evaluate(fixture, frozen, output):
    if output.exists(): raise FileExistsError(output)
    freeze_bytes = (frozen/'freeze.json').read_bytes()
    freeze_hash = hashlib.sha256(freeze_bytes).hexdigest()
    record = json.loads(freeze_bytes)
    if sha(fixture/'manifest.json') != record['manifest_sha256'] or sha(frozen/'prediction.f32') != record['prediction_sha256']:
        raise ValueError('frozen binding mismatch')
    manifest = json.loads((fixture/'manifest.json').read_bytes())
    truth = json.loads((fixture/'truth.json').read_bytes())
    if truth['native_manifest_sha256'] != record['manifest_sha256']:
        raise ValueError('truth input differs from frozen input')
    w, h = record['width'], record['height']
    raw = np.frombuffer((frozen/'prediction.f32').read_bytes(), '<f4').reshape(240, 320)
    pred = restore_disparity(raw, w, h)
    entries = truth['eyes']['left']
    def plane(name):
        e = entries[name]
        return np.frombuffer(read_entry(fixture, e), dtype=e['dtype']).reshape(e['shape'])
    gt = plane('disparity'); visible = plane('visible').astype(bool)
    clean = visible & ~plane('out_of_range').astype(bool) & ~plane('mixed_bilinear').astype(bool)
    support = plane('remap_support').astype(bool)
    # The complete 5x5 descriptor footprint must have rectification support.
    census = cv2.erode(support.astype(np.uint8), np.ones((5,5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    clean &= census
    p1, p2 = np.asarray(manifest['calibration']['P1']), np.asarray(manifest['calibration']['P2'])
    focal, baseline = p1[0,0], -p2[0,3]/p2[0,0]
    offset = float(p1[0,2]-p2[0,2])
    def scored(domain): return score(pred, gt, domain, principal_offset=offset)
    metrics = {'visible':scored(visible), 'native_clean':scored(clean)}
    for name in ('thin','edges','near','far'):
        metrics[name] = scored(clean & plane('region_'+name).astype(bool))
    edges = plane('region_edges').astype(bool)
    edge_neighborhood = cv2.dilate(edges.astype(np.uint8), np.ones((9,9), np.uint8)).astype(bool)
    metrics['edge_neighborhood_4px'] = scored(clean & edge_neighborhood)
    metrics['away_from_edges_4px'] = scored(clean & ~edge_neighborhood)
    occluded = plane('occluded').astype(bool)
    _, positive_depth = depth_from_disparity(pred, focal, baseline, p1[0,2]-p2[0,2])
    report = dict(scope='analytic synthetic development only; not physical, held-out, or quality promotion',
                  prediction_sha256=record['prediction_sha256'], freeze_sha256=freeze_hash, manifest_sha256=record['manifest_sha256'], prefilter_sigma=record.get('prefilter_sigma', 0),
                  model_hashes=record.get('model_hashes'), producer_source_hashes=record.get('source_hashes'), producer_runtime=record.get('runtime'), evaluator_sha256=sha(__file__),
                  truth_sha256=sha(fixture/'truth.json'), disparity_assumption='signed x_left-x_right in model pixels; positive effective disparity required; behavior witness only',
                  warning='all values are bilinearly interpolated predictions; finite positive effective disparity is numerical validity only, not independent verification or confidence; false accepted counts refer to this numerical lane, not LR checks',
                  source_grid=[w,h], model_grid=[320,240], horizontal_scale=w/320., vertical_scale=h/240.,
                  principal_offset=float(p1[0,2]-p2[0,2]), baseline=float(baseline), focal=float(focal),
                  positive_metric_denominator_pixels=int(positive_depth.sum()), metrics=metrics,
                  occluded_pixels=int(occluded.sum()), occluded_numerically_positive=int((occluded & np.isfinite(pred) & (pred>0)).sum()),
                  occluded_positive_metric_denominator=int((occluded & positive_depth).sum()) )
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd', required=True)
    a=sub.add_parser('witness'); a.add_argument('--model',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('freeze'); a.add_argument('--model',type=Path,required=True); a.add_argument('--fixture',type=Path,required=True); a.add_argument('--output',type=Path,required=True); a.add_argument('--sigma',type=int,choices=[0,2],default=0)
    a=sub.add_parser('evaluate'); a.add_argument('--fixture',type=Path,required=True); a.add_argument('--frozen',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); globals()[args.cmd](**{k:v for k,v in vars(args).items() if k!='cmd'})

if __name__=='__main__': main()
