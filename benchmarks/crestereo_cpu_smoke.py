#!/usr/bin/env python3
"""Offline CPU-only CREStereo feasibility on a private rectified RGB fixture.

No confidence, calibration validation, GPU result or quality claim is made.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


EXPECTED = {'crestereo.onnx': '0f0c59b353c09b571508791b0ecf5dcf657a389d1934a0912db5d2c52e897de9',
            'crestereo.data': '8e09346b8300891f72440b436eba8b1106f3eafd49fb7d0bad490ae6857dfd82',
            'metadata.json': 'c142ae8a1e98d8772c1b17bad451a21665f63d932e8c799480f9e38ab91f4c7d'}


def verify_model(directory):
    hashes = {name: digest(directory / name) for name in EXPECTED}
    if hashes != EXPECTED:
        raise ValueError('model archive members differ from pinned archive')
    return hashes


def prepare_inputs(left, right):
    if left.dtype != np.uint8 or right.dtype != np.uint8 or left.ndim != 3 or left.shape != right.shape or left.shape[2] != 3:
        raise ValueError('matching uint8 HWC RGB images required')
    result = {}
    for name, rgb in (('left', left), ('right', right)):
        # Resize is deliberately nonuniform for 16:9 sources. This changes
        # sampling and aspect ratio, but horizontal disparity scales by width.
        bgr = cv2.resize(rgb[..., ::-1], (320, 240), interpolation=cv2.INTER_LINEAR)
        f = bgr.astype(np.float32) / np.float32(255)
        x = np.linspace(0, 319, 160, dtype=np.float32)
        y = np.linspace(0, 239, 120, dtype=np.float32)
        mx, my = np.meshgrid(x, y)
        # remap uses interpolation-table quantization; use NumPy bilinear for
        # precise PyTorch align_corners=True sampling instead.
        x0 = np.floor(mx).astype(np.int32); y0 = np.floor(my).astype(np.int32)
        x1 = np.minimum(x0 + 1, 319); y1 = np.minimum(y0 + 1, 239)
        dx = (mx-x0)[..., None]; dy = (my-y0)[..., None]
        half = ((f[y0,x0]*(1-dx)+f[y0,x1]*dx)*(1-dy) + (f[y1,x0]*(1-dx)+f[y1,x1]*dx)*dy).astype(np.float32)
        result[name+'_image'] = np.ascontiguousarray(f.transpose(2,0,1)[None])
        result[name+'_image_dw2'] = np.ascontiguousarray(half.transpose(2,0,1)[None])
    return result


def scale_disparity(disparity, width, height):
    if disparity.shape != (240, 320) or width <= 0 or height <= 0:
        raise ValueError('expected 240x320 disparity and positive output size')
    return cv2.resize(disparity, (width, height), interpolation=cv2.INTER_LINEAR) * (width / 320.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True, help='private existing Daxa preprocess fixture')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--frames', type=int, default=10)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.warmup < 0 or a.frames < 1 or a.threads < 1:
        p.error('warmup must be nonnegative; frames and threads must be positive')
    import onnxruntime as ort
    archive_sha = digest(a.archive)
    if archive_sha != 'acfe65a00216c9eafc274f55b64acfd51b8d69c14c3e15a1425066b6dec6b549':
        raise ValueError('unexpected model archive hash')
    model = a.model_dir / 'crestereo.onnx'; data = a.model_dir / 'crestereo.data'; metadata = a.model_dir / 'metadata.json'
    model_hashes = verify_model(a.model_dir)
    source_hash = digest(__file__)
    if a.output.exists():
        raise FileExistsError('refusing to overwrite output')
    manifest_path = a.fixture / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    cv2.setNumThreads(1)
    width, height = manifest['rectified']['width'], manifest['rectified']['height']
    images = []
    source_hashes = {}; image_paths = {}
    for key in ('rect_left_rgb', 'rect_right_rgb'):
        desc = manifest['expected'][key]; path = a.fixture / desc['file']
        image_paths[key] = path
        image_bytes = path.read_bytes()
        source_hashes[key] = hashlib.sha256(image_bytes).hexdigest()
        if len(image_bytes) != width * height * 3 or len(image_bytes) != desc['bytes']:
            raise ValueError('fixture image byte length mismatch')
        if source_hashes[key] != desc['sha256']:
            raise ValueError('fixture image hash mismatch')
        images.append(np.frombuffer(image_bytes, np.uint8).reshape(height, width, 3))
    options = ort.SessionOptions(); options.intra_op_num_threads = a.threads; options.inter_op_num_threads = 1
    start = time.perf_counter_ns()
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    init_ms = (time.perf_counter_ns()-start)/1e6
    if session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('unexpected execution provider')
    if {x.name: x.shape for x in session.get_inputs()} != {'left_image':[1,3,240,320], 'right_image':[1,3,240,320], 'left_image_dw2':[1,3,120,160], 'right_image_dw2':[1,3,120,160]}:
        raise ValueError('unexpected input contract')
    samples = []
    for i in range(a.warmup + a.frames):
        t0 = time.perf_counter_ns(); inputs = prepare_inputs(*images); t1 = time.perf_counter_ns()
        out = session.run(['disparity'], inputs)[0]; t2 = time.perf_counter_ns()
        if out.shape != (1,1,240,320) or not np.isfinite(out).all():
            raise ValueError('invalid output shape or nonfinite disparity')
        restored = scale_disparity(out[0,0], width, height); t3 = time.perf_counter_ns()
        if i >= a.warmup:
            samples.append({'preprocess_ms':(t1-t0)/1e6,'inference_ms':(t2-t1)/1e6,'postprocess_ms':(t3-t2)/1e6,'total_ms':(t3-t0)/1e6})
    if (verify_model(a.model_dir) != model_hashes or digest(a.archive) != archive_sha
            or digest(__file__) != source_hash or digest(manifest_path) != manifest_sha
            or any(digest(image_paths[k]) != v for k, v in source_hashes.items())):
        raise ValueError('model, archive or script changed during run')
    report = {'scope':'offline CPU smoke; repeated rectified captured frame; no quality, confidence, GPU or live throughput claim; before/after hashes do not guarantee atomic model loading',
              'archive_sha256':archive_sha, 'upstream_repository':'https://github.com/quic/ai-hub-models',
              'upstream_tag':'v0.61.0', 'upstream_tag_commit':'5975a79b55b40f5cbc61f3ac5e52abe47d9d8bd5',
              'model_sha256':digest(model), 'external_data_sha256':digest(data), 'metadata_sha256':digest(metadata),
              'fixture_manifest_sha256':manifest_sha, 'image_sha256':source_hashes,
              'source_sha256':source_hash, 'provider':session.get_providers(), 'onnxruntime_version':ort.__version__,
              'numpy_version':np.__version__, 'opencv_version':cv2.__version__, 'python_version':platform.python_version(),
              'host':platform.platform(), 'threads':a.threads, 'opencv_threads':cv2.getNumThreads(), 'warmup':a.warmup, 'frames':a.frames, 'session_init_ms':init_ms,
              'geometry':{'source_width':width,'source_height':height,'model_width':320,'model_height':240,'horizontal_scale':width/320.,'vertical_scale':height/240.,'resize':'OpenCV INTER_LINEAR, nonuniform; half input bilinear align_corners=True','color':'RGB8 to BGR float32 [0,1]'},
              'last_output':{'raw_min':float(out.min()),'raw_max':float(out.max()),'rescaled_min':float(restored.min()),'rescaled_max':float(restored.max()),'units_and_sign':'unverified','finite':bool(np.isfinite(restored).all())},
              'samples':samples}
    with a.output.open('x') as output:
        output.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('scope','provider','onnxruntime_version','last_output','samples')},indent=2))


if __name__ == '__main__':
    main()
