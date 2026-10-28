"""Cold analytic-only bridge to the native canonical RGB bundle format.

This copies validated portable records; it does not infer SDK rectification,
qualify physical support, decode pixels, or execute the resident producer.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import stat

import numpy as np

from . import schema

MANIFEST_SCHEMA = 'stereoforge.canonical_record_input.v1'
SUPPLEMENT_SCHEMA = 'stereoforge.canonical_geometry.v1'
SAMPLING_PROFILE = 'canonical_rgb8_box2x2_halfup_gray15_support_and4_halfpixel_v1'
RESERVED = ('native-input.json', 'native-geometry.json', 'native-coarse-calibration.json')
SUPPLEMENT_KEYS = {'schema', 'input_identity', 'record_sha256', 'raw_calibration_sha256',
                   'rectified_calibration_sha256', 'R1', 'R2', 'P1', 'P2',
                   'support_qualification', 'producer'}
MAX_PLANE_BYTES = 512 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(path, *, regular=True):
    path = Path(path)
    _require('..' not in path.parts and '\\' not in str(path), 'unsafe filesystem path')
    path = path.absolute()
    for component in (path, *path.parents):
        _require(not component.is_symlink(), 'symlink component forbidden')
    if regular:
        _require(stat.S_ISREG(path.stat().st_mode), 'regular input file required')
    return path


def _snapshot(path):
    path = _path(path)
    before = path.stat()
    digest = sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(schema.HASH_CHUNK_BYTES):
            digest.update(chunk)
    after = path.stat()
    def identity(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    _require(identity(before) == identity(after), 'input changed while hashing')
    return dict(sha256=digest.hexdigest(), size_bytes=after.st_size), identity(after)


def _strict_json(path):
    document = schema._read_json(_path(path))
    def depth(value, level=0):
        _require(level <= 64, 'JSON nesting exceeds 64')
        if isinstance(value, dict):
            for child in value.values():
                depth(child, level + 1)
        elif isinstance(value, list):
            for child in value:
                depth(child, level + 1)
    depth(document)
    return document


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _matrix(value, rows, columns, name):
    _require(type(value) is list and len(value) == rows and all(type(row) is list and len(row) == columns for row in value), name + ': invalid matrix shape')
    for row in value:
        for element in row:
            schema._number(element, name)
    return np.asarray(value, dtype=np.float64)


def _close(actual, expected):
    return bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)) and
                np.all(np.abs(actual - expected) <= 1e-6 + 1e-5 * np.abs(expected)))


def _rotation(rotation):
    _require(_close(rotation.T @ rotation, np.eye(3)) and _close(np.linalg.det(rotation), 1.), 'improper rectification rotation')


def _geometry(raw, rectified, rotations, projections):
    """Mirror DaxaRectifiedGeometry's setup association, without producing maps."""
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            raw_pose = _matrix(raw['right_to_left'], 4, 4, 'raw pose')
            rect_pose = _matrix(rectified['right_to_left'], 4, 4, 'rectified pose')
            baseline = math.hypot(*raw_pose[:3, 3])
            rect_baseline = math.hypot(*rect_pose[:3, 3])
            _require(baseline > 0 and rect_baseline > 0 and math.isfinite(baseline) and math.isfinite(rect_baseline), 'finite positive baseline required')
            for pose in (raw_pose, rect_pose):
                _rotation(pose[:3, :3])
                _require(bool(np.all(np.abs(pose[3]-[0,0,0,1]) <= 1e-8 + 1e-5*np.abs([0,0,0,1]))), 'invalid affine pose')
            first, second = rotations
            _rotation(first); _rotation(second)
            _require(_close(rect_baseline / baseline, 1.) and _close(rect_pose[:3, :3], np.eye(3)), 'rectified pose differs from raw baseline')
            _require(rect_pose[0,3] > 0 and _close(rect_pose[:3,3]/baseline, np.array([1.,0,0])), 'rectified baseline must align positive X')
            translated = first @ (raw_pose[:3,3]/baseline)
            _require(translated[0] > 0 and _close(translated, np.array([1.,0,0])) and _close(first @ raw_pose[:3,:3] @ second.T, np.eye(3)), 'R1/R2 disagree with raw stereo pose')
            for eye, projection in zip(('left', 'right'), projections):
                camera = rectified[eye]
                translation = -camera['fx']*rect_baseline if eye == 'right' else 0.
                expected = np.array([[camera['fx'],0,camera['cx'],translation], [0,camera['fy'],camera['cy'],0], [0,0,1,0]], dtype=np.float64)
                for row in range(3):
                    for column in range(4):
                        actual, wanted = projection[row,column], expected[row,column]
                        if column == 3 and eye == 'right' and row == 0:
                            _require(math.isfinite(translation) and translation < 0 and actual < 0 and _close(actual/translation,1.), 'P2 translation disagrees with baseline')
                        elif column == 3:
                            _require(actual == 0, 'projection structural translation must be exactly zero')
                        else:
                            _require(_close(actual,wanted), 'projection differs from rectified pinhole')
            offset = rectified['left']['cx'] - rectified['right']['cx']
            limits = np.finfo(np.float32)
            for value in (rectified['left']['fx'], rect_baseline, offset):
                _require(math.isfinite(value) and abs(value) <= limits.max and (value == 0 or abs(value) >= limits.smallest_subnormal), 'metric calibration not representable as float32')
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError('geometry arithmetic failed') from exc


def _supplement(value, record, record_digest, raw, rectified):
    schema._keys(value, SUPPLEMENT_KEYS, 'canonical geometry supplement')
    _require(value['schema'] == SUPPLEMENT_SCHEMA and value['support_qualification'] == 'analytic_known_geometry', 'unsupported analytic geometry qualification')
    schema._text(value['producer'], 'producer')
    bindings = dict(input_identity=record['input_identity'], record_sha256=record_digest,
                    raw_calibration_sha256=record['calibration']['raw']['sha256'],
                    rectified_calibration_sha256=record['calibration']['rectified']['sha256'])
    for key, expected in bindings.items():
        schema._hex(value[key],64,key)
        _require(value[key] == expected, 'supplement identity/hash binding mismatch: ' + key)
    rotations = [_matrix(value[key],3,3,key) for key in ('R1','R2')]
    projections = [_matrix(value[key],3,4,key) for key in ('P1','P2')]
    _geometry(raw,rectified,rotations,projections)
    coarse = deepcopy(rectified)
    for eye in ('left','right'):
        camera = coarse[eye]
        camera['width'] //= 2; camera['height'] //= 2
        camera['fx'] /= 2; camera['fy'] /= 2
        camera['cx'] = (camera['cx']+.5)/2-.5
        camera['cy'] = (camera['cy']+.5)/2-.5
    coarse_projections = []
    for projection in projections:
        derived = projection.copy(); derived[:2] /= 2
        derived[0,2] = (projection[0,2]+.5)/2-.5
        derived[1,2] = (projection[1,2]+.5)/2-.5
        coarse_projections.append(derived)
    _geometry(raw,coarse,rotations,coarse_projections)
    return coarse


def _descriptors(value):
    if type(value) is dict:
        if set(value) in (schema.FILE_KEYS, schema.PLANE_KEYS):
            schema._descriptor(value, plane=set(value) == schema.PLANE_KEYS)
            yield value
        else:
            for child in value.values():
                yield from _descriptors(child)
    elif type(value) is list:
        for child in value:
            yield from _descriptors(child)


def _preflight_record(record):
    # Bound cold image work before complete portable admission hashes/scans files.
    # The schema remains authoritative for every descriptor/content invariant.
    images = record.get('images')
    schema._keys(images, schema.IMAGE_NAMES, 'images')
    total = 0
    for image in images.values():
        _require(type(image) is dict, 'image descriptor required')
        for key in ('width','height'):
            schema._integer(image.get(key),key,1,32766)
        schema._integer(image.get('size_bytes'),'size_bytes')
        total += image['size_bytes']
        _require(total <= MAX_PLANE_BYTES, 'native six-plane byte ceiling exceeded')
    _require(not (images['rectified_left']['width'] % 2 or images['rectified_left']['height'] % 2), 'native coarse grid requires even rectified dimensions')


def write_native_input(record_path, supplement_path, output_dir):
    """Write a fresh byte-preserving analytic bundle and return its manifest.

    Offline stat/hash checks detect observed input mutation, not an atomic
    filesystem snapshot. On failure only this call's exclusive directory is removed.
    """
    record_path, supplement_path = _path(record_path), _path(supplement_path)
    output = _path(output_dir, regular=False)
    _require(output.parent.is_dir(), 'output parent directory required')
    candidate = _strict_json(record_path)
    _preflight_record(candidate)
    record = schema.load_records([record_path])[0]
    _require(candidate == record, 'record changed during portable admission')
    _require(record['source']['kind'] == 'analytic', 'native bridge admits analytic source records only')
    _require(record['calibration']['geometry']['coordinate_frame'] == 'left_optical', 'native metric frame must be left_optical')
    capture = record['timestamps']['capture']
    _require(capture['value_ns'] > 0 and capture['clock_domain'].strip() and capture['clock_domain'] != 'unknown', 'known positive capture timestamp required')
    images = record['images']
    record_desc, record_stat = _snapshot(record_path)
    supplement_desc, supplement_stat = _snapshot(supplement_path)
    supplement = _strict_json(supplement_path)
    raw = _strict_json(record_path.parent/record['calibration']['raw']['path'])
    rectified = _strict_json(record_path.parent/record['calibration']['rectified']['path'])
    coarse = _supplement(supplement,record,record_desc['sha256'],raw,rectified)
    dependencies = {record_path.name: record_desc}
    for descriptor in _descriptors(record):
        name = descriptor['path']; content = {key:descriptor[key] for key in ('sha256','size_bytes')}
        _require(name not in dependencies or dependencies[name] == content, 'conflicting shared file descriptors')
        dependencies[name] = content
    names = sorted(dependencies)
    for name in names:
        _require(not any(name == reserved or name.startswith(reserved+'/') for reserved in RESERVED), 'dependency conflicts with reserved native path')
        _require(not any(other.startswith(name+'/') for other in names if other != name), 'dependency file/directory collision')
    snapshots = {}
    for name, descriptor in dependencies.items():
        path = _path(record_path.parent/name)
        actual, identity = _snapshot(path)
        _require(actual == descriptor, 'dependency hash/size mismatch')
        snapshots[path] = (actual,identity)
    snapshots[record_path] = (record_desc,record_stat)
    _require(supplement_path not in snapshots or snapshots[supplement_path] == (supplement_desc,supplement_stat), 'supplement conflicts with input snapshot')
    snapshots[supplement_path] = (supplement_desc,supplement_stat)
    created = False
    try:
        output.mkdir(exist_ok=False); created = True
        for name, descriptor in dependencies.items():
            destination = output/name; destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(record_path.parent/name,destination)
            _require(_snapshot(destination)[0] == descriptor, 'copied dependency hash/size mismatch')
        shutil.copyfile(supplement_path,output/'native-geometry.json')
        _require(_snapshot(output/'native-geometry.json')[0] == supplement_desc, 'copied supplement changed')
        coarse_bytes = _canonical(coarse)
        (output/'native-coarse-calibration.json').write_bytes(coarse_bytes)
        schema.load_records([output/record_path.name])
        _require(schema.load_records([record_path])[0] == record, 'record changed during conversion')
        for path, expected in snapshots.items():
            _require(_snapshot(path) == expected, 'input changed during conversion')
        manifest = dict(schema=MANIFEST_SCHEMA,
                        source_record=dict(path=record_path.name,**record_desc),
                        geometry_supplement=dict(path='native-geometry.json',**supplement_desc),
                        coarse_calibration=dict(path='native-coarse-calibration.json',sha256=sha256(coarse_bytes).hexdigest(),size_bytes=len(coarse_bytes)),
                        sampling_profile=SAMPLING_PROFILE)
        for key,names in [('original_pixel_identity',('original_left','original_right')),
                          ('rectified_pixel_identity',('rectified_left','rectified_right','support_left','support_right'))]:
            payload = {name:{field:value for field,value in images[name].items() if field != 'path'} for name in names}
            manifest[key] = sha256(_canonical(payload)).hexdigest()
        manifest_path = output/'native-input.json'
        manifest_path.write_bytes(_canonical(manifest))
        return manifest_path
    except BaseException:
        if created:
            shutil.rmtree(output)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record',type=Path)
    parser.add_argument('supplement',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args(argv)
    print(write_native_input(args.record,args.supplement,args.output))


if __name__ == '__main__':
    main()
