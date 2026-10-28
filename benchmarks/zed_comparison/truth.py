"""Strict independent truth admission, with one-frame cold CPU materialization.

Registration declarations describe coordinates already registered to the left
rectified grid. No interpolation, sensor registration or prediction is performed.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

from . import schema as s

TRUTH_SCHEMA = 'stereoforge.zed_comparison.truth.v1'
MAX_FRAME_BYTES = 512 * 1024 * 1024
MASK_NAMES = ('geometric_support', 'known_negative', 'visible', 'occluded',
              'mixed', 'boundary_edges')


def _plane(desc, geometry, formats):
    # Portable plane shape/path rules, extended only locally for truth float64.
    s._keys(desc, s.PLANE_KEYS, 'truth plane')
    s._descriptor({k: desc[k] for k in s.FILE_KEYS})
    for key in ('width', 'height', 'stride_bytes'):
        s._integer(desc[key], key, 1)
    s._require(desc['format'] in formats, 'unsupported truth plane format')
    bpp = {'f32_le_m': 4, 'f64_le_m': 8, 'u8_mask': 1}[desc['format']]
    s._require(desc['stride_bytes'] >= desc['width'] * bpp, 'truth stride smaller than pixel row')
    s._require(desc['size_bytes'] == desc['stride_bytes'] * desc['height'], 'wrong truth plane extent')
    s._require((desc['width'], desc['height']) == (geometry['width'], geometry['height']),
               'truth plane geometry mismatch')


def _rigid(value):
    s._require(type(value) is list and len(value) == 4 and
               all(type(row) is list and len(row) == 4 for row in value), 'registration transform must be 4x4')
    for row in value:
        for x in row:
            s._number(x, 'registration transform element')
    t = np.asarray(value, dtype=np.float64)
    r = t[:3, :3]
    s._require(bool(np.allclose(t[3], [0, 0, 0, 1], rtol=0, atol=1e-12)) and
               bool(np.all(np.abs(r) <= 1 + 1e-6)) and
               bool(np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-6)) and
               bool(np.isclose(np.linalg.det(r), 1, rtol=0, atol=1e-6)),
               'registration transform must be proper rigid')


def _registration(value):
    s._keys(value, {'method', 'transform_to_left', 'max_error_m', 'justification'}, 'registration')
    s._text(value['method'], 'registration method')
    _rigid(value['transform_to_left'])
    s._number(value['max_error_m'], 'registration max_error_m', minimum=0)
    s._text(value['justification'], 'registration justification')


def _timestamp_uncertainty(value):
    s._keys(value, {'max_error_ns', 'justification'}, 'timestamp uncertainty')
    s._integer(value['max_error_ns'], 'timestamp max_error_ns')
    s._text(value['justification'], 'timestamp justification')


def _immutable(array, dtype):
    """Detached immutable bytes keep caller mutation away from admitted content."""
    a = np.asarray(array, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=dtype).reshape(a.shape)


class TruthBundle:
    """Metadata and file fingerprints only; no dataset-sized array cache."""
    def __init__(self, document, root, hashes, fingerprints, calibrations, frame_paths):
        self.metadata = deepcopy(document)
        self._frames = {f['input_identity']: deepcopy(f) for f in document['frames']}
        self._root = root
        self._hashes = hashes
        self._fingerprints = fingerprints
        self._calibrations = calibrations
        self._frame_paths = frame_paths

    def recheck(self):
        """Recheck all admitted fingerprints once at a whole-run boundary."""
        with ExitStack() as stack:
            s._Files(self._root, stack, self._hashes, self._fingerprints).recheck()

    @contextmanager
    def frame(self, input_identity):
        s._require(input_identity in self._frames, 'missing independent truth for input')
        f = self._frames[input_identity]
        with ExitStack() as stack:
            paths = self._frame_paths[input_identity]
            files = s._Files(self._root, stack,
                             {p: self._hashes[p] for p in paths},
                             {p: self._fingerprints[p] for p in paths})
            files.recheck()
            def plane(desc, mask=False):
                path = files.check(desc)
                mapping = np.memmap(path, mode='r', dtype='u1')
                stack.callback(mapping._mmap.close)
                dtype = {'f32_le_m': '<f4', 'f64_le_m': '<f8', 'u8_mask': 'u1'}[desc['format']]
                a = np.ndarray((desc['height'], desc['width']), dtype=dtype, buffer=mapping,
                               strides=(desc['stride_bytes'], np.dtype(dtype).itemsize))
                if mask:
                    s._require(bool(np.all((a == 0) | (a == 1))), 'truth mask must be binary')
                return _immutable(a, bool if mask else np.float64)
            sample = plane(f['masks']['sample'], True)
            depth = plane(f['depth'])
            s._require(bool(np.all(np.isfinite(depth[sample]) & (depth[sample] > 0))),
                       'sampled metric truth must be finite positive')
            if f['depth_convention'] == 'ray_range':
                c = self._calibrations[input_identity]['left']
                with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                    x = (np.arange(c['width'], dtype=np.float64) - c['cx']) / c['fx']
                    y = (np.arange(c['height'], dtype=np.float64) - c['cy']) / c['fy']
                    factor = np.hypot(1, np.hypot(y[:, None], x[None, :]))
                    converted = depth / factor
                s._require(bool(np.all(np.isfinite(factor))) and
                           bool(np.all(np.isfinite(converted[sample]) & (converted[sample] > 0))),
                           'nonfinite or unsupported range conversion')
                depth = _immutable(converted, np.float64)
            out = dict(metadata=deepcopy(f), geometry=deepcopy(f['geometry']), depth_m=depth,
                       sample=sample, registration=deepcopy(f['registration']),
                       timestamp_uncertainty=deepcopy(f['timestamp_uncertainty']))
            def optional(value):
                return plane(value['value'], True) if value['status'] == 'available' else None
            for name in MASK_NAMES:
                out[name] = optional(f['masks'][name])
            if out['known_negative'] is not None:
                s._require(not bool(np.any(out['known_negative'] & sample)),
                           'known-negative truth intersects sample')
            out['regions'] = {name: optional(value) for name, value in f['regions'].items()}
            files.recheck()
            try:
                yield out
            finally:
                files.recheck()


def load_truth(truth, records, *, root):
    """Validate strict truth and exact portable-input bindings under explicit root.

    Records may contain multiple implementations of an identical input. Portable
    record file admission belongs to their loader/coordinator; metadata is checked
    here independently and truth file contents are always checked under root.
    """
    s._require(root is not None, 'explicit truth root required')
    root = Path(root).resolve()
    document = s._read_json(truth) if isinstance(truth, (str, Path)) else deepcopy(truth)
    s._json_value(document)
    s._require(len(s._canonical(document).encode()) <= s.MAX_JSON_BYTES, 'truth JSON exceeds bounded size')
    s._keys(document, {'schema', 'independent', 'provenance', 'frames'}, 'truth')
    s._require(document['schema'] == TRUTH_SCHEMA and document['independent'] is True,
               'independent truth schema required')
    s._require(type(records) is list, 'records must be a list')
    inputs = {}
    for record in records:
        s._validate(record)
        identity = record['input_identity']
        if identity in inputs:
            old = inputs[identity]
            s._require(all(record[k] == old[k] for k in ('partition', 'scene_id', 'sequence_id')),
                       'duplicate input has conflicting partition or scene/sequence binding')
        inputs[identity] = record
    hashes, fingerprints, calibrations, frame_paths = {}, {}, {}, {}
    with ExitStack() as stack:
        files = s._Files(root, stack, hashes, fingerprints)
        def file(desc):
            s._descriptor(desc)
            files.check(desc)
        p = document['provenance']
        s._keys(p, {'kind', 'artifact', 'generator'}, 'truth provenance')
        s._require(p['kind'] in ('analytic', 'measured'), 'unsupported independent truth kind')
        file(p['artifact'])
        generator = s._availability(p['generator'], file, 'truth generator')
        provenance_paths = set(fingerprints)
        s._require(type(document['frames']) is list, 'truth frames must be a list')
        seen = set()
        # Metadata/budget validation precedes any plane hashing/materialization.
        planes_by_frame = []
        for f in document['frames']:
            s._keys(f, {'input_identity', 'source_id', 'capture', 'partition', 'calibration',
                        'rectified_pixel_identity', 'geometry', 'depth_convention', 'depth', 'masks',
                        'regions', 'registration', 'timestamp_uncertainty'}, 'truth frame')
            s._hex(f['input_identity'], 64, 'input_identity')
            identity = f['input_identity']
            s._require(identity not in seen, 'duplicate truth input')
            seen.add(identity)
            s._require(identity in inputs, 'truth input missing portable record binding')
            r = inputs[identity]
            s._require(f['source_id'] == r['source_id'] and type(f['source_id']) is int,
                       'truth source ID mismatch')
            s._timestamp(f['capture'])
            s._require(f['capture'] == r['timestamps']['capture'], 'truth capture mismatch')
            s._require(f['partition'] == r['partition'], 'truth partition mismatch')
            s._require(f['geometry'] == r['calibration']['geometry'], 'truth geometry mismatch')
            for key in ('width', 'height'):
                s._integer(f['geometry'][key], 'truth geometry ' + key, 1)
            s._normalized_geometry(f['geometry'])
            s._descriptor(f['calibration'])
            s._require(s._content(f['calibration']) == s._content(r['calibration']['rectified']),
                       'truth calibration content mismatch')
            payload = {k: s._content(r['images'][k]) for k in
                       ('rectified_left', 'rectified_right', 'support_left', 'support_right')}
            s._require(f['rectified_pixel_identity'] == sha256(s._canonical(payload).encode()).hexdigest(),
                       'truth rectified pixel identity mismatch')
            s._require(f['depth_convention'] in ('axial_z', 'ray_range'), 'unsupported truth depth convention')
            planes = []
            def plane(desc, formats=('u8_mask',)):
                _plane(desc, f['geometry'], formats)
                planes.append(desc)
            plane(f['depth'], ('f32_le_m', 'f64_le_m'))
            s._keys(f['masks'], {'sample', *MASK_NAMES}, 'truth masks')
            plane(f['masks']['sample'])
            for name in MASK_NAMES:
                s._availability(f['masks'][name], plane, name)
            s._require(type(f['regions']) is dict, 'truth regions must be an object')
            for name, value in f['regions'].items():
                s._text(name, 'truth region name')
                s._require(name != 'all', 'all is reserved for implicit full region')
                s._availability(value, plane, 'truth region')
            s._require(sum(d['size_bytes'] for d in planes) <= MAX_FRAME_BYTES,
                       'truth frame exceeds 512 MiB declared plane extent budget')
            s._availability(f['registration'], _registration, 'registration')
            s._availability(f['timestamp_uncertainty'], _timestamp_uncertainty, 'timestamp uncertainty')
            zero = ((f['registration']['status'] == 'available' and f['registration']['value']['max_error_m'] == 0) or
                    (f['timestamp_uncertainty']['status'] == 'available' and f['timestamp_uncertainty']['value']['max_error_ns'] == 0))
            s._require(not (p['kind'] == 'analytic' and zero) or generator,
                       'analytic zero bounds require bound generator artifact')
            planes_by_frame.append((f, planes))
        for f, planes in planes_by_frame:
            files.check(f['calibration'], json_object=True)
            snapshot = files.json[f['calibration']['path']]
            s._calibration_snapshot(snapshot, 'rectified', f['geometry'], inputs[f['input_identity']]['images'])
            calibrations[f['input_identity']] = snapshot
            for desc in planes:
                files.check(desc)
            frame_paths[f['input_identity']] = provenance_paths | {f['calibration']['path']} | {d['path'] for d in planes}
        files.recheck()
    bundle = TruthBundle(document, root, hashes, fingerprints, calibrations, frame_paths)
    # Admission inspects binary masks and finite sampled truth one frame at a time.
    for identity in seen:
        with bundle.frame(identity):
            pass
    return bundle
