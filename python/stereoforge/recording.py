"""Lossless RGB recordings with ordered, checksum-bound calibration metadata."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import zipfile
import numpy as np
from .types import Calibration, StereoFrame


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _integer(value):
    return type(value) is int and value >= 0


def _order(entries):
    last_id = last_time = -1
    for entry in entries:
        fid, ts = entry['frame_id'], entry['timestamp_ns']
        if not _integer(fid) or not _integer(ts) or fid <= last_id or ts <= last_time:
            raise ValueError('frame IDs and timestamps must increase strictly')
        if not isinstance(entry['clock_domain'], str) or not entry['clock_domain']:
            raise ValueError('missing clock domain')
        if not isinstance(entry['sensor_samples'], list):
            raise ValueError('invalid sensor samples')
        last_id, last_time = fid, ts


class Recorder:
    def __init__(self, path, calibration: Calibration, config: dict):
        self.path = Path(path)
        if self.path.exists() and (not self.path.is_dir() or any(self.path.iterdir())):
            raise ValueError('recording directory must be new or empty')
        if not isinstance(config, dict):
            raise ValueError('config must be a JSON object')
        self.calibration = calibration
        self.manifest = dict(schema_version=1, finalized=False,
                             calibration=calibration.to_dict(), calibration_id=calibration.identity,
                             config=json.loads(_json(config)), frames=[])
        self.path.mkdir(parents=True, exist_ok=True)
        self.closed = False
        self._manifest()

    def _manifest(self):
        temp = self.path/'manifest.json.tmp'
        with temp.open('w', encoding='utf-8') as out:
            out.write(_json(self.manifest)); out.flush(); os.fsync(out.fileno())
        os.replace(temp, self.path/'manifest.json')

    def __enter__(self):
        return self

    def write(self, frame: StereoFrame):
        if self.closed:
            raise ValueError('recorder is closed')
        if frame.calibration.identity != self.calibration.identity:
            raise ValueError('calibration mismatch')
        # Revalidate mutable frame fields before accepting bytes.
        StereoFrame(frame.left, frame.right, frame.calibration, frame.timestamp_ns,
                    frame.frame_id, frame.clock_domain, frame.sensor_samples)
        entry = dict(path=f'frame_{len(self.manifest["frames"]):08d}.npz',
                     frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns,
                     clock_domain=frame.clock_domain,
                     sensor_samples=json.loads(_json(frame.sensor_samples)),
                     calibration_id=self.calibration.identity)
        _order(self.manifest['frames'] + [entry])
        target = self.path/entry['path']
        with target.open('xb') as out:
            np.savez_compressed(out, left=frame.left, right=frame.right)
            out.flush(); os.fsync(out.fileno())
        entry['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.manifest['frames'].append(entry)
        self._manifest()

    def close(self):
        if not self.closed:
            self.manifest['finalized'] = True
            self._manifest()
            self.closed = True

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self.closed = True


class Replay:
    def __init__(self, path):
        self.path = Path(path).resolve()
        try:
            self.manifest = json.loads((self.path/'manifest.json').read_text())
            d = self.manifest
            if type(d['schema_version']) is not int or d['schema_version'] != 1 or d['finalized'] is not True:
                raise ValueError('unsupported or nonfinalized recording')
            self.calibration = Calibration.from_dict(d['calibration'])
            if self.calibration.identity != d['calibration_id']:
                raise ValueError('calibration mismatch')
            self.config = d['config']
            if not isinstance(self.config, dict) or not isinstance(d['frames'], list):
                raise ValueError('invalid manifest')
            _json(d)
            _order(d['frames'])
            seen = set()
            for entry in d['frames']:
                name = entry['path']
                if not isinstance(name, str):
                    raise ValueError('invalid frame path')
                relative = PurePosixPath(name)
                target = self.path/name
                if relative.is_absolute() or '..' in relative.parts or '\\' in name or target.resolve().parent != self.path or target.is_symlink() or name in seen:
                    raise ValueError('unsafe or duplicate frame path')
                seen.add(name)
                if entry['calibration_id'] != self.calibration.identity:
                    raise ValueError('frame calibration mismatch')
                if len(entry['sha256']) != 64:
                    raise ValueError('invalid checksum')
        except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise ValueError('invalid recording manifest') from exc

    def __iter__(self):
        for entry in self.manifest['frames']:
            try:
                target = self.path/entry['path']
                payload = target.read_bytes()
                if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                    raise ValueError('frame checksum mismatch')
                # Read the verified bytes, avoiding a second potentially changed file read.
                import io
                with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                    if set(arrays.files) != {'left', 'right'}:
                        raise ValueError('invalid frame arrays')
                    yield StereoFrame(arrays['left'], arrays['right'], self.calibration,
                                      entry['timestamp_ns'], entry['frame_id'],
                                      entry['clock_domain'], entry['sensor_samples'])
            except (OSError, KeyError, EOFError, TypeError, zipfile.BadZipFile) as exc:
                raise ValueError('missing or damaged frame') from exc
