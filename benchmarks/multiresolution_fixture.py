#!/usr/bin/env python3
"""Export a smaller rectified grid from a frozen native fixture, offline only.

Original capture/calibration/RGB bytes and calibrated rectification rotations are
retained. This exporter does not estimate calibration or qualify GPU D96 output.
"""
import argparse
import copy
from fractions import Fraction
import math
from pathlib import Path
import shutil

import compare_daxa_preprocess as fixture
from compare_daxa_stereo import create_output, sha256, strict_json

PIXEL_CENTER_CONVENTION = 'u_target+0.5=s*(u_donor+0.5); v_target+0.5=s*(v_donor+0.5)'


def nominal_near_depth(fx, baseline_m, cx_offset, disparities):
    """Nominal maximum-disparity metric Z, not an accuracy or visibility bound."""
    if (type(disparities) is not int or disparities < 1 or
            not all(math.isfinite(v) for v in (fx, baseline_m, cx_offset)) or fx <= 0 or baseline_m <= 0):
        raise ValueError('invalid calibrated nominal-depth parameters')
    denominator = (disparities - 1) - cx_offset
    if denominator <= 0:
        return None
    result = fx * baseline_m / denominator
    if not math.isfinite(result) or result <= 0:
        raise ValueError('nominal near depth outside finite positive domain')
    return result


def _no_symlinks(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('symlink fixture inputs/destinations are not allowed')
    return path.resolve()


def _inventory(path, manifest):
    root = path.parent
    result = {path.name: sha256(path)}
    entries = [manifest['capture'], manifest['calibration']['original_json'], manifest['calibration']['rectified_json']]
    entries += list(manifest['maps'].values()) + list(manifest['expected'].values())
    for entry in entries:
        file = _no_symlinks(fixture.fixture_file(root, entry))
        result[entry['file']] = sha256(file)
    return dict(sorted(result.items()))


def export_profile(source_manifest: Path, output: Path, width: int, height: int) -> Path:
    import cv2
    import numpy as np
    from stereoforge.types import Calibration, CameraModel

    source_manifest = _no_symlinks(source_manifest)
    if not source_manifest.is_file():
        raise ValueError('missing donor manifest')
    output = _no_symlinks(output)
    if output.exists():
        raise ValueError('output must be a fresh nonexisting directory')
    if output.is_relative_to(source_manifest.parent):
        raise ValueError('output cannot be inside the immutable donor fixture')
    donor, _ = fixture.validate_fixture(source_manifest)
    before = _inventory(source_manifest, donor)
    donor_width, donor_height = donor['rectified']['width'], donor['rectified']['height']
    if (type(width) is not int or type(height) is not int or min(width, height) <= 0 or
            width > donor_width or height > donor_height or width * donor_height != height * donor_width):
        raise ValueError('target requires a positive equal-xy downscale of the donor rectified grid')
    cv_binary = next(Path(cv2.__file__).parent.glob('*.so'))
    if cv2.__version__ != '4.14.0' or sha256(cv_binary) != donor['provenance']['cv2_binary_sha256']:
        raise ValueError('actual OpenCV differs from frozen donor profile')
    root = source_manifest.parent
    original = Calibration.from_dict(strict_json(fixture.fixture_file(root, donor['calibration']['original_json']).read_text()))
    rectified = Calibration.from_dict(strict_json(fixture.fixture_file(root, donor['calibration']['rectified_json']).read_text()))
    ratio = Fraction(width, donor_width)
    scale = float(ratio)
    affine = np.array([[scale, 0., (scale - 1.) / 2.], [0., scale, (scale - 1.) / 2.], [0., 0., 1.]], np.float64)
    projections = [affine @ np.asarray(donor['calibration'][key], np.float64) for key in ('P1', 'P2')]
    cameras = [CameraModel(width, height, float(p[0, 0]), float(p[1, 1]), float(p[0, 2]), float(p[1, 2])) for p in projections]
    scaled = Calibration(*cameras, rectified.right_to_left, rectified.serial)
    maps, images = {}, {}
    for eye, camera, rotation, projection in zip(('left', 'right'), (original.left, original.right),
                                                (donor['calibration']['R1'], donor['calibration']['R2']), projections):
        distortion = np.asarray(camera.distortion, np.float64) if camera.distortion else np.zeros(5)
        mx, my = cv2.initUndistortRectifyMap(camera.K, distortion, np.asarray(rotation, np.float64), projection,
                                           (width, height), cv2.CV_32FC1)
        maps[eye + '_x'], maps[eye + '_y'] = mx.astype('<f4'), my.astype('<f4')
        rgb = np.frombuffer(fixture.fixture_file(root, donor['expected']['original_' + eye + '_rgb']).read_bytes(), np.uint8)
        rgb = rgb.reshape(camera.height, camera.width, 3)
        rect = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR)
        images['rect_' + eye + '_rgb'] = rect
        images[eye + '_luma'] = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY).astype('<u4')
        images[eye + '_support'] = fixture.support_mask(mx, my, camera.width, camera.height)
    # Validate all inputs and compute outputs before creating the fresh directory.
    if before != _inventory(source_manifest, donor):
        raise ValueError('donor changed during profile computation')
    create_output(output)

    def save(name, raw):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        return dict(file=name, sha256=sha256(path), bytes=len(raw))

    def copy_entry(entry, name):
        source = fixture.fixture_file(root, entry)
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as stream, path.open('xb') as target:
            shutil.copyfileobj(stream, target)
        if sha256(path) != entry['sha256']:
            raise ValueError('copied donor bytes changed')
        return dict(file=name, sha256=entry['sha256'], bytes=path.stat().st_size)

    manifest = copy.deepcopy(donor)
    manifest['rectified'] = dict(width=width, height=height)
    capture = copy_entry(donor['capture'], 'capture/capture.sfp')
    capture.pop('bytes')  # Preserve the existing capture descriptor's schema.
    capture['record_ordinal'] = donor['capture']['record_ordinal']
    manifest['capture'] = capture
    manifest['calibration'].update(original_json=copy_entry(donor['calibration']['original_json'], 'original-calibration.json'),
                                   rectified_json=save('rectified-calibration.json', fixture.canonical(scaled.to_dict())),
                                   rectified_identity=scaled.identity, P1=projections[0].tolist(), P2=projections[1].tolist())
    manifest['maps'] = {key: save('maps/' + key + '.f32', value.tobytes()) for key, value in maps.items()}
    manifest['expected'] = {}
    for key, suffix in fixture.PLANES.items():
        manifest['expected'][key] = (copy_entry(donor['expected'][key], 'expected/' + suffix) if key.startswith('original_')
                                     else save('expected/' + suffix, images[key].tobytes()))
    after = _inventory(source_manifest, donor)
    if before != after:
        raise ValueError('donor changed during profile export')
    manifest['provenance']['exporter_sha256'] = sha256(__file__)
    manifest['provenance']['multiresolution_profile'] = dict(
        schema='stereoforge.multiresolution_profile.v1', donor_manifest=dict(file=source_manifest.name, sha256=before[source_manifest.name]),
        donor_files_sha256=before, donor_manifest_sha256_before=before[source_manifest.name], donor_manifest_sha256_after=after[source_manifest.name],
        donor_rectified=donor['rectified'], target_rectified=manifest['rectified'],
        scale=dict(numerator=ratio.numerator, denominator=ratio.denominator), A=affine.tolist(),
        pixel_center_convention=PIXEL_CENTER_CONVENTION, exporter_sha256=sha256(__file__),
        sampling='unchanged frozen bilinear remap; no antialiasing added',
        geometry='P_target=A@P_donor; original models, R1/R2 and rectified metric pose retained',
        nominal_near_depth_m_by_disparities={str(d): nominal_near_depth(scaled.left.fx, scaled.baseline_m,
            scaled.left.cx-scaled.right.cx, d) for d in (64, 96, 128)},
        qualification='offline fixture only; native D96 scalar/packed core+sync GPU gate pending')
    result = output / 'manifest.json'
    result.write_bytes(fixture.canonical(manifest))
    fixture.validate_fixture(result)
    if before != _inventory(source_manifest, donor):
        raise ValueError('donor changed before export completed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    args = parser.parse_args()
    try:
        print(export_profile(args.source_manifest, args.output, args.width, args.height))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
