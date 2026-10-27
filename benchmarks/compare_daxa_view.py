#!/usr/bin/env python3
"""Separate-process matched resident RHI/Daxa POC; exact eleven-plane equality.

Diagnostic output dumps occur after GPU measurement. This static fixture result
cannot qualify capture, full-pipeline throughput, or physical geometry.
"""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
from compare_native_resident import PLANES, compare, fixture_manifest, sha256

ROOT = Path(__file__).resolve().parents[1]


def dependencies(binary, daxa):
    text = subprocess.check_output(['ldd', str(binary)], text=True)
    forbidden = ['libsgl', 'slangpy', 'libpython']
    if daxa:
        forbidden += ['slang-rhi', 'libslang-compiler']
    if any(name in text.lower() for name in forbidden):
        raise ValueError(f'Forbidden runtime dependency: {binary}')
    if not daxa and 'libslang-compiler' not in text:
        raise ValueError('RHI compiler identity missing')
    files = {}
    for line in text.splitlines():
        parts = line.split('=>')
        if len(parts) == 2:
            path = Path(parts[1].strip().split()[0])
            if path.is_file():
                files[str(path)] = sha256(path)
    return dict(ldd=text.splitlines(), sha256=files)


DIAGNOSTIC_PATTERNS = {
    'vuid': re.compile(r'\bVUID-[A-Za-z0-9_-]+', re.I),
    'validation_error': re.compile(r'\bValidation\s*Error\b', re.I),
    'warning': re.compile(r'\bWarning\b', re.I),
    'daxa_assertion': re.compile(r'Daxa[^\n]*assert|\bassertion\s+(?:failed|failure)|IMPOSSIBLE CASE', re.I),
    'synchronization_hazard': re.compile(r'SYNC-HAZARD', re.I),
}
VALIDATION_ENVIRONMENT = ('SF_DAXA_VALIDATION', 'VK_INSTANCE_LAYERS', 'VK_LAYER_VALIDATE_SYNC',
                          'VK_LAYER_ENABLES', 'VK_LOADER_LAYERS_ENABLE')


def parse_diagnostics(text):
    matches = []
    for line_number, line in enumerate(text.splitlines(), 1):
        categories = [name for name, pattern in DIAGNOSTIC_PATTERNS.items() if pattern.search(line)]
        if categories:
            matches.append(dict(line=line_number, categories=categories, text=line))
    return dict(line_count=len(matches), matches=matches)


def validation_requested(explicit, environment):
    return bool(explicit or 'SF_DAXA_VALIDATION' in environment or
                'validation' in environment.get('VK_INSTANCE_LAYERS', '').lower() or
                'validation' in environment.get('VK_LOADER_LAYERS_ENABLE', '').lower() or
                environment.get('VK_LAYER_VALIDATE_SYNC', '').lower() in ('1', 'true') or
                'VALIDATION' in environment.get('VK_LAYER_ENABLES', '').upper())


def run_child(command, environment, output, case):
    stdout_file, stderr_file = output/f'{case}.stdout', output/f'{case}.stderr'
    try:
        result = subprocess.run(command, env=environment, capture_output=True, text=True, errors='replace')
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except OSError as error:
        code, stdout, stderr = None, '', str(error)
    stdout_file.write_text(stdout); stderr_file.write_text(stderr)
    return dict(returncode=code, stdout_file=str(stdout_file), stderr_file=str(stderr_file),
                stdout_diagnostics=parse_diagnostics(stdout), stderr_diagnostics=parse_diagnostics(stderr))


def passed(numeric_parity, children, validation_required):
    return bool(numeric_parity and all(child['returncode'] == 0 and not child.get('report_error') and
        (not validation_required or (child['stdout_diagnostics']['line_count'] == 0 and
         child['stderr_diagnostics']['line_count'] == 0 and child.get('reported_validation_errors', 0) == 0 and
         child.get('reported_validation_warnings', 0) == 0)) for child in children.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--daxa-binary', type=Path, default=ROOT/'build/daxa/daxa_view_benchmark')
    p.add_argument('--rhi-binary', type=Path, default=ROOT/'build/standalone/native_view_benchmark')
    p.add_argument('--fixture', type=Path, default=ROOT/'artifacts/fixture-720p')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=30)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--validation', action='store_true')
    a = p.parse_args()
    if a.frames < 30 or a.warmup < 3:
        p.error('matched qualification requires at least 30 samples and 3 warmups')
    a.fixture = a.fixture.resolve(); a.output = a.output.resolve()
    binaries = dict(rhi=a.rhi_binary.resolve(), daxa=a.daxa_binary.resolve())
    fixture = fixture_manifest(a.fixture)
    identities = {name: dict(binary_sha256=sha256(binary), dependencies=dependencies(binary, name=='daxa'))
                  for name, binary in binaries.items()}
    shader_directory = Path(subprocess.check_output([str(binaries['daxa']), '--shader-directory'], text=True).strip()).resolve()
    spirv = sorted(shader_directory.glob('daxa-view-*.spv'))
    if len(spirv) != 9:
        raise ValueError('Expected all nine offline Daxa view SPIR-V files')
    cache = (shader_directory/'CMakeCache.txt').read_text()
    sdk_rows = [line.split('=', 1)[1] for line in cache.splitlines()
                if line.startswith('FETCHCONTENT_SOURCE_DIR_SF_DAXA_SLANG:')]
    compiler = Path(sdk_rows[0])/'bin/slangc' if sdk_rows and sdk_rows[0] else shader_directory/'_deps/sf_daxa_slang-src/bin/slangc'
    identities['daxa']['offline_compiler'] = dict(path=str(compiler), sha256=sha256(compiler))
    identities['daxa']['offline_spirv_sha256'] = {str(path): sha256(path) for path in spirv}
    commands = {}
    for mode in ('strict', 'preview'):
        for runtime, binary in binaries.items():
            case = f'{mode}-{runtime}'
            commands[case] = [str(binary), '--fixture', str(a.fixture), '--mode', mode,
                '--frames', str(a.frames), '--warmup', str(a.warmup), '--output', str(a.output/f'{case}.json'),
                '--dump-prefix', str(a.output/case)]
            if runtime == 'rhi':
                commands[case] += ['--input', 'resident', '--readback', 'gpu']
    sources = ['src/daxa_view.cpp', 'include/stereoforge/daxa_view.hpp',
        'include/stereoforge/daxa_view_parameters.hpp', 'apps/daxa_view_benchmark.cpp',
        'python/stereoforge/shaders/ops/projection.slang', 'python/stereoforge/shaders/daxa/view.slang',
        'python/stereoforge/shaders/fusion/project.slang', 'src/geometry.cpp',
        'apps/native_view_benchmark.cpp', 'benchmarks/compare_daxa_view.py', 'cmake/Daxa.cmake', 'CMakeLists.txt']
    patches = sorted((ROOT/'cmake/patches').glob('daxa*.patch'))
    sources += [str(path.relative_to(ROOT)) for path in patches]
    daxa_rows = [line.split('=', 1)[1] for line in cache.splitlines()
                 if line.startswith('FETCHCONTENT_SOURCE_DIR_SF_DAXA:')]
    daxa_source = Path(daxa_rows[0]) if daxa_rows and daxa_rows[0] else shader_directory/'_deps/sf_daxa-src'
    patched_files = {line[6:] for patch in patches for line in patch.read_text().splitlines()
                     if line.startswith('+++ b/')}
    identities['daxa']['applied_dependency_source_sha256'] = {
        str(daxa_source/name): sha256(daxa_source/name) for name in sorted(patched_files)}
    validation_active = validation_requested(a.validation, os.environ)
    evidence = dict(schema=2, fixture=fixture, identities=identities, commands=commands,
        source_sha256={name: sha256(ROOT/name) for name in sources},
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        frames=a.frames, warmup=a.warmup, validation=a.validation, validation_required=validation_active,
        inherited_validation_environment={name: os.environ[name] for name in VALIDATION_ENVIRONMENT if name in os.environ},
        validation_request='Daxa required layer guard plus both subprocesses VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation when --validation; preserve stderr and inspect actual layer execution.',
        equality_contract='Exact byte equality of all eleven planes; no float or RGB tolerances.',
        precision_contract='Matching float32 half-pixel profile; float64 camera boundary is not a migration waiver.',
        gpu_timing_contract='Daxa timestamp first scheduled pre through last scheduled post task callback, including eight dispatches and internal barriers, excluding graph boundary barriers. RHI report has CPU submit/completion only; do not compare it as GPU timestamp time.',
        memory_contract='Daxa requested buffer bytes include retained staging/parameter storage; actual driver/VMA committed peak is not measured.',
        validation_coverage='Core/synchronization validation requested independently. Physical buffer address shader accesses may not be fully inferred by synchronization validation.',
        scope='Static resident fixture only, no capture/stereo/temporal/full-pipeline qualification.')
    if a.verify_only:
        print(json.dumps(evidence, indent=2)); return
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Output directory must be new or empty')
    a.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update(PYTHONHOME='/nonexistent', PYTHONPATH='')
    if a.validation:
        env['SF_DAXA_VALIDATION'] = '1'
        env['VK_INSTANCE_LAYERS'] = 'VK_LAYER_KHRONOS_validation'
    reports, children = {}, {}
    for case, command in commands.items():
        child = children[case] = run_child(command, env, a.output, case)
        try:
            reports[case] = json.loads((a.output/f'{case}.json').read_text())
            child['reported_validation_errors'] = reports[case].get('validation_errors', 0)
            child['reported_validation_warnings'] = reports[case].get('validation_warnings', 0)
        except (OSError, ValueError) as error:
            child['report_error'] = str(error)
    count = fixture['target_width']*fixture['target_height']
    comparisons = {}
    for mode in ('strict', 'preview'):
        try:
            result = compare(a.output/f'{mode}-rhi', a.output/f'{mode}-daxa', count)
            result['passed'] = result['passed'] and all(result['planes'][name]['bytes_equal'] for name in PLANES)
        except (OSError, ValueError) as error:
            result = dict(passed=False, comparison_error=str(error))
        comparisons[mode] = result
    numeric_parity = all(x['passed'] for x in comparisons.values())
    evidence.update(reports=reports, children=children, comparisons=comparisons,
        numeric_parity=numeric_parity,
        validation_clean=all(child['stdout_diagnostics']['line_count'] == 0 and child['stderr_diagnostics']['line_count'] == 0 and
            child.get('reported_validation_errors', 0) == 0 and child.get('reported_validation_warnings', 0) == 0 for child in children.values()),
        passed=passed(numeric_parity, children, validation_active))
    evidence['files'] = {path.name: sha256(path) for path in a.output.iterdir() if path.is_file()}
    (a.output/'comparison.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output), numeric_parity=numeric_parity, validation_clean=evidence['validation_clean'], passed=evidence['passed']), indent=2))
    if not evidence['passed']: raise SystemExit('Daxa/RHI parity, child execution, or required validation failed; inspect comparison.json and per-case stdout/stderr')


if __name__ == '__main__': main()
