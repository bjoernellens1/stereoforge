"""Fail-closed offline decisions for a deliberately bounded variant list."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
import compare_kernel_variants as variants


def identity():
    return dict(physical_gpu='uuid', driver='driver', daxa='revision+patch', slang='2026.12',
        compiler_sha256='a'*64, shader_sha256='b'*64, specialization=json.dumps(dict(entrypoints=['metadata_fused'],workgroup=[64,1,1],readback_memory=dict(policy='first-coherent',type_index=2,property_flags=6))),
        source_width=1280, source_height=720, target_width=1280, target_height=720,
        disparities=0, paths=0, precision='uint32-pairs', quality_profile='metadata-exact-v1', subgroup=64)


@pytest.mark.parametrize('field', list(identity()))
def test_each_execution_identity_component_invalidates_cached_choice(field):
    before=identity(); after=copy.deepcopy(before)
    after[field] = ('d'*64 if field.endswith('sha256') else after[field]+1 if isinstance(after[field], int) else after[field]+'changed')
    with pytest.raises(ValueError, match='stale'):
        variants.validate_cache(before, after)


def test_missing_unknown_identity_fields_fail_closed():
    for field in identity():
        broken=identity(); del broken[field]
        with pytest.raises(ValueError): variants.validate_cache(broken, broken)
    broken=identity(); broken['unexpected']='x'
    with pytest.raises(ValueError): variants.validate_cache(broken,broken)
    broken=identity(); broken['driver']=''
    with pytest.raises(ValueError): variants.validate_cache(broken,broken)


@pytest.mark.parametrize('subgroup', [0,16,48,128])
def test_unsupported_subgroup_rejected(subgroup):
    with pytest.raises(ValueError): variants.validate_subgroup(subgroup,32,64,32)


def test_supported_subgroup_requires_compute_stage():
    variants.validate_subgroup(64,32,64,32)
    with pytest.raises(ValueError): variants.validate_subgroup(64,32,64,0)


def test_unknown_variant_and_changed_contract_rejected():
    config=variants.load_config()
    assert len(config['variants']) == 2
    variants.validate_config(config)
    changed=copy.deepcopy(config); changed['variants'][0]['name']='unknown'
    with pytest.raises(ValueError): variants.validate_config(changed)
    changed=copy.deepcopy(config); changed['variants'][0]['workgroup']=[8,8,1]
    with pytest.raises(ValueError): variants.validate_config(changed)


@pytest.mark.parametrize('field,value', [('numeric_parity',False),('validation_errors',1),
    ('validation_warnings',1),('gpu_ms_samples',[float('nan')]*30),
    ('transfer_inclusive_ms_samples',[-1]*30),('gpu_ms_samples',[1]*29),('warmup',2),
    ('allocated_bytes',2**40)])
def test_failed_numeric_quality_timing_and_resource_gates_reject(field,value):
    report=dict(numeric_parity=True,validation_errors=0,validation_warnings=0,warmup=3,
        gpu_ms_samples=[1.0]*30,transfer_inclusive_ms_samples=[2.0]*30,allocated_bytes=4096,
        resident_ms_samples=[1.0]*30,readback_memory_policy='first-coherent',readback_memory_type_index=2,readback_memory_property_flags=6,
        resident_cpu_phase_ms=[dict(input_stage=0,enqueue=0.2,completion_wait=0.7,readback_copy=0,query=0.1,total=1)]*30,
        transfer_cpu_phase_ms=[dict(input_stage=0.1,enqueue=0.2,completion_wait=0.7,readback_copy=0.9,query=0.1,total=2)]*30)
    variants.validate_measurement(report,variants.load_config()['quality'])
    report[field]=value
    with pytest.raises(ValueError): variants.validate_measurement(report,variants.load_config()['quality'])


def test_approval_requires_exact_validated_measurement_and_identity():
    key=identity(); report_sha='c'*64
    approval=dict(variant='metadata-fused-64',identity=key,measurement_sha256=report_sha,measurement_file='validated.json',reviewer='manual reviewer')
    variants.validate_approval(approval,'metadata-fused-64',key,report_sha)
    for field in ('reviewer','measurement_sha256','measurement_file','variant'):
        changed=copy.deepcopy(approval); changed[field]=''
        with pytest.raises(ValueError): variants.validate_approval(changed,'metadata-fused-64',key,report_sha)


def test_variants_only_compare_same_hardware_driver_shape_quality_profile():
    before=identity(); after=identity()
    after['shader_sha256']='d'*64
    specialization=json.loads(after['specialization']);specialization['entrypoints']=['metadata_ids','metadata_ages']
    after['specialization']=json.dumps(specialization)
    variants.validate_comparable(before,after)
    after['driver']='updated driver'
    with pytest.raises(ValueError,match='different execution profiles'):
        variants.validate_comparable(before,after)


def test_changed_benchmark_bytes_invalidate_approved_execution_identity(tmp_path):
    """Control-flow/fixture changes matter even with identical SPIR-V and libs."""
    import shutil
    binary=tmp_path/'benchmark';shutil.copyfile('/bin/true',binary);binary.chmod(0o755)
    compiler=tmp_path/'slangc';compiler.write_bytes(b'compiler identity fixture')
    shader=tmp_path/'daxa-metadata-metadata_fused.spv';shader.write_bytes(b'compiled SPIR-V identity fixture')
    source=tmp_path/'daxa'
    for patch in (variants.ROOT/'cmake/patches').glob('daxa*.patch'):
        for line in patch.read_text().splitlines():
            if line.startswith('+++ b/'):
                path=source/line[6:];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'applied source fixture')
    device=dict(device_uuid='physical uuid',driver_uuid='driver uuid',driver_name='driver',driver_info='version',
        driver_version=1,vulkan_api_version=1,daxa_commit='pinned',slang_version='2026.12',
        slangc_sha256=variants.sha256(compiler),subgroups=dict(default_size=64,min_size=32,max_size=64,supported_stages=32),memory_types=[dict(index=2,flags=6)])
    report=dict(device_report=device,shader_files=[str(shader)],width=1280,height=720,readback_memory_policy='first-coherent',readback_memory_type_index=2,readback_memory_property_flags=6)
    variant=variants.load_config()['variants'][0];quality=variants.load_config()['quality']
    before=variants.execution_identity(report,variant,binary,quality,compiler,source)
    with binary.open('ab') as out: out.write(b'changed fixture generator or native control flow')
    after=variants.execution_identity(report,variant,binary,quality,compiler,source)
    with pytest.raises(ValueError,match='stale'):
        variants.validate_cache(before,after)


def test_source_provenance_uses_binary_compiled_path_and_rejects_unrelated_source(tmp_path):
    import shlex
    actual=tmp_path/'actual-patched-source';actual.mkdir()
    unrelated=tmp_path/'unpatched-archive';unrelated.mkdir()
    binary=tmp_path/'source-path-query'
    binary.write_text('#!/bin/sh\nprintf "%s\\n" '+shlex.quote(str(actual))+'\n')
    binary.chmod(0o755)
    assert variants.compiled_daxa_source(binary)==actual
    assert variants.compiled_daxa_source(binary,actual)==actual
    with pytest.raises(ValueError,match='compiled Daxa source'):
        variants.compiled_daxa_source(binary,unrelated)
