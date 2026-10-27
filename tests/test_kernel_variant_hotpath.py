"""CPU-only regression guards for the standalone metadata microbenchmark.

These intentionally guard performance contracts that CPU execution cannot observe
without opening Vulkan. Mutation probes keep the checks from silently going stale.
"""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]


def body(source, signature):
    start = source.index('{', source.index(signature))
    depth = 0
    # Ignore braces in comments and literals while locating a C++/Slang body.
    tokens = re.finditer(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\'|[{}]', source[start:])
    for token in tokens:
        if token.group() == '{':
            depth += 1
        elif token.group() == '}':
            depth -= 1
            if not depth:
                return source[start + 1:start + token.start()]
    raise AssertionError('unterminated function body')


def check_sample_hotpath(source):
    sample = body(source, 'Sample run(')
    forbidden = r'\b(?:allocate|create_buffer|create_memory|create_compute_pipeline|create_command_recorder|make_unique|make_shared|resize|reserve|get_query_results|wait_idle)\s*\('
    assert not re.search(forbidden, sample), 'per-sample allocation/resource construction/global wait'
    assert sample.count('->execute(') == 1, 'one retained graph execution per sample'
    assert sample.count('.wait_on_submit(') == 1, 'one completion wait per sample'
    assert '.collect_garbage(' in sample, 'retire completed command resources each sample'
    assert re.search(r'if\s*\(copy\)\s*stage\(f\)', sample), 'resident sample must not upload CPU planes'
    assert re.search(r'if\s*\(copy\)\s*unpack\(f\)', sample), 'resident sample must not read back CPU planes'
    assert 'daxa_timeline_query_pool_query_results(' in sample, 'retrieve timestamps into caller-owned storage'
    assert sample.index('daxa_timeline_query_pool_query_results(') < sample.index('auto end=Clock::now()'), 'include timestamp retrieval in per-frame CPU wall cost'


def test_variant_sample_keeps_graphs_resources_and_timestamp_storage_retained():
    check_sample_hotpath((ROOT / 'apps/daxa_variant_benchmark.cpp').read_text())


@pytest.mark.parametrize('regression', ['d.wait_idle();', 'auto x=queries.get_query_results(0,2);', 'allocate(8,false);'])
def test_hotpath_guard_detects_inserted_global_wait_and_allocations(regression):
    source = (ROOT / 'apps/daxa_variant_benchmark.cpp').read_text()
    marker = 'Sample run(bool copy,Fixture& f){'
    assert marker in source
    with pytest.raises(AssertionError, match='per-sample'):
        check_sample_hotpath(source.replace(marker, marker + regression, 1))


def check_readback_policy_isolation(source):
    constructor = body(source, 'explicit Engine(')
    ordinary = body(source, 'daxa::BufferId allocate(')
    readback = body(source, 'daxa::BufferId allocate_readback(')
    assert constructor.count('allocate_readback(') == 1
    assert re.search(r'readback\s*=\s*allocate_readback\(n\*16,o.readback_memory\)', constructor), 'policy changes readback only'
    assert 'select_readback_memory_type' not in ordinary and 'readback_memory' not in ordinary, 'uploads and parameters must retain baseline memory policy'
    assert 'context.coherent_readback_requirements(info)' in ordinary
    pin = re.search(r'requirements.memory_type_bits\s*=\s*1u\s*<<\s*readback_type_index', readback)
    assert pin and pin.start() < readback.index('d.create_memory('), 'allocation must pin the reported compatible memory type'
    assert 'select_readback_memory_type(' not in body(source, 'Sample run('), 'memory selection is setup only'


def test_memory_experiment_changes_only_readback_and_pins_its_reported_type():
    check_readback_policy_isolation((ROOT / 'apps/daxa_variant_benchmark.cpp').read_text())


def test_readback_policy_guard_detects_unpinned_allocator_memory_choice():
    source = (ROOT / 'apps/daxa_variant_benchmark.cpp').read_text()
    changed = source.replace('requirements.memory_type_bits=1u<<readback_type_index;', '')
    assert changed != source
    with pytest.raises(AssertionError, match='pin the reported'):
        check_readback_policy_isolation(changed)


def check_contiguous_phase_boundaries(source):
    sample = body(source, 'Sample run(')
    checkpoints = [sample.index('stage(f)'), sample.index('auto staged=Clock::now()'),
                   sample.index('->execute('), sample.index('auto submitted=Clock::now()'),
                   sample.index('.wait_on_submit('), sample.index('auto completed=Clock::now()'),
                   sample.index('unpack(f)'), sample.index('auto copied=Clock::now()'),
                   sample.index('daxa_timeline_query_pool_query_results('), sample.index('auto end=Clock::now()')]
    assert checkpoints == sorted(checkpoints), 'CPU phase timestamps must follow the actual stage/submit/wait/copy/query operations'
    intervals = re.findall(r'elapsed\((\w+),(\w+)\)', sample[sample.index('return {'):])
    assert intervals[-1] == ('start', 'end')
    assert intervals[:-1] == list(zip(('start', 'staged', 'submitted', 'completed', 'copied'),
                                      ('staged', 'submitted', 'completed', 'copied', 'end'))), 'reported phases must partition the complete wall interval'


def test_cpu_phase_timing_covers_each_operation_without_gaps_or_overlap():
    check_contiguous_phase_boundaries((ROOT / 'apps/daxa_variant_benchmark.cpp').read_text())


def test_phase_guard_detects_completion_timestamp_before_the_wait():
    source = (ROOT / 'apps/daxa_variant_benchmark.cpp').read_text()
    marker = 'auto completed=Clock::now();'
    assert marker in source
    # initialize() also waits; target only the sample body for this regression.
    sample = body(source, 'Sample run(')
    changed_sample = sample.replace(marker, '', 1).replace('d.wait_on_submit(', marker + 'd.wait_on_submit(', 1)
    changed = source.replace(sample, changed_sample, 1)
    with pytest.raises(AssertionError, match='timestamps must follow'):
        check_contiguous_phase_boundaries(changed)


def check_split_dependencies(source):
    graph = body(source, 'std::unique_ptr<daxa::TaskGraph> graph(')
    compute = body(graph, 'for(unsigned pass=')
    assert not re.search(r'\.reads\(out\[', compute), 'split metadata planes must not declare a fabricated output read/barrier'
    assert 'visited++==pass' not in compute, 'timestamp boundaries follow actual scheduling, not shader/pass index'


def test_split_baseline_has_only_algorithmic_input_and_output_hazards():
    check_split_dependencies((ROOT / 'apps/daxa_variant_benchmark.cpp').read_text())


def test_split_dependency_guard_detects_artificial_baseline_barrier():
    source = (ROOT / 'apps/daxa_variant_benchmark.cpp').read_text()
    changed = source.replace('task.reads(par)', 'task.reads(out[0]);task.reads(par)', 1)
    assert changed != source
    with pytest.raises(AssertionError, match='fabricated'):
        check_split_dependencies(changed)


def check_shader_bounds(source, ops):
    predicate = body(ops, 'bool metadata_has_owner(')
    assert re.search(r'owner_index\s*<\s*source_count', predicate), 'physical-pointer owners need an upper bound'
    for entry in ('metadata_fused', 'metadata_ids', 'metadata_ages'):
        kernel = body(source, 'void ' + entry + '(')
        assert kernel.index('if(t>=p.count)return') < kernel.index('owners[t]'), 'tail lanes must return before pointer access'
        assert 'metadata_has_owner(validity[t],index,p.count)' in kernel
        assert kernel.index('if(present)') < min((kernel.index(load) for load in ('source_ids[index]', 'source_ages[index]') if load in kernel))


def test_all_metadata_entrypoints_bound_tail_and_source_owner_pointer_loads():
    check_shader_bounds((ROOT / 'python/stereoforge/shaders/daxa/metadata.slang').read_text(),
                        (ROOT / 'python/stereoforge/shaders/ops/stereo_output.slang').read_text())


def test_bounds_guard_detects_removed_owner_upper_bound():
    source = (ROOT / 'python/stereoforge/shaders/daxa/metadata.slang').read_text()
    ops = (ROOT / 'python/stereoforge/shaders/ops/stereo_output.slang').read_text()
    changed = ops.replace(' && owner_index < source_count', '')
    assert changed != ops
    with pytest.raises(AssertionError, match='upper bound'):
        check_shader_bounds(source, changed)


def test_benchmark_checks_both_static_fixture_batches_against_cpu_bytes():
    source = (ROOT / 'apps/daxa_variant_benchmark.cpp').read_text()
    main = body(source, 'int main(')
    resident = main.index('for(auto& s:resident)')
    readback = main.index('e.read_outputs(f)')
    resident_oracle = main.index('f.parity()', readback)
    transfer = main.index('for(auto& s:transfer)')
    transfer_oracle = main.index('f.parity()', transfer)
    assert resident < readback < resident_oracle < transfer < transfer_oracle
    assert 'memcmp(result_ids.data(),expected_ids.data()' in source
    assert 'memcmp(result_ages.data(),expected_ages.data()' in source
    assert 'f.parity()' not in body(source, 'Sample run('), 'oracle is batch diagnostic, not timed frame cost'
