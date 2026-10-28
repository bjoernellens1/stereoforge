from pathlib import Path
import itertools

ROOT = Path(__file__).resolve().parents[1]

def test_candidate_has_explicit_independent_eye_quad_entrypoints():
    source = (ROOT / 'python/stereoforge/shaders/daxa/stereo_overwrite_quad.slang').read_text()
    assert 'reference_sum<Components,Bits,RightReference>' in source
    assert 'RightReference?parameters.right_desc_address:parameters.left_desc_address' in source
    assert 'RightReference?parameters.left_desc_address:parameters.right_desc_address' in source
    assert source.count('WaveReadLaneAt(') == 3
    assert source.rindex('WaveReadLaneAt(') < source.index('if(lane<stride)')
    assert source.index('return;') > source.rindex('WaveReadLaneAt(')
    for entry in ('select_overwrite_quad_left8', 'select_overwrite_quad_left16', 'select_overwrite_quad_right8', 'select_overwrite_quad_right16'):
        assert entry in source


def test_four_lane_reduction_ties_and_tail():
    # Exhaust all ternary cost patterns, including tied minima and lane tails.
    for count in range(1, 9):
        for values in itertools.product(range(3), repeat=count):
            lanes = []
            for lane in range(4):
                ordered = sorted((values[d], d) for d in range(lane, count, 4))
                lanes.append((ordered[0] if ordered else (2**32-1, 2**32-1), ordered[1][0] if len(ordered) > 1 else 2**32-1))
            for stride in (2, 1):
                old = lanes[:]
                for lane in range(stride):
                    best, second = old[lane]
                    other, other_second = old[lane + stride]
                    lanes[lane] = (other, min(other_second, best[0])) if other < best else (best, min(second, other[0]))
            ordered = sorted((v, d) for d, v in enumerate(values))
            assert lanes[0][0] == ordered[0]
            assert lanes[0][1] == (ordered[1][0] if count > 1 else 2**32-1)


def test_candidate_identity_covers_all_executed_source_dependencies():
    source = (ROOT / 'src/daxa_stereo.cpp').read_text()
    identity = source.split('std::string DaxaStereo::overwrite_quad_shader_identity()', 1)[1].split('}', 1)[0]
    cmake = (ROOT / 'cmake/DaxaStereo.cmake').read_text()
    for name, filename in (
        ('OVERWRITE_QUAD', 'shaders/daxa/stereo_overwrite_quad.slang'),
        ('OVERWRITE_WRAPPER', 'shaders/daxa/stereo_overwrite.slang'),
        ('OVERWRITE_OPS', 'shaders/ops/sgm_overwrite.slang'),
        ('WRAPPER', 'shaders/daxa/stereo.slang'),
        ('PARAMETERS', 'shaders/daxa/stereo_parameters.slang'),
        ('SPECIALIZED', 'shaders/ops/sgm_specialized.slang'),
        ('OPS', 'shaders/ops/sgm.slang'),
    ):
        macro = f'SF_DAXA_STEREO_{name}_SHA256'
        assert macro in identity
        assert f'{filename}" {macro})' in cmake
        assert f'{filename}' in cmake


def test_reference_cost_direction_masks_and_subpixel_neighbors():
    source = (ROOT / 'python/stereoforge/shaders/ops/sgm_specialized.slang').read_text()
    assert 'x>=d&&reference_valid' in source
    assert 'right_valid[i-d]!=0' in source
    assert 'd<width-x&&reference_valid' in source
    assert 'target_valid[i+d]!=0' in source
    quad = (ROOT / 'python/stereoforge/shaders/daxa/stereo_overwrite_quad.slang').read_text()
    assert 'best>0&&best+1<disparities' in quad
    assert 'width,x,i,best-1' in quad and 'width,x,i,best+1' in quad
    for width in range(1, 7):
        for x in range(width):
            for d in range(8):
                for reference_valid, target_valid in itertools.product((False, True), repeat=2):
                    for right in (False, True):
                        target = x + d if right else x - d
                        supported = reference_valid and target_valid and (d < width - x if right else x >= d)
                        assert supported == (reference_valid and target_valid and 0 <= target < width)
                        if supported:
                            assert target == (x + d if right else x - d)
