from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def test_first_path_overwrite_matches_clear_add_with_dirty_reused_slots_and_odd_tails():
    rng = random.Random(463)
    for bits in (8, 16):
        components = 32 // bits
        for disparities in (1, 3, 5, 63, 65, 127, 128, 129):
            words = (disparities + components - 1) // components
            slot = [rng.getrandbits(32) for _ in range(11 * words)]
            for _frame in range(3):
                paths = []
                for _ in range(4):
                    additions = []
                    for _pixel in range(11):
                        for word in range(words):
                            value = 0
                            for component in range(components):
                                if word * components + component < disparities:
                                    value |= rng.randrange(0, 32) << (bits * component)
                            additions.append(value)
                    paths.append(additions)
                expected = [sum(values) for values in zip(*paths)]
                slot[:] = paths[0]
                for addition in paths[1:]:
                    slot[:] = [a + b for a, b in zip(slot, addition)]
                assert slot == expected


def test_experimental_first_path_declares_write_and_bypasses_clear():
    source = (ROOT / 'src/daxa_stereo.cpp').read_text()
    shader = (ROOT / 'python/stereoforge/shaders/ops/sgm_overwrite.slang').read_text()
    assert 'PackedSpecializedOverwrite' in source
    assert 'if(word<row_words)words[i*row_words+word]=addition;' in shader
    assert 'if(packing&&(variant!=DaxaSgmVariant::PackedSpecializedOverwrite&&variant!=DaxaSgmVariant::PackedSpecializedOverwriteQuad))add(2,7' in source
    assert 'bool overwrite_first_path=(variant==DaxaSgmVariant::PackedSpecializedOverwrite||variant==DaxaSgmVariant::PackedSpecializedOverwriteQuad)&&path==0' in source
    assert 'if(overwrite_first_path)add(stage,' in source
    assert '{LeftDesc,RightDesc,LeftValid,RightValid},{Penalties},{})' in source
    assert 'line_count(config,dx,dy),{LeftDesc,RightDesc,LeftValid,RightValid},{},{Penalties})' in source
    wrappers = (ROOT / 'python/stereoforge/shaders/daxa/stereo_overwrite.slang').read_text()
    for reference in ('left', 'right'):
        for bits in ('8', '16'):
            assert f'void aggregate_init_{reference}{bits}' in wrappers
    assert 'l.variant==DaxaSgmVariant::PackedSpecialized&&r.variant==l.variant' in source


def test_overwrite_body_is_frozen_baseline_with_only_store_changed():
    baseline = (ROOT / 'python/stereoforge/shaders/ops/sgm_specialized.slang').read_text()
    overwrite = (ROOT / 'python/stereoforge/shaders/ops/sgm_overwrite.slang').read_text()
    start = baseline.index('void aggregate_specialized_body<')
    end = baseline.index('void select_specialized_body<', start)
    expected = baseline[start:end].replace('aggregate_specialized_body', 'aggregate_overwrite_body').replace('words[i*row_words+word]+=addition;', 'words[i*row_words+word]=addition;')
    assert expected.strip() in overwrite
