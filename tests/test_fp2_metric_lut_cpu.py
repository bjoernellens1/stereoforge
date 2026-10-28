"""Pre-rounded numerator LUT checked against NumPy and exact rational distances."""
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT/'build/daxa/fp2_selection_fixture'


def bits(x):
    return int(np.float32(x).view(np.uint32))


def nearest_even_check(actual, numerator, denominator):
    exact = Fraction.from_float(float(numerator)) / Fraction.from_float(float(denominator))
    value = Fraction.from_float(float(actual))
    difference = abs(value-exact)
    for toward in (np.float32(0), np.float32(np.inf)):
        neighbor = np.nextafter(actual, toward)
        if not np.isfinite(neighbor):
            continue
        neighbor_difference = abs(Fraction.from_float(float(neighbor))-exact)
        assert difference <= neighbor_difference
        if difference == neighbor_difference:
            assert not (bits(actual) & 1)


@pytest.mark.skipif(not BINARY.exists(), reason='optional selection fixture not built')
@pytest.mark.parametrize('numerator,offset', [
    (np.float32(63.125), np.float32(.25)),
    (np.float32(1), np.nextafter(np.float32(2), np.float32(0))),
    (np.finfo(np.float32).tiny, np.float32(-127)),
    (np.finfo(np.float32).max, np.float32(-.5)),
    (np.float32(1), -np.finfo(np.float32).max),
    (np.float32(0.1), np.float32(126.99999)),
])
def test_all_integer_disparities(numerator, offset):
    table = np.array(json.loads(subprocess.check_output([str(BINARY), '--lut', str(bits(numerator)), str(bits(offset))], text=True)), dtype=np.uint32)
    assert table.shape == (128, 2)
    for d in range(128):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            denominator = np.float32(np.float32(d)-offset)
            depth = np.float32(numerator/denominator)
        valid = d>0 and denominator>0 and np.isfinite(depth) and depth>0
        assert int(table[d, 1]) == valid
        assert int(table[d, 0]) == (bits(depth) if valid else 0x7fc00000)
        if valid and np.isfinite(denominator):
            nearest_even_check(depth, numerator, denominator)


@pytest.mark.skipif(not BINARY.exists(), reason='optional selection fixture not built')
@pytest.mark.parametrize('numerator,offset', [(0, 0), (1, 0), (0x7f800000, 0), (0x3f800000, 1), (0x3f800000, 0x7fc00000)])
def test_unsupported_inputs_fail_closed(numerator, offset):
    result = subprocess.run([str(BINARY), '--lut', str(numerator), str(offset)], capture_output=True)
    assert result.returncode != 0


def test_original_precision_and_binary32_input_routes_are_not_equivalent():
    original = 1 + 2**-24
    original_product = np.float32(original * original)
    rounded_product = np.float32(np.float32(original) * np.float32(original))
    assert bits(original_product) != bits(rounded_product)


def spirv_opcodes(data):
    import struct
    if len(data) < 20 or len(data) % 4:
        raise ValueError('malformed SPIR-V size')
    words = struct.unpack('<%dI' % (len(data)//4), data)
    if words[0] != 0x07230203:
        raise ValueError('malformed SPIR-V magic')
    ops = []
    pos = 5
    while pos < len(words):
        count, opcode = words[pos] >> 16, words[pos] & 0xffff
        if not count or pos+count > len(words):
            raise ValueError('malformed SPIR-V instruction')
        ops.append(opcode)
        pos += count
    return ops


def require_no_fdiv(data):
    if 136 in spirv_opcodes(data):  # SPIR-V OpFDiv
        raise ValueError('metric SPIR-V contains OpFDiv')


@pytest.mark.skipif(not BINARY.exists(), reason='optional selection fixture not built')
def test_metric_spirv_has_no_fdiv_and_mutation_fails():
    import struct
    identity = json.loads(subprocess.check_output([str(BINARY), '--identity'], text=True))
    data = (Path(identity['shader_dir'])/'fp2-select-metric.spv').read_bytes()
    require_no_fdiv(data)
    with pytest.raises(ValueError, match='contains OpFDiv'):
        require_no_fdiv(data+struct.pack('<5I', (5<<16)|136, 1, 2, 3, 4))
    with pytest.raises(ValueError, match='malformed'):
        require_no_fdiv(data+b'X')


@pytest.mark.skipif(not BINARY.exists(), reason='optional selection fixture not built')
def test_mxcsr_modes_rejected_and_restored_in_subprocess():
    output = subprocess.check_output([str(BINARY), '--test-mxcsr'], text=True)
    assert 'MXCSR rejection passed' in output
