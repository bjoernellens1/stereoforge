"""Decode the actual compiled SPIR-V and reject missing reduction instructions."""
import struct
from pathlib import Path
import pytest
from benchmarks.fp2_subgroup_spirv_check import validate

SPV=Path(__file__).resolve().parents[1]/'build/daxa/fp2-subgroup.spv'


def remove_instruction(data, opcode, operand=None):
    words=list(struct.unpack('<%dI'%(len(data)//4),data)); i=5
    while i<len(words):
        n=words[i]>>16
        if (words[i]&0xffff)==opcode and (operand is None or words[i+1]==operand):
            del words[i:i+n];return struct.pack('<%dI'%len(words),*words)
        i+=n
    raise AssertionError('mutation target missing')


def test_compiled_shader_structure():
    assert validate(SPV.read_bytes()) == {'subgroup_elect':1,'subgroup_umin':1,'workgroup_atomic_umin':1,'workgroup_barriers':5}


@pytest.mark.parametrize('opcode,operand',[(17,61),(17,63),(333,None),(354,None),(237,None),(224,None)])
def test_removed_required_instruction_rejected(opcode,operand):
    with pytest.raises(ValueError):validate(remove_instruction(SPV.read_bytes(),opcode,operand))


def test_truncation_rejected():
    with pytest.raises(ValueError):validate(SPV.read_bytes()[:-1])
