"""CPU-only comparison and identity tests for integrated 720p diagnostic."""
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import fp2_integrated_diagnostic as diag
from benchmarks.fp2_720p_prepare import sha
from stereoforge.refinement import lr_consistency

ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'build/daxa/fp2_integrated_fixture'


@pytest.mark.skipif(not BINARY.exists(),reason='optional integrated fixture not built')
def test_binary_source_and_spirv_identity():
    assert diag.verify_binary(BINARY)['binary_sha256']==sha(BINARY)


def test_mismatched_words_fail():
    diag.equal_bits(np.array([1,2],np.uint32),np.array([1,2],np.uint32),'sample')
    with pytest.raises(ValueError,match='sample'):
        diag.equal_bits(np.array([1,3],np.uint32),np.array([1,2],np.uint32),'sample')


def test_small_complete_output_compare_and_mutations(tmp_path,monkeypatch):
    shape=(2,1,6);monkeypatch.setattr(diag,'SHAPE',shape)
    prep=tmp_path/'prepared';prep.mkdir();output=tmp_path/'output';output.mkdir();artifact=tmp_path/'artifact';artifact.mkdir()
    n=6;raw=np.zeros((*shape,7),np.uint32);raw[...,0]=2;raw[...,1]=1;raw[...,2]=3;raw[...,3]=8
    raw[...,4]=np.float32(2).view(np.uint32);raw[...,5]=np.float32(2/3).view(np.uint32);raw[...,6]=1
    disparity=np.full(shape,2,np.float32);valid=np.ones(shape,bool);lr=lr_consistency(disparity,valid,1.)
    metric=lr.copy();metric[0,0,4]=False
    depth=np.where(metric,np.float32(3.5),np.float32(np.nan)).astype(np.float32)
    source=0x12345678abcdef01
    obs=np.zeros((*shape,9),np.uint32);obs[...,0]=lr;obs[...,1]=depth.view(np.uint32);obs[...,2]=np.where(metric,raw[...,5],0);obs[...,3]=metric;obs[...,4]=metric;obs[...,5]=np.where(metric,source&0xffffffff,0xffffffff);obs[...,6]=np.where(metric,source>>32,0xffffffff)
    for eye in (0,1):
        directory=prep/('eye'+str(eye));directory.mkdir()
        for name,arr in {'winner':np.full(n,2,np.int32),'best':np.ones(n,np.uint32),'second':np.full(n,3,np.uint32),'count':np.full(n,8,np.uint32),'lo':np.zeros(n,np.uint32),'hi':np.full(n,8,np.uint32)}.items():
            arr.tofile(directory/name)
    a=artifact/'compact-fixed.npz';b=artifact/'compact-fixed.npz.selection.npz';c=artifact/'compact-fixed.json'
    np.savez(a,lo=np.zeros(shape,np.uint32),hi=np.full(shape,8,np.uint32))
    np.savez(b,**{'raw_disparity.f32':disparity,'raw_validity.u32':valid,'validity.u32':lr,'metric_validity.u32':metric,'depth.f32':depth})
    c.write_text('{}');monkeypatch.setitem(diag.ARTIFACTS,'compact-fixed',(sha(a),sha(b),sha(c)))
    raw.tofile(output/'raw.bin');obs.tofile(output/'obs.bin')
    result=diag.compare_outputs(output,prep,artifact,source)
    assert result['raw_valid']==[6,6]
    for col in (0,1,2,3,4,5,6,7,8):
        mutated=obs.copy();mutated[0,0,4,col]^=1;mutated.tofile(output/'obs.bin')
        with pytest.raises(ValueError):diag.compare_outputs(output,prep,artifact,source)
    obs.tofile(output/'obs.bin');raw[0,0,3,5]^=1;raw.tofile(output/'raw.bin')
    with pytest.raises(ValueError,match='confidence'):diag.compare_outputs(output,prep,artifact,source)


@pytest.mark.parametrize('raw,lr,metric', [([0,1],[1,1],[1,1]),([1,1],[1,0],[1,1]),([1,1],[1,1],[0,1]),([1],[1,1],[1,1])])
def test_empty_eye_population_rejected(raw,lr,metric):
    with pytest.raises(ValueError,match='empty'):
        diag.require_nonempty(raw,lr,metric)
