"""CPU and compile parity for the opt-in released shared reduction."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import numpy as np
import pytest
from stereoforge._sintergpu_reduction import reduction_source

ROOT=Path(__file__).resolve().parents[1]
SHADER=ROOT/'python/stereoforge/shaders/stereo/sgm.slang'
BASELINE=ROOT/'tests/fixtures/sintergpu_sgm_baseline.slang'
HELPER_SHA='b963fc6f1cad8365068c60579f57b55c73c1ad3c3a696a1d3573b0cf627d9388'


def released():
    try: import sintergpu
    except ImportError:
        if os.environ.get('SINTERGPU_TEST_RELEASED') == '1': pytest.fail('released SinterGPU package required')
        pytest.skip('released SinterGPU package unavailable')
    return sintergpu


def spirv_entry_names(blob):
    assert len(blob)%4==0
    words=struct.unpack('<'+'I'*(len(blob)//4),blob)
    assert words[0]==0x07230203
    names=set();i=5
    while i<len(words):
        count=words[i]>>16; opcode=words[i]&0xffff
        assert count and i+count<=len(words)
        if opcode==15:
            assert words[i+1]==5  # GLCompute
            name=b''.join(struct.pack('<I',w) for w in words[i+3:i+count]).split(b'\0',1)[0].decode('utf-8')
            names.add(name)
        i+=count
    return names


def test_default_does_not_load_optional_dependency(monkeypatch):
    import builtins
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        if name=='sintergpu': raise AssertionError('unexpected optional import')
        return original(name,*args,**kwargs)
    monkeypatch.delenv('SINTERGPU_SHARED_REDUCTION',raising=False)
    monkeypatch.setattr(builtins,'__import__',guard)
    source,identity=reduction_source(SHADER)
    assert source.encode()==SHADER.read_bytes()
    assert identity=={'source_sha256':hashlib.sha256(SHADER.read_bytes()).hexdigest(),'shared_reduction':False}


def test_released_bytes_identity_and_fail_closed(monkeypatch):
    pkg=released();monkeypatch.setenv('SINTERGPU_SHARED_REDUCTION','1')
    source,identity=reduction_source(SHADER);helper=pkg.module_path('common.block_reduce').read_bytes()
    assert pkg.__version__=='0.2.11' and hashlib.sha256(helper).hexdigest()==HELPER_SHA
    assert source.encode()==b'#define SINTERGPU_USE_SHARED_REDUCTION 1\n'+helper+b'\n'+SHADER.read_bytes()
    assert identity['source_sha256']==hashlib.sha256(source.encode()).hexdigest()
    monkeypatch.setattr(pkg,'__version__','0.2.10')
    with pytest.raises(RuntimeError,match='0.2.11'): reduction_source(SHADER)
    monkeypatch.setattr(pkg,'__version__','0.2.11')
    class Changed:
        def read_bytes(self): return helper+b' changed'
    monkeypatch.setattr(pkg,'module_path',lambda _: Changed())
    with pytest.raises(RuntimeError,match='hash mismatch'): reduction_source(SHADER)


def test_only_original_reduction_changed():
    before=BASELINE.read_text(); after=SHADER.read_text()
    block='''        for(uint stride=128;stride>0;stride>>=1){
            if(d<stride)reduction[d]=min(reduction[d],reduction[d+stride]);
            GroupMemoryBarrierWithGroupSync();
        }'''
    replacement='#if SINTERGPU_USE_SHARED_REDUCTION\n        SINTERGPU_REDUCE_256(reduction, d, 1, min)\n#else\n'+block+'\n#endif'
    assert before.count(block)==1 and after==before.replace(block,replacement)


@pytest.mark.parametrize('active',[1,2,3,31,32,64,128,255,256])
def test_uint_min_tree_and_padding_match(active):
    rng=np.random.default_rng(active); values=rng.integers(0,1000000,256,dtype=np.uint32);values[active:]=1000000
    original=values.copy(); shared=values.copy()
    for stride in (128,64,32,16,8,4,2,1):
        for lane in range(stride):original[lane]=min(original[lane],original[lane+stride])
        for lane in range(stride):
            for channel in range(1):
                offset=channel*256+lane;shared[offset]=min(shared[offset],shared[offset+stride])
        np.testing.assert_array_equal(original,shared)
    assert shared[0]==values.min()


@pytest.mark.parametrize('version',['2026.12','2026.19'])
def test_baseline_and_shared_compile_with_matching_resource_layout(version,tmp_path,monkeypatch):
    released();key='SINTERGPU_SLANGC_'+version.replace('.','_');compiler=os.environ.get(key)
    if not compiler:
        if key in os.environ: pytest.fail(f'{key} is empty')
        pytest.skip(f'pinned Slang {version} unavailable')
    result=subprocess.run([compiler,'-version'],capture_output=True,text=True,check=True)
    assert (result.stdout+result.stderr).strip()==version
    monkeypatch.setenv('SINTERGPU_SHARED_REDUCTION','1')
    composed,_=reduction_source(SHADER); new=tmp_path/'new.slang';new.write_text(composed)
    layouts=[]
    for index,src in enumerate((BASELINE,new)):
        spv=tmp_path/f'all-{index}.spv';reflection=tmp_path/f'all-{index}.json'
        # Automatic entry discovery avoids the standalone compiler's collision
        # between `-entry select` and its built-in select overloads.
        p=subprocess.run([compiler,str(src),'-target','spirv','-profile','glsl_460','-o',str(spv),'-reflection-json',str(reflection)],capture_output=True,text=True)
        assert p.returncode==0,p.stdout+p.stderr
        data=json.loads(reflection.read_text());entries={e['name']:e for e in data['entryPoints']}
        assert set(entries)=={'census','cost','aggregate','select','depth_convert'}
        assert spirv_entry_names(spv.read_bytes())==set(entries)
        assert entries['aggregate']['threadGroupSize']==[256,1,1]
        assert entries['select']['threadGroupSize']==[256,1,1]
        layouts.append({name:(entry['threadGroupSize'],[(x['name'],x['binding']) for x in entry['bindings'] if x['binding'].get('used')==1]) for name,entry in entries.items()})
    assert layouts[0]==layouts[1]


def test_engine_compiles_pinned_composed_source_without_hidden_device(monkeypatch):
    import sys
    from types import SimpleNamespace
    released();monkeypatch.setenv('SINTERGPU_SHARED_REDUCTION','1')
    seen=[]
    class Module:
        def entry_point(self,name): return name
    class Device:
        info=SimpleNamespace(adapter_name='fake')
        def load_module_from_source(self,name,source): seen.append(source);return Module()
        def link_program(self,modules,entries): return entries
        def create_compute_kernel(self,program): return program
        def create_query_pool(self,*args): return object()
    devices=[]
    def create_device(*args,**kwargs):
        d=Device();devices.append(d);return d
    fake=SimpleNamespace(create_device=create_device,DeviceType=SimpleNamespace(vulkan=object()),QueryType=SimpleNamespace(timestamp=object()))
    monkeypatch.setitem(sys.modules,'slangpy',fake)
    from stereoforge.gpu import VulkanStereo
    engine=VulkanStereo()
    expected,identity=reduction_source(SHADER)
    assert len(devices)==1 and seen==[expected] and engine.shader_source_identity==identity
