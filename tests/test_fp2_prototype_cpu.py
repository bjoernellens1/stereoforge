"""CPU-only prototype build identity and contract tests; never invokes Vulkan."""
import hashlib,json,subprocess
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'build/daxa/fp2_prototype_fixture'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(identity,shader,host,spirv,compiler):
    for key,path in [('shader_source_sha256',shader),('host_source_sha256',host),('spirv_sha256',spirv),('compiler_sha256',compiler)]:
        if digest(path)!=identity[key]:raise ValueError('identity mismatch '+key)

def test_identity_mutation_rejected(tmp_path):
    files=[]
    for name in ('shader','host','spirv','compiler'):
        p=tmp_path/name;p.write_bytes(name.encode());files.append(p)
    identity={k:digest(p) for k,p in zip(('shader_source_sha256','host_source_sha256','spirv_sha256','compiler_sha256'),files)}
    verify(identity,*files);files[2].write_bytes(b'mutated')
    with pytest.raises(ValueError,match='spirv_sha256'):verify(identity,*files)

@pytest.mark.skipif(not BINARY.exists(),reason='optional FP2 prototype fixture not built')
def test_compiled_identity_and_cpu_contract():
    identity=json.loads(subprocess.check_output([str(BINARY),'--identity'],text=True))
    shader=ROOT/'python/stereoforge/shaders/daxa/fp2_banded.slang';host=ROOT/'apps/fp2_prototype_fixture.cpp'
    spirv=Path(identity['spirv_path']);compiler=Path(identity['compiler_path'])
    verify(identity,shader,host,spirv,compiler)
    assert 'contracts passed' in subprocess.check_output([str(BINARY),'--cpu-only'],text=True)

def validate_pipeline_size(source):
    assert 'bytes%4==0' in source and 'std::vector<uint32_t> words(bytes/4)' in source
    assert '.byte_code_size=uint32_t(words.size())' in source
    assert '.byte_code_size=uint32_t(bytes)' not in source

def test_shader_binary_size_is_words_and_mutation_rejected():
    source=(ROOT/'apps/fp2_prototype_fixture.cpp').read_text();validate_pipeline_size(source)
    with pytest.raises(AssertionError):validate_pipeline_size(source.replace('.byte_code_size=uint32_t(words.size())','.byte_code_size=uint32_t(bytes)'))
