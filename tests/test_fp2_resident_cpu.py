"""CPU-only source equivalence and identity contracts for resident FP2."""
import re
from pathlib import Path

import pytest

from benchmarks.fp2_resident_diagnostic import verify_binary, verify_tiny

ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'build/daxa/fp2_resident_fixture'
TINY=ROOT/'build/daxa/fp2_resident_tiny'


def no_path_source(source):
    source=source.replace('// Isolated correctness prototype. One invocation owns one scanline.', '// Isolated no-path recurrence; one invocation owns one scanline.')
    source=source.replace('uint* cost=(uint*)p.costs; uint* sums=(uint*)p.sums; uint* paths=(uint*)p.paths;', 'uint* cost=(uint*)p.costs; uint* sums=(uint*)p.sums;')
    source=source.replace('    uint64_t allocated=offsets[uint64_t(p.width)*p.height];\n','')
    source=source.replace('                paths[uint64_t(pc.pass)*allocated+j]=v;\n','')
    return source


def test_no_path_specialization_only_removes_path_store():
    baseline=(ROOT/'python/stereoforge/shaders/daxa/fp2_banded.slang').read_text()
    no_path=(ROOT/'python/stereoforge/shaders/daxa/fp2_banded_no_path.slang').read_text()
    assert no_path_source(baseline)==no_path
    assert 'p.paths' not in no_path
    assert no_path_source(baseline.replace('v+=best-minimum;', 'v+=best-minimum+1;'))!=no_path


def test_sum_reset_covers_full_capacity_and_not_paths():
    source=(ROOT/'python/stereoforge/shaders/daxa/fp2_sum_reset.slang').read_text()
    compact=re.sub(r'\s+','',source)
    assert 'for(uinti=tid.x;i<pc.words;i+=stride)((uint*)pc.sums)[i]=0xffffffffu;' in compact
    assert 'pc.paths' not in compact
    assert 'i<pc.words;i+=stride' not in compact.replace('i<pc.words;i+=stride','i<pc.words-1;i+=stride')


@pytest.mark.skipif(not BINARY.exists() or not TINY.exists(),reason='optional resident fixtures not built')
def test_compiled_identities():
    assert verify_binary(BINARY)['binary_sha256']
    assert verify_tiny(TINY)['binary_sha256']


def test_parity_proof_identity_and_output_mutations(tmp_path):
    import json
    from benchmarks.fp2_720p_prepare import sha
    from benchmarks.fp2_resident_diagnostic import check_proof
    baseline={'gpu_outputs':{'raw_sha256':'r','obs_sha256':'o'}}
    proof={'scope':'resident parity','manifest_sha256':'m','host':{'id':'h'},'tiny':{'id':'t'},
           'harness_sha256':'s','baseline_sha256':'b','tiny_output':'contracts passed',
           'gpu_outputs':{'raw_sha256':'r','obs_sha256':'o'}}
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof));digest=sha(path)
    check_proof(path,digest,'m',{'id':'h'},{'id':'t'},'s','b',baseline)
    with pytest.raises(ValueError):check_proof(path,'0'*64,'m',{'id':'h'},{'id':'t'},'s','b',baseline)
    with pytest.raises(ValueError):check_proof(path,digest,'m',{'id':'changed'},{'id':'t'},'s','b',baseline)
    proof['gpu_outputs']['obs_sha256']='changed';path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):check_proof(path,sha(path),'m',{'id':'h'},{'id':'t'},'s','b',baseline)
