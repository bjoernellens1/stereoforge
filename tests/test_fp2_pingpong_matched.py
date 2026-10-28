import json
from pathlib import Path
import pytest
from benchmarks import fp2_pingpong_matched as m


def test_results_require_proof_binding_and_timing_contract(tmp_path):
    files=[tmp_path/x for x in ('harness','binary','tiny')]
    for f in files: f.write_bytes(f.name.encode())
    harness,binary,tiny=files
    proof={'host':{'binary_sha256':m.sha(binary)},'tiny':{'binary_sha256':m.sha(tiny)},
           'harness_sha256':m.sha(harness),'manifest_sha256':m.MANIFEST,'baseline_sha256':m.BASELINE}
    path=tmp_path/'proof';path.write_text(json.dumps(proof))
    result={**proof,'scope':'pingpong timing','sample_count':100,'warmup':10,
            'input_scope':'one repeated preuploaded analytic frame; one in-flight slot'}
    m.check_result(result,'new',path,harness,binary,tiny)
    for key,value in [('scope','parallel timing'),('sample_count',99),('warmup',9),('manifest_sha256','bad')]:
        changed={**result,key:value}
        with pytest.raises(ValueError):m.check_result(changed,'new',path,harness,binary,tiny)
    binary.write_bytes(b'mutation')
    with pytest.raises(ValueError,match='binary mismatch'):m.check_result(result,'new',path,harness,binary,tiny)


def test_loop_and_lock_guards():
    source=Path(m.__file__).read_text()
    assert "('old','new','old')" in source
    assert 'fcntl.LOCK_EX|fcntl.LOCK_NB' in source
    assert 'start_new_session=True' in source and 'os.killpg' in source
    assert "'synchronization_validation':'requested, not independently attested'" in source
    assert '128 lanes and tree reduction per path pixel' in source
