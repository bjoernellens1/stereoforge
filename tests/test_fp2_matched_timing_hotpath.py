"""CPU-only process/mutation guards for matched timing orchestration."""
import json
import signal
import subprocess
from pathlib import Path
import pytest
from benchmarks import fp2_matched_timing as m


def guard(src):
    for token in ("fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)", "enumerate(('old','new','old'))",
                  'start_new_session=True', 'status=process.wait(timeout=300)',
                  'os.killpg(process.pid,signal.SIGKILL);process.wait();raise',
                  "'--time','--proof',str(proof),'--proof-sha256',PROOFS[variant]",
                  'if sha(proof)!=PROOFS[variant]', 'identities.items()',
                  'check_result(result,variant,proof,harness,binary,tiny)',
                  "sha(destination/'output/samples.csv')!=result['samples_sha256']",
                  "result.get('sample_count')!=100", "result.get('warmup')!=10",
                  "'one repeated preuploaded analytic frame; one in-flight slot'",
                  "'128 lanes and tree reduction per path pixel including inactive candidates'"):
        if token not in src:raise ValueError('orchestrator invariant '+token)
    start=src.index('args=[str(ROOT/');end=src.index('identities=',start)
    args=src[start:end]
    if "'--parity'" in args or "'--gpu'" in args:raise ValueError('timing invokes readback path')


def test_source():guard(Path(m.__file__).read_text())


@pytest.mark.parametrize('old,new',[
 ('fcntl.LOCK_EX|fcntl.LOCK_NB','fcntl.LOCK_SH'),
 ("enumerate(('old','new','old'))","enumerate(('new','new','new'))"),
 ('start_new_session=True','start_new_session=False'),
 ('status=process.wait(timeout=300)','status=process.wait()'),
 ('os.killpg(process.pid,signal.SIGKILL);process.wait();raise','process.wait();raise'),
 ("'--time','--proof',str(proof),'--proof-sha256',PROOFS[variant]", "'--gpu'"),
 ("result.get('sample_count')!=100", "result.get('sample_count')!=10"),
 ("sha(destination/'output/samples.csv')!=result['samples_sha256']",'False'),
])
def test_source_mutation(old,new):
    source=Path(m.__file__).read_text();assert old in source
    with pytest.raises(ValueError):guard(source.replace(old,new,1))


def test_actual_run_block_fake_child(monkeypatch,tmp_path):
    """Exercise actual run_block subprocess boundary; no GPU backend is invoked."""
    proof=tmp_path/'proof';proof.write_text('{}'); calls=[]
    monkeypatch.setattr(m,'PROOFS',{'old':'fixed','new':'fixed'})
    monkeypatch.setattr(m,'sha',lambda path:'fixed')
    monkeypatch.setattr(m,'check_result',lambda *args:None)
    class Fake:
        pid=123
        def __init__(self,args,**kwargs):
            calls.append((args,kwargs));self.args=args;self.log=kwargs['stdout']
            self.log.write(b'AMD Radeon 8060S Graphics (RADV STRIX_HALO) libVkLayer_khronos_validation.so\n');self.log.flush()
            out=Path(args[args.index('--output-root')+1]);out.mkdir()
            (out/'samples.csv').write_text('gpu_ms\n1\n')
            (out/'result.json').write_text(json.dumps({'samples_sha256':'fixed','summary_ms':{}}))
        def wait(self,timeout=None):calls.append(('wait',timeout));return 0
    monkeypatch.setattr(m.subprocess,'Popen',Fake)
    report=m.run_block('new',tmp_path/'arm',proof,{})
    args,opts=calls[0]
    assert '--time' in args and '--proof' in args and '--gpu' not in args and '--parity' not in args
    assert opts['start_new_session'] is True and calls[1]==('wait',300)
    assert report['variant']=='new' and (tmp_path/'arm/exit.txt').read_text()=='0\n'


def test_actual_run_block_timeout_kills_and_reaps(monkeypatch,tmp_path):
    proof=tmp_path/'proof';proof.write_text('{}');calls=[]
    monkeypatch.setattr(m,'PROOFS',{'old':'fixed','new':'fixed'})
    monkeypatch.setattr(m,'sha',lambda path:'fixed')
    class Fake:
        pid=987
        def __init__(self,*a,**kw):pass
        def wait(self,timeout=None):
            calls.append(('wait',timeout))
            if timeout is not None:raise subprocess.TimeoutExpired('fake',timeout)
            return -9
    monkeypatch.setattr(m.subprocess,'Popen',Fake)
    monkeypatch.setattr(m.os,'killpg',lambda pid,sig:calls.append(('killpg',pid,sig)))
    with pytest.raises(subprocess.TimeoutExpired):m.run_block('old',tmp_path/'arm',proof,{})
    assert calls==[('wait',300),('killpg',987,signal.SIGKILL),('wait',None)]
    assert (tmp_path/'arm/exit.txt').read_text()=='timeout\n'
