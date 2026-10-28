"""Locked old/new/old FP2 timing of pinned parity-proven synthetic input."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
from pathlib import Path

from benchmarks.fp2_720p_prepare import sha

ROOT=Path(__file__).resolve().parents[1]
MANIFEST='7f70c357ec371d5c434c435e4a2500d87a149e43a5859e1695ada38cf021ce17'
BASELINE='e690b1895b63cf2c9c158419bf0662e7e316426d1d3c164f8c078d3700090041'
PROOFS={'old':'ce5610937f578f7e7a15c302374bb5efa69687f041739f282f0b63cfe5f07c18',
        'new':None}


def check_result(result, variant, proof, harness, binary, tiny):
    expected_scope='parallel timing' if variant=='old' else 'subgroup timing'
    if result.get('scope')!=expected_scope or result.get('manifest_sha256')!=MANIFEST or result.get('baseline_sha256')!=BASELINE:
        raise ValueError('result scope/input mismatch')
    if result.get('sample_count')!=100 or result.get('warmup')!=10 or result.get('input_scope')!='one repeated preuploaded analytic frame; one in-flight slot':
        raise ValueError('result timing contract mismatch')
    if result.get('harness_sha256')!=sha(harness) or result['host']['binary_sha256']!=sha(binary) or result['tiny']['binary_sha256']!=sha(tiny):
        raise ValueError('result binary mismatch')
    p=json.loads(Path(proof).read_text())
    for field in ('host','tiny','harness_sha256','manifest_sha256','baseline_sha256'):
        if result[field]!=p[field]: raise ValueError('result proof identity mismatch '+field)


def run_block(variant, destination, proof, env):
    prefix='parallel' if variant=='old' else 'subgroup'
    harness=ROOT/'benchmarks'/('fp2_'+prefix+'_diagnostic.py')
    binary=ROOT/'build/daxa'/('fp2_'+prefix+'_fixture')
    tiny=ROOT/'build/daxa'/('fp2_'+prefix+'_tiny')
    if sha(proof)!=PROOFS[variant]: raise ValueError('proof hash mismatch')
    args=[str(ROOT/'.venv/bin/python'),str(harness),
          '--prepared',str(ROOT/'artifacts/fp2-720p-cpu-20261028-v1/prepared'),
          '--manifest-sha256',MANIFEST,'--host',str(binary),'--tiny',str(tiny),
          '--artifact-root',str(ROOT/'artifacts/compact-sgm-step-20261028-v2'),
          '--family-root',str(ROOT/'artifacts/fast60-refinement-20261028/analytic-family-720p'),
          '--baseline-result',str(ROOT/'artifacts/fp2-integrated-20261029-v1/outputs/result.json'),
          '--output-root',str(destination/'output'),'--time','--proof',str(proof),'--proof-sha256',PROOFS[variant]]
    identities={str(path):sha(path) for path in (proof,harness,binary,tiny,Path(__file__))}
    destination.mkdir()
    try:
        with (destination/'run.log').open('xb') as log:
            process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True)
            try: status=process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise
        (destination/'exit.txt').write_text(str(status)+'\n')
        if status: raise RuntimeError('block failed: '+variant)
    except subprocess.TimeoutExpired:
        (destination/'exit.txt').write_text('timeout\n')
        raise
    if any(sha(path)!=digest for path,digest in identities.items()):raise ValueError('identity changed during block')
    (destination/'identities.json').write_text(json.dumps(identities,sort_keys=True,indent=2)+'\n')
    log=(destination/'run.log').read_text(errors='replace')
    if 'AMD Radeon 8060S Graphics (RADV STRIX_HALO)' not in log or 'libVkLayer_khronos_validation.so' not in log:
        raise ValueError('device or validation layer not attested')
    result=json.loads((destination/'output/result.json').read_text())
    check_result(result,variant,proof,harness,binary,tiny)
    if sha(destination/'output/samples.csv')!=result['samples_sha256']:
        raise ValueError('sample hash mismatch')
    return {'variant':variant,'result_sha256':sha(destination/'output/result.json'),
            'log_sha256':sha(destination/'run.log'),'samples_sha256':result['samples_sha256'],
            'summary_ms':result['summary_ms']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--new-proof',required=True);p.add_argument('--new-proof-sha256',required=True);a=p.parse_args()
    if len(a.new_proof_sha256)!=64 or any(x not in '0123456789abcdef' for x in a.new_proof_sha256):raise ValueError('invalid candidate proof hash')
    PROOFS['new']=a.new_proof_sha256
    output=Path(a.output_root).resolve()
    if output.exists():raise ValueError('output root exists')
    old=ROOT/'artifacts/fp2-parallel-20261029-v1/parity/result.json'
    new=Path(a.new_proof).resolve()
    for name,path in [('old',old),('new',new)]:
        if sha(path)!=PROOFS[name]:raise ValueError('proof hash mismatch '+name)
    env=os.environ.copy();env.pop('VK_LAYER_ENABLES',None)
    env.update({'VK_ICD_FILENAMES':'/usr/share/vulkan/icd.d/radeon_icd.x86_64.json',
                'VK_LAYER_PATH':str(ROOT/'build/validation-layer/usr/share/vulkan/explicit_layer.d'),
                'LD_LIBRARY_PATH':str(ROOT/'build/validation-layer/usr/lib64')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),
                'VK_LAYER_VALIDATE_SYNC':'1','VK_LOADER_DEBUG':'layer','PYTHONPATH':str(ROOT/'python')+':'+str(ROOT)})
    with open('/tmp/stereoforge-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output.mkdir(parents=True,exist_ok=False)
        blocks=[]
        for index,variant in enumerate(('old','new','old')):
            blocks.append(run_block(variant,output/f'{index}-{variant}',old if variant=='old' else new,env))
        (output/'summary.json').write_text(json.dumps({'blocks':blocks,'order':'old/new/old','warmups_per_block':10,
            'samples_per_block':100,'scope':'one repeated preuploaded analytic frame; one in-flight slot',
            'work_by_variant':{'old':'128 lanes per path pixel; shared tree reduction, including inactive candidates',
                               'new':'128 lanes per path pixel; subgroup minimum and leader atomic minimum, including inactive candidates'},
            'synchronization_validation':'requested, not independently attested'},indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
