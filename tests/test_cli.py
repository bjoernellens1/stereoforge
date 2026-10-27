import subprocess,sys
import json

def test_cli_synthetic_cpu_smoke(tmp_path):
    run=subprocess.run([sys.executable,'-m','stereoforge','synthetic','--backend','cpu_census','--width','64','--height','24','--disparities','8','--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    report=json.loads((tmp_path/'out'/'metrics.json').read_text())
    assert report['valid_fraction']>0
    assert (tmp_path/'out'/'depth.npz').is_file()

def test_cli_does_not_accept_cuda():
    run=subprocess.run([sys.executable,'-m','stereoforge','synthetic','--backend','cuda'],capture_output=True,text=True)
    assert run.returncode!=0
