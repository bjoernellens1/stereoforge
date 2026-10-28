"""CPU admission exercises the production loader without Vulkan or Daxa."""
from hashlib import sha256
import json
import subprocess
from pathlib import Path

import pytest
from test_zed_native_input import native_record
from benchmarks.zed_comparison.native_input import write_native_input

ROOT = Path(__file__).resolve().parents[1]

_loader_exe = None

@pytest.fixture
def loader(tmp_path_factory, json_include):
    global _loader_exe
    if _loader_exe is not None: return _loader_exe
    d = tmp_path_factory.mktemp('canonical-loader')
    source = d/'probe.cpp'
    source.write_text(r'''
#include <stereoforge/canonical_record.hpp>
#include <fstream>
#include <iostream>
int main(int argc,char**argv){try{
 auto r=stereoforge::load_canonical_record(argv[1]); auto f=r.frame.source; auto s=f->canonical_rgb;
 if(!r.frame.admitted()||r.ordinal!=0||f->frame_id!=uint64_t(INT64_MAX)||f->source_width!=8||f->height!=4||f->stride_bytes||!f->packed_yuyv.empty()||f->camera_timestamp_ns!=100||f->camera_clock_domain!="camera"||f->host_ingress_timestamp_ns||f->host_clock_domain!="unknown")return 3;
 if(s->rgb[0].bytes.size()!=60||s->rgb[0].stride_bytes!=15||s->rgb[0].bytes[14]!=0xe7||s->support[0].bytes.size()!=20||s->support[0].bytes[2*5+1]!=0||r.full_geometry.rectified.eyes[0].fx!=8||r.coarse_geometry.rectified.eyes[0].cx!=.5||r.coarse_geometry.P2[3]!=-.48)return 4;
 if(r.files.size()!=13||r.producer!="independent analytic fixture"||s->original_calibration_identity!=r.full_geometry.original_identity||s->rectified_calibration_identity!=r.full_geometry.rectified_identity||s->coarse_rectified_calibration_identity!=r.coarse_geometry.rectified_identity||r.coarse_geometry.original_identity!=r.full_geometry.original_identity||f->calibration_identity!=s->original_calibration_identity||!s->backing)return 7;
 for(auto const&p:s->rgb)for(unsigned y=0;y<4;++y){for(unsigned x=0;x<4;++x){auto i=y*15+x*3;if(p.bytes[i]!=x+y*4||p.bytes[i+1]!=19+x||p.bytes[i+2]!=223-y)return 8;}for(unsigned i=12;i<15;++i)if(p.bytes[y*15+i]!=0xe7)return 9;}
 for(auto const&p:s->support)for(unsigned y=0;y<4;++y){for(unsigned x=0;x<4;++x)if(p.bytes[y*5+x]!=unsigned(x!=1||y!=2))return 10;if(p.bytes[y*5+4]!=0xe7)return 11;}
 auto saved=s->rgb[0].bytes[0];
 if(argc>2){std::ofstream o(r.files.back().path,std::ios::binary|std::ios::app);o<<'!';o.close();try{stereoforge::verify_canonical_record_files(r);return 5;}catch(const std::exception&){} r={};f.reset();if(s->rgb[0].bytes[0]!=saved)return 6;}
 else stereoforge::verify_canonical_record_files(r);
 std::cout<<s->original_pixel_identity<<"\n"<<s->rectified_pixel_identity<<"\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what();return 1;}}
''')
    exe=d/'loader'
    subprocess.run(['c++','-std=c++20','-O0','-pthread','-I'+str(ROOT/'include'),'-I'+str(json_include),str(source),str(ROOT/'src/canonical_record.cpp'),str(ROOT/'src/daxa_calibration.cpp'),str(ROOT/'src/capture.cpp'),'-lcrypto','-o',str(exe)],check=True)
    _loader_exe = exe
    return exe

@pytest.fixture
def bundle(native_record,tmp_path):
    _,record,_,supplement=native_record
    return write_native_input(record,supplement,tmp_path/'bundle')

def run(loader,bundle,*args):
    return subprocess.run([str(loader),str(bundle),*args],text=True,capture_output=True)

def rewrite(path,value): path.write_text(json.dumps(value))
def seal(manifest,path,key):
    value=json.loads(manifest.read_text());data=path.read_bytes();value[key].update(sha256=sha256(data).hexdigest(),size_bytes=len(data));rewrite(manifest,value)

def test_load_and_owned_lifetime(loader,bundle):
    manifest=json.loads(bundle.read_text())
    result=run(loader,bundle)
    assert result.returncode==0,result.stderr
    assert result.stdout.splitlines()==[manifest['original_pixel_identity'],manifest['rectified_pixel_identity']]
    assert result.stdout.splitlines()==['380d8cc1becb902b098e771cf33a81708ec22a58b73430ec220588c1f3a118b9',
                                       '94d9b419d83ed2df3744ce72069908d78da88a13c07d300a18a47d93ce088090']
    result=run(loader,bundle,'mutate');assert result.returncode==0,result.stderr

@pytest.mark.parametrize('case',['unknown_manifest','duplicate','nonfinite','deep','large','pixel_identity','profile','unsafe','symlink','hash','size','nonanalytic','timestamp','unknown_clock','source_id','nonbinary','stride','extent','format','dimensions','supplement_extra','supplement_binding','matrix_bool','matrix_shape','rotation','projection','coarse','coarse_pose','calibration_bool','calibration_shape','calibration_distortion','plane_bound','plane_sum','parent_symlink','traversal','duplicate_nested','ingress_bool','hash_uppercase','coordinate_frame','nul_path'])
def test_reject_malformed(loader,bundle,case):
    m=json.loads(bundle.read_text());root=bundle.parent
    record=root/m['source_record']['path'];r=json.loads(record.read_text())
    supplement=root/m['geometry_supplement']['path'];s=json.loads(supplement.read_text())
    if case=='nul_path':
        m['source_record']['path']+='\0hidden';rewrite(bundle,m)
    elif case=='hash_uppercase':
        m['source_record']['sha256']=m['source_record']['sha256'].upper();rewrite(bundle,m)
    elif case=='duplicate_nested':
        supplement.write_text(supplement.read_text().replace('\"R1\":', '\"R1\":0,\"R1\":'));seal(bundle,supplement,'geometry_supplement')
    elif case.startswith('calibration_'):
        desc=r['calibration']['raw'];p=root/desc['path'];v=json.loads(p.read_text())
        if case=='calibration_bool':v['left']['fx']=True
        elif case=='calibration_shape':v['right_to_left']=[1]*16
        else:v['left']['distortion']={'model':'brown_conrady','coefficients':[0]*3}
        rewrite(p,v);desc.update(sha256=sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
        rewrite(record,r);seal(bundle,record,'source_record');s.update(raw_calibration_sha256=desc['sha256'],record_sha256=sha256(record.read_bytes()).hexdigest());rewrite(supplement,s);seal(bundle,supplement,'geometry_supplement')
    elif case=='parent_symlink':
        parent=root/'linked';parent.symlink_to(root,target_is_directory=True);m['source_record']['path']='linked/'+m['source_record']['path'];rewrite(bundle,m)
    elif case=='traversal':
        m['source_record']['path']='x/../'+m['source_record']['path'];rewrite(bundle,m)
    elif case=='unknown_manifest':m['extra']=1;rewrite(bundle,m)
    elif case=='duplicate':bundle.write_text(bundle.read_text()[:-1]+',"schema":"duplicate"}')
    elif case=='nonfinite':bundle.write_text('{"number":NaN}')
    elif case=='deep':bundle.write_text('['*65+'0'+']'*65)
    elif case=='large':bundle.write_bytes(b' '* (16*1024*1024+1))
    elif case=='pixel_identity':m['original_pixel_identity']='0'*64;rewrite(bundle,m)
    elif case=='profile':m['sampling_profile']='wrong';rewrite(bundle,m)
    elif case=='unsafe':m['source_record']['path']='../record.json';rewrite(bundle,m)
    elif case=='symlink':record.rename(root/'actual');record.symlink_to(root/'actual')
    elif case=='hash':record.write_bytes(record.read_bytes()+b' ')
    elif case=='size':m['source_record']['size_bytes']=True;rewrite(bundle,m)
    elif case.startswith('coarse'):
        p=root/m['coarse_calibration']['path'];v=json.loads(p.read_text())
        if case=='coarse':v['left']['cx']+=.1
        else:v['right_to_left'][0][3]*=2
        rewrite(p,v);seal(bundle,p,'coarse_calibration')
    elif case.startswith('supplement') or case.startswith('matrix') or case in ('rotation','projection'):
        if case=='supplement_extra':s['extra']=True
        elif case=='supplement_binding':s['input_identity']='0'*64
        elif case=='matrix_bool':s['R1'][0][0]=True
        elif case=='matrix_shape':s['R1']=[1]*9
        elif case=='rotation':s['R1'][0][0]=-1
        else:s['P2'][0][3]=0
        rewrite(supplement,s);seal(bundle,supplement,'geometry_supplement')
    else:
        if case=='coordinate_frame':r['calibration']['geometry']['coordinate_frame']='world'
        elif case=='nonanalytic':r['source']['kind']='physical'
        elif case=='timestamp':r['timestamps']['capture']['value_ns']=0
        elif case=='unknown_clock':r['timestamps']['capture']['clock_domain']='unknown'
        elif case=='source_id':r['source_id']=2**63
        elif case=='ingress_bool':r['timestamps']['ingress']={'status':'available','value':{'value_ns':True,'clock_domain':'host'}}
        elif case in ('plane_bound','plane_sum'):
            for desc in r['images'].values():
                desc.update(stride_bytes=(512*1024*1024//4+1 if case=='plane_bound' else 128*1024*1024//4),size_bytes=(512*1024*1024+4 if case=='plane_bound' else 128*1024*1024))
        elif case=='nonbinary':
            desc=r['images']['support_left'];p=root/desc['path'];data=bytearray(p.read_bytes());data[0]=2;p.write_bytes(data);desc['sha256']=sha256(data).hexdigest()
        elif case=='stride':r['images']['original_left']['stride_bytes']=True
        elif case=='extent':r['images']['original_left']['size_bytes']-=1
        elif case=='format':r['images']['original_left']['format']='rgba8'
        elif case=='dimensions':r['images']['original_left']['width']=32767
        rewrite(record,r);seal(bundle,record,'source_record')
        if case=='nonbinary':
            m=json.loads(bundle.read_text());payload={name:{k:v for k,v in desc.items() if k!='path'} for name,desc in r['images'].items() if name not in ('original_left','original_right')};m['rectified_pixel_identity']=sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest();rewrite(bundle,m)
        s['record_sha256']=sha256(record.read_bytes()).hexdigest();rewrite(supplement,s);seal(bundle,supplement,'geometry_supplement')
    result=run(loader,bundle)
    assert result.returncode==1,(case,result.returncode,result.stderr)
