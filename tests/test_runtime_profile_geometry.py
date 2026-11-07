from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_native_profile_geometry_and_search_admission(tmp_path):
    compiler=shutil.which('c++')
    if not compiler: pytest.skip('C++ compiler unavailable')
    source=tmp_path/'profile.cpp'
    source.write_text(r'''#include "stereo_runtime_profiles.hpp"
#include <limits>
#include <iostream>
using namespace stereoforge::runtime_profile;
int main(){
 auto live=geometry("live-quality",1280,720,1280,720,63.0413850625,0);
 auto q720=geometry("quality-reference",1280,720,1280,720,63.0413850625,0);
 auto q1080=geometry("quality-reference",1920,1080,1920,1080,126.88673,0);
 if(live.disparities!=128||live.slots!=2||live.fps!=60||live.margin_px!=0)return 1;
 if(q720.disparities!=256||q720.slots!=2||q720.fps!=60||q720.margin_px!=16)return 2;
 if(q1080.disparities!=288||q1080.slots!=1||q1080.fps!=30||q1080.margin_px!=16)return 3;
 unsigned rejected=0;
 auto reject=[&](const char* profile,unsigned ow,unsigned oh,unsigned w,unsigned h,double fxb,double cx){
  try{(void)geometry(profile,ow,oh,w,h,fxb,cx);}catch(const std::invalid_argument&){++rejected;}
 };
 reject("live-quality",1920,1080,1920,1080,126.88673,0);
 reject("quality-reference",1920,1080,1280,720,63.04138,0);
 reject("quality-reference",640,360,640,360,30,0);
 reject("quality-reference",1280,720,1280,720,0,0);
 reject("quality-reference",1280,720,1280,720,-1,0);
 reject("quality-reference",1280,720,1280,720,std::numeric_limits<double>::quiet_NaN(),0);
 reject("quality-reference",1280,720,1280,720,63,std::numeric_limits<double>::infinity());
 reject("live-quality",1280,720,1280,720,64,0);
 reject("quality-reference",1920,1080,1920,1080,136,0);
 reject("quality-reference",1280,720,1280,720,63,-100);
 reject("unknown",1280,720,1280,720,63,0);
 if(rejected!=11)return 4;
 std::cout<<"three recipes and eleven invalid geometry/range controls passed";
}
''')
    compiled=subprocess.run([compiler,'-std=c++20','-I'+str(ROOT/'apps'),str(source),'-o',str(tmp_path/'profile')],capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stderr
    result=subprocess.run([str(tmp_path/'profile')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
