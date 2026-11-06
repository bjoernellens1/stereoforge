"""Calibration-only cloud layout/ray contracts; no ROS or GPU SDK required."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_calibration_and_layout_contract(tmp_path):
    source = r'''
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <stereoforge/point_cloud_contract.hpp>
#include <stereoforge/fixed_json_uint_fields.hpp>
int main(){try{
 using namespace stereoforge;
 auto check=[](bool ok){if(!ok)throw std::runtime_error("cloud calibration/layout mismatch");};
 auto rejects=[&](auto fn){bool caught=false;try{fn();}catch(std::invalid_argument const&){caught=true;}check(caught);};
 FixedJsonUintFields<2> metadata("{\"id\":\"00000000000000000000\",\"ticket\":\"00000000000000000000\"}", {"id","ticket"});
 std::string output(metadata.template_bytes());auto capacity=output.capacity();
 metadata.stamp(std::span<char>(output),{UINT64_MAX,7});
 check(output=="{\"id\":\"18446744073709551615\",\"ticket\":\"00000000000000000007\"}"&&output.capacity()==capacity);
 rejects([&]{metadata.stamp(std::span<char>(output).first(1),{1,2});});
 rejects([]{FixedJsonUintFields<1> bad("{}",{"missing"});});
 static_assert(sizeof(XyzRgb)==16&&offsetof(XyzRgb,rgb)==12);
 PointCloudLayout full(7,5,1),sampled(7,5,3);
 check(full.width==7&&full.height==5&&full.row_step==112&&full.bytes==560);
 check(sampled.width==3&&sampled.height==2&&sampled.row_step==48&&sampled.bytes==96);
 check(sampled.source_index(0)==0&&sampled.source_index(1)==3&&sampled.source_index(2)==6&&sampled.source_index(3)==21&&sampled.source_index(5)==27);
 rejects([&]{sampled.source_index(6);});rejects([]{PointCloudLayout bad(0,5,1);});rejects([]{PointCloudLayout bad(7,5,0);});rejects([]{PointCloudLayout bad(8193,1,1);});
 DaxaCameraModel k{7,5,8,10,2.25,1.75,{}};
 auto rays=point_cloud_rays(k);check(rays.size()==35);
 for(unsigned y=0;y<5;++y)for(unsigned x=0;x<7;++x){auto ray=rays[y*7+x];check(ray[0]==float((double(x)-2.25)/8)&&ray[1]==float((double(y)-1.75)/10));}
 check(point_cloud_rgb(0xff030201u)==0x00010203u);check(point_cloud_rgb(0xff000000u)==0u);check(point_cloud_rgb(0xffffffffu)==0x00ffffffu);
 k.fx=0;rejects([&]{point_cloud_rays(k);});k.fx=8;k.distortion={.1};rejects([&]{point_cloud_rays(k);});k.distortion={0,0,0,0,0};point_cloud_rays(k);
 return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
'''
    path = tmp_path / 'cloud-contract.cpp'
    path.write_text(source)
    binary = tmp_path / 'cloud-contract'
    subprocess.run(['c++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'include'), str(path), '-o', str(binary)], check=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
