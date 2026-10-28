"""Actual CPU View setup path: source-aware admission, cache parity and cost.

Only the GPU-facing DaxaRectification declaration is isolated in a temporary
header. The View/calibration translation units and geometry types are actual
production sources. This does not initialize Vulkan or measure a frame loop.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import tempfile

import pytest
from test_daxa_depth_hotpath import definition
from test_kernel_variant_hotpath import body
from test_daxa_view_depth_hotpath import check_view_hotpath

ROOT=Path(__file__).resolve().parents[1]
ARTIFACTS=ROOT/'artifacts/zed-task3/performance/view-setup'


def source_hashes():
    paths=('src/daxa_view.cpp','src/daxa_view_calibration.cpp','src/daxa_calibration.cpp',
           'src/daxa_preprocess.cpp','src/capture.cpp','include/stereoforge/daxa_view_calibration.hpp',
           'include/stereoforge/canonical_rgb.hpp','include/stereoforge/daxa_preprocess.hpp',
           'include/stereoforge/daxa_calibration.hpp','include/stereoforge/geometry.hpp',
           'tests/test_daxa_view_canonical_setup_hotpath.py')
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


PROGRAM=r'''
#include <stereoforge/daxa_view_calibration.hpp>
#include <stereoforge/canonical_rgb.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
std::atomic<bool> tracking=false;std::atomic<unsigned> allocations=0;
void* operator new(std::size_t n){if(tracking)++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{std::free(p);}void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
namespace stereoforge {
namespace {void require(bool v,const char* m){if(!v)throw std::invalid_argument(m);}}
ACTUAL_RAW_DEFINITIONS
}
using namespace stereoforge;
void check(bool v,const char* m){if(!v)throw std::runtime_error(m);}
template<class F>void reject(F f,const char* m="invalid setup admitted"){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}check(caught,m);}
ACTUAL_SMALL_FIXTURE
DaxaRectification large_fixture(){
 auto c=fixture();c.eye_width=c.width=1280;c.source_height=c.height=720;c.stride_bytes=5120;
 auto& g=*c.geometry;
 for(unsigned eye=0;eye<2;++eye){g.original.eyes[eye]={1280,720,1000,1000,639.5,359.5,{}};g.rectified.eyes[eye]=g.original.eyes[eye];}
 g.P1={1000,0,639.5,0,0,1000,359.5,0,0,0,1,0};g.P2={1000,0,639.5,-100,0,1000,359.5,0,0,0,1,0};
 for(unsigned eye=0;eye<2;++eye){auto& mx=c.maps[eye*2];auto& my=c.maps[eye*2+1];mx.resize(1280*720);my.resize(1280*720);for(unsigned y=0;y<720;++y)for(unsigned x=0;x<1280;++x){mx[y*1280+x]=float(x);my[y*1280+x]=float(y);}}
 return c;
}
CaptureFrame canonical_source(DaxaRectification const& c){
 CaptureFrame f;f.source_kind=CaptureSourceKind::CanonicalRgbMetadata;f.source_width=2*c.eye_width;f.height=c.source_height;f.calibration_identity=c.original_identity;f.frame_id=17;f.camera_timestamp_ns=123;f.camera_clock_domain="fixture_camera";
 auto storage=std::make_shared<std::array<std::vector<uint8_t>,6>>();auto s=std::make_shared<CanonicalRgbSource>();s->backing=storage;s->input_identity="prepared";s->original_pixel_identity="original";s->rectified_pixel_identity="rectified";s->original_calibration_identity=c.original_identity;s->rectified_calibration_identity=c.rectified_identity;
 for(unsigned i=0;i<6;++i){auto& p=i<4?s->rgb[i]:s->support[i-4];p.width=i<2?c.eye_width:c.width;p.height=i<2?c.source_height:c.height;p.stride_bytes=p.width*(i<4?3:1);p.format=i<4?CanonicalRgbFormat::Rgb8:CanonicalRgbFormat::SupportU8;(*storage)[i].resize(size_t(p.stride_bytes)*p.height,i<4?127:1);p.bytes=(*storage)[i];}
 f.canonical_rgb=s;return f;
}
void parity(DaxaViewCalibration const& a,DaxaViewCalibration const& b){
 check(a.rays.size()==b.rays.size()&&a.preview_indices==b.preview_indices,"canonical/raw preview or ray count mismatch");
 check(std::memcmp(a.rays.data(),b.rays.data(),a.rays.size()*sizeof(Ray3))==0,"canonical/raw ray bytes mismatch");
 check(a.effective_view.source_to_virtual_left==b.effective_view.source_to_virtual_left&&a.effective_view.source_to_virtual_right==b.effective_view.source_to_virtual_right,"canonical/raw poses mismatch");
 check(a.original_width==b.original_width&&a.original_height==b.original_height&&a.preview_valid_count==b.preview_valid_count,"canonical/raw cache metadata mismatch");
}
int main(){try{
 for(bool large:{false,true}){
  auto raw=large?large_fixture():fixture();auto canonical=raw;canonical.stride_bytes=0;for(auto& map:canonical.maps)map.clear();auto source=canonical_source(raw);
  GeometryConfig cfg{raw.width,raw.height,raw.eye_width,raw.source_height};VirtualView v;auto cam=raw.geometry->original.eyes[0];v.intrinsics={float(cam.fx),float(cam.fy),float(cam.cx),float(cam.cy)};v.source_to_virtual_right[3]=.1f;v.mode=ViewMode::Preview;
  // Existing raw API must remain strict. Oracle includes real raw map/model association.
  reject([&]{prepare_daxa_view_calibration(canonical,cfg,v);});auto expected=prepare_daxa_view_calibration(raw,cfg,v);
  auto admitted=prepare_daxa_view_calibration(canonical,cfg,v,source);parity(expected,admitted);
  auto first=admitted.rays.front();check(first.z==1&&first.x==float(-raw.geometry->rectified.eyes[0].cx/raw.geometry->rectified.eyes[0].fx)&&first.y==float(-raw.geometry->rectified.eyes[0].cy/raw.geometry->rectified.eyes[0].fy),"analytic unit-Z ray mismatch");
  if(large){check(admitted.preview_valid_count==2ull*1280*720,"identity preview support mismatch");for(unsigned eye=0;eye<2;++eye)for(unsigned i:{0u,1279u,1280u,1280*720-1u})check(admitted.preview_indices[eye*1280*720+i]==eye*1280*720+i,"identity preview index mismatch");}
  auto bad_source=source;bad_source.calibration_identity="foreign";reject([&]{prepare_daxa_view_calibration(canonical,cfg,v,bad_source);});
  auto bad=canonical;bad.geometry->R1[0]=0;reject([&]{prepare_daxa_view_calibration(bad,cfg,v,source);});
  bad=canonical;bad.geometry->P2[3]=0;reject([&]{prepare_daxa_view_calibration(bad,cfg,v,source);});
  bad=canonical;bad.geometry->original.eyes[0].distortion={0,0,0};reject([&]{prepare_daxa_view_calibration(bad,cfg,v,source);},"unsupported original distortion count accepted");
  bad=canonical;bad.geometry->original.eyes[0].distortion.resize(15);reject([&]{prepare_daxa_view_calibration(bad,cfg,v,source);},"oversized original distortion accepted");
  auto raw_source=source;raw_source.source_kind=CaptureSourceKind::PackedYuyv;raw_source.canonical_rgb.reset();reject([&]{prepare_daxa_view_calibration(canonical,cfg,v,raw_source);});
  for(unsigned run=0;run<3;++run){allocations=0;tracking=true;auto start=std::chrono::steady_clock::now();auto cache=prepare_daxa_view_calibration(canonical,cfg,v,source);auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();tracking=false;parity(expected,cache);
   std::cout<<"{\"scope\":\"actual_CPU_source_aware_View_setup_cache_rebuild_only_no_GPU_or_frame_loop\",\"large\":"<<(large?"true":"false")<<",\"run\":"<<run<<",\"setup_ms\":"<<elapsed<<",\"setup_allocations\":"<<allocations<<",\"ray_count\":"<<cache.rays.size()<<",\"preview_count\":"<<cache.preview_indices.size()<<",\"preview_valid_count\":"<<cache.preview_valid_count<<",\"raw_canonical_exact_parity\":true,\"strict_raw_and_bad_geometry_rejected\":true}\n";
  }
 }
 return 0;}catch(const std::exception& e){tracking=false;std::cerr<<e.what()<<'\n';return 1;}}
'''


@pytest.mark.parametrize('skip_model_count_guard',[False,True])
def test_actual_source_aware_view_setup_parity_and_three_cost_samples(skip_model_count_guard):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ compiler unavailable')
    prep=(ROOT/'src/daxa_preprocess.cpp').read_text();capture=(ROOT/'src/capture.cpp').read_text()
    definitions=''.join(definition(prep,s) for s in ('MetricCalibration DaxaRectification::metric_calibration(', 'uint64_t DaxaRectification::requested_bytes(', 'void DaxaRectification::validate(', 'void DaxaRectification::validate_source('))
    definitions+=definition(capture,'void CaptureFrame::require_calibration(')
    program=PROGRAM.replace('ACTUAL_RAW_DEFINITIONS',definitions).replace('ACTUAL_SMALL_FIXTURE',definition((ROOT/'tests/daxa_view_depth.cpp').read_text(),'static DaxaRectification fixture('))
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='view-cpu-',dir=ARTIFACTS) as tmp:
        temp=Path(tmp);shadow=temp/'stereoforge';shadow.mkdir()
        declaration=definition((ROOT/'include/stereoforge/daxa_preprocess.hpp').read_text(),'struct DaxaRectification')+';'
        (shadow/'daxa_preprocess.hpp').write_text('#pragma once\n#include <stereoforge/daxa_calibration.hpp>\n#include <stereoforge/capture.hpp>\n#include <optional>\nnamespace stereoforge {\n'+declaration+'\n}\n')
        code=temp/'probe.cpp';code.write_text(program);binary=temp/'probe'
        setup_source=ROOT/'src/daxa_view_calibration.cpp'
        if skip_model_count_guard:
            text=setup_source.read_text();mutated=re.sub(r'require\(count==0\|\|count==4[^;]+;', '', text, count=1)
            assert text!=mutated
            setup_source=temp/'mutated_setup.cpp';setup_source.write_text(mutated)
        command=[compiler,'-std=c++20','-O3','-I'+str(temp),'-I'+str(ROOT/'include'),str(code),str(setup_source),str(ROOT/'src/daxa_calibration.cpp'),'-o',str(binary)]
        built=subprocess.run(command,capture_output=True,text=True,timeout=60);assert built.returncode==0,built.stderr
        result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
        if skip_model_count_guard:
            assert result.returncode!=0 and 'unsupported original distortion count accepted' in result.stderr,result.stderr
            return
        assert result.returncode==0,result.stderr
        rows=[json.loads(row) for row in result.stdout.splitlines()];assert len(rows)==6
        shutil.copyfile(code,ARTIFACTS/'probe.cpp')
        saved_include=ARTIFACTS/'include';(saved_include/'stereoforge').mkdir(parents=True,exist_ok=True)
        shutil.copyfile(shadow/'daxa_preprocess.hpp',saved_include/'stereoforge/daxa_preprocess.hpp')
        command=[compiler,'-std=c++20','-O3','-I'+str(saved_include),'-I'+str(ROOT/'include'),str(ARTIFACTS/'probe.cpp'),str(ROOT/'src/daxa_view_calibration.cpp'),str(ROOT/'src/daxa_calibration.cpp'),'-o',str(ARTIFACTS/'probe')]
        report={'samples':rows,'source_sha256':source_hashes(),'build_command':command,'scope':'Actual complete View/calibration CPU translation units; temporary preprocessing header contains exact production declaration. No device/GPU implementation; input preparation excluded from timing.'}
        (ARTIFACTS/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(result.stdout)


def check_cold_dispatch(setup, view):
    check_view_hotpath(view)
    ctor=body(view,'DaxaView::DaxaView(GeometryConfig c,VirtualView const& v,std::shared_ptr<DaxaFoundation> context,DaxaObservationFrame const& prototype,unsigned count)')
    assert 'prepare_daxa_view_calibration(*maps,c,v,*prototype.source())' in ctor
    impl=body(setup,'DaxaViewCalibration prepare_impl(')
    branch=body(impl,'if(canonical)')
    assert 'c.stride_bytes==0' in branch and 'map.empty()' in branch and 'c.metric_calibration()' in branch
    assert 'else c.validate()' in impl
    assert 'if(!canonical)g.validate_map_association(' in impl
    assert 'g.validate()' in impl and 'rigid(view.source_to_virtual_left)' in impl and 'rigid(view.source_to_virtual_right)' in impl
    dispatch=body(setup,'DaxaViewCalibration prepare_daxa_view_calibration(DaxaRectification const& c,GeometryConfig config,VirtualView const& view,const CaptureFrame& source)')
    assert 'validate_canonical_rgb_source(source,false)' in dispatch
    assert 'canonical_rgb_grid_association(source,' in dispatch and 'prepare_impl(c,config,view,true)' in dispatch
    assert 'c.validate_source(source)' in dispatch and 'prepare_impl(c,config,view,false)' in dispatch
    assert not re.search(r'\b(?:wait\w*|readback\w*|create_\w+)\s*\(',impl+dispatch)


def test_source_dispatch_and_all_existing_hotpath_invariants():
    check_cold_dispatch((ROOT/'src/daxa_view_calibration.cpp').read_text(),(ROOT/'src/daxa_view.cpp').read_text())


@pytest.mark.parametrize('old,new', [
 ('validate_canonical_rgb_source(source,false)', 'validate_canonical_rgb_source(source,true)'),
 ('if(!canonical)g.validate_map_association(', 'g.validate_map_association('),
 ('prepare_daxa_view_calibration(*maps,c,v,*prototype.source())','prepare_daxa_view_calibration(*maps,c,v)'),
])
def test_setup_guards_reject_scan_raw_map_validation_or_legacy_typed_dispatch(old,new):
    setup=(ROOT/'src/daxa_view_calibration.cpp').read_text();view=(ROOT/'src/daxa_view.cpp').read_text()
    assert old in setup+view
    with pytest.raises(AssertionError):check_cold_dispatch(setup.replace(old,new,1),view.replace(old,new,1))
