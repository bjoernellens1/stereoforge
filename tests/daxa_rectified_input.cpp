#include <stereoforge/daxa_rectified_input.hpp>
#include <stereoforge/daxa_refinement.hpp>
#include <stereoforge/daxa_view.hpp>
#include <type_traits>
#include <iostream>
#include <algorithm>
#include <bit>
#include <cmath>
using namespace stereoforge;
static_assert(std::is_same_v<decltype(std::declval<const DaxaRectifiedInput::Frame&>().full()),DaxaFrame>);
static_assert(std::is_same_v<decltype(std::declval<const DaxaRectifiedInput::Frame&>().coarse()),DaxaFrame>);
static_assert(std::is_constructible_v<DaxaDepthPipeline,DaxaDepthConfig,std::shared_ptr<DaxaFoundation>,const DaxaFrame&>);
static_assert(std::is_same_v<decltype(static_cast<DaxaDepthPipeline::Frame (DaxaDepthPipeline::*)(const DaxaFrame&)>(&DaxaDepthPipeline::enqueue_preprocessed)),DaxaDepthPipeline::Frame (DaxaDepthPipeline::*)(const DaxaFrame&)>);
static_assert(std::is_constructible_v<DaxaRefinement,std::shared_ptr<DaxaFoundation>,const DaxaDepthPipeline::Frame&,const DaxaFrame&,DaxaRefinementConfig>);
static_assert(std::is_same_v<decltype(std::declval<DaxaRefinement&>().process(std::declval<const DaxaDepthPipeline::Frame&>(),std::declval<const DaxaFrame&>())),DaxaRefinement::Frame>);
static void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> static void reject(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,"expected rejection");}
static DaxaRectification grid(bool full){
    DaxaRectification c;c.eye_width=4;c.source_height=4;c.width=full?4:2;c.height=full?4:2;c.original_identity="original";c.rectified_identity=full?"canonical-full":"canonical-coarse";
    DaxaRectifiedGeometry g;g.original_identity=c.original_identity;g.rectified_identity=c.rectified_identity;
    for(auto& e:g.original.eyes)e={4,4,8,8,1.5,1.5,{}};
    double f=full?8:4,p=full?1.5:.5;for(auto& e:g.rectified.eyes)e={c.width,c.height,f,f,p,p,{}};
    g.original.right_to_left=g.rectified.right_to_left={1,0,0,.1,0,1,0,0,0,0,1,0,0,0,0,1};g.R1=g.R2={1,0,0,0,1,0,0,0,1};
    g.P1={f,0,p,0,0,f,p,0,0,0,1,0};g.P2={f,0,p,-f*.1,0,f,p,0,0,0,1,0};c.geometry=g;return c;
}
struct Fixture {
    std::shared_ptr<std::array<std::vector<uint8_t>,6>> backing=std::make_shared<std::array<std::vector<uint8_t>,6>>();
    std::shared_ptr<CanonicalRgbSource> sidecar=std::make_shared<CanonicalRgbSource>();
    std::shared_ptr<CaptureFrame> capture=std::make_shared<CaptureFrame>();
    Fixture(){
        sidecar->backing=backing;sidecar->input_identity="analytic-awkward-pitch";sidecar->original_pixel_identity="original-pixels";sidecar->rectified_pixel_identity="rectified-pixels";sidecar->original_calibration_identity="original";sidecar->rectified_calibration_identity="canonical-full";sidecar->coarse_rectified_calibration_identity="canonical-coarse";
        for(unsigned p=0;p<6;++p){unsigned stride=p<4?13+p:7+p;(*backing)[p].assign(stride*4,0xed);CanonicalRgbPlane plane{(*backing)[p],4,4,stride,p<4?CanonicalRgbFormat::Rgb8:CanonicalRgbFormat::SupportU8};if(p<4){sidecar->rgb[p]=plane;for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x){auto i=size_t(y)*stride+x*3;(*backing)[p][i]=uint8_t(p*29+x*3+y);(*backing)[p][i+1]=uint8_t(23+x+y*3);(*backing)[p][i+2]=uint8_t(211-x-y);}}else{sidecar->support[p-4]=plane;for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)(*backing)[p][y*stride+x]=1;}}
        // Black remains geometrically supported; colored pixels need not be.
        for(unsigned ch=0;ch<3;++ch)(*backing)[2][ch]=0;
        (*backing)[4][1]=0;
        capture->source_kind=CaptureSourceKind::CanonicalRgbMetadata;capture->canonical_rgb=sidecar;capture->source_width=8;capture->height=4;capture->frame_id=7;capture->camera_timestamp_ns=11;capture->host_ingress_timestamp_ns=12;capture->camera_clock_domain="analytic-camera";capture->host_clock_domain="host-steady";capture->calibration_identity="original";capture->device_identity="analytic";
    }
    RectifiedRgbFrame frame()const{return RectifiedRgbFrame{capture};}
};
static void offline_admission_contracts(){
    Fixture original,second;
    auto admitted=original.frame();check(admitted.admitted(),"original immutable record admitted once offline");
    auto admitted_second=second.frame();check(admitted_second.admitted()&&admitted_second.source!=admitted.source,"valid fresh second record independently admitted");
    Fixture malformed;(*malformed.backing)[4][0]=2;reject([&]{RectifiedRgbFrame invalid{malformed.capture};});
    RectifiedRgbFrame unadmitted;unadmitted.source=second.capture;check(!unadmitted.admitted(),"default public source assignment cannot bypass admission");
    auto replaced=admitted;replaced.source=second.capture;check(!replaced.admitted()&&admitted.admitted(),"replacement lease invalidates copied capability without changing original");
    auto full=grid(true),coarse=grid(false);reject([&]{validate_daxa_rectified_input_geometry(full,coarse,{},unadmitted);});reject([&]{validate_daxa_rectified_input_geometry(full,coarse,{},replaced);});
    std::weak_ptr<const CaptureFrame> weak=original.capture;std::weak_ptr<const void> backing=original.sidecar->backing;original.capture.reset();original.sidecar.reset();original.backing.reset();
    check(!weak.expired()&&!backing.expired()&&admitted.admitted(),"admission retains original immutable source and backing");
    // A public-source replacement also keeps the privately admitted original
    // lease alive; it still cannot be used as an admitted replacement record.
    admitted={};check(!weak.expired()&&!replaced.admitted(),"private admission lease retained after public replacement");replaced={};check(weak.expired()&&backing.expired(),"immutable original released when all capability leases release");
    Fixture offline;offline.capture->host_ingress_timestamp_ns=0;offline.capture->host_clock_domain="unknown";check(offline.frame().admitted(),"truthfully unavailable offline host ingress remains accepted");
}
static void cpu_oracle(){
    Fixture f;auto full=grid(true),coarse=grid(false);auto r=daxa_rectified_input_reference(f.frame(),full,coarse);
    for(unsigned p=0;p<4;++p)for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x){auto const& in=f.sidecar->rgb[p];auto q=in.bytes.data()+y*in.stride_bytes+x*3;auto expected=uint32_t(q[0])|(uint32_t(q[1])<<8)|(uint32_t(q[2])<<16)|0xff000000u;check(r.full.planes[p][y*4+x]==expected,"exact RGB channels/pitch must survive packing");}
    check(r.full.planes[6][0]==1&&r.full.planes[2][0]==0xff000000u,"black supported pixel must remain valid");check(r.full.planes[6][1]==0&&r.full.planes[2][1]!=0xff000000u,"colored unsupported pixel must remain invalid");
    for(unsigned eye=0;eye<2;++eye)for(unsigned y=0;y<2;++y)for(unsigned x=0;x<2;++x){unsigned sums[3]{};bool valid=true;for(unsigned dy=0;dy<2;++dy)for(unsigned dx=0;dx<2;++dx){auto const& p=f.sidecar->rgb[2+eye];auto q=p.bytes.data()+(y*2+dy)*p.stride_bytes+(x*2+dx)*3;for(unsigned ch=0;ch<3;++ch)sums[ch]+=q[ch];auto const& mask=f.sidecar->support[eye];valid&=mask.bytes[(y*2+dy)*mask.stride_bytes+x*2+dx]==1;}uint32_t c=0xff000000u;for(unsigned ch=0;ch<3;++ch)c|=((sums[ch]+2)/4)<<(ch*8);auto i=y*2+x;check(r.coarse->planes[2+eye][i]==c,"coarse RGB box per-channel halfup");auto gray=((c&255)*9798+((c>>8)&255)*19235+((c>>16)&255)*3735+16384)>>15;check(r.coarse->planes[4+eye][i]==gray,"gray15 applied after RGB box");check(r.coarse->planes[6+eye][i]==unsigned(valid),"coarse support must AND all four inputs");}
    uint64_t upload=0;for(unsigned i=0;i<6;++i){auto const& p=i<4?f.sidecar->rgb[i]:f.sidecar->support[i-4];uint64_t extent=uint64_t(p.height-1)*p.stride_bytes+uint64_t(p.width)*(i<4?3:1);upload+=(extent+3)&~uint64_t{3};}
    check(daxa_rectified_input_required_bytes(full,coarse,{},f.frame())==2*(2*upload+200+8*16+24*16+24*4),"exact2slot pitched staging/raw/full/coarse budget counts original aliases once");
    check(r.coarse->planes[0]==r.full.planes[0]&&r.coarse->planes[1]==r.full.planes[1],"original images preserved across grids");
    check(canonical_rgb_grid_association(*f.capture,4,4,4,4,"original","canonical-full"),"full sidecar identity association");check(canonical_rgb_grid_association(*f.capture,4,4,2,2,"original","canonical-coarse"),"coarse sidecar identity association");check(!canonical_rgb_grid_association(*f.capture,4,4,2,2,"original","canonical-full"),"identity must bind exact dimensions");
    for(unsigned n=0;n<13;++n){Fixture bad;if(n==0)bad.sidecar->rgb[0].stride_bytes=11;if(n==1)bad.sidecar->rgb[0].bytes=bad.sidecar->rgb[0].bytes.first(2);if(n==2)bad.sidecar->support[0].format=CanonicalRgbFormat::Rgb8;if(n==3)bad.sidecar->rgb[0].format=CanonicalRgbFormat::SupportU8;if(n==4)bad.capture->packed_yuyv.resize(64);if(n==5)bad.capture->stride_bytes=16;if(n==6)bad.capture->frame_id=UINT64_MAX;if(n==7)bad.sidecar->backing.reset();if(n==8)bad.sidecar->sampling_profile="F1-remap";if(n==9)(*bad.backing)[4][0]=2;if(n==10)bad.sidecar->coarse_rectified_calibration_identity="foreign";if(n==11)bad.sidecar->rectified_calibration_identity="foreign";if(n==12)bad.capture->camera_clock_domain="unknown";reject([&]{daxa_rectified_input_reference(bad.frame(),full,coarse);});}
    Fixture offline;offline.capture->host_ingress_timestamp_ns=0;offline.capture->host_clock_domain="unknown";validate_canonical_rgb_source(*offline.capture,true);offline.capture->host_clock_domain.clear();reject([&]{validate_canonical_rgb_source(*offline.capture);});
    auto bad=coarse;bad.geometry->P2[3]*=2;reject([&]{daxa_rectified_input_reference(f.frame(),full,bad);});bad=coarse;bad.geometry->rectified.eyes[0].cx+=.1;bad.geometry->P1[2]+=.1;reject([&]{daxa_rectified_input_reference(f.frame(),full,bad);});bad=full;bad.maps[0].resize(16);reject([&]{daxa_rectified_input_reference(f.frame(),bad,coarse);});bad=full;bad.width=3;reject([&]{daxa_rectified_input_reference(f.frame(),bad,coarse);});
    Fixture only;only.sidecar->coarse_rectified_calibration_identity.clear();DaxaRectifiedInputConfig cfg;cfg.coarse_2x=false;check(!daxa_rectified_input_reference(only.frame(),full,std::nullopt,cfg).coarse,"full-only configuration explicit");
}

static DaxaRectification gpu_grid(bool high){
    auto c=grid(high);c.eye_width=128;c.source_height=64;c.width=high?128:64;c.height=high?64:32;
    auto& g=*c.geometry;for(auto& e:g.original.eyes)e={128,64,96,96,63.5,31.5,{}};
    double f=high?96:48,cx=high?63.5:31.5,cy=high?31.5:15.5;
    for(auto& e:g.rectified.eyes)e={c.width,c.height,f,f,cx,cy,{}};
    g.P1={f,0,cx,0,0,f,cy,0,0,0,1,0};g.P2={f,0,cx,-f*.1,0,f,cy,0,0,0,1,0};return c;
}
static Fixture gpu_fixture(){
    Fixture f;f.capture->source_width=256;f.capture->height=64;
    auto texture=[](unsigned x,unsigned y,unsigned ch){uint32_t v=(x+1)*0x9e3779b9u^(y+3)*0x85ebca6bu^(ch+7)*0xc2b2ae35u;v^=v>>16;v*=0x7feb352du;v^=v>>15;return uint8_t(v>>24);};
    for(unsigned p=0;p<6;++p){unsigned stride=p<4?128*3+1+2*p:128+3+2*p;auto& bytes=(*f.backing)[p];bytes.assign(stride*64,0xe7);CanonicalRgbPlane plane{bytes,128,64,stride,p<4?CanonicalRgbFormat::Rgb8:CanonicalRgbFormat::SupportU8};if(p<4){f.sidecar->rgb[p]=plane;for(unsigned y=0;y<64;++y)for(unsigned x=0;x<128;++x)for(unsigned ch=0;ch<3;++ch)bytes[y*stride+x*3+ch]=texture(x+(p&1?8:0)+(p<2?13:0),y,ch);}else{f.sidecar->support[p-4]=plane;for(unsigned y=0;y<64;++y)for(unsigned x=0;x<128;++x)bytes[y*stride+x]=x>=2&&x<126&&y>=2&&y<62?1:0;}}
    // Explicit interior support holes plus supported black pixels exercise
    // support independently of color and preserve awkward final-word reads.
    (*f.backing)[4][17*f.sidecar->support[0].stride_bytes+51]=0;
    for(unsigned ch=0;ch<3;++ch)(*f.backing)[2][20*f.sidecar->rgb[2].stride_bytes+72*3+ch]=0;
    return f;
}
static void compare_float(const std::vector<float>& actual,const std::vector<float>& expected,const char* message){
    check(actual.size()==expected.size(),message);for(size_t i=0;i<actual.size();++i){if(std::isnan(expected[i]))check(std::isnan(actual[i]),message);else check(std::isfinite(actual[i])&&std::abs(actual[i]-expected[i])<=2e-6f*std::max(1.f,std::abs(expected[i])),message);}
}
static void check_depth(DaxaDepthPipeline& pipeline,const DaxaDepthPipeline::Frame& frame,const DaxaDepthConfig& cfg,const DaxaPreprocessReadback& images){
    pipeline.check_stereo_integer(frame,images);
    auto left=pipeline.readback_stereo(frame,DaxaStereoReference::Left,false),right=pipeline.readback_stereo(frame,DaxaStereoReference::Right,false);
    DaxaDepthStereoInput input;input.disparity={left.disparity,right.disparity};input.confidence={left.confidence,right.confidence};input.validity={left.validity,right.validity};input.support={images.planes[6],images.planes[7]};
    auto expected=daxa_depth_reference(cfg,frame.metric_calibration(),input,frame.source()->frame_id),actual=pipeline.readback(frame);
    check(actual.lr_validity==expected.lr_validity&&actual.metric_validity==expected.metric_validity&&actual.provenance==expected.provenance&&actual.source_ids==expected.source_ids&&actual.age_ns==expected.age_ns,"canonical depth integer validity/source/provenance exact");
    compare_float(actual.lr_disparity,expected.lr_disparity,"canonical LR disparity actual-stereo oracle");compare_float(actual.lr_confidence,expected.lr_confidence,"canonical LR confidence oracle");compare_float(actual.depth,expected.depth,"canonical metric depth float oracle");compare_float(actual.metric_confidence,expected.metric_confidence,"canonical metric confidence oracle");
    check(std::count(actual.metric_validity.begin(),actual.metric_validity.end(),1u)>64,"analytic canonical stereo must produce nonvacuous metric support");
    auto timing=frame.timings();check(!timing.preprocessing_timings_available&&!timing.combined_gpu_timing_available&&timing.stereo_to_metric_gpu_ms>0,"external depth timing boundary explicit");
    reject([&]{pipeline.readback_images(frame);});reject([&]{pipeline.readback_stereo(frame);});reject([&]{pipeline.last_upload_cpu_ms();});
}
static void gpu_stream_contracts(const std::shared_ptr<DaxaFoundation>& context){
    // Static analytic texture cycling is a lifetime/visibility fixture. It is
    // neither a camera stream nor a throughput measurement.
    std::array<Fixture,3> variants{gpu_fixture(),gpu_fixture(),gpu_fixture()};
    for(unsigned variant=1;variant<3;++variant){auto& f=variants[variant];for(unsigned p=0;p<4;++p){auto const& plane=f.sidecar->rgb[p];for(unsigned y=0;y<plane.height;++y)for(unsigned x=0;x<plane.width*3;++x)(*f.backing)[p][size_t(y)*plane.stride_bytes+x]^=uint8_t(variant*0x39);}
        for(unsigned eye=0;eye<2;++eye){auto const& plane=f.sidecar->support[eye];(*f.backing)[4+eye][17*plane.stride_bytes+51]=1;(*f.backing)[4+eye][(21+variant)*plane.stride_bytes+50+variant]=0;}}
    for(unsigned variant=0;variant<3;++variant){variants[variant].sidecar->input_identity="analytic-static-cycling-variant-"+std::to_string(variant);variants[variant].sidecar->original_pixel_identity="original-rgb-variant-"+std::to_string(variant);variants[variant].sidecar->rectified_pixel_identity="rectified-rgb-variant-"+std::to_string(variant);}
    auto high=gpu_grid(true),low=gpu_grid(false);
    std::array<DaxaRectifiedInputReference,3> expected;
    for(unsigned variant=0;variant<3;++variant)expected[variant]=daxa_rectified_input_reference(variants[variant].frame(),high,low);
    check(expected[0].full.planes[2]!=expected[1].full.planes[2]&&expected[1].full.planes[2]!=expected[2].full.planes[2]&&expected[0].full.planes[6]!=expected[1].full.planes[6]&&expected[1].full.planes[6]!=expected[2].full.planes[6],"three deterministic texture/support variants differ");
    std::vector<RectifiedRgbFrame> prepared;prepared.reserve(300);
    for(unsigned i=0;i<300;++i){auto record=std::make_shared<CaptureFrame>(*variants[i%3].capture);auto sidecar=std::make_shared<CanonicalRgbSource>(*record->canonical_rgb);sidecar->input_identity="analytic-static-cycling-record-"+std::to_string(i);record->canonical_rgb=std::move(sidecar);record->frame_id=1000+i;record->camera_timestamp_ns=1'000'000+i*1000;record->host_ingress_timestamp_ns=2'000'000+i*1000;prepared.emplace_back(record);}
    auto setup_source=std::make_shared<CaptureFrame>(*variants[0].capture);setup_source->frame_id=999;RectifiedRgbFrame setup{setup_source};
    auto producer=std::make_unique<DaxaRectifiedInput>(high,low,DaxaRectifiedInputConfig{},context,setup);
    auto prototype=producer->enqueue(setup);auto prototype_full=prototype.full(),prototype_coarse=prototype.coarse();
    DaxaDepthConfig cfg{StereoConfig{64,32,32,4,8,32,true},1.f,2,32};auto depth=std::make_unique<DaxaDepthPipeline>(cfg,context,prototype_coarse);
    auto prototype_depth=depth->enqueue_preprocessed(prototype_coarse);DaxaRefinementConfig refine_cfg;refine_cfg.disparities=32;refine_cfg.radius=2;
    auto refine=std::make_unique<DaxaRefinement>(context,prototype_depth,prototype_full,refine_cfg);auto prototype_refined=refine->process(prototype_depth,prototype_full);
    GeometryConfig geometry{128,64,128,64};VirtualView virtual_view;virtual_view.intrinsics={96,96,63.5f,31.5f};virtual_view.source_to_virtual_right[3]=.1f;virtual_view.mode=ViewMode::Strict;
    auto view=std::make_unique<DaxaView>(geometry,virtual_view,context,prototype_refined.observations(),2);
    prototype_refined.wait();prototype_refined={};prototype_depth={};prototype_full={};prototype_coarse={};prototype={};refine->collect();depth->collect();producer->collect();
    struct Pending {DaxaRefinement::Frame refined;DaxaView::Frame projected;unsigned index=0;bool active=false;};
    std::array<Pending,2> pending;std::array<bool,300> completed{};unsigned accepted=0,finished=0,inflight=0,max_inflight=0;
    auto retire=[&](Pending& p){if(!p.active)return;p.projected.wait();auto const& source=prepared[p.index].source;
        check(p.projected.source()==source&&p.projected.source()->frame_id==1000+p.index&&p.projected.timestamp_ns()==int64_t(source->host_ingress_timestamp_ns),"stream output preserves unique source ID and current timestamps");
        check(!completed[p.index],"stream source consumed exactly once");completed[p.index]=true;++finished;
        if(p.index<3||p.index==299){auto diagnostic=refine->readback(p.refined);auto const& truth=expected[p.index%3].full;for(unsigned eye=0;eye<2;++eye){size_t n=128*64;check(diagnostic.rectified_rgb[eye]==truth.planes[2+eye],"selected static-cycling stream RGB exact");check(std::equal(truth.planes[4+eye].begin(),truth.planes[4+eye].end(),diagnostic.luma.begin()+eye*n),"selected static-cycling stream luma exact");check(std::equal(truth.planes[6+eye].begin(),truth.planes[6+eye].end(),diagnostic.support.begin()+eye*n),"selected static-cycling stream support exact");}
            unsigned valid=0;for(size_t j=0;j<diagnostic.metric_validity.size();++j)if(diagnostic.metric_validity[j]){++valid;check(diagnostic.source_ids[j]==int64_t(source->frame_id),"stream GPU metric source IDs match current record");}check(valid>0,"selected refined stream output contains real metric support");
            auto projected=view->readback(p.projected);unsigned projected_valid=0;for(size_t j=0;j<projected.validity.size();++j)if(projected.validity[j]){++projected_valid;check(projected.source_ids[j]==int64_t(source->frame_id),"stream View GPU metadata retains current source ID");}check(projected_valid>0,"selected strict View output contains real projected metric support");}
        p.projected={};p.refined={};p.active=false;--inflight;view->collect();refine->collect();depth->collect();producer->collect();
    };
    for(unsigned i=0;i<300;++i){auto& slot=pending[i%2];retire(slot);
        auto input=producer->enqueue(prepared[i]);auto full=input.full(),coarse=input.coarse();auto coarse_depth=depth->enqueue_preprocessed(coarse);slot.refined=refine->process(coarse_depth,full);slot.projected=view->project(slot.refined.observations());slot.index=i;slot.active=true;++accepted;++inflight;max_inflight=std::max(max_inflight,inflight);
        check(slot.projected.source()==prepared[i].source,"each accepted stream record reaches View without metadata substitution");
    }
    retire(pending[0]);retire(pending[1]);check(accepted==300&&finished==300&&inflight==0&&max_inflight==2&&std::all_of(completed.begin(),completed.end(),[](bool value){return value;}),"300 unique accepted/completed records with bounded2 inflight and no gaps");
    // A final independent lease case is outside the 300-record stream count.
    auto survivor_source=std::make_shared<CaptureFrame>(*variants[2].capture);survivor_source->frame_id=2000;survivor_source->camera_timestamp_ns=4'000'000;survivor_source->host_ingress_timestamp_ns=5'000'000;RectifiedRgbFrame survivor_record{survivor_source};
    auto input=producer->enqueue(survivor_record);auto full=input.full(),coarse=input.coarse();auto coarse_depth=depth->enqueue_preprocessed(coarse);auto refined=refine->process(coarse_depth,full);auto original=full.plane(DaxaPreprocessPlane::OriginalLeft).buffer();auto retained_view=view->project(refined.observations());
    input={};full={};coarse={};coarse_depth={};refined={};view.reset();refine.reset();depth.reset();
    // No facade destruction releases the source chain held by View's output.
    auto blocker=producer->enqueue(prepared[0]);blocker.full().wait();producer->collect();reject([&]{producer->enqueue(prepared[1]);});
    check(retained_view.source()==survivor_source&&retained_view.source_observations().image(DaxaObservationImage::OriginalLeft).buffer()==original,"View output independently retains original RGB/source chain after upstream facade destruction");
    blocker={};producer.reset();retained_view.wait();check(retained_view.source()->frame_id==2000&&!retained_view.buffer("output_rgb").is_empty(),"View source/output survives all facade destruction through completion");retained_view={};
    std::cout<<"static_texture_cycling_stress accepted=300 completed=300 unique_ids=300 gaps=0 max_inflight=2 view_retention=true throughput_claim=false\n";
}
static void gpu_contracts(){
    require_validation_layer();auto context=std::make_shared<DaxaFoundation>(true);
    {
        auto fixture=gpu_fixture();auto high=gpu_grid(true),low=gpu_grid(false);auto expected=daxa_rectified_input_reference(fixture.frame(),high,low);
        auto producer=std::make_unique<DaxaRectifiedInput>(high,low,DaxaRectifiedInputConfig{},context,fixture.frame());
        auto packed=producer->enqueue(fixture.frame());auto full=packed.full(),coarse=packed.coarse();full.wait();
        check(producer->requested_bytes()==daxa_rectified_input_required_bytes(high,low,{},fixture.frame()),"actual producer requested bytes equal checked preflight");
        check(producer->readback(full).planes==expected.full.planes&&producer->readback(coarse).planes==expected.coarse->planes,"GPU canonical all8 full/coarse planes exact CPU oracle");
        check(full.source()==fixture.capture&&coarse.source()==full.source()&&full.calibration()!=coarse.calibration(),"canonical grids share exact capture and retain independent calibrations");
        for(auto plane:{DaxaPreprocessPlane::OriginalLeft,DaxaPreprocessPlane::OriginalRight})check(full.plane(plane).buffer()==coarse.plane(plane).buffer(),"canonical originals share actual GPU buffer IDs");
        auto upload_timing=producer->timings(packed);check(upload_timing.total_ms>0&&upload_timing.original_ms>=0&&upload_timing.rectified_ms>=0&&upload_timing.coarse_ms>=0,"canonical producer explicit timing stages");
        DaxaDepthConfig high_cfg{StereoConfig{128,64,32,4,8,32,true},1.f,2,32};auto low_cfg=high_cfg;low_cfg.stereo.width=64;low_cfg.stereo.height=32;
        reject([&]{DaxaDepthPipeline invalid(high_cfg,std::shared_ptr<DaxaFoundation>{},full);});
        auto high_depth=std::make_unique<DaxaDepthPipeline>(high_cfg,context,full);auto low_depth=std::make_unique<DaxaDepthPipeline>(low_cfg,context,coarse);
        reject([&]{high_depth->enqueue_preprocessed(DaxaFrame{});});reject([&]{high_depth->enqueue_preprocessed(coarse);});
        auto high_out=high_depth->enqueue_preprocessed(full),low_out=low_depth->enqueue_preprocessed(coarse);high_out.wait();low_out.wait();
        check_depth(*high_depth,high_out,high_cfg,expected.full);check_depth(*low_depth,low_out,low_cfg,*expected.coarse);
        DaxaRefinementConfig refine_cfg;refine_cfg.disparities=32;refine_cfg.radius=2;
        auto refine=std::make_unique<DaxaRefinement>(context,low_out,full,refine_cfg);reject([&]{refine->process(high_out,full);});reject([&]{refine->process(low_out,coarse);});
        auto refined=refine->process(low_out,full);refined.wait();auto refined_pixels=refine->readback(refined);auto observation=refined.observations();
        for(unsigned eye=0;eye<2;++eye){check(refined_pixels.rectified_rgb[eye]==expected.full.planes[2+eye],"canonical refinement RGB must copy full GPU input exactly");size_t n=128*64;check(std::equal(expected.full.planes[4+eye].begin(),expected.full.planes[4+eye].end(),refined_pixels.luma.begin()+eye*n),"canonical refinement layered luma copy exact");check(std::equal(expected.full.planes[6+eye].begin(),expected.full.planes[6+eye].end(),refined_pixels.support.begin()+eye*n),"canonical refinement layered support copy exact");check(observation.image(eye?DaxaObservationImage::OriginalRight:DaxaObservationImage::OriginalLeft).buffer()==full.plane(eye?DaxaPreprocessPlane::OriginalRight:DaxaPreprocessPlane::OriginalLeft).buffer(),"canonical observation retains original GPU RGB");}
        check(observation.source()==fixture.capture&&observation.map_owner()==full.calibration(),"refinement exact source/full calibration owner");check(refined.timings().canonical_pack&&std::string_view(refined.timings().first_stage_name)=="canonical_pack","canonical refinement stage is device pack");
        // Prepare a distinct immutable record before submission, then exercise
        // cross-graph visibility without a diagnostic wait between consumers.
        auto fresh=gpu_fixture();fresh.capture->frame_id=19;
        for(unsigned p=0;p<4;++p){auto const& plane=fresh.sidecar->rgb[p];for(unsigned y=0;y<plane.height;++y)for(unsigned x=0;x<plane.width*3;++x)(*fresh.backing)[p][size_t(y)*plane.stride_bytes+x]^=0x5a;}
        for(unsigned eye=0;eye<2;++eye){auto const& plane=fresh.sidecar->support[eye];(*fresh.backing)[4+eye][17*plane.stride_bytes+51]=1;(*fresh.backing)[4+eye][21*plane.stride_bytes+50]=0;}
        auto prepared=fresh.frame();auto fresh_expected=daxa_rectified_input_reference(prepared,high,low);
        check(fresh_expected.full.planes[2]!=expected.full.planes[2]&&fresh_expected.full.planes[6]!=expected.full.planes[6],"second prepared record changes RGB and support truth");
        auto fresh_packed=producer->enqueue(prepared);auto fresh_full=fresh_packed.full(),fresh_coarse=fresh_packed.coarse();
        auto fresh_depth=low_depth->enqueue_preprocessed(fresh_coarse);
        auto fresh_refined=refine->process(fresh_depth,fresh_full);
        fresh_refined.wait(); // First wait in this connected producer/SGM/pack chain.
        auto fresh_pixels=refine->readback(fresh_refined);
        for(unsigned eye=0;eye<2;++eye){size_t n=128*64;check(fresh_pixels.rectified_rgb[eye]==fresh_expected.full.planes[2+eye],"unsynchronized connected chain sees changed full RGB");check(std::equal(fresh_expected.full.planes[4+eye].begin(),fresh_expected.full.planes[4+eye].end(),fresh_pixels.luma.begin()+eye*n),"unsynchronized connected chain sees changed full luma");check(std::equal(fresh_expected.full.planes[6+eye].begin(),fresh_expected.full.planes[6+eye].end(),fresh_pixels.support.begin()+eye*n),"unsynchronized connected chain sees changed full support");}
        check(fresh_refined.observations().source()==fresh.capture,"connected fresh source identity retained");
        check(producer->readback(fresh_coarse).planes==fresh_expected.coarse->planes,"connected chain coarse GPU output reflects changed input");
        fresh_refined={};fresh_depth={};fresh_full={};fresh_coarse={};fresh_packed={};refine->collect();low_depth->collect();producer->collect();
        // A separately constructed producer has equal identity strings but a
        // different immutable calibration owner; admission must reject it.
        {DaxaRectifiedInput alien(high,low,{},context,fixture.frame());auto other=alien.enqueue(fixture.frame());reject([&]{high_depth->enqueue_preprocessed(other.full());});reject([&]{refine->process(low_out,other.full());});other.full().wait();}
        auto invalid_source=std::make_shared<CaptureFrame>(*fixture.capture);invalid_source->frame_id=UINT64_MAX;auto ticket=context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);reject([&]{producer->enqueue(RectifiedRgbFrame{invalid_source});});check(ticket==context->device.latest_queue_submit_index(daxa::QUEUE_MAIN),"invalid source rejects before GPU upload/submission");
        auto other_source=std::make_shared<CaptureFrame>(*fixture.capture);other_source->frame_id=8;auto second=producer->enqueue(RectifiedRgbFrame{other_source});second.full().wait();reject([&]{refine->process(low_out,second.full());});reject([&]{producer->enqueue(fixture.frame());});
        // Leave only the exported observation and the other ring slot alive.
        auto old_generation=full.generation();packed={};full={};coarse={};high_out={};low_out={};refined={};refine.reset();high_depth.reset();low_depth.reset();producer->collect();reject([&]{producer->enqueue(fixture.frame());});
        observation={};producer->collect();auto reused=producer->enqueue(fixture.frame());check(reused.full().generation()>old_generation,"released observation permits inclusive completed slot reuse");reused.full().wait();reused={};second={};producer->collect();
        // Each producer lease category independently consumes ring capacity.
        for(unsigned kind=0;kind<4;++kind){auto first=producer->enqueue(fixture.frame());auto other=producer->enqueue(fixture.frame());first.full().wait();other.full().wait();uint64_t generation=first.full().generation();DaxaFrame held;std::optional<DaxaImageView> image;if(kind==1)held=first.full();if(kind==2)held=first.coarse();if(kind==3)image=first.full().plane(DaxaPreprocessPlane::OriginalLeft);if(kind!=0)first={};producer->collect();reject([&]{producer->enqueue(fixture.frame());});first={};held={};image.reset();producer->collect();auto next=producer->enqueue(fixture.frame());check(next.full().generation()>generation,"independent producer lease release advances generation");next.full().wait();next={};other={};producer->collect();}
        // A retained child owns the engine and buffers beyond facade lifetime.
        auto last=producer->enqueue(fixture.frame());auto survivor=last.coarse();last={};producer.reset();survivor.wait();check(survivor.source()==fixture.capture&&!survivor.plane(DaxaPreprocessPlane::OriginalRight).buffer().is_empty(),"canonical child survives producer facade destruction");survivor={};
    }
    gpu_stream_contracts(context);
    std::cout<<"canonical_device_report="<<context->report()<<'\n';
    context->close();check(context->validation_errors==0&&context->validation_warnings==0,"canonical connected GPU core/sync validation clean through teardown");
}
int main(int argc,char** argv){try{
    check(argc==2&&(std::string_view(argv[1])=="--cpu-only"||std::string_view(argv[1])=="--gpu"),"expected --cpu-only or --gpu");
    offline_admission_contracts();cpu_oracle();
    DaxaRectifiedInputConfig config;config.validate();check(config.slots==2,"default producer uses two slots");
    for(unsigned n:{0u,1u,3u}){auto bad=config;bad.slots=n;reject([&]{bad.validate();});}
    if(std::string_view(argv[1])=="--gpu"){gpu_contracts();std::cout<<"Rectified input connected GPU contracts passed; metric_float_relative_bound=2e-6; stress_300_frames=true\n";}else std::cout<<"Rectified input CPU API contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
