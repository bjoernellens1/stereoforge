#include <stereoforge/daxa_host_export.hpp>
#include <stereoforge/daxa_rectified_input.hpp>
#include <stereoforge/daxa_depth.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
using namespace stereoforge;
static void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F>static void reject(F fn){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}check(caught,"expected host export rejection");}
// Same immutable analytic 128x64 fixture as daxa_rectified_input contracts.
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

static void memory_contract(const DaxaHostExport& output,unsigned requested_planes){
 auto memory=output.memory();constexpr auto required=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
 check((memory.property_flags&required)==required,"host export memory must be visible and coherent");
 check(memory.host_cached==bool(memory.property_flags&VK_MEMORY_PROPERTY_HOST_CACHED_BIT),"host export cached metadata must match actual flags");
 check(memory.bytes_per_slot==size_t(requested_planes)*128*64*4,"requested payload bytes per slot exact");
 check(output.requested_bytes()==size_t(2)*requested_planes*128*64*4,"requested bytes across two slots exact");
}
static void gpu_contracts(bool validation){
 auto context=std::make_shared<DaxaFoundation>(validation);
 auto fixture=gpu_fixture();auto high=gpu_grid(true);DaxaRectifiedInputConfig input_config;input_config.coarse_2x=false;
 auto expected_rgb=daxa_rectified_input_reference(fixture.frame(),high,std::nullopt,input_config).full.planes[2];
 DaxaRectifiedInput input(high,std::nullopt,input_config,context,fixture.frame());
 auto producer=input.enqueue(fixture.frame());auto prototype=producer.full();
 DaxaHostExport rgb(context,prototype,{2,false,true});memory_contract(rgb,1);
 {auto frame=rgb.enqueue(prototype);frame.wait();check(frame.depth_bytes().empty()&&frame.rgba_bytes().size()==expected_rgb.size()*4,"RGB-only exact extent");check(std::memcmp(frame.rgba_bytes().data(),expected_rgb.data(),frame.rgba_bytes().size())==0,"RGB-only native packed bytes exact");}
 rgb.drain();
 DaxaDepthConfig config{StereoConfig{128,64,32,4,8,32,true},1.f,2,32};
 DaxaDepthPipeline pipeline(config,context,prototype);prototype.wait();prototype={};producer={};input.collect();
 auto native=input.enqueue(fixture.frame());auto depth=pipeline.enqueue_preprocessed(native.full());auto observation=depth.observations();
 auto expected=pipeline.readback(depth);check(std::count(expected.metric_validity.begin(),expected.metric_validity.end(),1u)>64,"nonvacuous analytic metric support");
 DaxaHostExport output(context,observation,{2,true,true});memory_contract(output,2);
 auto a=output.enqueue(observation),b=output.enqueue(observation);reject([&]{output.enqueue(observation);});
 check(a.producer_ticket()==observation.submit_ticket()&&a.copy_ticket()>a.producer_ticket(),"producer and copy ticket boundaries");
 a.wait();b.wait();output.collect();
 auto compare=[&](const DaxaHostExport::Frame& frame){
  auto bytes=size_t(128)*64*4;check(frame.width()==128&&frame.height()==64&&frame.depth_bytes().size()==bytes&&frame.rgba_bytes().size()==bytes,"only left plane extents");
  check(std::memcmp(frame.depth_bytes().data(),expected.depth.data(),bytes)==0,"byte exact depth including NaN payload");
  check(std::memcmp(frame.rgba_bytes().data(),expected_rgb.data(),bytes)==0,"byte exact rectified-left RGBA");
 };
 compare(a);compare(b);a={};auto reused=output.enqueue(observation);reused.wait();compare(reused);b={};reused={};output.collect();
 {DaxaHostExport only_depth(context,observation,{2,true,false});memory_contract(only_depth,1);auto result=only_depth.enqueue(observation);result.wait();check(result.rgba_bytes().empty(),"disabled RGB omitted");check(result.depth_bytes().size()==size_t(128)*64*4,"depth only extent");}
 DaxaHostExport::Frame retained;{DaxaHostExport temporary(context,observation,{2,true,true});retained=temporary.enqueue(observation);}retained.wait();compare(retained);retained={};
 observation={};depth={};native={};pipeline.drain();input.collect();
 // Static analytic source cycling tests resource ownership and queue visibility.
 // It is not a fresh-camera or throughput benchmark.
 for(unsigned iteration=0;iteration<30;++iteration){
  auto in=input.enqueue(fixture.frame());auto produced=pipeline.enqueue_preprocessed(in.full());auto observed=produced.observations();auto host=output.enqueue(observed);
  produced={};observed={};in={};host.wait();compare(host);output.collect();pipeline.collect();input.collect();
 }
 output.drain();rgb.drain();pipeline.drain();input.collect();
 check(context->validation_errors.load()==0,"Vulkan validation errors");
 auto selected=output.memory();
 std::cout<<"native host export GPU cases passed: 128x64, 30 static cycles, exact selective bytes"
  <<" memory_type_index="<<selected.type_index<<" property_flags="<<selected.property_flags
  <<" host_cached="<<selected.host_cached<<" bytes_per_slot="<<selected.bytes_per_slot
  <<" requested_bytes="<<output.requested_bytes()<<'\n';
}
int main(int argc,char** argv){try{
 bool gpu=false,validation=false;for(int i=1;i<argc;++i){std::string_view argument(argv[i]);if(argument=="--gpu")gpu=true;else if(argument=="--validation")validation=true;else throw std::invalid_argument("unknown host export test argument");}
 check(gpu,"native export contracts require explicit --gpu; CPU mocks run through pytest");gpu_contracts(validation);return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
