#include <stereoforge/daxa_stereo.hpp>
#include <stereoforge/daxa_preprocess.hpp>
#include <daxa/utils/task_graph.hpp>
#include <daxa/c/device.h>
#include <daxa/c/types.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <thread>
namespace stereoforge {
namespace {
constexpr std::array<std::array<int,2>,8> directions{{{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,-1},{1,-1},{-1,1}}};
void require(bool v,const char* m){if(!v)throw std::invalid_argument(m);}
float invalid_float(){return std::bit_cast<float>(uint32_t{0x7fc00000});}
bool inside(StereoConfig c,int x,int y){return x>=0&&y>=0&&x<int(c.width)&&y<int(c.height);}
uint32_t step(uint32_t cost,uint32_t own,uint32_t lo,uint32_t hi,uint32_t minimum,StereoConfig c,bool first){auto best=std::min({own,lo+c.p1,hi+c.p1,minimum+c.p2});return cost+(first?0:best-minimum);}
uint32_t line_count(StereoConfig c,int dx,int dy){return !dy?c.height:!dx?c.width:c.width+c.height-1;}
std::array<int,2> line_start(StereoConfig c,int dx,int dy,uint32_t line){if(!dy)return {dx>0?0:int(c.width)-1,int(line)};if(!dx)return {int(line),dy>0?0:int(c.height)-1};if(line<c.width)return {int(line),dy>0?0:int(c.height)-1};return {dx>0?0:int(c.width)-1,dy>0?int(line-c.width)+1:int(c.height)-2-int(line-c.width)};}
}
std::array<uint32_t,3> DaxaStereo::dispatch_grid(uint64_t groups,std::array<uint32_t,3> limits){require(groups&&limits[0]&&limits[1]&&limits[2],"invalid compute dispatch limits");auto x=uint32_t(std::min<uint64_t>(groups,limits[0]));auto y=(groups+x-1)/x;require(y<=limits[1],"compute grid exceeds actual 2D dispatch limits");return {x,uint32_t(y),1};}
void DaxaStereo::validate_shape(StereoConfig c,unsigned slots,std::optional<MetricCalibration> metric,DaxaSgmVariant variant){
    c.validate();require(variant==DaxaSgmVariant::Baseline||variant==DaxaSgmVariant::Wave||variant==DaxaSgmVariant::Packed||variant==DaxaSgmVariant::PackedSpecialized||variant==DaxaSgmVariant::PackedD128||variant==DaxaSgmVariant::PackedQuad||variant==DaxaSgmVariant::PackedShuffle,"unknown Daxa SGM variant");require(variant!=DaxaSgmVariant::PackedD128||c.disparities==128,"packed-d128 requires exactly128 disparities");require(slots>=1&&slots<=2,"Daxa stereo requires one or two slots");auto n=uint64_t(c.width)*c.height;require(n<=uint64_t(UINT32_MAX)/c.disparities,"stereo volume exceeds uint32 indexing bound");
    auto storage=(variant==DaxaSgmVariant::Packed||variant==DaxaSgmVariant::PackedSpecialized||variant==DaxaSgmVariant::PackedD128||variant==DaxaSgmVariant::PackedQuad||variant==DaxaSgmVariant::PackedShuffle)?n*((c.disparities+32/packed_bits(c)-1)/(32/packed_bits(c)))*4:n*c.disparities*8;
    require(n*16+slots*(storage+n*40+256)<=2ull*1024*1024*1024,"Daxa stereo retained allocation exceeds 2 GiB ceiling");if(metric)metric->validate();
}
void DaxaStereo::validate(StereoConfig c,std::span<const uint32_t> left,std::span<const uint32_t> right,unsigned slots,std::optional<MetricCalibration> metric,DaxaSgmVariant variant){
    validate_shape(c,slots,metric,variant);auto n=uint64_t(c.width)*c.height;require(left.size()==n&&right.size()==n,"retained luma dimensions mismatch");
    require(std::all_of(left.begin(),left.end(),[](auto x){return x<=255;})&&std::all_of(right.begin(),right.end(),[](auto x){return x<=255;}),"uint32 luma values must be in [0,255]");
}
uint32_t DaxaStereo::packed_bits(StereoConfig c){c.validate();auto bound=uint64_t(c.paths)*c.p2;if(bound<=255)return 8;if(bound<=65535)return 16;throw std::invalid_argument("packed normalized penalty bound exceeds 16 bits; baseline/wave remains available");}
std::vector<uint32_t> daxa_sgm_packed_model(StereoConfig c,std::span<const uint32_t> costs,unsigned lanes){
    require(lanes==32||lanes==64,"packed model requires full 32/64 lanes");auto bits=DaxaStereo::packed_bits(c),components=32/bits,words=(c.disparities+components-1)/components,chunks=(words+lanes-1)/lanes;require(costs.size()==size_t(c.width)*c.height*c.disparities,"packed model cost shape mismatch");require(std::all_of(costs.begin(),costs.end(),[](auto x){return x<=10000;}),"packed model cost bound");
    std::vector<uint32_t> packed(size_t(c.width)*c.height*words);std::array<std::array<uint32_t,64>,8> previous{},next{};
    for(unsigned path=0;path<c.paths;++path){auto [dx,dy]=directions[path];for(uint32_t line=0;line<line_count(c,dx,dy);++line){for(auto& r:previous)r.fill(0);auto [x0,y0]=line_start(c,dx,dy,line);bool first=true;
        for(int x=x0,y=y0;inside(c,x,y);x+=dx,y+=dy){uint32_t minimum=1000000;for(unsigned lane=0;lane<lanes;++lane)for(unsigned k=0;k<chunks;++k)for(unsigned component=0;component<components;++component)if((k*lanes+lane)*components+component<c.disparities)minimum=std::min(minimum,previous[k*components+component][lane]);
            for(unsigned k=0;k<chunks;++k)for(unsigned lane=0;lane<lanes;++lane){uint32_t addition=0;auto word=k*lanes+lane;for(unsigned component=0;component<components;++component){auto d=word*components+component,reg=k*components+component;if(d>=c.disparities){next[reg][lane]=0;continue;}uint32_t lo=1000000,hi=1000000;
                    if(d)lo=component?previous[reg-1][lane]:lane?previous[reg-1+components][lane-1]:previous[reg-1][lanes-1];
                    if(d+1<c.disparities)hi=component+1<components?previous[reg+1][lane]:lane+1<lanes?previous[reg+1-components][lane+1]:previous[reg+1][0];
                    auto cost=costs[(size_t(y)*c.width+x)*c.disparities+d];next[reg][lane]=step(cost,previous[reg][lane],lo,hi,minimum,c,first);auto residual=next[reg][lane]-cost;require(residual<=c.p2,"normalized SGM residual exceeds proof bound");addition|=residual<<(component*bits);
                }if(word<words)packed[(size_t(y)*c.width+x)*words+word]+=addition;
            }previous=next;first=false;
        }
    }}return packed;
}
std::vector<uint32_t> daxa_sgm_d128_model(StereoConfig c,std::span<const uint32_t> costs,unsigned lanes){
    require(c.disparities==128,"D128 model requires exactly128 disparities");require(lanes==32||lanes==64,"packed model requires full 32/64 lanes");auto bits=DaxaStereo::packed_bits(c),components=32/bits,words=(c.disparities+components-1)/components,chunks=(words+lanes-1)/lanes;require(chunks*components<=4,"D128 register bound");require(costs.size()==size_t(c.width)*c.height*c.disparities,"packed model cost shape mismatch");require(std::all_of(costs.begin(),costs.end(),[](auto x){return x<=10000;}),"packed model cost bound");
    std::vector<uint32_t> packed(size_t(c.width)*c.height*words);std::array<std::array<uint32_t,64>,4> previous{},next{};
    for(unsigned path=0;path<c.paths;++path){auto [dx,dy]=directions[path];for(uint32_t line=0;line<line_count(c,dx,dy);++line){for(auto& r:previous)r.fill(0);auto [x0,y0]=line_start(c,dx,dy,line);bool first=true;
        for(int x=x0,y=y0;inside(c,x,y);x+=dx,y+=dy){uint32_t minimum=1000000;for(unsigned lane=0;lane<lanes;++lane)for(unsigned k=0;k<chunks;++k)for(unsigned component=0;component<components;++component)if((k*lanes+lane)*components+component<c.disparities)minimum=std::min(minimum,previous[k*components+component][lane]);
            for(unsigned k=0;k<chunks;++k)for(unsigned lane=0;lane<lanes;++lane){uint32_t addition=0;auto word=k*lanes+lane;for(unsigned component=0;component<components;++component){auto d=word*components+component,reg=k*components+component;if(d>=c.disparities){next[reg][lane]=0;continue;}uint32_t lo=1000000,hi=1000000;
                    if(d)lo=component?previous[reg-1][lane]:lane?previous[reg-1+components][lane-1]:previous[reg-1][lanes-1];
                    if(d+1<c.disparities)hi=component+1<components?previous[reg+1][lane]:lane+1<lanes?previous[reg+1-components][lane+1]:previous[reg+1][0];
                    auto cost=costs[(size_t(y)*c.width+x)*c.disparities+d];next[reg][lane]=step(cost,previous[reg][lane],lo,hi,minimum,c,first);auto residual=next[reg][lane]-cost;require(residual<=c.p2,"normalized SGM residual exceeds proof bound");addition|=residual<<(component*bits);
                }if(word<words)packed[(size_t(y)*c.width+x)*words+word]+=addition;
            }previous=next;first=false;
        }
    }}return packed;
}
std::vector<uint32_t> daxa_wave_min_model(std::span<const uint32_t> values){
    require(values.size()==32||values.size()==64,"minimum model requires full32/64waves");std::vector<uint32_t> lanes(values.begin(),values.end());
    for(unsigned offset=1;offset<lanes.size();offset*=2){auto previous=lanes;for(unsigned lane=0;lane<lanes.size();++lane){auto partner=lane^offset;require(partner<lanes.size(),"minimum shuffle source must be active");lanes[lane]=std::min(previous[lane],previous[partner]);}}return lanes;
}
DaxaQuadSelection daxa_sgm_quad_model(StereoConfig c,std::span<const uint32_t> sums,unsigned lanes){
    DaxaStereo::validate_shape(c,1,std::nullopt,DaxaSgmVariant::PackedQuad);require(lanes==32||lanes==64,"quad model requires full32/64waves");size_t n=size_t(c.width)*c.height;require(sums.size()==n*c.disparities,"quad sum shape mismatch");require(std::all_of(sums.begin(),sums.end(),[&](auto x){return x<=c.paths*20000;}),"quad sum bound");
    DaxaQuadSelection out;out.best.resize(n);out.best_cost.resize(n);out.second.resize(n);
    for(size_t group=0;group<(n+15)/16;++group)for(unsigned subgroup=0;subgroup<64/lanes;++subgroup){
        std::array<uint32_t,64> best,best_cost,second;best.fill(UINT32_MAX);best_cost.fill(UINT32_MAX);second.fill(UINT32_MAX);
        for(unsigned lane=0;lane<lanes;++lane){size_t pixel=group*16+subgroup*(lanes/4)+lane/4;if(pixel>=n)continue;for(unsigned d=lane%4;d<c.disparities;d+=4){auto value=sums[pixel*c.disparities+d];if(value<best_cost[lane]||(value==best_cost[lane]&&d<best[lane])){second[lane]=best_cost[lane];best_cost[lane]=value;best[lane]=d;}else second[lane]=std::min(second[lane],value);}}
        for(unsigned stride=2;stride;stride/=2){auto old_best=best,old_cost=best_cost,old_second=second;for(unsigned lane=0;lane<lanes;++lane){unsigned source=lane^stride;require(source<lanes&&source/4==lane/4,"quad shuffle active source stays in pixel");auto other_cost=old_cost[source],other_best=old_best[source],other_second=old_second[source];if(lane%4<stride){if(other_cost<old_cost[lane]||(other_cost==old_cost[lane]&&other_best<old_best[lane])){second[lane]=std::min(other_second,old_cost[lane]);best_cost[lane]=other_cost;best[lane]=other_best;}else second[lane]=std::min(old_second[lane],other_cost);}}}
        for(unsigned lane=0;lane<lanes;lane+=4){size_t pixel=group*16+subgroup*(lanes/4)+lane/4;if(pixel<n){out.best[pixel]=best[lane];out.best_cost[pixel]=best_cost[lane];out.second[pixel]=c.disparities==1?best_cost[lane]:second[lane];}}
    }return out;
}
std::array<std::vector<uint32_t>,2> daxa_stereo_fixture(StereoConfig c){c.validate();std::array<std::vector<uint32_t>,2> out;for(auto& v:out)v.resize(size_t(c.width)*c.height);uint32_t state=1234567;for(auto& x:out[0]){state^=state<<13;state^=state>>17;state^=state<<5;x=state&255;}for(uint32_t y=0;y<c.height;++y)for(uint32_t x=0;x<c.width;++x)out[1][y*c.width+x]=x+4<c.width?out[0][y*c.width+x+4]:0;return out;}
DaxaStereoReadback daxa_stereo_reference(StereoConfig c,std::span<const uint32_t> left,std::span<const uint32_t> right,std::optional<MetricCalibration> metric){
    DaxaStereo::validate(c,left,right,1,metric);size_t n=size_t(c.width)*c.height;std::array<std::vector<uint32_t>,2> desc,valid;
    for(unsigned eye=0;eye<2;++eye){auto image=eye?right:left;desc[eye].resize(n);valid[eye].resize(n);for(uint32_t y=2;y+2<c.height;++y)for(uint32_t x=2;x+2<c.width;++x){uint32_t bits=0,bit=0,i=y*c.width+x;for(int oy=-2;oy<=2;++oy)for(int ox=-2;ox<=2;++ox){if(!ox&&!oy)continue;if(image[(int(y)+oy)*c.width+int(x)+ox]<image[i])bits|=1u<<bit;++bit;}desc[eye][i]=bits;valid[eye][i]=1;}}
    DaxaStereoReadback out;out.census=desc;out.census_valid=valid;out.costs.resize(n*c.disparities);out.sums.resize(out.costs.size());
    for(size_t i=0;i<n;++i)for(uint32_t d=0;d<c.disparities;++d){bool ok=i%c.width>=d&&valid[0][i]&&valid[1][i-d];out.costs[i*c.disparities+d]=ok?std::popcount(desc[0][i]^desc[1][i-d]):10000;}
    std::array<uint32_t,256> previous{},next{};
    for(unsigned path=0;path<c.paths;++path){auto [dx,dy]=directions[path];
        // Independent scalar start enumeration: predecessor outside image.
        for(uint32_t sy=0;sy<c.height;++sy)for(uint32_t sx=0;sx<c.width;++sx){if(inside(c,int(sx)-dx,int(sy)-dy))continue;previous.fill(0);bool first=true;
            for(int x=sx,y=sy;inside(c,x,y);x+=dx,y+=dy){auto minimum=*std::min_element(previous.begin(),previous.begin()+c.disparities);auto offset=(size_t(y)*c.width+x)*c.disparities;
                for(uint32_t d=0;d<c.disparities;++d){next[d]=step(out.costs[offset+d],previous[d],d?previous[d-1]:1000000,d+1<c.disparities?previous[d+1]:1000000,minimum,c,first);out.sums[offset+d]+=next[d];}previous=next;first=false;}
        }
    }
    out.disparity.resize(n,invalid_float());out.confidence.resize(n);out.validity.resize(n);if(metric){out.depth.resize(n,invalid_float());out.depth_validity.resize(n);}
    for(size_t i=0;i<n;++i){size_t offset=i*c.disparities;uint32_t best=UINT32_MAX,cost=UINT32_MAX,second=UINT32_MAX;
        for(uint32_t d=0;d<c.disparities;++d){auto value=out.sums[offset+d];if(value<cost){second=cost;cost=value;best=d;}else second=std::min(second,value);}if(c.disparities==1)second=cost;
        bool ok=best>0&&cost<10000&&second>cost;float disparity=float(best);
        if(c.subpixel&&best>0&&best+1<c.disparities){float lo=float(out.sums[offset+best-1]),hi=float(out.sums[offset+best+1]),denom=lo+hi-2*float(cost);if(denom>0)disparity+=.5f*(lo-hi)/denom;}
        if(ok){out.disparity[i]=disparity;out.validity[i]=1;out.confidence[i]=float(second-cost)/float(std::max(second,1u));}
        if(metric&&ok){float corrected=disparity-metric->cx_offset,z=metric->fx*metric->baseline_m/corrected;if(std::isfinite(corrected)&&corrected>0&&std::isfinite(z)&&z>0){out.depth[i]=z;out.depth_validity[i]=1;}}
    }return out;
}
std::vector<uint32_t> daxa_sgm_wave_model(StereoConfig c,std::span<const uint32_t> costs,unsigned lanes){
    c.validate();require(lanes==32||lanes==64,"wave CPU model requires 32 or 64 lanes");require(costs.size()==size_t(c.width)*c.height*c.disparities,"cost volume dimensions mismatch");require(std::all_of(costs.begin(),costs.end(),[](auto x){return x<=10000;}),"SGM costs exceed fixed integer contract");
    std::vector<uint32_t> sums(costs.size());unsigned chunks=(c.disparities+lanes-1)/lanes;
    std::array<std::array<uint32_t,64>,8> prev{},next{};
    for(unsigned path=0;path<c.paths;++path){auto [dx,dy]=directions[path];for(uint32_t line=0;line<line_count(c,dx,dy);++line){for(auto& v:prev)v.fill(0);auto [x0,y0]=line_start(c,dx,dy,line);bool first=true;
        for(int x=x0,y=y0;inside(c,x,y);x+=dx,y+=dy){uint32_t minimum=1000000;for(unsigned lane=0;lane<lanes;++lane){uint32_t local=1000000;for(unsigned k=0;k<chunks;++k)if(k*lanes+lane<c.disparities)local=std::min(local,prev[k][lane]);minimum=std::min(minimum,local);}
            for(unsigned k=0;k<chunks;++k)for(unsigned lane=0;lane<lanes;++lane){auto d=k*lanes+lane;if(d>=c.disparities)continue;uint32_t lo=d?prev[k-(lane==0)][lane?lane-1:lanes-1]:1000000,hi=d+1<c.disparities?prev[k+(lane+1==lanes)][lane+1==lanes?0:lane+1]:1000000;auto index=(size_t(y)*c.width+x)*c.disparities+d;next[k][lane]=step(costs[index],prev[k][lane],lo,hi,minimum,c,first);sums[index]+=next[k][lane];}prev=next;first=false;
        }
    }}return sums;
}
namespace {
enum B:unsigned { Left,Right,LeftDesc,RightDesc,LeftValid,RightValid,Costs,Sums,Disparity,Confidence,Validity,Depth,DepthValid,Counters,Penalties,Params,BufferCount };
constexpr const char* resource_names[BufferCount]={"left_luma","right_luma","left_census","right_census","left_census_valid","right_census_valid","dense_costs","dense_sums","disparity","confidence","validity","metric_depth","metric_validity","inspect_counters","packed_penalties","immutable_parameters"};
constexpr const char* entries[]={"census","cost","aggregate_baseline","aggregate_wave","select_output","depth_convert","inspect","clear_packed","aggregate_packed8","aggregate_packed16","select_packed","aggregate_specialized8","aggregate_specialized16","select_specialized8","select_specialized16","aggregate_d1288","aggregate_d12816","select_quad8","select_quad16","aggregate_shuffle8","aggregate_shuffle16"};
struct Parameters {std::array<uint64_t,14> address{};uint32_t width,height,disparities,paths,p1,p2,subpixel,metric;float fx,baseline_m,cx_offset;uint32_t packing_bits=0;uint64_t penalties_address=0;};
static_assert(sizeof(Parameters)==168&&offsetof(Parameters,width)==112&&offsetof(Parameters,fx)==144);
struct Push {uint64_t parameters;uint32_t pass=0,groups_x=1;};static_assert(sizeof(Push)==16);
std::vector<uint32_t> load_shader(const char* name){auto path=std::filesystem::path(SF_DAXA_STEREO_SPIRV_DIRECTORY)/(std::string("daxa-stereo-")+name+".spv");std::ifstream file(path,std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("missing offline stereo SPIR-V: "+path.string());auto bytes=file.tellg();require(bytes>0&&bytes%4==0,"invalid stereo SPIR-V bytes");std::vector<uint32_t> words(size_t(bytes)/4);file.seekg(0);file.read(reinterpret_cast<char*>(words.data()),bytes);if(!file)throw std::runtime_error("SPIR-V read failed");return words;}
unsigned buffer_index(DaxaStereoPlane plane){switch(plane){case DaxaStereoPlane::Disparity:return Disparity;case DaxaStereoPlane::Confidence:return Confidence;case DaxaStereoPlane::Validity:return Validity;case DaxaStereoPlane::Depth:return Depth;case DaxaStereoPlane::DepthValidity:return DepthValid;case DaxaStereoPlane::Costs:return Costs;case DaxaStereoPlane::Sums:return Sums;case DaxaStereoPlane::LeftCensus:return LeftDesc;case DaxaStereoPlane::RightCensus:return RightDesc;case DaxaStereoPlane::LeftCensusValid:return LeftValid;case DaxaStereoPlane::RightCensusValid:return RightValid;case DaxaStereoPlane::PackedPenalties:return Penalties;}throw std::invalid_argument("unknown stereo plane");}
}
struct DaxaStereo::Slot {
    std::array<daxa::BufferId,BufferCount> buffers{};
    std::array<daxa::ExternalTaskBuffer,BufferCount> external;
    std::unique_ptr<daxa::TaskGraph> graph;
    daxa::TimelineQueryPool queries;
    uint64_t submit=0,generation=0;
    DaxaFrame input;CaptureLease source;
    unsigned started=0,finished=0;
};
struct DaxaStereo::Impl {
    std::shared_ptr<DaxaFoundation> context;
    StereoConfig config;
    DaxaSgmVariant variant;
    std::shared_ptr<const DaxaRectification> calibration;
    std::optional<MetricCalibration> metric;
    std::array<daxa::ComputePipeline,21> pipelines;
    std::vector<daxa::BufferId> owned;
    std::vector<daxa::MemoryBlock> memory;
    std::vector<std::shared_ptr<Slot>> slots;
    uint64_t allocated=0,next_generation=1,storage_range_limit=0;
    uint32_t timestamp_bits=0;
    float timestamp_period=0;
    unsigned stages=0,query_count=0;
    uint32_t wave_size=0,packing=0;std::array<uint32_t,3> group_limits{};
    const std::thread::id submission_thread=std::this_thread::get_id();
    void require_thread()const{require(std::this_thread::get_id()==submission_thread,"Daxa stereo requires its single host submission thread");}
    Impl(StereoConfig c,DaxaSgmVariant v,std::optional<MetricCalibration> m,std::shared_ptr<DaxaFoundation> shared,bool validation,uint32_t requested_wave):context(shared?std::move(shared):std::make_shared<DaxaFoundation>(validation)),config(c),variant(v),metric(m){
        require(v==DaxaSgmVariant::Baseline||v==DaxaSgmVariant::Wave||v==DaxaSgmVariant::Packed||v==DaxaSgmVariant::PackedSpecialized||v==DaxaSgmVariant::PackedD128||v==DaxaSgmVariant::PackedQuad||v==DaxaSgmVariant::PackedShuffle,"unknown Daxa SGM variant");require(!validation||context->messenger,"shared foundation does not have required validation enabled");stages=4+c.paths+unsigned(bool(metric));query_count=2+stages*2;
        VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};VkPhysicalDeviceSubgroupSizeControlProperties control{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};subgroup.pNext=&control;VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};properties.pNext=&subgroup;
        auto physical=daxa_dvc_get_vk_physical_device(context->device.get());vkGetPhysicalDeviceProperties2(physical,&properties);timestamp_period=properties.properties.limits.timestampPeriod;
        packing=(v==DaxaSgmVariant::Packed||v==DaxaSgmVariant::PackedSpecialized||v==DaxaSgmVariant::PackedD128||v==DaxaSgmVariant::PackedQuad||v==DaxaSgmVariant::PackedShuffle)?DaxaStereo::packed_bits(c):0;auto largest=packing?uint64_t(c.width)*c.height*((c.disparities+32/packing-1)/(32/packing))*4:uint64_t(c.width)*c.height*c.disparities*4;
        auto const& limits=properties.properties.limits;storage_range_limit=limits.maxStorageBufferRange;require(std::max(largest,uint64_t(sizeof(Parameters)))<=limits.maxStorageBufferRange,"dense stereo buffer exceeds actual maxStorageBufferRange");require(limits.maxComputeWorkGroupInvocations>=256&&limits.maxComputeWorkGroupSize[0]>=256,"stereo selection requires 256-invocation workgroups");std::copy_n(limits.maxComputeWorkGroupCount,3,group_limits.begin());
        if(v!=DaxaSgmVariant::Baseline){VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};f.pNext=&f13;vkGetPhysicalDeviceFeatures2(physical,&f);require(f13.subgroupSizeControl&&f13.computeFullSubgroups,"wave SGM requires enabled Daxa-required subgroup size/full subgroup features");require((control.requiredSubgroupSizeStages&VK_SHADER_STAGE_COMPUTE_BIT)!=0,"required subgroup size unavailable for compute");require((subgroup.supportedStages&VK_SHADER_STAGE_COMPUTE_BIT)!=0,"compute subgroups unavailable");require((subgroup.supportedOperations&(VK_SUBGROUP_FEATURE_ARITHMETIC_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_BIT))==(VK_SUBGROUP_FEATURE_ARITHMETIC_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_BIT),"wave SGM requires subgroup arithmetic and shuffle");wave_size=requested_wave?requested_wave:subgroup.subgroupSize;require((wave_size==32||wave_size==64)&&wave_size>=control.minSubgroupSize&&wave_size<=control.maxSubgroupSize&&64/wave_size<=control.maxComputeWorkgroupSubgroups,"requested full subgroup size 32/64 unsupported");}else require(!requested_wave,"wave size supplied for baseline variant");
        uint32_t count=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);std::vector<VkQueueFamilyProperties> queues(count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,queues.data());uint32_t family=UINT32_MAX;auto result=daxa_dvc_get_vk_queue(context->device.get(),{DAXA_QUEUE_TYPE_MAIN,0},nullptr,&family);require(result==DAXA_RESULT_SUCCESS&&family<count,"cannot identify Daxa main queue");timestamp_bits=queues[family].timestampValidBits;require(timestamp_bits>0,"Daxa main queue does not support timestamps");
    }
    ~Impl(){auto& d=context->device;d.wait_idle();slots.clear();pipelines={};for(auto b:owned)if(d.is_id_valid(b))d.destroy_buffer(b);d.collect_garbage();memory.clear();}
    daxa::BufferId allocate(size_t size,const std::string& name,bool host=false){auto& d=context->device;daxa::BufferInfo info{.size=size,.name=name};daxa::BufferId b;if(host){auto m=d.create_memory({.requirements=context->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});b=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=m});memory.push_back(std::move(m));}else b=d.create_buffer(info);owned.push_back(b);allocated+=size;return b;}
    void setup(std::span<const uint32_t> left,std::span<const uint32_t> right,unsigned count,const DaxaFrame* prototype=nullptr);
    void completed(Frame const& f)const{f.validate();require(f.slot_->submit<context->device.oldest_pending_submit_index(),"GPU completion required before query access");}
};
void DaxaStereo::Impl::setup(std::span<const uint32_t> left,std::span<const uint32_t> right,unsigned count,const DaxaFrame* prototype){
    auto& d=context->device;size_t n=size_t(config.width)*config.height,volume=n*config.disparities;std::array<daxa::BufferId,2> inputs;
    if(prototype){inputs[0]=prototype->plane(DaxaPreprocessPlane::LeftLuma).buffer();inputs[1]=prototype->plane(DaxaPreprocessPlane::RightLuma).buffer();}else{auto recorder=d.create_command_recorder({.name="one-time retained stereo luma upload"});
    for(unsigned eye=0;eye<2;++eye){auto image=eye?right:left;inputs[eye]=allocate(n*4,resource_names[eye]);auto staging=allocate(n*4,std::string("setup_staging_")+resource_names[eye],true);std::memcpy(d.buffer_host_address(staging).value(),image.data(),n*4);recorder.copy_buffer_to_buffer({.src_buffer=staging,.dst_buffer=inputs[eye],.size=n*4});}
    recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ});auto commands=recorder.complete_current_commands();d.submit_commands({.command_lists={&commands,1}});d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=d.latest_queue_submit_index(daxa::QUEUE_MAIN),.timeout=30'000'000'000ull});d.collect_garbage();
    }
    for(unsigned i=0;i<pipelines.size();++i){
        bool specialized=variant==DaxaSgmVariant::PackedSpecialized,d128=variant==DaxaSgmVariant::PackedD128,quad=variant==DaxaSgmVariant::PackedQuad,shuffle=variant==DaxaSgmVariant::PackedShuffle;
        if((i==2&&variant!=DaxaSgmVariant::Baseline)||(i==3&&variant!=DaxaSgmVariant::Wave)||(i==7&&!packing)||(i>=8&&i<=10&&variant!=DaxaSgmVariant::Packed)||(i>=11&&i<=12&&!specialized&&!quad)||(i>=13&&i<=14&&!specialized&&!d128&&!shuffle)||(i>=15&&i<=16&&!d128)||(i>=17&&i<=18&&!quad)||(i>=19&&!shuffle)||((i==8||i==11||i==13||i==15||i==17||i==19)&&packing!=8)||((i==9||i==12||i==14||i==16||i==18||i==20)&&packing!=16))continue;
        auto words=load_shader(entries[i]);daxa::ShaderInfo shader{.byte_code=words.data(),.byte_code_size=uint32_t(words.size())};if(i==3||i==8||i==9||i==11||i==12||i==15||i==16||i==17||i==18||i==19||i==20){shader.create_flags=daxa::ShaderCreateFlagBits::REQUIRE_FULL_SUBGROUPS;shader.required_subgroup_size=wave_size;}pipelines[i]=d.create_compute_pipeline({.shader_info=shader,.push_constant_size=sizeof(Push),.name=entries[i]});}

    slots.reserve(count);
    for(unsigned index=0;index<count;++index){auto s=std::make_shared<Slot>();s->buffers[Left]=inputs[0];s->buffers[Right]=inputs[1];auto prefix="stereo_slot_"+std::to_string(index)+"_";
        for(unsigned b=LeftDesc;b<=Counters;++b){if((b==Depth||b==DepthValid)&&!metric)continue;if(packing&&(b==Costs||b==Sums))continue;size_t bytes=b==Costs||b==Sums?volume*4:b==Counters?16:n*4;s->buffers[b]=allocate(bytes,prefix+resource_names[b]);}
        if(packing)s->buffers[Penalties]=allocate(n*((config.disparities+32/packing-1)/(32/packing))*4,prefix+resource_names[Penalties]);
        Parameters p{};for(unsigned b=0;b<14;++b)if(!s->buffers[b].is_empty())p.address[b]=d.device_address(s->buffers[b]).value();p.packing_bits=packing;if(packing)p.penalties_address=d.device_address(s->buffers[Penalties]).value();p.width=config.width;p.height=config.height;p.disparities=config.disparities;p.paths=config.paths;p.p1=config.p1;p.p2=config.p2;p.subpixel=config.subpixel;p.metric=bool(metric);if(metric){p.fx=metric->fx;p.baseline_m=metric->baseline_m;p.cx_offset=metric->cx_offset;}
        s->buffers[Params]=allocate(sizeof(p),prefix+resource_names[Params],true);std::memcpy(d.buffer_host_address(s->buffers[Params]).value(),&p,sizeof(p));auto address=d.device_address(s->buffers[Params]).value();s->queries=d.create_timeline_query_pool({.query_count=query_count,.name=prefix+"stage_queries"});auto* slot=s.get();
        s->graph=std::make_unique<daxa::TaskGraph>(daxa::TaskGraphInfo{.device=d,
            .pre_task_callback=[this,slot](daxa::TaskInterface ti){if(slot->started++==0){if(calibration)ti.recorder.pipeline_barrier({.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ});ti.recorder.reset_timestamps({.query_pool=slot->queries,.start_index=0,.count=query_count});ti.recorder.write_timestamp({.query_pool=slot->queries,.pipeline_stage=daxa::PipelineStageFlagBits::TOP_OF_PIPE,.query_index=0});}},
            .post_task_callback=[this,slot](daxa::TaskInterface ti){if(++slot->finished==stages)ti.recorder.write_timestamp({.query_pool=slot->queries,.pipeline_stage=daxa::PipelineStageFlagBits::BOTTOM_OF_PIPE,.query_index=1});},.name=prefix+"resident_single_reference"});
        auto& graph=*s->graph;std::array<daxa::TaskBufferView,BufferCount> views;
        for(unsigned b=0;b<BufferCount;++b)if(!s->buffers[b].is_empty()){s->external[b]=daxa::ExternalTaskBuffer({.buffer=s->buffers[b],.name=prefix+resource_names[b]});views[b]=graph.register_buffer(s->external[b]);}
        auto add=[&](unsigned stage,unsigned pipeline,unsigned pass,uint32_t groups,std::initializer_list<unsigned> reads,std::initializer_list<unsigned> writes,std::initializer_list<unsigned> readwrites){auto grid=DaxaStereo::dispatch_grid(groups,group_limits);auto task=daxa::InlineTask::Compute(std::string("stage_")+std::to_string(stage));task.reads(views[Params]);for(auto b:reads)task.reads(views[b]);for(auto b:writes)task.writes(views[b]);for(auto b:readwrites)task.reads_writes(views[b]);
            graph.add_task(task.executes([this,slot,address,stage,pipeline,pass,grid](daxa::TaskInterface ti){if(calibration&&stage<2){ti.recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ});}ti.recorder.write_timestamp({.query_pool=slot->queries,.pipeline_stage=daxa::PipelineStageFlagBits::COMPUTE_SHADER,.query_index=2+stage*2});ti.recorder.set_pipeline(pipelines[pipeline]);ti.recorder.push_constant(Push{address,pass,grid[0]});ti.recorder.dispatch({grid[0],grid[1],1});ti.recorder.write_timestamp({.query_pool=slot->queries,.pipeline_stage=daxa::PipelineStageFlagBits::COMPUTE_SHADER,.query_index=3+stage*2});}));};
        add(0,0,0,uint32_t((n+63)/64),{Left},{LeftDesc,LeftValid},{});add(1,0,1,uint32_t((n+63)/64),{Right},{RightDesc,RightValid},{});
        if(packing)add(2,7,0,uint32_t((n*((config.disparities+32/packing-1)/(32/packing))+63)/64),{},{Penalties},{});
        else add(2,1,0,uint32_t((volume+63)/64),{LeftDesc,RightDesc,LeftValid,RightValid},{Costs,Sums},{});
        for(unsigned path=0;path<config.paths;++path){auto [dx,dy]=directions[path];if(packing)add(3+path,variant==DaxaSgmVariant::PackedShuffle?(packing==8?19:20):variant==DaxaSgmVariant::PackedD128?(packing==8?15:16):(variant==DaxaSgmVariant::PackedSpecialized||variant==DaxaSgmVariant::PackedQuad||variant==DaxaSgmVariant::PackedShuffle)?(packing==8?11:12):(packing==8?8:9),path,line_count(config,dx,dy),{LeftDesc,RightDesc,LeftValid,RightValid},{},{Penalties});else add(3+path,variant==DaxaSgmVariant::Baseline?2:3,path,line_count(config,dx,dy),{Costs},{},{Sums});}
        if(packing)add(3+config.paths,variant==DaxaSgmVariant::PackedQuad?(packing==8?17:18):(variant==DaxaSgmVariant::PackedSpecialized||variant==DaxaSgmVariant::PackedD128||variant==DaxaSgmVariant::PackedQuad||variant==DaxaSgmVariant::PackedShuffle)?(packing==8?13:14):10,0,uint32_t((n+(variant==DaxaSgmVariant::PackedQuad?15:63))/(variant==DaxaSgmVariant::PackedQuad?16:64)),{LeftDesc,RightDesc,LeftValid,RightValid,Penalties},{Disparity,Confidence,Validity},{});else add(3+config.paths,4,0,uint32_t((n+63)/64),{Sums},{Disparity,Confidence,Validity},{});if(metric)add(4+config.paths,5,0,uint32_t((n+63)/64),{Disparity,Validity},{Depth,DepthValid},{});
        graph.submit({});graph.complete({});slots.push_back(std::move(s));
    }
}
DaxaStereo::DaxaStereo(StereoConfig c,std::span<const uint32_t> left,std::span<const uint32_t> right,DaxaSgmVariant v,unsigned slots,bool validation,std::optional<MetricCalibration> metric,std::shared_ptr<DaxaFoundation> owner,uint32_t wave_size){validate(c,left,right,slots,metric,v);impl_=std::make_shared<Impl>(c,v,metric,std::move(owner),validation,wave_size);impl_->setup(left,right,slots);}
DaxaStereo::Frame::Frame(std::shared_ptr<Impl> owner,std::shared_ptr<Slot> slot):owner_(std::move(owner)),slot_(std::move(slot)),generation_(slot_->generation){}
DaxaStereo::Frame& DaxaStereo::Frame::operator=(Frame other) noexcept{slot_.swap(other.slot_);owner_.swap(other.owner_);std::swap(generation_,other.generation_);return *this;}
void DaxaStereo::Frame::validate()const{require(owner_&&slot_&&generation_==slot_->generation,"empty or stale Daxa stereo frame");}
DaxaStereo::DaxaStereo(StereoConfig c,const DaxaFrame& prototype,DaxaSgmVariant variant,unsigned slots,std::optional<MetricCalibration> metric,std::shared_ptr<DaxaFoundation> context,uint32_t wave){
    validate_shape(c,slots,metric,variant);prototype.validate();require(context&&prototype.owner_->context==context,"foreign Daxa GPU input context");auto maps=prototype.calibration();require(maps->width==c.width&&maps->height==c.height,"GPU luma shape mismatch");impl_=std::make_shared<Impl>(c,variant,metric,std::move(context),false,wave);impl_->calibration=std::move(maps);impl_->setup({}, {},slots,&prototype); // No initial Frame/slot marker retained.
}
size_t DaxaStereo::collect_completed_inputs_at(uint64_t oldest){auto& e=*impl_;e.require_thread();size_t retired=0;for(auto& slot:e.slots)if(slot->input&&detail::completed_submission(slot->submit,oldest)){slot->input={};++retired;}for(auto& slot:e.slots)if(slot->source&&detail::available_slot(slot.use_count(),slot->submit,oldest))slot->source.reset();return retired;}
size_t DaxaStereo::collect_completed_inputs(){impl_->require_thread();return collect_completed_inputs_at(impl_->context->device.oldest_pending_submit_index());}
DaxaStereo::Frame DaxaStereo::process(){impl_->require_thread();return process_at(nullptr,impl_->context->device.oldest_pending_submit_index());}
DaxaStereo::Frame DaxaStereo::process(const DaxaFrame& input){impl_->require_thread();return process_at(&input,impl_->context->device.oldest_pending_submit_index());}
DaxaStereo::Frame DaxaStereo::process_at(const DaxaFrame* input,uint64_t oldest){
    auto& e=*impl_;e.require_thread();require(bool(input)==bool(e.calibration),"wrong stereo input mode");if(input){input->validate();require(input->owner_->context==e.context,"foreign GPU input context");auto maps=input->calibration();require(maps==e.calibration&&maps->width==e.config.width&&maps->height==e.config.height&&maps->rectified_identity==e.calibration->rectified_identity&&maps->original_identity==e.calibration->original_identity,"GPU luma calibration/map-cache owner/shape mismatch");}
    collect_completed_inputs_at(oldest);auto& d=e.context->device;d.collect_garbage();for(auto& slot:e.slots){if(!detail::available_slot(slot.use_count(),slot->submit,oldest))continue;
        if(input){slot->input=*input;slot->source=input->source();for(unsigned eye=0;eye<2;++eye){auto buffer=input->plane(eye?DaxaPreprocessPlane::RightLuma:DaxaPreprocessPlane::LeftLuma).buffer();slot->buffers[eye]=buffer;slot->external[eye].set_buffer(buffer);auto* params=reinterpret_cast<Parameters*>(d.buffer_host_address(slot->buffers[Params]).value());params->address[eye]=d.device_address(buffer).value();}}
        slot->generation=e.next_generation++;slot->started=0;slot->finished=0;try{slot->graph->execute({});}catch(...){slot->submit=d.latest_queue_submit_index(daxa::QUEUE_MAIN);throw;}require(slot->started==e.stages&&slot->finished==e.stages,"stereo graph callback count changed");slot->submit=d.latest_queue_submit_index(daxa::QUEUE_MAIN);return Frame(impl_,slot);
    }throw std::runtime_error("Daxa stereo slots retained or still in flight");
}
CaptureLease DaxaStereo::Frame::source()const{validate();return slot_->source;}const std::string& DaxaStereo::Frame::calibration_identity()const{validate();require(bool(owner_->calibration),"retained-fixture stereo has no calibrated source metadata");return owner_->calibration->rectified_identity;}

void DaxaStereo::Frame::wait()const{validate();auto& d=owner_->context->device;d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=slot_->submit,.timeout=30'000'000'000ull});}
DaxaStereoTiming DaxaStereo::Frame::timings()const{validate();owner_->completed(*this);constexpr size_t max_queries=2+DaxaStereoTiming::max_stages*2;std::array<uint64_t,max_queries*2> results{};auto status=daxa_timeline_query_pool_query_results(slot_->queries.get(),0,owner_->query_count,results.data());require(status==DAXA_RESULT_SUCCESS,"stereo timestamp query failed");for(unsigned i=0;i<owner_->query_count;++i)require(results[i*2+1]!=0,"stereo timestamp unavailable");uint64_t mask=owner_->timestamp_bits==64?UINT64_MAX:(uint64_t{1}<<owner_->timestamp_bits)-1;auto elapsed=[&](unsigned first,unsigned last){return double((results[last*2]-results[first*2])&mask)*double(owner_->timestamp_period)/1e6;};DaxaStereoTiming out;out.stage_count=owner_->stages;out.total_gpu_ms=elapsed(0,1);for(unsigned i=0;i<out.stage_count;++i)out.stage_gpu_ms[i]=elapsed(2+i*2,3+i*2);return out;}
daxa::BufferId DaxaStereo::Frame::buffer(DaxaStereoPlane plane)const{validate();auto id=slot_->buffers[buffer_index(plane)];require(!id.is_empty(),"plane unavailable for this variant/calibration; packed penalties are not dense costs or sums");return id;}
uint64_t DaxaStereo::Frame::generation()const{validate();return generation_;}
uint64_t DaxaStereo::Frame::context_id()const{validate();return reinterpret_cast<uintptr_t>(owner_->context->device.get());}
const char* DaxaStereo::stage_name(unsigned i)const{require(i<impl_->stages,"invalid stereo stage");if(i==0)return "census_left";if(i==1)return "census_right";if(i==2)return impl_->packing?"zero_packed_penalties":"cost_and_zero_sums";constexpr const char* names[]={"aggregate_pos_x","aggregate_neg_x","aggregate_pos_y","aggregate_neg_y","aggregate_pos_x_pos_y","aggregate_neg_x_neg_y","aggregate_pos_x_neg_y","aggregate_neg_x_pos_y"};if(i<3+impl_->config.paths)return names[i-3];return i==3+impl_->config.paths?"select":"depth_convert";}
std::string DaxaStereo::device_report()const{return impl_->context->report();}
uint32_t DaxaStereo::packing_bits()const{return impl_->packing;}
uint32_t DaxaStereo::Frame::packing_bits()const{validate();return owner_->packing;}
uint32_t DaxaStereo::required_wave_size()const{return impl_->wave_size;}
uint64_t DaxaStereo::allocated_bytes()const{return impl_->allocated;}
unsigned DaxaStereo::validation_errors()const{return impl_->context->validation_errors.load();}
const char* DaxaStereo::variant_name(DaxaSgmVariant v){if(v==DaxaSgmVariant::Baseline)return "baseline";if(v==DaxaSgmVariant::Wave)return "wave";if(v==DaxaSgmVariant::Packed)return "packed";if(v==DaxaSgmVariant::PackedSpecialized)return "packed-specialized";if(v==DaxaSgmVariant::PackedD128)return "packed-d128";if(v==DaxaSgmVariant::PackedQuad)return "packed-quad";if(v==DaxaSgmVariant::PackedShuffle)return "packed-shuffle";throw std::invalid_argument("unknown stereo variant");}
std::string DaxaStereo::shader_directory(){return SF_DAXA_STEREO_SPIRV_DIRECTORY;}
std::string DaxaStereo::shader_identity(){return std::string(SF_DAXA_STEREO_WRAPPER_SHA256)+":"+SF_DAXA_STEREO_OPS_SHA256;}
namespace {
daxa::MemoryRequirements cached_coherent(DaxaFoundation& context,daxa::BufferInfo const& info){
    auto requirements=context.device.buffer_memory_requirements(info);VkPhysicalDeviceMemoryProperties properties{};vkGetPhysicalDeviceMemoryProperties(daxa_dvc_get_vk_physical_device(context.device.get()),&properties);
    constexpr auto required=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for(bool cached:{true,false})for(uint32_t i=0;i<properties.memoryTypeCount;++i){auto flags=properties.memoryTypes[i].propertyFlags;if((requirements.memory_type_bits&(1u<<i))&&(flags&required)==required&&(!cached||(flags&VK_MEMORY_PROPERTY_HOST_CACHED_BIT))){requirements.memory_type_bits=1u<<i;return requirements;}}
    throw std::runtime_error("diagnostic readback needs HOST_VISIBLE|HOST_COHERENT memory; unqualified noncoherent invalidate path rejected");
}
}
DaxaStereoReadback DaxaStereo::readback(Frame const& frame,bool intermediate)const{
    impl_->require_thread();require(frame.owner_==impl_,"foreign Daxa stereo readback context");frame.wait();auto& e=*impl_;auto& d=e.context->device;auto& slot=*frame.slot_;size_t n=size_t(e.config.width)*e.config.height,volume=n*e.config.disparities;
    std::array<unsigned,12> planes{Disparity,Confidence,Validity,Depth,DepthValid,Costs,Sums,LeftDesc,RightDesc,LeftValid,RightValid,Penalties};std::array<size_t,12> offsets{},sizes{};size_t total=0;
    for(size_t i=0;i<planes.size();++i){if(slot.buffers[planes[i]].is_empty())continue;if((i==3||i==4)&&!e.metric)continue;if(i>=5&&!intermediate)continue;offsets[i]=total;sizes[i]=4*(i==5||i==6?volume:i==11?n*((e.config.disparities+32/e.packing-1)/(32/e.packing)):n);total+=sizes[i];}
    require(total<=e.storage_range_limit,"combined diagnostic export exceeds maxStorageBufferRange; use output-only readback");
    daxa::BufferInfo info{.size=total,.name="explicit diagnostic stereo readback"};auto requirements=cached_coherent(*e.context,info);auto memory=d.create_memory({.requirements=requirements,.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});auto host=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=memory});
    struct Cleanup{daxa::Device& device;daxa::BufferId id;~Cleanup(){device.destroy_buffer(id);device.collect_garbage();}}cleanup{d,host};
    daxa::ExternalTaskBuffer external({.buffer=host,.name="diagnostic_host_export"});daxa::TaskGraph graph({.device=d,.name="explicit stereo diagnostic export"});auto destination=graph.register_buffer(external);std::array<daxa::TaskBufferView,12> views;auto task=daxa::InlineTask::Transfer("copy selected stereo planes");
    for(size_t i=0;i<planes.size();++i)if(sizes[i]){views[i]=graph.register_buffer(slot.external[planes[i]]);task.reads(views[i]);}
    task.writes(destination);
    graph.add_task(task.executes([views,destination,sizes,offsets](daxa::TaskInterface ti){for(size_t i=0;i<sizes.size();++i)if(sizes[i])ti.recorder.copy_buffer_to_buffer({.src_buffer=ti.get(views[i]).id,.dst_buffer=ti.get(destination).id,.dst_offset=offsets[i],.size=sizes[i]});}));
    graph.add_task(daxa::InlineTask::Transfer("diagnostic host visibility").host.reads(destination).executes([](daxa::TaskInterface){}));graph.submit({});graph.complete({});graph.execute({});d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=d.latest_queue_submit_index(daxa::QUEUE_MAIN),.timeout=30'000'000'000ull});
    auto* data=static_cast<const std::byte*>(d.buffer_host_address(host).value());DaxaStereoReadback out;auto copy=[&](auto& values,size_t i){if(sizes[i]){values.resize(sizes[i]/4);std::memcpy(values.data(),data+offsets[i],sizes[i]);}};copy(out.disparity,0);copy(out.confidence,1);copy(out.validity,2);copy(out.depth,3);copy(out.depth_validity,4);copy(out.costs,5);copy(out.sums,6);for(unsigned eye=0;eye<2;++eye){copy(out.census[eye],7+eye);copy(out.census_valid[eye],9+eye);}out.packing_bits=e.packing;copy(out.packed_penalties,11);
    if(e.packing&&intermediate){out.intermediates_reconstructed=true;out.costs.resize(volume);out.sums.resize(volume);auto components=32/e.packing,words=(e.config.disparities+components-1)/components,mask=(1u<<e.packing)-1;
        // Explicit diagnostic reconstruction from ACTUAL GPU Census/masks/words.
        // These are not claimed to be dense GPU buffers in the packed variant.
        for(size_t i=0;i<n;++i)for(uint32_t disparity=0;disparity<e.config.disparities;++disparity){auto offset=i*e.config.disparities+disparity;bool ok=i%e.config.width>=disparity&&out.census_valid[0][i]&&out.census_valid[1][i-disparity];auto cost=ok?uint32_t(std::popcount(out.census[0][i]^out.census[1][i-disparity])):10000u;out.costs[offset]=cost;auto penalty=(out.packed_penalties[i*words+disparity/components]>>((disparity%components)*e.packing))&mask;out.sums[offset]=e.config.paths*cost+penalty;}
    }return out;
}
std::array<uint32_t,4> DaxaStereo::inspect(Frame const& frame)const{
    impl_->require_thread();require(frame.owner_==impl_,"foreign Daxa stereo inspect context");frame.wait();auto& e=*impl_;auto& d=e.context->device;auto& slot=*frame.slot_;
    daxa::BufferInfo info{.size=16,.name="diagnostic sixteen-byte counter export"};auto memory=d.create_memory({.requirements=cached_coherent(*e.context,info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});auto host=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=memory});struct Cleanup{daxa::Device& device;daxa::BufferId id;~Cleanup(){device.destroy_buffer(id);device.collect_garbage();}}cleanup{d,host};
    daxa::ExternalTaskBuffer external({.buffer=host,.name="counter_host_export"});daxa::TaskGraph graph({.device=d,.name="separate GPU stereo output consumer"});auto destination=graph.register_buffer(external);auto counter=graph.register_buffer(slot.external[Counters]);auto parameters=graph.register_buffer(slot.external[Params]);
    graph.add_task(daxa::InlineTask::Transfer("clear diagnostic counters").writes(counter).executes([counter](daxa::TaskInterface ti){ti.recorder.clear_buffer({.buffer=ti.get(counter).id,.offset=0,.size=16,.clear_value=0});}));auto task=daxa::InlineTask::Compute("inspect resident stereo final outputs");task.reads(parameters).reads_writes(counter);
    for(auto b:{Disparity,Confidence,Validity})task.reads(graph.register_buffer(slot.external[b]));
    if(e.metric)task.reads(graph.register_buffer(slot.external[DepthValid]));
    auto address=d.device_address(slot.buffers[Params]).value();auto n=e.config.width*e.config.height;auto grid=dispatch_grid((uint64_t(n)+63)/64,e.group_limits);
    graph.add_task(task.executes([&e,address,grid](daxa::TaskInterface ti){ti.recorder.set_pipeline(e.pipelines[6]);ti.recorder.push_constant(Push{address,0,grid[0]});ti.recorder.dispatch({grid[0],grid[1],1});}));
    graph.add_task(daxa::InlineTask::Transfer("export sixteen counter bytes").reads(counter).writes(destination).executes([counter,destination](daxa::TaskInterface ti){ti.recorder.copy_buffer_to_buffer({.src_buffer=ti.get(counter).id,.dst_buffer=ti.get(destination).id,.size=16});}));graph.add_task(daxa::InlineTask::Transfer("counter host visibility").host.reads(destination).executes([](daxa::TaskInterface){}));graph.submit({});graph.complete({});graph.execute({});d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=d.latest_queue_submit_index(daxa::QUEUE_MAIN),.timeout=30'000'000'000ull});std::array<uint32_t,4> result;std::memcpy(result.data(),d.buffer_host_address(host).value(),16);return result;
}
}
