// Causal select-only microbenchmark: retained deterministic costs, reusable
// commands/resources and identical 64-pixel dispatch groups for either shader.
#include <stereoforge/native_device.hpp>
#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace stereoforge;
using rhi::ComPtr;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static void checked(SlangResult code,const char* message){if(SLANG_FAILED(code))throw std::runtime_error(message);}
static double ms(Clock::time_point begin,Clock::time_point end){return std::chrono::duration<double,std::milli>(end-begin).count();}
static std::string quote(const std::string& value){std::ostringstream out;out<<'"';for(unsigned char c:value){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else out<<c;}out<<'"';return out.str();}
static unsigned number(const std::string& text){std::size_t used=0;const auto value=std::stoul(text,&used);if(used!=text.size()||text.empty()||text.front()=='-'||value>UINT32_MAX)throw std::invalid_argument("invalid unsigned argument");return unsigned(value);}
struct Options{fs::path shader,output,dump;unsigned width=1280,height=720,disparities=64,seed=42,frames=30,warmup=3,validation=0;};
static Options options(int argc,char** argv){Options out;for(int i=1;i<argc;++i){const std::string name=argv[i];if(++i==argc)throw std::invalid_argument("missing argument value");const std::string value=argv[i];
 if(name=="--shader")out.shader=value;else if(name=="--output")out.output=value;else if(name=="--dump-prefix")out.dump=value;
 else if(name=="--width")out.width=number(value);else if(name=="--height")out.height=number(value);else if(name=="--disparities")out.disparities=number(value);else if(name=="--seed")out.seed=number(value);else if(name=="--frames")out.frames=number(value);else if(name=="--warmup")out.warmup=number(value);else if(name=="--validation")out.validation=number(value);else throw std::invalid_argument("unknown argument: "+name);}
 const auto n=std::uint64_t(out.width)*out.height;
 if(out.shader.empty()||out.output.empty()||!fs::is_regular_file(out.shader))throw std::invalid_argument("existing --shader and --output are required");
 if(!n||out.disparities<3||out.disparities>256||n>UINT32_MAX/out.disparities||(n+63)/64>65535||!out.frames||out.frames>100000||out.warmup>100000||out.validation>1)throw std::invalid_argument("invalid bounded dimensions or iteration count");return out;}
static std::vector<std::uint32_t> costs(const Options& opt){const std::size_t n=std::size_t(opt.width)*opt.height,d=opt.disparities;std::vector<std::uint32_t> out(n*d);auto state=opt.seed;
 for(auto& value:out){state=1664525u*state+1013904223u;value=(state>>8)%1000;}
 for(std::size_t pixel=0;pixel<n;++pixel){auto first=out.begin()+pixel*d;switch(pixel%257){
 case 0:std::fill_n(first,d,10);break; // All-way tie, invalid.
 case 1:std::fill_n(first,d,100);first[1]=first[2]=1;break;
 case 2:std::fill_n(first,d,100);first[0]=0;break; // Unique disparity zero, invalid.
 case 3:std::fill_n(first,d,100);first[d-1]=0;break; // Upper disparity boundary.
 case 4:std::fill_n(first,d,10000);break; // Cost threshold, invalid.
 case 5:std::fill_n(first,d,100);first[0]=10;first[1]=5;first[2]=20;break; // Asymmetric subpixel parabola.
 default:break;}}
 return out;}
static std::string fingerprint(const std::vector<std::uint32_t>& data){std::uint64_t hash=14695981039346656037ULL;for(auto value:data)for(unsigned byte=0;byte<4;++byte){hash^=(value>>(8*byte))&255;hash*=1099511628211ULL;}std::ostringstream out;out<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return out.str();}
static ComPtr<rhi::IBuffer> make_buffer(rhi::IDevice* device,std::size_t count){rhi::BufferDesc desc{};desc.size=count*4;desc.elementSize=4;desc.usage=rhi::BufferUsage::ShaderResource|rhi::BufferUsage::UnorderedAccess|rhi::BufferUsage::CopySource|rhi::BufferUsage::CopyDestination;desc.defaultState=rhi::ResourceState::UnorderedAccess;ComPtr<rhi::IBuffer> out;checked(device->createBuffer(desc,nullptr,out.writeRef()),"create select buffer");return out;}
static void submit(rhi::ICommandQueue* queue,rhi::ICommandBuffer* command,rhi::IFence* fence,std::uint64_t value){rhi::ICommandBuffer* commands[]={command};rhi::IFence* fences[]={fence};rhi::SubmitDesc desc{};desc.commandBuffers=commands;desc.commandBufferCount=1;desc.signalFences=fences;desc.signalFenceValues=&value;desc.signalFenceCount=1;checked(queue->submit(desc),"submit select work");}
static void wait(rhi::IDevice* device,rhi::IFence* fence,std::uint64_t value){rhi::IFence* fences[]={fence};checked(device->waitForFences(1,fences,&value,true,rhi::kTimeoutInfinite),"wait select work");}
struct Sample{double gpu,submit,completion,wall,query_read;};
static double percentile(std::vector<double> values,double f){std::sort(values.begin(),values.end());const double p=f*(values.size()-1);auto lo=std::size_t(p),hi=std::min(lo+1,values.size()-1);return values[lo]+(values[hi]-values[lo])*(p-lo);}
static void summary(std::ostream& out,const std::vector<Sample>& samples,double Sample::*field){std::vector<double> values;for(const auto& sample:samples)values.push_back(sample.*field);out<<"{\"p50\":"<<percentile(values,.5)<<",\"p95\":"<<percentile(values,.95)<<",\"minimum\":"<<*std::min_element(values.begin(),values.end())<<",\"maximum\":"<<*std::max_element(values.begin(),values.end())<<'}';}
int main(int argc,char** argv){try{
 const auto opt=options(argc,argv);const auto init_begin=Clock::now();NativeDevice shared(opt.validation!=0);auto device=shared.rhi_device();auto queue=shared.rhi_queue();
 std::ifstream text(opt.shader);const std::string source((std::istreambuf_iterator<char>(text)),{});ComPtr<slang::IBlob> diagnostics;
 ComPtr<slang::IModule> module(device->getSlangSession()->loadModuleFromSourceString("select_microbenchmark",opt.shader.string().c_str(),source.c_str(),diagnostics.writeRef()));
 if(!module)throw std::runtime_error(diagnostics?static_cast<const char*>(diagnostics->getBufferPointer()):"load select module");
 ComPtr<slang::IEntryPoint> entry;checked(module->findEntryPointByName("select",entry.writeRef()),"find select entry");slang::IComponentType* entries[]={entry};rhi::ShaderProgramDesc program_desc{};program_desc.slangGlobalScope=module;program_desc.slangEntryPoints=entries;program_desc.slangEntryPointCount=1;
 ComPtr<rhi::IShaderProgram> program;auto compiled=device->createShaderProgram(program_desc,program.writeRef(),diagnostics.writeRef());if(SLANG_FAILED(compiled))throw std::runtime_error(diagnostics?static_cast<const char*>(diagnostics->getBufferPointer()):"compile select shader");
 rhi::ComputePipelineDesc pipeline_desc{};pipeline_desc.program=program;ComPtr<rhi::IComputePipeline> pipeline;checked(device->createComputePipeline(pipeline_desc,pipeline.writeRef()),"create select pipeline");
 const std::size_t n=std::size_t(opt.width)*opt.height;auto input=costs(opt);const auto input_hash=fingerprint(input);
 auto sums=make_buffer(device,input.size()),disparity=make_buffer(device,n),confidence=make_buffer(device,n),validity=make_buffer(device,n);
 ComPtr<rhi::IFence> fence;checked(device->createFence({},fence.writeRef()),"create select fence");std::uint64_t generation=1;
 const auto upload_begin=Clock::now();auto upload_encoder=queue->createCommandEncoder();checked(upload_encoder->uploadBufferData(sums,0,input.size()*4,input.data()),"upload deterministic costs");upload_encoder->globalBarrier();ComPtr<rhi::ICommandBuffer> upload_command;checked(upload_encoder->finish(upload_command.writeRef()),"finish upload");submit(queue,upload_command,fence,generation);wait(device,fence,generation);const auto upload_end=Clock::now();
 rhi::QueryPoolDesc query_desc{};query_desc.count=2;ComPtr<rhi::IQueryPool> queries;checked(device->createQueryPool(query_desc,queries.writeRef()),"create select timestamps");
 auto encoder=queue->createCommandEncoder();auto pass=encoder->beginComputePass();rhi::ShaderCursor cursor(pass->bindPipeline(pipeline));
 for(auto binding:{std::pair{"sums",sums.get()},std::pair{"disparity",disparity.get()},std::pair{"confidence",confidence.get()},std::pair{"validity",validity.get()}})checked(cursor[binding.first].setBinding(rhi::Binding(binding.second)),binding.first);
 checked(cursor["width"].setData(opt.width),"width");checked(cursor["height"].setData(opt.height),"height");checked(cursor["disparity_count"].setData(opt.disparities),"disparity_count");const unsigned subpixel=1;checked(cursor["subpixel"].setData(subpixel),"subpixel");
 pass->writeTimestamp(queries,0);pass->dispatchCompute(std::uint32_t((n+63)/64),1,1);pass->writeTimestamp(queries,1);pass->end();encoder->globalBarrier();ComPtr<rhi::ICommandBuffer> command;checked(encoder->finish(command.writeRef()),"finish reusable select command");const auto init_end=Clock::now();
 const auto frequency=device->getInfo().timestampFrequency;if(!frequency)throw std::runtime_error("timestamp frequency unavailable");
 std::vector<Sample> samples;samples.reserve(opt.frames);
 // No per-iteration encoders, pipelines, buffers, uploads or output downloads.
 for(unsigned iteration=0;iteration<opt.warmup+opt.frames;++iteration){
  checked(queries->reset(),"reset select timestamps");const auto start=Clock::now();submit(queue,command,fence,++generation);const auto submitted=Clock::now();wait(device,fence,generation);const auto completed=Clock::now();
  std::uint64_t ticks[2]{};checked(queries->getResult(0,2,ticks),"read completed select timestamps");const auto read=Clock::now();
  if(iteration>=opt.warmup)samples.push_back({double(ticks[1]-ticks[0])*1000/frequency,ms(start,submitted),ms(submitted,completed),ms(start,completed),ms(completed,read)});
 }
 if(!opt.dump.empty()){
  if(!opt.dump.parent_path().empty())fs::create_directories(opt.dump.parent_path());
  for(auto binding:{std::pair{"disparity.f32",disparity.get()},std::pair{"confidence.f32",confidence.get()},std::pair{"validity.u32",validity.get()}}){std::vector<std::uint32_t> output(n);checked(device->readBuffer(binding.second,0,n*4,output.data()),"read final select output");std::ofstream file(opt.dump.string()+"."+binding.first,std::ios::binary);if(!file.write(reinterpret_cast<const char*>(output.data()),n*4))throw std::runtime_error("write select dump");}
 }
 if(!opt.output.parent_path().empty())fs::create_directories(opt.output.parent_path());std::ofstream out(opt.output);if(!out)throw std::runtime_error("create select report");out<<std::setprecision(12);
 out<<"{\"schema\":1,\"device\":"<<quote(shared.name())<<",\"shader\":"<<quote(opt.shader.string())<<",\"width\":"<<opt.width<<",\"height\":"<<opt.height<<",\"disparities\":"<<opt.disparities<<",\"subpixel\":true,\"seed\":"<<opt.seed<<",\"costs_fnv1a64\":"<<quote(input_hash)<<",\"costs_bytes\":"<<input.size()*4<<",\"resident_buffer_bytes\":"<<(input.size()+3*n)*4<<",\"dispatch_groups\":"<<(n+63)/64<<",\"pixels_per_group\":64,\"frames\":"<<opt.frames<<",\"warmup\":"<<opt.warmup<<",\"timestamp_frequency\":"<<frequency<<",\"initialization_ms\":"<<ms(init_begin,init_end)<<",\"initial_upload_ms\":"<<ms(upload_begin,upload_end)<<",\"scope\":\"Select kernel only on a static resident cost volume; excludes stereo aggregation, camera, output download and setup; not 60 Hz qualification.\",\"summary_ms\":{\"gpu\":";
 summary(out,samples,&Sample::gpu);out<<",\"submit\":";summary(out,samples,&Sample::submit);out<<",\"completion_wait\":";summary(out,samples,&Sample::completion);out<<",\"submit_to_completion\":";summary(out,samples,&Sample::wall);out<<",\"timestamp_read\":";summary(out,samples,&Sample::query_read);out<<"},\"samples_ms\":[";
 for(std::size_t i=0;i<samples.size();++i){if(i)out<<',';const auto& v=samples[i];out<<"{\"gpu\":"<<v.gpu<<",\"submit\":"<<v.submit<<",\"completion_wait\":"<<v.completion<<",\"submit_to_completion\":"<<v.wall<<",\"timestamp_read\":"<<v.query_read<<'}';}out<<"]}\n";if(!out)throw std::runtime_error("write select report");std::cout<<opt.output<<'\n';return 0;
 }catch(const std::exception& error){std::cerr<<"native_select_benchmark: "<<error.what()<<'\n';return 2;}}
