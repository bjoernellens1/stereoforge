#include "daxa_view_depth_common.hpp"
#include <stereoforge/daxa_temporal.hpp>
#include <daxa/c/device.h>
#include <limits>
using namespace view_app;
namespace {
constexpr size_t pool_capacity=8;
int64_t host_ns(){auto n=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();check(n>=0,"host steady timestamp outside signed64 domain");return n;}
DaxaTimedRigPose pose(int64_t time){return {{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1},time,"host_steady"};}
}
int main(int argc,char** argv){try{
    if(argc==2&&std::string_view(argv[1])=="--shader-directory"){std::cout<<DaxaTemporal::shader_directory()<<'\n';return 0;}
    if(argc==2&&std::string_view(argv[1])=="--shader-source-identities"){auto j=shader_identities();j["temporal_shader_source_sha256_pair"]=DaxaTemporal::shader_identity();std::cout<<j.dump()<<'\n';return 0;}
    // Parse the one benchmark-specific switch before the established shared CLI.
    std::string variant="depth_temporal";bool variant_seen=false;std::vector<char*> arguments;arguments.reserve(size_t(argc));arguments.push_back(argv[0]);
    for(int i=1;i<argc;++i){if(std::string_view(argv[i])=="--variant"){check(!variant_seen&&i+1<argc,"duplicate/missing variant");variant_seen=true;variant=argv[++i];}else arguments.push_back(argv[i]);}
    if(variant=="depth")variant="depth_alone";check(variant=="depth_alone"||variant=="depth_temporal"||variant=="depth_temporal_view","variant must be depth_alone, depth_temporal or depth_temporal_view");
    bool use_temporal=variant!="depth_alone",use_view=variant=="depth_temporal_view";
    Options options(int(arguments.size()),arguments.data(),false);Prepared prepared(options);auto frames=options.u("--frames",100),warmup=options.u("--warmup",3);
    check(frames>=1&&frames<=10000&&warmup<=1000,"bounded frames1..10000/warmup0..1000 required");check(prepared.depth.slots>=2&&prepared.depth.slots<=3,"matched benchmark slots must be2..3 for all variants");check(use_view||!options.values.contains("--dump-prefix"),"dump-prefix requires depth_temporal_view variant");
    // Allocate/copy all capture pixels and owned metadata before measurement.
    // Unique current capture IDs/host times describe repeated static bytes only.
    std::array<std::shared_ptr<CaptureFrame>,pool_capacity> captures;
    for(auto& c:captures){c=std::make_shared<CaptureFrame>(*prepared.source);c->device_identity="static-content-temporal-benchmark";c->host_clock_domain="host_steady";c->host_ingress_timestamp_ns=uint64_t(host_ns());c->frame_id=0;}
    auto context=std::make_shared<DaxaFoundation>(options.validation);auto depth=std::make_unique<DaxaDepthPipeline>(prepared.fixture.rectification,prepared.depth,context,captures[0]);
    auto prototype=depth->enqueue(captures[0]);std::unique_ptr<DaxaTemporal> temporal;std::unique_ptr<DaxaView> view;
    DaxaTemporalConfig temporal_config;temporal_config.slots=prepared.depth.slots;
    if(use_temporal)temporal=std::make_unique<DaxaTemporal>(context,prototype,temporal_config);
    if(use_view)view=std::make_unique<DaxaView>(prepared.config,prepared.view,context,prototype.observations(),prepared.depth.slots);
    prototype.wait();prototype={};depth->collect();
    auto depth_bytes=depth->requested_combined_buffer_bytes(),temporal_bytes=temporal?temporal->requested_buffer_bytes():0,view_bytes=view?view->allocated_bytes():0;
    check(depth_bytes+temporal_bytes+view_bytes<=2ull*1024*1024*1024,"combined requested chain buffers exceed2GiB");
    enum Sample:unsigned {Enqueue,Wait,Query,Wall,Age,Chain,Depth,Upload,Decode,Rectify,LeftStereo,RightStereo,LR,Metadata,Temporal,Clear,Z,Owner,Fuse,View,SampleCount};
    std::array<std::vector<double>,SampleCount> samples;for(auto& v:samples)v.resize(frames);
    DaxaDepthPipeline::Frame raw;DaxaTemporal::Frame fused;DaxaView::Frame projected;
    uint64_t offered=0,completed=0,skipped_admission=0,skipped_capture_pool=0,explicit_waits=0;
    uint64_t retired_view=0,retired_temporal=0,retired_depth=0,last_host=0,first_host=0;size_t cursor=0;
    auto measurement_start=Clock::now();
    while(completed<uint64_t(frames)+warmup){
        check(offered<4*(uint64_t(frames)+warmup),"bounded benchmark admission failed to progress");++offered;
        projected={};fused={};raw={};auto start=Clock::now();
        // Release downstream leases before Depth's capture/upload admission.
        if(view)retired_view+=view->collect();if(temporal)retired_temporal+=temporal->collect();retired_depth+=depth->collect();
        if((view&&!view->can_project())||(temporal&&!temporal->can_process())){++skipped_admission;continue;}
        size_t selected=pool_capacity;for(size_t n=0;n<pool_capacity;++n){auto index=(cursor+n)%pool_capacity;if(captures[index].use_count()==1){selected=index;break;}}
        if(selected==pool_capacity){++skipped_capture_pool;continue;}cursor=(selected+1)%pool_capacity;auto& capture=captures[selected];
        auto now=uint64_t(host_ns());check(last_host<uint64_t(INT64_MAX),"host benchmark timestamp overflow");auto timestamp=std::max(now,last_host+1);check(timestamp<=uint64_t(INT64_MAX),"host benchmark timestamp outside signed64 domain");last_host=timestamp;if(first_host==0)first_host=timestamp;
        capture->frame_id=completed+1;capture->host_ingress_timestamp_ns=timestamp;
        raw=depth->enqueue(capture);if(temporal)fused=temporal->process(raw,pose(int64_t(timestamp)));if(view)projected=view->project(fused.observations());auto submitted=Clock::now();
        if(view)projected.wait();else if(temporal)fused.wait();else raw.wait();++explicit_waits;auto complete=Clock::now();
        auto dt=raw.timings();DaxaTemporalTiming tt;if(temporal)tt=fused.timings();double view_ms=view?projected.gpu_ms():0;
        auto end_tick=view?projected.end_tick():temporal?tt.end_tick:dt.preprocessing.end_tick;
        // Depth's terminal metric pass ends at its own combined queue span.
        double chain_ms=dt.combined_gpu_ms;
        if(temporal){auto bits=dt.preprocessing.timestamp_bits;auto mask=bits==64?UINT64_MAX:(uint64_t{1}<<bits)-1;check(tt.timestamp_bits==bits&&tt.timestamp_period==dt.preprocessing.timestamp_period,"chain timestamp properties differ");chain_ms=double((end_tick-dt.preprocessing.start_tick)&mask)*dt.preprocessing.timestamp_period/1e6;}
        auto queried=Clock::now();check(std::chrono::duration<long double,std::nano>(queried-start).count()<std::ldexp(static_cast<long double>(dt.preprocessing.timestamp_period),int(dt.preprocessing.timestamp_bits)),"chain timestamp wrap ambiguous");
        check(dt.per_reference_stereo_timings_available,"benchmark requires matched sequential independent SGM timings");
        if(completed>=warmup){auto j=size_t(completed-warmup);std::array<double,SampleCount> values{
            std::chrono::duration<double,std::milli>(submitted-start).count(),std::chrono::duration<double,std::milli>(complete-submitted).count(),std::chrono::duration<double,std::milli>(queried-complete).count(),std::chrono::duration<double,std::milli>(queried-start).count(),double(std::chrono::duration_cast<std::chrono::nanoseconds>(complete.time_since_epoch()).count()-int64_t(timestamp))/1e6,
            chain_ms,dt.combined_gpu_ms,dt.preprocessing.upload_ms,dt.preprocessing.decode_ms,dt.preprocessing.rectify_ms,dt.left.total_gpu_ms,dt.right.total_gpu_ms,dt.lr_gpu_ms,dt.metric_metadata_gpu_ms,tt.total_gpu_ms,tt.clear_gpu_ms,tt.z_gpu_ms,tt.owner_gpu_ms,tt.fuse_gpu_ms,view_ms};
            for(unsigned k=0;k<SampleCount;++k)samples[k][j]=values[k];}
        ++completed;
    }
    auto measurement_end=Clock::now();check(depth->requested_combined_buffer_bytes()==depth_bytes&&(!temporal||temporal->requested_buffer_bytes()==temporal_bytes)&&(!view||view->allocated_bytes()==view_bytes),"resident requested buffers grew during benchmark");
    auto diagnostic_start=Clock::now();GeometryReadback exported;if(options.values.contains("--dump-prefix")){exported=view->readback(projected);dump(exported,options.values.at("--dump-prefix"));}double diagnostic_ms=std::chrono::duration<double,std::milli>(Clock::now()-diagnostic_start).count();
    auto result=report(prepared,false);result["schema"]="stereoforge.daxa_temporal_benchmark.v1";result["variant"]=variant;
    result["scope"]="serial_known_identity_motion_actual_unmodified_SGM_over_static_recorded_pixels_unique_synthetic_capture_metadata_no_live_or_physical_throughput_claim";
    result["current_RGB_view_enabled"]=use_view;result["observations_overridden_for_fixture"]=false;result["frame_age_scope"]="host_steady_stamp_before_Depth_upload_through_terminal_completion_excludes_original_camera_exposure_age";
    result["unique_physical_frames"]=false;result["source_pixels_static"]=true;result["capture_metadata_policy"]="preallocated8capture_pool_unique_ID_and_host_steady_timestamp_only_while_entry_exclusively_owned";
    result["recorded_source_metadata"]=result["source_metadata"];result.erase("source_metadata");result["recorded_source_id"]=result["source_id"];result.erase("source_id");result["benchmark_capture_epoch"]={{"device_identity","static-content-temporal-benchmark"},{"host_clock_domain","host_steady"},{"first_frame_id",1},{"last_frame_id",completed},{"first_host_ingress_timestamp_ns",first_host},{"last_host_ingress_timestamp_ns",last_host}};
    result["known_motion"]="identity_world_from_original_left_rig";result["frames"]=frames;result["warmup"]=warmup;result["offered_including_warmup"]=offered;result["completed_including_warmup"]=completed;result["skipped_admission"]=skipped_admission;result["skipped_capture_pool"]=skipped_capture_pool;result["explicit_terminal_waits_including_warmup"]=explicit_waits;
    result["wall_scope"]="serial_enqueue_wait_query; samples_do_not_establish_pipelined_or_live_FPS";result["all_warmup_and_measurement_wall_ms"]=std::chrono::duration<double,std::milli>(measurement_end-measurement_start).count();
    result["same_queue_chain_scope"]="Depth_upload_start_through_terminal_stage_end_including_intergraph_gaps";result["diagnostic_after_timing_ms"]=diagnostic_ms;
    result["requested_depth_buffer_bytes"]=depth_bytes;result["requested_temporal_buffer_bytes"]=temporal_bytes;result["requested_view_buffer_bytes"]=view_bytes;result["requested_chain_buffer_bytes"]=depth_bytes+temporal_bytes+view_bytes;
    result["memory_scope"]="sum_of_disjoint_stage_requested_buffers_excludes_driver_process_peak_queries_command_pools_and_scoped_diagnostics";result["driver_process_peak_memory_bytes"]=nullptr;result["capture_pool_entries"]=pool_capacity;result["requested_capture_pool_pixel_bytes"]=pool_capacity*prepared.source->packed_yuyv.size();
    result["collection_order"]="View_then_Temporal_then_Depth_before_capture_admission";result["shared_context_garbage_collection"]="Depth_enqueue_plus_nonblocking_Temporal_collect_each_offer; optional_View_existing_collection; final_stage_destructors_wait_and_collect";
    result["slots"]=prepared.depth.slots;result["disparities"]=prepared.depth.stereo.disparities;result["paths"]=prepared.depth.stereo.paths;result["p1"]=prepared.depth.stereo.p1;result["p2"]=prepared.depth.stereo.p2;result["required_wave_size"]=prepared.depth.required_wave_size;
    result["temporal_config"]={{"slots",temporal_config.slots},{"max_contributors",temporal_config.max_contributors},{"max_reuses",temporal_config.max_reuses},{"max_age_ns",temporal_config.max_age_ns},{"relative_depth_tolerance",temporal_config.relative_depth_tolerance},{"confidence_floor",temporal_config.confidence_floor}};
    result["shader_source_identities"]["temporal_shader_source_sha256_pair"]=DaxaTemporal::shader_identity();
    constexpr std::array<char const*,SampleCount> names{"enqueue_collect_admission_cpu_ms","explicit_wait_cpu_ms","query_cpu_ms","host_wall_ms","host_ingress_to_terminal_complete_ms","gpu_chain_queue_span_ms","gpu_depth_queue_span_ms","gpu_upload_ms","gpu_decode_ms","gpu_rectify_ms","gpu_left_stereo_ms","gpu_right_stereo_ms","gpu_lr_ms","gpu_metric_metadata_ms","gpu_temporal_queue_span_ms","gpu_temporal_clear_ms","gpu_temporal_z_ms","gpu_temporal_owner_ms","gpu_temporal_fuse_ms","gpu_view_ms"};
    result["quantile_policy"]="p99_nearest_rank_ceil_0.99N; existing_p50_p95_unchanged";
    for(unsigned k=0;k<SampleCount;++k)if((k<Temporal||temporal)&&(k!=View||view)){
        result[names[k]]=summary(samples[k]);
        auto sorted=samples[k];std::sort(sorted.begin(),sorted.end());
        result[names[k]]["p99"]=sorted[(99ull*sorted.size()+99)/100-1];
        result[names[k]]["max"]=sorted.back();
        result["raw_samples_ms"][names[k]]=samples[k];
    }
    if(options.values.contains("--dump-prefix"))coverage(result,exported);
    projected={};fused={};raw={};if(view)retired_view+=view->collect();if(temporal)retired_temporal+=temporal->collect();retired_depth+=depth->collect();result["retired_view_inputs"]=retired_view;result["retired_temporal_sources"]=retired_temporal;result["retired_depth_outputs"]=retired_depth;
    view.reset();temporal.reset();depth->drain();depth.reset();result["device"]=json::parse(context->report());context->close();check(context->validation_errors==0&&context->validation_warnings==0,"temporal benchmark validation failed through teardown");result["validation_requested"]=options.validation;result["validation_errors_after_teardown"]=context->validation_errors.load();result["validation_warnings_after_teardown"]=context->validation_warnings.load();prepared.verify_inputs(options);finish(options,result);return 0;
}catch(std::exception const& e){std::cerr<<"daxa_temporal_benchmark: "<<e.what()<<'\n';return 1;}}
