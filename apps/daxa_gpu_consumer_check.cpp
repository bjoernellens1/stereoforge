#include "daxa_view_depth_common.hpp"
#include <stereoforge/daxa_gpu_consumer.hpp>
#include <array>
#include <cstring>
#include <limits>
using namespace view_app;
namespace {
// This is an explicit diagnostic readback, never a real-time pipeline stage.
void compare_exact(DaxaFoundation& context, DaxaObservationFrame const& producer,
                   DaxaGpuConsumer::Frame& consumer) {
    auto& d = context.device;
    const size_t bytes = size_t(consumer.width()) * consumer.height() * 4;
    std::array<daxa::BufferId, 4> staging{};
    std::array<daxa::MemoryBlock, 4> memory{};
    try {
        for (unsigned i=0;i<4;++i) {
            daxa::BufferInfo info{.size=bytes,.name="GPU consumer diagnostic readback"};
            memory[i]=d.create_memory({.requirements=context.coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
            staging[i]=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=memory[i]});
        }
        auto recorder=d.create_command_recorder({.name="consumer exact diagnostic readback"});
        recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE|daxa::AccessConsts::TRANSFER_WRITE,
                                   .dst_access=daxa::AccessConsts::TRANSFER_READ});
        const std::array<daxa::BufferId,4> sources{producer.buffer(DaxaObservationPlane::Depth),consumer.depth(),
            producer.image(DaxaObservationImage::RectifiedLeft).buffer(),consumer.rgba()};
        for(unsigned i=0;i<4;++i) recorder.copy_buffer_to_buffer({.src_buffer=sources[i],.dst_buffer=staging[i],.size=bytes});
        recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ});
        auto commands=recorder.complete_current_commands();
        auto ticket=d.submit_commands({.command_lists={&commands,1}});
        consumer.retain_until(ticket);
        d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=ticket,.timeout=30'000'000'000ull});
        check(std::memcmp(d.buffer_host_address(staging[0]).value(),d.buffer_host_address(staging[1]).value(),bytes)==0,"consumer depth differs from producer bytes");
        check(std::memcmp(d.buffer_host_address(staging[2]).value(),d.buffer_host_address(staging[3]).value(),bytes)==0,"consumer RGBA differs from producer bytes");
        for(auto id:staging)d.destroy_buffer(id);
        d.collect_garbage();
    } catch (...) {
        // If a command may have been submitted, do not destroy its resources early.
        d.wait_idle();
        for(auto id:staging)if(!id.is_empty()&&d.is_id_valid(id))d.destroy_buffer(id);
        d.collect_garbage();
        throw;
    }
}
}
int main(int argc,char** argv){try {
    Options options(argc,argv,false);
    check(options.validation,"--validation is required for this diagnostic");
    check(options.u("--slots",2)==2,"diagnostic requires exactly two slots");
    Prepared prepared(options);
    check(prepared.source->device_identity.find("synthetic")!=std::string::npos,"synthetic fixture required");
    check(prepared.source->frame_id<=std::numeric_limits<uint64_t>::max()-2,"source frame ID overflow");
    auto source2=std::make_shared<CaptureFrame>(*prepared.source);
    auto source3=std::make_shared<CaptureFrame>(*prepared.source);
    ++source2->frame_id;source3->frame_id+=2;
    for(size_t i=0;i<source2->packed_yuyv.size();i+=2){source2->packed_yuyv[i]^=0x55;source3->packed_yuyv[i]^=0xaa;}
    auto context=std::make_shared<DaxaFoundation>(true);
    auto pipeline=std::make_unique<DaxaDepthPipeline>(prepared.fixture.rectification,prepared.depth,context,prepared.source);
    auto first=pipeline->enqueue(prepared.source);
    auto first_observation=first.observations();
    auto consumer=std::make_unique<DaxaGpuConsumer>(context,first_observation,2);
    auto a=consumer->enqueue(first_observation);
    auto second=pipeline->enqueue(source2);
    auto second_observation=second.observations();
    auto b=consumer->enqueue(second_observation);
    check(!consumer->has_capacity(),"retained consumer ring must be exhausted");
    bool rejected=false;
    try { auto extra=consumer->enqueue(second_observation); } catch(std::runtime_error const&) { rejected=true; }
    check(rejected,"consumer ring must reject excess enqueue");
    bool empty_rejected=false;
    try { auto extra=consumer->enqueue(DaxaObservationFrame{}); } catch(std::invalid_argument const&) { empty_rejected=true; }
    check(empty_rejected,"empty observation must be rejected");
    check(a.source()->frame_id==prepared.source->frame_id&&b.source()->frame_id==source2->frame_id,"consumer source identity mismatch");
    compare_exact(*context,first_observation,a);
    compare_exact(*context,second_observation,b);
    auto freed_depth=a.depth();
    a={};first_observation={};first={};consumer->collect();pipeline->collect();
    check(consumer->has_capacity(),"completed released slot must become available");
    auto third=pipeline->enqueue(source3);
    auto third_observation=third.observations();
    auto c=consumer->enqueue(third_observation);
    check(c.source()->frame_id==source3->frame_id,"reused consumer source identity mismatch");
    check(c.depth()==freed_depth,"released consumer slot was not reused");
    check(!consumer->has_capacity(),"retained slot and reused slot must fill ring");
    compare_exact(*context,third_observation,c);
    compare_exact(*context,second_observation,b); // retained output survives reuse
    json result={{"schema","stereoforge.daxa_gpu_consumer_check.v1"},
      {"scope","synthetic_repeated_source_exact_diagnostic_not_live_or_performance"},
      {"source_frame_ids",{prepared.source->frame_id,source2->frame_id,source3->frame_id}},
      {"source_variants","synthetic_luma_xor_00_55_aa"},{"capture_sha256",prepared.fixture.capture_sha256},
      {"manifest_sha256",prepared.manifest_hash},{"width",c.width()},{"height",c.height()},
      {"depth_and_rgba_exact",true},{"retained_ring_exhaustion_checked",true},
      {"released_slot_reuse_checked",true},{"retained_output_after_reuse_checked",true},
      {"empty_input_rejected",true},{"foreign_context_checked",false},
      {"diagnostic_host_readback",true},{"performance_qualified",false},
      {"requested_consumer_bytes",consumer->requested_bytes()}};
    c={};b={};third_observation={};second_observation={};third={};second={};
    consumer->drain();consumer.reset();pipeline->drain();pipeline.reset();
    result["device"]=json::parse(context->report());context->close();
    check(context->validation_errors==0&&context->validation_warnings==0,"Vulkan validation failed through teardown");
    result["validation_errors_after_teardown"]=context->validation_errors.load();
    result["validation_warnings_after_teardown"]=context->validation_warnings.load();
    prepared.verify_inputs(options);finish(options,result);return 0;
}catch(std::exception const& e){std::cerr<<"daxa_gpu_consumer_check: "<<e.what()<<'\n';return 1;}}
