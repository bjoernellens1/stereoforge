#include "timing_metadata.hpp"
#include "daxa_refinement_common.hpp"
#include <stereoforge/canonical_record.hpp>
#include <stereoforge/daxa_rectified_input.hpp>
#include <cmath>

using namespace refinement_app;
namespace {
struct CanonicalOptions {
    std::map<std::string,std::string> values;
    bool validation=false;
    CanonicalOptions(int argc,char** argv) {
        for(int i=1;i<argc;++i) {
            std::string key=argv[i];
            if(key=="--validation") { check(!validation,"duplicate validation option");validation=true;continue; }
            check(key=="--manifest"||key=="--output"||key=="--variant"||key=="--frames"||key=="--warmup"||key=="--radius"||key=="--dump-prefix","unknown canonical benchmark option");
            check(++i<argc,"missing option value");
            check(values.emplace(key,argv[i]).second,"duplicate option");
        }
        check(values.contains("--manifest")&&values.contains("--output"),"manifest and output required");
    }
    unsigned get(char const* key,unsigned fallback)const { return values.contains(key)?number(values.at(key)):fallback; }
};
DaxaRectification grid(DaxaRectifiedGeometry const& geometry) {
    DaxaRectification out;
    out.eye_width=geometry.original.eyes[0].width;out.source_height=geometry.original.eyes[0].height;
    out.width=geometry.rectified.eyes[0].width;out.height=geometry.rectified.eyes[0].height;
    out.original_identity=geometry.original_identity;out.rectified_identity=geometry.rectified_identity;
    out.geometry=geometry; // Canonical input has no raw layout or remapping tables.
    return out;
}
json shader_hashes(std::string const& directory,std::string const& prefix) {
    json result=json::object();
    for(auto const& entry:std::filesystem::directory_iterator(directory))
        if(entry.is_regular_file()&&entry.path().extension()==".spv"&&entry.path().filename().string().starts_with(prefix))
            result[entry.path().filename().string()]=daxa_preprocess_file_sha256(entry.path());
    return result;
}
void dump_depth_plane(json& files,std::string const& prefix,char const* name,auto const& plane) {
    auto path=prefix+"."+name;write(path,plane.data(),plane.size()*sizeof(typename std::remove_reference_t<decltype(plane)>::value_type));
    files[name]={{"path",path},{"sha256",daxa_preprocess_file_sha256(path)},{"elements",plane.size()}};
}
json dump_depth(DaxaDepthReadback const& data,std::string const& prefix) {
    json files=json::object();
    dump_depth_plane(files,prefix,"disparity.f32",data.lr_disparity);
    dump_depth_plane(files,prefix,"confidence.f32",data.lr_confidence);
    dump_depth_plane(files,prefix,"validity.u32",data.lr_validity);
    dump_depth_plane(files,prefix,"depth.f32",data.depth);
    dump_depth_plane(files,prefix,"metric_confidence.f32",data.metric_confidence);
    dump_depth_plane(files,prefix,"metric_validity.u32",data.metric_validity);
    dump_depth_plane(files,prefix,"provenance.u32",data.provenance);
    dump_depth_plane(files,prefix,"source_ids.i64",data.source_ids);
    dump_depth_plane(files,prefix,"age_ns.i64",data.age_ns);
    return files;
}
}

int main(int argc,char** argv) { try {
    CanonicalOptions options(argc,argv);
    auto frames=options.get("--frames",100),warmup=options.get("--warmup",20),radius=options.get("--radius",4);
    check(frames>=1&&frames<=300&&warmup<=30,"frames1..300/warmup0..30 required");
    check(radius==2||radius==4||radius==8,"radius must be2/4/8");
    auto variant=options.values.contains("--variant")?options.values.at("--variant"):"full_sgm";
    check(variant=="full_sgm"||variant=="coarse_refine","unknown canonical benchmark variant");
    bool refine=variant=="coarse_refine";
    auto record=load_canonical_record(options.values.at("--manifest"));
    auto full=grid(record.full_geometry);
    std::optional<DaxaRectification> coarse;
    if(refine)coarse=grid(record.coarse_geometry);
    DaxaRectifiedInputConfig producer_config;producer_config.slots=2;producer_config.coarse_2x=refine;
    DaxaDepthConfig depth_config;
    auto const& stereo_grid=refine?*coarse:full;
    depth_config.stereo={stereo_grid.width,stereo_grid.height,refine?64u:128u,4,8,32,true};
    depth_config.slots=2;depth_config.right_reference_mode=DaxaRightReferenceMode::IndependentSGM;
    depth_config.validate();
    DaxaRefinementConfig refinement_config;refinement_config.slots=2;refinement_config.radius=radius;refinement_config.validate();

    // All admission, file hashing, compilation and graph construction precede timing.
    auto context=std::make_shared<DaxaFoundation>(options.validation);
    auto producer=std::make_unique<DaxaRectifiedInput>(full,coarse,producer_config,context,record.frame);
    auto bootstrap=producer->enqueue(record.frame);
    auto depth=std::make_unique<DaxaDepthPipeline>(depth_config,context,refine?bootstrap.coarse():bootstrap.full());
    std::unique_ptr<DaxaRefinement> refinement;
    if(refine) {
        auto prototype=depth->enqueue_preprocessed(bootstrap.coarse());
        prototype.wait();
        refinement=std::make_unique<DaxaRefinement>(context,prototype,bootstrap.full(),refinement_config);
        prototype={};depth->drain();
    }
    bootstrap={};depth->collect();producer->collect();

    enum : unsigned {Enqueue,Wait,Query,Wall,UploadCpu,UploadGpu,OriginalGpu,PackGpu,CoarseGpu,ProducerSpan,
        LeftSgm,RightSgm,Lr,Metric,StereoMetricSpan,RefinePack,RefineCensus,RefineMatch,RefineLr,RefineMetric,RefineSpan,CombinedSpan,Count};
    constexpr const char* keys[Count]={"enqueue_cpu_ms","wait_cpu_ms","query_cpu_ms","host_wall_ms","upload_cpu_ms",
        "gpu_upload_ms","gpu_original_pack_ms","gpu_rectified_pack_ms","gpu_coarse_pack_ms","gpu_producer_queue_span_ms",
        "gpu_left_sgm_ms","gpu_right_sgm_ms","gpu_lr_ms","gpu_metric_ms","gpu_stereo_to_metric_queue_span_ms",
        "gpu_refinement_pack_ms","gpu_refinement_census_ms","gpu_refinement_match_ms","gpu_refinement_lr_ms","gpu_refinement_metric_ms","gpu_refinement_queue_span_ms","gpu_combined_queue_span_ms"};
    std::array<std::vector<double>,Count> samples;
    for(auto& column:samples)column.resize(frames);
    std::array<std::array<std::vector<double>,DaxaStereoTiming::max_stages>,2> stereo_stages;
    for(auto& eye:stereo_stages)for(auto& stage:eye)stage.resize(frames);
    std::array<unsigned,2> stage_counts{};
    DaxaRectifiedInput::Frame input;
    DaxaDepthPipeline::Frame stereo;
    DaxaRefinement::Frame output;
    for(unsigned i=0;i<frames+warmup;++i) {
        auto a=Clock::now();
        output={};if(refinement)refinement->collect();
        stereo={};depth->collect();input={};producer->collect();
        if(refinement)check(refinement->can_process(),"refinement admission failed");
        input=producer->enqueue(record.frame);
        stereo=depth->enqueue_preprocessed(refine?input.coarse():input.full());
        if(refinement)output=refinement->process(stereo,input.full());
        auto b=Clock::now();
        if(refinement)output.wait();else stereo.wait();
        auto c=Clock::now();
        auto pt=producer->timings(input);auto dt=stereo.timings();
        DaxaRefinementTiming rt;if(refinement)rt=output.timings();
        auto end=refinement?rt.end_tick:dt.end_tick;
        auto bits=refinement?rt.timestamp_bits:dt.timestamp_bits;
        auto period=refinement?rt.timestamp_period:dt.timestamp_period;
        check(pt.timestamp_bits>0&&pt.timestamp_bits<=64&&bits==pt.timestamp_bits&&period==pt.timestamp_period,"incompatible queue timestamp domains");
        auto wrap_ns=std::ldexp(static_cast<long double>(period),int(bits));
        check(std::chrono::duration<long double,std::nano>(Clock::now()-a).count()<wrap_ns,"combined timestamp wrap interval ambiguous");
        auto mask=bits==64?UINT64_MAX:(uint64_t(1)<<bits)-1;
        double combined=double((end-pt.start_tick)&mask)*period/1e6;
        check(std::isfinite(combined),"nonfinite combined GPU span");
        auto d=Clock::now();
        if(i>=warmup) {
            std::array<double,Count> row{std::chrono::duration<double,std::milli>(b-a).count(),std::chrono::duration<double,std::milli>(c-b).count(),std::chrono::duration<double,std::milli>(d-c).count(),std::chrono::duration<double,std::milli>(d-a).count(),producer->last_upload_cpu_ms(),pt.upload_ms,pt.original_ms,pt.rectified_ms,pt.coarse_ms,pt.total_ms,dt.left.total_gpu_ms,dt.right.total_gpu_ms,dt.lr_gpu_ms,dt.metric_metadata_gpu_ms,dt.stereo_to_metric_gpu_ms,rt.stage_gpu_ms[0],rt.stage_gpu_ms[1],rt.stage_gpu_ms[2],rt.stage_gpu_ms[3],rt.stage_gpu_ms[4],rt.total_gpu_ms,combined};
            for(unsigned k=0;k<Count;++k) { check(std::isfinite(row[k]),"nonfinite timing sample");samples[k][i-warmup]=row[k]; }
            stage_counts={dt.left.stage_count,dt.right.stage_count};
            check(stage_counts[0]<=DaxaStereoTiming::max_stages&&stage_counts[1]<=DaxaStereoTiming::max_stages,"invalid stereo stage count");
            for(unsigned eye=0;eye<2;++eye)for(unsigned k=0;k<stage_counts[eye];++k)
                stereo_stages[eye][k][i-warmup]=(eye?dt.right:dt.left).stage_gpu_ms[k];
        }
    }
    auto calibration=full.metric_calibration();auto const& source=*record.frame.source;
    json result={{"schema","stereoforge.daxa_depth_rectified_benchmark.v1"},{"variant",variant},{"frames",frames},{"warmup",warmup},{"slots",2},{"radius",radius},{"subpixel",refine?refinement_config.subpixel:depth_config.stereo.subpixel},{"sgm_subpixel",depth_config.stereo.subpixel},
        {"scope","static_repeated_canonical_record; no_live_quality_or_pipeline_FPS_claim"},{"percentile_policy","nearest_rank_ceil_p_times_N_minus1"},
        {"manifest_sha256",record.manifest_sha256},{"record_sha256",record.record_sha256},{"geometry_supplement_sha256",record.geometry_supplement_sha256},{"coarse_calibration_sha256",record.coarse_calibration_sha256},
        {"producer",record.producer},{"record_ordinal",record.ordinal},{"source_id",source.frame_id},{"camera_timestamp_ns",source.camera_timestamp_ns},
        {"input_identity",source.canonical_rgb->input_identity},{"original_pixel_identity",source.canonical_rgb->original_pixel_identity},{"rectified_pixel_identity",source.canonical_rgb->rectified_pixel_identity},
        {"original_calibration_identity",full.original_identity},{"rectified_calibration_identity",full.rectified_identity},{"sampling_profile",canonical_rgb_sampling_profile},
        {"width",full.width},{"height",full.height},{"sgm_disparities",depth_config.stereo.disparities},{"sgm_directions",4},{"right_reference_mode","independent_sgm"},{"lr_dependency","independent_references"},
        {"metric_calibration",{{"fx",calibration.fx},{"baseline_m",calibration.baseline_m},{"cx_offset",calibration.cx_offset}}},
        {"device",json::parse(context->report())},{"validation_requested",options.validation},
        {"requested_producer_buffer_bytes",producer->requested_bytes()},{"requested_depth_buffer_bytes",depth->requested_combined_buffer_bytes()},{"requested_refinement_buffer_bytes",refinement?refinement->requested_buffer_bytes():0},
        {"memory_scope","requested buffer bytes; excludes driver allocator overhead, CPU admission and optional diagnostics"},
        {"combined_scope","direct shared queue timestamp difference from producer upload start through final metric end including queue gaps"},
        {"shader_compiler",{{"version",SF_DAXA_SLANG_VERSION},{"compiler_sha256",SF_DAXA_COMPILER_SHA256},{"floating_point_mode","default"}}},
        {"producer_shader_identity",DaxaRectifiedInput::shader_identity()},{"depth_shader_identity",DaxaDepthPipeline::shader_identity()},
        {"producer_spirv_sha256",shader_hashes(DaxaRectifiedInput::shader_directory(),"daxa-rectified-input-")},{"depth_spirv_sha256",shader_hashes(DaxaDepthPipeline::shader_directory(),"daxa-depth-")}};
    result["requested_combined_buffer_bytes"]=producer->requested_bytes()+depth->requested_combined_buffer_bytes()+(refinement?refinement->requested_buffer_bytes():0);
    result["stereo_spirv_sha256"]=shader_hashes(DaxaStereo::shader_directory(),"daxa-stereo-");
    result["shader_hash_scope"]="compiled shader family bundles; includes unused variants";
    result["manifest_path"]=std::filesystem::absolute(options.values.at("--manifest")).string();
    result["camera_clock_domain"]=source.camera_clock_domain;result["host_clock_domain"]=source.host_clock_domain;
    result["host_ingress_timestamp_ns"]=source.host_ingress_timestamp_ns;result["device_identity"]=source.device_identity;
    for(auto const& file:record.files)result["admitted_files"].push_back({{"path",file.path.string()},{"sha256",file.sha256},{"size_bytes",file.size_bytes}});
    if(refinement) {
        result["refinement_shader_compiler"]={{"version",DaxaRefinement::compiler_version()},{"commit",DaxaRefinement::compiler_commit()},{"package_sha256",DaxaRefinement::compiler_package_sha256()},{"compiler_sha256",DaxaRefinement::compiler_sha256()},{"floating_point_mode","precise"}};
        result["prior_confidence_floor"]=refinement_config.prior_confidence_floor;result["max_coarse_span"]=refinement_config.max_coarse_span;result["min_margin"]=refinement_config.min_margin;result["lr_tolerance"]=refinement_config.lr_tolerance;result["recovery"]=false;
        result["coarse_width"]=coarse->width;result["coarse_height"]=coarse->height;result["coarse_rectified_identity"]=coarse->rectified_identity;
        result["refinement_subpixel"]=refinement_config.subpixel;
        result["final_estimator"]="independent-local-census";result["final_right_reference_mode"]="independent-local-census";
        result["refinement_disparities"]=refinement_config.disparities;result["refinement_shader_identity"]=DaxaRefinement::shader_identity();
        result["refinement_spirv_sha256"]=shader_hashes(DaxaRefinement::shader_directory(),"daxa-refinement-");
    } else result["final_estimator"]="independent_sgm";
    for(unsigned k=0;k<Count;++k) {
        bool available=(refine||!(k>=RefinePack&&k<=RefineSpan))&&(refine||k!=CoarseGpu);
        result[keys[k]]=available?summary(samples[k]):json(nullptr);
        result["raw_samples_ms"][keys[k]]=available?json(samples[k]):json(nullptr);
    }
    for(unsigned eye=0;eye<2;++eye)for(unsigned k=0;k<stage_counts[eye];++k) {
        auto key=std::string(eye?"right_":"left_")+std::to_string(k);
        result["stereo_stage_ms"][key]=summary(stereo_stages[eye][k]);
        result["stereo_stage_raw_samples_ms"][key]=stereo_stages[eye][k];
    }
    add_native_timing_metadata(result,frames,warmup,"nearest_rank_ceil_p_times_N_minus1");
    if(options.values.contains("--dump-prefix")) {
        auto start=Clock::now();auto const& prefix=options.values.at("--dump-prefix");
        if(refinement)result["files"]=dump(refinement->readback(output),prefix);
        else result["files"]=dump_depth(depth->readback(stereo),prefix);
        result["diagnostic_after_timing_ms"]=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    }
    output={};if(refinement) { refinement->collect();result["refinement_source_retirements"]=refinement->completed_source_retirements();refinement.reset(); }
    stereo={};depth->drain();result["depth_output_retirements"]=depth->completed_output_retirements();depth.reset();
    input={};producer->collect();producer.reset();context->close();
    result["validation_errors_after_teardown"]=context->validation_errors.load();result["validation_warnings_after_teardown"]=context->validation_warnings.load();
    check(!context->validation_errors&&!context->validation_warnings,"canonical benchmark validation failed through teardown");
    verify_canonical_record_files(record);
    auto bytes=result.dump(2)+'\n';write(options.values.at("--output"),bytes.data(),bytes.size());
    return 0;
} catch(std::exception const& e) { std::cerr<<"daxa_depth_rectified_benchmark: "<<e.what()<<'\n';return 1; } }
