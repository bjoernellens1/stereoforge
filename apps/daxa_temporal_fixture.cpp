// Diagnostic only: controlled observations replace actual completed Depth
// output. This executable does not measure SGM accuracy, physical geometry or
// performance.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <limits>
#include <map>
#include <stereoforge/daxa_temporal.hpp>
#include <stereoforge/daxa_view.hpp>
#include <type_traits>
#include <unistd.h>
using namespace stereoforge;
using json = nlohmann::json;
namespace {
void check(bool b, const char *m) {
  if (!b)
    throw std::invalid_argument(m);
}
using Matrix = std::array<double, 16>;
constexpr Matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
void write_file(std::filesystem::path const &path, void const *data,
                size_t bytes) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
  check(fd >= 0, "fresh exclusive fixture output required");
  size_t at = 0;
  while (at < bytes) {
    auto n = ::write(fd, static_cast<char const *>(data) + at, bytes - at);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0) {
      ::close(fd);
      throw std::runtime_error("fixture write failed");
    }
    at += size_t(n);
  }
  check(::close(fd) == 0, "fixture output close failed");
}
json read_json(std::filesystem::path const &path) {
  std::ifstream f(path);
  check(bool(f), "fixture JSON unavailable");
  json out;
  f >> out;
  return out;
}
std::string digest(std::filesystem::path const &p) {
  return daxa_preprocess_file_sha256(p);
}
Matrix matrix(json const &value) {
  std::vector<double> flat;
  auto walk = [&](auto &&self, json const &a) -> void {
    if (a.is_array())
      for (auto const &b : a)
        self(self, b);
    else {
      check(a.is_number(), "numeric matrix required");
      flat.push_back(a.get<double>());
    }
  };
  walk(walk, value);
  check(flat.size() == 16, "pose requires16 scalars");
  Matrix out;
  std::copy(flat.begin(), flat.end(), out.begin());
  return out;
}
std::array<double, 9> rotation_transpose(Matrix const &m) {
  std::array<double, 9> r{};
  for (unsigned y = 0; y < 3; ++y)
    for (unsigned x = 0; x < 3; ++x)
      r[y * 3 + x] = m[x * 4 + y];
  return r;
}
Matrix eye_to_rig(DaxaRectifiedGeometry const &g, unsigned eye) {
  auto r = eye ? g.R2 : g.R1;
  Matrix rt = identity, out{};
  for (unsigned y = 0; y < 3; ++y)
    for (unsigned x = 0; x < 3; ++x)
      rt[y * 4 + x] = r[x * 3 + y];
  if (!eye)
    return rt;
  for (unsigned y = 0; y < 4; ++y)
    for (unsigned x = 0; x < 4; ++x)
      for (unsigned k = 0; k < 4; ++k)
        out[y * 4 + x] += g.original.right_to_left[y * 4 + k] * rt[k * 4 + x];
  return out;
}
void match_geometry(DaxaRectification const &maps, json const &geometry) {
  check(maps.width == geometry.at("width") &&
            maps.height == geometry.at("height") && maps.geometry.has_value(),
        "producer/sequence grid differs");
  auto const &g = *maps.geometry;
  for (unsigned eye = 0; eye < 2; ++eye) {
    auto const &k = g.rectified.eyes[eye];
    auto expected =
        geometry.at("intrinsics").at(eye).get<std::array<double, 4>>();
    std::array<double, 4> actual{k.fx, k.fy, k.cx, k.cy};
    for (unsigned i = 0; i < 4; ++i)
      check(actual[i] == expected[i], "producer/sequence intrinsics differ");
    auto a = eye_to_rig(g, eye),
         wanted = matrix(geometry.at("eye_to_rig").at(eye));
    for (unsigned i = 0; i < 16; ++i)
      check(std::abs(a[i] - wanted[i]) <= 1e-10,
            "producer/sequence eye-to-rig differs");
  }
  g.validate_map_association(
      {maps.maps[0], maps.maps[1], maps.maps[2], maps.maps[3]});
}
struct Producer {
  DaxaRectification maps;
  std::shared_ptr<CaptureFrame> prototype;
  json binding;
  Producer(json const &sequence, std::filesystem::path const &path) {
    auto const &geometry = sequence.at("geometry");
    auto const &first = sequence.at("frames").at(0);
    prototype = std::make_shared<CaptureFrame>();
    if (!sequence.at("producer_manifest").is_null()) {
      auto manifest = sequence.at("producer_manifest");
      auto relative =
          std::filesystem::path(manifest.at("file").get<std::string>());
      check(!relative.is_absolute(), "relative producer manifest required");
      for (auto const &part : relative)
        check(part != "..",
              "producer manifest cannot escape sequence directory");
      auto file = path.parent_path() / relative;
      check(digest(file) == manifest.at("sha256").get<std::string>(),
            "producer manifest hash mismatch");
      auto fixture = load_daxa_preprocess_fixture(file);
      maps = fixture.rectification;
      auto replay = packed_replay_source(fixture.capture_directory);
      for (uint64_t i = 0; i <= fixture.record_ordinal; ++i)
        check(replay(*prototype, std::chrono::milliseconds(100)) ==
                  CaptureSourceStatus::recorded_frame,
              "producer capture missing");
      maps.validate_source(*prototype);
      binding = {{"kind", "bound_synthetic_recorded_producer"},
                 {"manifest_file", std::filesystem::absolute(file).string()},
                 {"manifest_sha256", digest(file)},
                 {"capture_file",
                  (fixture.capture_directory / "capture.sfp").string()},
                 {"capture_sha256", fixture.capture_sha256}};
    } else {
      maps.width = geometry.at("width");
      maps.height = geometry.at("height");
      maps.eye_width = (maps.width + 1u) & ~1u;
      maps.source_height = maps.height;
      maps.stride_bytes = maps.eye_width * 4;
      maps.original_identity = first.at("calibration_id");
      maps.rectified_identity = maps.original_identity + "/rectified";
      DaxaRectifiedGeometry g;
      g.original_identity = maps.original_identity;
      g.rectified_identity = maps.rectified_identity;
      auto left = matrix(geometry.at("eye_to_rig").at(0)),
           right = matrix(geometry.at("eye_to_rig").at(1));
      check(left[3] == 0 && left[7] == 0 && left[11] == 0,
            "original-left grid transform cannot translate");
      g.R1 = rotation_transpose(left);
      g.R2 = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      g.original.right_to_left = right;
      g.rectified.right_to_left = identity;
      double baseline = 0;
      for (unsigned k = 0; k < 3; ++k)
        baseline += g.R1[k] * right[k * 4 + 3];
      g.rectified.right_to_left[3] = baseline;
      for (unsigned eye = 0; eye < 2; ++eye) {
        auto k = geometry.at("intrinsics").at(eye).get<std::array<double, 4>>();
        g.rectified.eyes[eye] = {maps.width, maps.height, k[0], k[1],
                                 k[2],       k[3],        {}};
        g.original.eyes[eye] = {
            maps.eye_width, maps.source_height, k[0], k[1], k[2], k[3], {}};
        auto &p = eye ? g.P2 : g.P1;
        p = {k[0], 0, k[2], eye ? -k[0] * baseline : 0, 0, k[1], k[3], 0, 0,
             0,    1, 0};
        auto a = eye ? right : left;
        for (unsigned y = 0; y < maps.height; ++y)
          for (unsigned x = 0; x < maps.width; ++x) {
            std::array<double, 3> ray{(double(x) - k[2]) / k[0],
                                      (double(y) - k[3]) / k[1], 1},
                q{};
            if (eye)
              q = ray;
            else
              for (unsigned i = 0; i < 3; ++i)
                for (unsigned j = 0; j < 3; ++j)
                  q[i] += a[i * 4 + j] * ray[j];
            check(q[2] > 0, "unsupported synthetic calibration hemisphere");
            auto uv = detail::daxa_calibration_distort(
                g.original.eyes[eye], q[0] / q[2], q[1] / q[2]);
            maps.maps[2 * eye].push_back(float(uv[0]));
            maps.maps[2 * eye + 1].push_back(float(uv[1]));
          }
      }
      maps.geometry = g;
      prototype->source_width = 2 * maps.eye_width;
      prototype->height = maps.source_height;
      prototype->stride_bytes = maps.stride_bytes;
      prototype->packed_yuyv.resize(size_t(maps.stride_bytes) *
                                    maps.source_height);
      for (size_t i = 0; i < prototype->packed_yuyv.size(); ++i)
        prototype->packed_yuyv[i] = uint8_t((i * 17 + 3) & 255);
      prototype->calibration_identity = maps.original_identity;
      binding = {
          {"kind", "sequence_bound_generated_synthetic_calibrated_depth"},
          {"recorded_producer_qualified", false},
          {"source_pixels", "deterministic byte(i*17+3) modulo256"}};
    }
    match_geometry(maps, geometry);
    maps.metric_calibration();
    prototype->frame_id = 1;
    prototype->host_ingress_timestamp_ns = 0;
    prototype->host_clock_domain = "fixture-setup";
    prototype->device_identity = "synthetic-temporal-fixture-setup";
    binding["original_identity"] = maps.original_identity;
    binding["rectified_identity"] = maps.rectified_identity;
    binding["rectified_width"] = maps.width;
    binding["rectified_height"] = maps.height;
  }
  void verify() const {
    if (binding.contains("manifest_file")) {
      check(digest(binding.at("manifest_file").get<std::string>()) ==
                binding.at("manifest_sha256").get<std::string>(),
            "producer manifest changed");
      check(digest(binding.at("capture_file").get<std::string>()) ==
                binding.at("capture_sha256").get<std::string>(),
            "producer capture changed");
    }
  }
};
template <class T>
std::vector<T> plane(json const &input, unsigned width, unsigned height,
                     bool depth = false) {
  check(input.is_array() && input.size() == 2,
        "two-eye literal plane required");
  std::vector<T> out;
  out.reserve(size_t(width) * height * 2);
  for (auto const &eye : input) {
    check(eye.is_array() && eye.size() == height, "literal height mismatch");
    for (auto const &row : eye) {
      check(row.is_array() && row.size() == width, "literal width mismatch");
      for (auto const &value : row) {
        if (depth && value.is_null()) {
          if constexpr (std::is_same_v<T, float>)
            out.push_back(std::bit_cast<float>(0x7fc00000u));
          else
            throw std::invalid_argument("null integer");
        } else {
          check(value.is_number(), "numeric literal plane required");
          if constexpr (std::is_integral_v<T>) {
            check(value.is_number_integer() || value.is_number_unsigned(),
                  "integer literal plane required");
            if (value.is_number_unsigned())
              check(value.get<uint64_t>() <=
                        uint64_t(std::numeric_limits<T>::max()),
                    "integer literal overflow");
            else {
              auto v = value.get<int64_t>();
              check(v >= int64_t(std::numeric_limits<T>::min()) &&
                        v <= int64_t(std::numeric_limits<T>::max()),
                    "integer literal range mismatch");
            }
          }
          out.push_back(value.get<T>());
        }
      }
    }
  }
  return out;
}
struct Fresh {
  std::vector<float> z, confidence;
  std::vector<uint32_t> valid, provenance, support;
  std::vector<int64_t> ids, age;
  Fresh(json const &row, json const &geometry, unsigned w, unsigned h)
      : z(plane<float>(row.at("fresh").at("depth"), w, h, true)),
        confidence(plane<float>(row.at("fresh").at("confidence"), w, h)),
        valid(plane<uint32_t>(row.at("fresh").at("validity"), w, h)),
        provenance(plane<uint32_t>(row.at("fresh").at("provenance"), w, h)),
        support(plane<uint32_t>(geometry.at("support"), w, h)),
        ids(plane<int64_t>(row.at("fresh").at("source_ids"), w, h)),
        age(plane<int64_t>(row.at("fresh").at("age_ns"), w, h)) {
    for (size_t i = 0; i < z.size(); ++i) {
      check(valid[i] <= 1 && support[i] <= 1,
            "binary validity/support required");
      check(std::isfinite(confidence[i]) && confidence[i] >= 0 &&
                confidence[i] <= 1,
            "fresh confidence invalid");
      check(valid[i] ? (std::isfinite(z[i]) && z[i] > 0 &&
                        (provenance[i] == 1 || provenance[i] == 5) &&
                        ids[i] >= 0 && age[i] >= 0)
                     : (std::bit_cast<uint32_t>(z[i]) == 0x7fc00000u &&
                        confidence[i] == 0 && provenance[i] == 0 &&
                        ids[i] == -1 && age[i] == 0),
            "malformed fresh planes are diagnostic-reference-only");
    }
  }
  void upload(DaxaDepthPipeline::Frame const &frame,
              DaxaFoundation &context) const {
    frame.wait();
    auto &d = context.device;
    std::vector<daxa::BufferId> staging;
    std::vector<daxa::MemoryBlock> memory;
    auto recorder = d.create_command_recorder(
        {.name = "diagnostic exclusive controlled fresh planes and support"});
    auto put = [&](daxa::BufferId dst, auto const &values) {
      daxa::BufferInfo info{.size = values.size() * sizeof(values[0]),
                            .name = "temporal fixture diagnostic staging"};
      auto block = d.create_memory(
          {.requirements = context.coherent_readback_requirements(info),
           .flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
      auto id = d.create_buffer_from_memory_block(
          {.buffer_info = info, .memory_block = block});
      memory.push_back(std::move(block));
      staging.push_back(id);
      std::memcpy(d.buffer_host_address(id).value(), values.data(),
                  size_t(info.size));
      recorder.copy_buffer_to_buffer(
          {.src_buffer = id, .dst_buffer = dst, .size = info.size});
    };
    recorder.pipeline_barrier(
        {.src_access = daxa::AccessConsts::HOST_WRITE |
                       daxa::AccessConsts::COMPUTE_SHADER_WRITE,
         .dst_access = daxa::AccessConsts::TRANSFER_READ |
                       daxa::AccessConsts::TRANSFER_WRITE});
    put(frame.buffer(DaxaDepthPlane::Depth), z);
    put(frame.buffer(DaxaDepthPlane::MetricConfidence), confidence);
    put(frame.buffer(DaxaDepthPlane::MetricValidity), valid);
    put(frame.buffer(DaxaDepthPlane::Provenance), provenance);
    put(frame.buffer(DaxaDepthPlane::SourceIds), ids);
    put(frame.buffer(DaxaDepthPlane::AgeNs), age);
    size_t n = z.size() / 2;
    for (unsigned eye = 0; eye < 2; ++eye) {
      std::vector<uint32_t> values(support.begin() + eye * n,
                                   support.begin() + (eye + 1) * n);
      put(frame
              .image(eye ? DaxaPreprocessPlane::RightSupport
                         : DaxaPreprocessPlane::LeftSupport)
              .buffer(),
          values);
    }
    recorder.pipeline_barrier(
        {.src_access = daxa::AccessConsts::TRANSFER_WRITE,
         .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ});
    auto command = recorder.complete_current_commands();
    d.submit_commands({.command_lists = {&command, 1}});
    auto ticket = d.latest_queue_submit_index(daxa::QUEUE_MAIN);
    d.wait_on_submit({.queue = daxa::QUEUE_MAIN,
                      .queue_submit_index = ticket,
                      .timeout = 30'000'000'000ull});
    for (auto id : staging)
      d.destroy_buffer(id);
    d.collect_garbage();
  }
};
json counters(DaxaTemporalCounters const &c) {
  return {{"fresh_domain_rejected", c.fresh_domain_rejected},
          {"history_input_domain_rejected", c.history_input_domain_rejected},
          {"projected_domain_rejected", c.projected_domain_rejected},
          {"arithmetic_faults", c.arithmetic_faults},
          {"history_scanned", c.history_scanned},
          {"history_eligible", c.history_eligible},
          {"history_projected", c.history_projected},
          {"history_winners", c.history_winners},
          {"filled", c.filled},
          {"fused", c.fused}};
}
template <class T>
json dump(std::filesystem::path const &directory, unsigned index,
          char const *name, std::vector<T> const &values) {
  char prefix[32];
  std::snprintf(prefix, sizeof(prefix), "frame-%04u.", index);
  auto filename = std::string(prefix) + name;
  auto p = directory / filename;
  write_file(p, values.data(), values.size() * sizeof(T));
  return {{"file", filename}, {"sha256", digest(p)}};
}
json dump_temporal(std::filesystem::path const &directory, unsigned index,
                   DaxaTemporalReadback const &r) {
  json p;
  p["depth.f32"] = dump(directory, index, "depth.f32", r.depth);
  p["confidence.f32"] = dump(directory, index, "confidence.f32", r.confidence);
  p["validity.u32"] = dump(directory, index, "validity.u32", r.validity);
  p["provenance.u32"] = dump(directory, index, "provenance.u32", r.provenance);
  p["source_ids.i64"] = dump(directory, index, "source_ids.i64", r.source_ids);
  p["age_ns.i64"] = dump(directory, index, "age_ns.i64", r.age_ns);
  p["contributor_ids.i64"] =
      dump(directory, index, "contributor_ids.i64", r.contributor_ids);
  p["contributor_count.u32"] =
      dump(directory, index, "contributor_count.u32", r.count);
  p["oldest_age_ns.i64"] =
      dump(directory, index, "oldest_age_ns.i64", r.oldest_age_ns);
  p["reuse.u32"] = dump(directory, index, "reuse.u32", r.reuse);
  p["origin_mask.u32"] =
      dump(directory, index, "origin_mask.u32", r.origin_mask);
  return p;
}
json view_checks(std::shared_ptr<DaxaFoundation> const &context,
                 DaxaObservationFrame const &common,
                 DaxaTemporalReadback const &temporal,
                 std::filesystem::path const &directory, unsigned index) {
  auto maps = common.map_owner();
  auto const &g = *maps->geometry;
  GeometryConfig cfg{maps->width, maps->height, maps->eye_width,
                     maps->source_height};
  VirtualView v;
  auto const &k = g.original.eyes[0];
  v.intrinsics = {float(k.fx), float(k.fy), float(k.cx), float(k.cy)};
  for (unsigned i = 0; i < 16; ++i)
    v.source_to_virtual_right[i] = float(g.original.right_to_left[i]);
  json result = json::array();
  for (auto mode : {ViewMode::Strict, ViewMode::Preview}) {
    v.mode = mode;
    DaxaView view(cfg, v, context, common, 1);
    auto output = view.project(common);
    output.wait();
    auto retained = output.source_observations();
    auto ancestry = retained.ancestry();
    check(retained.generation() == common.generation() &&
              retained.source() == common.source() &&
              ancestry.kind() ==
                  DaxaObservationAncestryKind::BoundedContributors,
          "View must retain actual Temporal lineage source");
    for (unsigned p = 0; p < 5; ++p)
      check(ancestry.buffer(DaxaObservationAncestryPlane(p)) ==
                common.ancestry().buffer(DaxaObservationAncestryPlane(p)),
            "View lineage buffer association changed");
    auto r = view.readback(output);
    std::vector<int64_t> gathered(r.owner.size() * 4, -1);
    std::vector<uint32_t> count(r.owner.size());
    for (size_t i = 0; i < r.owner.size(); ++i) {
      if (r.owner[i] == UINT32_MAX) {
        check(!r.validity[i], "preview owner must not carry metric lineage");
        continue;
      }
      auto owner = r.owner[i];
      check(owner < temporal.depth.size() && r.validity[i] &&
                temporal.validity[owner],
            "View metric owner outside Temporal observation namespace");
      count[i] = temporal.count[owner];
      for (unsigned a = 0; a < 4; ++a)
        gathered[i * 4 + a] = temporal.contributor_ids[owner * 4 + a];
      check(r.source_ids[i] == temporal.source_ids[owner] &&
                r.age_ns[i] == temporal.age_ns[owner] &&
                r.provenance[i] == temporal.provenance[owner],
            "View selected metric metadata owner mismatch");
    }
    auto name = mode == ViewMode::Strict ? "view-strict" : "view-preview";
    result.push_back(
        {{"mode", name},
         {"owner", dump(directory, index,
                        (std::string(name) + ".owner.u32").c_str(), r.owner)},
         {"validity",
          dump(directory, index, (std::string(name) + ".validity.u32").c_str(),
               r.validity)},
         {"contributor_ids",
          dump(directory, index,
               (std::string(name) + ".contributor_ids.i64").c_str(), gathered)},
         {"contributor_count",
          dump(directory, index,
               (std::string(name) + ".contributor_count.u32").c_str(), count)},
         {"scope",
          "actual_GPU_View_owners_and_retained_GPU_lineage_handles_with_CPU_"
          "diagnostic_owner_gather_not_GPU_ancestry_gather"}});
  }
  return result;
}
json identities() {
  json out;
  auto root = std::filesystem::path(__FILE__).parent_path().parent_path();
  for (auto file :
       {"apps/daxa_temporal_fixture.cpp",
        "include/stereoforge/daxa_temporal.hpp", "src/daxa_temporal.cpp",
        "include/stereoforge/daxa_observation.hpp", "src/daxa_observation.cpp",
        "include/stereoforge/daxa_calibration.hpp", "src/daxa_calibration.cpp",
        "src/daxa_view_calibration.cpp", "src/daxa_view.cpp",
        "python/stereoforge/shaders/daxa/temporal.slang",
        "python/stereoforge/shaders/ops/temporal.slang", "CMakeLists.txt"})
    out["source_sha256"][file] = digest(root / file);
  out["binary_sha256"] = digest("/proc/self/exe");
  out["compiler_sha256"] = SF_DAXA_COMPILER_SHA256;
  for (auto const &entry :
       std::filesystem::recursive_directory_iterator(DaxaTemporal::shader_directory()))
    if (entry.path().extension() == ".spv")
      out["spirv_sha256"][std::filesystem::absolute(entry.path()).string()] =
          digest(entry.path());
  return out;
}
} // namespace
int main(int argc, char **argv) {
  try {
    check(std::endian::native == std::endian::little,
          "little-endian diagnostic export required");
    std::map<std::string, std::string> args;
    bool validation = false;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--validation") {
        check(!validation, "duplicate validation flag");
        validation = true;
      } else {
        check(key == "--sequence" || key == "--output" ||
                  key == "--dump-directory",
              "unknown temporal fixture option");
        check(++i < argc, "missing option value");
        check(args.emplace(key, argv[i]).second, "duplicate option");
      }
    }
    check(args.size() == 3, "sequence/output/dump-directory required");
    auto sequence_path = std::filesystem::absolute(args.at("--sequence"));
    auto sequence = read_json(sequence_path);
    auto sequence_hash = digest(sequence_path);
    check(sequence.at("schema") == "stereoforge.temporal.sequence.v1",
          "sequence schema mismatch");
    auto directory = std::filesystem::path(args.at("--dump-directory"));
    check(!std::filesystem::exists(directory) &&
              !std::filesystem::exists(args.at("--output")),
          "fresh output destinations required");
    std::filesystem::create_directories(directory);
    if (sequence.at("geometry").at("profile") == "diagnostic_only") {
      json skipped = {{"schema", "stereoforge.daxa_temporal.fixture.v1"},
                      {"sequence_sha256", sequence_hash},
                      {"native_executed", false},
                      {"qualified", false},
                      {"skip_reason",
                       "diagnostic_only_geometry_outside_actual_Depth_domain"},
                      {"frames", json::array()}};
      unsigned index = 0;
      for (auto const &row : sequence.at("frames"))
        skipped["frames"].push_back(
            {{"index", index++},
             {"qualification_scope", row.at("qualification_scope")},
             {"accepted", false},
             {"native_executed", false},
             {"skip_reason", "diagnostic_only_geometry"}});
      auto text = skipped.dump(2) + '\n';
      write_file(args.at("--output"), text.data(), text.size());
      return 0;
    }
    check(sequence.at("geometry").at("profile") == "native_depth_compatible",
          "unknown geometry profile");
    auto frozen = identities();
    Producer producer(sequence, sequence_path);
    auto context = std::make_shared<DaxaFoundation>(validation);
    json report = {{"schema", "stereoforge.daxa_temporal.fixture.v1"},
                   {"sequence_sha256", sequence_hash},
                   {"observations_overridden_for_fixture", true},
                   {"support_overridden_for_fixture", true},
                   {"measured_stereo_output", false},
                   {"performance_qualified", false},
                   {"physical_qualified", false},
                   {"validation_requested", validation},
                   {"scope",
                    "actual_synthetic_calibrated_Depth_to_Temporal_to_View_"
                    "controlled_planes_no_SGM_quality_or_performance"},
                   {"producer", producer.binding},
                   {"producer_binding",
                    {{"kind", "literal_controlled_planes_after_actual_Depth"},
                     {"geometry_models_checked", true},
                     {"calibration_map_association_checked", true}}},
                   {"identities", frozen},
                   {"frames", json::array()}};
    {
      DaxaDepthConfig dc{
          StereoConfig{producer.maps.width, producer.maps.height, 3, 4}, 1.f, 2,
          32};
      DaxaDepthPipeline depth(producer.maps, dc, context, producer.prototype);
      auto setup = depth.enqueue(producer.prototype);
      auto const &cfg = sequence.at("config");
      DaxaTemporalConfig tc;
      tc.slots = 2;
      tc.max_contributors = cfg.at("max_contributors");
      tc.max_reuses = cfg.at("max_reuse");
      tc.max_age_ns = cfg.at("max_age_ns");
      tc.relative_depth_tolerance = cfg.at("relative_z_tolerance");
      tc.confidence_floor = cfg.at("min_confidence");
      DaxaTemporal temporal(context, setup, tc);
      setup.wait();
      setup = {};
      depth.collect();
      unsigned index = 0;
      for (auto const &row : sequence.at("frames")) {
        std::string scope = row.at("qualification_scope");
        json actual = {{"index", index},
                       {"accepted", false},
                       {"qualification_scope", scope},
                       {"native_executed", false}};
        if (scope == "diagnostic_reference_only") {
          actual["skip_reason"] = "diagnostic_reference_only";
          report["frames"].push_back(actual);
          ++index;
          continue;
        }
        check(scope == "native" || scope == "native_host_rejection",
              "unknown qualification scope");
        Fresh fresh(row, sequence.at("geometry"), producer.maps.width,
                    producer.maps.height);
        auto source = std::make_shared<CaptureFrame>(*producer.prototype);
        source->frame_id = row.at("capture_id");
        source->host_ingress_timestamp_ns = row.at("host_ingress_timestamp_ns");
        source->host_clock_domain = row.at("host_clock_domain");
        source->device_identity = row.at("device_id");
        source->calibration_identity = row.at("calibration_id");
        if (row.at("reset").get<bool>())
          temporal.reset();
        temporal.collect();
        depth.collect();
        DaxaDepthPipeline::Frame input;
        auto const before_depth = context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);
        try {
          input = depth.enqueue(source);
        } catch (std::invalid_argument const &e) {
          check(scope == "native_host_rejection",
                "unexpected upstream Depth admission rejection");
          actual["native_executed"] = true;
          actual["rejection_stage"] = "actual_Depth_producer_admission";
          actual["rejection"] = e.what();
          actual["rejected_before_submission"] =
              before_depth == context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);
          check(actual["rejected_before_submission"].get<bool>(),
                "upstream host rejection submitted GPU work");
          report["frames"].push_back(std::move(actual));
          ++index;
          continue;
        }
        fresh.upload(input, *context);
        std::optional<DaxaTimedRigPose> pose;
        std::string domain;
        if (!row.at("pose").is_null()) {
          auto const &p = row.at("pose");
          domain = p.at("clock_domain");
          pose = DaxaTimedRigPose{matrix(p.at("world_from_original_left")),
                                  p.at("timestamp_ns").get<int64_t>(), domain};
        }
        actual["native_executed"] = true;
        auto before =
            context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);
        DaxaTemporal::Frame output;
        try {
          output = temporal.process(input, pose);
        } catch (std::invalid_argument const &e) {
          actual["rejection_stage"] = "actual_Temporal_host_admission";
          actual["rejection"] = e.what();
          actual["rejected_before_submission"] =
              before ==
              context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);
          check(actual["rejected_before_submission"].get<bool>(),
                "host rejection submitted GPU work");
          check(scope == "native_host_rejection",
                "unexpected native Temporal admission rejection");
        }
        if (output) {
          check(scope != "native_host_rejection",
                "declared native host rejection was accepted");
          output.wait();
          auto r = temporal.readback(output);
          actual["accepted"] = true;
          actual["planes"] = dump_temporal(directory, index, r);
          auto all_counters = counters(r.counters);
          actual["domain_counters"] = json::object();
          actual["work_counters"] = json::object();
          for (auto const *key :
               {"fresh_domain_rejected", "history_input_domain_rejected",
                "projected_domain_rejected", "arithmetic_faults"})
            actual["domain_counters"][key] = all_counters.at(key);
          for (auto const *key :
               {"history_scanned", "history_eligible", "history_projected",
                "history_winners", "filled", "fused"})
            actual["work_counters"][key] = all_counters.at(key);
          actual["source_metadata"] = {
              {"capture_id", source->frame_id},
              {"host_ingress_timestamp_ns", source->host_ingress_timestamp_ns},
              {"host_clock_domain", source->host_clock_domain},
              {"device_id", source->device_identity},
              {"calibration_id", source->calibration_identity}};
          auto common = output.observations();
          if (sequence.value("check_view_lineage", true))
            actual["view_checks"] =
                view_checks(context, common, r, directory, index);
        }
        output = {};
        input = {};
        temporal.collect();
        depth.collect();
        report["frames"].push_back(std::move(actual));
        ++index;
      }
      report["requested_temporal_bytes"] = temporal.requested_buffer_bytes();
      report["completed_source_retirements"] =
          temporal.completed_source_retirements();
      depth.drain();
    }
    context->close();
    report["validation_errors"] = context->validation_errors.load();
    report["validation_warnings"] = context->validation_warnings.load();
    check(context->validation_errors == 0 && context->validation_warnings == 0,
          "validation diagnostics through fixture teardown");
    check(digest(sequence_path) == sequence_hash,
          "sequence changed during fixture run");
    producer.verify();
    check(identities() == frozen,
          "source/compiler/binary/SPIRV identity changed during fixture run");
    auto text = report.dump(2) + '\n';
    write_file(args.at("--output"), text.data(), text.size());
    std::cout << "Actual native diagnostic sequence exported; no stereo "
                 "quality/performance qualification\n";
    return 0;
  } catch (std::exception const &e) {
    std::cerr << "daxa_temporal_fixture: " << e.what() << '\n';
    return 1;
  }
}
