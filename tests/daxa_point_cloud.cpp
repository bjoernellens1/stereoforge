#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stereoforge/daxa_depth.hpp>
#include <stereoforge/daxa_point_cloud.hpp>
#include <stereoforge/daxa_preprocess.hpp>
#include <vector>
using namespace stereoforge;
static void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
template <class F> static void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (std::exception const &) {
    caught = true;
  }
  check(caught, "expected cloud rejection");
}
struct GateRelease {
  daxa::TimelineSemaphore gate;
  bool released = false;
  void release() {
    if (!released) {
      gate.signal_value(1);
      released = true;
    }
  }
  ~GateRelease() {
    if (!released) {
      try {
        release();
      } catch (...) {
        std::terminate();
      }
    }
  }
};

// Unwind releases the queue via the later semaphore guard before freeing this.
struct TemporaryBuffer {
  std::shared_ptr<DaxaFoundation> context;
  daxa::BufferId id;
  ~TemporaryBuffer() noexcept {
    try {
      auto &device = context->device;
      auto ticket = device.latest_queue_submit_index(daxa::QUEUE_MAIN);
      if (ticket)
        device.wait_on_submit({.queue = daxa::QUEUE_MAIN,
                               .queue_submit_index = ticket,
                               .timeout = 30'000'000'000ull});
      device.destroy_buffer(id);
      device.collect_garbage();
    } catch (...) {
      std::terminate();
    }
  }
};

static DaxaRectification grid(bool extreme = false) {
  DaxaRectification c;
  c.eye_width = 8;
  c.width = 7;
  c.source_height = c.height = 5;
  c.stride_bytes = 32;
  c.original_identity = "cloud-original";
  c.rectified_identity = "cloud-rectified";
  DaxaRectifiedGeometry g;
  g.original_identity = c.original_identity;
  g.rectified_identity = c.rectified_identity;
  for (auto &k : g.original.eyes)
    k = {8, 5, 8, 10, 2.25, 1.75, {}};
  g.rectified = g.original;
  for (auto &k : g.rectified.eyes)
    k.width = 7;
  g.original.right_to_left = g.rectified.right_to_left = {
      1, 0, 0, .1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  g.R1 = g.R2 = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  g.P1 = {8, 0, 2.25, 0, 0, 10, 1.75, 0, 0, 0, 1, 0};
  g.P2 = {8, 0, 2.25, -.8, 0, 10, 1.75, 0, 0, 0, 1, 0};
  if (extreme) {
    for (auto *pair : {&g.original, &g.rectified})
      for (auto &eye : pair->eyes)
        eye.cx = -16;
    g.P1[2] = g.P2[2] = -16;
  }
  c.geometry = g;
  for (unsigned p = 0; p < 4; ++p) {
    c.maps[p].resize(35);
    for (unsigned y = 0; y < 5; ++y)
      for (unsigned x = 0; x < 7; ++x)
        c.maps[p][y * 7 + x] = float(p % 2 ? y : x);
  }
  return c;
}
static CaptureLease capture(uint64_t id) {
  auto s = std::make_shared<CaptureFrame>();
  s->source_width = 16;
  s->height = 5;
  s->stride_bytes = 32;
  s->frame_id = id;
  s->camera_timestamp_ns = id;
  s->host_ingress_timestamp_ns = id;
  s->camera_clock_domain = "cloud-camera";
  s->calibration_identity = "cloud-original";
  s->device_identity = "cloud-test";
  s->packed_yuyv.assign(160, 128);
  return s;
}
template <class T>
static void upload(std::shared_ptr<DaxaFoundation> ctx, daxa::BufferId dst,
                   std::span<const T> data) {
  auto &d = ctx->device;
  daxa::BufferInfo info{.size = data.size_bytes(), .name = "cloud test upload"};
  auto memory = d.create_memory(
      {.requirements = ctx->coherent_readback_requirements(info),
       .flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
  auto src = d.create_buffer_from_memory_block(
      {.buffer_info = info, .memory_block = memory});
  std::memcpy(d.buffer_host_address(src).value(), data.data(),
              data.size_bytes());
  auto rec = d.create_command_recorder({.name = "cloud test input override"});
  rec.pipeline_barrier({.src_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE |
                                      daxa::AccessConsts::HOST_WRITE,
                        .dst_access = daxa::AccessConsts::TRANSFER_READ |
                                      daxa::AccessConsts::TRANSFER_WRITE});
  rec.copy_buffer_to_buffer(
      {.src_buffer = src, .dst_buffer = dst, .size = data.size_bytes()});
  rec.pipeline_barrier({.src_access = daxa::AccessConsts::TRANSFER_WRITE,
                        .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ});
  auto commands = rec.complete_current_commands();
  auto ticket = d.submit_commands({.command_lists = {&commands, 1}});
  d.wait_on_submit({.queue = daxa::QUEUE_MAIN,
                    .queue_submit_index = ticket,
                    .timeout = 30'000'000'000ull});
  d.destroy_buffer(src);
  d.collect_garbage();
}
static void gpu() {
  auto ctx = std::make_shared<DaxaFoundation>(true);
  auto source = capture((uint64_t(1) << 60) + 3);
  DaxaDepthConfig cfg{StereoConfig{7, 5, 3, 4}, 1.f, 2, 32};
  DaxaDepthPipeline depth(grid(), cfg, ctx, source);
  auto input = depth.enqueue(source);
  input.wait();
  auto observation = input.observations();
  std::vector<float> z(70);
  std::vector<uint32_t> validity(70, 1), rgba(35);
  for (unsigned i = 0; i < 70; ++i)
    z[i] = 1.f + float(i % 7) * .25f;
  for (unsigned i = 0; i < 35; ++i)
    rgba[i] =
        0xff000000u | ((i + 1) & 255u) | ((17 + i) << 8) | ((201 - i) << 16);
  z[1] = NAN;
  z[2] = 0;
  z[3] = -1;
  z[4] = INFINITY;
  validity[5] = 0;
  upload<float>(ctx, input.buffer(DaxaDepthPlane::Depth), z);
  upload<uint32_t>(ctx, input.buffer(DaxaDepthPlane::MetricValidity), validity);
  upload<uint32_t>(
      ctx, observation.image(DaxaObservationImage::RectifiedLeft).buffer(),
      rgba);
  DaxaPointCloud resident(ctx, observation, {2, 0}),
      exported(ctx, observation, {2, 3}), full_export(ctx, observation, {2, 1});
  check(resident.requested_export_bytes() == 0,
        "resident-only path allocated export storage");
  auto a = resident.enqueue(observation), b = exported.enqueue(observation);
  a.wait();
  b.wait();
  check(a.width() == 7 && a.height() == 5 && a.point_bytes() == 560,
        "full-resolution resident geometry lost");
  check(!a.has_export() && b.has_export(), "requested export switch");
  rejects([&] { a.export_bytes(); });
  check(b.export_width() == 3 && b.export_height() == 2 &&
            b.export_bytes().size() == 96,
        "sampled export layout");
  auto full = full_export.enqueue(observation);
  full.wait();
  auto rays =
      point_cloud_rays(observation.calibration()->geometry->rectified.eyes[0]);
  PointCloudLayout layout(7, 5, 3);
  auto verify = [&](const DaxaPointCloud::Frame &frame,
                    const PointCloudLayout &samples) {
    auto exported_bytes = frame.export_bytes();
    for (size_t i = 0; i < size_t(samples.width) * samples.height; ++i) {
      XyzRgb p;
      std::memcpy(&p, exported_bytes.data() + i * sizeof(p), sizeof(p));
      auto index = samples.source_index(i);
      bool valid =
          validity[index] == 1 && std::isfinite(z[index]) && z[index] > 0;
      if (valid)
        check(p.x == rays[index][0] * z[index] &&
                  p.y == rays[index][1] * z[index] && p.z == z[index],
              "axial ray geometry differs");
      else
        check(std::isnan(p.x) && std::isnan(p.y) && std::isnan(p.z),
              "invalid point not marked NaN");
      check(p.rgb == point_cloud_rgb(rgba[index]),
            "RGB byte order differs from CPU/PCL");
    }
    auto timing = frame.timings();
    check(std::isfinite(timing.generation_gpu_ms) &&
              timing.generation_gpu_ms >= 0 &&
              std::isfinite(timing.export_gpu_ms) && timing.export_gpu_ms >= 0,
          "cloud GPU timestamps invalid");
  };
  verify(b, layout);
  verify(full, PointCloudLayout(7, 5, 1));
  auto no_export = full_export.enqueue(observation, false);
  no_export.wait();
  check(!no_export.has_export(), "unrequested per-frame cloud readback");
  rejects([&] { no_export.export_bytes(); });
  no_export = {};
  full_export.collect();
  {
    auto &device = ctx->device;
    TemporaryBuffer scratch{
        ctx, device.create_buffer(
                 {.size = 560, .name = "pending export test copy"})};
    auto block = device.create_command_recorder(
        {.name = "gate cloud export completion"});
    block.pipeline_barrier(
        {.src_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
         .dst_access = daxa::AccessConsts::TRANSFER_READ});
    block.copy_buffer_to_buffer(
        {.src_buffer = a.buffer(), .dst_buffer = scratch.id, .size = 560});
    auto commands = block.complete_current_commands();
    auto gate = device.create_timeline_semaphore(
        {.initial_value = 0, .name = "pending cloud export gate"});
    GateRelease unblock{gate};
    std::array waits{std::pair{gate, uint64_t(1)}};
    auto ticket = device.submit_commands(
        {.command_lists = {&commands, 1}, .wait_timeline_semaphores = waits});
    a.retain_until(ticket);
    auto pending = full_export.enqueue(observation);
    check(!pending.ready(),
          "cloud behind blocked main queue unexpectedly ready");
    rejects([&] { pending.export_bytes(); });
    rejects([&] { pending.timings(); });
    unblock.release();
    pending.wait();
    verify(pending, PointCloudLayout(7, 5, 1));
    pending = {};
    full_export.collect();
  }
  check(full_export.memory().export_bytes == 1120 &&
            resident.memory().export_bytes == 0,
        "cloud export accounting");
  rejects([&] { resident.enqueue({}); });
  rejects([&] { DaxaPointCloud bad(ctx, observation, {1, 0}); });
  auto before_invalid = ctx->device.latest_queue_submit_index(daxa::QUEUE_MAIN);
  rejects([&] { DaxaPointCloud bad({}, observation); });
  rejects([&] { DaxaPointCloud bad(ctx, observation, {4, 0}); });
  rejects([&] { DaxaPointCloud bad(ctx, observation, {2, 8193}); });
  DaxaPointCloud::Frame empty;
  check(!empty, "empty cloud frame became valid");
  rejects([&] { empty.buffer(); });
  rejects([&] { empty.export_bytes(); });
  check(ctx->device.latest_queue_submit_index(daxa::QUEUE_MAIN) ==
            before_invalid,
        "invalid cloud request submitted GPU work");
  {
    auto foreign_context = std::make_shared<DaxaFoundation>(true);
    auto foreign_before =
        foreign_context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);
    rejects([&] { DaxaPointCloud bad(foreign_context, observation); });
    check(foreign_context->device.latest_queue_submit_index(daxa::QUEUE_MAIN) ==
              foreign_before,
          "foreign cloud constructor submitted GPU work");
    {
      DaxaDepthPipeline foreign_depth(grid(), cfg, foreign_context,
                                      capture(101));
      auto foreign_frame = foreign_depth.enqueue(capture(101));
      foreign_frame.wait();
      auto before = ctx->device.latest_queue_submit_index(daxa::QUEUE_MAIN);
      rejects([&] { resident.enqueue(foreign_frame.observations()); });
      check(ctx->device.latest_queue_submit_index(daxa::QUEUE_MAIN) == before,
            "foreign cloud enqueue submitted GPU work");
      check(resident.has_capacity(),
            "foreign cloud enqueue consumed a ring slot");
    }
    foreign_context->close();
    check(foreign_context->validation_errors.load() == 0 &&
              foreign_context->validation_warnings.load() == 0,
          "foreign-context rejection failed Vulkan validation");
  }
  DaxaDepthPipeline foreign(grid(true), cfg, ctx, capture(77));
  auto other = foreign.enqueue(capture(77));
  other.wait();
  rejects([&] { resident.enqueue(other.observations()); });
  {
    auto extreme_z = z;
    extreme_z.at(34) = std::numeric_limits<float>::max();
    upload<float>(ctx, other.buffer(DaxaDepthPlane::Depth), extreme_z);
    upload<uint32_t>(ctx, other.buffer(DaxaDepthPlane::MetricValidity),
                     validity);
    DaxaPointCloud extreme(ctx, other.observations(), {2, 1});
    auto frame = extreme.enqueue(other.observations());
    frame.wait();
    XyzRgb point;
    auto bytes = frame.export_bytes();
    std::memcpy(&point, bytes.data() + 34 * sizeof(point), sizeof(point));
    check(std::isnan(point.x) && std::isnan(point.y) && std::isnan(point.z),
          "overflowing XYZ not rejected as NaN");
  }
  other = {};
  foreign.drain();
  DaxaPointCloud::Frame survivor;
  {
    auto owner = std::make_unique<DaxaPointCloud>(ctx, observation,
                                                  DaxaPointCloudConfig{2, 1});
    survivor = owner->enqueue(observation);
  }
  survivor.wait();
  verify(survivor, PointCloudLayout(7, 5, 1));
  survivor = {};

  check(b.observations().source()->frame_id == source->frame_id &&
            b.observations().buffer(DaxaObservationPlane::Provenance) ==
                observation.buffer(DaxaObservationPlane::Provenance),
        "cloud lost source/provenance lease");
  auto retained = b;
  auto second = exported.enqueue(observation);
  rejects([&] { exported.enqueue(observation); });
  second.wait();
  second = {};
  exported.collect();
  second = exported.enqueue(observation);
  second.wait();
  b = {};
  exported.collect();
  rejects([&] { exported.enqueue(observation); });
  retained = {};
  exported.collect();
  auto reused = exported.enqueue(observation);
  reused.wait();
  auto before = ctx->device.latest_queue_submit_index(daxa::QUEUE_MAIN);
  rejects([&] { reused.retain_until(before + 1); });
  auto &device = ctx->device;
  auto downstream =
      device.create_buffer({.size = 560, .name = "cloud downstream test"});
  auto read = device.create_command_recorder({.name = "cloud downstream read"});
  read.pipeline_barrier({.src_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
                         .dst_access = daxa::AccessConsts::TRANSFER_READ});
  read.copy_buffer_to_buffer(
      {.src_buffer = reused.buffer(), .dst_buffer = downstream, .size = 560});
  auto read_commands = read.complete_current_commands();
  auto gate = device.create_timeline_semaphore(
      {.initial_value = 0, .name = "hold cloud downstream test"});
  GateRelease unblock{gate};
  std::array waits{std::pair{gate, uint64_t(1)}};
  auto read_ticket =
      device.submit_commands({.command_lists = {&read_commands, 1},
                              .wait_timeline_semaphores = waits});
  reused.retain_until(read_ticket);
  reused = {};
  exported.collect();
  check(!exported.has_capacity(),
        "released cloud reused before downstream GPU read");
  rejects([&] { exported.enqueue(observation); });
  unblock.release();
  device.wait_on_submit({.queue = daxa::QUEUE_MAIN,
                         .queue_submit_index = read_ticket,
                         .timeout = 30'000'000'000ull});
  check(exported.has_capacity(),
        "completed downstream read did not release cloud slot");
  reused = exported.enqueue(observation);
  reused.wait();
  device.destroy_buffer(downstream);
  device.collect_garbage();
  full = {};
  full_export.drain();
  a = {};
  reused = {};
  second = {};
  observation = {};
  input = {};
  resident.collect();
  exported.collect();
  depth.collect();
  resident.drain();
  exported.drain();
  depth.drain();
  check(ctx->validation_errors.load() == 0 &&
            ctx->validation_warnings.load() == 0,
        "cloud Vulkan validation failed");
  std::cout << "native cloud GPU geometry/export/lease contracts passed\n";
}
int main() {
  try {
    gpu();
    return 0;
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
