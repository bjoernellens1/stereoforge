#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <daxa/c/device.h>
#include <filesystem>
#include <fstream>
#include <stereoforge/daxa_point_cloud.hpp>
#include <stereoforge/daxa_preprocess.hpp>
#include <thread>
#include <utility>
namespace stereoforge {
namespace {
void require(bool ok, const char *why) {
  if (!ok)
    throw std::invalid_argument(why);
}
struct Parameters {
  uint64_t depth = 0, validity = 0, rgba = 0, rays = 0, points = 0,
           export_points = 0;
  uint32_t width = 0, height = 0, export_width = 0, export_height = 0,
           stride = 0, pad = 0;
};
static_assert(sizeof(Parameters) == 72);
std::vector<uint32_t> shader(const char *entry) {
  std::ifstream f(std::filesystem::path(SF_DAXA_CLOUD_SPIRV_DIRECTORY) /
                      (std::string("daxa-cloud-") + entry + ".spv"),
                  std::ios::binary | std::ios::ate);
  require(bool(f), "missing offline cloud shader");
  auto size = f.tellg();
  require(size > 0 && size % 4 == 0, "invalid cloud SPIRV");
  std::vector<uint32_t> words(size_t(size) / 4);
  f.seekg(0);
  f.read(reinterpret_cast<char *>(words.data()), size);
  require(bool(f), "cloud SPIRV read failed");
  return words;
}
} // namespace
struct DaxaPointCloud::Slot {
  std::shared_ptr<DaxaFoundation> context;
  std::thread::id thread = std::this_thread::get_id();
  daxa::BufferId points{}, parameters{}, output{};
  daxa::MemoryBlock parameter_memory, output_memory;
  std::byte *parameter_ptr = nullptr;
  const std::byte *output_ptr = nullptr;
  daxa::TimelineQueryPool queries;
  bool exported = false;
  DaxaObservationFrame producer;
  uint64_t generation = 0, submit = 0, read = 0;
  std::chrono::steady_clock::time_point started;
  ~Slot() noexcept {
    if (thread != std::this_thread::get_id())
      std::terminate();
    try {
      auto &d = context->device;
      auto ticket = std::max(submit, read);
      if (!detail::completed_submission(ticket,
                                        d.oldest_pending_submit_index()))
        d.wait_on_submit({.queue = daxa::QUEUE_MAIN,
                          .queue_submit_index = ticket,
                          .timeout = 30'000'000'000ull});
      for (auto b : {points, parameters, output})
        if (!b.is_empty())
          d.destroy_buffer(b);
      d.collect_garbage();
    } catch (...) {
      std::terminate();
    }
  }
};
struct DaxaPointCloud::Impl {
  std::shared_ptr<DaxaFoundation> context;
  std::shared_ptr<const DaxaRectification> maps;
  DaxaPointCloudConfig config;
  PointCloudLayout full, exported;
  DaxaPointCloudMemory accounting;
  daxa::BufferId rays{};
  uint32_t timestamp_bits = 0;
  float timestamp_period = 0;
  std::array<daxa::ComputePipeline, 2> pipelines;
  std::array<std::shared_ptr<Slot>, 3> slots{};
  std::shared_ptr<Slot> setup_upload;
  std::thread::id thread = std::this_thread::get_id();
  bool poisoned = false;
  Impl(std::shared_ptr<DaxaFoundation> c,
       std::shared_ptr<const DaxaRectification> m, DaxaPointCloudConfig cfg)
      : context(std::move(c)), maps(std::move(m)), config(cfg),
        full(maps->width, maps->height, 1),
        exported(maps->width, maps->height,
                 cfg.export_stride ? cfg.export_stride : 1) {}
  void check() const {
    require(thread == std::this_thread::get_id(),
            "cloud requires producer submission thread");
    require(!poisoned, "cloud submission failed; reconstruct cloud owner");
  }
  ~Impl() noexcept {
    if (thread != std::this_thread::get_id())
      std::terminate();
    try {
      setup_upload.reset();
      for (auto const &slot : slots)
        if (slot) {
          auto t = std::max(slot->submit, slot->read);
          if (t && !detail::completed_submission(
                       t, context->device.oldest_pending_submit_index()))
            context->device.wait_on_submit({.queue = daxa::QUEUE_MAIN,
                                            .queue_submit_index = t,
                                            .timeout = 30'000'000'000ull});
        }
      if (!rays.is_empty())
        context->device.destroy_buffer(rays);
      context->device.collect_garbage();
    } catch (...) {
      std::terminate();
    }
  }
  void validate(const DaxaObservationFrame &input) const {
    require(bool(input) && input.context_id() == reinterpret_cast<uintptr_t>(
                                                     context->device.get()),
            "foreign/empty cloud observation context");
    require(input.calibration() == maps && input.width() == full.width &&
                input.height() == full.height,
            "foreign cloud calibration/grid");
    require(input.size_bytes(DaxaObservationPlane::Depth) >=
                    size_t(full.width) * full.height * 4 &&
                input.size_bytes(DaxaObservationPlane::Validity) >=
                    size_t(full.width) * full.height * 4,
            "short cloud metric planes");
    auto rgb = input.image(DaxaObservationImage::RectifiedLeft);
    require(rgb.width() == full.width && rgb.height() == full.height &&
                rgb.size_bytes() >= size_t(full.width) * full.height * 4,
            "cloud RGB registration mismatch");
  }
  std::shared_ptr<Slot> &available() {
    auto boundary = context->device.oldest_pending_submit_index();
    for (unsigned i = 0; i < config.slots; ++i)
      if (detail::available_slot(slots[i].use_count(),
                                 std::max(slots[i]->submit, slots[i]->read),
                                 boundary))
        return slots[i];
    throw std::runtime_error(
        "cloud ring exhausted: retained lease or GPU read");
  }
};
DaxaPointCloud::DaxaPointCloud(std::shared_ptr<DaxaFoundation> context,
                               const DaxaObservationFrame &prototype,
                               DaxaPointCloudConfig config) {
  require(context && context->device.get() && bool(prototype),
          "cloud needs resident context/prototype");
  require(config.slots >= 2 && config.slots <= 3,
          "cloud requires2..3bounded slots");
  require(prototype.context_id() ==
              reinterpret_cast<uintptr_t>(context->device.get()),
          "foreign cloud prototype context");
  auto maps = prototype.calibration();
  require(maps && maps->geometry.has_value(),
          "cloud needs native rectified metric calibration");
  maps->metric_calibration();
  auto e = std::make_shared<Impl>(std::move(context), std::move(maps), config);
  e->validate(prototype);
  auto &d = e->context->device;
  auto physical = daxa_dvc_get_vk_physical_device(d.get());
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical, &properties);
  require(properties.limits.maxComputeWorkGroupInvocations >= 64 &&
              properties.limits.maxComputeWorkGroupSize[0] >= 64 &&
              (uint64_t(e->full.width) * e->full.height + 63) / 64 <=
                  properties.limits.maxComputeWorkGroupCount[0] &&
              e->full.bytes <= properties.limits.maxStorageBufferRange,
          "cloud dispatch/storage exceeds device limits");
  e->timestamp_period = properties.limits.timestampPeriod;
  uint32_t family = 0, count = 0;
  require(daxa_dvc_get_vk_queue(d.get(), {DAXA_QUEUE_TYPE_MAIN, 0}, nullptr,
                                &family) == DAXA_RESULT_SUCCESS,
          "cloud needs shared main queue");
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
  require(family < count && families[family].timestampValidBits,
          "cloud timestamps unavailable");
  e->timestamp_bits = families[family].timestampValidBits;
  for (unsigned i = 0; i < 2; ++i) {
    auto code = shader(i == 0 ? "generate" : "sample_points");
    e->pipelines[i] = d.create_compute_pipeline(
        {.shader_info = {.byte_code = code.data(),
                         .byte_code_size = uint32_t(code.size())},
         .push_constant_size = 8,
         .name = "resident XYZRGB"});
  }
  auto rays = point_cloud_rays(e->maps->geometry->rectified.eyes[0]);
  require(rays.size() == size_t(e->full.width) * e->full.height,
          "cloud ray grid mismatch");
  size_t ray_bytes = rays.size() * sizeof(rays[0]);
  e->rays = d.create_buffer(
      {.size = ray_bytes, .name = "immutable rectified cloud rays"});
  // One calibration upload at setup, never per-frame geometry readback/upload.
  e->setup_upload = std::make_shared<Slot>();
  auto &upload = *e->setup_upload;
  upload.context = e->context;
  daxa::BufferInfo staging_info{.size = ray_bytes,
                                .name = "cloud calibration upload"};
  upload.parameter_memory = d.create_memory(
      {.requirements = e->context->coherent_readback_requirements(staging_info),
       .flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
  upload.parameters = d.create_buffer_from_memory_block(
      {.buffer_info = staging_info, .memory_block = upload.parameter_memory});
  std::memcpy(d.buffer_host_address(upload.parameters).value(), rays.data(),
              ray_bytes);
  auto before_upload = d.latest_queue_submit_index(daxa::QUEUE_MAIN);
  try {
    auto rec = d.create_command_recorder({.name = "cloud calibration upload"});
    rec.pipeline_barrier({.src_access = daxa::AccessConsts::HOST_WRITE,
                          .dst_access = daxa::AccessConsts::TRANSFER_READ});
    rec.copy_buffer_to_buffer({.src_buffer = upload.parameters,
                               .dst_buffer = e->rays,
                               .size = ray_bytes});
    rec.pipeline_barrier(
        {.src_access = daxa::AccessConsts::TRANSFER_WRITE,
         .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ});
    auto commands = rec.complete_current_commands();
    upload.submit = d.submit_commands({.command_lists = {&commands, 1}});
    require(upload.submit > before_upload,
            "cloud calibration submit did not advance queue");
    d.wait_on_submit({.queue = daxa::QUEUE_MAIN,
                      .queue_submit_index = upload.submit,
                      .timeout = 30'000'000'000ull});
  } catch (...) {
    try {
      upload.submit = std::max(upload.submit,
                               d.latest_queue_submit_index(daxa::QUEUE_MAIN));
    } catch (...) {
      std::terminate();
    }
    throw;
  }
  e->setup_upload.reset();
  e->accounting.resident_bytes =
      ray_bytes + config.slots * (e->full.bytes + sizeof(Parameters));
  daxa::MemoryRequirements export_requirements{};
  daxa::BufferInfo export_info{.size = e->exported.bytes,
                               .name = "requested cloud host export"};
  if (config.export_stride) {
    export_requirements = d.buffer_memory_requirements(export_info);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    bool found = false;
    for (unsigned pass = 0; pass < 2 && !found; ++pass)
      for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        auto f = memory.memoryTypes[i].propertyFlags;
        constexpr auto coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((export_requirements.memory_type_bits & (1u << i)) &&
            (f & coherent) == coherent &&
            (pass || (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT))) {
          export_requirements.memory_type_bits = 1u << i;
          e->accounting.export_memory_type = i;
          e->accounting.export_memory_flags = f;
          found = true;
          break;
        }
      }
    require(found, "no coherent cloud export memory; noncoherent invalidation "
                   "unqualified");
    e->accounting.export_bytes = config.slots * e->exported.bytes;
  }
  for (unsigned i = 0; i < config.slots; ++i) {
    auto s = std::make_shared<Slot>();
    s->context = e->context;
    s->points = d.create_buffer(
        {.size = e->full.bytes, .name = "full-resolution resident XYZRGB"});
    daxa::BufferInfo parameter_info{.size = sizeof(Parameters),
                                    .name = "cloud parameters"};
    s->parameter_memory = d.create_memory(
        {.requirements =
             e->context->coherent_readback_requirements(parameter_info),
         .flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
    s->parameters = d.create_buffer_from_memory_block(
        {.buffer_info = parameter_info, .memory_block = s->parameter_memory});
    s->parameter_ptr = reinterpret_cast<std::byte *>(
        d.buffer_host_address(s->parameters).value());
    if (config.export_stride) {
      s->output_memory =
          d.create_memory({.requirements = export_requirements,
                           .flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
      s->output = d.create_buffer_from_memory_block(
          {.buffer_info = export_info, .memory_block = s->output_memory});
      s->output_ptr = reinterpret_cast<const std::byte *>(
          d.buffer_host_address(s->output).value());
      require(s->output_ptr, "cloud export is not mapped");
    }
    s->queries = d.create_timeline_query_pool(
        {.query_count = 4, .name = "cloud generation/export timing"});
    e->slots[i] = std::move(s);
  }
  impl_ = std::move(e);
}
std::string DaxaPointCloud::shader_directory() {
  return SF_DAXA_CLOUD_SPIRV_DIRECTORY;
}
DaxaPointCloud::~DaxaPointCloud() noexcept {
  if (impl_ && impl_->thread != std::this_thread::get_id())
    std::terminate();
}
DaxaPointCloud::Frame DaxaPointCloud::enqueue(const DaxaObservationFrame &input,
                                              bool request_export) {
  auto &e = *impl_;
  e.check();
  e.validate(input);
  auto slot = e.available();
  auto &s = *slot;
  require(s.generation != UINT64_MAX, "cloud generation exhausted");
  auto &d = e.context->device;
  s.exported = request_export && e.config.export_stride != 0;
  Parameters p;
  p.depth = d.device_address(input.buffer(DaxaObservationPlane::Depth)).value();
  p.validity =
      d.device_address(input.buffer(DaxaObservationPlane::Validity)).value();
  p.rgba = d.device_address(
                input.image(DaxaObservationImage::RectifiedLeft).buffer())
               .value();
  p.rays = d.device_address(e.rays).value();
  p.points = d.device_address(s.points).value();
  if (s.exported)
    p.export_points = d.device_address(s.output).value();
  p.width = e.full.width;
  p.height = e.full.height;
  p.export_width = e.exported.width;
  p.export_height = e.exported.height;
  p.stride = e.config.export_stride;
  std::memcpy(s.parameter_ptr, &p, sizeof(p));
  auto before = d.latest_queue_submit_index(daxa::QUEUE_MAIN);
  s.producer = input;
  ++s.generation;
  s.submit = s.read = 0;
  s.started = std::chrono::steady_clock::now();
  try {
    auto rec = d.create_command_recorder(
        {.name = "resident cloud generation and requested export"});
    rec.pipeline_barrier(
        {.src_access = daxa::AccessConsts::HOST_WRITE |
                       daxa::AccessConsts::COMPUTE_SHADER_WRITE |
                       daxa::AccessConsts::TRANSFER_WRITE,
         .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ |
                       daxa::AccessConsts::COMPUTE_SHADER_WRITE});
    rec.reset_timestamps(
        {.query_pool = s.queries, .start_index = 0, .count = 4});
    rec.write_timestamp(
        {.query_pool = s.queries,
         .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER,
         .query_index = 0});
    rec.set_pipeline(e.pipelines[0]);
    rec.push_constant(d.device_address(s.parameters).value());
    rec.dispatch(
        {uint32_t((size_t(e.full.width) * e.full.height + 63) / 64), 1, 1});
    rec.write_timestamp(
        {.query_pool = s.queries,
         .pipeline_stage = daxa::PipelineStageFlagBits::COMPUTE_SHADER,
         .query_index = 1});
    rec.pipeline_barrier(
        {.src_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
         .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ |
                       daxa::AccessConsts::TRANSFER_READ |
                       daxa::AccessConsts::TRANSFER_WRITE});
    rec.write_timestamp(
        {.query_pool = s.queries,
         .pipeline_stage = daxa::PipelineStageFlagBits::ALL_COMMANDS,
         .query_index = 2});
    if (s.exported && e.config.export_stride == 1)
      rec.copy_buffer_to_buffer({.src_buffer = s.points,
                                 .dst_buffer = s.output,
                                 .size = e.full.bytes});
    else if (s.exported) {
      rec.set_pipeline(e.pipelines[1]);
      rec.push_constant(d.device_address(s.parameters).value());
      rec.dispatch(
          {uint32_t((size_t(e.exported.width) * e.exported.height + 63) / 64),
           1, 1});
    }
    rec.pipeline_barrier(
        {.src_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE |
                       daxa::AccessConsts::TRANSFER_WRITE,
         .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ |
                       daxa::AccessConsts::TRANSFER_READ |
                       daxa::AccessConsts::HOST_READ});
    rec.write_timestamp(
        {.query_pool = s.queries,
         .pipeline_stage = daxa::PipelineStageFlagBits::ALL_COMMANDS,
         .query_index = 3});
    auto commands = rec.complete_current_commands();
    s.submit = d.submit_commands({.command_lists = {&commands, 1}});
    require(s.submit > before, "cloud submit did not advance queue");
  } catch (...) {
    e.poisoned = true;
    try {
      s.submit =
          std::max(s.submit, d.latest_queue_submit_index(daxa::QUEUE_MAIN));
      if (s.submit <= before)
        s.submit = 0;
    } catch (...) {
      std::terminate();
    }
    throw;
  }
  return Frame(impl_, std::move(slot));
}
bool DaxaPointCloud::has_capacity() const {
  impl_->check();
  auto boundary = impl_->context->device.oldest_pending_submit_index();
  for (unsigned i = 0; i < impl_->config.slots; ++i) {
    auto const &s = impl_->slots[i];
    if (detail::available_slot(s.use_count(), std::max(s->submit, s->read),
                               boundary))
      return true;
  }
  return false;
}
size_t DaxaPointCloud::collect() {
  impl_->check();
  auto boundary = impl_->context->device.oldest_pending_submit_index();
  size_t n = 0;
  for (unsigned i = 0; i < impl_->config.slots; ++i) {
    auto &s = impl_->slots[i];
    if (s.use_count() == 1 && s->producer &&
        detail::completed_submission(std::max(s->submit, s->read), boundary)) {
      s->producer = {};
      ++n;
    }
  }
  impl_->context->device.collect_garbage();
  return n;
}
void DaxaPointCloud::drain() {
  impl_->check();
  for (unsigned i = 0; i < impl_->config.slots; ++i) {
    auto const &s = impl_->slots[i];
    auto ticket = std::max(s->submit, s->read);
    if (ticket)
      impl_->context->device.wait_on_submit({.queue = daxa::QUEUE_MAIN,
                                             .queue_submit_index = ticket,
                                             .timeout = 30'000'000'000ull});
  }
  collect();
}
size_t DaxaPointCloud::requested_export_bytes() const {
  impl_->check();
  return impl_->accounting.export_bytes;
}
DaxaPointCloudMemory DaxaPointCloud::memory() const {
  impl_->check();
  return impl_->accounting;
}
DaxaPointCloud::Frame::Frame(std::shared_ptr<Impl> owner,
                             std::shared_ptr<Slot> slot)
    : owner_(std::move(owner)), slot_(std::move(slot)),
      generation_(slot_->generation) {}
DaxaPointCloud::Frame::~Frame() noexcept {
  if (owner_ && owner_->thread != std::this_thread::get_id())
    std::terminate();
}
DaxaPointCloud::Frame &DaxaPointCloud::Frame::operator=(Frame other) noexcept {
  owner_.swap(other.owner_);
  slot_.swap(other.slot_);
  std::swap(generation_, other.generation_);
  return *this;
}
void DaxaPointCloud::Frame::validate() const {
  require(owner_ && slot_ && generation_ == slot_->generation,
          "empty/stale cloud frame");
  owner_->check();
}
DaxaPointCloud::Frame::operator bool() const {
  return owner_ && slot_ && generation_ == slot_->generation;
}
daxa::BufferId DaxaPointCloud::Frame::buffer() const {
  validate();
  return slot_->points;
}
size_t DaxaPointCloud::Frame::point_bytes() const {
  validate();
  return owner_->full.bytes;
}
uint32_t DaxaPointCloud::Frame::width() const {
  validate();
  return owner_->full.width;
}
uint32_t DaxaPointCloud::Frame::height() const {
  validate();
  return owner_->full.height;
}
DaxaObservationFrame DaxaPointCloud::Frame::observations() const {
  validate();
  return slot_->producer;
}
uint64_t DaxaPointCloud::Frame::submit_ticket() const {
  validate();
  return slot_->submit;
}
bool DaxaPointCloud::Frame::ready() const {
  validate();
  return detail::completed_submission(
      slot_->submit, owner_->context->device.oldest_pending_submit_index());
}
void DaxaPointCloud::Frame::wait() const {
  validate();
  owner_->context->device.wait_on_submit({.queue = daxa::QUEUE_MAIN,
                                          .queue_submit_index = slot_->submit,
                                          .timeout = 30'000'000'000ull});
}
bool DaxaPointCloud::Frame::has_export() const {
  validate();
  return slot_->exported;
}
uint32_t DaxaPointCloud::Frame::export_width() const {
  require(has_export(), "cloud host export not requested");
  return owner_->exported.width;
}
uint32_t DaxaPointCloud::Frame::export_height() const {
  require(has_export(), "cloud host export not requested");
  return owner_->exported.height;
}
std::span<const std::byte> DaxaPointCloud::Frame::export_bytes() const {
  require(has_export(), "cloud host export not requested");
  require(ready(), "cloud export completion required before CPU access");
  return {slot_->output_ptr, owner_->exported.bytes};
}
void DaxaPointCloud::Frame::retain_until(uint64_t ticket) {
  validate();
  require(ticket > slot_->submit &&
              ticket <= owner_->context->device.latest_queue_submit_index(
                            daxa::QUEUE_MAIN),
          "invalid downstream cloud main-queue ticket");
  slot_->read = std::max(slot_->read, ticket);
}
DaxaPointCloudTiming DaxaPointCloud::Frame::timings() const {
  validate();
  require(ready(), "cloud completion required before timing");
  std::array<uint64_t, 8> values{};
  require(daxa_timeline_query_pool_query_results(
              slot_->queries.get(), 0, 4, values.data()) == DAXA_RESULT_SUCCESS,
          "cloud timestamp query failed");
  for (unsigned i = 0; i < 4; ++i)
    require(values[2 * i + 1], "cloud timestamp unavailable");
  long double wrap =
      std::ldexp(static_cast<long double>(owner_->timestamp_period),
                 int(owner_->timestamp_bits));
  require(std::chrono::duration<long double, std::nano>(
              std::chrono::steady_clock::now() - slot_->started)
                  .count() < wrap,
          "cloud timestamp wrap ambiguous");
  auto mask = owner_->timestamp_bits == 64
                  ? UINT64_MAX
                  : (uint64_t(1) << owner_->timestamp_bits) - 1;
  auto ms = [&](unsigned a, unsigned b) {
    return double((values[b * 2] - values[a * 2]) & mask) *
           double(owner_->timestamp_period) / 1e6;
  };
  return {ms(0, 1), slot_->exported ? ms(2, 3) : 0};
}
} // namespace stereoforge
