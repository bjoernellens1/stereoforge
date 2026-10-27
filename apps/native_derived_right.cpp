// Diagnostic-only independent dense RHI oracle; no production packed code.
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>
#include <stereoforge/contracts.hpp>
#include <stereoforge/native_device.hpp>
#include <unordered_map>
#include <vector>
using rhi::ComPtr;
static void check(SlangResult r, const char *s) {
  if (SLANG_FAILED(r))
    throw std::runtime_error(s);
}
static std::vector<uint32_t> read(const std::string &p, size_t n) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f || f.tellg() != std::streamoff(n * 4))
    throw std::runtime_error("input byte extent: " + p);
  f.seekg(0);
  std::vector<uint32_t> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * 4);
  return v;
}
int main(int argc, char **argv) {
  try {
    std::unordered_map<std::string, std::string> o;
    bool validation = false;
    for (int a = 1; a < argc; a++) {
      std::string k = argv[a];
      if (k == "--validation") {
        validation = true;
        continue;
      }
      if (++a == argc)
        throw std::invalid_argument("missing argument");
      if (!o.emplace(k, argv[a]).second)
        throw std::invalid_argument("duplicate argument");
    }
    for (auto &[k, v] : o)
      if (k != "--width" && k != "--height" && k != "--disparities" &&
          k != "--paths" && k != "--p1" && k != "--p2" && k != "--subpixel" &&
          k != "--left" && k != "--right" && k != "--left-valid" &&
          k != "--right-valid" && k != "--output" && k != "--fx" &&
          k != "--baseline-m" && k != "--principal-offset" &&
          k != "--shader-directory" && k != "--reference-shader")
        throw std::invalid_argument("unknown argument " + k);
    auto u = [&](const char *k) {
      size_t end = 0;
      auto s = o.at(k);
      auto v = std::stoul(s, &end);
      if (end != s.size() || s.front() == '-' || v > UINT32_MAX)
        throw std::invalid_argument(k);
      return uint32_t(v);
    };
    stereoforge::StereoConfig c{
        u("--width"), u("--height"), u("--disparities"),  u("--paths"),
        u("--p1"),    u("--p2"),     u("--subpixel") != 0};
    c.validate();
    auto f = [&](const char *k) {
      size_t e = 0;
      float v = std::stof(o.at(k), &e);
      if (e != o.at(k).size())
        throw std::invalid_argument(k);
      return v;
    };
    stereoforge::MetricCalibration m{f("--fx"), f("--baseline-m"),
                                     f("--principal-offset")};
    m.validate();
    const size_t n = size_t(c.width) * c.height;
    auto l = read(o.at("--left"), n), r = read(o.at("--right"), n),
         lv = read(o.at("--left-valid"), n),
         rv = read(o.at("--right-valid"), n);
    for (size_t i = 0; i < n; i++)
      if (lv[i] > 1 || rv[i] > 1)
        throw std::invalid_argument("nonbinary census support");
    stereoforge::NativeDevice context(validation);
    auto device = context.rhi_device();
    auto queue = context.rhi_queue();
    std::unordered_map<std::string, ComPtr<rhi::IBuffer>> buffers;
    for (const char *name :
         {"left", "right", "left_desc", "right_desc", "left_valid",
          "right_valid", "census_left_valid", "census_right_valid", "costs",
          "sums", "disparity", "confidence", "validity", "metric_depth_output",
          "metric_validity_output"}) {
      rhi::BufferDesc d{};
      d.size = n * 4 *
               ((std::string(name) == "costs" || std::string(name) == "sums")
                    ? c.disparities
                    : 1);
      d.elementSize = 4;
      d.usage =
          rhi::BufferUsage::ShaderResource | rhi::BufferUsage::UnorderedAccess |
          rhi::BufferUsage::CopySource | rhi::BufferUsage::CopyDestination;
      d.defaultState = rhi::ResourceState::UnorderedAccess;
      check(device->createBuffer(d, nullptr, buffers[name].writeRef()),
            "buffer");
    }
    std::unordered_map<std::string, ComPtr<rhi::IComputePipeline>> pipelines;
    auto compile = [&](std::string path, std::vector<std::string> names) {
      std::ifstream f(path);
      if (!f)
        throw std::runtime_error("shader missing");
      std::string s((std::istreambuf_iterator<char>(f)), {});
      ComPtr<slang::IBlob> diag;
      ComPtr<slang::IModule> mod(
          device->getSlangSession()->loadModuleFromSourceString(
              path.c_str(), path.c_str(), s.c_str(), diag.writeRef()));
      if (!mod)
        throw std::runtime_error(
            diag ? static_cast<const char *>(diag->getBufferPointer())
                 : "module");
      for (auto &name : names) {
        ComPtr<slang::IEntryPoint> e;
        check(mod->findEntryPointByName(name.c_str(), e.writeRef()), "entry");
        slang::IComponentType *entries[] = {e};
        rhi::ShaderProgramDesc pd{};
        pd.slangGlobalScope = mod;
        pd.slangEntryPoints = entries;
        pd.slangEntryPointCount = 1;
        ComPtr<rhi::IShaderProgram> program;
        auto result = device->createShaderProgram(pd, program.writeRef(),
                                                  diag.writeRef());
        if (SLANG_FAILED(result))
          throw std::runtime_error(
              diag ? static_cast<const char *>(diag->getBufferPointer())
                   : "compile");
        rhi::ComputePipelineDesc d{};
        d.program = program;
        check(device->createComputePipeline(d, pipelines[name].writeRef()),
              "pipeline");
      }
    };
    compile(o.at("--shader-directory") + "/sgm.slang",
            {"census", "cost", "aggregate", "select", "depth_convert"});
    compile(o.at("--reference-shader"), {"derived_right"});
    auto encoder = queue->createCommandEncoder();
    for (auto input : {std::pair{"left", &l}, std::pair{"right", &r},
                       std::pair{"census_left_valid", &lv},
                       std::pair{"census_right_valid", &rv}})
      check(encoder->uploadBufferData(buffers.at(input.first), 0, n * 4,
                                      input.second->data()),
            "upload");
    encoder->globalBarrier();
    auto dispatch = [&](const std::string &stage, uint32_t groups,
                        bool right = false, int dx = 0, int dy = 0,
                        uint32_t lines = 0) {
      auto pass = encoder->beginComputePass();
      rhi::ShaderCursor root(pass->bindPipeline(pipelines.at(stage)));
      auto bind = [&](const char *param, const char *name) {
        auto field = root[param];
        if (field.isValid())
          check(field.setBinding(rhi::Binding(buffers.at(name))), param);
      };
      for (auto &[name, b] : buffers)
        bind(name.c_str(), name.c_str());
      bind("image", right ? "right" : "left");
      bind("descriptor", right ? "right_desc" : "left_desc");
      bind("descriptor_valid", right ? "right_valid" : "left_valid");
      if (stage == "derived_right") {
        bind("left_valid", "census_left_valid");
        bind("right_valid", "census_right_valid");
      }
      bind("lower", "left_valid");
      bind("upper", "right_valid");
      auto scalar = [&](const char *name, auto v) {
        auto field = root[name];
        if (field.isValid())
          check(field.setData(v), name);
      };
      scalar("width", c.width);
      scalar("height", c.height);
      scalar("disparity_count", c.disparities);
      scalar("path_count", c.paths);
      scalar("p1", c.p1);
      scalar("p2", c.p2);
      scalar("bounded", 0u);
      scalar("subpixel", uint32_t(c.subpixel));
      scalar("dx", dx);
      scalar("dy", dy);
      scalar("line_count", lines);
      scalar("metric_fx", m.fx);
      scalar("metric_baseline", m.baseline_m);
      scalar("metric_principal_offset", m.cx_offset);
      pass->dispatchCompute(groups, 1, 1);
      pass->end();
      encoder->globalBarrier();
    };
    dispatch("census", (n + 63) / 64);
    dispatch("census", (n + 63) / 64, true);
    dispatch("cost", (n * c.disparities + 63) / 64);
    int dirs[8][2] = {{1, 0}, {-1, 0},  {0, 1},  {0, -1},
                      {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
    for (uint32_t p = 0; p < c.paths; p++) {
      int dx = dirs[p][0], dy = dirs[p][1];
      auto lines =
          dx == 0 ? c.width : (dy == 0 ? c.height : c.width + c.height - 1);
      dispatch("aggregate", lines, false, dx, dy, lines);
    }
    dispatch("select", (n + 63) / 64);
    dispatch("depth_convert", (n + 63) / 64);
    auto submit = [&]() {
      ComPtr<rhi::ICommandBuffer> cmd;
      check(encoder->finish(cmd.writeRef()), "finish");
      rhi::ICommandBuffer *cmds[] = {cmd};
      rhi::SubmitDesc s{};
      s.commandBuffers = cmds;
      s.commandBufferCount = 1;
      check(queue->submit(s), "submit");
      check(queue->waitOnHost(), "wait");
    };
    auto dump = [&](const std::string &eye) {
      for (auto pair :
           {std::pair{"disparity", "disparity.f32"},
            std::pair{"confidence", "confidence.f32"},
            std::pair{"validity", "validity.u32"},
            std::pair{"metric_depth_output", "depth.f32"},
            std::pair{"metric_validity_output", "depth_validity.u32"}}) {
        std::vector<uint32_t> v(n);
        check(device->readBuffer(buffers.at(pair.first), 0, n * 4, v.data()),
              "read");
        std::ofstream out(o.at("--output") + "." + eye + "." + pair.second,
                          std::ios::binary);
        out.write(reinterpret_cast<const char *>(v.data()), n * 4);
        if (!out)
          throw std::runtime_error("dump");
      }
    };
    submit();
    dump("left");
    encoder = queue->createCommandEncoder();
    dispatch("derived_right", (n + 63) / 64);
    dispatch("depth_convert", (n + 63) / 64);
    submit();
    dump("right");
    std::cout << "{\"schema\":\"stereoforge.rhi_derived_right.v1\",\"adapter\":"
              << std::quoted(context.name())
              << ",\"vulkan_validation\":" << (validation ? "true" : "false")
              << ",\"width\":" << c.width << ",\"height\":" << c.height
              << ",\"disparities\":" << c.disparities
              << ",\"paths\":" << c.paths << ",\"metric_depth\":true}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "native_derived_right: " << e.what() << '\n';
    return 1;
  }
}
