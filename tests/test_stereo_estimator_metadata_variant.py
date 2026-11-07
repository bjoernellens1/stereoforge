from pathlib import Path
import os
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_profile_rejects_experimental_variant_at_runtime(tmp_path):
    compiler = shutil.which('c++')
    json_include = Path(os.environ.get('STEREOFORGE_JSON_INCLUDE', ROOT / 'build/daxa/_deps/sf_native_json-src'))
    if not compiler or not (json_include / 'json.hpp').exists():
        pytest.skip('C++ compiler and configured native JSON dependency required')
    stub = tmp_path / 'stereoforge'
    stub.mkdir()
    (stub / 'daxa_depth.hpp').write_text('''#pragma once
#include <string>
namespace stereoforge {
enum class DaxaSgmVariant { PackedSpecialized, PackedSpecializedOverwrite };
enum class DaxaRightReferenceMode { IndependentSGM, LeftAggregatedSelection };
enum class DaxaDepthStereoSchedule { Sequential, PairedPerPath, LayeredPerPath, SharedCensusSequential, Hierarchical1to8Selective1to4 };
struct StereoConfig { unsigned width=1280,height=720,disparities=64,paths=4,p1=8,p2=32; bool subpixel=true,ortho_hybrid=false; };
struct DaxaDepthConfig { StereoConfig stereo; DaxaDepthStereoSchedule stereo_schedule=DaxaDepthStereoSchedule::Sequential; float lr_tolerance=1; DaxaRightReferenceMode right_reference_mode=DaxaRightReferenceMode::IndependentSGM; DaxaSgmVariant stereo_variant=DaxaSgmVariant::PackedSpecialized; void validate() const {} };
inline std::string daxa_preprocess_file_sha256(const std::filesystem::path&) { return "hash"; }
}
'''.replace('#include <string>', '#include <string>\n#include <filesystem>'))
    src = tmp_path / 'test.cpp'
    src.write_text('''#include "stereo_estimator_metadata.hpp"
#include <set>
using namespace stereoforge;
int main() {
  DaxaDepthConfig config;
  nlohmann::json shaders={{"shader", "hash"}};
  auto result=native_estimator::profile(config,native_estimator::instantaneous(),shaders,shaders,"hash");
  if(result.at("algorithm")!="independent_sgm")return 1;
  auto baseline=result;
  config.stereo.ortho_hybrid=true;
  auto hybrid=native_estimator::profile(config,native_estimator::instantaneous(),shaders,shaders,"hash");
  if(hybrid==baseline)return 3;
  if(hybrid.at("descriptor_mode")!="ortho-hybrid" || baseline.at("descriptor_mode")!="census")return 4;
  config.stereo.ortho_hybrid=false;
  std::set<std::string> identities;
  for(auto schedule:{DaxaDepthStereoSchedule::Sequential,DaxaDepthStereoSchedule::PairedPerPath,
      DaxaDepthStereoSchedule::LayeredPerPath,DaxaDepthStereoSchedule::SharedCensusSequential,
      DaxaDepthStereoSchedule::Hierarchical1to8Selective1to4}) {
    config.stereo_schedule=schedule;
    identities.insert(native_estimator::profile(config,native_estimator::instantaneous(),shaders,shaders,"hash").dump());
  }
  if(identities.size()!=5)return 5;
  config.stereo_schedule=DaxaDepthStereoSchedule::Sequential;
  config.stereo_variant=DaxaSgmVariant::PackedSpecializedOverwrite;
  try { (void)native_estimator::profile(config,native_estimator::instantaneous(),shaders,shaders,"hash"); }
  catch(const std::invalid_argument&) {return 0;}
  return 2;
}
''')
    executable = tmp_path / 'test'
    subprocess.run([compiler, '-std=c++20', '-I'+str(tmp_path), '-I'+str(ROOT/'apps'), '-I'+str(json_include), str(src), '-o', str(executable)], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
