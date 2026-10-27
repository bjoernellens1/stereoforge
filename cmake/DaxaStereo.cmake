include_guard(GLOBAL)
if(NOT TARGET stereoforge_daxa_foundation)
  message(FATAL_ERROR "DaxaStereo requires the qualified pinned Daxa foundation")
endif()
set(sf_daxa_stereo_shared shaders/daxa/stereo_parameters.slang shaders/ops/sgm_specialized.slang)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS shaders/daxa/stereo.slang shaders/ops/sgm.slang shaders/daxa/stereo_layered.slang ${sf_daxa_stereo_shared})
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/shaders/daxa/stereo_parameters.slang" SF_DAXA_STEREO_PARAMETERS_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ops/sgm_specialized.slang" SF_DAXA_STEREO_SPECIALIZED_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/shaders/daxa/stereo_layered.slang" SF_DAXA_STEREO_LAYERED_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/shaders/daxa/stereo.slang" SF_DAXA_STEREO_WRAPPER_SHA256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ops/sgm.slang" SF_DAXA_STEREO_OPS_SHA256)
set(sf_daxa_stereo_entries census cost aggregate_baseline aggregate_wave select_output depth_convert inspect clear_packed aggregate_packed8 aggregate_packed16 select_packed aggregate_specialized8 aggregate_specialized16 select_specialized8 select_specialized16 aggregate_d1288 aggregate_d12816 select_quad8 select_quad16 aggregate_shuffle8 aggregate_shuffle16 aggregate_right_specialized8 aggregate_right_specialized16 select_right_specialized8 select_right_specialized16 aggregate_layered8 aggregate_layered16 select_layered8 select_layered16)
set(sf_daxa_stereo_spirv)
foreach(entry IN LISTS sf_daxa_stereo_entries)
  set(output "${CMAKE_CURRENT_BINARY_DIR}/daxa-stereo-${entry}.spv")
  set(source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/daxa/stereo.slang")
  if(entry MATCHES "_layered")
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/daxa/stereo_layered.slang")
  endif()
  add_custom_command(OUTPUT "${output}"
    COMMAND "${SF_DAXA_SLANGC}" "${source}"
      -target spirv -profile spirv_1_5 -capability spvGroupNonUniform+spvGroupNonUniformShuffle+spvGroupNonUniformArithmetic -restrictive-capability-check -matrix-layout-row-major -fvk-use-scalar-layout -O3
      -entry "${entry}" -stage compute -o "${output}"
    DEPENDS "${source}" shaders/ops/sgm.slang ${sf_daxa_stereo_shared} VERBATIM)
  list(APPEND sf_daxa_stereo_spirv "${output}")
endforeach()
add_custom_target(daxa_stereo_shaders DEPENDS ${sf_daxa_stereo_spirv})
add_library(stereoforge_daxa_frame src/daxa_frame.cpp)
target_link_libraries(stereoforge_daxa_frame PUBLIC stereoforge_daxa_foundation)
add_library(stereoforge_daxa_stereo src/daxa_stereo.cpp)
target_link_libraries(stereoforge_daxa_stereo PUBLIC stereoforge_daxa_foundation stereoforge_daxa_frame PRIVATE GPUOpen::VulkanMemoryAllocator)
target_compile_definitions(stereoforge_daxa_stereo PRIVATE
  SF_DAXA_STEREO_SPIRV_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}"
  SF_DAXA_STEREO_WRAPPER_SHA256="${SF_DAXA_STEREO_WRAPPER_SHA256}"
  SF_DAXA_STEREO_OPS_SHA256="${SF_DAXA_STEREO_OPS_SHA256}"
  SF_DAXA_STEREO_PARAMETERS_SHA256="${SF_DAXA_STEREO_PARAMETERS_SHA256}"
  SF_DAXA_STEREO_SPECIALIZED_SHA256="${SF_DAXA_STEREO_SPECIALIZED_SHA256}"
  SF_DAXA_STEREO_LAYERED_SHA256="${SF_DAXA_STEREO_LAYERED_SHA256}")
add_dependencies(stereoforge_daxa_stereo daxa_stereo_shaders)
add_executable(daxa_stereo_benchmark apps/daxa_stereo_benchmark.cpp)
target_link_libraries(daxa_stereo_benchmark PRIVATE stereoforge_daxa_stereo)
if(BUILD_TESTING)
  add_executable(daxa_stereo_contracts tests/daxa_stereo.cpp)
  target_link_libraries(daxa_stereo_contracts PRIVATE stereoforge_daxa_stereo)
  add_test(NAME daxa_stereo_cpu_contracts COMMAND daxa_stereo_contracts --cpu-only)
  foreach(variant baseline wave32 wave64 packed32 packed64 packed-specialized32 packed-specialized64 packed-d12832 packed-d12864 packed-quad32 packed-quad64 packed-shuffle32 packed-shuffle64 packed-specialized-right32 packed-specialized-right64)
    add_test(NAME daxa_stereo_${variant}_contracts COMMAND daxa_stereo_contracts --gpu "${variant}")
    set_tests_properties(daxa_stereo_${variant}_contracts PROPERTIES LABELS "vulkan;hardware;daxa")
  endforeach()
endif()

option(STEREOFORGE_DAXA_PREPROCESS "Build staged native GPU preprocessing and luma bridge" OFF)
if(STEREOFORGE_DAXA_PREPROCESS)
  include("${CMAKE_CURRENT_LIST_DIR}/DaxaPreprocess.cmake")
  if(BUILD_TESTING)
    target_link_libraries(daxa_stereo_contracts PRIVATE stereoforge_daxa_preprocess)
    target_compile_definitions(daxa_stereo_contracts PRIVATE SF_DAXA_STEREO_TEST_PREPROCESS=1)
  endif()
endif()
