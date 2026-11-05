include_guard(GLOBAL)
set(sf_vo_spirv)
foreach(entry detect track)
  set(output "${CMAKE_CURRENT_BINARY_DIR}/daxa-vo-${entry}.spv")
  add_custom_command(OUTPUT "${output}"
    COMMAND "${SF_DAXA_SLANGC}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/daxa/stereo_vo.slang"
      -target spirv -profile spirv_1_5 -restrictive-capability-check -matrix-layout-row-major -fvk-use-scalar-layout -O3
      -entry "${entry}" -stage compute -o "${output}"
    DEPENDS shaders/daxa/stereo_vo.slang VERBATIM)
  list(APPEND sf_vo_spirv "${output}")
endforeach()
add_custom_target(daxa_stereo_vo_shaders DEPENDS ${sf_vo_spirv})
add_library(stereoforge_daxa_stereo_vo src/daxa_stereo_vo.cpp)
target_link_libraries(stereoforge_daxa_stereo_vo PUBLIC stereoforge_daxa_depth stereoforge_stereo_motion)
target_compile_definitions(stereoforge_daxa_stereo_vo PRIVATE SF_DAXA_VO_SPIRV_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(stereoforge_daxa_stereo_vo daxa_stereo_vo_shaders)
if(BUILD_TESTING)
  add_executable(daxa_stereo_vo_contracts tests/daxa_stereo_vo.cpp)
  target_link_libraries(daxa_stereo_vo_contracts PRIVATE stereoforge_daxa_stereo_vo stereoforge_daxa_rectified_input)
  add_test(NAME daxa_stereo_vo_gpu_contracts COMMAND daxa_stereo_vo_contracts)
  set_tests_properties(daxa_stereo_vo_gpu_contracts PROPERTIES LABELS "vulkan;hardware;daxa")
endif()
