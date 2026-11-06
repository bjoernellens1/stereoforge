include_guard(GLOBAL)
set(sf_cloud_spirv)
foreach(entry generate sample_points)
  set(output "${CMAKE_CURRENT_BINARY_DIR}/daxa-cloud-${entry}.spv")
  add_custom_command(OUTPUT "${output}"
    COMMAND "${SF_DAXA_SLANGC}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/daxa/point_cloud.slang"
      -target spirv -profile spirv_1_5 -restrictive-capability-check -matrix-layout-row-major -fvk-use-scalar-layout -O3
      -entry "${entry}" -stage compute -o "${output}"
    DEPENDS shaders/daxa/point_cloud.slang VERBATIM)
  list(APPEND sf_cloud_spirv "${output}")
endforeach()
add_custom_target(daxa_point_cloud_shaders DEPENDS ${sf_cloud_spirv})
add_library(stereoforge_daxa_point_cloud src/daxa_point_cloud.cpp)
target_link_libraries(stereoforge_daxa_point_cloud PUBLIC stereoforge_daxa_depth PRIVATE GPUOpen::VulkanMemoryAllocator)
target_compile_definitions(stereoforge_daxa_point_cloud PRIVATE SF_DAXA_CLOUD_SPIRV_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(stereoforge_daxa_point_cloud daxa_point_cloud_shaders)
if(BUILD_TESTING)
  add_executable(daxa_point_cloud_contracts tests/daxa_point_cloud.cpp)
  target_link_libraries(daxa_point_cloud_contracts PRIVATE stereoforge_daxa_point_cloud)
  add_test(NAME daxa_point_cloud_gpu_contracts COMMAND daxa_point_cloud_contracts)
  set_tests_properties(daxa_point_cloud_gpu_contracts PROPERTIES LABELS "vulkan;hardware;daxa")
endif()
