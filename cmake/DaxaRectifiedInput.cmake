include_guard(GLOBAL)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS shaders/daxa/rectified_input.slang)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/shaders/daxa/rectified_input.slang" SF_DAXA_RECTIFIED_INPUT_SHADER_SHA256)
set(sf_rectified_input_spirv)
foreach(entry pack_original pack_rectified coarse_box)
  set(output "${CMAKE_CURRENT_BINARY_DIR}/daxa-rectified-input-${entry}.spv")
  add_custom_command(OUTPUT "${output}"
    COMMAND "${SF_DAXA_SLANGC}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/daxa/rectified_input.slang"
    -target spirv -profile spirv_1_5 -restrictive-capability-check -matrix-layout-row-major -fvk-use-scalar-layout -O3
    -entry "${entry}" -stage compute -o "${output}"
    DEPENDS shaders/daxa/rectified_input.slang VERBATIM)
  list(APPEND sf_rectified_input_spirv "${output}")
endforeach()
add_custom_target(daxa_rectified_input_shaders DEPENDS ${sf_rectified_input_spirv})
add_library(stereoforge_daxa_rectified_input src/daxa_rectified_input.cpp)
target_link_libraries(stereoforge_daxa_rectified_input PUBLIC stereoforge_daxa_preprocess PRIVATE GPUOpen::VulkanMemoryAllocator)
target_compile_definitions(stereoforge_daxa_rectified_input PRIVATE SF_DAXA_RECTIFIED_INPUT_SPIRV_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}" SF_DAXA_RECTIFIED_INPUT_SHADER_SHA256="${SF_DAXA_RECTIFIED_INPUT_SHADER_SHA256}")
add_dependencies(stereoforge_daxa_rectified_input daxa_rectified_input_shaders)
