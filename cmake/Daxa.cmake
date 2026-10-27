# Independent Daxa owner; offline Linux Slang -> SPIR-V, no RHI/SGL/Python.
include(FetchContent)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  message(FATAL_ERROR "Pinned Daxa foundation requires Linux x86_64")
endif()
set(STEREOFORGE_DAXA_REVISION "00761e80c1dd6d1972d5f4569ce49620a7b4d384")
set(STEREOFORGE_DAXA_SLANG_VERSION "2026.12")
set(STEREOFORGE_DAXA_SLANG_SDK_ARCHIVE "" CACHE FILEPATH "Cached official Slang Linux SDK archive")
set(sf_daxa_slang_url "https://github.com/shader-slang/slang/releases/download/v2026.12/slang-2026.12-linux-x86_64.tar.gz")
if(STEREOFORGE_DAXA_SLANG_SDK_ARCHIVE)
  set(sf_daxa_slang_url "${STEREOFORGE_DAXA_SLANG_SDK_ARCHIVE}")
endif()
FetchContent_Declare(sf_daxa_slang URL "${sf_daxa_slang_url}"
  URL_HASH SHA256=2409ecc7b6710783e05fd59792558c275c637e3e3c37fb5054fcfc858e570d82
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE SOURCE_SUBDIR no-cmake-project)
FetchContent_MakeAvailable(sf_daxa_slang)
set(SF_DAXA_SLANGC "${sf_daxa_slang_SOURCE_DIR}/bin/slangc")
execute_process(COMMAND "${SF_DAXA_SLANGC}" -version
  OUTPUT_VARIABLE sf_daxa_slang_actual_version ERROR_VARIABLE sf_daxa_slang_actual_version
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT sf_daxa_slang_actual_version STREQUAL STEREOFORGE_DAXA_SLANG_VERSION)
  message(FATAL_ERROR "Expected Linux Slang ${STEREOFORGE_DAXA_SLANG_VERSION}, got ${sf_daxa_slang_actual_version}")
endif()
file(SHA256 "${SF_DAXA_SLANGC}" SF_DAXA_COMPILER_SHA256)
message(STATUS "Daxa compiler ${SF_DAXA_SLANGC}, version ${sf_daxa_slang_actual_version}, SHA256 ${SF_DAXA_COMPILER_SHA256}")
# First declaration wins, overriding the upstream mutable VMA tag.
FetchContent_Declare(vma URL "https://codeload.github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/tar.gz/009ecd192c1289c7529bff248a16cfe896254816"
  URL_HASH SHA256=5ed5125086a92666f1698df907a29f54f11197c382996094b556a1b22186ecaf
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
foreach(feature TESTS TOOLS STATIC_ANALYSIS UTILS_IMGUI UTILS_FSR2 UTILS_FSR3 UTILS_PIPELINE_MANAGER_GLSLANG UTILS_PIPELINE_MANAGER_SLANG UTILS_PIPELINE_MANAGER_SPIRV_VALIDATION)
  set(DAXA_ENABLE_${feature} OFF CACHE BOOL "Minimal pinned Daxa foundation" FORCE)
endforeach()
set(DAXA_ENABLE_UTILS_TASK_GRAPH ON CACHE BOOL "Daxa owns scheduling" FORCE)
FetchContent_Declare(sf_daxa URL "https://codeload.github.com/Ipotrick/Daxa/tar.gz/${STEREOFORGE_DAXA_REVISION}"
  URL_HASH SHA256=0005d3601b4b3b01f7bdc0a1cacf957aa56dfdb70686fa84043ab79528672679
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(sf_daxa)
# Fail closed on unexpected pinned-source or patch drift; accept exact already-applied patches.
find_program(SF_DAXA_PATCH patch REQUIRED)
function(sf_daxa_apply_patch filename)
  set(patch_file "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/${filename}")
  execute_process(COMMAND "${SF_DAXA_PATCH}" -p1 --batch --fuzz=0 --dry-run --forward --input "${patch_file}"
    WORKING_DIRECTORY "${sf_daxa_SOURCE_DIR}" RESULT_VARIABLE apply_result OUTPUT_QUIET ERROR_QUIET)
  if(apply_result EQUAL 0)
    execute_process(COMMAND "${SF_DAXA_PATCH}" -p1 --batch --fuzz=0 --forward --input "${patch_file}"
      WORKING_DIRECTORY "${sf_daxa_SOURCE_DIR}" COMMAND_ERROR_IS_FATAL ANY)
  else()
    execute_process(COMMAND "${SF_DAXA_PATCH}" -p1 --batch --fuzz=0 --dry-run --reverse --input "${patch_file}"
      WORKING_DIRECTORY "${sf_daxa_SOURCE_DIR}" RESULT_VARIABLE reverse_result OUTPUT_QUIET ERROR_QUIET)
    if(NOT reverse_result EQUAL 0)
      message(FATAL_ERROR "Pinned Daxa patch does not apply cleanly: ${filename}")
    endif()
  endif()
endfunction()
sf_daxa_apply_patch(daxa-3.6-linux-headless.patch)
sf_daxa_apply_patch(daxa-3.6-external-resource-views.patch)
sf_daxa_apply_patch(daxa-3.6-first-access-group-batch.patch)
set(SF_DAXA_SOURCE_DIR "${sf_daxa_SOURCE_DIR}")
# Retain upstream runtime assertions in the optimized qualification build.
target_compile_definitions(daxa PUBLIC DAXA_VALIDATION=1)
find_package(Vulkan REQUIRED)
set(SF_DAXA_SPIRV "${CMAKE_CURRENT_BINARY_DIR}/daxa-foundation.spv")
add_custom_command(OUTPUT "${SF_DAXA_SPIRV}"
  COMMAND "${SF_DAXA_SLANGC}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/daxa/foundation.slang"
    -target spirv -profile spirv_1_5 -entry main -stage compute -o "${SF_DAXA_SPIRV}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/daxa/foundation.slang"
  VERBATIM)
add_custom_target(daxa_foundation_shader DEPENDS "${SF_DAXA_SPIRV}")
