# Matching source-built RHI and official Slang SDK. No Python/SGL discovery.
# Source revision is also the RHI revision used by the research wheel, avoiding
# an ABI guess while moving the production build off that wheel entirely.
include(FetchContent)
set(STEREOFORGE_RHI_REVISION "1a9768741246b1f356116d5dfa305933034ce049")
set(STEREOFORGE_SLANG_VERSION "2026.12")
set(STEREOFORGE_RHI_SOURCE_ARCHIVE "" CACHE FILEPATH "Optional cached exact pinned RHI source archive")
set(STEREOFORGE_SLANG_SDK_ARCHIVE "" CACHE FILEPATH "Optional cached exact pinned official Slang SDK archive")
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  message(FATAL_ERROR "The pinned standalone SDK currently qualifies Linux x86_64 only")
endif()
set(sf_rhi_url "https://codeload.github.com/shader-slang/slang-rhi/tar.gz/${STEREOFORGE_RHI_REVISION}")
set(sf_slang_url "https://github.com/shader-slang/slang/releases/download/v${STEREOFORGE_SLANG_VERSION}/slang-${STEREOFORGE_SLANG_VERSION}-linux-x86_64.tar.gz")
if(STEREOFORGE_RHI_SOURCE_ARCHIVE)
  set(sf_rhi_url "${STEREOFORGE_RHI_SOURCE_ARCHIVE}")
endif()
if(STEREOFORGE_SLANG_SDK_ARCHIVE)
  set(sf_slang_url "${STEREOFORGE_SLANG_SDK_ARCHIVE}")
endif()
FetchContent_Declare(sf_slang_sdk URL "${sf_slang_url}"
  URL_HASH SHA256=2409ecc7b6710783e05fd59792558c275c637e3e3c37fb5054fcfc858e570d82
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE SOURCE_SUBDIR no-cmake-project)
FetchContent_MakeAvailable(sf_slang_sdk)
set(SLANG_RHI_FETCH_SLANG OFF CACHE BOOL "Use the pinned complete SDK above" FORCE)
set(SLANG_RHI_SLANG_INCLUDE_DIR "${sf_slang_sdk_SOURCE_DIR}/include" CACHE STRING "Pinned SDK headers" FORCE)
set(SLANG_RHI_SLANG_BINARY_DIR "${sf_slang_sdk_SOURCE_DIR}" CACHE STRING "Pinned SDK binary root" FORCE)
set(SLANG_RHI_BUILD_SHARED OFF CACHE BOOL "Static production RHI" FORCE)
set(SLANG_RHI_BUILD_TESTS OFF CACHE BOOL "No upstream tests" FORCE)
set(SLANG_RHI_BUILD_TESTS_WITH_GLFW OFF CACHE BOOL "No GLFW" FORCE)
set(SLANG_RHI_BUILD_EXAMPLES OFF CACHE BOOL "No upstream examples" FORCE)
set(SLANG_RHI_INSTALL OFF CACHE BOOL "No system installation" FORCE)
foreach(backend CPU CUDA OPTIX METAL D3D11 D3D12 AGILITY_SDK NVAPI AFTERMATH WGPU)
  set(SLANG_RHI_ENABLE_${backend} OFF CACHE BOOL "Vulkan-only production runtime" FORCE)
endforeach()
set(SLANG_RHI_ENABLE_VULKAN ON CACHE BOOL "Vulkan production backend" FORCE)
set(SLANG_RHI_FETCH_DXC OFF CACHE BOOL "No D3D compiler" FORCE)
set(SLANG_RHI_OPTIX_VERSIONS "" CACHE STRING "No OptiX headers" FORCE)
FetchContent_Declare(sf_slang_rhi URL "${sf_rhi_url}"
  URL_HASH SHA256=ab0daed27b68bf3878358d059694a35e36bf99ed38ac7b9b1926cefc84481396
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(sf_slang_rhi)
add_library(stereoforge_rhi_provider INTERFACE)
target_link_libraries(stereoforge_rhi_provider INTERFACE slang-rhi slang-rhi-vulkan-headers)
set(SF_RHI_RUNTIME_LIBRARY_DIR "${sf_slang_sdk_SOURCE_DIR}/lib")
message(STATUS "StereoForge standalone RHI ${STEREOFORGE_RHI_REVISION}, Slang ${STEREOFORGE_SLANG_VERSION}; Vulkan only")
# GCC 16 diagnoses the upstream COM multiple-inheritance virtual delete thunks
# under -O3 as free-nonheap-object. Keep that warning visible without allowing
# upstream's blanket -Werror to block this pinned dependency on the host compiler.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
  target_compile_options(slang-rhi PRIVATE -Wno-error=free-nonheap-object)
endif()
