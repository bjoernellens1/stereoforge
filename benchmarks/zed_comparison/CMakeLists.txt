find_package(ZED 5.5 REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL 3 REQUIRED COMPONENTS Crypto)
include(FetchContent)
FetchContent_Declare(sf_zed_reference_json
  URL https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp
  URL_HASH SHA256=9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6
  DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(sf_zed_reference_json)
add_executable(export_sdk export_sdk.cpp)
target_compile_features(export_sdk PRIVATE cxx_std_20)
target_include_directories(export_sdk SYSTEM PRIVATE ${ZED_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(export_sdk PRIVATE "${sf_zed_reference_json_SOURCE_DIR}")
target_link_directories(export_sdk PRIVATE "${ZED_LIBRARY_DIR}")
target_link_libraries(export_sdk PRIVATE ${ZED_LIBRARIES} CUDA::cudart OpenSSL::Crypto)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/reference-licenses")
foreach(name nlohmann-json.LICENSE.MIT nlohmann-json.NOTICE openssl.LICENSE.Apache-2.0)
  configure_file("${PROJECT_SOURCE_DIR}/third_party/licenses/${name}"
    "${CMAKE_CURRENT_BINARY_DIR}/reference-licenses/${name}" COPYONLY)
endforeach()
set(sf_zed_export_source "${CMAKE_CURRENT_SOURCE_DIR}/export_sdk.cpp")
file(SHA256 "${sf_zed_export_source}" SF_ZED_EXPORT_SOURCE_SHA256)
set(sf_zed_export_io "${CMAKE_CURRENT_SOURCE_DIR}/export_io.hpp")
file(SHA256 "${sf_zed_export_io}" SF_ZED_EXPORT_IO_SHA256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${sf_zed_export_source}" "${sf_zed_export_io}")
set(SF_ZED_EXPORT_COMMIT "unavailable")
set(SF_ZED_EXPORT_DIRTY "unavailable")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE sf_git_head_result OUTPUT_VARIABLE sf_git_head OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(sf_git_head_result EQUAL 0 AND sf_git_head MATCHES "^[0-9a-fA-F]+$")
    set(SF_ZED_EXPORT_COMMIT "${sf_git_head}")
  endif()
  execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE sf_git_status_result OUTPUT_VARIABLE sf_git_status OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(sf_git_status_result EQUAL 0)
    if(sf_git_status STREQUAL "")
      set(SF_ZED_EXPORT_DIRTY "false")
    else()
      set(SF_ZED_EXPORT_DIRTY "true")
    endif()
  endif()
endif()
target_compile_definitions(export_sdk PRIVATE
  SF_ZED_EXPORT_IO_SHA256="${SF_ZED_EXPORT_IO_SHA256}"
  SF_ZED_EXPORT_SOURCE_SHA256="${SF_ZED_EXPORT_SOURCE_SHA256}"
  SF_ZED_EXPORT_COMMIT="${SF_ZED_EXPORT_COMMIT}"
  SF_ZED_EXPORT_DIRTY="${SF_ZED_EXPORT_DIRTY}")
