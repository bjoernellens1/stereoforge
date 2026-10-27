# Private upstream build: no global compiler options, no SDK/CUDA dependency.
include(FetchContent)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir OUTPUT_VARIABLE SF_PYBIND_DIR OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE SF_PYBIND_RESULT)
if(NOT SF_PYBIND_RESULT EQUAL 0)
  message(FATAL_ERROR "Capture needs pybind11==2.13.6 in the selected Python environment")
endif()
find_package(pybind11 2.13.6 CONFIG REQUIRED PATHS "${SF_PYBIND_DIR}" NO_DEFAULT_PATH)
FetchContent_Declare(sf_zed
  URL https://codeload.github.com/stereolabs/zed-open-capture/tar.gz/93739154903a66c112ee52c3c6fbfd83b7218617
  URL_HASH SHA256=738a097d809248c191434d082389c25b9fec6e2ef55a29d0484b871987f329da
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR stereoforge_no_upstream_build)
FetchContent_Declare(sf_hid_headers
  URL https://codeload.github.com/libusb/hidapi/tar.gz/d6b2a974608dec3b76fb1e36c189f22b9cf3650c
  URL_HASH SHA256=a93a9bfb599831b82c6cef50f42359e0f5f31abfc8f6954862c328a464cbc4d0
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR stereoforge_no_upstream_build)
# Fetch sources without evaluating upstream CMake (which rewrites global flags).
FetchContent_MakeAvailable(sf_zed sf_hid_headers)
find_library(SF_HIDAPI_LIBRARY NAMES hidapi-hidraw hidapi-libusb libhidapi-hidraw.so.0 libhidapi-libusb.so.0 REQUIRED)
find_package(Threads REQUIRED)

# Upstream's getters return mutable producer references after unlocking. Add
# snapshot methods to build-local copies, preserving upstream files and license.
# These wait via the upstream APIs then copy under the actual producer mutex.
set(SF_ZED_ADAPTER_DIR "${CMAKE_CURRENT_BINARY_DIR}/zed-adapter")
file(MAKE_DIRECTORY "${SF_ZED_ADAPTER_DIR}")
foreach(name defines.hpp videocapture_def.hpp sensorcapture_def.hpp)
  configure_file("${sf_zed_SOURCE_DIR}/include/${name}" "${SF_ZED_ADAPTER_DIR}/${name}" COPYONLY)
endforeach()
file(READ "${sf_zed_SOURCE_DIR}/include/videocapture.hpp" SF_VIDEO_HEADER)
string(REPLACE "    void grabThreadFunc();" "public:\n    // Modified by StereoForge: generated producer-locked snapshot declaration.\n    Frame snapshotFrame(std::vector<uint8_t>& bytes, uint64_t timeout_msec);\nprivate:\n    void grabThreadFunc();" SF_VIDEO_HEADER "${SF_VIDEO_HEADER}")
if(NOT SF_VIDEO_HEADER MATCHES "snapshotFrame")
  message(FATAL_ERROR "Pinned ZED VideoCapture adapter insertion point missing")
endif()
file(WRITE "${SF_ZED_ADAPTER_DIR}/videocapture.hpp" "${SF_VIDEO_HEADER}")
file(READ "${sf_zed_SOURCE_DIR}/include/sensorcapture.hpp" SF_SENSOR_HEADER)
string(REPLACE "    void grabThreadFunc();" "public:\n    // Modified by StereoForge: generated producer-locked snapshot declaration.\n    data::Imu snapshotIMU(uint64_t timeout_usec);\nprivate:\n    void grabThreadFunc();" SF_SENSOR_HEADER "${SF_SENSOR_HEADER}")
# Fail rather than silently compile an unsafe binding if upstream structure drifts.
if(NOT SF_SENSOR_HEADER MATCHES "snapshotIMU")
  message(FATAL_ERROR "Pinned ZED SensorCapture adapter insertion point missing")
endif()
file(WRITE "${SF_ZED_ADAPTER_DIR}/sensorcapture.hpp" "${SF_SENSOR_HEADER}")
configure_file("${sf_zed_SOURCE_DIR}/src/videocapture.cpp" "${SF_ZED_ADAPTER_DIR}/videocapture.cpp" COPYONLY)
configure_file("${sf_zed_SOURCE_DIR}/src/sensorcapture.cpp" "${SF_ZED_ADAPTER_DIR}/sensorcapture.cpp" COPYONLY)
file(WRITE "${SF_ZED_ADAPTER_DIR}/snapshots.cpp" [=[
#include "videocapture.hpp"
#include "sensorcapture.hpp"
sl_oc::video::Frame sl_oc::video::VideoCapture::snapshotFrame(std::vector<uint8_t>& bytes,uint64_t timeout) {
    getLastFrame(timeout);
    std::lock_guard<std::mutex> lock(mBufMutex);
    Frame copy=mLastFrame;
    size_t size=static_cast<size_t>(copy.width)*copy.height*copy.channels;
    if(copy.data && size) bytes.assign(copy.data,copy.data+size);
    copy.data=bytes.empty()?nullptr:bytes.data();
    return copy;
}
sl_oc::sensors::data::Imu sl_oc::sensors::SensorCapture::snapshotIMU(uint64_t timeout) {
    getLastIMUData(timeout);
    std::lock_guard<std::mutex> lock(mIMUMutex);
    return mLastIMUData;
}
]=])
add_library(stereoforge_zed_private STATIC
  "${SF_ZED_ADAPTER_DIR}/videocapture.cpp" "${SF_ZED_ADAPTER_DIR}/sensorcapture.cpp" "${SF_ZED_ADAPTER_DIR}/snapshots.cpp")
set_target_properties(stereoforge_zed_private PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(stereoforge_zed_private PUBLIC VIDEO_MOD_AVAILABLE SENSORS_MOD_AVAILABLE)
target_include_directories(stereoforge_zed_private PUBLIC "${SF_ZED_ADAPTER_DIR}" "${sf_hid_headers_SOURCE_DIR}/hidapi")
target_link_libraries(stereoforge_zed_private PUBLIC "${SF_HIDAPI_LIBRARY}" Threads::Threads)
# The pinned video module uses Linux V4L2 directly; it does not include/use libusb.
pybind11_add_module(_zed_capture "${CMAKE_CURRENT_LIST_DIR}/../src/zed_capture.cpp")
target_link_libraries(_zed_capture PRIVATE stereoforge_zed_private)
target_compile_features(_zed_capture PRIVATE cxx_std_20)
set_target_properties(_zed_capture PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native")

# Preserve full license notices beside the optional native capture binary.
set(SF_CAPTURE_LICENSE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/licenses")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/native/licenses")
foreach(name zed-open-capture-MIT.txt hidapi-BSD-3-Clause.txt pybind11-BSD-3-Clause.txt)
  configure_file("${SF_CAPTURE_LICENSE_SOURCE_DIR}/${name}" "${CMAKE_BINARY_DIR}/native/licenses/${name}" COPYONLY)
endforeach()
