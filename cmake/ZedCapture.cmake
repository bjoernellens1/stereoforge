include_guard(GLOBAL)
include(FetchContent)
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
set(SF_FRAME_TIMESTAMP "    uint64_t timestamp = 0;         //!< Timestamp in nanoseconds")
string(FIND "${SF_VIDEO_HEADER}" "${SF_FRAME_TIMESTAMP}" SF_FRAME_TIMESTAMP_POS)
if(SF_FRAME_TIMESTAMP_POS EQUAL -1)
  message(FATAL_ERROR "Pinned ZED Frame timing insertion point missing")
endif()
string(REPLACE "${SF_FRAME_TIMESTAMP}" "${SF_FRAME_TIMESTAMP}
    // StereoForge: unqualified raw driver evidence copied with the pixels.
    bool driver_timing_available = false;
    uint64_t raw_timestamp_ns = 0;
    uint32_t driver_sequence = 0, driver_flags = 0;
    uint64_t dequeue_begin_monotonic_ns = 0, dequeue_end_monotonic_ns = 0;
    uint64_t sdk_wall_anchor_ns = 0, sdk_first_timestamp_ns = 0;" SF_VIDEO_HEADER "${SF_VIDEO_HEADER}")
string(REPLACE "    void grabThreadFunc();" "public:\n    // Modified by StereoForge: generated producer-locked snapshot declaration.\n    Frame snapshotFrame(std::vector<uint8_t>& bytes, uint64_t timeout_msec, uint64_t previous_timestamp=0);\nprivate:\n    void grabThreadFunc();" SF_VIDEO_HEADER "${SF_VIDEO_HEADER}")
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
file(READ "${sf_zed_SOURCE_DIR}/src/videocapture.cpp" SF_VIDEO_SOURCE)
set(SF_DEQUEUE "        int ret = ioctl(mFileDesc, VIDIOC_DQBUF, &buf);")
set(SF_TIMESTAMP_ASSIGN "                mLastFrame.timestamp = mStartTs + rel_ts;")
foreach(anchor SF_DEQUEUE SF_TIMESTAMP_ASSIGN)
  string(FIND "${SF_VIDEO_SOURCE}" "${${anchor}}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "Pinned ZED producer timing insertion point missing: ${anchor}")
  endif()
endforeach()
string(REPLACE "${SF_DEQUEUE}" "        // StereoForge: bracket this exact dequeue in host monotonic time.
        const auto sf_dequeue_begin = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
${SF_DEQUEUE}
        const auto sf_dequeue_end = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();" SF_VIDEO_SOURCE "${SF_VIDEO_SOURCE}")
string(REPLACE "${SF_TIMESTAMP_ASSIGN}" "${SF_TIMESTAMP_ASSIGN}
                // Preserve returned metadata under the same producer lock as pixels.
                mLastFrame.raw_timestamp_ns = ts_uvc * 1000;
                mLastFrame.driver_sequence = buf.sequence;
                mLastFrame.driver_flags = buf.flags;
                mLastFrame.dequeue_begin_monotonic_ns = sf_dequeue_begin;
                mLastFrame.dequeue_end_monotonic_ns = sf_dequeue_end;
                mLastFrame.sdk_wall_anchor_ns = mStartTs;
                mLastFrame.sdk_first_timestamp_ns = mInitTs * 1000;
                mLastFrame.driver_timing_available = true;" SF_VIDEO_SOURCE "${SF_VIDEO_SOURCE}")
file(WRITE "${SF_ZED_ADAPTER_DIR}/videocapture.cpp" "${SF_VIDEO_SOURCE}")
configure_file("${sf_zed_SOURCE_DIR}/src/sensorcapture.cpp" "${SF_ZED_ADAPTER_DIR}/sensorcapture.cpp" COPYONLY)
file(WRITE "${SF_ZED_ADAPTER_DIR}/snapshots.cpp" [=[
#include "videocapture.hpp"
#include "sensorcapture.hpp"
sl_oc::video::Frame sl_oc::video::VideoCapture::snapshotFrame(std::vector<uint8_t>& bytes,uint64_t timeout,uint64_t previous_timestamp) {
    getLastFrame(timeout);
    std::lock_guard<std::mutex> lock(mBufMutex);
    Frame copy=mLastFrame;
    if(copy.timestamp==previous_timestamp) { copy.data=nullptr; return copy; }
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
