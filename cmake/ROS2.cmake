# Optional ROS delivery tools. Default builds do not discover ROS or SDKs.
include_guard(GLOBAL)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
include(FetchContent)
if(NOT sf_native_json_SOURCE_DIR)
  FetchContent_Declare(sf_native_json
    URL https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp
    URL_HASH SHA256=9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6
    DOWNLOAD_NO_EXTRACT TRUE)
  FetchContent_MakeAvailable(sf_native_json)
endif()
add_executable(stereoforge_ros2_delivery_probe apps/ros2_delivery_probe.cpp)
target_include_directories(stereoforge_ros2_delivery_probe PRIVATE "${sf_native_json_SOURCE_DIR}")
target_link_libraries(stereoforge_ros2_delivery_probe stereoforge_contracts)
ament_target_dependencies(stereoforge_ros2_delivery_probe rclcpp sensor_msgs)

if(STEREOFORGE_ROS2_PUBLISHER)
  if(NOT STEREOFORGE_NATIVE_CAPTURE OR NOT TARGET stereoforge_daxa_host_export OR NOT TARGET stereoforge_daxa_depth OR NOT TARGET stereoforge_daxa_refinement OR NOT TARGET stereoforge_capture)
    message(FATAL_ERROR "Native ROS publisher requires Daxa stereo, preprocess, depth and native capture")
  endif()
  add_executable(stereoforge_ros2_depth_publisher apps/ros2_depth_publisher.cpp)
  target_include_directories(stereoforge_ros2_depth_publisher PRIVATE "${sf_native_json_SOURCE_DIR}")
  target_link_libraries(stereoforge_ros2_depth_publisher stereoforge_daxa_host_export stereoforge_daxa_depth stereoforge_daxa_refinement stereoforge_capture)
  ament_target_dependencies(stereoforge_ros2_depth_publisher rclcpp sensor_msgs)
endif()

# CPU-only publisher boundary benchmark; it does not initialize capture or Vulkan.
add_executable(stereoforge_ros2_publish_microbench apps/ros2_publish_microbench.cpp)
ament_target_dependencies(stereoforge_ros2_publish_microbench rclcpp sensor_msgs)

if(STEREOFORGE_ROS2_PUBLISHER AND BUILD_TESTING)
  find_package(std_msgs REQUIRED)
  add_executable(stereoforge_ros2_gpu_frame_test tests/test_ros_gpu_frame.cpp)
  target_link_libraries(stereoforge_ros2_gpu_frame_test stereoforge_daxa_depth)
  ament_target_dependencies(stereoforge_ros2_gpu_frame_test rclcpp std_msgs)
  add_test(NAME ros2_gpu_frame_contract COMMAND stereoforge_ros2_gpu_frame_test)
endif()

# Same-process GPU handoff; remote ROS wire messages contain metadata only.
option(STEREOFORGE_ROS2_GPU_COMPOSED "Build the experimental same-process ROS GPU handoff" OFF)
if(STEREOFORGE_ROS2_GPU_COMPOSED)
  if(NOT STEREOFORGE_ROS2_PUBLISHER OR NOT TARGET stereoforge_daxa_depth OR NOT TARGET stereoforge_capture)
    message(FATAL_ERROR "ROS GPU handoff requires STEREOFORGE_ROS2_PUBLISHER and native capture/depth")
  endif()
  find_package(std_msgs REQUIRED)
  add_library(stereoforge_daxa_gpu_consumer STATIC src/daxa_gpu_consumer.cpp)
  target_link_libraries(stereoforge_daxa_gpu_consumer PUBLIC stereoforge_daxa_depth)
  add_executable(stereoforge_ros2_gpu_composed apps/ros2_gpu_composed.cpp)
  target_include_directories(stereoforge_ros2_gpu_composed PRIVATE "${sf_native_json_SOURCE_DIR}")
  target_link_libraries(stereoforge_ros2_gpu_composed stereoforge_daxa_gpu_consumer stereoforge_capture)
  ament_target_dependencies(stereoforge_ros2_gpu_composed rclcpp std_msgs)
  add_executable(daxa_gpu_consumer_check apps/daxa_gpu_consumer_check.cpp)
  target_include_directories(daxa_gpu_consumer_check PRIVATE "${sf_native_json_SOURCE_DIR}")
  target_link_libraries(daxa_gpu_consumer_check PRIVATE stereoforge_daxa_gpu_consumer stereoforge_daxa_view)
endif()

# CPU-only representative adapted intra-process ownership handoff; no GPU or capture.
find_package(std_msgs REQUIRED)
add_executable(stereoforge_ros2_gpu_handoff_microbench apps/ros2_gpu_handoff_microbench.cpp)
ament_target_dependencies(stereoforge_ros2_gpu_handoff_microbench rclcpp std_msgs)
