#pragma once

#include <cstdlib>
#include <stdexcept>

namespace stereoforge {
// Call before rclcpp::init. An explicit environment value, including an empty
// value, belongs to the caller and is never silently replaced.
inline void set_default_ros_rmw() {
    const char* const selected = std::getenv("RMW_IMPLEMENTATION");
    if (selected != nullptr && selected[0] == '\0') {
        throw std::runtime_error("RMW_IMPLEMENTATION is empty; unset it for Zenoh or select an installed RMW");
    }
    if (selected == nullptr &&
        ::setenv("RMW_IMPLEMENTATION", "rmw_zenoh_cpp", 0) != 0) {
        throw std::runtime_error("could not set default ROS middleware to rmw_zenoh_cpp");
    }
}
} // namespace stereoforge
