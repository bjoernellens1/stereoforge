#pragma once

#include <stereoforge/daxa_observation.hpp>
#include <stereoforge/daxa_preprocess.hpp>
#include <rclcpp/type_adapter.hpp>
#include <std_msgs/msg/header.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace stereoforge {
// Same-process Daxa observation. Copies retain the producer slot and source.
// Buffers returned by observation() are borrowed: retain this frame for the
// complete consumer GPU submission, and establish producer/consumer ordering.
// This type does not wait, copy pixels, or make buffers accessible on a
// different Vulkan device/process. The ROS wire representation is Header only.
class RosGpuFrame {
    DaxaObservationFrame observation_;
    std_msgs::msg::Header header_;
public:
    RosGpuFrame() = default; // Required by ROS; an empty frame is not publishable.
    RosGpuFrame(DaxaObservationFrame observation, std_msgs::msg::Header header)
        : observation_(std::move(observation)), header_(std::move(header)) {
        validate();
    }
    void validate() const {
        if (!observation_ || header_.frame_id.empty() || header_.stamp.sec < 0 ||
            header_.stamp.nanosec >= 1000000000u ||
            (header_.stamp.sec == 0 && header_.stamp.nanosec == 0))
            throw std::invalid_argument("GPU frame requires a lease, frame ID and valid mapped capture stamp");
        const auto source = observation_.source();
        const auto maps = observation_.calibration();
        if (!source || !maps || source->device_identity.empty() ||
            source->calibration_identity.empty() || source->camera_clock_domain.empty() ||
            source->camera_clock_domain == "unknown" || !source->camera_timestamp_ns ||
            maps->original_identity != source->calibration_identity || maps->rectified_identity.empty() ||
            !observation_.width() || !observation_.height())
            throw std::invalid_argument("GPU frame requires identified capture and calibrated geometry");
    }
    const DaxaObservationFrame& observation() const { validate(); return observation_; }
    const std_msgs::msg::Header& header() const { validate(); return header_; }
};
} // namespace stereoforge

namespace rclcpp {
// This wire format is deliberately metadata-only. A remote subscriber cannot
// reconstruct an observation or treat a header as a GPU frame.
template<> struct TypeAdapter<stereoforge::RosGpuFrame, std_msgs::msg::Header> {
    using is_specialized = std::true_type;
    using custom_type = stereoforge::RosGpuFrame;
    using ros_message_type = std_msgs::msg::Header;
    static void convert_to_ros_message(const custom_type& source, ros_message_type& destination) {
        destination = source.header();
    }
    static void convert_to_custom(const ros_message_type&, custom_type&) {
        throw std::runtime_error("metadata-only ROS Header cannot create a GPU observation");
    }
};
} // namespace rclcpp
