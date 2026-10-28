#include <stereoforge/ros_gpu_frame.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

using Adapter = rclcpp::TypeAdapter<stereoforge::RosGpuFrame, std_msgs::msg::Header>;
static_assert(Adapter::is_specialized::value);
static_assert(std::is_copy_constructible_v<stereoforge::RosGpuFrame>);
static_assert(std::is_same_v<Adapter::ros_message_type, std_msgs::msg::Header>);

int main(int argc, char** argv) {
    stereoforge::RosGpuFrame empty;
    std_msgs::msg::Header header;
    bool rejected = false;
    try { Adapter::convert_to_ros_message(empty, header); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return EXIT_FAILURE;
    rejected = false;
    try { Adapter::convert_to_custom(header, empty); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) return EXIT_FAILURE;
    header.frame_id = "left_optical";
    header.stamp.sec = 1;
    rejected = false;
    try { stereoforge::RosGpuFrame invalid({}, header); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return EXIT_FAILURE;

    // Instantiate the actual Jazzy publisher and subscriber with intra-process
    // enabled. No synthetic object can stand in for a genuine GPU lease, so
    // this checks registration only, not delivery or device lifetime.
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    auto node = std::make_shared<rclcpp::Node>("gpu_frame_contract_test", options);
    using Adapted = rclcpp::TypeAdapter<stereoforge::RosGpuFrame, std_msgs::msg::Header>;
    auto pub = node->create_publisher<Adapted>("gpu_frame_contract", rclcpp::QoS(1));
    auto sub = node->create_subscription<Adapted>("gpu_frame_contract", rclcpp::QoS(1),
        [](std::shared_ptr<const stereoforge::RosGpuFrame>) {});
    if (!pub || !sub) return EXIT_FAILURE;
    sub.reset(); pub.reset(); node.reset();
    rclcpp::shutdown();
}
