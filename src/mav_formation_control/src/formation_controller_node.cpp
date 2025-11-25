#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <vector>

struct FormationOffset {
  double x, y, z;
  std::string leader_id;
  double r, g, b; // Color for visualization
};

class FormationControllerNode : public rclcpp::Node {
public:
  FormationControllerNode() : Node("formation_controller_node") {
    // Declare parameters
    this->declare_parameter<int>("num_leaders", 2);
    this->declare_parameter<double>("formation_radius", 2.0);
    this->declare_parameter<std::string>("frame_id", "map");
    this->declare_parameter<double>("publish_rate", 50.0);
    this->declare_parameter<double>("formation_angle_offset", 0.0);

    // Get parameters
    int num_leaders = this->get_parameter("num_leaders").as_int();
    formation_radius_ = this->get_parameter("formation_radius").as_double();
    frame_id_ = this->get_parameter("frame_id").as_string();
    double publish_rate = this->get_parameter("publish_rate").as_double();
    double angle_offset = this->get_parameter("formation_angle_offset").as_double();

    // Setup TF
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // Define formation offsets for leaders
    // Arrange leaders in a circle around virtual leader
    formation_offsets_.clear();
    for (int i = 0; i < num_leaders; i++) {
      FormationOffset offset;
      double angle = (2.0 * M_PI * i / num_leaders) + angle_offset;
      offset.x = formation_radius_ * std::cos(angle);
      offset.y = formation_radius_ * std::sin(angle);
      offset.z = 0.0;
      offset.leader_id = "leader_" + std::to_string(i);
      
      // Assign colors: red, blue, etc.
      if (i == 0) {
        offset.r = 1.0; offset.g = 0.0; offset.b = 0.0; // Red
      } else if (i == 1) {
        offset.r = 0.0; offset.g = 0.0; offset.b = 1.0; // Blue
      } else if (i == 2) {
        offset.r = 1.0; offset.g = 1.0; offset.b = 0.0; // Yellow
      } else {
        offset.r = 1.0; offset.g = 0.0; offset.b = 1.0; // Magenta
      }
      
      formation_offsets_.push_back(offset);
    }

    // Subscriber for virtual leader pose
    virtual_leader_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "virtual_leader/pose", 10,
      std::bind(&FormationControllerNode::virtualLeaderCallback, this, std::placeholders::_1));

    // Publishers for each leader
    for (size_t i = 0; i < formation_offsets_.size(); i++) {
      std::string topic = "leader_" + std::to_string(i) + "/pose";
      leader_pubs_.push_back(
        this->create_publisher<geometry_msgs::msg::PoseStamped>(topic, 10));
    }

    // Marker publisher
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "formation_leaders/markers", 10);

    // Timer
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
      std::bind(&FormationControllerNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Formation Controller Node started with %zu leaders", 
                formation_offsets_.size());
  }

private:
  void virtualLeaderCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    virtual_leader_pose_ = *msg;
    has_virtual_leader_pose_ = true;
  }

  void timerCallback() {
    if (!has_virtual_leader_pose_) {
      return;
    }

    // Calculate and publish poses for each leader
    visualization_msgs::msg::MarkerArray marker_array;

    for (size_t i = 0; i < formation_offsets_.size(); i++) {
      geometry_msgs::msg::PoseStamped leader_pose;
      // Use current time instead of copying old header timestamp
      leader_pose.header.stamp = this->now();
      leader_pose.header.frame_id = virtual_leader_pose_.header.frame_id;
      
      // Get virtual leader orientation
      double yaw = 2.0 * std::atan2(
        virtual_leader_pose_.pose.orientation.z,
        virtual_leader_pose_.pose.orientation.w);

      // Rotate formation offset by virtual leader's orientation
      double cos_yaw = std::cos(yaw);
      double sin_yaw = std::sin(yaw);
      
      double offset_x = formation_offsets_[i].x * cos_yaw - formation_offsets_[i].y * sin_yaw;
      double offset_y = formation_offsets_[i].x * sin_yaw + formation_offsets_[i].y * cos_yaw;

      // Calculate leader position
      leader_pose.pose.position.x = virtual_leader_pose_.pose.position.x + offset_x;
      leader_pose.pose.position.y = virtual_leader_pose_.pose.position.y + offset_y;
      leader_pose.pose.position.z = virtual_leader_pose_.pose.position.z + formation_offsets_[i].z;
      
      // Leader orientation same as virtual leader
      leader_pose.pose.orientation = virtual_leader_pose_.pose.orientation;

      // Publish leader pose
      leader_pubs_[i]->publish(leader_pose);

      // Create marker for visualization
      visualization_msgs::msg::Marker marker;
      marker.header.stamp = this->now();
      marker.header.frame_id = leader_pose.header.frame_id;
      marker.ns = "formation_leaders";
      marker.id = static_cast<int>(i);
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose = leader_pose.pose;
      marker.scale.x = 0.25;
      marker.scale.y = 0.25;
      marker.scale.z = 0.25;
      marker.color.r = formation_offsets_[i].r;
      marker.color.g = formation_offsets_[i].g;
      marker.color.b = formation_offsets_[i].b;
      marker.color.a = 1.0; // Opaque for leaders
      marker.lifetime = rclcpp::Duration(0, 0);
      marker_array.markers.push_back(marker);
    }

    marker_pub_->publish(marker_array);
  }

  std::vector<FormationOffset> formation_offsets_;
  geometry_msgs::msg::PoseStamped virtual_leader_pose_;
  bool has_virtual_leader_pose_ = false;
  double formation_radius_;
  std::string frame_id_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr virtual_leader_sub_;
  std::vector<rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> leader_pubs_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FormationControllerNode>());
  rclcpp::shutdown();
  return 0;
}

