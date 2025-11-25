#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <vector>
#include <map>

struct FollowerOffset {
  double x, y, z;
  std::string follower_id;
};

class FollowerControllerNode : public rclcpp::Node {
public:
  FollowerControllerNode() : Node("follower_controller_node") {
    // Declare parameters
    this->declare_parameter<int>("num_leaders", 2);
    this->declare_parameter<int>("followers_per_leader", 2);
    this->declare_parameter<double>("follower_radius", 1.0);
    this->declare_parameter<std::string>("frame_id", "map");
    this->declare_parameter<double>("publish_rate", 50.0);
    this->declare_parameter<double>("follower_angle_offset", 0.0);

    // Get parameters
    int num_leaders = this->get_parameter("num_leaders").as_int();
    int followers_per_leader = this->get_parameter("followers_per_leader").as_int();
    follower_radius_ = this->get_parameter("follower_radius").as_double();
    frame_id_ = this->get_parameter("frame_id").as_string();
    double publish_rate = this->get_parameter("publish_rate").as_double();
    double angle_offset = this->get_parameter("follower_angle_offset").as_double();

    // Setup TF
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // Define follower offsets for each leader
    // Arrange followers in a circle around their leader
    for (int leader_idx = 0; leader_idx < num_leaders; leader_idx++) {
      std::vector<FollowerOffset> leader_followers;
      
      for (int follower_idx = 0; follower_idx < followers_per_leader; follower_idx++) {
        FollowerOffset offset;
        double angle = (2.0 * M_PI * follower_idx / followers_per_leader) + angle_offset;
        offset.x = follower_radius_ * std::cos(angle);
        offset.y = follower_radius_ * std::sin(angle);
        offset.z = 0.0;
        offset.follower_id = "follower_" + std::to_string(leader_idx) + "_" + std::to_string(follower_idx);
        leader_followers.push_back(offset);
      }
      
      follower_offsets_[leader_idx] = leader_followers;
    }

    // Subscribers for leader poses
    for (int i = 0; i < num_leaders; i++) {
      std::string topic = "leader_" + std::to_string(i) + "/pose";
      leader_subs_.push_back(
        this->create_subscription<geometry_msgs::msg::PoseStamped>(
          topic, 10,
          [this, i](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            leaderPoseCallback(i, msg);
          }));
    }

    // Publishers for each follower
    for (int leader_idx = 0; leader_idx < num_leaders; leader_idx++) {
      for (size_t follower_idx = 0; follower_idx < follower_offsets_[leader_idx].size(); follower_idx++) {
        std::string topic = follower_offsets_[leader_idx][follower_idx].follower_id + "/pose";
        follower_pubs_[leader_idx].push_back(
          this->create_publisher<geometry_msgs::msg::PoseStamped>(topic, 10));
      }
    }

    // Marker publisher
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "formation_followers/markers", 10);

    // Get leader colors from formation controller (hardcoded for now)
    // Red, Blue, Yellow, Magenta
    leader_colors_[0] = {1.0, 0.0, 0.0}; // Red
    leader_colors_[1] = {0.0, 0.0, 1.0}; // Blue
    leader_colors_[2] = {1.0, 1.0, 0.0}; // Yellow
    leader_colors_[3] = {1.0, 0.0, 1.0}; // Magenta

    // Timer
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
      std::bind(&FollowerControllerNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Follower Controller Node started");
    RCLCPP_INFO(this->get_logger(), "  %d leaders, %d followers per leader", 
                num_leaders, followers_per_leader);
  }

private:
  void leaderPoseCallback(int leader_idx, const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    leader_poses_[leader_idx] = *msg;
    has_leader_pose_[leader_idx] = true;
  }

  void timerCallback() {
    visualization_msgs::msg::MarkerArray marker_array;
    int marker_id = 0;

    // Process each leader's followers
    for (auto& [leader_idx, followers] : follower_offsets_) {
      if (!has_leader_pose_[leader_idx]) {
        continue;
      }

      const geometry_msgs::msg::PoseStamped& leader_pose = leader_poses_[leader_idx];

      // Get leader color
      auto color_it = leader_colors_.find(leader_idx);
      double r = 1.0, g = 0.0, b = 0.0;
      if (color_it != leader_colors_.end()) {
        r = color_it->second[0];
        g = color_it->second[1];
        b = color_it->second[2];
      }

      // Calculate and publish poses for each follower of this leader
      for (size_t follower_idx = 0; follower_idx < followers.size(); follower_idx++) {
        geometry_msgs::msg::PoseStamped follower_pose;
        // Use current time instead of copying old header timestamp
        follower_pose.header.stamp = this->now();
        follower_pose.header.frame_id = leader_pose.header.frame_id;

        // Get leader orientation
        double yaw = 2.0 * std::atan2(
          leader_pose.pose.orientation.z,
          leader_pose.pose.orientation.w);

        // Rotate follower offset by leader's orientation
        double cos_yaw = std::cos(yaw);
        double sin_yaw = std::sin(yaw);
        
        double offset_x = followers[follower_idx].x * cos_yaw - followers[follower_idx].y * sin_yaw;
        double offset_y = followers[follower_idx].x * sin_yaw + followers[follower_idx].y * cos_yaw;

        // Calculate follower position
        follower_pose.pose.position.x = leader_pose.pose.position.x + offset_x;
        follower_pose.pose.position.y = leader_pose.pose.position.y + offset_y;
        follower_pose.pose.position.z = leader_pose.pose.position.z + followers[follower_idx].z;
        
        // Follower orientation same as leader
        follower_pose.pose.orientation = leader_pose.pose.orientation;

        // Publish follower pose
        follower_pubs_[leader_idx][follower_idx]->publish(follower_pose);

        // Create marker for visualization (transparent)
        visualization_msgs::msg::Marker marker;
        marker.header.stamp = this->now();
        marker.header.frame_id = follower_pose.header.frame_id;
        marker.ns = "formation_followers";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose = follower_pose.pose;
        marker.scale.x = 0.2;
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.color.a = 0.5; // Transparent for followers
        marker.lifetime = rclcpp::Duration(0, 0);
        marker_array.markers.push_back(marker);
      }
    }

    marker_pub_->publish(marker_array);
  }

  std::map<int, std::vector<FollowerOffset>> follower_offsets_;
  std::map<int, geometry_msgs::msg::PoseStamped> leader_poses_;
  std::map<int, bool> has_leader_pose_;
  std::map<int, std::vector<rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr>> follower_pubs_;
  std::map<int, std::array<double, 3>> leader_colors_;
  double follower_radius_;
  std::string frame_id_;

  std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> leader_subs_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FollowerControllerNode>());
  rclcpp::shutdown();
  return 0;
}

