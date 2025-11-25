#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

struct Waypoint {
  double x, y, z;
};

class VirtualLeaderTrajectoryNode : public rclcpp::Node {
public:
  VirtualLeaderTrajectoryNode() : Node("virtual_leader_trajectory_node") {
    // Declare parameters
    this->declare_parameter<std::string>("waypoint_file", "config/waypoints.txt");
    this->declare_parameter<double>("trajectory_speed", 1.0);
    this->declare_parameter<double>("waypoint_tolerance", 0.1);
    this->declare_parameter<std::string>("frame_id", "map");
    this->declare_parameter<double>("publish_rate", 50.0);
    this->declare_parameter<double>("max_yaw_rate", 1.0); // rad/s

    // Get parameters
    waypoint_file_ = this->get_parameter("waypoint_file").as_string();
    trajectory_speed_ = this->get_parameter("trajectory_speed").as_double();
    waypoint_tolerance_ = this->get_parameter("waypoint_tolerance").as_double();
    frame_id_ = this->get_parameter("frame_id").as_string();
    double publish_rate = this->get_parameter("publish_rate").as_double();
    max_yaw_rate_ = this->get_parameter("max_yaw_rate").as_double();

    // Load waypoints
    if (!loadWaypoints(waypoint_file_)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load waypoints from %s", waypoint_file_.c_str());
      return;
    }

    if (waypoints_.empty()) {
      RCLCPP_ERROR(this->get_logger(), "No waypoints loaded!");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints", waypoints_.size());

    // Publishers
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "virtual_leader/pose", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "virtual_leader/marker", 10);

    // Initialize trajectory
    current_waypoint_idx_ = 0;
    current_pose_.position.x = waypoints_[0].x;
    current_pose_.position.y = waypoints_[0].y;
    current_pose_.position.z = waypoints_[0].z;
    current_pose_.orientation.w = 1.0;
    
    // Initialize current yaw based on direction to first waypoint
    if (waypoints_.size() > 1) {
      double dx = waypoints_[1].x - waypoints_[0].x;
      double dy = waypoints_[1].y - waypoints_[0].y;
      current_yaw_ = std::atan2(dy, dx);
    } else {
      current_yaw_ = 0.0;
    }
    current_pose_.orientation.z = std::sin(current_yaw_ / 2.0);
    current_pose_.orientation.w = std::cos(current_yaw_ / 2.0);

    // Timer
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
      std::bind(&VirtualLeaderTrajectoryNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Virtual Leader Trajectory Node started");
  }

private:
  bool loadWaypoints(const std::string& filename) {
    std::string full_path = ament_index_cpp::get_package_share_directory("mav_formation_control") 
                            + "/" + filename;
    
    std::ifstream file(full_path);
    if (!file.is_open()) {
      RCLCPP_WARN(this->get_logger(), "Could not open %s, trying direct path", full_path.c_str());
      file.open(filename);
      if (!file.is_open()) {
        return false;
      }
    }

    waypoints_.clear();
    std::string line;
    while (std::getline(file, line)) {
      // Skip empty lines and comments
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::istringstream iss(line);
      Waypoint wp;
      if (iss >> wp.x >> wp.y >> wp.z) {
        waypoints_.push_back(wp);
      }
    }

    file.close();
    return !waypoints_.empty();
  }

  void timerCallback() {
    // Update trajectory
    updateTrajectory();

    // Publish pose
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = frame_id_;
    pose_msg.pose = current_pose_;
    pose_pub_->publish(pose_msg);

    // Publish marker for visualization
    publishMarker();
  }

  void updateTrajectory() {
    if (waypoints_.empty() || current_waypoint_idx_ >= waypoints_.size()) {
      return;
    }

    const Waypoint& target = waypoints_[current_waypoint_idx_];
    double dx = target.x - current_pose_.position.x;
    double dy = target.y - current_pose_.position.y;
    double dz = target.z - current_pose_.position.z;
    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (distance < waypoint_tolerance_) {
      // Reached waypoint, move to next
      current_waypoint_idx_++;
      if (current_waypoint_idx_ >= waypoints_.size()) {
        // Loop back to first waypoint
        current_waypoint_idx_ = 0;
        RCLCPP_INFO(this->get_logger(), "Completed waypoint loop, restarting");
      } else {
        RCLCPP_INFO(this->get_logger(), "Reached waypoint %zu", current_waypoint_idx_);
      }
      return;
    }

    // Move towards target waypoint
    double dt = 1.0 / this->get_parameter("publish_rate").as_double();
    double step = trajectory_speed_ * dt;
    
    if (distance > step) {
      // Normalize direction and move
      current_pose_.position.x += (dx / distance) * step;
      current_pose_.position.y += (dy / distance) * step;
      current_pose_.position.z += (dz / distance) * step;

      // Update orientation smoothly to face direction of movement
      if (std::sqrt(dx*dx + dy*dy) > 0.01) {
        double desired_yaw = std::atan2(dy, dx);
        current_yaw_ = smoothYawUpdate(current_yaw_, desired_yaw, dt);
        
        // Update quaternion from yaw
        current_pose_.orientation.z = std::sin(current_yaw_ / 2.0);
        current_pose_.orientation.w = std::cos(current_yaw_ / 2.0);
        current_pose_.orientation.x = 0.0;
        current_pose_.orientation.y = 0.0;
      }
    } else {
      // Close enough, snap to waypoint
      current_pose_.position.x = target.x;
      current_pose_.position.y = target.y;
      current_pose_.position.z = target.z;
      
      // Still update orientation smoothly even when snapping position
      double desired_yaw = std::atan2(dy, dx);
      current_yaw_ = smoothYawUpdate(current_yaw_, desired_yaw, dt);
      current_pose_.orientation.z = std::sin(current_yaw_ / 2.0);
      current_pose_.orientation.w = std::cos(current_yaw_ / 2.0);
      current_pose_.orientation.x = 0.0;
      current_pose_.orientation.y = 0.0;
    }
  }

  // Smooth yaw update with maximum angular velocity constraint
  double smoothYawUpdate(double current_yaw, double desired_yaw, double dt) {
    // Normalize angles to [-pi, pi]
    current_yaw = normalizeAngle(current_yaw);
    desired_yaw = normalizeAngle(desired_yaw);
    
    // Calculate shortest angular distance
    double yaw_error = normalizeAngle(desired_yaw - current_yaw);
    
    // Limit the change by maximum yaw rate
    double max_yaw_change = max_yaw_rate_ * dt;
    double yaw_change = std::copysign(std::min(std::abs(yaw_error), max_yaw_change), yaw_error);
    
    // Update yaw
    double new_yaw = current_yaw + yaw_change;
    return normalizeAngle(new_yaw);
  }

  // Normalize angle to [-pi, pi]
  double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
  }

  void publishMarker() {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id_;
    marker.header.stamp = this->now();
    marker.ns = "virtual_leader";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = current_pose_;
    marker.scale.x = 0.3;
    marker.scale.y = 0.3;
    marker.scale.z = 0.3;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    marker.lifetime = rclcpp::Duration(0, 0); // 0 means never expires
    marker_pub_->publish(marker);
  }

  std::string waypoint_file_;
  std::vector<Waypoint> waypoints_;
  size_t current_waypoint_idx_;
  geometry_msgs::msg::Pose current_pose_;
  double trajectory_speed_;
  double waypoint_tolerance_;
  std::string frame_id_;
  double max_yaw_rate_;
  double current_yaw_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VirtualLeaderTrajectoryNode>());
  rclcpp::shutdown();
  return 0;
}

