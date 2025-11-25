cmake_minimum_required(VERSION 3.8)
project(mav_formation_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Include directories
include_directories(include)

# Virtual Leader Trajectory Generator Node
add_executable(virtual_leader_trajectory_node
  src/virtual_leader_trajectory_node.cpp
)
ament_target_dependencies(virtual_leader_trajectory_node
  rclcpp
  geometry_msgs
  std_msgs
  visualization_msgs
  ament_index_cpp
)

# Formation Controller Node
add_executable(formation_controller_node
  src/formation_controller_node.cpp
)
ament_target_dependencies(formation_controller_node
  rclcpp
  geometry_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Follower Controller Node
add_executable(follower_controller_node
  src/follower_controller_node.cpp
)
ament_target_dependencies(follower_controller_node
  rclcpp
  geometry_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Install executables
install(TARGETS
  virtual_leader_trajectory_node
  formation_controller_node
  follower_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

