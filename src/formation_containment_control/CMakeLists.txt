cmake_minimum_required(VERSION 3.8)
project(formation_containment_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages (use separate target name to avoid conflict with ament_python_install_package)
rosidl_generate_interfaces(${PROJECT_NAME}_msgs
  "msg/FormationState.msg"
  "msg/ControllerState.msg"
  "msg/AgentState.msg"
  "srv/SetFormation.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

# Install Python modules manually (cannot use ament_python_install_package because
# rosidl_generator_py already creates a Python package with the same name for message bindings)
install(DIRECTORY
  ${PROJECT_NAME}/
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
)

# Install Python scripts as executables
install(PROGRAMS
  scripts/formation_containment_node.py
  scripts/virtual_leader_node.py
  scripts/visualization_node.py
  scripts/crazyswarm_bridge_node.py
  scripts/simulation_node.py
  scripts/standalone_demo.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  # Add pytest tests
  ament_add_pytest_test(test_graph_theory test/test_graph_theory.py)
  ament_add_pytest_test(test_controllers test/test_controllers.py)
  ament_add_pytest_test(test_convex_hull test/test_convex_hull.py)
endif()

ament_package()

