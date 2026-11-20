cmake_minimum_required(VERSION 3.8)
project(x2p_cmd_relay)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(aimdk_msgs REQUIRED)

add_executable(x2p_cmd_relay src/cmd_relay.cpp)
ament_target_dependencies(x2p_cmd_relay rclcpp aimdk_msgs)
install(TARGETS x2p_cmd_relay DESTINATION lib/${PROJECT_NAME})

ament_package()
