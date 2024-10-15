cmake_minimum_required(VERSION 3.0.2)
project(lidar2car)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES HesaiLidar_sub
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros pcl_conversions
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

set(SOURCES
  src/rpzCalib.cpp
  src/GroundExtractor.cpp
  src/lidar2car_node.cpp
)

set(DEP_LIBS
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(lidar2car_node ${SOURCES})

# Specify libraries to link a library or executable target against
target_link_libraries(lidar2car_node ${DEP_LIBS} -lgomp)

