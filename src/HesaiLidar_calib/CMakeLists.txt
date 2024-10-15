cmake_minimum_required(VERSION 3.0.2)
project(HesaiLidar_calib)

set(CMAKE_CXX_STANDARD 17)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
)

find_package(PCL REQUIRED COMPONENTS visualization)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(G2O_DIR "/home/max/Documents/01_Project/00_thirdparty/g2o/")
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES HesaiLidar_sub
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros pcl_conversions
  DEPENDS G2O
#  DEPENDS system_lib
)

include_directories(
  include
  src
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)


# add_executable(hesaiLidar_calib_node src/hesailidar_calib_node.cpp)

# # Specify libraries to link a library or executable target against
# target_link_libraries(hesaiLidar_calib_node
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
# )

set(SOURCES
  # src/hesailidar_calib_node.cpp
  src/lidar_calib_node.cpp
  src/GroundExtract.cpp
  src/PlateExtract.cpp
)

add_executable(${PROJECT_NAME}_node ${SOURCES})

# Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Eigen_LIBRARIES} 
  ${G2O_LIBRARIES}
  ${G2O_LIBS}
  g2o_core 
  g2o_stuff
  # /home/max/Documents/01_Project/00_thirdparty/g2o/build/lib/libg2o_core.so
  # /home/max/Documents/01_Project/00_thirdparty/g2o/build/lib/libg2o_stuff.so
  # /home/max/Documents/01_Project/00_thirdparty/g2o/build/lib/libg2o.so
)


# add_dependencies(hesailidar_sub_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})