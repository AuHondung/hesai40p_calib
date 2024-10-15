# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "hesai_lidar: 2 messages, 0 services")

set(MSG_I_FLAGS "-Ihesai_lidar:/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(hesai_lidar_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg" NAME_WE)
add_custom_target(_hesai_lidar_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "hesai_lidar" "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg" ""
)

get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg" NAME_WE)
add_custom_target(_hesai_lidar_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "hesai_lidar" "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg" "std_msgs/Header:hesai_lidar/PandarPacket"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(hesai_lidar
  "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/hesai_lidar
)
_generate_msg_cpp(hesai_lidar
  "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/hesai_lidar
)

### Generating Services

### Generating Module File
_generate_module_cpp(hesai_lidar
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/hesai_lidar
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(hesai_lidar_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(hesai_lidar_generate_messages hesai_lidar_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg" NAME_WE)
add_dependencies(hesai_lidar_generate_messages_cpp _hesai_lidar_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg" NAME_WE)
add_dependencies(hesai_lidar_generate_messages_cpp _hesai_lidar_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(hesai_lidar_gencpp)
add_dependencies(hesai_lidar_gencpp hesai_lidar_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS hesai_lidar_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(hesai_lidar
  "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/hesai_lidar
)
_generate_msg_eus(hesai_lidar
  "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/hesai_lidar
)

### Generating Services

### Generating Module File
_generate_module_eus(hesai_lidar
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/hesai_lidar
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(hesai_lidar_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(hesai_lidar_generate_messages hesai_lidar_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg" NAME_WE)
add_dependencies(hesai_lidar_generate_messages_eus _hesai_lidar_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg" NAME_WE)
add_dependencies(hesai_lidar_generate_messages_eus _hesai_lidar_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(hesai_lidar_geneus)
add_dependencies(hesai_lidar_geneus hesai_lidar_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS hesai_lidar_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(hesai_lidar
  "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/hesai_lidar
)
_generate_msg_lisp(hesai_lidar
  "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/hesai_lidar
)

### Generating Services

### Generating Module File
_generate_module_lisp(hesai_lidar
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/hesai_lidar
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(hesai_lidar_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(hesai_lidar_generate_messages hesai_lidar_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg" NAME_WE)
add_dependencies(hesai_lidar_generate_messages_lisp _hesai_lidar_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg" NAME_WE)
add_dependencies(hesai_lidar_generate_messages_lisp _hesai_lidar_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(hesai_lidar_genlisp)
add_dependencies(hesai_lidar_genlisp hesai_lidar_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS hesai_lidar_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(hesai_lidar
  "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/hesai_lidar
)
_generate_msg_nodejs(hesai_lidar
  "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/hesai_lidar
)

### Generating Services

### Generating Module File
_generate_module_nodejs(hesai_lidar
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/hesai_lidar
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(hesai_lidar_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(hesai_lidar_generate_messages hesai_lidar_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg" NAME_WE)
add_dependencies(hesai_lidar_generate_messages_nodejs _hesai_lidar_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg" NAME_WE)
add_dependencies(hesai_lidar_generate_messages_nodejs _hesai_lidar_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(hesai_lidar_gennodejs)
add_dependencies(hesai_lidar_gennodejs hesai_lidar_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS hesai_lidar_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(hesai_lidar
  "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/hesai_lidar
)
_generate_msg_py(hesai_lidar
  "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/hesai_lidar
)

### Generating Services

### Generating Module File
_generate_module_py(hesai_lidar
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/hesai_lidar
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(hesai_lidar_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(hesai_lidar_generate_messages hesai_lidar_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarPacket.msg" NAME_WE)
add_dependencies(hesai_lidar_generate_messages_py _hesai_lidar_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/max/Documents/01_Project/17_lidar/HesaiLidar_ws/src/HesaiLidar_General_ROS/msg/PandarScan.msg" NAME_WE)
add_dependencies(hesai_lidar_generate_messages_py _hesai_lidar_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(hesai_lidar_genpy)
add_dependencies(hesai_lidar_genpy hesai_lidar_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS hesai_lidar_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/hesai_lidar)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/hesai_lidar
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(hesai_lidar_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/hesai_lidar)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/hesai_lidar
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(hesai_lidar_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/hesai_lidar)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/hesai_lidar
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(hesai_lidar_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/hesai_lidar)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/hesai_lidar
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(hesai_lidar_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/hesai_lidar)
  install(CODE "execute_process(COMMAND \"/home/max/anaconda3/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/hesai_lidar\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/hesai_lidar
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(hesai_lidar_generate_messages_py std_msgs_generate_messages_py)
endif()
