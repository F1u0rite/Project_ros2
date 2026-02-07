cmake_minimum_required(VERSION 3.8)
project(orbslam3_rgbd_ros2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =========================
# ROS2 dependencies
# =========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

# =========================
# Non-ROS deps
# =========================
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)

# =========================
# Pangolin (CMake config)
# =========================
find_package(Pangolin REQUIRED)

# =========================
# ORB-SLAM3 paths
# =========================
set(ORB_SLAM3_ROOT "/home/fluorite/AAE5303/ORB_SLAM3")

# =========================
# Common include dirs
# =========================
set(ORB_SLAM3_INCLUDES
  ${ORB_SLAM3_ROOT}
  ${ORB_SLAM3_ROOT}/include
  ${ORB_SLAM3_ROOT}/include/CameraModels
  ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
  ${ORB_SLAM3_ROOT}/Thirdparty/g2o
  ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

# =========================
# Common library dirs
# =========================
set(ORB_SLAM3_LIBDIRS
  ${ORB_SLAM3_ROOT}/lib
  ${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib
  ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib
)

# =========================
# Common link libs
# =========================
set(ORB_SLAM3_LINK_LIBS
  ORB_SLAM3
  ${OpenCV_LIBS}
  OpenGL::GL
)

# ============================================================
# rgbd_node
# ============================================================
add_executable(rgbd_node
  src/rgbd_node.cc
)

ament_target_dependencies(rgbd_node
  rclcpp
  sensor_msgs
  cv_bridge
  message_filters
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
)

target_include_directories(rgbd_node PRIVATE
  ${ORB_SLAM3_INCLUDES}
)

# Pangolin include (兼容不同变量名)
if(DEFINED Pangolin_INCLUDE_DIRS)
  target_include_directories(rgbd_node PRIVATE ${Pangolin_INCLUDE_DIRS})
endif()
if(DEFINED Pangolin_INCLUDE_DIR)
  target_include_directories(rgbd_node PRIVATE ${Pangolin_INCLUDE_DIR})
endif()

target_link_directories(rgbd_node PRIVATE
  ${ORB_SLAM3_LIBDIRS}
)

target_link_libraries(rgbd_node
  ${ORB_SLAM3_LINK_LIBS}
)

# ---- Pangolin link: support multiple export styles ----
if(TARGET Pangolin::pangolin)
  message(STATUS "Linking Pangolin via target: Pangolin::pangolin")
  target_link_libraries(rgbd_node Pangolin::pangolin)
elseif(TARGET pangolin)
  message(STATUS "Linking Pangolin via target: pangolin")
  target_link_libraries(rgbd_node pangolin)
elseif(DEFINED Pangolin_LIBRARIES)
  message(STATUS "Linking Pangolin via variable: Pangolin_LIBRARIES")
  target_link_libraries(rgbd_node ${Pangolin_LIBRARIES})
elseif(DEFINED PANGOLIN_LIBRARIES)
  message(STATUS "Linking Pangolin via variable: PANGOLIN_LIBRARIES")
  target_link_libraries(rgbd_node ${PANGOLIN_LIBRARIES})
else()
  message(FATAL_ERROR "Pangolin was found but no usable target/variables exported.")
endif()

set_target_properties(rgbd_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "/usr/local/lib;${ORB_SLAM3_ROOT}/lib;${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib;${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib"
)

install(TARGETS rgbd_node
  DESTINATION lib/${PROJECT_NAME}
)

# ============================================================
# stereo_ir_node (NEW)
# ============================================================
add_executable(stereo_ir_node
  src/stereo_ir_node.cc
)

ament_target_dependencies(stereo_ir_node
  rclcpp
  sensor_msgs
  cv_bridge
  message_filters
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
)

target_include_directories(stereo_ir_node PRIVATE
  ${ORB_SLAM3_INCLUDES}
)

# Pangolin include (兼容不同变量名)
if(DEFINED Pangolin_INCLUDE_DIRS)
  target_include_directories(stereo_ir_node PRIVATE ${Pangolin_INCLUDE_DIRS})
endif()
if(DEFINED Pangolin_INCLUDE_DIR)
  target_include_directories(stereo_ir_node PRIVATE ${Pangolin_INCLUDE_DIR})
endif()

target_link_directories(stereo_ir_node PRIVATE
  ${ORB_SLAM3_LIBDIRS}
)

target_link_libraries(stereo_ir_node
  ${ORB_SLAM3_LINK_LIBS}
)

# ---- Pangolin link: same as rgbd_node ----
if(TARGET Pangolin::pangolin)
  target_link_libraries(stereo_ir_node Pangolin::pangolin)
elseif(TARGET pangolin)
  target_link_libraries(stereo_ir_node pangolin)
elseif(DEFINED Pangolin_LIBRARIES)
  target_link_libraries(stereo_ir_node ${Pangolin_LIBRARIES})
elseif(DEFINED PANGOLIN_LIBRARIES)
  target_link_libraries(stereo_ir_node ${PANGOLIN_LIBRARIES})
else()
  message(FATAL_ERROR "Pangolin was found but no usable target/variables exported.")
endif()

set_target_properties(stereo_ir_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "/usr/local/lib;${ORB_SLAM3_ROOT}/lib;${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib;${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib"
)

install(TARGETS stereo_ir_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
