cmake_minimum_required(VERSION 3.0.2)
project(motion_planning)

find_package(catkin REQUIRED COMPONENTS)

find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  )

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${pinocchio_INCLUDE_DIRS}
  )
