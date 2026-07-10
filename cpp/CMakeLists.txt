cmake_minimum_required(VERSION 3.16)
project(omr_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(omr_core
  src/image.cpp
  src/preprocess.cpp
  src/morph.cpp
  src/staff.cpp
  src/segment.cpp
  src/heads.cpp
  src/descriptors.cpp
  src/knn.cpp
  src/decode.cpp
  src/npy.cpp
  src/duration_knn.cpp
  src/midi.cpp
  src/io_primus.cpp
)

target_include_directories(omr_core PUBLIC src third_party)

add_executable(omr_cpp src/main.cpp)
target_link_libraries(omr_cpp PRIVATE omr_core)
