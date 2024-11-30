cmake_minimum_required(VERSION 3.8)
project(yolov8_s)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(main yolov8_s.cpp)
target_link_libraries(main ${OpenCV_LIBS})