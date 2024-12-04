cmake_minimum_required(VERSION 3.8)
project(yolov8_s)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/home/bing/code/open-source/ncnn/build/install/include")
link_directories("/home/bing/code/open-source/ncnn/build/install/lib")

# enable OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(main yolov8_s.cpp)
target_link_libraries(main ncnnd ${OpenCV_LIBS} ${OpenMP_LIBRARIES})