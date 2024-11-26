cmake_minimum_required(VERSION 3.8)
project(resnet18)



include_directories("/home/bing/code/open-source/ncnn/build_release_nobench/install/include")
link_directories("/home/bing/code/open-source/ncnn/build_release_nobench/install/lib")

# add executable 
add_executable(resnet18 resnet18.cpp)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# link libraries 
target_link_libraries(resnet18 ncnn ${OpenCV_LIBS} ${OpenMP_LIBRARIES})
