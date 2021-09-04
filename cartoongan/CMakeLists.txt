cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(cartoonGan)

set(Torch_DIR /opt/libtorch/share/cmake/Torch)

set(OpenCV_DIR /opt/opencv/lib/cmake/opencv4)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
