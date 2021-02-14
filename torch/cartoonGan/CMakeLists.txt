cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(cartoonGan)

#set(Torch_DIR /home/hlz/blackhighsea/torch/libtorch/share/cmake/Torch)

#set(OpenCV_DIR /home/hlz/blackhighsea/torch/opencv/lib/cmake/opencv4)

#find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

#target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries(${PROJECT_NAME} "${OpenCV_LIBS}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)