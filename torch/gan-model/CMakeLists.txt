cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(gan-model gan-model.cpp)
target_link_libraries(gan-model "${TORCH_LIBRARIES}")
set_property(TARGET gan-model PROPERTY CXX_STANDARD 14)