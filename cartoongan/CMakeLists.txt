cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(cartoonGan)

set(Torch_DIR /opt/libtorch/share/cmake/Torch)

set(OpenCV_DIR /opt/opencv/lib/cmake/opencv4)

set(CMAKE_CXX_STANDARD 14)

# # set(POCO_LIBS "/opt/cos/poco/libPocoCrypto.so"
# #             "/opt/cos/poco/libPocoFoundation.so"
# #             "/opt/cos/poco/libPocoJSON.so"
# #             "/opt/cos/poco/libPocoNet.so"
# #             "/opt/cos/poco/libPocoNetSSL.so"
# #             "/opt/cos/poco/libPocoUtil.so"
# #             "/opt/cos/poco/libPocoXML.so")

# include_directories(/opt/cos/include_poco)

# include_directories(/opt/cos/include/)

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

# find_library(LIBCOSSDK cossdk /opt/cos/linux)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# target_link_libraries(${PROJECT_NAME} "/opt/cos/poco/libPocoCrypto.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/poco/libPocoFoundation.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/poco/libPocoJSON.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/poco/libPocoNet.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/poco/libPocoNetSSL.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/poco/libPocoUtil.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/poco/libPocoXML.so")

# # target_link_libraries(${PROJECT_NAME} "/opt/cos/linux/libcossdk-shared.so")
# target_link_libraries(${PROJECT_NAME} ${LIBBOX2D})