cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(cartoonGan)

set(Torch_DIR /opt/libtorch/share/cmake/Torch)

set(OpenCV_DIR /opt/opencv/lib/cmake/opencv4)

# set(SYSTEM_LIBS stdc++ pthread)

# set(CMAKE_C_FLAGS -pthread)
# set(CMAKE_CXX_FLAGS "-pthread")

set(CMAKE_CXX_STANDARD 14)

# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(POCO_LIBS "/opt/cos/third_party/lib/linux/poco/libPocoCrypto.so"
#             "/opt/cos/third_party/lib/linux/poco/libPocoFoundation.so"
#             "/opt/cos/third_party/lib/linux/poco/libPocoJSON.so"
#             "/opt/cos/third_party/lib/linux/poco/libPocoNet.so"
#             "/opt/cos/third_party/lib/linux/poco/libPocoNetSSL.so"
#             "/opt/cos/third_party/lib/linux/poco/libPocoUtil.so"
#             "/opt/cos/third_party/lib/linux/poco/libPocoXML.so")

# include_directories(/opt/cos/third_party/include)

# include_directories(/opt/cos/include)
# include_directories(/home/cos-cpp-sdk-v5/include)

include_directories(/opt/aliyun-oss-cpp-sdk/sdk/include)

# add_subdirectory(/opt/cos/third_party/include)

# add_subdirectory(/opt/cos/include)

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

# find_package (Threads REQUIRED)

# set_target_properties(${PROJECT_NAME} PROPERTIES LINK_LIBRARIES -pthread)

# find_library(LIBCOSSDK cossdk /opt/cos/linux)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# target_link_libraries(${PROJECT_NAME} "/opt/cos/third_party/lib/linux/poco/libPocoCrypto.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/third_party/lib/linux/poco/libPocoFoundation.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/third_party/lib/linux/poco/libPocoJSON.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/third_party/lib/linux/poco/libPocoNet.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/third_party/lib/linux/poco/libPocoNetSSL.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/third_party/lib/linux/poco/libPocoUtil.so")
# target_link_libraries(${PROJECT_NAME} "/opt/cos/third_party/lib/linux/poco/libPocoXML.so")

# target_link_libraries(${PROJECT_NAME} ${LIBBOX2D})

# target_link_libraries(${PROJECT_NAME} pthread)

# target_link_libraries(${PROJECT_NAME} /home/cos-cpp-sdk-v5/build/lib/libcossdk.a)
# target_link_libraries(${PROJECT_NAME} /opt/cos/libs/linux/libcossdk.a)
# target_link_libraries(${PROJECT_NAME} /opt/cos/libs/linux/libcossdk-shared.so)

target_link_libraries(${PROJECT_NAME} /opt/aliyun-oss-cpp-sdk/build/lib/libalibabacloud-oss-cpp-sdk.so)