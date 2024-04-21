cmake_minimum_required(VERSION 3.21.1)
project(onnx_mnist)

set(CMAKE_CXX_STANDARD 20)

add_executable(onnx_mnist onnx_mnist.cpp)
target_include_directories(onnx_mnist PRIVATE onnxruntime-linux-x64-1.17.3/include)
target_link_libraries(onnx_mnist PRIVATE ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.17.3/lib/libonnxruntime.so)

# add openCV dependency
find_package(OpenCV REQUIRED)
target_include_directories(onnx_mnist PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(onnx_mnist PRIVATE ${OpenCV_LIBS})
