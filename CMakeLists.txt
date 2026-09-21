cmake_minimum_required(VERSION 3.22)
project(cuda_linear_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

find_path(CUDNN_INCLUDE_DIR cudnn.h
  HINTS /usr/include /usr/local/cuda/include
)
find_library(CUDNN_LIBRARY cudnn
  HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64
)

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
  message(FATAL_ERROR "cuDNN not found")
endif()

add_executable(cuda_linear_bench src/main.cu)
target_include_directories(cuda_linear_bench PRIVATE ${CUDNN_INCLUDE_DIR})
target_link_libraries(cuda_linear_bench PRIVATE CUDA::cudart CUDA::cublas ${CUDNN_LIBRARY})
