cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

set(SYCL_ROOT "/usr/local/dpcpp-cuda")
set(CMAKE_CXX_COMPILER "/usr/local/dpcpp-cuda/bin/clang++")

# SYCL installation
if (NOT SYCL_ROOT) 
  message(FATAL_ERROR "No SYCL installation detected")
endif(NOT SYCL_ROOT)

set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/lib/clang/11.0.0/include/")
set(DPCT_INCLUDE_DIR "/mnt/TFG")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")
set(SYCL_FLAGS "-fsycl" 
      "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice" 
      "-fsycl-unnamed-lambda")


# Build the SYCL code

add_executable (sycl_b+tree main.c.dp.cpp)
target_compile_features(sycl_b+tree PUBLIC cxx_std_17)
target_compile_options(sycl_b+tree PUBLIC ${SYCL_FLAGS})
target_link_libraries(sycl_b+tree PUBLIC ${SYCL_FLAGS})
target_include_directories(sycl_b+tree PUBLIC ${SYCL_INCLUDE_DIR})
target_include_directories(sycl_b+tree PUBLIC ${DPCT_INCLUDE_DIR})
target_link_libraries(sycl_b+tree PUBLIC ${SYCL_LIB})

