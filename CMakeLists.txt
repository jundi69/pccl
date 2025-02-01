cmake_minimum_required(VERSION 3.16)
project(pccl)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Detect CPU architecture
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(IS_AMD64 TRUE)
else ()
    set(IS_AMD64 FALSE)
endif ()
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(IS_ARM64 TRUE)
else ()
    set(IS_ARM64 FALSE)
endif ()

# Handle CUDA build
option(PCCL_BUILD_CUDA_SUPPORT "Build PCCL with CUDA support" ON)

if (PCCL_BUILD_CUDA_SUPPORT)
    find_package(CUDAToolkit)

    # check if the cuda toolkit was found
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit was found!")
        # You can now use CUDA-specific targets and include directories.
    else()
        message(WARNING "CUDA Toolkit not found! CUDA support will be disabled!")
        set(PCCL_BUILD_CUDA_SUPPORT OFF)
    endif()
endif ()

if (PCCL_BUILD_CUDA_SUPPORT)
    set(CMAKE_CUDA_STANDARD 20)

    # target sm_50 as a lower bound. We override this in a target specific way when necessary.
    set(CMAKE_CUDA_ARCHITECTURES "50")

    # enable cuda language
    enable_language(CUDA)
endif ()

# add third party libraries
add_subdirectory(third_party)

enable_testing()

# include testing functions
include(cmake/testing.cmake)

# add internal libraries
add_subdirectory(common_utils)
add_subdirectory(log)
add_subdirectory(ccoip)
add_subdirectory(ccoip_master)
add_subdirectory(tinysockets)

add_library(pccl SHARED src/pccl.cpp)
if (NOT WIN32)
    target_compile_options(pccl PRIVATE -Wall -Wextra -Werror -Wpedantic -Wno-unused-parameter)
endif ()

if (PCCL_BUILD_CUDA_SUPPORT)
    target_compile_definitions(pccl PRIVATE PCCL_HAS_CUDA_SUPPORT)
endif ()

target_include_directories(pccl PUBLIC include)
target_include_directories(pccl PRIVATE internal)

target_link_libraries(pccl PRIVATE ccoip_private)
target_link_libraries(pccl PUBLIC ccoip_public)
target_link_libraries(pccl PRIVATE pccl_log)

add_subdirectory(tests)
