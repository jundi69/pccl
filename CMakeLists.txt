cmake_minimum_required(VERSION 3.28)
project(pccl)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(IS_AMD64 TRUE)
else()
    set(IS_AMD64 FALSE)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(IS_ARM64 TRUE)
else()
    set(IS_ARM64 FALSE)
endif()

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
endif()
target_include_directories(pccl PUBLIC include)
target_include_directories(pccl PRIVATE internal)

target_link_libraries(pccl PRIVATE ccoip_private)
target_link_libraries(pccl PUBLIC ccoip_public)

add_subdirectory(tests)
