cmake_minimum_required(VERSION 3.29)
project(pccl)

set(CMAKE_CXX_STANDARD 20)

add_library(pccl STATIC src/pccl.cpp)

add_subdirectory(ccoip)
add_subdirectory(tinysockets)

target_include_directories(pccl PUBLIC include)
target_include_directories(pccl PRIVATE internal)

target_link_libraries(pccl PRIVATE ccoip_private)
target_link_libraries(pccl PUBLIC ccoip_public)

add_subdirectory(tests)
