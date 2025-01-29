cmake_minimum_required(VERSION 3.10)

project(ring_reduce_experiment CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(ring_reduce_experiment ring_reduce_experiment.cpp)
target_compile_definitions(ring_reduce_experiment PRIVATE RING_REDUCE_EXPERIMENT_MAIN_IMPL=1)

# hack
add_subdirectory(googletest)

add_executable(ring_reduce_impl_test ring_reduce_impl_test.cpp ring_reduce_experiment.cpp)
target_link_libraries(ring_reduce_impl_test gtest_main)