add_subdirectory(libuv)
add_subdirectory(googletest)

set(LIBTSP_BUILD_TESTS OFF)
add_subdirectory(libtsp)

add_subdirectory(threadpark)

set(QUANT_BUILD_TESTS OFF)
# add_subdirectory(quantization-kernels)

add_subdirectory(threadpool)