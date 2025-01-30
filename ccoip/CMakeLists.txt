add_library(ccoip_internal STATIC
        src/ccoip_master.cpp src/ccoip_client_handler.cpp src/ccoip_master_handler.cpp
        src/bandwidth_store.cpp src/benchmark_runner.cpp src/topolgy_optimizer.cpp
        src/ccoip_packets.cpp src/ccoip_packet_buffer.cpp src/ccoip_client.cpp src/uuid_utils.cpp src/crc32.cpp
        src/ccoip_master_state.cpp src/ccoip_client_state.cpp src/reduce_kernels.cpp src/reduce.cpp src/quantize.cpp
        src/quantize_kernels.cpp)

target_link_libraries(ccoip_internal PRIVATE tinysockets)
target_link_libraries(ccoip_internal PRIVATE pccl_log)
target_link_libraries(ccoip_internal PRIVATE tsp)

# Arch specific compilation options.
# Excludes older x86-64 CPUs released before 2008 (Intel) or 2011 (AMD) and excludes ARM CPUs before 2013.
# These CPUs do not support the required instruction sets.
# Instructions are needed for the fast CRC32 implementation.
if(${IS_AMD64})
    target_compile_options(ccoip_internal PRIVATE -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpclmul)
elseif(${IS_ARM64})
    target_compile_options(ccoip_internal PRIVATE -march=armv8-a+simd)
endif()

# add stub C++ language targets ccoip_private & ccoip_public
add_library(ccoip_private STATIC cmake_hacks/dummy.cpp)
add_library(ccoip_public STATIC cmake_hacks/dummy.cpp)

target_include_directories(ccoip_internal PUBLIC internal)
target_include_directories(ccoip_internal PUBLIC internal_include)
target_include_directories(ccoip_internal PUBLIC include)

target_include_directories(ccoip_private PUBLIC internal_include)
target_include_directories(ccoip_private PUBLIC public_include)
target_include_directories(ccoip_public PUBLIC public_include)

# we need to hide ccoip_internal, hence no INTERFACE possible
target_link_libraries(ccoip_private PRIVATE ccoip_internal)
target_link_libraries(ccoip_public PRIVATE ccoip_internal)

add_subdirectory(tests)