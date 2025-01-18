add_library(ccoip_private STATIC
        src/ccoip_master.cpp src/ccoip_client_handler.cpp src/ccoip_master_handler.cpp
        src/ccoip_packets.cpp src/ccoip_packet_buffer.cpp src/ccoip_client.cpp src/uuid_utils.cpp src/crc32.cpp
        src/ccoip_master_state.cpp src/ccoip_client_state.cpp src/reduce_kernels.cpp src/reduce.cpp src/quantize.cpp
        src/quantize_kernels.cpp)

add_library(ccoip_public INTERFACE)
target_include_directories(ccoip_public INTERFACE public_include)

# Arch specific compilation options.
# Excludes older x86-64 CPUs released before 2008 (Intel) or 2011 (AMD) and excludes ARM CPUs before 2013.
# These CPUs do not support the required instruction sets.
# Instructions are needed for the fast CRC32 implementation.
if(${IS_AMD64})
    target_compile_options(ccoip_private PRIVATE -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpclmul)
elseif(${IS_ARM64})
    target_compile_options(ccoip_private PRIVATE -march=armv8-a+simd)
endif()

target_include_directories(ccoip_private PUBLIC internal_include)
target_link_libraries(ccoip_public INTERFACE tinysockets)
target_link_libraries(ccoip_private PUBLIC ccoip_public)

target_include_directories(ccoip_private PRIVATE internal)

target_link_libraries(ccoip_private PRIVATE pccl_log)

add_subdirectory(tests)