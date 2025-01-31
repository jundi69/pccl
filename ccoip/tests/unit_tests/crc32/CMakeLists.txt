# test crc32 cpu hash
add_sanitized_gtest(crc32_cpu_test crc32_cpu_test.cpp)
target_link_libraries(crc32_cpu_test PRIVATE ccoip_private)