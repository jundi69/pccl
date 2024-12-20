# test crc32 hash
add_sanitized_gtest(crc32_test crc32_test.cpp)
target_link_libraries(crc32_test PRIVATE ccoip_private)