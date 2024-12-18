# test fvn1a hash
add_sanitized_gtest(fvn1a_test fvn1a_test.cpp)
target_link_libraries(fvn1a_test PRIVATE ccoip_private)