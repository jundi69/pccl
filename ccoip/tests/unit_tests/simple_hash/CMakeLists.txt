# test simplehash cuda hash
if (PCCL_BUILD_CUDA_SUPPORT)
    # NOTE: CUDA doesn't work with ASAN, hence use un-sanitized tests here
    add_unsanitized_gtest(simplehash_cuda_test simplehash_cuda_test.cpp)
    target_link_libraries(simplehash_cuda_test PRIVATE simplehash_cuda)
    target_link_libraries(simplehash_cuda_test PRIVATE ccoip_private CUDA::cuda_driver)

    # test simplehash cpu hash
    add_unsanitized_gtest(simplehash_cpu_test simplehash_cpu_test.cu)
    target_link_libraries(simplehash_cpu_test PRIVATE simplehash_cpu)
    target_link_libraries(simplehash_cpu_test PRIVATE simplehash_cuda)
    target_link_libraries(simplehash_cpu_test PRIVATE ccoip_private CUDA::cudart)
endif ()


# test simplehash cpu hash
add_sanitized_gtest(simplehash_cpu_no_cuda_test simplehash_cpu_test_no_cuda.cpp)
target_link_libraries(simplehash_cpu_no_cuda_test PRIVATE ccoip_private simplehash_cpu)

