# test simplehash cuda hash
if (PCCL_BUILD_CUDA_SUPPORT AND NOT DEFINED $ENV{IS_CPU_CI})
    add_sanitized_gtest(simplehash_cuda_test simplehash_cuda_test.cpp)
    target_link_libraries(simplehash_cuda_test PRIVATE simplehash_cuda)
    target_link_libraries(simplehash_cuda_test PRIVATE ccoip_private CUDA::cuda_driver)

    # test simplehash cpu hash
    add_sanitized_gtest(simplehash_cpu_test simplehash_cpu_test.cu)
    target_link_libraries(simplehash_cpu_test PRIVATE simplehash_cpu)
    target_link_libraries(simplehash_cpu_test PRIVATE simplehash_cuda)
    target_link_libraries(simplehash_cpu_test PRIVATE ccoip_private CUDA::cudart)
endif ()

