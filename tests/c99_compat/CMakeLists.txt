enable_language(C)
set(CMAKE_C_STANDARD 99)
add_executable(pccl_c99_compat enforce_c99_compat.c)
target_link_libraries(pccl_c99_compat pccl)
