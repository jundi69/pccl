option(PCCL_SANITIZE_TESTS "Enable address sanitizer for tests" OFF)

if (DEFINED $ENV{IS_CI})
    message(STATUS "Running in CI, enabling sanitizers in tests")
endif ()

# ─── BEGIN SANITIZER + STATIC-RUNTIME SETUP ─────────────────────────────────

# 1) Which sanitizers to turn on at compile time (C & C++ only)
set(SAN_FLAGS
        -fsanitize=address
        -fsanitize=leak
        -fsanitize=undefined
)

# 2) Ask the compiler where its static archives live
execute_process(
        COMMAND ${CMAKE_C_COMPILER}   -print-file-name libasan.a
        OUTPUT_VARIABLE ASAN_A
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${CMAKE_C_COMPILER}   -print-file-name liblsan.a
        OUTPUT_VARIABLE LSAN_A
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${CMAKE_C_COMPILER}   -print-file-name libubsan.a
        OUTPUT_VARIABLE UBSAN_A
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name libstdc++.a
        OUTPUT_VARIABLE STDCXX_A
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 3) Make sure those files actually exist
foreach(_archive IN LISTS ASAN_A LSAN_A UBSAN_A STDCXX_A)
    if(NOT EXISTS "${_archive}")
        message(FATAL_ERROR "Static archive not found: ${_archive}")
    endif()
endforeach()

# 4) Apply -fsanitize=… to every C and C++ compile
add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:${SAN_FLAGS}>
        $<$<COMPILE_LANGUAGE:C>:  ${SAN_FLAGS}>
)

# 5) Build the link flags that:
#    - carry over the same instrumentation
#    - wrap the four .a archives in --whole-archive so no symbol gets left as a .so dep
set(STATIC_ARCHIVES
        ${ASAN_A}
        ${LSAN_A}
        ${UBSAN_A}
        ${STDCXX_A}
)

add_link_options(
        # instrumentation during link:
        $<$<LINK_LANGUAGE:CXX>:${SAN_FLAGS}>
        $<$<LINK_LANGUAGE:C>:  ${SAN_FLAGS}>

        # now force the archives in:
        $<$<LINK_LANGUAGE:CXX>:-Wl,--whole-archive>
        ${STATIC_ARCHIVES}
        $<$<LINK_LANGUAGE:CXX>:-Wl,--no-whole-archive>

        $<$<LINK_LANGUAGE:C>:-Wl,--whole-archive>
        ${STATIC_ARCHIVES}
        $<$<LINK_LANGUAGE:C>:-Wl,--no-whole-archive>
)

# ───── END SANITIZER + STATIC-RUNTIME SETUP ─────────────────────────────────

function(add_sanitized_gtest target_name test_file)
    add_executable(${target_name} ${test_file})
    target_link_libraries(${target_name} PRIVATE gtest_main)
    add_test(NAME ${target_name} COMMAND ${target_name})

    # sanitized test
    if (PCCL_SANITIZE_TESTS)
        if (APPLE)
            target_link_options(${target_name} PRIVATE -fsanitize=address -fsanitize=undefined)
            target_compile_options(${target_name} PRIVATE -fsanitize=address -fsanitize=undefined)
        else ()
            target_link_options(${target_name} PRIVATE -fsanitize=address -fsanitize=leak -fsanitize=undefined)
            target_compile_options(${target_name} PRIVATE -fsanitize=address -fsanitize=leak -fsanitize=undefined)
        endif ()
    endif ()
endfunction()

function(add_unsanitized_gtest target_name test_file)
    add_executable(${target_name} ${test_file})
    target_link_libraries(${target_name} PRIVATE gtest_main)
    add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()