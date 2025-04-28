option(PCCL_SANITIZE_TESTS "Enable address sanitizer for tests" OFF)

if (DEFINED $ENV{IS_CI})
    message(STATUS "Running in CI, enabling sanitizers in tests")
endif ()

# ─── BEGIN SANITIZER + STATIC-RUNTIME SETUP ───────────────────────────────────

# 1) Which sanitizers to turn on everywhere
set(SAN_FLAGS
        -fsanitize=address
        -fsanitize=leak
        -fsanitize=undefined
)

# 2) Ask the compiler where its .a files live
execute_process(
        COMMAND ${CMAKE_C_COMPILER}   -print-file-name=libasan.a
        OUTPUT_VARIABLE ASAN_A
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${CMAKE_C_COMPILER}   -print-file-name=liblsan.a
        OUTPUT_VARIABLE LSAN_A
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${CMAKE_C_COMPILER}   -print-file-name=libubsan.a
        OUTPUT_VARIABLE UBSAN_A
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.a
        OUTPUT_VARIABLE STDCXX_A
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 3) Make sure those files actually exist
set(STATIC_ARCHIVES
        ${ASAN_A}
        ${LSAN_A}
        ${UBSAN_A}
        ${STDCXX_A}
)
foreach(_archive IN ITEMS ${STATIC_ARCHIVES})
    if(NOT EXISTS "${_archive}")
        message(FATAL_ERROR "Static archive not found: ${_archive}")
    endif()
endforeach()

# 4) Turn on sanitizers at compile time for every target
add_compile_options(${SAN_FLAGS})

# 5) Build the flags that will force-archive all objects from those .a’s
set(ALL_SHARED_LINK_FLAGS
        ${SAN_FLAGS}
        -Wl,--whole-archive
        ${STATIC_ARCHIVES}
        -Wl,--no-whole-archive
)

# 6) Apply those flags to every kind of link
set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    ${ALL_SHARED_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ALL_SHARED_LINK_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${ALL_SHARED_LINK_FLAGS}")

# ───── END SANITIZER + STATIC-RUNTIME SETUP ───────────────────────────────────

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