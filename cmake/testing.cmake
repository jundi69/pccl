option(PCCL_SANITIZE_TESTS "Enable address sanitizer for tests" OFF)

if (DEFINED $ENV{IS_CI})
    message(STATUS "Running in CI, enabling sanitizers in tests")
endif ()

# ─── STATIC-LINK ASAN + UBSAN + libstdc++ SETUP ─────────────────────────────

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # 1) Hard-coded static archives:
    set(ASAN_A    "/usr/lib/gcc/x86_64-linux-gnu/11/libasan.a")
    set(UBSAN_A  "/usr/lib/gcc/x86_64-linux-gnu/11/libubsan.a")
    set(STDCXX_A "/usr/lib/gcc/x86_64-linux-gnu/11/libstdc++.a")

    # 2) Sanity check:
    foreach(_ARCH ${ASAN_A} ${UBSAN_A} ${STDCXX_A})
        if (NOT EXISTS "${_ARCH}")
            message(FATAL_ERROR "Missing static archive: ${_ARCH}")
        endif()
    endforeach()

    # 3) Sanitizer flags (no leak sanitizer):
    set(SAN_FLAGS "-fsanitize=address -fsanitize=undefined")

    # 4) Apply to all C/C++ compiles only (CUDA untouched):
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${SAN_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SAN_FLAGS}")

    # 5) Build your static-link block:
    set(STATIC_LINK_FLAGS
            "-Wl,--whole-archive ${ASAN_A} ${UBSAN_A} ${STDCXX_A} -Wl,--no-whole-archive"
    )

    # 6) Inject into every link step:
    set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    ${SAN_FLAGS} ${STATIC_LINK_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SAN_FLAGS} ${STATIC_LINK_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${SAN_FLAGS} ${STATIC_LINK_FLAGS}")
endif()

# ───── END STATIC-LINK ASAN SETUP ─────────────────────────────────────────────

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