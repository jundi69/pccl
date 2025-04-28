option(PCCL_SANITIZE_TESTS "Enable address sanitizer for tests" OFF)

if (DEFINED $ENV{IS_CI})
    message(STATUS "Running in CI, enabling sanitizers in tests")
endif ()

# ─── STATIC-LINK ASAN + libstdc++ SETUP ──────────────────────────────────────

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # 1) Hard-coded static archives you found:
    set(ASAN_A    "/usr/lib/gcc/x86_64-linux-gnu/11/libasan.a")
    set(STDCXX_A "/usr/lib/gcc/x86_64-linux-gnu/11/libstdc++.a")

    # 2) Sanity-check they exist:
    foreach(_ARCH ${ASAN_A} ${STDCXX_A})
        if (NOT EXISTS "${_ARCH}")
            message(FATAL_ERROR "Missing static archive: ${_ARCH}")
        endif()
    endforeach()

    # 3) AddressSanitizer compile flag only:
    set(ASAN_FLAG "-fsanitize=address")

    # 4) Apply to all C & C++ compilation steps (CUDA untouched):
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${ASAN_FLAG}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAG}")

    # 5) Build the --whole-archive block for linking:
    set(STATIC_LINK_ARCHIVES
            ${ASAN_A}
            ${STDCXX_A}
    )
    # join into one space-separated string:
    list(JOIN STATIC_LINK_ARCHIVES " " STATIC_ARCHIVES_STR)
    set(WHOLE_ARCHIVE_FLAGS
            "-Wl,--whole-archive ${STATIC_ARCHIVES_STR} -Wl,--no-whole-archive"
    )

    # 6) Combine ASan + static-archive into one link-flag string:
    set(ASAN_LINK_FLAGS "${ASAN_FLAG} ${WHOLE_ARCHIVE_FLAGS}")

    # 7) Inject into every final link (exe, shared, MODULE):
    set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    ${ASAN_LINK_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ASAN_LINK_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${ASAN_LINK_FLAGS}")
endif()

# ───── END STATIC-LINK ASAN SETUP ────────────────────────────────────────────

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