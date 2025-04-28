option(PCCL_SANITIZE_TESTS "Enable address sanitizer for tests" OFF)

if (DEFINED $ENV{IS_CI})
    message(STATUS "Running in CI, enabling sanitizers in tests")
endif ()

# ─── STATIC-LINK ASAN/LSAN/UBSAN + libstdc++ SETUP ────────────────────────────

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # 1) Hard-coded static archive paths:
    set(ASAN_A    "/usr/lib/gcc/x86_64-linux-gnu/11/libasan.a")
    set(LSAN_A   "/usr/lib/gcc/x86_64-linux-gnu/11/liblsan.a")
    set(UBSAN_A  "/usr/lib/gcc/x86_64-linux-gnu/11/libubsan.a")
    set(STDCXX_A "/usr/lib/gcc/x86_64-linux-gnu/11/libstdc++.a")

    # 2) Verify they exist:
    foreach(_ARCH ${ASAN_A} ${LSAN_A} ${UBSAN_A} ${STDCXX_A})
        if (NOT EXISTS "${_ARCH}")
            message(FATAL_ERROR "Missing static archive: ${_ARCH}")
        endif()
    endforeach()

    # 3) Your sanitizer flags as a list:
    set(SAN_FLAGS_LIST
            -fsanitize=address
            -fsanitize=leak
            -fsanitize=undefined
    )

    # 4) Apply sanitizers *only* to C/C++ compile steps (CUDA untouched):
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:${SAN_FLAGS_LIST}>
            $<$<COMPILE_LANGUAGE:C>:  ${SAN_FLAGS_LIST}>
    )

    # 5) Join that list into a single string:
    list(JOIN SAN_FLAGS_LIST " " SAN_FLAGS_STR)

    # 6) Build the whole-archive block:
    set(STATIC_ARCHIVES
            ${ASAN_A}
            ${LSAN_A}
            ${UBSAN_A}
            ${STDCXX_A}
    )
    list(JOIN STATIC_ARCHIVES " " STATIC_ARCHIVES_STR)
    set(WHOLE_ARCHIVE_FLAGS
            "-Wl,--whole-archive ${STATIC_ARCHIVES_STR} -Wl,--no-whole-archive"
    )

    # 7) Combine sanitize+whole-archive into one string:
    set(SAN_LINK_FLAGS "${SAN_FLAGS_STR} ${WHOLE_ARCHIVE_FLAGS}")

    # 8) Inject into every link type:
    set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    ${SAN_LINK_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SAN_LINK_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${SAN_LINK_FLAGS}")
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