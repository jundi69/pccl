option(PCCL_SANITIZE_TESTS "Enable address sanitizer for tests" OFF)

if (DEFINED $ENV{IS_CI})
    message(STATUS "Running in CI, enabling sanitizers in tests")
endif ()

# ─── STATIC-LINK ASAN/LSAN/UBSAN + libstdc++ ──────────────────────────────────

# 1) Where the compiler’s static sanitizer & C++ archives live:
set(ASAN_A    /usr/lib/gcc/x86_64-linux-gnu/11/libasan.a)
set(LSAN_A   /usr/lib/gcc/x86_64-linux-gnu/11/liblsan.a)
set(UBSAN_A  /usr/lib/gcc/x86_64-linux-gnu/11/libubsan.a)
set(STDCXX_A /usr/lib/gcc/x86_64-linux-gnu/11/libstdc++.a)

# 2) Turn on Address/Leak/UB sanitizers at compile time for all C & C++ code:
add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address -fsanitize=leak -fsanitize=undefined>
        $<$<COMPILE_LANGUAGE:C>:-fsanitize=address -fsanitize=leak -fsanitize=undefined>
)

# 3) Build the link flags that both enable sanitizers and force-archive the static libs:
set(SAN_LINK_FLAGS
        -fsanitize=address
        -fsanitize=leak
        -fsanitize=undefined
        -Wl,--whole-archive
        ${ASAN_A}
        ${LSAN_A}
        ${UBSAN_A}
        ${STDCXX_A}
        -Wl,--no-whole-archive
)

# 4) Apply those flags to every final link (executables, shared libs, MODULE libs):
set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    ${SAN_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SAN_LINK_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${SAN_LINK_FLAGS}")

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