option(PCCL_SANITIZE_TESTS "Enable address sanitizer for tests" OFF)

if (DEFINED $ENV{IS_CI})
    message(STATUS "Running in CI, enabling sanitizers in tests")
endif ()

# BEGIN NUCLEAR BULLSHIT

# 1) Instrumentation flags for the compile stage
set(SAN_FLAGS
        -fsanitize=address
        -fsanitize=leak
        -fsanitize=undefined
)

# 2) Locate the static archives (adjust paths if your distro is different)
find_library(ASAN_A asan   PATHS /usr/lib/gcc/* NO_DEFAULT_PATH)
find_library(LSAN_A lsan   PATHS /usr/lib/gcc/* NO_DEFAULT_PATH)
find_library(UBSAN_A ubsan PATHS /usr/lib/gcc/* NO_DEFAULT_PATH)
find_library(STDCXX_A stdc++ PATHS /usr/lib/gcc/* NO_DEFAULT_PATH)

if(NOT ASAN_A OR NOT LSAN_A OR NOT UBSAN_A OR NOT STDCXX_A)
    message(FATAL_ERROR "Couldn’t find one of: libasan.a, liblsan.a, libubsan.a, libstdc++.a")
endif()

# 3) Apply the compiler flags everywhere
add_compile_options(${SAN_FLAGS})

# 4) Push the same flags + force‐archive into *all* shared & module links
#    --whole-archive/--no-whole-archive makes sure every single object in those .a’s
#    is sucked into *your* .so
set(ALL_SHARED_LINK_FLAGS
        ${SAN_FLAGS}
        -Wl,--whole-archive
        ${ASAN_A}
        ${LSAN_A}
        ${UBSAN_A}
        ${STDCXX_A}
        -Wl,--no-whole-archive
)
# for executables (if you care):
set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} ${ALL_SHARED_LINK_FLAGS}"
)
# for shared libraries:
set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} ${ALL_SHARED_LINK_FLAGS}"
)
# for MODULE libraries (e.g. Python extensions):
set(CMAKE_MODULE_LINKER_FLAGS
        "${CMAKE_MODULE_LINKER_FLAGS} ${ALL_SHARED_LINK_FLAGS}"
)

# END OF NUCLEAR BULLSHIT

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