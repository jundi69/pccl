option(PCCL_SANITIZE_TESTS "Enable address sanitizer for tests" OFF)

if (DEFINED $ENV{IS_CI})
    message(STATUS "Running in CI, enabling sanitizers in tests")
endif ()

# ─── BEGIN NUCLEAR SANITIZER SETUP ────────────────────────────────────────────

# 1) What sanitizers we want everywhere (C & C++ only)
set(SAN_FLAGS
        -fsanitize=address
        -fsanitize=leak
        -fsanitize=undefined
)

# 2) Discover the static .a archives from the compiler itself
foreach(_lib IN ITEMS libasan.a liblsan.a libubsan.a libstdc++.a)
    # pick C vs C++ based on name:
    if(_lib STREQUAL libstdc++.a)
        set(CMD "${CMAKE_CXX_COMPILER} -print-file-name=${_lib}")
    else()
        set(CMD "${CMAKE_C_COMPILER} -print-file-name=${_lib}")
    endif()
    execute_process(
            COMMAND ${CMD}
            OUTPUT_VARIABLE "${_lib}_PATH"
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # sanity-check it’s real
    if(NOT EXISTS "${${_lib}_PATH}")
        message(FATAL_ERROR "Static archive not found: ${_lib}_PATH=\"${${_lib}_PATH}\"")
    endif()
endforeach()

# 3) Collect them
set(STATIC_SANITIZER_ARCHIVES
        ${libasan.a_PATH}
        ${liblsan.a_PATH}
        ${libubsan.a_PATH}
        ${libstdc++.a_PATH}
)

# 4) Turn on -fsanitize=… for *compile* but only in C/C++ (never for CUDA)
add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:${SAN_FLAGS}>
        $<$<COMPILE_LANGUAGE:C>:  ${SAN_FLAGS}>
)

# 5) Globally inject the same flags into the *link* line for C/C++ targets,
#    and wrap the four .a archives in --whole-archive so nothing is left as a .so dep.
add_link_options(
        # instrumentation:
        $<$<LINK_LANGUAGE:CXX>:${SAN_FLAGS}>
        $<$<LINK_LANGUAGE:C>:  ${SAN_FLAGS}>

        # now force in all objects from the static runtimes:
        $<$<LINK_LANGUAGE:CXX>:-Wl,--whole-archive>
        $<$<LINK_LANGUAGE:CXX>:${STATIC_SANITIZER_ARCHIVES}>
        $<$<LINK_LANGUAGE:CXX>:-Wl,--no-whole-archive>

        $<$<LINK_LANGUAGE:C>:-Wl,--whole-archive>
        $<$<LINK_LANGUAGE:C>:${STATIC_SANITIZER_ARCHIVES}>
        $<$<LINK_LANGUAGE:C>:-Wl,--no-whole-archive>
)

# ───── END NUCLEAR SANITIZER SETUP ─────────────────────────────────────────────

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