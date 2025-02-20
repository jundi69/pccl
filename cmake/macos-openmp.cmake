# Extrawirschtl for OpenMP on macos to avoid collision with pytorch-loaded libomp.dylib

function(fix_macos_abspaths target)
    if(NOT APPLE)
        return()  # Do nothing on non-macOS platforms
    endif()

    # The idea: we'll look at the target's *link libraries* variable(s).
    # Usually, find_package(OpenMP) sets OpenMP_CXX_LIBRARIES or similar.
    # We'll gather them all and rewrite them if they are actual files.

    set(_candidate_paths
            "${OpenMP_CXX_LIBRARIES}"
    )

    # 1) Build up a list of -change arguments for install_name_tool
    set(_INSTALL_NAME_CHANGE_ARGS "")
    foreach(item IN LISTS _candidate_paths)
        # If the item is an actual file on disk, we assume it's an absolute path to a .dylib
        if(EXISTS "${item}")
            # Extract just the filename (e.g. "libomp.dylib")
            get_filename_component(_lib_name "${item}" NAME)

            # We'll run a small script to get the 'LC_ID_DYLIB' from that file.
            execute_process(
                    COMMAND otool -D "${item}"
                    OUTPUT_VARIABLE _otool_output
                    OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            string(REPLACE "\n" ";" _otool_output_list "${_otool_output}")
            list(GET _otool_output_list -1 _dylib_id)

            message(STATUS "LC_ID_DYLIB for libomp is: ${_dylib_id}")

            # We'll rewrite e.g. "/opt/homebrew/.../libomp.dylib" -> "@rpath/libomp.dylib"
            list(APPEND _INSTALL_NAME_CHANGE_ARGS
                    "-change"
                    "${_dylib_id}"
                    "@rpath/${_lib_name}"
            )
        endif()
    endforeach()

    # 2) If we found any absolute libs, add a custom command to rewrite them all in one go.
    if(_INSTALL_NAME_CHANGE_ARGS)
        message(STATUS "Fix absolute library references in target '${target}' -> @rpath/<filename>; ${_INSTALL_NAME_CHANGE_ARGS}")
        add_custom_command(
                TARGET "${target}"
                POST_BUILD
                COMMAND install_name_tool
                ${_INSTALL_NAME_CHANGE_ARGS}
                "$<TARGET_FILE:${target}>"
                COMMENT "Rewriting absolute install_name paths to @rpath for target '${target}'"
        )
    endif()
endfunction()
