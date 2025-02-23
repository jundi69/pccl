cmake_minimum_required(VERSION 3.16)
project(pccl)

include(cmake/macos-openmp.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# make sure we reference dynamic libraries such as OpenMP via RPATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_MACOSX_RPATH ON)

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@rpath")
elseif(UNIX)
    # For Linux, we often use $ORIGIN so it’s relative to the .so location.
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

# Detect CPU architecture
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(IS_AMD64 TRUE)
else ()
    set(IS_AMD64 FALSE)
endif ()
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(IS_ARM64 TRUE)
else ()
    set(IS_ARM64 FALSE)
endif ()

# Handle CUDA build
option(PCCL_BUILD_CUDA_SUPPORT "Build PCCL with CUDA support" ON)
option(PCCL_BUILD_OPENMP_SUPPORT "Build PCCL with OpenMP support" ON)

if (PCCL_BUILD_CUDA_SUPPORT)
    find_package(CUDAToolkit)

    # check if the cuda toolkit was found
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit was found!")
        # You can now use CUDA-specific targets and include directories.
    else()
        message(WARNING "CUDA Toolkit not found! CUDA support will be disabled!")
        set(PCCL_BUILD_CUDA_SUPPORT OFF)
    endif()
endif ()

if (PCCL_BUILD_OPENMP_SUPPORT)
    set(OpenMP_USE_STATIC_LIBS ON)  # Request static OpenMP
    find_package(OpenMP QUIET)

    if (OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP. Compiling with OpenMP support")
        set(PCCL_BUILD_OPENMP_SUPPORT ON)
    else()
        message(STATUS "OpenMP not found through find_package(OpenMP). Checking compiler flags manually...")

        # Manual check of compiler flags
        include(CheckCXXCompilerFlag)

        if (MSVC)
            # MSVC uses /openmp
            check_cxx_compiler_flag("/openmp" MSVC_OPENMP_SUPPORTED)
            if (MSVC_OPENMP_SUPPORTED)
                message(STATUS "OpenMP supported by MSVC. Enabling /openmp flag.")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
            else()
                message(STATUS "OpenMP is not supported by this version of MSVC.")
                set(PCCL_BUILD_OPENMP_SUPPORT OFF)
            endif()
        else()
            # GCC, Clang, etc. typically use -fopenmp
            check_cxx_compiler_flag("-fopenmp" GCC_CLANG_OPENMP_SUPPORTED)
            if (GCC_CLANG_OPENMP_SUPPORTED)
                message(STATUS "OpenMP supported by GCC/Clang. Enabling -fopenmp flag.")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
            else()
                message(STATUS "OpenMP is not supported by this compiler or version.")
                set(PCCL_BUILD_OPENMP_SUPPORT OFF)
            endif()
        endif()
    endif()
else()
    message(STATUS "PCCL_BUILD_OPENMP_SUPPORT is OFF. Skipping OpenMP checks.")
endif()

if (PCCL_BUILD_OPENMP_SUPPORT)
    add_compile_definitions(PCCL_BUILD_OPENMP_SUPPORT)
endif ()

if (PCCL_BUILD_CUDA_SUPPORT)
    set(CMAKE_CUDA_STANDARD 20)

    # target sm_50 as a lower bound. We override this in a target specific way when necessary.
    set(CMAKE_CUDA_ARCHITECTURES "50")

    # enable cuda language
    enable_language(CUDA)
endif ()

# add third party libraries
add_subdirectory(third_party)

enable_testing()

# include testing functions
include(cmake/testing.cmake)

# add internal libraries
add_subdirectory(common_utils)
add_subdirectory(log)
add_subdirectory(ccoip)
add_subdirectory(ccoip_master)
add_subdirectory(tinysockets)

option(PCCL_BUILD_STATIC_LIB "Build PCCL as a static library" ON)

if (PCCL_BUILD_STATIC_LIB)
    add_library(pccl STATIC src/pccl.cpp)
else ()
    add_library(pccl SHARED src/pccl.cpp)
endif ()

if (NOT WIN32)
    target_compile_options(pccl PRIVATE -Wall -Wextra -Werror -Wpedantic -Wno-unused-parameter)
endif ()

if (PCCL_BUILD_CUDA_SUPPORT)
    target_compile_definitions(pccl PRIVATE PCCL_HAS_CUDA_SUPPORT)
endif ()

target_include_directories(pccl PUBLIC include)
target_include_directories(pccl PRIVATE internal)

target_link_libraries(pccl PRIVATE ccoip_private)
target_link_libraries(pccl PUBLIC ccoip_public)
target_link_libraries(pccl PRIVATE pccl_log)

if (NOT PCCL_BUILD_STATIC_LIB)
    fix_macos_abspaths(pccl)
endif ()

add_subdirectory(tests)
