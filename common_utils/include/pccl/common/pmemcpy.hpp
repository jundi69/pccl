#pragma once

#include <cstddef>

#ifdef PCCL_BUILD_OPENMP_SUPPORT
#include <omp.h>
#endif

#ifndef _MSC_VER
#define FORCE_INLINE inline __attribute__((always_inline))
#else
#define FORCE_INLINE inline __forceinline
#endif

FORCE_INLINE void pmemcpy(void *dst, const void *src, const size_t length) {
#if defined(PCCL_BUILD_OPENMP_SUPPORT) and not defined(__APPLE__) // omp_target_memcpy is not supported under macOS
    const int device = omp_get_initial_device();
    omp_target_memcpy(dst, src, length, 0, 0, device, device);
#else
    std::memcpy(dst, src, length);
#endif
}
