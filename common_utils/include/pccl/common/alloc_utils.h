#pragma once

#ifndef __APPLE__
#include <malloc.h>
#endif

#include <stddef.h>

inline void *do_aligned_alloc(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}