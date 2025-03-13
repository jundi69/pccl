#pragma once

#ifndef __APPLE__
#include <malloc.h>
#endif

#include <cstddef>
#include <memory>
#include <stdexcept>

#define CEIL_TO_MULTIPLE(x, m) (((x) + (m) - 1) / (m) * (m))

inline void *do_aligned_alloc(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

inline void do_aligned_free(void *ptr) {
    free(ptr);
}

struct AlignedFreeDeleter {
    void operator()(void *ptr) const noexcept {
        do_aligned_free(ptr);
    }
};

template<typename T>
std::unique_ptr<T[], AlignedFreeDeleter>
do_aligned_alloc_unique(const size_t alignment, const size_t count) {
    const size_t numBytes = CEIL_TO_MULTIPLE(count * sizeof(T), alignment);
    // ReSharper disable once CppDFAMemoryLeak
    void *rawPtr = do_aligned_alloc(alignment, numBytes);
    if (!rawPtr) {
        throw std::bad_alloc();
    }
    return std::unique_ptr<T[], AlignedFreeDeleter>(static_cast<T *>(rawPtr));
}
