#include "pccl/common/fmemcpy.hpp"

#include <iostream>
#include <cstdint>

#ifdef _MSC_VER
  // MSVC doesn't support __builtin_assume_aligned, so we provide our own fallback.
  #define ASSUME_ALIGNED(ptr, alignment) (ptr)
#else
#define ASSUME_ALIGNED(ptr, alignment) \
static_cast<decltype(ptr)>(__builtin_assume_aligned((ptr), (alignment)))
#endif

#define release_assert(condition) if (!(condition)) { std::cerr << "Condition " << #condition << " failed!"; std::abort(); }

void fast_memcpy(void *dst, const void *src, const size_t length) {
    // check for pointer overlap
    {
        const auto *src_beg = src;
        const auto *src_end = src_beg + length;
        const auto *dst_beg = dst;
        const auto *dst_end = dst_beg + length;
        const bool overlap = !((src_end <= dst_beg) || (dst_end <= src_beg));
        release_assert(!overlap && "src and dst pointers do overlap!");
    }
    if (reinterpret_cast<uintptr_t>(dst) % 32 == 0 && reinterpret_cast<uintptr_t>(src) % 32 == 0) {
        // Let the compiler know these pointers are 32-byte aligned
        auto *aligned_dst = ASSUME_ALIGNED(static_cast<uint64_t*>(dst), 32);
        auto *aligned_src = ASSUME_ALIGNED(static_cast<const uint64_t*>(src), 32);
        const size_t n_words = length / sizeof(uint64_t);

        for (size_t i = 0; i < n_words; i++) {
            aligned_dst[i] = aligned_src[i];
        }
    } else {
        auto *dst_uint8s = static_cast<uint8_t *>(dst);
        const auto src_uint8s = static_cast<const uint8_t *>(src);

        for (size_t i = 0; i < length; i++) {
            dst_uint8s[i] = src_uint8s[i];
        }
    }
}
