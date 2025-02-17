#include <cstdint>

#if defined(_MSC_VER)
// For Windows: intrin.h provides __cpuid(), _mm_crc32_u8, etc.
#include <intrin.h>
#else
// For GCC/Clang on x86:
#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

//------------------- x86 SSE4.2 Implementation -------------
// Simple SSE4.2-only version (no PCLMUL):
#if defined(__x86_64__) || defined(_M_X64)
[[nodiscard]] uint32_t CRC32_x86_sse42(const void *data, size_t size) {
    if (!data || !size) [[unlikely]] return 0;
    const auto *buf = static_cast<const std::uint8_t *>(data);

    std::uint32_t crc = ~0u;
    // Align
    while (size && (reinterpret_cast<std::uintptr_t>(buf) & 7)) {
        crc = _mm_crc32_u8(crc, *buf++);
        --size;
    }
    // 8-byte chunks
    while (size >= 8) {
        crc = _mm_crc32_u64(crc, *reinterpret_cast<const std::uint64_t *>(buf));
        buf += 8;
        size -= 8;
    }
    // leftover
    while (size > 0) {
        crc = _mm_crc32_u8(crc, *buf++);
        --size;
    }
    return ~crc;
}
#endif