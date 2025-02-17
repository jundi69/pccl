#include <cstdint>
#include <cstddef>

#if defined(_MSC_VER)
// For Windows: intrin.h provides __cpuid(), _mm_crc32_u8, etc.
#include <intrin.h>
#else
#if defined(__aarch64__)
// For __crc32cb, __crc32cd, etc. (arm_acle.h) and NEON:
#include <arm_neon.h>
#include <arm_acle.h>
#endif
#endif

//------------------- ARMv8 PMULL + CRC32 Implementation -------------
#if defined(__aarch64__)
[[nodiscard]] uint32_t CRC32_armv8(const void *data, size_t size)
{
    if (!data || !size) [[unlikely]] return 0;

    const auto* buf = static_cast<const std::uint8_t*>(data);

    // PMULL + CRC inline assembly approach
    auto clmul_lo_e = [](uint64x2_t a, uint64x2_t b, uint64x2_t c) noexcept -> uint64x2_t {
        uint64x2_t r;
        __asm__ __volatile__(
            "pmull  %0.1q, %2.1d, %3.1d\n"
            "eor    %0.16b, %0.16b, %1.16b\n"
            : "=w"(r), "+w"(c) : "w"(a), "w"(b)
        );
        return r;
    };
    auto clmul_hi_e = [](uint64x2_t a, uint64x2_t b, uint64x2_t c) noexcept -> uint64x2_t {
        uint64x2_t r;
        __asm__ __volatile__(
            "pmull2 %0.1q, %2.2d, %3.2d\n"
            "eor    %0.16b, %0.16b, %1.16b\n"
            : "=w"(r), "+w"(c) : "w"(a), "w"(b)
        );
        return r;
    };

    std::uint32_t crc = ~0u;

    // Align to 8 if possible
    while (size && ((reinterpret_cast<std::uintptr_t>(buf) & 7) != 0)) {
        crc = __crc32cb(crc, *buf++);
        --size;
    }
    // possibly consume one 8-byte chunk
    if (((reinterpret_cast<std::uintptr_t>(buf) & 8) != 0) && (size >= 8)) {
        crc = __crc32cd(crc, *reinterpret_cast<const std::uint64_t*>(buf));
        buf += 8;
        size -= 8;
    }

    // Bulk PMULL logic
    if (size >= 192) {
        uint64x2_t x0 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf)), y0;
        uint64x2_t x1 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+16)), y1;
        uint64x2_t x2 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+32)), y2;
        uint64x2_t x3 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+48)), y3;
        uint64x2_t x4 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+64)), y4;
        uint64x2_t x5 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+80)), y5;
        uint64x2_t x6 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+96)), y6;
        uint64x2_t x7 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+112)), y7;
        uint64x2_t x8 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+128)), y8;
        uint64x2_t x9 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+144)), y9;
        uint64x2_t x10 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+160)), y10;
        uint64x2_t x11 = vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+176)), y11;
        uint64x2_t k;
        {
            static constexpr uint64_t k_ alignas(16)[] = {0xa87ab8a8, 0xab7aff2a};
            k = vld1q_u64(k_);
        }
        x0 = veorq_u64( (uint64x2_t){crc, 0}, x0);
        buf += 192;
        size -= 192;

        while (size >= 192) {
            y0  = clmul_lo_e(x0, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf)));
            x0  = clmul_hi_e(x0, k, y0);
            y1  = clmul_lo_e(x1, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+16)));
            x1  = clmul_hi_e(x1, k, y1);
            y2  = clmul_lo_e(x2, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+32)));
            x2  = clmul_hi_e(x2, k, y2);
            y3  = clmul_lo_e(x3, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+48)));
            x3  = clmul_hi_e(x3, k, y3);
            y4  = clmul_lo_e(x4, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+64)));
            x4  = clmul_hi_e(x4, k, y4);
            y5  = clmul_lo_e(x5, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+80)));
            x5  = clmul_hi_e(x5, k, y5);
            y6  = clmul_lo_e(x6, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+96)));
            x6  = clmul_hi_e(x6, k, y6);
            y7  = clmul_lo_e(x7, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+112)));
            x7  = clmul_hi_e(x7, k, y7);
            y8  = clmul_lo_e(x8, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+128)));
            x8  = clmul_hi_e(x8, k, y8);
            y9  = clmul_lo_e(x9, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+144)));
            x9  = clmul_hi_e(x9, k, y9);
            y10 = clmul_lo_e(x10, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+160)));
            x10 = clmul_hi_e(x10, k, y10);
            y11 = clmul_lo_e(x11, k, vld1q_u64(reinterpret_cast<const std::uint64_t*>(buf+176)));
            x11 = clmul_hi_e(x11, k, y11);

            buf += 192;
            size -= 192;
        }
        {
            static constexpr uint64_t k_ alignas(16)[] = {0xf20c0dfe, 0x493c7d27};
            k = vld1q_u64(k_);
        }
        y0  = clmul_lo_e(x0, k, x1);
        x0  = clmul_hi_e(x0, k, y0);
        y2  = clmul_lo_e(x2, k, x3);
        x2  = clmul_hi_e(x2, k, y2);
        y4  = clmul_lo_e(x4, k, x5);
        x4  = clmul_hi_e(x4, k, y4);
        y6  = clmul_lo_e(x6, k, x7);
        x6  = clmul_hi_e(x6, k, y6);
        y8  = clmul_lo_e(x8, k, x9);
        x8  = clmul_hi_e(x8, k, y8);
        y10 = clmul_lo_e(x10, k, x11);
        x10 = clmul_hi_e(x10, k, y10);

        {
            static constexpr uint64_t k_ alignas(16)[] = {0x3da6d0cb, 0xba4fc28e};
            k = vld1q_u64(k_);
        }
        y0  = clmul_lo_e(x0, k, x2);
        x0  = clmul_hi_e(x0, k, y0);
        y4  = clmul_lo_e(x4, k, x6);
        x4  = clmul_hi_e(x4, k, y4);
        y8  = clmul_lo_e(x8, k, x10);
        x8  = clmul_hi_e(x8, k, y8);

        {
            static constexpr uint64_t k_ alignas(16)[] = {0x740eef02, 0x9e4addf8};
            k = vld1q_u64(k_);
        }
        y0  = clmul_lo_e(x0, k, x4);
        x0  = clmul_hi_e(x0, k, y0);
        x4  = x8;
        y0  = clmul_lo_e(x0, k, x4);
        x0  = clmul_hi_e(x0, k, y0);

        crc = __crc32cd(0, vgetq_lane_u64(x0, 0));
        crc = __crc32cd(crc, vgetq_lane_u64(x0, 1));
    }

    // Handle remainder in 8-byte increments
    while (size >= 8) {
        crc = __crc32cd(crc, *reinterpret_cast<const std::uint64_t*>(buf));
        buf  += 8;
        size -= 8;
    }
    // Final leftover
    while (size > 0) {
        crc = __crc32cb(crc, *buf++);
        --size;
    }

    return ~crc;
}
#endif // __aarch64__
