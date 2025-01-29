#pragma once

#include <type_traits>

#if __has_include(<bit>)
#include <bit>
#endif

#ifdef _MSC_VER
#include <intrin.h> // for _byteswap_ushort, _byteswap_ulong, _byteswap_uint64
#endif

namespace ccoip::internal::network_order_utils {
    //------------------------------------------------------------------
    // Shared "swap bytes" helper that always reverses the byte order
    // for 2-, 4-, or 8-byte integrals, regardless of endianness.
    //------------------------------------------------------------------
    template<typename T>
    constexpr T bswap(T val) noexcept {
        static_assert(std::is_integral_v<T>, "bswap only valid for integral types");
        using U = std::make_unsigned_t<T>;
        U uval = static_cast<U>(val);

        if constexpr (sizeof(T) == 1) {
            // 1-byte: nothing to swap
            return val;
        } else if constexpr (sizeof(T) == 2) {
#if defined(__GNUC__) || defined(__clang__)
            uval = __builtin_bswap16(uval);
#elif defined(_MSC_VER)
        uval = _byteswap_ushort(uval);
#else
        // Fallback: manual swap
        uval = static_cast<U>(((uval & 0x00FFU) << 8) |
                              ((uval & 0xFF00U) >> 8));
#endif
        } else if constexpr (sizeof(T) == 4) {
#if defined(__GNUC__) || defined(__clang__)
            uval = __builtin_bswap32(uval);
#elif defined(_MSC_VER)
        uval = _byteswap_ulong(uval);
#else
        // Fallback: manual swap
        uval = static_cast<U>(((uval & 0x000000FFUL) << 24) |
                              ((uval & 0x0000FF00UL) <<  8) |
                              ((uval & 0x00FF0000UL) >>  8) |
                              ((uval & 0xFF000000UL) >> 24));
#endif
        } else if constexpr (sizeof(T) == 8) {
#if defined(__GNUC__) || defined(__clang__)
            uval = __builtin_bswap64(uval);
#elif defined(_MSC_VER)
        uval = _byteswap_uint64(uval);
#else
        // Fallback: manual swap
        uval = static_cast<U>(
            ((uval & 0x00000000000000FFULL) << 56) |
            ((uval & 0x000000000000FF00ULL) << 40) |
            ((uval & 0x0000000000FF0000ULL) << 24) |
            ((uval & 0x00000000FF000000ULL) <<  8) |
            ((uval & 0x000000FF00000000ULL) >>  8) |
            ((uval & 0x0000FF0000000000ULL) >> 24) |
            ((uval & 0x00FF000000000000ULL) >> 40) |
            ((uval & 0xFF00000000000000ULL) >> 56));
#endif
        } else {
            static_assert(sizeof(T) <= 8,
                          "bswap not implemented for this integral size");
        }

        return static_cast<T>(uval);
    }

    //------------------------------------------------------------------
    // network_to_host and host_to_network, both rely on bswap if host
    // is little-endian, otherwise return unchanged if big-endian.
    //------------------------------------------------------------------
    template<typename T>
    constexpr T network_to_host(T val) noexcept {
        static_assert(std::is_integral_v<T>,
                      "network_to_host can only be used with integral types.");

#if defined(__cpp_lib_endian) && (__cpp_lib_endian >= 201907L)
        if constexpr (std::endian::native == std::endian::big) {
            // Host is big-endian => same as network => no swap
            return val;
        } else {
            // Host is little-endian => swap
            return bswap(val);
        }
#else
    // If we can't query endianness, assume little-endian if needed
    // or just unconditionally swap if you know your target is LE.
    return bswap(val);
#endif
    }

    template<typename T>
    constexpr T host_to_network(T val) noexcept {
        // It's exactly the same logic, just reversed in name.
        // On a little-endian system, we do the same bswap to get big-endian.
        return network_to_host(val);
    }
}; // namespace ccoip::internal::network_order_utils
