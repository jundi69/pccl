#pragma once

#include <span>
#include <limits>
#include <type_traits>
#include <quantize.hpp>

#ifndef _MSC_VER
#define QUANTIZE_FORCE_INLINE inline __attribute__((always_inline))
#else
#define QUANTIZE_FORCE_INLINE inline __forceinline
#endif

namespace ccoip::internal::quantize {
    template<typename I, typename O, typename = std::enable_if_t<
        std::is_integral_v<I> && std::is_unsigned_v<I> && std::is_floating_point_v<O>>>
    [[nodiscard]]
    QUANTIZE_FORCE_INLINE O deQuantizeMinMaxScalar(I value, O min, O max) {
        const auto inv = 1.0 / static_cast<double>(std::numeric_limits<I>::max());
        O relative = static_cast<O>(static_cast<double>(value) * inv);
        return min + relative * (max - min);
    }

    [[nodiscard]] DeQuantizationMetaData performMinMaxQuantization(const std::span<std::byte> &dst_span,
                                                                   const std::span<const std::byte> &src_bytes,
                                                                   ccoip_data_type_t quantized_type,
                                                                   ccoip_data_type_t data_type);

    // This method is needed because we need a fast way to pretend we quantized and de-quantized.
    // The reason for this is that to ensure the result is the same for all peers, we need to
    // pretend the peer that has access to the original un-quantized data also just receives
    // the same quantized data as other peers. For this reason, we need to quantize and de-quantize locally.
    // However, we want to just do it fused and fast, hence this method.
    void performMinMaxQuantizationAndDequantization(const std::span<std::byte> &dst_span,
                                                    const std::span<const std::byte> &src_bytes,
                                                    ccoip_data_type_t quantized_type,
                                                    ccoip_data_type_t data_type);
}; // namespace ccoip::internal::quantize

#undef QUANTIZE_FORCE_INLINE
