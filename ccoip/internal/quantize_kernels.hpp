#pragma once

#include <span>
#include <type_traits>
#include <quantize.hpp>

#ifndef _MSC_VER
#define QUANTIZE_FORCE_INLINE inline __attribute__((always_inline))
#else
#define QUANTIZE_FORCE_INLINE inline __forceinline
#endif

namespace ccoip::internal::quantize {

    template<typename I, typename O, typename = std::enable_if_t<
        std::is_integral_v<I> && std::is_unsigned_v<I> && std::is_floating_point_v<O>> >
    [[nodiscard]]
    QUANTIZE_FORCE_INLINE O deQuantizeMinMaxScalar(I value, O min, O max) {
        O relative = static_cast<O>(value) / static_cast<O>(std::numeric_limits<I>::max());
        return min + relative * (max - min);
    }

    [[nodiscard]] DeQuantizationMetaData performMinMaxQuantization(const std::span<std::byte> &dst_span,
                                                                   const std::span<const std::byte> &src_bytes,
                                                                   ccoip_data_type_t quantized_type,
                                                                   ccoip_data_type_t data_type);

}; // namespace ccoip::internal::quantize

#undef QUANTIZE_FORCE_INLINE