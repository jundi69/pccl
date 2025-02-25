#pragma once

#include <cassert>
#include <span>
#include <type_traits>
#include <vector>
#include <cstring>
#include <ccoip_types.hpp>
#include <network_order_utils.hpp>

namespace ccoip::internal::quantize {
    /// Contains meta-data needed for de-quantization.
    struct DeQuantizationMetaData {
        /// Defines the data type for how to interpret @code min_value@endcode and @code max_value@endcode.
        ccoip_data_type_t data_type;

        /// Minimum value summary statistic
        std::vector<uint8_t> min_value;

        /// Maximum value summary statistic
        std::vector<uint8_t> max_value;

        template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        [[nodiscard]]
        static DeQuantizationMetaData Make(T min_value, T max_value) {
            using namespace tinysockets;
            constexpr ccoip_data_type_t data_type = ccoip_data_type_from_type<T>();

            DeQuantizationMetaData meta_data{};
            meta_data.data_type = data_type;
            const size_t data_type_size = ccoip_data_type_size(data_type);
            assert(data_type_size == sizeof(T));

            if constexpr (std::is_integral_v<T>) {
                // if is integer, convert to network byte order
                min_value = network_order_utils::host_to_network(min_value);
                max_value = network_order_utils::host_to_network(max_value);
            }

            // floats and doubles are always in host byte order

            meta_data.min_value.resize(data_type_size);
            meta_data.max_value.resize(data_type_size);

            std::memcpy(meta_data.min_value.data(), &min_value, data_type_size);
            std::memcpy(meta_data.max_value.data(), &max_value, data_type_size);

            return meta_data;
        }
    };

    [[nodiscard]] DeQuantizationMetaData performQuantization(const std::span<std::byte> &dst_span,
                                                             const std::span<const std::byte> &src_span,
                                                             ccoip_quantization_algorithm_t quantization_algorithm,
                                                             ccoip_data_type_t quantized_type,
                                                             ccoip_data_type_t data_type);

    // This method is needed because we need a fast way to pretend we quantized and de-quantized.
    // The reason for this is that to ensure the result is the same for all peers, we need to
    // pretend the peer that has access to the original un-quantized data also just receives
    // the same quantized data as other peers. For this reason, we need to quantize and de-quantize locally.
    // However, we want to just do it fused and fast, hence this method.
    void performQuantizationAndDequantization(const std::span<std::byte> &dst_span,
                                              const std::span<const std::byte> &src_span,
                                              ccoip_quantization_algorithm_t quantization_algorithm,
                                              ccoip_data_type_t quantized_type,
                                              ccoip_data_type_t data_type);
};
