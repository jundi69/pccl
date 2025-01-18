#pragma once

#include <cassert>
#include <span>
#include <type_traits>
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

        template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> > >
        [[nodiscard]]
        static DeQuantizationMetaData Make(T min_value, T max_value) {
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
};
