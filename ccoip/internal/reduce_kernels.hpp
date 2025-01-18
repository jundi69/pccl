#pragma once

#include <span>
#include <cstddef>
#include <ccoip_types.hpp>
#include <quantize.hpp>

namespace ccoip::internal::reduce {
    void performReduction(const std::span<std::byte> &dst, const std::span<const std::byte> &src,
                          ccoip_data_type_t dst_type, ccoip_data_type_t src_type,
                          ccoip_quantization_algorithm_t quantization_algorithm,
                          ccoip_reduce_op_t op,
                          const quantize::DeQuantizationMetaData &meta_data);

    void performReduceFinalization(const std::span<std::byte> &dst, ccoip_data_type_t accumulator_type,
                                   size_t world_size,
                                   ccoip_reduce_op_t op);
};
