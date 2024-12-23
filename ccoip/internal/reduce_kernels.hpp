#pragma once

#include <span>
#include <cstddef>
#include <ccoip_types.hpp>

namespace ccoip {
    void performReduction(std::span<std::byte> &span, const std::span<const std::byte> &data,
                          ccoip_data_type_t data_type, ccoip_reduce_op_t op);
};
