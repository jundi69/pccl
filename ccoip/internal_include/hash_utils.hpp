#pragma once

#include <cstdint>
#include <cstddef>

namespace ccoip::hash_utils {
    [[nodiscard]] uint32_t CRC32(const void *data, size_t n_bytes);

    [[nodiscard]] uint32_t simplehash_cuda(const void *data, size_t n_bytes);

    [[nodiscard]] uint32_t simplehash_cpu(const void *data, size_t n_bytes);
}
