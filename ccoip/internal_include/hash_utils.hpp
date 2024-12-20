#pragma once

#include <cstdint>
#include <cstddef>

namespace ccoip::hash_utils {
    [[nodiscard]] uint32_t CRC32(const void *data, size_t size);
}
