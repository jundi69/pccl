#pragma once

#include <cstdint>
#include <cstddef>

namespace ccoip::hash_utils {
    uint64_t FVN1a_512Hash(const void *data, size_t size);
    uint64_t FVN1a_512HashAccel(const void *data, size_t size) noexcept;
}
