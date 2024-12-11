#pragma once

#include <cstdint>
#include <array>
#include <functional>
#include <string>

enum CCoIPDataType : uint8_t {
    CCOIP_DTYPE_FLOAT16 = 0x01,
    CCOIP_DTYPE_BFLOAT16 = 0x02,
    CCOIP_DTYPE_FLOAT32 = 0x03,
    CCOIP_DTYPE_FLOAT64 = 0x04,
    CCOIP_DTYPE_INT8 = 0x05,
    CCOIP_DTYPE_INT16 = 0x06,
    CCOIP_DTYPE_INT32 = 0x07,
    CCOIP_DTYPE_INT64 = 0x08,
    CCOIP_DTYPE_UINT8 = 0x09,
    CCOIP_DTYPE_UINT16 = 0x0A,
    CCOIP_DTYPE_UINT32 = 0x0B,
    CCOIP_DTYPE_UINT64 = 0x0C
};

constexpr std::size_t CCOIP_UUID_N_BYTES = 16;
using ccoip_uuid = std::array<uint8_t, CCOIP_UUID_N_BYTES>;

typedef uint8_t boolean;

struct ccoip_uuid_t
{
    ccoip_uuid data;

    friend bool operator==(const ccoip_uuid_t& lhs, const ccoip_uuid_t& rhs)
    {
        return lhs.data == rhs.data;
    }

    friend bool operator!=(const ccoip_uuid_t& lhs, const ccoip_uuid_t& rhs)
    {
        return !(lhs == rhs);
    }
};
static_assert(sizeof(ccoip_uuid_t) == 16);

// Custom hash specialization for uuid_t
template <>
struct std::hash<ccoip_uuid_t>
{
    std::size_t operator()(const ccoip_uuid_t& uuid) const noexcept
    {
        // Combine the hash of each byte in the UUID array
        std::size_t hash_value = 0;
        for (const auto& byte : uuid.data)
        {
            hash_value = (hash_value * 31) + byte; // or use a better mixing function if desired
        }
        return hash_value;
    }
};

inline std::string uuid_to_string(const ccoip_uuid_t& uuid)
{
    std::string uuid_str;
    uuid_str.reserve(CCOIP_UUID_N_BYTES * 2 + 4);
    for (const auto& byte : uuid.data)
    {
        char byte_str[3];
        snprintf(byte_str, sizeof(byte_str), "%02X", byte);
        uuid_str += byte_str;
        if (uuid_str.size() == 8 || uuid_str.size() == 13 || uuid_str.size() == 18 || uuid_str.size() == 23)
        {
            uuid_str += "-";
        }
    }
    return uuid_str;
}
