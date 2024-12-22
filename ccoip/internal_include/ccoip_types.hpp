#pragma once

#include <cstdint>
#include <array>
#include <string>

namespace ccoip {
    enum ccoip_data_type_t {
        ccoipUint8 = 0,
        ccoipInt8 = 1,
        ccoipUint16 = 2,
        ccoipUint32 = 3,
        ccoipInt16 = 4,
        ccoipInt32 = 5,
        ccoipUint64 = 6,
        ccoipInt64 = 7,
        ccoipFloat = 8,
        ccoipDouble = 9,
    };

    enum ccoip_reduce_op_t {
        ccoipOpSum = 0,
        ccoipOpAvg = 1,
        ccoipOpProd = 2,
        ccoipOpMax = 3,
        ccoipOpMin = 4
    };


    inline size_t ccoip_data_type_size(const ccoip_data_type_t datatype) {
        switch (datatype) {
            case ccoipUint8:
            case ccoipInt8:
                return 1;
            case ccoipUint16:
            case ccoipInt16:
                return 2;
            case ccoipUint32:
            case ccoipInt32:
            case ccoipFloat:
                return 4;
            case ccoipUint64:
            case ccoipInt64:
            case ccoipDouble:
                return 8;
        }
        return 0;
    }
};


constexpr std::size_t CCOIP_UUID_N_BYTES = 16;
using ccoip_uuid = std::array<uint8_t, CCOIP_UUID_N_BYTES>;

typedef uint8_t boolean;

struct ccoip_uuid_t {
    ccoip_uuid data;

    friend bool operator==(const ccoip_uuid_t &lhs, const ccoip_uuid_t &rhs) {
        return lhs.data == rhs.data;
    }

    friend bool operator!=(const ccoip_uuid_t &lhs, const ccoip_uuid_t &rhs) {
        return !(lhs == rhs);
    }
};

static_assert(sizeof(ccoip_uuid_t) == 16);

// Custom hash specialization for uuid_t
template<>
struct std::hash<ccoip_uuid_t> {
    std::size_t operator()(const ccoip_uuid_t &uuid) const noexcept {
        // Combine the hash of each byte in the UUID array
        std::size_t hash_value = 0;
        for (const auto &byte: uuid.data) {
            hash_value = (hash_value * 31) + byte; // or use a better mixing function if desired
        }
        return hash_value;
    }
};

inline std::string uuid_to_string(const ccoip_uuid_t &uuid) {
    std::string uuid_str;
    uuid_str.reserve(CCOIP_UUID_N_BYTES * 2 + 4);
    for (const auto &byte: uuid.data) {
        char byte_str[3];
        snprintf(byte_str, sizeof(byte_str), "%02X", byte);
        uuid_str += byte_str;
        if (uuid_str.size() == 8 || uuid_str.size() == 13 || uuid_str.size() == 18 || uuid_str.size() == 23) {
            uuid_str += "-";
        }
    }
    return uuid_str;
}
