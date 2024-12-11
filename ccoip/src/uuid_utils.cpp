#include "uuid_utils.hpp"

#include <ccoip_types.hpp>
#include <random>

std::random_device rd;
std::mt19937_64 generator(rd());
std::uniform_int_distribution<uint32_t> dist(0, 255);

void ccoip::uuid_utils::generate_uuid(ccoip_uuid &uuid_out) {
    // Generate 16 random bytes
    for (int i = 0; i < 16; ++i) {
        uuid_out[i] = static_cast<uint8_t>(dist(generator));
    }

    // Set version to 4 (UUIDv4)
    uuid_out[6] = (uuid_out[6] & 0x0F) | 0x40;
    // Set variant to 2 (RFC 4122 variant)
    uuid_out[8] = (uuid_out[8] & 0x3F) | 0x80;
}
