#include "hash_utils.hpp"

#include <cstring>

// FNV-1a with 512-bit chunks
// Motivation: FNV-1a is a fast, non-cryptographic hash function that is suitable for hash tables.
// However, the strict sequential nature of the algorithm makes it difficult for instruction level parallelism
// and SIMD parallelism to be exploited. By combining 8x64-bit hashes at a time, we can achieve better
// parallelism and throughput.

uint64_t FVN1a_512Hash(const void *data, const size_t size) {
    const auto words = static_cast<const uint64_t *>(data);
    uint64_t hash = 0xcbf29ce484222325;
    const size_t num_words = size / 8;
    for (size_t i = 0; i < num_words; i += 8) {
        uint64_t local_hashes[8]{};
        for (size_t k = 0; k < 8; k++) {
            for (size_t j = 0; j < 8; j++) {
                local_hashes[j] ^= words[i + j] >> k * 8 & 0xFF;
                local_hashes[j] *= 0x100000001b3;
            }
        }

        // Combine the local hashes
        for (size_t j = 0; j < 8; j++) { // NOLINT(*-loop-convert)
            hash ^= local_hashes[j];
            hash *= 0x100000001b3;
        }
    }
    return hash;
}
