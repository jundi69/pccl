#include "hash_utils.hpp"

// FNV-1a with 512-bit chunks
// Motivation: FNV-1a is a fast, non-cryptographic hash function that is suitable for hash tables.
// However, the strict sequential nature of the algorithm makes it difficult for instruction level parallelism
// and SIMD parallelism to be exploited. By combining 8x64-bit hashes at a time, we can achieve better
// parallelism and throughput.

uint64_t ccoip::hash_utils::FVN1a_512Hash(const void *data, const size_t size) {
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
        for (size_t j = 0; j < 8; j++) {
            // NOLINT(*-loop-convert)
            hash ^= local_hashes[j];
            hash *= 0x100000001b3;
        }
    }
    return hash;
}


constexpr uint64_t INITIAL_OFFSET = 0xcbf29ce484222325ULL;
constexpr uint64_t PRIME = 0x100000001b3ULL;

static inline uint64_t mix_hash(uint64_t h, uint64_t v) {
    h ^= v;
    h = (h << 31) | (h >> (64 - 31));
    h += 0x9e3779b97f4a7c16ULL;
    return h;
}

uint64_t ccoip::hash_utils::FVN1a_512HashAccel(const void *data, const size_t size) noexcept {
    // Assume size is a multiple of 512 for simplicity
    const auto * __restrict words = static_cast<const uint64_t *>(data);
    const size_t num_words = size / 8;

    constexpr size_t block_size = 64; // process 64 words at a time
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i + block_size - 1 < num_words; i += block_size) {
        constexpr size_t lanes = 8;
        uint64_t lane_hash[lanes];
        for (int l = 0; l < lanes; ++l) { // NOLINT(*-loop-convert)
            lane_hash[l] = 0xcbf29ce484222325ULL;
        }

        // Each lane processes 8 words, total 64 words.
#pragma unroll (8)
        for (int step = 0; step < 8; ++step) {
            const size_t offset = i + step * lanes;
            // Load 8 words in parallel
            const uint64_t w0 = words[offset + 0];
            const uint64_t w1 = words[offset + 1];
            const uint64_t w2 = words[offset + 2];
            const uint64_t w3 = words[offset + 3];
            const uint64_t w4 = words[offset + 4];
            const uint64_t w5 = words[offset + 5];
            const uint64_t w6 = words[offset + 6];
            const uint64_t w7 = words[offset + 7];

            // Mix each lane
            lane_hash[0] = mix_hash(lane_hash[0], w0);
            lane_hash[1] = mix_hash(lane_hash[1], w1);
            lane_hash[2] = mix_hash(lane_hash[2], w2);
            lane_hash[3] = mix_hash(lane_hash[3], w3);
            lane_hash[4] = mix_hash(lane_hash[4], w4);
            lane_hash[5] = mix_hash(lane_hash[5], w5);
            lane_hash[6] = mix_hash(lane_hash[6], w6);
            lane_hash[7] = mix_hash(lane_hash[7], w7);
        }

        // Combine after processing the entire block
        // This reduces per-iteration dependency
        for (int l = 0; l < lanes; ++l) { // NOLINT(*-loop-convert)
            hash = mix_hash(hash, lane_hash[l]);
        }
    }

    // Handle remainder
    const size_t remainder = num_words % block_size;
    const size_t start = num_words - remainder;
    for (size_t r = start; r < num_words; ++r) {
        hash = mix_hash(hash, words[r]);
    }

    return hash;
}
