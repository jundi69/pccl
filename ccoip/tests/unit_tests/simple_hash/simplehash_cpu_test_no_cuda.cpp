#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <hash_utils.hpp>

uint32_t next_uint32(uint32_t &seed, const uint32_t lo, const uint32_t hi) {
    seed = 1664525u * seed + 1013904223u;
    const uint64_t range = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo) + 1ULL;
    const uint32_t rndInRange = static_cast<uint32_t>(static_cast<uint64_t>(seed) % range) + lo;
    return rndInRange;
}

TEST(SimpleHashTest, BenchmarkAgainstBaseline) {
    // Use the same buffer size once used in the CUDA tests.
    constexpr size_t size = 154533888; // in bytes
    constexpr int n_repeat = 500;

    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    // init random
    {
        uint32_t seed = 42;
        for (uint32_t i = 0; i < size; i++) { // NOLINT(*-loop-convert)
            static_cast<uint8_t *>(host_buffer)[i] = next_uint32(seed, 0, 255);
        }
    }

    volatile uint64_t simple_hashes[n_repeat] = {0};

    // CPU baseline
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = ccoip::hash_utils::simplehash_cpu(host_buffer, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU: " << duration.count() << " us" << std::endl;
    double bandwidth = (static_cast<double>(n_repeat * size) / 1e9) / (static_cast<double>(duration.count()) / 1e6);
    std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Ensure all iterations produced exactly the same hash.
    for (int i = 1; i < n_repeat; ++i) {
        ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
    }

    // If you want to compare to a known reference, insert your updated ASSERT here:
    ASSERT_EQ(3391090508, simple_hashes[0]);

    free(host_buffer);
}

TEST(SimpleHashTest, TestSizeOneByte) {
    constexpr size_t size = 1;
    constexpr int n_repeat = 100;

    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    // init random
    {
        uint32_t seed = 42;
        for (uint32_t i = 0; i < size; i++) { // NOLINT(*-loop-convert)
            static_cast<uint8_t *>(host_buffer)[i] = next_uint32(seed, 0, 255);
        }
    }

    volatile uint64_t simple_hashes[n_repeat] = {0};

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = ccoip::hash_utils::simplehash_cpu(host_buffer, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU (1 byte): " << duration.count() << " us" << std::endl;
    double bandwidth = (static_cast<double>(n_repeat * size) / 1e9) / (static_cast<double>(duration.count()) / 1e6);
    std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // All iterations should produce the same hash for identical input.
    for (int i = 1; i < n_repeat; ++i) {
        ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
    }

    // Insert an updated reference if desired.
    ASSERT_EQ(344386053, simple_hashes[0]);

    free(host_buffer);
}

TEST(SimpleHashTest, TestSizeFourBytes) {
    constexpr size_t size = 4;
    constexpr int n_repeat = 100;

    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    // init random
    {
        uint32_t seed = 42;
        for (uint32_t i = 0; i < size; i++) { // NOLINT(*-loop-convert)
            static_cast<uint8_t *>(host_buffer)[i] = next_uint32(seed, 0, 255);
        }
    }

    volatile uint64_t simple_hashes[n_repeat] = {0};

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = ccoip::hash_utils::simplehash_cpu(host_buffer, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU (4 bytes): " << duration.count() << " us" << std::endl;
    double bandwidth = (static_cast<double>(n_repeat * size) / 1e9) / (static_cast<double>(duration.count()) / 1e6);
    std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

    for (int i = 1; i < n_repeat; ++i) {
        ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
    }

    // Insert an updated reference if desired.
    ASSERT_EQ(3765247898, simple_hashes[0]);

    free(host_buffer);
}

TEST(SimpleHashTest, TestOneVecPlus2WordsPlus1Byte) {
    // 1 vector = 16 bytes, plus 2 words = 8 bytes, plus 1 byte => total 25 bytes
    constexpr size_t size = 25;
    constexpr int n_repeat = 100;

    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    // init random
    {
        uint32_t seed = 42;
        for (uint32_t i = 0; i < size; i++) { // NOLINT(*-loop-convert)
            static_cast<uint8_t *>(host_buffer)[i] = next_uint32(seed, 0, 255);
        }
    }

    volatile uint64_t simple_hashes[n_repeat] = {0};

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = ccoip::hash_utils::simplehash_cpu(host_buffer, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU (25 bytes): " << duration.count() << " us" << std::endl;
    double bandwidth = (static_cast<double>(n_repeat * size) / 1e9) / (static_cast<double>(duration.count()) / 1e6);
    std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

    for (int i = 1; i < n_repeat; ++i) {
        ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
    }

    // Insert an updated reference if desired.
    ASSERT_EQ(3651434421, simple_hashes[0]);

    free(host_buffer);
}

int main(int argc, char **argv) {
    // Standard GTest main with no CUDA context creation.
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
