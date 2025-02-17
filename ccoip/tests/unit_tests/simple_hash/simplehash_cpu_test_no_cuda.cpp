#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

// Declaration of the CPU hash function (implemented elsewhere).
extern "C" uint64_t simplehash(const void *data, size_t n_bytes);

// CPU-based random initialization of a buffer.
static void cpu_fill_random(char *data, const size_t size) {
    std::mt19937_64 generator(42);
    std::uniform_int_distribution<uint32_t> dist(0, 255);

    for (size_t i = 0; i < size; i++) {
        reinterpret_cast<unsigned char *>(data)[i] = dist(generator);
    }
}

TEST(SimpleHashTest, BenchmarkAgainstBaseline) {
    // Use the same buffer size once used in the CUDA tests.
    constexpr size_t size = 154533888; // in bytes
    constexpr int n_repeat = 500;

    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    // Fill host_buffer using CPU RNG.
    cpu_fill_random(static_cast<char *>(host_buffer), size);

    volatile uint64_t simple_hashes[n_repeat] = {0};

    // CPU baseline
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = simplehash(host_buffer, size);
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
    ASSERT_EQ(1683357801, simple_hashes[0]);

    free(host_buffer);
}

TEST(SimpleHashTest, TestSizeOneByte) {
    constexpr size_t size = 1;
    constexpr int n_repeat = 100;

    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    // Fill the host buffer via CPU RNG.
    cpu_fill_random(static_cast<char *>(host_buffer), size);

    volatile uint64_t simple_hashes[n_repeat] = {0};

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = simplehash(host_buffer, size);
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
    ASSERT_EQ(3683124237, simple_hashes[0]);

    free(host_buffer);
}

TEST(SimpleHashTest, TestSizeFourBytes) {
    constexpr size_t size = 4;
    constexpr int n_repeat = 100;

    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    cpu_fill_random(static_cast<char *>(host_buffer), size);

    volatile uint64_t simple_hashes[n_repeat] = {0};

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = simplehash(host_buffer, size);
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
    ASSERT_EQ(2375994381, simple_hashes[0]);

    free(host_buffer);
}

TEST(SimpleHashTest, TestOneVecPlus2WordsPlus1Byte) {
    // 1 vector = 16 bytes, plus 2 words = 8 bytes, plus 1 byte => total 25 bytes
    constexpr size_t size = 25;
    constexpr int n_repeat = 100;

    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    cpu_fill_random(static_cast<char *>(host_buffer), size);

    volatile uint64_t simple_hashes[n_repeat] = {0};

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = simplehash(host_buffer, size);
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
    ASSERT_EQ(1032845947, simple_hashes[0]);

    free(host_buffer);
}

int main(int argc, char **argv) {
    // Standard GTest main with no CUDA context creation.
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
