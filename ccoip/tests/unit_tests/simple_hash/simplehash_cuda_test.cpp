#include <chrono>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <random>

extern "C" uint64_t simplehash_cuda_kernel(const void *data, size_t n_bytes);

uint32_t next_uint32(uint32_t &seed, const uint32_t lo, const uint32_t hi) {
    seed = 1664525u * seed + 1013904223u;
    const uint64_t range = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo) + 1ULL;
    const uint32_t rndInRange = static_cast<uint32_t>(static_cast<uint64_t>(seed) % range) + lo;
    return rndInRange;
}

TEST(SimpleHashTest, BenchmarkAgainstBaseline) {
    std::vector<uint8_t> data(154533888, 0);

    // init random
    {
        uint32_t seed = 42;
        for (uint32_t i = 0; i < data.size(); i++) { // NOLINT(*-loop-convert)
            data[i] = next_uint32(seed, 0, 255);
        }
    }

    constexpr int n_repeat = 100;

    volatile uint64_t simple_hashes[n_repeat]{};
    CUdeviceptr data_ptr{};
    cuMemAlloc_v2(&data_ptr, data.size());
    cuMemcpyHtoD_v2(data_ptr, data.data(), data.size());

    // launch benchmark
    {
        const auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_repeat; ++i) { // NOLINT(*-loop-convert)
            simple_hashes[i] = simplehash_cuda_kernel(reinterpret_cast<void *>(data_ptr), data.size());
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "CUDA: " << duration.count() << " us" << std::endl;
        const double bandwidth = (static_cast<double>(n_repeat * data.size()) / 1e9) / (
                                     static_cast<double>(duration.count()) / 1e6);
        std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

        // assert all hashes are the same
        for (int i = 1; i < n_repeat; ++i) {
            ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
        }
    }
    cuMemFree_v2(data_ptr);

    for (int i = 0; i < n_repeat; ++i) {
        // check if the hashes are the same
        ASSERT_EQ(1502689209, simple_hashes[i]);
    }
}

TEST(SimpleHashTest, TestSizeOneByte) {
    std::vector<uint8_t> data(1, 0);

    // init random
    {
        uint32_t seed = 42;
        for (uint32_t i = 0; i < data.size(); i++) { // NOLINT(*-loop-convert)
            data[i] = next_uint32(seed, 0, 255);
        }
    }

    constexpr int n_repeat = 100;

    volatile uint64_t simple_hashes[n_repeat]{};
    CUdeviceptr data_ptr{};
    cuMemAlloc_v2(&data_ptr, data.size());
    cuMemcpyHtoD_v2(data_ptr, data.data(), data.size());

    // launch benchmark
    {
        const auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_repeat; ++i) {
            // NOLINT(*-loop-convert)
            simple_hashes[i] = simplehash_cuda_kernel(reinterpret_cast<void *>(data_ptr), data.size());
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "CUDA: " << duration.count() << " us" << std::endl;
        const double bandwidth = (static_cast<double>(n_repeat * data.size()) / 1e9) / (
                                     static_cast<double>(duration.count()) / 1e6);
        std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

        // assert all hashes are the same
        for (int i = 1; i < n_repeat; ++i) {
            ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
        }
    }
    cuMemFree_v2(data_ptr);

    for (int i = 0; i < n_repeat; ++i) {
        // check if the hashes are the same
        ASSERT_EQ(366970640, simple_hashes[i]);
    }
}


TEST(SimpleHashTest, TestSizeFourBytes) {
    std::vector<uint8_t> data(4, 0);

    // init random
    {
        uint32_t seed = 42;
        for (uint32_t i = 0; i < data.size(); i++) { // NOLINT(*-loop-convert)
            data[i] = next_uint32(seed, 0, 255);
        }
    }

    constexpr int n_repeat = 100;

    volatile uint64_t simple_hashes[n_repeat]{};
    CUdeviceptr data_ptr{};
    cuMemAlloc_v2(&data_ptr, data.size());
    cuMemcpyHtoD_v2(data_ptr, data.data(), data.size());

    // launch benchmark
    {
        const auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_repeat; ++i) { // NOLINT(*-loop-convert)
            simple_hashes[i] = simplehash_cuda_kernel(reinterpret_cast<void *>(data_ptr), data.size());
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "CUDA: " << duration.count() << " us" << std::endl;
        const double bandwidth = (static_cast<double>(n_repeat * data.size()) / 1e9) / (
                                     static_cast<double>(duration.count()) / 1e6);
        std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

        // assert all hashes are the same
        for (int i = 1; i < n_repeat; ++i) {
            ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
        }
    }
    cuMemFree_v2(data_ptr);

    for (int i = 0; i < n_repeat; ++i) {
        // check if the hashes are the same
        ASSERT_EQ(3727309584, simple_hashes[i]);
    }
}

TEST(SimpleHashTest, TestOneVecPlus2WordsPlus1Byte) {
    std::vector<uint8_t> data(1 * 4 * 4 + 2 * 4 + 1, 0); // 1 * 4 * 4 = 1 vec; 2 * 4 = 2 words; 1 = 1 byte

    // init random
    {
        uint32_t seed = 42;
        for (uint32_t i = 0; i < data.size(); i++) { // NOLINT(*-loop-convert)
            data[i] = next_uint32(seed, 0, 255);
        }
    }

    constexpr int n_repeat = 100;

    volatile uint64_t simple_hashes[n_repeat]{};
    CUdeviceptr data_ptr{};
    cuMemAlloc_v2(&data_ptr, data.size());
    cuMemcpyHtoD_v2(data_ptr, data.data(), data.size());

    // launch benchmark
    {
        const auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_repeat; ++i) { // NOLINT(*-loop-convert)
            simple_hashes[i] = simplehash_cuda_kernel(reinterpret_cast<void *>(data_ptr), data.size());
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "CUDA: " << duration.count() << " us" << std::endl;
        const double bandwidth = (static_cast<double>(n_repeat * data.size()) / 1e9) / (
                                     static_cast<double>(duration.count()) / 1e6);
        std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

        // assert all hashes are the same
        for (int i = 1; i < n_repeat; ++i) {
            ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
        }
    }
    cuMemFree_v2(data_ptr);

    for (int i = 0; i < n_repeat; ++i) {
        // check if the hashes are the same
        ASSERT_EQ(3836545445, simple_hashes[i]);
    }
}

int main(int argc, char **argv) {
    cuInit(0);
    CUcontext ctx{};
    int device_count{};
    cuDeviceGetCount(&device_count);
    if (device_count < 1) {
        std::cerr << "Can't run CUDA-tests when no CUDA-supported GPUs are available!" << std::endl;
        return -1;
    }
    if (cuCtxCreate_v2(&ctx, 0, 0) != CUDA_SUCCESS) {
        std::cerr << "Can't create CUDA context!" << std::endl;
        return -1;
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
