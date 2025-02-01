#include <random>
#include <cstring>
#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" uint64_t simplehash_cuda_kernel(const void *data, size_t n_bytes);

TEST(SimpleHashTest, BenchmarkAgainstBaseline) {
    std::vector<uint8_t> data(1024 * 1024 * 1024, 0);

    // init random
    {
        std::mt19937_64 generator(42);
        std::uniform_int_distribution<uint32_t> dist(0, 255);

        for (unsigned char &i: data) {
            // NOLINT(*-loop-convert)
            i = dist(generator);
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
        const double bandwidth = (static_cast<double>(n_repeat * data.size()) / 1e9) / (static_cast<double>(duration.count()) / 1e6);
        std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

        // assert all hashes are the same
        for (int i = 1; i < n_repeat; ++i) {
            ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
        }
    }
    cuMemFree_v2(data_ptr);

    for (int i = 0; i < n_repeat; ++i) {
        // check if the hashes are the same
        ASSERT_EQ(3804779624, simple_hashes[i]);
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
    cuCtxCreate_v2(&ctx, 0, 0);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

/*
int main() {
    int seed{};
    std::cout << "Enter seed: ";
    std::cin >> seed;

    std::vector<uint8_t> data(1024 * 1024 * 10, 0); // 10 MB

    // init random
    {
        std::mt19937_64 generator(seed);
        std::uniform_int_distribution<uint32_t> dist(0, 255);

        for (size_t i = 0; i < data.size(); ++i) {// NOLINT(*-loop-convert)
            data[i] = dist(generator);
        }
    }

    for (size_t i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(data[i]) << " ";
    }
    std::cout << std::endl;

    constexpr int n_repeat = 100;
    uint64_t hashes[n_repeat]{};

    void *data_ptr{};
    cudaMalloc(&data_ptr, data.size());
    cudaMemcpy(data_ptr, data.data(), data.size(), cudaMemcpyHostToDevice); {
        const auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_repeat; ++i) { // NOLINT(*-loop-convert)
            hashes[i] = simplehash_cuda_kernel(data_ptr, data.size());
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "CUDA: " << duration.count() << " ms" << std::endl;
        const double bandwidth = (static_cast<double>(n_repeat * data.size()) / 1e9) / (static_cast<double>(duration.count()) / 1000.0);
        std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

        // assert all hashes are the same
        for (int i = 1; i < n_repeat; ++i) {
            if (hashes[i] != hashes[0]) {
                std::cerr << "Hash function did not return deterministic results for iteration " << i << " compared to iteration zero!" << std::endl;
            }
        }
    }
    std::cout << "SimpleHash: " << hashes[0] << std::endl;
    cudaFree(data_ptr);
}
*/
