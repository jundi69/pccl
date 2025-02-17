#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

// Declaration of the CPU hash function (implemented elsewhere)
extern "C" uint64_t simplehash(const void *data, size_t n_bytes);

namespace ccoip::hash_utils {
    [[nodiscard]] uint32_t simplehash_cuda(const void *data, size_t n_bytes);
} // namespace ccoip::hash_utils

// Declaration of the CUDA random–init kernel.
__global__ void random_init_kernel(uint64_t *data, size_t N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nt = blockDim.x * gridDim.x;
    for (size_t i = tid; i < N; i += nt) {
        data[i] = (tid * nt ^ i & N) * 0xaabaababab1ull;
    }
}

// Helper: use CUDA to quickly fill host memory with random data.
static void gpu_fill_random(void *host_buffer, const size_t size) {
    CUdeviceptr dev_ptr = 0;
    CUresult res = cuMemAlloc_v2(&dev_ptr, size);
    if (res != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuMemAlloc_v2 failed\n");
        abort();
    }
    // Launch random_init_kernel to initialize device memory.
    const size_t num_elements = size / sizeof(uint64_t);
    random_init_kernel<<<8, 256>>>(reinterpret_cast<uint64_t *>(dev_ptr), num_elements);

    // Wait for kernel completion.
    res = cuStreamSynchronize(nullptr);
    if (res != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuStreamSynchronize failed\n");
        abort();
    }
    // Copy data from device to host.
    res = cuMemcpyDtoH_v2(host_buffer, dev_ptr, size);
    if (res != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuMemcpyDtoH_v2 failed\n");
        abort();
    }
    cuMemFree_v2(dev_ptr);
}

uint32_t next_uint32(uint32_t &seed, const uint32_t lo, const uint32_t hi) {
    seed = 1664525u * seed + 1013904223u;
    const uint64_t range = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo) + 1ULL;
    const uint32_t rndInRange = static_cast<uint32_t>(static_cast<uint64_t>(seed) % range) + lo;
    return rndInRange;
}

static void cpu_fill_random(uint8_t *data, const size_t size) {
    uint32_t seed = 42;
    for (uint32_t i = 0; i < size; i++) { // NOLINT(*-loop-convert)
        data[i] = static_cast<uint8_t>(next_uint32(seed, 0, 255));
    }
}

TEST(SimpleHashTest, BenchmarkAgainstBaseline) {
    // Use the same buffer size used in the CUDA tests.
    constexpr size_t size = 154533888; // in bytes
    constexpr int n_repeat = 500;

    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    // Fill host_buffer using the CUDA random-init kernel.
    gpu_fill_random(host_buffer, size);

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

    CUdeviceptr dev_ptr = 0;
    CUresult res = cuMemAlloc_v2(&dev_ptr, size);
    ASSERT_EQ(res, CUDA_SUCCESS) << "cuMemAlloc_v2 failed!";
    res = cuMemcpyHtoD_v2(dev_ptr, host_buffer, size);
    ASSERT_EQ(res, CUDA_SUCCESS) << "cuMemcpyHtoD_v2 failed!";
    const uint64_t reference = ccoip::hash_utils::simplehash_cuda(reinterpret_cast<void *>(dev_ptr), size);
    cuMemFree_v2(dev_ptr);

    // Check result
    for (int i = 0; i < n_repeat; ++i) {
        ASSERT_EQ(reference, simple_hashes[i]);
        ASSERT_EQ(1885439893, simple_hashes[i]);
    }
    free(host_buffer);
}

TEST(SimpleHashTest, TestSizeOneByte) {
    constexpr size_t size = 1;
    constexpr int n_repeat = 100;
    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    // Fill the host buffer via CPU random initialization
    cpu_fill_random(static_cast<uint8_t *>(host_buffer), size);

    volatile uint64_t simple_hashes[n_repeat] = {0};

    // CPU
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = simplehash(host_buffer, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU (1 byte): " << duration.count() << " us" << std::endl;
    double bandwidth = (static_cast<double>(n_repeat * size) / 1e9) / (static_cast<double>(duration.count()) / 1e6);
    std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

    for (int i = 1; i < n_repeat; ++i) {
        ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
    }

    CUdeviceptr dev_ptr = 0;
    CUresult res = cuMemAlloc_v2(&dev_ptr, size);
    ASSERT_EQ(res, CUDA_SUCCESS) << "cuMemAlloc_v2 failed!";
    res = cuMemcpyHtoD_v2(dev_ptr, host_buffer, size);
    ASSERT_EQ(res, CUDA_SUCCESS) << "cuMemcpyHtoD_v2 failed!";
    const uint64_t reference = ccoip::hash_utils::simplehash_cuda(reinterpret_cast<void *>(dev_ptr), size);
    cuMemFree_v2(dev_ptr);

    // Compare with expected
    for (int i = 0; i < n_repeat; ++i) {
        ASSERT_EQ(reference, simple_hashes[i]);
        ASSERT_EQ(2429690320, simple_hashes[i]);
    }

    free(host_buffer);
}

TEST(SimpleHashTest, TestSizeFourBytes) {
    constexpr size_t size = 4;
    constexpr int n_repeat = 100;
    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    cpu_fill_random(static_cast<uint8_t *>(host_buffer), size);

    volatile uint64_t simple_hashes[n_repeat] = {0};

    // CPU
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

    CUdeviceptr dev_ptr = 0;
    CUresult res = cuMemAlloc_v2(&dev_ptr, size);
    ASSERT_EQ(res, CUDA_SUCCESS) << "cuMemAlloc_v2 failed!";
    res = cuMemcpyHtoD_v2(dev_ptr, host_buffer, size);
    ASSERT_EQ(res, CUDA_SUCCESS) << "cuMemcpyHtoD_v2 failed!";
    const uint64_t reference = ccoip::hash_utils::simplehash_cuda(reinterpret_cast<void *>(dev_ptr), size);
    cuMemFree_v2(dev_ptr);

    // Compare with expected
    for (int i = 0; i < n_repeat; ++i) {
        ASSERT_EQ(reference, simple_hashes[i]);
        ASSERT_EQ(1120185552, simple_hashes[i]);
    }

    free(host_buffer);
}

TEST(SimpleHashTest, TestOneVecPlus2WordsPlus1Byte) {
    // 1 vector = 16 bytes, plus 2 words = 8 bytes, plus 1 byte => total 25 bytes
    constexpr size_t size = 25;
    constexpr int n_repeat = 100;
    void *host_buffer = malloc(size);
    ASSERT_NE(host_buffer, nullptr) << "Allocation failed";

    cpu_fill_random(static_cast<uint8_t *>(host_buffer), size);

    volatile uint64_t simple_hashes[n_repeat] = {0};

    // CPU
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

    CUdeviceptr dev_ptr = 0;
    CUresult res = cuMemAlloc_v2(&dev_ptr, size);
    ASSERT_EQ(res, CUDA_SUCCESS) << "cuMemAlloc_v2 failed!";
    res = cuMemcpyHtoD_v2(dev_ptr, host_buffer, size);
    ASSERT_EQ(res, CUDA_SUCCESS) << "cuMemcpyHtoD_v2 failed!";
    const uint64_t reference = ccoip::hash_utils::simplehash_cuda(reinterpret_cast<void *>(dev_ptr), size);
    cuMemFree_v2(dev_ptr);

    // Compare with expected
    for (int i = 0; i < n_repeat; ++i) {
        ASSERT_EQ(reference, simple_hashes[i]);
        ASSERT_EQ(654648064, simple_hashes[i]);
    }

    free(host_buffer);
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
