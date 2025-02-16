#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <cuda.h>

#include <gtest/gtest.h>

// Declaration of the CPU hash function (implemented elsewhere)
extern "C" uint64_t simplehash(const void *data, size_t n_bytes);

namespace ccoip::hash_utils {
    [[nodiscard]] uint32_t simplehash_cuda(const void *data, size_t n_bytes);
}


// Declaration of the CUDA random–init kernel.
__global__ void random_init_kernel(uint64_t *data, size_t N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nt = blockDim.x * gridDim.x;
    for (size_t i = tid; i < N; i += nt) {
        data[i] = (tid * nt ^ i & N) * 0xaabaababab1ull;
    }
}

// Helper: use CUDA to quickly fill host memory with random data.
// This routine allocates device memory, launches random_init_kernel,
// waits for kernel completion, then copies the result back to host (via cuMemcpyDtoH_v2)
// before finally freeing the device memory.
static void gpu_fill_random(void *host_buffer, const size_t size) {
    CUdeviceptr dev_ptr = 0;
    CUresult res = cuMemAlloc_v2(&dev_ptr, size);
    if (res != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuMemAlloc_v2 failed\n");
        abort();
    }
    // Launch random_init_kernel to initialize device memory.
    // We assume that each 64–bit entry gets randomized.
    const size_t num_elements = size / sizeof(uint64_t);
    random_init_kernel<<<8, 256>>>(reinterpret_cast<uint64_t*>(dev_ptr), num_elements);
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

static void cpu_fill_random(char *data, const size_t size) {
    std::mt19937_64 generator(42);
    std::uniform_int_distribution<uint32_t> dist(0, 255);

    for (size_t i = 0; i < size; i++) {
        // NOLINT(*-loop-convert)
        reinterpret_cast<unsigned char *>(data)[i] = dist(generator);
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

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = simplehash(host_buffer, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU: " << duration.count() << " us" << std::endl;
    double bandwidth =
        (static_cast<double>(n_repeat * size) / 1e9) /
        (static_cast<double>(duration.count()) / 1e6);
    std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Ensure all iterations produced exactly the same hash.
    for (int i = 1; i < n_repeat; ++i) {
        ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
    }
    // Expected value (from the CUDA version) is 649674942.
    const uint64_t reference = ccoip::hash_utils::simplehash_cuda(host_buffer, size);
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

    // Fill the host buffer via device random initialization.
    cpu_fill_random(static_cast<char *>(host_buffer), size);

    volatile uint64_t simple_hashes[n_repeat] = {0};

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = simplehash(host_buffer, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU (1 byte): " << duration.count() << " us" << std::endl;
    double bandwidth =
        (static_cast<double>(n_repeat * size) / 1e9) /
        (static_cast<double>(duration.count()) / 1e6);
    std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

    for (int i = 1; i < n_repeat; ++i) {
        ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
    }
    // Expected value (from CUDA tests).
    const uint64_t reference = ccoip::hash_utils::simplehash_cuda(host_buffer, size);
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

    cpu_fill_random(static_cast<char *>(host_buffer), size);

    volatile uint64_t simple_hashes[n_repeat] = {0};

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        simple_hashes[i] = simplehash(host_buffer, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU (4 bytes): " << duration.count() << " us" << std::endl;
    double bandwidth =
        (static_cast<double>(n_repeat * size) / 1e9) /
        (static_cast<double>(duration.count()) / 1e6);
    std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

    for (int i = 1; i < n_repeat; ++i) {
        ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
    }
    // Expected value (from CUDA tests).
    const uint64_t reference = ccoip::hash_utils::simplehash_cuda(host_buffer, size);
    for (int i = 0; i < n_repeat; ++i) {
        ASSERT_EQ(reference, simple_hashes[i]);
        ASSERT_EQ(1120185552, simple_hashes[i]);
    }
    free(host_buffer);
}

TEST(SimpleHashTest, TestOneVecPlus2WordsPlus1Byte) {
    // 1 vector = 4 * 4 bytes, plus 2 words (2 * 4 bytes), plus 1 byte => total 16 + 8 + 1 = 25 bytes.
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
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU (25 bytes): " << duration.count() << " us" << std::endl;
    double bandwidth =
        (static_cast<double>(n_repeat * size) / 1e9) /
        (static_cast<double>(duration.count()) / 1e6);
    std::cout << "Hashing-Bandwidth: " << bandwidth << " GB/s" << std::endl;

    for (int i = 1; i < n_repeat; ++i) {
        ASSERT_EQ(simple_hashes[i], simple_hashes[0]);
    }
    // Expected value (from CUDA tests).
    const uint64_t reference = ccoip::hash_utils::simplehash_cuda(host_buffer, size);
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