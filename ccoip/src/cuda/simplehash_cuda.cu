#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cuda.h>

struct HashCombine {
    __host__ __device__
    uint32_t operator()(uint32_t a, const uint32_t b) const {
        a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
        return a;
    }
};

extern "C" __host__ uint64_t simplehash_cuda_kernel(const void *data, const size_t n_bytes) {
    const auto n_ints = static_cast<std::ptrdiff_t>(n_bytes / sizeof(uint32_t));
    const thrust::device_ptr<const uint32_t> device_vector = thrust::device_pointer_cast<const uint32_t>(
        static_cast<const uint32_t *>(data));
    const auto remaining_bytes = static_cast<uint32_t>(n_bytes % sizeof(uint32_t));

    std::unique_ptr<uint8_t[]> trailing_buffer = nullptr;
    if (remaining_bytes > 0) {
        trailing_buffer = std::unique_ptr<uint8_t[]>(new uint8_t[remaining_bytes]);
        cuMemcpyDtoH_v2(trailing_buffer.get(),
                        reinterpret_cast<CUdeviceptr>(static_cast<const uint8_t *>(data) + n_ints * sizeof(uint32_t)),
                        remaining_bytes);
    }

    uint32_t result = reduce(device_vector, device_vector + n_ints, 0, HashCombine());
    cuStreamSynchronize(nullptr);
    if (remaining_bytes > 0) {
        for (uint32_t i = 0; i < remaining_bytes; ++i) {
            result ^= trailing_buffer[i] + 0x9e3779b97f4a7c15ULL + (result << 6) + (result >> 2);
        }
    }
    return result;
}
