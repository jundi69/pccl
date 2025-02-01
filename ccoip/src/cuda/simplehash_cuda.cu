// This is about as fast as thrust can do it
// TODO: OPTIMIZE THIS FURTHER

#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <vector_types.h>

/**
 * @brief Functor to combine two hash values.
 *
 * Note that the operation is not associative so the order of operands matters.
 */
struct HashCombine {
    /**
     * @brief Combines two hash values.
     *
     * @param a The first hash value.
     * @param b The second hash value.
     * @return The combined hash value.
     */
    __host__ __device__ __forceinline__
    uint32_t operator()(uint32_t a, const uint32_t b) const {
        a ^= b + 0x9e3779b1u;
        a = (a << 7) | (a >> 25);
        a *= 0x85ebca6bu;
        return a;
    }
};

/**
 * @brief Performs a warp–level reduction using shuffle intrinsics.
 *
 * This function reduces the value across all threads in a warp.
 *
 * @param val The input value to be reduced.
 * @return The reduced hash value.
 */
static __device__ __forceinline__
uint32_t warpReduceHash(uint32_t val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        uint32_t v = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = HashCombine()(val, v);
    }
    return val;
}

/**
 * @brief Performs a block–level reduction using warpReduceHash.
 *
 * Each warp writes its result into shared memory, and then the first warp reduces these values.
 *
 * @param val The input value to be reduced across the block.
 * @return The final reduced hash value for the block.
 */
__device__ uint32_t blockReduceHash(uint32_t val) {
    val = warpReduceHash(val);
    __shared__ uint32_t warpResults[32]; // Supports up to 1024 threads (32 warps)
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warpId = threadIdx.x >> 5;
    if (lane == 0) {
        warpResults[warpId] = val;
    }
    __syncthreads();

    uint32_t final_val = 0;
    if (warpId == 0) {
        final_val = warpResults[lane];
        final_val = warpReduceHash(final_val);
    }
    return final_val;
}

/**
 * @brief Kernel for the first pass (big–pass) of the reduction.
 *
 * The input is interpreted as a vectorized array (each element is a uint4,
 * i.e. 4 uint32_t’s). The vectorized data is divided evenly among blocks, and
 * each thread within a block processes its segment, enhancing memory coalescing.
 *
 * @param d_in Pointer to the input data in device memory.
 * @param d_partial Pointer to the output array of partial hash results in device memory.
 * @param n_words Number of 32–bit words in the input.
 */
__global__ void big_reduce_kernel(const uint32_t *__restrict d_in,
                                  uint32_t *__restrict d_partial,
                                  const size_t n_words) {
    uint32_t sum = 0;

    // Interpret the input as a vectorized array (each load gets 4 uint32_t's).
    const auto *d4 = reinterpret_cast<const uint4 *>(d_in);
    // Total number of vectorized elements.
    const auto totalVec = static_cast<unsigned int>(n_words / 4);

    // Divide the vectorized array evenly among the blocks.
    // Each block processes a contiguous chunk.
    const uint32_t elementsPerBlock = (totalVec + gridDim.x - 1) / gridDim.x;
    const uint32_t blockStart = blockIdx.x * elementsPerBlock;
    uint32_t blockEnd = blockStart + elementsPerBlock;
    if (blockEnd > totalVec) {
        blockEnd = totalVec;
    }

    // Each thread in the block processes its share within the block’s segment.
#pragma unroll 8
    for (uint32_t i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x) {
        uint4 v = __ldg(d4 + i);
        sum = HashCombine()(sum, v.x);
        sum = HashCombine()(sum, v.y);
        sum = HashCombine()(sum, v.z);
        sum = HashCombine()(sum, v.w);
    }

    // Reduce the per–thread results within the block.
    sum = blockReduceHash(sum);
    if (threadIdx.x == 0) {
        d_partial[blockIdx.x] = sum;
    }
}

/**
 * @brief Kernel for the final reduction pass.
 *
 * This kernel reduces the partial block results into a single final hash.
 *
 * @param d_partial Pointer to the partial hash results in device memory.
 * @param d_out Pointer to the output final hash in device memory.
 * @param n_blocks Number of partial results (blocks) to reduce.
 */
__global__ void final_reduce_kernel(const uint32_t *__restrict d_partial,
                                    uint32_t *__restrict d_out,
                                    const size_t n_blocks) {
    uint32_t sum = 0;
    const uint32_t idx = threadIdx.x;
    const uint32_t stride = blockDim.x;

    for (uint32_t i = idx; i < n_blocks; i += stride) {
        sum = HashCombine()(sum, d_partial[i]);
    }

    sum = blockReduceHash(sum);

    if (threadIdx.x == 0) {
        d_out[0] = sum;
    }
}

/**
 * @brief Computes a simple hash for device–resident data using CUDA kernels.
 *
 * This host function launches a two–pass reduction on the device. It handles the input
 * as 32–bit words for the bulk processing and processes any leftover bytes (if present)
 * on the host.
 *
 * @param data Device pointer to the input data.
 * @param n_bytes Number of bytes in the input data.
 * @return The computed hash value as a 64–bit unsigned integer.
 */
extern "C" __host__ uint64_t simplehash_cuda_kernel(const void *data, const size_t n_bytes) {
    if (n_bytes == 0) {
        return 0;
    }

    // Interpret the data as an array of 32–bit words.
    const size_t n_words = n_bytes / 4;
    const size_t tailBytes = n_bytes % 4;

    // Allocate device memory for the final result.
    CUdeviceptr d_out = 0;
    CUresult res = cuMemAlloc(&d_out, sizeof(uint32_t));
    if (res != CUDA_SUCCESS) {
        abort();
    }

    uint32_t finalHostVal = 0;

    if (n_words > 0) {
        // Use vectorized elements (each uint4 covers 4 uint32_t's) to base grid dimensions.
        constexpr int blockDim_x = 256;
        const auto totalVec = static_cast<uint32_t>(n_words / 4);
        uint32_t gridDim_x = (totalVec + blockDim_x - 1) / blockDim_x;
        if (gridDim_x > 960) {
            gridDim_x = 960;
        }

        // Allocate an array for partial block results.
        CUdeviceptr d_partial = 0;
        res = cuMemAlloc(&d_partial, gridDim_x * sizeof(uint32_t));
        if (res != CUDA_SUCCESS) {
            abort();
        }

        // Launch the big–pass kernel.
        const auto d_in_ptr = static_cast<const uint32_t *>(data);
        auto *d_partial_ptr = reinterpret_cast<uint32_t *>(d_partial);
        big_reduce_kernel<<<gridDim_x, blockDim_x>>>(d_in_ptr, d_partial_ptr, n_words);

        // Launch the final–pass kernel (using one block).
        final_reduce_kernel<<<1, blockDim_x>>>(d_partial_ptr,
                                               reinterpret_cast<uint32_t *>(d_out),
                                               gridDim_x);

        res = cuStreamSynchronize(nullptr);
        if (res != CUDA_SUCCESS) {
            abort();
        }

        // Copy the final hash result back to the host.
        res = cuMemcpyDtoH(&finalHostVal, d_out, sizeof(uint32_t));
        if (res != CUDA_SUCCESS) {
            abort();
        }

        // Free the partial results array.
        cuMemFree(d_partial);
    }

    // Process any leftover bytes (if n_bytes is not a multiple of 4).
    if (tailBytes > 0) {
        const CUdeviceptr tailPtr = reinterpret_cast<uintptr_t>(data) + n_words * 4;
        uint8_t tailHost[4] = {0, 0, 0, 0};

        res = cuMemcpyDtoH(tailHost, tailPtr, tailBytes);
        if (res != CUDA_SUCCESS) {
            abort();
        }

        uint32_t tailVal = 0;
        for (size_t i = 0; i < tailBytes; ++i) {
            tailVal |= static_cast<uint32_t>(tailHost[i]) << (8 * i);
        }
        finalHostVal = HashCombine()(finalHostVal, tailVal);
    }

    cuMemFree(d_out);
    return finalHostVal;
}
