#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>

constexpr int warpSize = 32;
constexpr size_t blockDim = 256; // Emulate 256 threads per block.

//----------------------------------------------------------------
// Scalar version of HashCombine (kept for reference)
// (The struct version is now replaced by a static inline function.)
static inline uint32_t hash_combine_scalar(uint32_t a, uint32_t b) {
    // Inspired by fnv1a.
    b ^= 0x9e3779b1u;
    b *= 0x85ebca6bu;
    a ^= b;
    return a;
}

//----------------------------------------------------------------
// Scalar warp–level reduction (32 elements) exactly like the CUDA code.
// (This implementation uses a temporary array of 32 numbers.)
static uint32_t warp_reduce(const uint32_t input[warpSize]) {
    alignas(32) uint32_t temp[warpSize];
    std::memcpy(temp, input, warpSize * sizeof(uint32_t));

    // The CUDA loop:
    //   for (int offset = 16; offset > 0; offset >>= 1) {
    //       uint32_t v = __shfl_down_sync(..., val, offset);
    //       val = HashCombine(val, v);
    //   }
    // When executed in lockstep by 32 threads, lane0 gets
    // contributions from lane16, then lane8 (which has merged 8+24),
    // then lane4, then lane2, then lane1.
    // To “simulate” the parallel update, we perform a series of iterations.

    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int i = 0; i < offset; i++) {
            temp[i] = hash_combine_scalar(temp[i], temp[i + offset]);
        }
    }
    return temp[0];
}

//----------------------------------------------------------------
// Scalar block–level reduction over 256 elements.
static uint32_t block_reduce(const uint32_t acc[blockDim]) {
    constexpr int numWarps = blockDim / warpSize; // Should be 8.
    alignas(32) uint32_t warpResults[32]; // At most 32 warps per block.
    for (int w = 0; w < numWarps; w++) {
        warpResults[w] = warp_reduce(&acc[w * warpSize]);
    }
    for (int i = numWarps; i < 32; i++) {
        warpResults[i] = 0;
    }
    const uint32_t blockResult = warp_reduce(warpResults);
    return blockResult;
}

static inline void compute_thread_block_perform_all_threads_step(
    uint32_t acc[blockDim], const uint32_t *__restrict current_words, const int n_threads_with_data) {
    for (int t = 0; t < n_threads_with_data; t++) {
        // Each vector element is 4 words.
        const uint32_t *vec = current_words + t * 4;
        // Important: combine the four words in order.
        acc[t] = hash_combine_scalar(acc[t], vec[0]);
        acc[t] = hash_combine_scalar(acc[t], vec[1]);
        acc[t] = hash_combine_scalar(acc[t], vec[2]);
        acc[t] = hash_combine_scalar(acc[t], vec[3]);
    }
}

//----------------------------------------------------------------
// Compute the result of a thread-block of blockIdx b.
static inline uint32_t compute_thread_block_result(const size_t b, const uint32_t *words, const size_t n_vec,
                                                   const size_t vectorsPerBlock) {
    const size_t blockStart = b * vectorsPerBlock;
    size_t blockEnd = blockStart + vectorsPerBlock;
    if (blockEnd > n_vec)
        blockEnd = n_vec;

    // Allocate 256 accumulators (one per virtual thread).
    alignas(32) uint32_t acc[blockDim] = {0};

    // Each virtual thread processes its strided subset of the vectorized data in this block.
    // This exactly mimics:
    //    for (i = blockStart + t; i < blockEnd; i += 256) { ... }
    for (size_t i = blockStart; i < blockEnd; i += blockDim) {
        const int n_threads_with_data = static_cast<int>(std::min(blockDim, blockEnd - i));
        const uint32_t *current_words = words + i * 4;
        compute_thread_block_perform_all_threads_step(acc, current_words, n_threads_with_data);
    }
    // Reduce the 256 accumulators into one hash for this block.
    const uint32_t block_hash = block_reduce(acc);
    return block_hash;
}

static bool openmp_configured = false;

#ifdef PCCL_BUILD_OPENMP_SUPPORT
#include <omp.h>
#endif

//----------------------------------------------------------------
// Compute the result of the final_reduce_kernel
static inline uint32_t compute_kernel_result(const size_t gridDim, const uint32_t *words, const size_t n_vec,
                                             const size_t vectorsPerBlock) {
#ifdef PCCL_BUILD_OPENMP_SUPPORT
    if (!openmp_configured) {
        // force OpenMP to use the max number of threads available
        int num_procs = omp_get_num_procs();

#if defined(__x86_64__) || defined(_M_X64)
        // on x86_64, we make the somewhat erroneous assumption that we always use half of the available cores
        // because we expect half of them to be hyper-threaded processors,
        // which imperially do only degrade performance when used
        num_procs /= 2;
#endif

        omp_set_num_threads(num_procs);
        openmp_configured = true;
    }
#endif
    // Process the blocks in a 2 layer reduce style.
    alignas(32) uint32_t thread_reduce_results[blockDim] = {0};

    // A section corresponds to the group of partial hashes that would be reduced in parallel by
    // the final_reduce_kernel in the CUDA implementation.
    const size_t n_sections = (gridDim + blockDim - 1) / blockDim;
    for (int section = 0; section < n_sections; section++) {
        const size_t n_threads = std::min(gridDim - section * blockDim, blockDim);
#ifdef PCCL_BUILD_OPENMP_SUPPORT
#pragma omp parallel for shared(thread_reduce_results, n_threads, section, blockDim, gridDim, words, n_vec, vectorsPerBlock) default(none)
#endif
        for (int thread_id = 0; thread_id < n_threads; thread_id++) {
            const size_t assigned_block_idx = section * blockDim + thread_id;
            const uint32_t block_hash = compute_thread_block_result(assigned_block_idx, words, n_vec, vectorsPerBlock);
            thread_reduce_results[thread_id] = hash_combine_scalar(thread_reduce_results[thread_id], block_hash);
        }
    }

    const uint32_t final_hash = block_reduce(thread_reduce_results);
    return final_hash;
}

//----------------------------------------------------------------
// The simple hash function which mimics the CUDA kernel’s reduction order.
extern "C" uint64_t simplehash(const void *data, const size_t n_bytes) {
    if (n_bytes == 0)
        return 0;
    // Check that the input pointer is 16–byte aligned.
    if (reinterpret_cast<uintptr_t>(data) % 16 != 0) {
        abort();
    }

    // Interpret the input as an array of 32–bit words.
    const size_t n_words = n_bytes / 4;
    const auto *words = static_cast<const uint32_t *>(data);

    // Process the vectorized portion.
    // We regard every 4 consecutive 32–bit words as a “vector” (like uint4).
    const size_t n_vec = n_words / 4; // number of complete 4–word vectors
    const size_t tail_ints = n_words % 4; // leftover full 32–bit words (if any)

    // Emulate the GPU’s big–pass reduction.
    // In the CUDA code the vectorized array is divided among blocks.
    size_t gridDim = (n_vec + blockDim - 1) / blockDim;
    if (gridDim > 960) {
        gridDim = 960;
    }

    uint32_t final_hash = 0;
    if (gridDim) {
        const size_t vectorsPerBlock = (n_vec + gridDim - 1) / gridDim;
        final_hash = compute_kernel_result(gridDim, words, n_vec, vectorsPerBlock);
    }

    // Process any leftover (tail) 32–bit words that didn’t form a complete vector.
    const size_t tailStart = n_vec * 4;
    for (size_t i = 0; i < tail_ints; i++) {
        final_hash = hash_combine_scalar(final_hash, words[tailStart + i]);
    }

    // Process any leftover bytes (if n_bytes is not a multiple of 4).
    const size_t tail_bytes = n_bytes % 4;
    if (tail_bytes > 0) {
        uint32_t tail_val = 0;
        const auto *bytes = static_cast<const uint8_t *>(data);
        const size_t byteStart = n_words * 4;
        for (size_t i = 0; i < tail_bytes; i++) {
            tail_val |= static_cast<uint32_t>(bytes[byteStart + i]) << (8 * i);
        }
        final_hash = hash_combine_scalar(final_hash, tail_val);
    }

    return final_hash;
}
