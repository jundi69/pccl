#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <emmintrin.h>   // SSE2 intrinsics
#include <immintrin.h>
#include <mutex>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

//----------------------------------------------------------------
// Constants
//----------------------------------------------------------------
constexpr int warpSize = 32;
constexpr size_t blockDim = 256; // Emulate 256 threads per block.

//----------------------------------------------------------------
// Anonymous namespace for internal functions, types, and globals.
//----------------------------------------------------------------
namespace {

//=========================
// Forward Declarations
//=========================

// CPU feature detection and dispatch types.
enum SimdCapability {
    SCALAR = 0,
    SSE2   = 1,
    AVX2   = 2
};

struct CPUFeatures {
    bool sse2;
    bool avx2;
};

CPUFeatures detectCPUFeatures();

SimdCapability getSimdCapability();

// Function pointer types for coarse–grained dispatch.
typedef uint32_t (*WarpReduceFunc)(const uint32_t input[warpSize]);
typedef uint32_t (*BlockReduceFunc)(const uint32_t acc[blockDim]);
typedef void (*ThreadBlockStepFunc)(uint32_t acc[blockDim],
                                    const uint32_t *current_words,
                                    int n_threads_with_data);

// Forward declarations of our hash and reduction functions.
uint32_t hash_combine_scalar(uint32_t a, uint32_t b);
__m128i mm_mullo_epi32_sse2(__m128i a, __m128i b);
__m128i hash_combine_sse(__m128i a, __m128i b);
__m256i hash_combine_avx2(__m256i a, __m256i b);

uint32_t warp_reduce_scalar(const uint32_t input[warpSize]);
uint32_t warp_reduce_sse(const uint32_t input[warpSize]);
uint32_t warp_reduce_avx2(const uint32_t input[warpSize]);

uint32_t block_reduce_scalar(const uint32_t acc[blockDim]);
uint32_t block_reduce_sse(const uint32_t acc[blockDim]);
uint32_t block_reduce_avx2(const uint32_t acc[blockDim]);

void compute_thread_block_perform_all_threads_step_scalar(uint32_t acc[blockDim],
                                                            const uint32_t *current_words,
                                                            int n_threads_with_data);
void compute_thread_block_perform_all_threads_step_sse2(uint32_t acc[blockDim],
                                                          const uint32_t *current_words,
                                                          int n_threads_with_data);
void compute_thread_block_perform_all_threads_step_avx2(uint32_t acc[blockDim],
                                                          const uint32_t *current_words,
                                                          int n_threads_with_data);

uint32_t compute_thread_block_result(const size_t b, const uint32_t *words,
                                       const size_t n_vec, const size_t vectorsPerBlock);
uint32_t compute_kernel_result(const size_t gridDim, const uint32_t *words,
                               const size_t n_vec, const size_t vectorsPerBlock);

//=========================
// CPU Detection & Dispatch Initialization
//=========================
CPUFeatures detectCPUFeatures() {
    CPUFeatures features { false, false };
#if defined(_MSC_VER)
    int cpu_info[4] = {0};
    __cpuid(cpu_info, 1);
  #if defined(SIMPLEHASH_ENABLE_SSE2)
    features.sse2 = (cpu_info[3] & (1 << 26)) != 0;
  #else
    features.sse2 = false;
  #endif
  #if !defined(SIMPLEHASH_DISABLE_AVX2)
    __cpuid(cpu_info, 0x7);
    features.avx2 = (cpu_info[1] & (1 << 5)) != 0;
  #else
    features.avx2 = false;
  #endif
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
  #if defined(SIMPLEHASH_ENABLE_SSE2)
    features.sse2 = (edx & (1 << 26)) != 0;
  #else
    features.sse2 = false;
  #endif
  #if !defined(SIMPLEHASH_DISABLE_AVX2)
    __get_cpuid(0x7, &eax, &ebx, &ecx, &edx);
    features.avx2 = (ebx & (1 << 5)) != 0;
  #else
    features.avx2 = false;
  #endif
#endif
    return features;
}

static SimdCapability g_simd_capability = SCALAR;
static std::once_flag g_simd_once;

SimdCapability getSimdCapability() {
    std::call_once(g_simd_once, [](){
        CPUFeatures features = detectCPUFeatures();
        if (features.avx2)
            g_simd_capability = AVX2;
        else if (features.sse2)
            g_simd_capability = SSE2;
        else
            g_simd_capability = SCALAR;
    });
    return g_simd_capability;
}

// Global function pointers for dispatch.
static WarpReduceFunc       g_warp_reduce_fn       = nullptr;
static BlockReduceFunc      g_block_reduce_fn      = nullptr;
static ThreadBlockStepFunc  g_thread_block_step_fn = nullptr;
static std::once_flag       g_dispatch_once;

void init_simd_dispatch() {
    std::call_once(g_dispatch_once, [](){
        SimdCapability cap = getSimdCapability();
        if (cap == AVX2) {
            g_warp_reduce_fn       = warp_reduce_avx2;
            g_block_reduce_fn      = block_reduce_avx2;
            g_thread_block_step_fn = compute_thread_block_perform_all_threads_step_avx2;
        } else if (cap == SSE2) {
            g_warp_reduce_fn       = warp_reduce_sse;
            g_block_reduce_fn      = block_reduce_sse;
            g_thread_block_step_fn = compute_thread_block_perform_all_threads_step_sse2;
        } else {
            g_warp_reduce_fn       = warp_reduce_scalar;
            g_block_reduce_fn      = block_reduce_scalar;
            g_thread_block_step_fn = compute_thread_block_perform_all_threads_step_scalar;
        }
    });
}

//=========================
// HashCombine Implementations
//=========================
uint32_t hash_combine_scalar(uint32_t a, uint32_t b) {
    b ^= 0x9e3779b1u;
    b *= 0x85ebca6bu;
    a ^= b;
    return a;
}

__m128i mm_mullo_epi32_sse2(__m128i a, __m128i b) {
    __m128i even = _mm_mul_epu32(a, b);
    __m128i a_shift = _mm_srli_si128(a, 4);
    __m128i b_shift = _mm_srli_si128(b, 4);
    __m128i odd = _mm_mul_epu32(a_shift, b_shift);
    return _mm_unpacklo_epi32(
        _mm_shuffle_epi32(even, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm_shuffle_epi32(odd,  _MM_SHUFFLE(2, 0, 2, 0))
    );
}

__m128i hash_combine_sse(__m128i a, __m128i b) {
    const __m128i c1 = _mm_set1_epi32(0x9e3779b1u);
    const __m128i c2 = _mm_set1_epi32(0x85ebca6bu);
    b = _mm_xor_si128(b, c1);
    b = mm_mullo_epi32_sse2(b, c2);
    a = _mm_xor_si128(a, b);
    return a;
}

__m256i hash_combine_avx2(__m256i a, __m256i b) {
    const __m256i c1 = _mm256_set1_epi32(0x9e3779b1u);
    const __m256i c2 = _mm256_set1_epi32(0x85ebca6bu);
    b = _mm256_xor_si256(b, c1);
    b = _mm256_mullo_epi32(b, c2);
    a = _mm256_xor_si256(a, b);
    return a;
}

//=========================
// Warp-Level Reduction Implementations
//=========================
uint32_t warp_reduce_scalar(const uint32_t input[warpSize]) {
    alignas(32) uint32_t temp[warpSize];
    std::memcpy(temp, input, warpSize * sizeof(uint32_t));
    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int i = 0; i < offset; i++) {
            temp[i] = hash_combine_scalar(temp[i], temp[i + offset]);
        }
    }
    return temp[0];
}

uint32_t warp_reduce_sse(const uint32_t input[warpSize]) {
    __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + 0));
    __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + 4));
    __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + 8));
    __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + 12));
    __m128i v4 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + 16));
    __m128i v5 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + 20));
    __m128i v6 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + 24));
    __m128i v7 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + 28));

    v0 = hash_combine_sse(v0, v4);
    v1 = hash_combine_sse(v1, v5);
    v2 = hash_combine_sse(v2, v6);
    v3 = hash_combine_sse(v3, v7);
    v0 = hash_combine_sse(v0, v2);
    v1 = hash_combine_sse(v1, v3);
    v0 = hash_combine_sse(v0, v1);

    uint32_t words[4];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(words), v0);
    uint32_t result = hash_combine_scalar(words[0], words[2]);
    uint32_t h = hash_combine_scalar(words[1], words[3]);
    return hash_combine_scalar(result, h);
}

uint32_t warp_reduce_avx2(const uint32_t input[warpSize]) {
    __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + 0));
    __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + 8));
    __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + 16));
    __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + 24));

    v0 = hash_combine_avx2(v0, v2);
    v1 = hash_combine_avx2(v1, v3);
    __m256i combined = hash_combine_avx2(v0, v1);
    alignas(32) uint32_t temp[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), combined);
    for (int offset = 4; offset > 0; offset >>= 1) {
        for (int i = 0; i < offset; i++) {
            temp[i] = hash_combine_scalar(temp[i], temp[i + offset]);
        }
    }
    return temp[0];
}

//=========================
// Block-Level Reduction Implementations
//=========================
uint32_t block_reduce_scalar(const uint32_t acc[blockDim]) {
    constexpr int numWarps = blockDim / warpSize;
    alignas(32) uint32_t warpResults[warpSize] = {0};
    for (int w = 0; w < numWarps; w++) {
        warpResults[w] = g_warp_reduce_fn(&acc[w * warpSize]);
    }
    for (int i = numWarps; i < warpSize; i++) {
        warpResults[i] = 0;
    }
    return g_warp_reduce_fn(warpResults);
}

uint32_t block_reduce_sse(const uint32_t acc[blockDim]) {
    return block_reduce_scalar(acc);
}

uint32_t block_reduce_avx2(const uint32_t acc[blockDim]) {
    constexpr int numWarps = blockDim / warpSize;
    alignas(32) uint32_t warpResults[warpSize] = {0};
    for (int w = 0; w < numWarps; w++) {
        warpResults[w] = warp_reduce_avx2(&acc[w * warpSize]);
    }
    for (int i = numWarps; i < warpSize; i++) {
        warpResults[i] = 0;
    }
    return warp_reduce_avx2(warpResults);
}

//=========================
// Per-Thread Block Step Implementations
//=========================
void compute_thread_block_perform_all_threads_step_scalar(uint32_t acc[blockDim],
                                                            const uint32_t *current_words,
                                                            int n_threads_with_data)
{
    for (int t = 0; t < n_threads_with_data; t++) {
        const uint32_t *vec = current_words + t * 4;
        acc[t] = hash_combine_scalar(acc[t], vec[0]);
        acc[t] = hash_combine_scalar(acc[t], vec[1]);
        acc[t] = hash_combine_scalar(acc[t], vec[2]);
        acc[t] = hash_combine_scalar(acc[t], vec[3]);
    }
}

static inline __m128i load_strided_sse2(const uint32_t *base, size_t stride_in_bytes) {
    return _mm_setr_epi32(
        base[0],
        base[stride_in_bytes / sizeof(uint32_t)],
        base[2 * stride_in_bytes / sizeof(uint32_t)],
        base[3 * stride_in_bytes / sizeof(uint32_t)]
    );
}

void compute_thread_block_perform_all_threads_step_sse2(uint32_t acc[blockDim],
                                                          const uint32_t *current_words,
                                                          int n_threads_with_data)
{
    for (int t = 0; t <= n_threads_with_data - 4; t += 4) {
        const uint32_t *vec = current_words + t * 4;
        __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&acc[t]));
        __m128i v;
        v = load_strided_sse2(vec, 4 * sizeof(uint32_t));
        a = hash_combine_sse(a, v);
        v = load_strided_sse2(vec + 1, 4 * sizeof(uint32_t));
        a = hash_combine_sse(a, v);
        v = load_strided_sse2(vec + 2, 4 * sizeof(uint32_t));
        a = hash_combine_sse(a, v);
        v = load_strided_sse2(vec + 3, 4 * sizeof(uint32_t));
        a = hash_combine_sse(a, v);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(&acc[t]), a);
    }
    for (int t = (n_threads_with_data / 4) * 4; t < n_threads_with_data; t++) {
        const uint32_t *vec = current_words + t * 4;
        acc[t] = hash_combine_scalar(acc[t], vec[0]);
        acc[t] = hash_combine_scalar(acc[t], vec[1]);
        acc[t] = hash_combine_scalar(acc[t], vec[2]);
        acc[t] = hash_combine_scalar(acc[t], vec[3]);
    }
}

void compute_thread_block_perform_all_threads_step_avx2(uint32_t acc[blockDim],
                                                          const uint32_t *current_words,
                                                          int n_threads_with_data)
{
    const __m256i offset0 = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i offset1 = _mm256_setr_epi32(1, 5, 9, 13, 17, 21, 25, 29);
    const __m256i offset2 = _mm256_setr_epi32(2, 6, 10, 14, 18, 22, 26, 30);
    const __m256i offset3 = _mm256_setr_epi32(3, 7, 11, 15, 19, 23, 27, 31);
    for (int t = 0; t <= n_threads_with_data - 8; t += 8) {
        const uint32_t *base = current_words + t * 4;
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&acc[t]));
        __m256i v;
        v = _mm256_i32gather_epi32(reinterpret_cast<const int *>(base), offset0, 4);
        a = hash_combine_avx2(a, v);
        v = _mm256_i32gather_epi32(reinterpret_cast<const int *>(base), offset1, 4);
        a = hash_combine_avx2(a, v);
        v = _mm256_i32gather_epi32(reinterpret_cast<const int *>(base), offset2, 4);
        a = hash_combine_avx2(a, v);
        v = _mm256_i32gather_epi32(reinterpret_cast<const int *>(base), offset3, 4);
        a = hash_combine_avx2(a, v);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&acc[t]), a);
    }
    for (int t = (n_threads_with_data / 8) * 8; t < n_threads_with_data; t++) {
        const uint32_t *vec = current_words + t * 4;
        acc[t] = hash_combine_scalar(acc[t], vec[0]);
        acc[t] = hash_combine_scalar(acc[t], vec[1]);
        acc[t] = hash_combine_scalar(acc[t], vec[2]);
        acc[t] = hash_combine_scalar(acc[t], vec[3]);
    }
}

//=========================
// Higher-Level Reduction Functions
//=========================
uint32_t compute_thread_block_result(const size_t b, const uint32_t *words,
                                       const size_t n_vec, const size_t vectorsPerBlock)
{
    const size_t blockStart = b * vectorsPerBlock;
    size_t blockEnd = blockStart + vectorsPerBlock;
    if (blockEnd > n_vec)
        blockEnd = n_vec;
    alignas(32) uint32_t acc[blockDim] = {0};
    for (size_t i = blockStart; i < blockEnd; i += blockDim) {
        int n_threads_with_data = static_cast<int>(std::min(blockDim, blockEnd - i));
        const uint32_t *current_words = words + i * 4;
        g_thread_block_step_fn(acc, current_words, n_threads_with_data);
    }
    return g_block_reduce_fn(acc);
}

uint32_t compute_kernel_result(const size_t gridDim, const uint32_t *words,
                               const size_t n_vec, const size_t vectorsPerBlock)
{
    alignas(32) uint32_t thread_reduce_results[blockDim] = {0};
    const size_t n_sections = (gridDim + blockDim - 1) / blockDim;
    for (size_t section = 0; section < n_sections; section++) {
        size_t n_threads = std::min(gridDim - section * blockDim, blockDim);
    #ifdef SIMPLEHASH_ENABLE_OMP
    #pragma omp parallel for shared(thread_reduce_results)
    #endif
        for (int thread_id = 0; thread_id < static_cast<int>(n_threads); thread_id++) {
            size_t assigned_block_idx = section * blockDim + thread_id;
            uint32_t block_hash = compute_thread_block_result(assigned_block_idx, words, n_vec, vectorsPerBlock);
            thread_reduce_results[thread_id] = hash_combine_scalar(thread_reduce_results[thread_id], block_hash);
        }
    }
    return g_block_reduce_fn(thread_reduce_results);
}

} // end anonymous namespace

//================================================================
// External simplehash() function (with C linkage)
//================================================================
extern "C" uint64_t simplehash(const void *data, const size_t n_bytes) {
    if (n_bytes == 0)
        return 0;
    if (reinterpret_cast<uintptr_t>(data) % 16 != 0) {
        abort();
    }
    const size_t n_words = n_bytes / 4;
    const uint32_t *words = static_cast<const uint32_t *>(data);
    const size_t n_vec = n_words / 4;      // number of complete 4-word vectors.
    const size_t tail_ints = n_words % 4;    // leftover 32-bit words.
    size_t gridDim = (n_vec + blockDim - 1) / blockDim;
    if (gridDim > 960)
        gridDim = 960;
    // Initialize the coarse–grained dispatch function pointers.
    init_simd_dispatch();
    uint32_t final_hash = 0;
    if (gridDim) {
        const size_t vectorsPerBlock = (n_vec + gridDim - 1) / gridDim;
        final_hash = compute_kernel_result(gridDim, words, n_vec, vectorsPerBlock);
    }
    const size_t tailStart = n_vec * 4;
    for (size_t i = 0; i < tail_ints; i++) {
        final_hash = hash_combine_scalar(final_hash, words[tailStart + i]);
    }
    const size_t tail_bytes = n_bytes % 4;
    if (tail_bytes > 0) {
        uint32_t tail_val = 0;
        const uint8_t *bytes = static_cast<const uint8_t *>(data);
        const size_t byteStart = n_words * 4;
        for (size_t i = 0; i < tail_bytes; i++) {
            tail_val |= static_cast<uint32_t>(bytes[byteStart + i]) << (8 * i);
        }
        final_hash = hash_combine_scalar(final_hash, tail_val);
    }
    return final_hash;
}
