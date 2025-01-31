#include "hash_utils.hpp"


extern "C" uint64_t simplehash_cuda_kernel(const void *data, size_t n_bytes);

uint32_t ccoip::hash_utils::simplehash_cuda(const void *data, const size_t n_bytes) {
#ifndef PCCL_HAS_CUDA_SUPPORT
    LOG(BUG) << "PCCL is not built with CUDA support. Cannot invoke simplehash_cuda!";
    return 0;
#endif
    return simplehash_cuda_kernel(data, n_bytes);
}
