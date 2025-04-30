// This file hooks into the global new/delete operators to use our malloc wrapper,
// which can optionally be configured to memprotect outside of the allocated region.
#include "alloc.hpp"

#include <new>

#define CCOIP_GUARD_ALLOCATIONS

namespace ccoip::alloc {
    void *malloc(const size_t size) {
#ifdef CCOIP_GUARD_ALLOCATIONS
        return internal::guarded_malloc(size);
#else
        return std::malloc(size);
#endif
    }

    void free(void *ptr) {
#ifdef CCOIP_GUARD_ALLOCATIONS
        internal::guarded_free(ptr);
#else
        std::free(ptr);
#endif
    }
}

void* operator new(std::size_t __sz) {
    void* p = ccoip::alloc::malloc(__sz);
    if (!p) throw std::bad_alloc();
    return p;
}

void* operator new[](std::size_t __sz) {
    void* p = ccoip::alloc::malloc(__sz);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete(void* __p) noexcept {
    ccoip::alloc::free(__p);
}
void operator delete[](void* __p) noexcept {
    ccoip::alloc::free(__p);
}

void operator delete(void* __p, std::size_t) noexcept { operator delete(__p); }
void operator delete[](void* __p, std::size_t) noexcept { operator delete[](__p); }