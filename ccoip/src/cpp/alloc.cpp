// This file hooks into the global new/delete operators to use our malloc wrapper,
// which can optionally be configured to memprotect outside of the allocated region.
#include "alloc.hpp"

#include <new>

#define CCOIP_GUARD_ALLOCATIONS
#define CCOIP_HOOK_NEW_OPERATOR

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

#ifdef CCOIP_HOOK_NEW_OPERATOR
// Ordinary (throwing) allocation
void* operator new(std::size_t sz) {
    void* p = ccoip::alloc::malloc(sz);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t sz) {
    void* p = ccoip::alloc::malloc(sz);
    if (!p) throw std::bad_alloc();
    return p;
}


// Ordinary deallocation
void operator delete(void* p) noexcept {
    ccoip::alloc::free(p);
}
void operator delete[](void* p) noexcept {
    ccoip::alloc::free(p);
}

// Sized deallocate
void operator delete(void* p, std::size_t) noexcept {
    ccoip::alloc::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    ccoip::alloc::free(p);
}

// Alignment‐aware (throwing) allocation
void* operator new(std::size_t sz, std::align_val_t al) {
    void* p = ccoip::alloc::malloc(sz);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t sz, std::align_val_t al) {
    void* p = ccoip::alloc::malloc(sz);
    if (!p) throw std::bad_alloc();
    return p;
}

// Alignment‐aware deallocation
void operator delete(void* p, std::align_val_t al) noexcept {
    ccoip::alloc::free(p);
}
void operator delete[](void* p, std::align_val_t al) noexcept {
    ccoip::alloc::free(p);
}
// sized + aligned deallocate
void operator delete(void* p, std::size_t, std::align_val_t al) noexcept {
    ccoip::alloc::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t al) noexcept {
    ccoip::alloc::free(p);
}

// nothrow‐new variants:
void* operator new(std::size_t sz, const std::nothrow_t&) noexcept {
    return ccoip::alloc::malloc(sz);
}
void* operator new[](std::size_t sz, const std::nothrow_t&) noexcept {
    return ccoip::alloc::malloc(sz);
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
    ccoip::alloc::free(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
    ccoip::alloc::free(p);
}

// aligned + nothrow
void* operator new(std::size_t sz, std::align_val_t al, const std::nothrow_t&) noexcept {
    return ccoip::alloc::malloc(sz);
}
void* operator new[](std::size_t sz, std::align_val_t al, const std::nothrow_t&) noexcept {
    return ccoip::alloc::malloc(sz);
}
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept {
    ccoip::alloc::free(p);
}
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept {
    ccoip::alloc::free(p);
}
#endif