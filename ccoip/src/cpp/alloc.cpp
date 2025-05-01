// This file hooks into the global new/delete operators to use our malloc wrapper,
// which can optionally be configured to memprotect outside of the allocated region.
#include "alloc.hpp"

#include <assert.h>
#include <cstdio>
#include <dlfcn.h>
#include <new>

//#define CCOIP_GUARD_ALLOCATIONS
//#define CCOIP_HOOK_NEW_OPERATOR

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


namespace op_delete_loader {
    using PFN_delete_pv = void(*)(void *) noexcept;
    using PFN_delete_pv_sz = void(*)(void *, std::size_t) noexcept;
    using PFN_delete_pv_align = void(*)(void *, std::align_val_t) noexcept;
    using PFN_delete_pv_sz_align = void(*)(void *, std::size_t, std::align_val_t) noexcept;
    using PFN_delete_pv_nothrow = void(*)(void *, const std::nothrow_t &) noexcept;
    using PFN_delete_pv_align_nothrow = void(*)(void *, std::align_val_t, const std::nothrow_t &) noexcept;

    extern PFN_delete_pv real_delete;
    extern PFN_delete_pv real_delete_array;
    extern PFN_delete_pv_sz real_delete_sz;
    extern PFN_delete_pv_sz real_delete_array_sz;
    extern PFN_delete_pv_align real_delete_align;
    extern PFN_delete_pv_align real_delete_array_align;
    extern PFN_delete_pv_sz_align real_delete_sz_align;
    extern PFN_delete_pv_sz_align real_delete_array_sz_align;
    extern PFN_delete_pv_nothrow real_delete_nothrow;
    extern PFN_delete_pv_nothrow real_delete_array_nothrow;
    extern PFN_delete_pv_align_nothrow real_delete_align_nothrow;
    extern PFN_delete_pv_align_nothrow real_delete_array_align_nothrow;

    PFN_delete_pv real_delete = nullptr;
    PFN_delete_pv real_delete_array = nullptr;
    PFN_delete_pv_sz real_delete_sz = nullptr;
    PFN_delete_pv_sz real_delete_array_sz = nullptr;
    PFN_delete_pv_align real_delete_align = nullptr;
    PFN_delete_pv_align real_delete_array_align = nullptr;
    PFN_delete_pv_sz_align real_delete_sz_align = nullptr;
    PFN_delete_pv_sz_align real_delete_array_sz_align = nullptr;
    PFN_delete_pv_nothrow real_delete_nothrow = nullptr;
    PFN_delete_pv_nothrow real_delete_array_nothrow = nullptr;
    PFN_delete_pv_align_nothrow real_delete_align_nothrow = nullptr;
    PFN_delete_pv_align_nothrow real_delete_array_align_nothrow = nullptr;

    // helper to dlsym + assert
    static void *load_sym(const char *name) {
        void *p = dlsym(RTLD_NEXT, name);
        if (!p) {
            fprintf(stderr, "op_delete_loader: failed to load symbol %s\n", name);
            abort();
        }
        return p;
    }

    bool init() {
        real_delete = reinterpret_cast<PFN_delete_pv>(load_sym("_ZdlPv"));
        real_delete_array = reinterpret_cast<PFN_delete_pv>(load_sym("_ZdaPv"));
        real_delete_sz = reinterpret_cast<PFN_delete_pv_sz>(load_sym("_ZdlPvm"));
        real_delete_array_sz = reinterpret_cast<PFN_delete_pv_sz>(load_sym("_ZdaPvm"));
        real_delete_align = reinterpret_cast<PFN_delete_pv_align>(load_sym("_ZdlPvSt11align_val_t"));
        real_delete_array_align = reinterpret_cast<PFN_delete_pv_align>(load_sym("_ZdaPvSt11align_val_t"));
        real_delete_sz_align = reinterpret_cast<PFN_delete_pv_sz_align>(load_sym("_ZdlPvmSt11align_val_t"));
        real_delete_array_sz_align = reinterpret_cast<PFN_delete_pv_sz_align>(load_sym("_ZdaPvmSt11align_val_t"));
        real_delete_nothrow = reinterpret_cast<PFN_delete_pv_nothrow>(load_sym("_ZdlPvRKSt9nothrow_t"));
        real_delete_array_nothrow = reinterpret_cast<PFN_delete_pv_nothrow>(load_sym("_ZdaPvRKSt9nothrow_t"));
        real_delete_align_nothrow = reinterpret_cast<PFN_delete_pv_align_nothrow>(load_sym("_ZdlPvSt11align_val_tRKSt9nothrow_t"));
        real_delete_array_align_nothrow
                = reinterpret_cast<PFN_delete_pv_align_nothrow>(load_sym("_ZdaPvSt11align_val_tRKSt9nothrow_t"));
        return true;
    }

    __attribute__((constructor))
    static void initializer() {
        init();
    }
} // namespace op_delete_loader

#ifdef CCOIP_HOOK_NEW_OPERATOR

// Ordinary (throwing) allocation
void *operator new(std::size_t sz) {
    void *p = ccoip::alloc::malloc(sz);
    if (!p) throw std::bad_alloc();
    return p;
}

void *operator new[](std::size_t sz) {
    void *p = ccoip::alloc::malloc(sz);
    if (!p) throw std::bad_alloc();
    return p;
}


// Ordinary deallocation
void operator delete(void *p) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete(p);
    }
}

void operator delete[](void *p) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete_array(p);
    }
}

// Sized deallocate
void operator delete(void *p, std::size_t s) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete_sz(p, s);
    }
}

void operator delete[](void *p, std::size_t s) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete_array_sz(p, s);
    }
}

// Alignment‐aware (throwing) allocation
void *operator new(std::size_t sz, std::align_val_t al) {
    void *p = ccoip::alloc::malloc(sz);
    if (!p) throw std::bad_alloc();
    return p;
}

void *operator new[](std::size_t sz, std::align_val_t al) {
    void *p = ccoip::alloc::malloc(sz);
    if (!p) throw std::bad_alloc();
    return p;
}

// Alignment‐aware deallocation
void operator delete(void *p, std::align_val_t al) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete_align(p, al);
    }
}

void operator delete[](void *p, std::align_val_t al) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete_array_align(p, al);
    }
}

// sized + aligned deallocate
void operator delete(void *p, std::size_t s, std::align_val_t al) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete_sz_align(p, s, al);
    }
}

void operator delete[](void *p, std::size_t s, std::align_val_t al) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete_array_sz_align(p, s, al);
    }
}

// nothrow‐new variants:
void *operator new(std::size_t sz, const std::nothrow_t &) noexcept {
    return ccoip::alloc::malloc(sz);
}

void *operator new[](std::size_t sz, const std::nothrow_t &) noexcept {
    return ccoip::alloc::malloc(sz);
}

void operator delete(void *p, const std::nothrow_t &) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete_nothrow(p, std::nothrow);
    }
}

void operator delete[](void *p, const std::nothrow_t &) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete_array_nothrow(p, std::nothrow);
    }
}

// aligned + nothrow
void *operator new(std::size_t sz, std::align_val_t al, const std::nothrow_t &) noexcept {
    return ccoip::alloc::malloc(sz);
}

void *operator new[](std::size_t sz, std::align_val_t al, const std::nothrow_t &) noexcept {
    return ccoip::alloc::malloc(sz);
}

void operator delete(void *p, std::align_val_t al, const std::nothrow_t &) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete_align_nothrow(p, al, std::nothrow);
    }
}

void operator delete[](void *p, std::align_val_t al, const std::nothrow_t &) noexcept {
    if (ccoip::alloc::internal::is_guarded_ptr(p)) {
        ccoip::alloc::free(p);
    } else {
        op_delete_loader::real_delete_array_align_nothrow(p, al, std::nothrow);
    }
}
#endif
