#pragma once

#include <unistd.h>
#include <sys/mman.h>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstdlib>

namespace ccoip::alloc::internal {
    static constexpr uint64_t GUARD_MAGIC = 0xDEADC0DECAFEBABEULL;

    struct Header {
        void *base;
        size_t total_bytes;
        uint64_t magic;
    };

    void *guarded_malloc(size_t size) {
        if (size == 0) return nullptr;

        long P = sysconf(_SC_PAGESIZE);
        assert(P > 0);

        size_t H = sizeof(Header);
        size_t payload_bytes = H + size;
        size_t payload_pages = (payload_bytes + P - 1) / P;

        size_t total_pages = payload_pages + 2;
        size_t total_bytes = total_pages * P;

        void *base = mmap(nullptr, total_bytes,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS,
                          -1, 0);
        assert(base != MAP_FAILED);

        char *b = static_cast<char *>(base);
        mprotect(b, P, PROT_NONE);
        mprotect(b + (total_pages - 1) * P, P, PROT_NONE);

        char *rw_end = b + total_bytes - P;
        char *user_ptr = rw_end - size;
        auto hdr = reinterpret_cast<Header *>(user_ptr - H);

        hdr->base = base;
        hdr->total_bytes = total_bytes;
        hdr->magic = GUARD_MAGIC;

        return user_ptr;
    }

    bool is_guarded_ptr(void *ptr) noexcept {
        if (!ptr) return false;

        long P = sysconf(_SC_PAGESIZE);
        if (P <= 0) return false;

        size_t H = sizeof(Header);
        auto hdr = reinterpret_cast<Header *>(
            static_cast<char *>(ptr) - H);

        if (hdr->magic != GUARD_MAGIC) return false;

        char *base = static_cast<char *>(hdr->base);
        char *rw_start = base + P;
        char *rw_end = base + hdr->total_bytes - P;
        return ptr >= rw_start && ptr < rw_end;
    }

    void guarded_free(void *ptr) {
        if (!ptr) return;
        if (!is_guarded_ptr(ptr)) {
            free(ptr);
            return;
        }
        auto hdr = reinterpret_cast<Header *>(
            static_cast<char *>(ptr) - sizeof(Header));
        munmap(hdr->base, hdr->total_bytes);
    }
} // namespace ccoip::alloc::internal
