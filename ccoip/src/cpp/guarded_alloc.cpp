#pragma once

#include <unistd.h>
#include <sys/mman.h>
#include <cstdint>
#include <cstddef>
#include <cassert>


namespace ccoip::alloc::internal {
    // Canary value to detect header corruption
    static constexpr uint64_t GUARD_MAGIC = 0xDEADC0DECAFEBABEULL;

    struct Header {
        void *base; // start of the mmap region
        size_t total_bytes; // total bytes allocated via mmap
        uint64_t magic; // sanity check
    };

    inline void *guarded_malloc(size_t size) {
        if (size == 0) return nullptr;

        // Page size
        const long P = sysconf(_SC_PAGESIZE);
        assert(P > 0);

        // Number of payload pages (round up)
        size_t pages = (size + P - 1) / P;

        // Total pages = guard + header + payload + guard
        size_t total_pages = pages + 3;
        size_t total_bytes = total_pages * P;

        // mmap entire region, start with no permissions
        void *base = mmap(nullptr, total_bytes,
                          PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS,
                          -1, 0);
        if (base == MAP_FAILED) {
            return nullptr;
        }

        // Layout:
        // [0] guard page (PROT_NONE)
        // [1] header page (PROT_READ only)
        // [2..2+pages-1] payload pages (PROT_READ|WRITE)
        // [2+pages] guard page (PROT_NONE)

        // Header page: read-only
        mprotect((char *) base + P, P, PROT_READ);
        // Payload pages: R/W
        mprotect((char *) base + 2 * P, pages * P, PROT_READ | PROT_WRITE);

        // Compute user pointer at start of payload region
        void *user_ptr = (char *) base + 2 * P;

        // Initialize header in the header page
        auto hdr = reinterpret_cast<Header *>((char *) base + P);
        hdr->base = base;
        hdr->total_bytes = total_bytes;
        hdr->magic = GUARD_MAGIC;

        return user_ptr;
    }

    inline void guarded_free(void *ptr) {
        if (!ptr) return;

        // Page size to find header
        const long P = sysconf(_SC_PAGESIZE);
        assert(P > 0);

        // Header is one page before the payload region
        auto hdr = reinterpret_cast<Header *>((char *) ptr - P);
        // Check magic
        assert(hdr->magic == GUARD_MAGIC && "Heap corruption detected: header overwritten");

        // Unmap the entire region
        munmap(hdr->base, hdr->total_bytes);
    }
} // namespace ccoip::alloc::internal
