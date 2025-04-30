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
        void *base;             // start of the mmap region
        size_t total_bytes;     // total bytes allocated via mmap
        uint64_t magic;         // sanity check
    };

    void *guarded_malloc(size_t size) {
        if (size == 0) return nullptr;

        // Page size
        const long P = sysconf(_SC_PAGESIZE);
        assert(P > 0);

        // Number of payload pages (round up)
        size_t pages = (size + P - 1) / P;

        // Layout: guard | header | payload... | guard
        size_t total_pages = pages + 3;
        size_t total_bytes = total_pages * P;

        // mmap entire region with no permissions
        void *base = mmap(nullptr, total_bytes,
                          PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS,
                          -1, 0);
        if (base == MAP_FAILED) {
            return nullptr;
        }

        // First, allow header page RW to initialize
        void *hdr_page = static_cast<char *>(base) + P;
        if (mprotect(hdr_page, P, PROT_READ | PROT_WRITE) != 0) {
            munmap(base, total_bytes);
            return nullptr;
        }

        // Initialize header
        auto hdr = reinterpret_cast<Header *>(hdr_page);
        hdr->base = base;
        hdr->total_bytes = total_bytes;
        hdr->magic = GUARD_MAGIC;

        // Protect header page read-only
        mprotect(hdr_page, P, PROT_READ);

        // Payload region RW
        void *payload = static_cast<char *>(base) + 2 * P;
        if (mprotect(payload, pages * P, PROT_READ | PROT_WRITE) != 0) {
            munmap(base, total_bytes);
            return nullptr;
        }

        // Guard pages remain PROT_NONE
        // [base + 0*P .. P) and [base + (2+pages)*P .. total_bytes) are still none

        // Return pointer at start of payload
        return payload;
    }

    void guarded_free(void *ptr) {
        if (!ptr) return;

        // Page size to find header
        const long P = sysconf(_SC_PAGESIZE);
        assert(P > 0);

        // Header page is one page before payload
        auto hdr_page = static_cast<char *>(ptr) - P;
        auto hdr = reinterpret_cast<Header *>(hdr_page);

        // Verify canary
        assert(hdr->magic == GUARD_MAGIC && "Heap corruption detected: header overwritten");

        // Unmap the entire region
        munmap(hdr->base, hdr->total_bytes);
    }
} // namespace ccoip::alloc::internal
