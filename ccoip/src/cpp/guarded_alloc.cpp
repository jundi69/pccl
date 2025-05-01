#pragma once

#include <unistd.h>
#include <sys/mman.h>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstdlib>
#include <sys/errno.h>


namespace ccoip::alloc::internal {
    // Magic value to tag all of *our* allocations
    static constexpr uint64_t GUARD_MAGIC = 0xDEADC0DECAFEBABEULL;

    struct Header {
        void *base; // start of the mmap region
        size_t total_bytes; // total bytes mapped
        size_t payload_pages; // how many pages worth of payload
        uint64_t magic; // == GUARD_MAGIC
    };

    void *guarded_malloc(size_t size) {
        if (size == 0) return nullptr;

        // --------------------------------------------------------------------
        // 1) Figure pages
        // --------------------------------------------------------------------
        long P = sysconf(_SC_PAGESIZE);
        assert(P > 0);

        size_t pages = (size + P - 1) / P; // how many pages we need
        // always sandwich with one header + one trailer guard
        size_t total_pages = 1 + pages + 1;
        size_t total_bytes = total_pages * P;

        // --------------------------------------------------------------------
        // 2) mmap everything RW to initialize
        // --------------------------------------------------------------------
        void *base = mmap(nullptr, total_bytes,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS,
                          -1, 0);
        assert(base != MAP_FAILED);

        // --------------------------------------------------------------------
        // 3) Initialize header on the *first* page
        // --------------------------------------------------------------------
        auto *hdr = reinterpret_cast<Header *>(base);
        hdr->base = base;
        hdr->total_bytes = total_bytes;
        hdr->payload_pages = pages;
        hdr->magic = GUARD_MAGIC;

        // --------------------------------------------------------------------
        // 4) Protect header & trailer, leave payload RW
        // --------------------------------------------------------------------
        // header page:
        mprotect(base, P, PROT_NONE);
        // trailer page:
        mprotect(static_cast<char *>(base) + (1 + pages) * P, P, PROT_NONE);
        // payload pages:
        mprotect(static_cast<char *>(base) + P, pages * P, PROT_READ | PROT_WRITE);

        // --------------------------------------------------------------------
        // 5) Compute user pointer
        //    - If it's a single‐page alloc, place it at end of that page so an
        //      overflow of +1 immediately hits the trailer guard.
        //    - Otherwise point to the start of the payload region.
        // --------------------------------------------------------------------
        void *payload_start = static_cast<char *>(base) + P;
        void *user_ptr;
        if (pages == 1 && size < size_t(P)) {
            // end‐aligned
            user_ptr = static_cast<char *>(payload_start) + (P - size);
        } else {
            // start‐aligned
            user_ptr = payload_start;
        }

        return user_ptr;
    }

    bool is_guarded_ptr(void *ptr) noexcept {
        if (!ptr) return false;
        long P = sysconf(_SC_PAGESIZE);
        if (P <= 0) return false;

        // Round ptr down to the start of its page, then back up one page → header page.
        auto *maybe_hdr = reinterpret_cast<char *>(ptr);
        maybe_hdr = reinterpret_cast<char *>((uintptr_t(maybe_hdr) & ~(P - 1)) - P);

        // Try to mprotect it read/write so we can inspect our magic.
        if (mprotect(maybe_hdr, P, PROT_READ | PROT_WRITE) != 0) {
            // Not one of ours → fall back to “false”
            return false;
        }

        // Read the header and restore PROT_NONE
        auto *hdr = reinterpret_cast<Header *>(maybe_hdr);
        bool ok = (hdr->magic == GUARD_MAGIC);
        mprotect(maybe_hdr, P, PROT_NONE);
        return ok;
    }

    void guarded_free(void *ptr) {
        if (!ptr) return;

        // --------------------------------------------------------------------
        // Try to see if this was one of ours:
        //   - compute where the header page *must* sit
        //     (it's always the very first page of the mapping)
        //   - mprotect it to RW so we can read our magic
        // --------------------------------------------------------------------
        long P = sysconf(_SC_PAGESIZE);
        assert(P > 0);

        // figure out a candidate header page by rounding down to page
        // boundary, then back up one page.  This works because:
        //  - Any user_ptr in the payload region lies ≥ base+P,
        //  - align_down(ptr,P) gives you the start of *some* payload page,
        //  - subtracting P gives you the header page start.
        auto *maybe_hdr = reinterpret_cast<char *>(ptr);
        maybe_hdr = reinterpret_cast<char *>((uintptr_t(maybe_hdr) & ~(P - 1)) - P);

        // try to make it R/W so we can inspect
        if (mprotect(maybe_hdr, P, PROT_READ | PROT_WRITE) != 0) {
            // not one of ours → fall back to libc free
            std::free(ptr);
            return;
        }

        // read header
        auto *hdr = reinterpret_cast<Header *>(maybe_hdr);
        if (hdr->magic != GUARD_MAGIC) {
            // not ours
            mprotect(maybe_hdr, P, PROT_NONE);
            std::free(ptr);
            return;
        }

        // it's ours!  unmap the entire region
        munmap(hdr->base, hdr->total_bytes);
    }
} // namespace ccoip::alloc::internal
