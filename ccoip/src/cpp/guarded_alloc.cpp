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
        size_t total_bytes; // size of that region
        uint64_t magic; // == GUARD_MAGIC if this really is ours
    };

    void *guarded_malloc(size_t size) {
        if (size == 0) return nullptr;

        long P = sysconf(_SC_PAGESIZE);
        assert(P > 0);

        // figure out how many pages we need
        size_t payload_pages = (size + P - 1) / P;
        // one header page + payload_pages
        size_t total_pages = 1 + payload_pages;
        size_t total_bytes = total_pages * P;

        // map it all RW initially
        void *base = mmap(nullptr, total_bytes,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS,
                          -1, 0);
        assert(base != MAP_FAILED);

        auto *p = static_cast<char *>(base);
        auto *hdr = reinterpret_cast<Header *>(p);
        hdr->base = base;
        hdr->total_bytes = total_bytes;
        hdr->magic = GUARD_MAGIC;

        // now protect the header page
        mprotect(p, P, PROT_NONE);
        // leave the payload pages RW
        mprotect(p + P, payload_pages * P, PROT_READ | PROT_WRITE);

        // hand back the start of the payload
        return p + P;
    }

    bool is_guarded_ptr(void *ptr) noexcept {
        if (!ptr) return false;
        long P = sysconf(_SC_PAGESIZE);
        if (P <= 0) return false;

        // the header must sit exactly one page below ptr
        void *hdr_page = static_cast<char *>(ptr) - P;

        // try to give ourselves read access to that page…
        // if it fails, it wasn’t mapped by us
        if (mprotect(hdr_page, P, PROT_READ | PROT_WRITE) != 0) {
            if (errno == ENOMEM || errno == EACCES) {
                return false;
            }
            // something odd—still consider it not-ours
            return false;
        }

        // now read the magic, then re-protect
        auto *hdr = static_cast<Header *>(hdr_page);
        bool ok = (hdr->magic == GUARD_MAGIC);
        // restore the “no access” on header
        mprotect(hdr_page, P, PROT_NONE);
        return ok;
    }

    void guarded_free(void *ptr) {
        if (!ptr) return;

        if (!is_guarded_ptr(ptr)) {
            // fall back to the real C heap
            std::free(ptr);
            return;
        }

        // it *is* ours, so unmap the whole region
        long P = sysconf(_SC_PAGESIZE);
        auto *hdr_page = static_cast<char*>(ptr) - P;
        // make header RW so we can read base/size
        mprotect(hdr_page, P, PROT_READ|PROT_WRITE);
        auto *hdr = reinterpret_cast<Header*>(hdr_page);
        munmap(hdr->base, hdr->total_bytes);
    }
} // namespace ccoip::alloc::internal
