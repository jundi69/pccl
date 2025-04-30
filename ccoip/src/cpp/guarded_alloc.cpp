#pragma once

#include <unistd.h>
#include <sys/mman.h>
#include <cstdint>
#include <cstddef>
#include <cassert>


namespace ccoip::alloc::internal {
    struct Header {
        void *base; // start of the mmap region
        size_t total_bytes; // total bytes allocated via mmap
    };

    void *guarded_malloc(size_t size) {
        if (size == 0) return nullptr;

        // 1) figure out pages
        long pagesize = sysconf(_SC_PAGESIZE);
        assert(pagesize > 0);
        size_t payload_pages = (size + pagesize - 1) / pagesize;

        // 2) map header page + payload pages in one shot
        size_t total_pages = 1 + payload_pages;
        size_t total_bytes = total_pages * pagesize;
        void *base = mmap(nullptr,
                          total_bytes,
                          PROT_READ | PROT_WRITE, // RW while we init
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(base != MAP_FAILED);

        auto *p = static_cast<char *>(base);
        char *header_page = p;
        char *payload = p + pagesize;

        // 3) write our Header into the first page
        auto h = reinterpret_cast<Header *>(header_page);
        h->base = base;
        h->total_bytes = total_bytes;

        // 4) now protect the header page completely
        mprotect(header_page, pagesize, PROT_NONE);

        // 5) leave the payload pages RW
        mprotect(payload, payload_pages * pagesize,
                 PROT_READ | PROT_WRITE);

        // 6) hand out the user pointer
        return payload;
    }

    void guarded_free(void *ptr) {
        if (!ptr) return;

        long pagesize = sysconf(_SC_PAGESIZE);
        assert(pagesize > 0);

        auto *payload = static_cast<char *>(ptr);
        char *header_page = payload - pagesize;

        // make header RW again so we can read our metadata
        mprotect(header_page, pagesize, PROT_READ | PROT_WRITE);

        auto h = reinterpret_cast<Header *>(header_page);
        void *base = h->base;
        size_t total_bytes = h->total_bytes;

        // unmap everything
        munmap(base, total_bytes);
    }
} // namespace ccoip::alloc::internal
