#include "alloc.hpp"

#include <unistd.h>     // sysconf
#include <sys/mman.h>   // mmap, mprotect, munmap
#include <cstdint>      // uintptr_t
#include <cerrno>

namespace ccoip {
    namespace alloc {
        namespace internal {
            struct Header {
                void *base;
                std::size_t total;
            };

            void *guarded_malloc(std::size_t size) {
                if (size == 0)
                    return nullptr;

                const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
                std::size_t n = (size + page - 1) / page;
                std::size_t offset = n * page - size;

                if (offset < sizeof(Header)) {
                    n += 1;
                    offset += page;
                }

                const std::size_t total_bytes = (n + 2) * page;

                void *base = mmap(nullptr,
                                  total_bytes,
                                  PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS,
                                  -1, 0);
                if (base == MAP_FAILED) {
                    return nullptr;
                }

                mprotect(base, page, PROT_NONE);
                mprotect(static_cast<char *>(base) + (1 + n) * page, page, PROT_NONE);

                void *ptr = static_cast<char *>(base) + page + offset;

                auto *hdr = reinterpret_cast<Header *>(
                    static_cast<char *>(ptr) - sizeof(Header));
                hdr->base = base;
                hdr->total = total_bytes;

                return ptr;
            }

            void guarded_free(void *ptr) {
                if (!ptr) return;
                // pull back our header
                auto *hdr = reinterpret_cast<Header *>(
                    static_cast<char *>(ptr) - sizeof(Header));
                munmap(hdr->base, hdr->total);
            }
        } // namespace internal
    } // namespace alloc
} // namespace ccoip
