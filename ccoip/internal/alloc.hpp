#pragma once

#include <cstdlib>

namespace ccoip::alloc {
    void *malloc(size_t size);

    void free(void *ptr);

    namespace internal {
        void *guarded_malloc(size_t size);
        void guarded_free(void *ptr);
        bool is_guarded_ptr(void *ptr) noexcept;
    }
}
