#pragma once

#include <functional>

namespace ccoip::guard_utils {
    /// RAII guard for a critical section;
    /// Wraps two functions: one for entering the critical section, and one for exiting it.
    /// The constructor calls the enter function, and the destructor calls the exit function.
    class phase_guard {
        std::function<void()> enter;
        std::function<void()> exit;

    public:
        phase_guard(const std::function<void()> &enter, const std::function<void()> &exit) : enter(enter), exit(exit) {
            enter();
        }

        ~phase_guard() {
            exit();
        }
    };
}; // namespace guard_utils
