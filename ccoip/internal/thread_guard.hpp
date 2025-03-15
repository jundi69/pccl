#pragma once

#ifdef _MSC_VER
#define FUNC_SIGNATURE() __FUNCSIG__
#else
#define FUNC_SIGNATURE() __PRETTY_FUNCTION__
#endif

#define THREAD_GUARD(thread_id) \
    if (std::this_thread::get_id() != thread_id) { \
        LOG(FATAL) << "Function " << FUNC_SIGNATURE() << " must be called from the main thread! This is a fatal bug!"; \
        std::terminate(); \
    }
