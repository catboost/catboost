#pragma once

#include "defaults.h"

extern "C" { // sanitizers API

#if defined(_asan_enabled_)
void __lsan_ignore_object(const void* p);
#endif

#if defined(_msan_enabled_)
void __msan_unpoison(const volatile void* a, size_t size);
void __msan_poison(const volatile void* a, size_t size);
void __msan_check_mem_is_initialized(const volatile void* x, size_t size);
#endif

}; // sanitizers API

namespace NSan {
    class TFiberContext {
    public:
        TFiberContext() noexcept;
        TFiberContext(const void* stack, size_t len, const char* contName) noexcept;

        ~TFiberContext() noexcept;

        void BeforeFinish() noexcept;
        void BeforeSwitch() noexcept;
        void AfterSwitch() noexcept;

        static void AfterStart() noexcept;

    private:
        void* Token_;
        const void* Stack_;
        size_t Len_;

#if defined(_tsan_enabled_)
        void* const CurrentTSanFiberContext_;
        const bool WasFiberCreated_;
#endif
    };

    // Returns plain if no sanitizer enabled or sanitized otherwise
    // Ment to be used in test code for constants (timeouts, etc)
    template <typename T>
    inline constexpr static T PlainOrUnderSanitizer(T plain, T sanitized) noexcept {
#if defined(_tsan_enabled_) || defined(_msan_enabled_) || defined(_asan_enabled_)
        Y_UNUSED(plain);
        return sanitized;
#else
        Y_UNUSED(sanitized);
        return plain;
#endif
    }

    // Determines if asan present
    inline constexpr static bool ASanIsOn() noexcept {
#if defined(_asan_enabled_)
        return true;
#else
        return false;
#endif
    }

    // Determines if tsan present
    inline constexpr static bool TSanIsOn() noexcept {
#if defined(_tsan_enabled_)
        return true;
#else
        return false;
#endif
    }

    // Determines if msan present
    inline constexpr static bool MSanIsOn() noexcept {
#if defined(_msan_enabled_)
        return true;
#else
        return false;
#endif
    }

    // Make memory region fully initialized (without changing its contents).
    inline static void Unpoison(const volatile void* a, size_t size) noexcept {
#if defined(_msan_enabled_)
        __msan_unpoison(a, size);
#else
        Y_UNUSED(a);
        Y_UNUSED(size);
#endif
    }

    // Make memory region fully uninitialized (without changing its contents).
    // This is a legacy interface that does not update origin information. Use __msan_allocated_memory() instead.
    inline static void Poison(const volatile void* a, size_t size) noexcept {
#if defined(_msan_enabled_)
        __msan_poison(a, size);
#else
        Y_UNUSED(a);
        Y_UNUSED(size);
#endif
    }

    // Checks that memory range is fully initialized, and reports an error if it is not.
    inline static void CheckMemIsInitialized(const volatile void* a, size_t size) noexcept {
#if defined(_msan_enabled_)
        __msan_check_mem_is_initialized(a, size);
#else
        Y_UNUSED(a);
        Y_UNUSED(size);
#endif
    }

    inline static void MarkAsIntentionallyLeaked(const void* ptr) noexcept {
#if defined(_asan_enabled_)
        __lsan_ignore_object(ptr);
#else
        Y_UNUSED(ptr);
#endif
    }
}
