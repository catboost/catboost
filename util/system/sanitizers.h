#pragma once

#include "defaults.h"

extern "C" { // sanitizers API

#if defined(_asan_enabled_)
    void __lsan_ignore_object(const void* p);
    void __sanitizer_start_switch_fiber(void** fake_stack_save, const void* bottom, size_t size);
    void __sanitizer_finish_switch_fiber(void* fake_stack_save, const void** old_bottom, size_t* old_size);
#endif

#if defined(_msan_enabled_)
    void __msan_unpoison(const volatile void* a, size_t size);
    void __msan_poison(const volatile void* a, size_t size);
    void __msan_check_mem_is_initialized(const volatile void* x, size_t size);
#endif

#if defined(_tsan_enabled_)
    void __tsan_acquire(void* a);
    void __tsan_release(void* a);
    void* __tsan_get_current_fiber(void);
    void __tsan_destroy_fiber(void* fiber);
    void __tsan_switch_to_fiber(void* fiber, unsigned flags);
#endif

} // sanitizers API

namespace NSan {
    class TFiberContext {
    public:
        TFiberContext() noexcept;
        TFiberContext(const void* stack, size_t len, const char* contName) noexcept;

        ~TFiberContext() noexcept {
            if (!IsMainFiber_) {
#if defined(_asan_enabled_)
                if (Token_) {
                    // destroy saved FakeStack
                    void* activeFakeStack = nullptr;
                    const void* activeStack = nullptr;
                    size_t activeStackSize = 0;
                    __sanitizer_start_switch_fiber(&activeFakeStack, (char*)Stack_, Len_);
                    __sanitizer_finish_switch_fiber(Token_, &activeStack, &activeStackSize);
                    __sanitizer_start_switch_fiber(nullptr, activeStack, activeStackSize);
                    __sanitizer_finish_switch_fiber(activeFakeStack, nullptr, nullptr);
                }
#endif
#if defined(_tsan_enabled_)
                __tsan_destroy_fiber(CurrentTSanFiberContext_);
#endif
            }
        }

        Y_FORCE_INLINE void BeforeSwitch(TFiberContext* old) noexcept {
#if defined(_asan_enabled_)
            __sanitizer_start_switch_fiber(old ? &old->Token_ : nullptr, (char*)Stack_, Len_);
#else
            (void)old;
#endif

#if defined(_tsan_enabled_)
            __tsan_switch_to_fiber(CurrentTSanFiberContext_, /*flags =*/0);
#endif
        }
        void AfterSwitch() noexcept {
#if defined(_asan_enabled_)
            __sanitizer_finish_switch_fiber(Token_, nullptr, nullptr);
#endif
        }

    private:
        void* Token_;
        const void* Stack_;
        size_t Len_;

        const bool IsMainFiber_;
#if defined(_tsan_enabled_)
        void* const CurrentTSanFiberContext_;
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

#if defined(_tsan_enabled_)
    // defined in .cpp to avoid exposing problematic C-linkage version of AnnotateBenignRaceSized(...)
    void AnnotateBenignRaceSized(const char* file, int line,
                                 const volatile void* address,
                                 size_t size,
                                 const char* description) noexcept;
#else
    inline static void AnnotateBenignRaceSized(const char* file, int line,
                                               const volatile void* address,
                                               size_t size,
                                               const char* description) noexcept {
        Y_UNUSED(file);
        Y_UNUSED(line);
        Y_UNUSED(address);
        Y_UNUSED(size);
        Y_UNUSED(description);
    }
#endif

    inline static void Acquire(void* a) {
#if defined(_tsan_enabled_)
        __tsan_acquire(a);
#else
        Y_UNUSED(a);
#endif
    }

    inline static void Release(void* a) {
#if defined(_tsan_enabled_)
        __tsan_release(a);
#else
        Y_UNUSED(a);
#endif
    }
}
