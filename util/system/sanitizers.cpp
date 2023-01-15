#include "sanitizers.h"
#include "thread.h"

#if defined(_asan_enabled_)
extern "C" {
    void __sanitizer_start_switch_fiber(void** fake_stack_save, const void* bottom, size_t size);
    void __sanitizer_finish_switch_fiber(void* fake_stack_save, const void** old_bottom, size_t* old_size);
}
#endif

#if defined(_tsan_enabled_)
    #if defined(__clang_major__) && (__clang_major__ >= 9)
extern "C" {
    void* __tsan_get_current_fiber(void);
    void* __tsan_create_fiber(unsigned flags);
    void __tsan_destroy_fiber(void* fiber);
    void __tsan_switch_to_fiber(void* fiber, unsigned flags);
    void __tsan_set_fiber_name(void* fiber, const char* name);
}
    #else
namespace {
    void* __tsan_get_current_fiber(void) {
        return nullptr;
    }
    void* __tsan_create_fiber(unsigned) {
        return nullptr;
    }
    void __tsan_destroy_fiber(void*) {
    }
    void __tsan_switch_to_fiber(void*, unsigned) {
    }
    void __tsan_set_fiber_name(void*, const char*) {
    }
}
    #endif
#endif

using namespace NSan;

TFiberContext::TFiberContext() noexcept
    : Token_(nullptr)
    , IsMainFiber_(true)
#if defined(_tsan_enabled_)
    , CurrentTSanFiberContext_(__tsan_get_current_fiber())
#endif
{
    TCurrentThreadLimits sl;

    Stack_ = sl.StackBegin;
    Len_ = sl.StackLength;

#if defined(_tsan_enabled_)
    static constexpr char MainFiberName[] = "main_fiber";
    __tsan_set_fiber_name(CurrentTSanFiberContext_, MainFiberName);
#endif
}

TFiberContext::TFiberContext(const void* stack, size_t len, const char* contName) noexcept
    : Token_(nullptr)
    , Stack_(stack)
    , Len_(len)
    , IsMainFiber_(false)
#if defined(_tsan_enabled_)
    , CurrentTSanFiberContext_(__tsan_create_fiber(/*flags =*/0))
#endif
{
    (void)contName;
#if defined(_tsan_enabled_)
    __tsan_set_fiber_name(CurrentTSanFiberContext_, contName);
#endif
}

TFiberContext::~TFiberContext() noexcept {
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

void TFiberContext::BeforeFinish() noexcept {
#if defined(_asan_enabled_)
    __sanitizer_start_switch_fiber(nullptr, nullptr, 0);
#else
    (void)Token_;
    (void)Stack_;
    (void)Len_;
#endif
}

void TFiberContext::BeforeSwitch(TFiberContext* old) noexcept {
#if defined(_asan_enabled_)
    __sanitizer_start_switch_fiber(old ? &old->Token_ : nullptr, (char*)Stack_, Len_);
#else
    (void)old;
#endif

#if defined(_tsan_enabled_)
    __tsan_switch_to_fiber(CurrentTSanFiberContext_, /*flags =*/0);
#endif
}

void TFiberContext::AfterSwitch() noexcept {
#if defined(_asan_enabled_)
    __sanitizer_finish_switch_fiber(Token_, nullptr, nullptr);
#endif
}

void TFiberContext::AfterStart() noexcept {
#if defined(_asan_enabled_)
    __sanitizer_finish_switch_fiber(nullptr, nullptr, nullptr);
#endif
}

#if defined(_tsan_enabled_)
extern "C" {
    // This function should not be directly exposed in headers
    // due to signature variations among contrib headers.
    void AnnotateBenignRaceSized(const char* file, int line,
                                 const volatile void* address,
                                 size_t size,
                                 const char* description);
}
void NSan::AnnotateBenignRaceSized(const char* file, int line,
                                   const volatile void* address,
                                   size_t size,
                                   const char* description) noexcept {
    ::AnnotateBenignRaceSized(file, line, address, size, description);
}
#endif
