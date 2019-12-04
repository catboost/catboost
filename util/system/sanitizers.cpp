#include "sanitizers.h"
#include "thread.h"

#if defined(_asan_enabled_)
extern "C" {
void __sanitizer_start_switch_fiber(void** fake_stack_save, const void* bottom, size_t size);
#if defined(__clang_major__) && (__clang_major__ >= 4)
#define NEW_ASAN_IFACE
void __sanitizer_finish_switch_fiber(void* fake_stack_save, const void** old_bottom, size_t* old_size);
#else
#undef NEW_ASAN_IFACE
void __sanitizer_finish_switch_fiber(void* fake_stack_save);
#endif
}
#endif

#if defined(_tsan_enabled_)
#if defined(__clang_major__) && (__clang_major__ >= 9)
extern "C" {
void *__tsan_get_current_fiber(void);
void *__tsan_create_fiber(unsigned flags);
void __tsan_destroy_fiber(void *fiber);
void __tsan_switch_to_fiber(void *fiber, unsigned flags);
void __tsan_set_fiber_name(void *fiber, const char *name);
}
#else
namespace {
void* __tsan_get_current_fiber(void) { return nullptr; }
void* __tsan_create_fiber(unsigned) { return nullptr; }
void __tsan_destroy_fiber(void*) {}
void __tsan_switch_to_fiber(void*, unsigned) {}
void __tsan_set_fiber_name(void*, const char*) {}
}
#endif
#endif

using namespace NSan;

TFiberContext::TFiberContext() noexcept
    : Token_(nullptr)
#if defined(_tsan_enabled_)
    , CurrentTSanFiberContext_(__tsan_get_current_fiber())
    , WasFiberCreated_(false)
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
#if defined(_tsan_enabled_)
    , CurrentTSanFiberContext_(__tsan_create_fiber(/*flags =*/ 0))
    , WasFiberCreated_(true)
#endif
{
    (void)contName;
#if defined(_tsan_enabled_)
    __tsan_set_fiber_name(CurrentTSanFiberContext_, contName);
#endif
}

TFiberContext::~TFiberContext() noexcept {
#if defined(_tsan_enabled_)
    if (WasFiberCreated_) {
        __tsan_destroy_fiber(CurrentTSanFiberContext_);
    }
#endif
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

void TFiberContext::BeforeSwitch() noexcept {
#if defined(_asan_enabled_)
    __sanitizer_start_switch_fiber(&Token_, (char*)Stack_, Len_);
#endif

#if defined(_tsan_enabled_)
    __tsan_switch_to_fiber(CurrentTSanFiberContext_, /*flags =*/ 0);
#endif
}

void TFiberContext::AfterSwitch() noexcept {
#if defined(_asan_enabled_)
#if defined(NEW_ASAN_IFACE)
    __sanitizer_finish_switch_fiber(Token_, nullptr, nullptr);
#else
    __sanitizer_finish_switch_fiber(Token_);
#endif
#endif
}

void TFiberContext::AfterStart() noexcept {
#if defined(_asan_enabled_)
#if defined(NEW_ASAN_IFACE)
    __sanitizer_finish_switch_fiber(nullptr, nullptr, nullptr);
#else
    __sanitizer_finish_switch_fiber(nullptr);
#endif
#endif
}
