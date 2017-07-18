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

using namespace NSan;

TFiberContext::TFiberContext() noexcept
    : Token_(nullptr)
{
    TCurrentThreadLimits sl;

    Stack_ = sl.StackBegin;
    Len_ = sl.StackLength;
}

TFiberContext::TFiberContext(const void* stack, size_t len) noexcept
    : Token_(nullptr)
    , Stack_(stack)
    , Len_(len)
{
}

TFiberContext::~TFiberContext() noexcept {
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
