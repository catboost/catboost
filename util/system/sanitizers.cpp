#include "sanitizers.h"
#include "thread.h"

using namespace NSan;

#if defined(_tsan_enabled_)
    #if defined(__clang_major__) && (__clang_major__ >= 9)
extern "C" { // sanitizers API

        #if defined(_tsan_enabled_)
    void* __tsan_create_fiber(unsigned flags);
    void __tsan_set_fiber_name(void* fiber, const char* name);
        #endif

} // sanitizers API
    #else
namespace {
    void* __tsan_create_fiber(unsigned) {
        return nullptr;
    }
    void __tsan_set_fiber_name(void*, const char*) {
    }
} // namespace
    #endif
#endif

TFiberContext::TFiberContext() noexcept
    : Token_(nullptr)
    , IsMainFiber_(true)
#if defined(_tsan_enabled_)
    , CurrentTSanFiberContext_(__tsan_get_current_fiber())
#endif
{
    TCurrentThreadLimits sl;
    (void)Token_;
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
