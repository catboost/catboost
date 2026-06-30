#pragma once

#include "align.h"
#include "defaults.h"
#include "compiler.h"

#include <util/generic/array_ref.h>
#include <util/generic/utility.h>
#include <util/generic/yexception.h>

#define STACK_ALIGN (8 * PLATFORM_DATA_ALIGN)

#if defined(_x86_64_) || defined(_i386_) || defined(_arm_) || defined(_ppc64_)
    #define STACK_GROW_DOWN 1
#else
    #error todo
#endif

#if defined(__clang_major__) && (__clang_major__ >= 9) && (defined(_asan_enabled_) || defined(_tsan_enabled_))
    #include "sanitizers.h"
    #define USE_SANITIZER_CONTEXT
#endif

/*
 * switch method
 */
#if defined(thread_sanitizer_enabled) && defined(_darwin_)
    #define _XOPEN_SOURCE 700
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#if defined(_bionic_) || defined(__IOS__)
    #define USE_GENERIC_CONT
#elif defined(_cygwin_)
    #define USE_UCONTEXT_CONT
#elif defined(_win_)
    #define USE_FIBER_CONT
#elif (defined(_i386_) || defined(_x86_64_) || defined(_arm64_)) && !defined(_k1om_)
    #define USE_JUMP_CONT
#else
    #define USE_UCONTEXT_CONT
#endif

#if defined(USE_JUMP_CONT)
    #if defined(_arm64_)
        #include "context_aarch64.h"
    #else
        #include "context_x86.h"
    #endif
#endif

#if defined(USE_UCONTEXT_CONT)
    #include <ucontext.h>
#endif

struct ITrampoLine {
    virtual ~ITrampoLine() = default;

    virtual void DoRun();
    virtual void DoRunNaked();
};

struct TContClosure {
    ITrampoLine* TrampoLine;
    TArrayRef<char> Stack;
    const char* ContName = nullptr;
};

#if defined(USE_UCONTEXT_CONT)
class TContMachineContext {
    typedef void (*ucontext_func_t)(void);

public:
    inline TContMachineContext() {
        getcontext(&Ctx_);
    }

    inline TContMachineContext(const TContClosure& c) {
        getcontext(&Ctx_);

        Ctx_.uc_link = 0;
        Ctx_.uc_stack.ss_sp = (void*)c.Stack.data();
        Ctx_.uc_stack.ss_size = c.Stack.size();
        Ctx_.uc_stack.ss_flags = 0;

        extern void ContextTrampoLine(void* arg);
        makecontext(&Ctx_, (ucontext_func_t)ContextTrampoLine, 1, c.TrampoLine);
    }

    inline ~TContMachineContext() {
    }

    inline void SwitchTo(TContMachineContext* next) noexcept {
        swapcontext(&Ctx_, &next->Ctx_);
    }

private:
    ucontext_t Ctx_;
};
#endif

#if defined(USE_GENERIC_CONT)
class TContMachineContext {
    struct TImpl;

public:
    TContMachineContext();
    TContMachineContext(const TContClosure& c);

    ~TContMachineContext();

    void SwitchTo(TContMachineContext* next) noexcept;

private:
    THolder<TImpl> Impl_;
};
#endif

#if defined(USE_FIBER_CONT)
class TContMachineContext {
public:
    TContMachineContext();
    TContMachineContext(const TContClosure& c);
    ~TContMachineContext();

    void SwitchTo(TContMachineContext* next) noexcept;

private:
    void* Fiber_;
    bool MainFiber_;
};
#endif

#if defined(USE_JUMP_CONT)
class TContMachineContext {
public:
    inline TContMachineContext() {
        Zero(Buf_);
    }

    TContMachineContext(const TContClosure& c);

    inline ~TContMachineContext() = default;

    void SwitchTo(TContMachineContext* next) noexcept;

private:
    __myjmp_buf Buf_;

    #if defined(USE_SANITIZER_CONTEXT)

    struct TSan: public ITrampoLine, public ::NSan::TFiberContext {
        TSan() noexcept;
        TSan(const TContClosure& c) noexcept;

        void DoRunNaked() override;

        ITrampoLine* TL;
    };

    TSan San_;
    #endif
};
#endif

static inline size_t MachineContextSize() noexcept {
    return sizeof(TContMachineContext);
}

/*
 * be polite
 */
#if !defined(FROM_CONTEXT_IMPL)
    #undef USE_SANITIZER_CONTEXT
    #undef USE_JUMP_CONT
    #undef USE_FIBER_CONT
    #undef USE_GENERIC_CONT
    #undef USE_UCONTEXT_CONT
    #undef PROGR_CNT
    #undef STACK_CNT
    #undef EXTRA_PUSH_ARGS
#endif

#if defined(_darwin_) && defined(thread_sanitizer_enabled)
    #pragma clang diagnostic pop
    #undef _XOPEN_SOURCE
#endif

struct TExceptionSafeContext: public TContMachineContext {
    using TContMachineContext::TContMachineContext;

    void SwitchTo(TExceptionSafeContext* to) noexcept;

#if defined(_unix_)
    void* Buf_[2] = {nullptr, nullptr};
#endif
};
