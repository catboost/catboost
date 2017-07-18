#include "compiler.h"
#include "defaults.h"
#include "event.h"
#include "thread.h"

#include <cstdlib> //for abort()

#if defined(_win_)
#include "winint.h"
#endif

#include <util/stream/output.h>
#include <util/generic/yexception.h>

#define FROM_CONTEXT_IMPL
#include "context.h"

static inline void Run(void* arg) {
    try {
        ((ITrampoLine*)arg)->DoRun();
    } catch (...) {
        Cerr << "Uncaught exception in coroutine: " << CurrentExceptionMessage() << "\n";
    }

    abort();
}

#if defined(USE_JUMP_CONT)
extern "C" void __mylongjmp(__myjmp_buf env, int val) __attribute__((__noreturn__));
extern "C" int __mysetjmp(__myjmp_buf env) __attribute__((__returns_twice__));

namespace {
    class TStack {
    public:
        inline TStack(TMemRegion range) noexcept
#if defined(STACK_GROW_DOWN)
            : Data_(range.Data() + range.Size())
#else
            : Data_(range.Data() + STACK_ALIGN)
#endif
        {
            ReAlign();
        }

        inline ~TStack() = default;

        inline void ReAlign() noexcept {
            Data_ = AlignStackPtr(Data_);
        }

        template <class T>
        inline void Push(T t) noexcept {
#if defined(STACK_GROW_DOWN)
            Data_ -= sizeof(T);
            *((T*)Data_) = t;
#else
            *((T*)Data_) = t;
            Data_ += sizeof(T);
#endif
        }

        inline char* StackPtr() noexcept {
            return Data_;
        }

    private:
        static inline char* AlignStackPtr(char* ptr) noexcept {
#if defined(STACK_GROW_DOWN)
            return AlignDown(ptr, STACK_ALIGN);
#else
            return AlignUp(ptr, STACK_ALIGN);
#endif
        }

    private:
        char* Data_;
    };

    static inline void*& JmpBufReg(__myjmp_buf& buf, size_t n) noexcept {
        return (((void**)(void*)(buf))[n]);
    }

    static inline void*& JmpBufStackReg(__myjmp_buf& buf) noexcept {
        return JmpBufReg(buf, STACK_CNT);
    }

    static inline void*& JmpBufProgrReg(__myjmp_buf& buf) noexcept {
        return JmpBufReg(buf, PROGR_CNT);
    }

    Y_NO_SANITIZE("address")
    Y_NO_SANITIZE("memory")
    static void ContextTrampoLine() {
        void** argPtr = (void**)((char*)AlignUp(&argPtr + EXTRA_PUSH_ARGS, STACK_ALIGN) + STACK_ALIGN);
        Y_ASSERT(*(argPtr - 1) == *(argPtr - 2));

        Run(*(argPtr - 1));
    }
}

TContMachineContext::TSan::TSan() noexcept
    : TL(nullptr)
{
}

TContMachineContext::TSan::TSan(const TContClosure& c) noexcept
    : NSan::TFiberContext(c.Stack.Data(), c.Stack.Size())
    , TL(c.TrampoLine)
{
}

void TContMachineContext::TSan::DoRun() {
    AfterSwitch();
    TL->DoRun();
    BeforeFinish();
}

TContMachineContext::TContMachineContext(const TContClosure& c)
#if defined(_asan_enabled_)
    : San_(c)
#endif
{
    TStack stack(c.Stack);

/*
     * arg, and align data
     */

#if defined(_asan_enabled_)
    auto trampoline = &San_;
#else
    auto trampoline = c.TrampoLine;
#endif

    stack.Push(trampoline);
    stack.Push(trampoline);
    stack.ReAlign();

    /*
     * fake return address
     */
    for (size_t i = 0; i < EXTRA_PUSH_ARGS; ++i) {
        stack.Push(nullptr);
    }

    __mysetjmp(Buf_);

    JmpBufProgrReg(Buf_) = ReinterpretCast<void*>(ContextTrampoLine);
    JmpBufStackReg(Buf_) = stack.StackPtr();
}

void TContMachineContext::SwitchTo(TContMachineContext* next) noexcept {
    if (Y_LIKELY(__mysetjmp(Buf_) == 0)) {
#if defined(_asan_enabled_)
        next->San_.BeforeSwitch();
#endif
        __mylongjmp(next->Buf_, 1);
    } else {
#if defined(_asan_enabled_)
        San_.AfterSwitch();
#endif
    }
}
#else
void ContextTrampoLine(void* arg) {
    Run(arg);
}
#endif

#if defined(USE_FIBER_CONT)
TContMachineContext::TContMachineContext()
    : Fiber_(ConvertThreadToFiber(this))
    , MainFiber_(true)
{
    Y_ENSURE(Fiber_, STRINGBUF("fiber error"));
}

TContMachineContext::TContMachineContext(const TContClosure& c)
    : Fiber_(CreateFiber(c.Stack.Size(), (LPFIBER_START_ROUTINE)ContextTrampoLine, (LPVOID)c.TrampoLine))
    , MainFiber_(false)
{
    Y_ENSURE(Fiber_, STRINGBUF("fiber error"));
}

TContMachineContext::~TContMachineContext() {
    if (MainFiber_) {
        ConvertFiberToThread();
    } else {
        DeleteFiber(Fiber_);
    }
}

void TContMachineContext::SwitchTo(TContMachineContext* next) noexcept {
    SwitchToFiber(next->Fiber_);
}
#endif

#if defined(USE_GENERIC_CONT)
#include <pthread.h>

struct TContMachineContext::TImpl {
    inline TImpl()
        : TL(nullptr)
        , Finish(false)
    {
    }

    inline TImpl(const TContClosure& c)
        : TL(c.TrampoLine)
        , Finish(false)
    {
        Thread.Reset(new TThread(TThread::TParams(Run, this).SetStackSize(c.Stack.Size()).SetStackPointer((void*)c.Stack.Data())));
        Thread->Start();
    }

    inline ~TImpl() {
        if (Thread) {
            Finish = true;
            Signal();
            Thread->Join();
        }
    }

    inline void SwitchTo(TImpl* next) noexcept {
        next->Signal();
        Wait();
    }

    static void* Run(void* self) {
        ((TImpl*)self)->DoRun();

        return nullptr;
    }

    inline void DoRun() {
        Wait();
        TL->DoRun();
    }

    inline void Signal() noexcept {
        Event.Signal();
    }

    inline void Wait() noexcept {
        Event.Wait();

        if (Finish) {
            // TODO - need proper TThread::Exit(), have some troubles in win32 now
            pthread_exit(0);
        }
    }

    TAutoEvent Event;
    THolder<TThread> Thread;
    ITrampoLine* TL;
    bool Finish;
};

TContMachineContext::TContMachineContext()
    : Impl_(new TImpl())
{
}

TContMachineContext::TContMachineContext(const TContClosure& c)
    : Impl_(new TImpl(c))
{
}

TContMachineContext::~TContMachineContext() {
}

void TContMachineContext::SwitchTo(TContMachineContext* next) noexcept {
    Impl_->SwitchTo(next->Impl_.Get());
}
#endif
