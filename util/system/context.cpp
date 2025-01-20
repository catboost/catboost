#include "compiler.h"
#include "defaults.h"
#include "event.h"
#include "thread.h"

#include <cstdlib> //for abort()

#if defined(_win_)
    #include "winint.h"
#endif

#if defined(_unix_)
    #include <cxxabi.h>

    #if !defined(Y_CXA_EH_GLOBALS_COMPLETE)
namespace __cxxabiv1 {
    struct __cxa_eh_globals {
        void* caughtExceptions;
        unsigned int uncaughtExceptions;
    };

    extern "C" __cxa_eh_globals* __cxa_get_globals();
} // namespace __cxxabiv1
    #endif
#endif

#include <util/stream/output.h>
#include <util/generic/yexception.h>

#define FROM_CONTEXT_IMPL
#include "context.h"

void ITrampoLine::DoRun() {
}

void ITrampoLine::DoRunNaked() {
    try {
        DoRun();
    } catch (...) {
        Cerr << "Uncaught exception in coroutine: " << CurrentExceptionMessage() << "\n";
    }

    abort();
}

static inline void Run(void* arg) {
    ((ITrampoLine*)arg)->DoRunNaked();
}

#if defined(USE_JUMP_CONT)
extern "C" void __mylongjmp(__myjmp_buf env, int val) __attribute__((__noreturn__));
extern "C" int __mysetjmp(__myjmp_buf env) __attribute__((__returns_twice__));

namespace {
    class TStackType {
    public:
        inline TStackType(TArrayRef<char> range) noexcept
    #if defined(STACK_GROW_DOWN)
            : Data_(range.data() + range.size())
    #else
            : Data_(range.data() + STACK_ALIGN)
    #endif
        {
            ReAlign();
        }

        inline ~TStackType() = default;

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

    static inline void*& JmpBufFrameReg(__myjmp_buf& buf) noexcept {
        return JmpBufReg(buf, FRAME_CNT);
    }

    #if defined(_x86_64_)
    // not sure if Y_NO_SANITIZE is needed
    Y_NO_SANITIZE("address")
    Y_NO_SANITIZE("memory") extern "C" void
    ContextTrampoLine(void*, void*, void*, void*, void*, void*, // register arguments, no defined value
                      /* first argument passed through the stack */ void* t1,
                      /* second argument passed through the stack */ void* t2) {
        Y_ASSERT(t1 == t2);
        Run(t1);
    }
    #else
    Y_NO_SANITIZE("address")
    Y_NO_SANITIZE("memory") static void
    ContextTrampoLine() {
        void** argPtr = (void**)((char*)AlignUp(&argPtr + EXTRA_PUSH_ARGS, STACK_ALIGN) + STACK_ALIGN);
        Y_ASSERT(*(argPtr - 1) == *(argPtr - 2));

        Run(*(argPtr - 1));
    }
    #endif
} // namespace

    #if defined(USE_SANITIZER_CONTEXT)

TContMachineContext::TSan::TSan() noexcept
    : TL(nullptr)
{
}

TContMachineContext::TSan::TSan(const TContClosure& c) noexcept
    : NSan::TFiberContext(c.Stack.data(), c.Stack.size(), c.ContName)
    , TL(c.TrampoLine)
{
}

void TContMachineContext::TSan::DoRunNaked() {
    AfterSwitch();
    TL->DoRunNaked();
}

    #endif

TContMachineContext::TContMachineContext(const TContClosure& c)
    #if defined(USE_SANITIZER_CONTEXT)
    : San_(c)
    #endif
{
    TStackType stack(c.Stack);

    /*
     * arg, and align data
     */

    #if defined(USE_SANITIZER_CONTEXT)
    auto trampoline = &San_;
    #else
    auto trampoline = c.TrampoLine;
    #endif

    #if defined(_x86_64_)
    stack.ReAlign();
    // push twice to preserve alignment by 16
    stack.Push(trampoline); // second stack argument
    stack.Push(trampoline); // first stack argument

    stack.Push(nullptr); // fake return address
    #else
    stack.Push(trampoline);
    stack.Push(trampoline);
    stack.ReAlign();
    /*
     * fake return address
     */
    for (size_t i = 0; i < EXTRA_PUSH_ARGS; ++i) {
        stack.Push(nullptr);
    }
    #endif

    __mysetjmp(Buf_);

    JmpBufProgrReg(Buf_) = reinterpret_cast<void*>(ContextTrampoLine);
    JmpBufStackReg(Buf_) = stack.StackPtr();
    JmpBufFrameReg(Buf_) = nullptr;
}

void TContMachineContext::SwitchTo(TContMachineContext* next) noexcept {
    if (Y_LIKELY(__mysetjmp(Buf_) == 0)) {
    #if defined(USE_SANITIZER_CONTEXT)
        next->San_.BeforeSwitch(&San_);
    #endif
        __mylongjmp(next->Buf_, 1);
    } else {
    #if defined(USE_SANITIZER_CONTEXT)
        San_.AfterSwitch();
    #endif
    }
}
#elif defined(_win_) && defined(_32_)
void __stdcall ContextTrampoLine(void* arg) {
    Run(arg);
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
    Y_ENSURE(Fiber_, TStringBuf("fiber error"));
}

TContMachineContext::TContMachineContext(const TContClosure& c)
    : Fiber_(CreateFiber(c.Stack.size(), (LPFIBER_START_ROUTINE)ContextTrampoLine, (LPVOID)c.TrampoLine))
    , MainFiber_(false)
{
    Y_ENSURE(Fiber_, TStringBuf("fiber error"));
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
        Thread.Reset(new TThread(TThread::TParams(Run, this).SetStackSize(c.Stack.size()).SetStackPointer((void*)c.Stack.data())));
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

void TExceptionSafeContext::SwitchTo(TExceptionSafeContext* to) noexcept {
#if defined(_unix_)
    static_assert(sizeof(__cxxabiv1::__cxa_eh_globals) == sizeof(Buf_), "size mismatch of __cxa_eh_globals structure");

    auto* eh = __cxxabiv1::__cxa_get_globals();
    ::memcpy(Buf_, eh, sizeof(Buf_));
    ::memcpy(eh, to->Buf_, sizeof(Buf_));
#endif

    TContMachineContext::SwitchTo(to);
}
