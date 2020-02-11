#include "tls.h"
#include "thread.h"
#include "thread.i"

#include <util/generic/map.h>
#include <util/generic/ptr.h>
#include <util/generic/ymath.h>
#include <util/generic/ylimits.h>
#include <util/generic/yexception.h>
#include "yassert.h"
#include <utility>

#include <cstdio>
#include <cstdlib>

#if !defined(_win_)
#include <pthread.h>
#endif

bool SetHighestThreadPriority() {
#ifdef _win_
    return SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#else
    struct sched_param sch;
    memset(&sch, 0, sizeof(sch));
    sch.sched_priority = 31;
    return pthread_setschedparam(pthread_self(), SCHED_RR, &sch) == 0;
#endif
}

namespace {
    using TParams = TThread::TParams;
    using TId = TThread::TId;

    inline void SetThrName(const TParams& p) {
        try {
            if (p.Name) {
                TThread::SetCurrentThreadName(p.Name.data());
            }
        } catch (...) {
            // ¯\_(ツ)_/¯
        }
    }

    inline size_t StackSize(const TParams& p) noexcept {
        if (p.StackSize) {
            return FastClp2(p.StackSize);
        }

        return 0;
    }

#if defined(_win_)
    class TWinThread {
        struct TMyParams: public TParams, public TThrRefBase {
            inline TMyParams(const TParams& p)
                : TParams(p)
                , Result(0)
            {
            }

            void* Result;
        };

        using TParamsRef = TIntrusivePtr<TMyParams>;

    public:
        inline TWinThread(const TParams& params)
            : P_(new TMyParams(params))
            , Handle(0)
#if _WIN32_WINNT < 0x0502
            , ThreadId(0)
#endif
        {
        }

        inline bool Running() const noexcept {
            return Handle != 0;
        }

        inline TId SystemThreadId() const noexcept {
#if _WIN32_WINNT < 0x0502
            return (TId)ThreadId;
#else
            return (TId)GetThreadId(Handle);
#endif
        }

        inline void* Join() {
            ::WaitForSingleObject(Handle, INFINITE);
            ::CloseHandle(Handle);

            return P_->Result;
        }

        inline void Detach() {
            ::CloseHandle(Handle);
        }

        static ui32 __stdcall Proxy(void* ptr) {
            NTls::TCleaner cleaner;

            (void)cleaner;

            {
                TParamsRef p((TMyParams*)(ptr));

                //drop counter, gotten in Start()
                p->UnRef();

                SetThrName(*p);
                p->Result = p->Proc(p->Data);
            }

            return 0;
        }

        inline void Start() {
#if _WIN32_WINNT < 0x0502
            Handle = reinterpret_cast<HANDLE>(::_beginthreadex(nullptr, (unsigned)StackSize(*P_), Proxy, (void*)P_.Get(), 0, &ThreadId));
#else
            Handle = reinterpret_cast<HANDLE>(::_beginthreadex(nullptr, (unsigned)StackSize(*P_), Proxy, (void*)P_.Get(), 0, nullptr));
#endif

            Y_ENSURE(Handle, AsStringBuf("failed to create a thread"));

            //do not do this, kids, at home
            P_->Ref();
        }

    private:
        TParamsRef P_;
        HANDLE Handle;
#if _WIN32_WINNT < 0x0502
        ui32 ThreadId;
#endif
    };

    using TThreadBase = TWinThread;
#else
    //unix

#define PCHECK(x, y)                                     \
    {                                                    \
        const int err_ = x;                              \
        if (err_) {                                      \
            ythrow TSystemError(err_) << AsStringBuf(y); \
        }                                                \
    }

    class TPosixThread {
    public:
        inline TPosixThread(const TParams& params)
            : P_(new TParams(params))
            , H_()
        {
            static_assert(sizeof(H_) == sizeof(TId), "expect sizeof(H_) == sizeof(TId)");
        }

        inline TId SystemThreadId() const noexcept {
            return (TId)H_;
        }

        inline void* Join() {
            void* tec = nullptr;
            PCHECK(pthread_join(H_, &tec), "can not join thread");

            return tec;
        }

        inline void Detach() {
            PCHECK(pthread_detach(H_), "can not detach thread");
        }

        inline bool Running() const noexcept {
            return (bool)H_;
        }

        inline void Start() {
            pthread_attr_t* pattrs = nullptr;
            pthread_attr_t attrs;

            if (P_->StackSize > 0) {
                Zero(attrs);
                pthread_attr_init(&attrs);
                pattrs = &attrs;

                if (P_->StackPointer) {
                    pthread_attr_setstack(pattrs, P_->StackPointer, P_->StackSize);
                } else {
                    pthread_attr_setstacksize(pattrs, StackSize(*P_));
                }
            }

            {
                TParams* holdP = P_.Release();
                int err = pthread_create(&H_, pattrs, ThreadProxy, holdP);
                if (err) {
                    H_ = {};
                    P_.Reset(holdP);
                    PCHECK(err, "failed to create thread");
                }
            }
        }

    private:
        static void* ThreadProxy(void* arg) {
            THolder<TParams> p((TParams*)arg);

            SetThrName(*p);

            return p->Proc(p->Data);
        }

    private:
        THolder<TParams> P_;
        pthread_t H_;
    };

#undef PCHECK

    using TThreadBase = TPosixThread;
#endif

    template <class T>
    static inline typename T::TValueType* Impl(T& t, const char* op, bool check = true) {
        if (!t) {
            ythrow yexception() << "can not " << op << " dead thread";
        }

        if (t->Running() != check) {
            static const char* const msg[] = {"running", "not running"};

            ythrow yexception() << "can not " << op << " " << msg[check] << " thread";
        }

        return t.Get();
    }
}

class TThread::TImpl: public TThreadBase {
public:
    inline TImpl(const TParams& params, THolder<TCallableBase> callable = {})
        : TThreadBase(params)
        , Callable_(std::move(callable))
    {
    }

    inline TId Id() const noexcept {
        return ThreadIdHashFunction(SystemThreadId());
    }

    static THolder<TImpl> Create(THolder<TCallableBase> callable) {
        TParams params(TCallableBase::ThreadWorker, callable.Get());
        return MakeHolder<TImpl>(std::move(params), std::move(callable));
    }

private:
    THolder<TCallableBase> Callable_;
};

TThread::TThread(const TParams& p)
    : Impl_(new TImpl(p))
{
}

TThread::TThread(TThreadProc threadProc, void* param)
    : Impl_(new TImpl(TParams(threadProc, param)))
{
}

TThread::TThread(TPrivateCtor, THolder<TCallableBase> callable)
    : Impl_(TImpl::Create(std::move(callable)))
{
}

TThread::~TThread() {
    Join();
}

void TThread::Start() {
    Impl(Impl_, "start", false)->Start();
}

void* TThread::Join() {
    if (Running()) {
        void* ret = Impl_->Join();

        Impl_.Destroy();

        return ret;
    }

    return nullptr;
}

void TThread::Detach() {
    if (Running()) {
        Impl_->Detach();
        Impl_.Destroy();
    }
}

bool TThread::Running() const noexcept {
    return Impl_ && Impl_->Running();
}

TThread::TId TThread::Id() const noexcept {
    if (Running()) {
        return Impl_->Id();
    }

    return ImpossibleThreadId();
}

TThread::TId TThread::CurrentThreadId() noexcept {
    return SystemCurrentThreadId();
}

TThread::TId TThread::ImpossibleThreadId() noexcept {
    return Max<TThread::TId>();
}

namespace {
    template <class T>
    static void* ThreadProcWrapper(void* param) {
        return reinterpret_cast<T*>(param)->ThreadProc();
    }
}

ISimpleThread::ISimpleThread(size_t stackSize)
    : TThread(TParams(ThreadProcWrapper<ISimpleThread>, reinterpret_cast<void*>(this), stackSize))
{
}

#if defined(_MSC_VER)
// This beautiful piece of code is borrowed from
// http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx

//
// Usage: WindowsCurrentSetThreadName (-1, "MainThread");
//
#include <windows.h>
#include <processthreadsapi.h>

const DWORD MS_VC_EXCEPTION = 0x406D1388;

#pragma pack(push, 8)
typedef struct tagTHREADNAME_INFO {
    DWORD dwType;     // Must be 0x1000.
    LPCSTR szName;    // Pointer to name (in user addr space).
    DWORD dwThreadID; // Thread ID (-1=caller thread).
    DWORD dwFlags;    // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

static void WindowsCurrentSetThreadName(DWORD dwThreadID, const char* threadName) {
    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = threadName;
    info.dwThreadID = dwThreadID;
    info.dwFlags = 0;

    __try {
        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
}
#endif

void TThread::SetCurrentThreadName(const char* name) {
    (void)name;

#if defined(_freebsd_)
    pthread_t thread = pthread_self();
    pthread_set_name_np(thread, name);
#elif defined(_linux_)
    Y_VERIFY(prctl(PR_SET_NAME, name, 0, 0, 0) == 0, "pctl failed: %s", strerror(errno));
#elif defined(_darwin_)
    Y_VERIFY(pthread_setname_np(name) == 0, "pthread_setname_np failed: %s", strerror(errno));
#elif defined(_MSC_VER)
    WindowsCurrentSetThreadName(DWORD(-1), name);
#else
// no idea
#endif // OS
}

TString TThread::CurrentThreadName() {
#if defined(_freebsd_)
// FreeBSD doesn't seem to have an API to get thread name.
#elif defined(_linux_)
    // > The buffer should allow space for up to 16 bytes; the returned string  will be
    // > null-terminated.
    // via `man prctl`
    char name[16];
    memset(name, 0, sizeof(name));
    Y_VERIFY(prctl(PR_GET_NAME, name, 0, 0, 0) == 0, "pctl failed: %s", strerror(errno));
    return name;
#elif defined(_darwin_)
    // available on Mac OS 10.6+
    const auto thread = pthread_self();
    char name[256];
    memset(name, 0, sizeof(name));
    Y_VERIFY(pthread_getname_np(thread, name, sizeof(name)) == 0, "pthread_getname_np failed: %s", strerror(errno));
    return name;
#elif defined(_MSC_VER)
// Apparently there is no way to get thread name for Windows in general case.
//
// Though there is an API for Windows 10:
// https://msdn.microsoft.com/en-us/library/windows/desktop/mt774972(v=vs.85).aspx
#else
// no idea
#endif // OS

    return {};
}

TCurrentThreadLimits::TCurrentThreadLimits() noexcept
    : StackBegin(nullptr)
    , StackLength(0)
{
#if defined(_linux_) || defined(_cygwin_) || defined(_freebsd_)
    pthread_attr_t attr;
    pthread_attr_init(&attr);

#if defined(_linux_) || defined(_cygwin_)
    Y_VERIFY(pthread_getattr_np(pthread_self(), &attr) == 0, "pthread_getattr failed");
#else
    Y_VERIFY(pthread_attr_get_np(pthread_self(), &attr) == 0, "pthread_attr_get_np failed");
#endif
    pthread_attr_getstack(&attr, (void**)&StackBegin, &StackLength);
    pthread_attr_destroy(&attr);

#elif defined(_darwin_)
    StackBegin = pthread_get_stackaddr_np(pthread_self());
    StackLength = pthread_get_stacksize_np(pthread_self());
#elif defined(_MSC_VER)

#if _WIN32_WINNT >= _WIN32_WINNT_WIN8
    ULONG_PTR b = 0;
    ULONG_PTR e = 0;

    GetCurrentThreadStackLimits(&b, &e);

    StackBegin = (const void*)b;
    StackLength = e - b;

#else
    // Copied from https://github.com/llvm-mirror/compiler-rt/blob/release_40/lib/sanitizer_common/sanitizer_win.cc#L91
    void* place_on_stack = alloca(16);
    MEMORY_BASIC_INFORMATION memory_info;
    Y_VERIFY(VirtualQuery(place_on_stack, &memory_info, sizeof(memory_info)));

    StackBegin = memory_info.AllocationBase;
    StackLength = static_cast<const char*>(memory_info.BaseAddress) + memory_info.RegionSize - static_cast<const char*>(StackBegin);

#endif

#else
#error port me
#endif
}
