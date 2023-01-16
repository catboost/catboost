#include "demangle_impl.h"
#include "platform.h"
#include "backtrace.h"

#include <util/stream/output.h>
#include <util/stream/format.h>
#include <util/generic/array_ref.h>
#include <util/generic/string.h>

#ifdef _win_
    #include "mutex.h"

    #ifndef OPTIONAL
        #define OPTIONAL
    #endif
    #include <dbghelp.h>
#endif

#if defined(_bionic_)
//TODO
#else
    #if !defined(HAVE_BACKTRACE) && defined(_cygwin_)
        #define CaptureStackBackTrace RtlCaptureStackBackTrace
extern "C" __stdcall unsigned short CaptureStackBackTrace(unsigned long FramesToSkip, unsigned long FramesToCapture, void** BackTrace, unsigned long* BackTraceHash);

        #define USE_WIN_BACKTRACE
        #define HAVE_BACKTRACE
    #endif

    #if !defined(HAVE_BACKTRACE) && defined(__IOS__)
        #define USE_GLIBC_BACKTRACE
        #define HAVE_BACKTRACE
    #endif

    #if !defined(HAVE_BACKTRACE) && defined(__GNUC__)
        #define USE_GCC_BACKTRACE
        #define HAVE_BACKTRACE
    #endif

    #if !defined(HAVE_BACKTRACE) && defined(_win_)
        #define USE_WIN_BACKTRACE
        #define HAVE_BACKTRACE
    #endif

    #if !defined(HAVE_BACKTRACE) && defined(_glibc_)
        #define USE_GLIBC_BACKTRACE
        #define HAVE_BACKTRACE
    #endif
#endif

#if defined(USE_GLIBC_BACKTRACE)
    #include <execinfo.h>

size_t BackTrace(void** p, size_t len) {
    return (size_t)backtrace(p, len);
}
#endif

#if defined(USE_GCC_BACKTRACE)
    #include <cxxabi.h>
    #include <unwind.h>

namespace {
    namespace NGCCBacktrace {
        struct TBackTraceContext {
            void** sym;
            size_t cnt;
            size_t size;
        };

        static _Unwind_Reason_Code Helper(struct _Unwind_Context* c, void* h) {
            TBackTraceContext* bt = (TBackTraceContext*)h;

            if (bt->cnt != 0) {
                bt->sym[bt->cnt - 1] = (void*)_Unwind_GetIP(c);
            }

            if (bt->cnt == bt->size) {
                return _URC_END_OF_STACK;
            }

            ++bt->cnt;

            return _URC_NO_REASON;
        }

        static inline size_t BackTrace(void** p, size_t len) {
            if (len >= 1) {
                TBackTraceContext bt = {p, 0, len};

                _Unwind_Backtrace(Helper, &bt);

                return bt.cnt - 1;
            }

            return 0;
        }
    }
}

size_t BackTrace(void** p, size_t len) {
    return NGCCBacktrace::BackTrace(p, len);
}
#endif

#if defined(USE_WIN_BACKTRACE)
size_t BackTrace(void** p, size_t len) {
    return CaptureStackBackTrace(0, len, p, nullptr);
}
#endif

#if !defined(HAVE_BACKTRACE)
size_t BackTrace(void**, size_t) {
    return 0;
}
#endif

#if defined(_unix_) && !defined(_cygwin_)
    #include <util/generic/strfcpy.h>

    #include <dlfcn.h>

    #if defined(_darwin_)
        #include <execinfo.h>
    #endif

static inline const char* CopyTo(const char* from, char* buf, size_t len) {
    strfcpy(buf, from, len);

    return buf;
}

TResolvedSymbol ResolveSymbol(void* sym, char* buf, size_t len) {
    TResolvedSymbol ret = {
        "??",
        sym,
    };

    Dl_info dli;

    Zero(dli);

    if (dladdr(sym, &dli) && dli.dli_sname) {
        ret.Name = CopyTo(NPrivate::TCppDemangler().Demangle(dli.dli_sname), buf, len);
        ret.NearestSymbol = dli.dli_saddr;
    }

    return ret;
}
#elif defined(_win_)
    #include <util/generic/singleton.h>

namespace {
    struct TWinSymbolResolverImpl {
        typedef BOOL(WINAPI* TSymInitializeFunc)(HANDLE, PCSTR, BOOL);
        typedef BOOL(WINAPI* TSymCleanupFunc)(HANDLE);
        typedef BOOL(WINAPI* TSymFromAddrFunc)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);

        TWinSymbolResolverImpl()
            : InitOk(FALSE)
        {
            Library = LoadLibraryA("Dbghelp.dll");
            if (!Library) {
                return;
            }

            SymInitializeFunc = (TSymInitializeFunc)GetProcAddress(Library, "SymInitialize");
            SymCleanupFunc = (TSymCleanupFunc)GetProcAddress(Library, "SymCleanup");
            SymFromAddrFunc = (TSymFromAddrFunc)GetProcAddress(Library, "SymFromAddr");
            if (SymInitializeFunc && SymCleanupFunc && SymFromAddrFunc) {
                InitOk = SymInitializeFunc(GetCurrentProcess(), nullptr, TRUE);
            }
        }

        ~TWinSymbolResolverImpl() {
            if (InitOk) {
                SymCleanupFunc(GetCurrentProcess());
            }

            if (Library) {
                FreeLibrary(Library);
            }
        }

        TResolvedSymbol Resolve(void* sym, char* buf, size_t len) {
            TGuard<TMutex> guard(Mutex);

            TResolvedSymbol ret = {
                "??",
                sym};

            if (!InitOk || (len <= 1 + sizeof(SYMBOL_INFO))) {
                return ret;
            }

            SYMBOL_INFO* symbol = (SYMBOL_INFO*)buf;
            Zero(*symbol);

            symbol->MaxNameLen = len - sizeof(SYMBOL_INFO) - 1;
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

            DWORD64 displacement = 0;
            BOOL res = SymFromAddrFunc(GetCurrentProcess(), (DWORD64)sym, &displacement, symbol);
            if (res) {
                ret.NearestSymbol = (void*)symbol->Address;
                ret.Name = symbol->Name;
            }

            return ret;
        }

        TMutex Mutex;
        HMODULE Library;
        TSymInitializeFunc SymInitializeFunc;
        TSymCleanupFunc SymCleanupFunc;
        TSymFromAddrFunc SymFromAddrFunc;
        BOOL InitOk;
    };
}

TResolvedSymbol ResolveSymbol(void* sym, char* buf, size_t len) {
    return Singleton<TWinSymbolResolverImpl>()->Resolve(sym, buf, len);
}
#else
TResolvedSymbol ResolveSymbol(void* sym, char*, size_t) {
    TResolvedSymbol ret = {
        "??",
        sym,
    };

    return ret;
}
#endif

void FormatBackTrace(IOutputStream* out, void* const* backtrace, size_t backtraceSize) {
    char tmpBuf[1024];

    for (size_t i = 0; i < backtraceSize; ++i) {
        TResolvedSymbol rs = ResolveSymbol(backtrace[i], tmpBuf, sizeof(tmpBuf));

        *out << rs.Name << "+" << ((ptrdiff_t)backtrace[i] - (ptrdiff_t)rs.NearestSymbol) << " (" << Hex((ptrdiff_t)backtrace[i], HF_ADDX) << ')' << '\n';
    }
}

TFormatBackTraceFn FormatBackTraceFn = FormatBackTrace;

TFormatBackTraceFn SetFormatBackTraceFn(TFormatBackTraceFn f) {
    TFormatBackTraceFn prevFn = FormatBackTraceFn;
    FormatBackTraceFn = f;
    return prevFn;
}

void FormatBackTrace(IOutputStream* out) {
    void* array[300];
    const size_t s = BackTrace(array, Y_ARRAY_SIZE(array));
    FormatBackTraceFn(out, array, s);
}

TFormatBackTraceFn GetFormatBackTraceFn() {
    return FormatBackTraceFn;
}

void PrintBackTrace() {
    FormatBackTrace(&Cerr);
}

TBackTrace::TBackTrace()
    : Size(0)
{
}

void TBackTrace::Capture() {
    Size = BackTrace(Data, CAPACITY);
}

void TBackTrace::PrintTo(IOutputStream& out) const {
    FormatBackTraceFn(&out, Data, Size);
}

TString TBackTrace::PrintToString() const {
    TStringStream ss;
    PrintTo(ss);
    return ss.Str();
}

size_t TBackTrace::size() const {
    return Size;
}

const void* const* TBackTrace::data() const {
    return Data;
}

TBackTrace::operator TBackTraceView() const {
    return TBackTraceView(Data, Size);
}

TBackTrace TBackTrace::FromCurrentException() {
#ifdef _YNDX_LIBUNWIND_EXCEPTION_BACKTRACE_SIZE
    TBackTrace result;
    result.Size = __cxxabiv1::__cxa_collect_current_exception_backtrace(result.Data, CAPACITY);
    return result;
#else
    return TBackTrace();
#endif
}
