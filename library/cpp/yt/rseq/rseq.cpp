#include "rseq.h"

#ifdef YT_RSEQ_AVAILABLE

#include "per_cpu.h"

#include <library/cpp/yt/misc/static_initializer.h>
#include <library/cpp/yt/misc/tls.h>

#include <util/system/types.h>

#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <utility>

namespace NYT::NRseq {

////////////////////////////////////////////////////////////////////////////////

// Set by the static initializer below before main(); see the header for the zero
// placeholder window.
std::ptrdiff_t CpuIdFieldOffset = 0;

namespace {

#if defined(__x86_64__)
    constexpr long RseqSyscallNumber = 334;
#elif defined(__aarch64__)
    constexpr long RseqSyscallNumber = 293;
#endif

// The kernel struct rseq is registered with this (original ABI) size; cpu_id sits at
// offset 4 and is the only field we read.
constexpr unsigned RseqRegistrationSize = 32;

// The conventional rseq signature shared by glibc, librseq and tcmalloc. We must pass
// the same one so that re-registering an already-registered area yields EBUSY (success)
// rather than EINVAL; see RegisterCurrentThread.
constexpr unsigned RseqSignature = 0x53053053;

struct alignas(32) TRseqArea
{
    ui32 CpuIdStart;
    ui32 CpuId;
    ui64 RseqCs;
    ui32 Flags;
    ui32 Padding[3];
};

static_assert(sizeof(TRseqArea) == RseqRegistrationSize);
static_assert(offsetof(TRseqArea, CpuId) == 4);

// Defined by glibc (>= 2.35) when it owns the rseq registration; absent (weak ->
// nullptr) otherwise, e.g. on glibc 2.31 and musl.
extern "C" {
extern const std::ptrdiff_t __rseq_offset __attribute__((weak));
extern const unsigned int __rseq_size __attribute__((weak));
} // extern "C"

// The TLS model for our weak __rseq_abi definition. initial-exec yields a
// link-time-constant offset from the thread pointer (a single TP-relative load), but it
// requires a static TLS block reserved at program startup -- which the dynamic loader
// cannot grant a module dlopen'd afterwards (e.g. a YQL UDF .so), failing with "cannot
// allocate memory in static TLS block". Position-independent objects (-fPIC/-fPIE, what
// such .so's are built as) therefore use global-dynamic instead. This only affects the
// cold &__rseq_abi accesses (ComputeCpuIdFieldOffset / RegisterCurrentThread); the hot
// path reads *(thread_pointer + CpuIdFieldOffset) off a cached offset and is unchanged.
// Mirrors the __PIC__/__PIE__ handling in contrib/libs/tcmalloc.
#if defined(__PIC__) || defined(__PIE__)
    #define YT_RSEQ_ABI_TLS_MODEL "global-dynamic"
#else
    #define YT_RSEQ_ABI_TLS_MODEL "initial-exec"
#endif

// The legacy per-thread rseq area. tcmalloc, librseq and pre-2.35 glibc all define and
// register this exact symbol; our definition is weak, so it coalesces with theirs when
// present (the common case in YT binaries -- tcmalloc owns it) and stands alone, with us
// registering it, otherwise (e.g. musl). CpuId starts at -1 so an unregistered thread
// takes the slow path.
extern "C" {
__thread TRseqArea __rseq_abi __attribute__((weak, tls_model(YT_RSEQ_ABI_TLS_MODEL), aligned(32))) = {
    .CpuId = static_cast<ui32>(-1),
};
} // extern "C"

#undef YT_RSEQ_ABI_TLS_MODEL

// True when we read __rseq_abi (not the glibc-owned area) and so must make sure each
// thread is registered.
bool OwnsRegistration = false;

bool RegisterCurrentThread()
{
    // flags = 0. We pass the shared signature and the standard size so that whoever of
    // {us, tcmalloc, librseq} runs first registers __rseq_abi and the rest get EBUSY
    // (success). The signature must be 0x53053053: it is the value emitted before abort_ip
    // in the per_cpu rseq critical sections (see per_cpu-inl.h), and on a kernel abort the
    // signature must match the registered one or the kernel delivers SIGSEGV.
    if (::syscall(RseqSyscallNumber, &__rseq_abi, RseqRegistrationSize, 0u, RseqSignature) == 0) {
        return true;
    }
    return errno == EBUSY;
}

YT_PREVENT_TLS_CACHING std::ptrdiff_t ComputeCpuIdFieldOffset()
{
    if (&__rseq_size != nullptr && __rseq_size != 0) {
        // glibc owns the registration and keeps every thread's cpu_id up to date in its
        // own area; just read it.
        return __rseq_offset + static_cast<std::ptrdiff_t>(offsetof(TRseqArea, CpuId));
    }
    // Otherwise use __rseq_abi. Register this (main) thread; other threads register
    // lazily on their first slow-path call. The offset points at __rseq_abi either way:
    // cpu_id holds the real value once the thread is registered and stays -1 (routing to
    // the slow path) until then.
    OwnsRegistration = true;
    RegisterCurrentThread();
    auto* threadPointer = static_cast<char*>(__builtin_thread_pointer());
    return (reinterpret_cast<char*>(&__rseq_abi) - threadPointer) +
        static_cast<std::ptrdiff_t>(offsetof(TRseqArea, CpuId));
}

YT_STATIC_INITIALIZER({
    CpuIdFieldOffset = ComputeCpuIdFieldOffset();
});

// Checks, on a freshly spawned thread, whether CpuIdFieldOffset names *this* thread's rseq
// area -- i.e. whether __rseq_abi sits at a fixed thread-pointer offset (a glibc-owned area or
// the static TLS block, incl. tcmalloc) rather than a dlopen'd module's dynamically allocated
// TLS, where the offset is valid only on the thread that computed it. Compares addresses
// without dereferencing the suspect offset, so it is safe even when the offset is bogus. See
// IsPerCpuFastPathSafe.
YT_PREVENT_TLS_CACHING bool ValidateFastPathOnFreshThread()
{
    if (OwnsRegistration) {
        RegisterCurrentThread();
        const auto* viaOffset =
            static_cast<const char*>(__builtin_thread_pointer()) + CpuIdFieldOffset;
        const auto* viaSymbol =
            reinterpret_cast<const char*>(&__rseq_abi) +
            static_cast<std::ptrdiff_t>(offsetof(TRseqArea, CpuId));
        if (viaOffset != viaSymbol) {
            // Not thread-pointer-stable: dynamically allocated TLS. Disable the fast path.
            return false;
        }
        // Stable; read cpu_id through the symbol and require the kernel to have registered us.
        return __rseq_abi.CpuId != static_cast<ui32>(-1);
    }
    // glibc owns the registration: __rseq_offset is a fixed thread-pointer-relative offset
    // valid on every thread. Usable iff a valid cpu_id is present.
    return ReadField<int>(CpuIdFieldOffset) >= 0;
}

void* RunFastPathProbe(void* result)
{
    *static_cast<bool*>(result) = ValidateFastPathOnFreshThread();
    return nullptr;
}

} // namespace

YT_PREVENT_TLS_CACHING bool EnsureCurrentThreadRegistered()
{
    if (OwnsRegistration) {
        // Register this thread once, on first use. Usually a no-op (EBUSY): in YT
        // binaries tcmalloc registers __rseq_abi for every thread before we get here.
        thread_local bool RegistrationAttempted = false;
        if (!std::exchange(RegistrationAttempted, true)) {
            RegisterCurrentThread();
        }
    }
    // Either way the thread is registered iff cpu_id reads as valid (>= 0).
    return ReadField<int>(CpuIdFieldOffset) >= 0;
}

bool IsPerCpuFastPathSafe()
{
    // Decided once, lazily, on a freshly spawned thread (the check is meaningful only off the
    // thread that computed the offset) and cached -- cost is one thread spawn at first use.
    // Reached from the first hot-sensor construction, a runtime event; were it ever reached
    // from a global constructor under the dynamic loader lock, spawning the probe thread could
    // deadlock, but hot sensors are not built during static initialization.
    static const bool Safe = [] {
        // Make sure CpuIdFieldOffset (static initializer) and CpuCount are in place.
        GetCpuCount();
        bool result = false;
        pthread_t thread;
        if (::pthread_create(&thread, /*attr*/ nullptr, &RunFastPathProbe, &result) != 0) {
            // Cannot validate -- stay on the safe atomic fallback.
            return false;
        }
        ::pthread_join(thread, /*retval*/ nullptr);
        return result;
    }();
    return Safe;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq

#else // YT_RSEQ_AVAILABLE

#include "per_cpu.h"

namespace NYT::NRseq {

////////////////////////////////////////////////////////////////////////////////

// No rseq fast path on this platform; hot sensors use the atomic fallback.
bool IsPerCpuFastPathSafe()
{
    return false;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq

#endif // YT_RSEQ_AVAILABLE
