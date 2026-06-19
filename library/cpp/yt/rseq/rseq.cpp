#include "rseq.h"

#ifdef YT_RSEQ_AVAILABLE

#include <library/cpp/yt/misc/static_initializer.h>
#include <library/cpp/yt/misc/tls.h>

#include <util/system/types.h>

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

// The legacy per-thread rseq area. tcmalloc, librseq and pre-2.35 glibc all define and
// register this exact symbol; our definition is weak, so it coalesces with theirs when
// present (the common case in YT binaries -- tcmalloc owns it) and stands alone, with us
// registering it, otherwise (e.g. musl). initial-exec gives a link-time-constant offset
// from the thread pointer; CpuId starts at -1 so an unregistered thread takes the slow
// path.
extern "C" {
__thread TRseqArea __rseq_abi __attribute__((weak, tls_model("initial-exec"), aligned(32))) = {
    .CpuId = static_cast<ui32>(-1),
};
} // extern "C"

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

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq

#endif // YT_RSEQ_AVAILABLE
