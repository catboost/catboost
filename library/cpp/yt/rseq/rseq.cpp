#include "rseq.h"

#ifdef YT_RSEQ_AVAILABLE

#include <library/cpp/yt/misc/static_initializer.h>
#include <library/cpp/yt/misc/tls.h>

#include <util/system/types.h>

#include <sys/syscall.h>
#include <unistd.h>

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

// Our own per-thread rseq area, used when glibc does not own the registration.
// initial-exec so its offset from the thread pointer is a link-time constant, and
// CpuId starts at -1 so an unregistered thread takes the slow path.
__thread TRseqArea OwnRseqArea __attribute__((tls_model("initial-exec"), aligned(32))) = {
    .CpuId = static_cast<ui32>(-1),
};

// True iff we (not glibc) own the registration and the kernel supports rseq.
bool OwnsRegistration = false;

bool RegisterCurrentThread()
{
    // flags = 0, signature = 0: we never use restartable critical sections, so the
    // signature is irrelevant (it is only checked at an rseq_cs abort handler).
    return ::syscall(RseqSyscallNumber, &OwnRseqArea, RseqRegistrationSize, 0u, 0u) == 0;
}

YT_PREVENT_TLS_CACHING std::ptrdiff_t ComputeCpuIdFieldOffset()
{
    if (&__rseq_size != nullptr && __rseq_size != 0) {
        // glibc owns the registration and keeps every thread's cpu_id up to date.
        return __rseq_offset + static_cast<std::ptrdiff_t>(offsetof(TRseqArea, CpuId));
    }
    // We own the registration. Probe kernel support by registering this (main) thread;
    // other threads register lazily on their first slow-path call. Point at our area
    // either way: cpu_id holds the real value once registered and stays -1 (routing to
    // the slow path) when it is not.
    if (RegisterCurrentThread()) {
        OwnsRegistration = true;
    }
    auto* threadPointer = static_cast<char*>(__builtin_thread_pointer());
    return (reinterpret_cast<char*>(&OwnRseqArea) - threadPointer) + static_cast<std::ptrdiff_t>(offsetof(TRseqArea, CpuId));
}

YT_STATIC_INITIALIZER({
    CpuIdFieldOffset = ComputeCpuIdFieldOffset();
});

} // namespace

YT_PREVENT_TLS_CACHING bool EnsureCurrentThreadRegistered()
{
    if (!OwnsRegistration) {
        // Either glibc owns the registration (every thread is already registered) or
        // rseq is unavailable. The two are told apart by what cpu_id reads: a valid
        // (>= 0) value means registered.
        return ReadField<int>(CpuIdFieldOffset) >= 0;
    }

    // We own the registration: register this thread once, on first use.
    thread_local bool RegistrationAttempted = false;
    if (!std::exchange(RegistrationAttempted, true)) {
        RegisterCurrentThread();
    }
    return ReadField<int>(CpuIdFieldOffset) >= 0;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq

#endif // YT_RSEQ_AVAILABLE
