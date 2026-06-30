#include "execution_stack.h"

#if defined(_unix_)
#   include <sys/mman.h>
#   include <limits.h>
#   include <unistd.h>
#   if !defined(__x86_64__) && !defined(__arm64__) && !defined(__aarch64__)
#       error Unsupported platform
#   endif
#endif

#include <library/cpp/yt/memory/ref.h>
#include <library/cpp/yt/memory/ref_tracked.h>

#include <library/cpp/yt/misc/tls.h>

#include <library/cpp/yt/error/error.h>

#include <library/cpp/yt/system/exit.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

TExecutionStackBase::TExecutionStackBase(size_t size)
    : Size_(RoundUpToPage(size))
{
    auto cookie = GetRefCountedTypeCookie<TExecutionStack>();
    TRefCountedTrackerFacade::AllocateInstance(cookie);
    TRefCountedTrackerFacade::AllocateSpace(cookie, Size_);
}

TExecutionStackBase::~TExecutionStackBase()
{
    auto cookie = GetRefCountedTypeCookie<TExecutionStack>();
    TRefCountedTrackerFacade::FreeInstance(cookie);
    TRefCountedTrackerFacade::FreeSpace(cookie, Size_);
}

void* TExecutionStackBase::GetStack() const
{
    return Stack_;
}

size_t TExecutionStackBase::GetSize() const
{
    return Size_;
}

////////////////////////////////////////////////////////////////////////////////

#if defined(_unix_)

TExecutionStack::TExecutionStack(size_t size)
    : TExecutionStackBase(size)
{
    size_t guardSize = GuardPageCount * GetPageSize();

    int flags =
#if defined(_darwin_)
        MAP_ANON | MAP_PRIVATE;
#else
        MAP_ANONYMOUS | MAP_PRIVATE;
#endif

    Base_ = reinterpret_cast<char*>(::mmap(
        0,
        guardSize * 2 + Size_,
        PROT_READ | PROT_WRITE,
        flags,
        -1,
        0));

    auto handleError = [&] {
        AbortProcessDramatically(
            EProcessExitCode::OutOfMemory,
            Format("Error creating execution stack (Size: %v): %v",
                Size_,
                TError::FromSystem()));
    };

    if (Base_ == MAP_FAILED) {
        handleError();
    }

    if (::mprotect(Base_, guardSize, PROT_NONE) == -1) {
        handleError();
    }

    if (::mprotect(Base_ + guardSize + Size_, guardSize, PROT_NONE) == -1) {
        handleError();
    }

    Stack_ = Base_ + guardSize;
    YT_VERIFY((reinterpret_cast<uintptr_t>(Stack_) & 15) == 0);
}

TExecutionStack::~TExecutionStack()
{
    const size_t guardSize = GuardPageCount * GetPageSize();
    ::munmap(Base_, guardSize * 2 + Size_);
}

#elif defined(_win_)

TExecutionStack::TExecutionStack(size_t size)
    : TExecutionStackBase(size)
    , Handle_(::CreateFiber(Size_, &FiberTrampoline, this))
{ }

TExecutionStack::~TExecutionStack()
{
    ::DeleteFiber(Handle_);
}

YT_DEFINE_THREAD_LOCAL(void*, FiberTrampolineOpaque);

void TExecutionStack::SetOpaque(void* opaque)
{
    FiberTrampolineOpaque() = opaque;
}

void* TExecutionStack::GetOpaque()
{
    return FiberTrampolineOpaque();
}

void TExecutionStack::SetTrampoline(void (*trampoline)(void*))
{
    YT_ASSERT(!Trampoline_);
    Trampoline_ = trampoline;
}

VOID CALLBACK TExecutionStack::FiberTrampoline(PVOID opaque)
{
    auto* stack = reinterpret_cast<TExecutionStack*>(opaque);
    stack->Trampoline_(FiberTrampolineOpaque());
}

#else
#   error Unsupported platform
#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
