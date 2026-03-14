#pragma once

#include "public.h"

#include <util/system/platform.h>

#if defined(_win_)
#include <windows.h>
#endif

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

class TExecutionStackBase
{
public:
    TExecutionStackBase(const TExecutionStackBase& other) = delete;
    TExecutionStackBase& operator=(const TExecutionStackBase& other) = delete;

    virtual ~TExecutionStackBase();

    void* GetStack() const;
    size_t GetSize() const;

protected:
    const size_t Size_;
    void* Stack_ = nullptr;

    explicit TExecutionStackBase(size_t size);
};

#if defined(_unix_)

//! Mapped memory with a few extra guard pages.
class TExecutionStack
    : public TExecutionStackBase
{
public:
    explicit TExecutionStack(size_t size);
    ~TExecutionStack();

private:
    char* Base_ = nullptr;

    static const int GuardPageCount = 256;
};

#elif defined(_win_)

//! Stack plus Windows fiber holder.
class TExecutionStack
    : public TExecutionStackBase
{
public:
    explicit TExecutionStack(size_t size);
    ~TExecutionStack();

    static void SetOpaque(void* opaque);
    static void* GetOpaque();

    void SetTrampoline(void (*callee)(void*));

private:
    friend class TExecutionContext;

    void* const Handle_;
    void (*Trampoline_)(void*) = nullptr;

    static VOID CALLBACK FiberTrampoline(PVOID opaque);

    friend TExecutionContext CreateExecutionContext(
        TExecutionStack* stack,
        void (*trampoline)(void*));
};

#else
#   error Unsupported platform
#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
