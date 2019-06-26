#pragma once

#include <util/generic/noncopyable.h>
#include <util/generic/ptr.h>
#include <util/system/context.h>
#include <util/system/defaults.h>

#if !defined(STACK_GROW_DOWN)
#   error "unsupported"
#endif

// TODO(velavokr): allow any std::function objects, not only TContFunc
class TCont;
typedef void (*TContFunc)(TCont*, void*);

namespace NCoro {
    class IScheduleCallback;

    // accounts for asan stack space overhead
    ui32 RealCoroStackSize(ui32 coroStackSize);
    TMaybe<ui32> RealCoroStackSize(TMaybe<ui32> coroStackSize);

    class TTrampoline : public ITrampoLine, TNonCopyable {
    public:
        TTrampoline(
            ui32 stackSize,
            TContFunc f,
            TCont* cont,
            void* arg
        ) noexcept;

        ~TTrampoline();

        void SwitchTo(TExceptionSafeContext* ctx) noexcept;

        void DoRun();

    public:
        const THolder<char, TFree> Stack_;
        const ui32 StackSize_;
        const TContClosure Clo_;
        TExceptionSafeContext Ctx_;
        TContFunc const Func_ = nullptr;
        TCont* const Cont_;
        size_t StackId_ = 0;
        void* const Arg_;
    };
}
