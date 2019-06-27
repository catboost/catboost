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
    class TStack : TNonCopyable {
    public:
        enum class EGuard {
            Canary /* "canary" */,
            Page /* "page" */,
        };

        explicit TStack(ui32 sz, EGuard) noexcept;
        ~TStack();

        TArrayRef<char> Get() noexcept;

        bool LowerCanaryOk() const noexcept;

        bool UpperCanaryOk() const noexcept;

    private:
        const EGuard Guard_;
        const ui32 Size_;
        char* const Data_;
        size_t StackId_ = 0;
    };


    class TTrampoline : public ITrampoLine, TNonCopyable {
    public:
        TTrampoline(
            ui32 stackSize,
            TStack::EGuard guard,
            TContFunc f,
            TCont* cont,
            void* arg
        ) noexcept;

        TArrayRef<char> Stack() noexcept;

        TExceptionSafeContext* Context() noexcept {
            return &Ctx_;
        }

        TContFunc Func() const noexcept {
            return Func_;
        }

        void* Arg() const noexcept {
            return Arg_;
        }

        void SwitchTo(TExceptionSafeContext* ctx) noexcept {
            Y_VERIFY(Stack_.LowerCanaryOk(), "Stack overflow");
            Y_VERIFY(Stack_.UpperCanaryOk(), "Stack override");
            Ctx_.SwitchTo(ctx);
        }

        void DoRun();

    private:
        TStack Stack_;
        const TContClosure Clo_;
        TExceptionSafeContext Ctx_;
        TContFunc const Func_ = nullptr;
        TCont* const Cont_;
        void* const Arg_;
    };


    // accounts for the stack space overhead in sanitizers and debug builds
    ui32 RealCoroStackSize(ui32 coroStackSize);

    TMaybe<ui32> RealCoroStackSize(TMaybe<ui32> coroStackSize);
}
