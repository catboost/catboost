#pragma once

#include "stack/stack_common.h"
#include "stack/stack.h"

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

    namespace NStack {
        class IAllocator;
    }

    class TTrampoline : public ITrampoLine, TNonCopyable {
    public:
        TTrampoline(
            NCoro::NStack::IAllocator& allocator,
            uint32_t stackSize,
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
            Y_VERIFY(Stack_.LowerCanaryOk(), "Stack overflow (%s)", ContName());
            Y_VERIFY(Stack_.UpperCanaryOk(), "Stack override (%s)", ContName());
            Ctx_.SwitchTo(ctx);
        }

        void DoRun() override;

        void DoRunNaked() override;

    private:
        const char* ContName() const noexcept;
    private:
        NStack::TStackHolder Stack_;
        const TContClosure Clo_;
        TExceptionSafeContext Ctx_;
        TContFunc const Func_ = nullptr;
        TCont* const Cont_;
        void* const Arg_;
    };
}
