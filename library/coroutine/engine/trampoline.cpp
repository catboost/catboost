#include "impl.h"
#include "trampoline.h"

#include <util/system/yassert.h>
#include <util/system/valgrind.h>

#include <cstdlib>

namespace NCoro {
    ui32 RealCoroStackSize(ui32 coroStackSize) {
#if defined(_san_enabled_) || !defined(NDEBUG)
        coroStackSize *= 4;
#endif
        return coroStackSize;
    }

    TMaybe<ui32> RealCoroStackSize(TMaybe<ui32> coroStackSize) {
        if (coroStackSize) {
            return RealCoroStackSize(*coroStackSize);
        } else {
            return Nothing();
        }
    }

    namespace {
        constexpr TStringBuf CANARY = AsStringBuf(
            "4ef8f9c2f7eb6cb8af66f2e441f4250c0f819a30d07821895b53e6017f90fbcd");
    }

    TTrampoline::TTrampoline(ui32 stackSize, TContFunc f, TCont* cont, void* arg) noexcept
        : Stack_((char*) malloc(stackSize))
        , StackSize_(stackSize)
        , Clo_{this, {
            Stack_.Get() + NCoro::CANARY.size(),
            StackSize_ - NCoro::CANARY.size()
        }}
        , Ctx_(Clo_)
        , Func_(f)
        , Cont_(cont)
        , Arg_(arg)
    {
        Y_VERIFY(Stack_, "out of memory");
        StackId_ = VALGRIND_STACK_REGISTER(
            Stack_.Get() + NCoro::CANARY.size(),
            Stack_.Get() + StackSize_ - NCoro::CANARY.size()
        );
        memcpy(Stack_.Get(), NCoro::CANARY.data(), NCoro::CANARY.size());
    }

    TTrampoline::~TTrampoline() {
        VALGRIND_STACK_DEREGISTER(StackId_);
    }

    void TTrampoline::SwitchTo(TExceptionSafeContext* ctx) noexcept {
        Y_VERIFY(
            TStringBuf(Stack_.Get(), NCoro::CANARY.size()) == NCoro::CANARY,
            "Stack overflow"
        );
        Ctx_.SwitchTo(ctx);
    }

    void TTrampoline::DoRun() {
        try {
            Func_(Cont_, Arg_);
        } catch (...) {
            Y_VERIFY(
                !Cont_->Executor()->FailOnError(),
                "uncaught exception %s", CurrentExceptionMessage().c_str()
            );
        }

        Cont_->Terminate();
    }
}
