#include "impl.h"
#include "trampoline.h"

#include <util/system/info.h>
#include <util/system/protect.h>
#include <util/system/valgrind.h>
#include <util/system/yassert.h>

#include <cstdlib>
#include <util/stream/format.h>

namespace NCoro {
    namespace {
        constexpr TStringBuf CANARY = AsStringBuf(
            "\x4e\xf8\xf9\xc2\xf7\xeb\x6c\xb8"
            "\xaf\x66\xf2\xe4\x41\xf4\x25\x0c"
            "\x0f\x81\x9a\x30\xd0\x78\x21\x89"
            "\x5b\x53\xe6\x01\x7f\x90\xfb\xcd"
        );

        ui32 GuardSize(TStack::EGuard guard) {
            static const ui32 pageSize = NSystemInfo::GetPageSize();
            switch (guard) {
            case TStack::EGuard::Canary:
                return CANARY.size();
            case TStack::EGuard::Page:
                return pageSize;
            }
        }

        void ProtectWithCanary(char* data, size_t sz) {
            constexpr ui32 guardSize = CANARY.size();
            memcpy(
                AlignUp(data, guardSize),
                NCoro::CANARY.data(),
                guardSize
            );
            memcpy(
                AlignUp(data, guardSize) + sz - guardSize,
                NCoro::CANARY.data(),
                guardSize
            );
        }

        void ProtectWithPages(char* data, size_t sz, EProtectMemory mode) {
            static const ui32 guardSize = NSystemInfo::GetPageSize();
            ProtectMemory(
                AlignUp(data, guardSize),
                guardSize,
                mode
            );
            ProtectMemory(
                AlignUp(data, guardSize) + sz - guardSize,
                guardSize,
                mode
            );
        }
    }

    TStack::TStack(ui32 sz, TStack::EGuard guard) noexcept
        : Guard_(guard)
        , Size_(
            AlignUp(sz, GuardSize(Guard_)) + 2 * GuardSize(Guard_)
        ) // reservation for unaligned allocation and 2 guards
        , Data_((char*) malloc(Size_ + GuardSize(Guard_)))
    {
        switch (Guard_) {
        case EGuard::Canary:
            ProtectWithCanary(Data_, Size_);
            break;
        case EGuard::Page:
            ProtectWithPages(Data_, Size_, PM_NONE);
            break;
        }

        StackId_ = VALGRIND_STACK_REGISTER(
            Get().data(),
            Get().size()
        );
    }

    TStack::~TStack()
    {
        if (Guard_ == EGuard::Page) {
            ProtectWithPages(Data_, Size_, PM_WRITE | PM_READ);
        }
        free(Data_);
        VALGRIND_STACK_DEREGISTER(StackId_);
    }

    TArrayRef<char> TStack::Get() noexcept {
        const size_t guardSize = GuardSize(Guard_);
        return {
            AlignUp(Data_, guardSize) + guardSize,
            AlignUp(Data_, guardSize) + Size_ - guardSize
        };
    }

    bool TStack::LowerCanaryOk() const noexcept {
        constexpr ui32 guardSize = NCoro::CANARY.size();
        return Guard_ != EGuard::Canary || TStringBuf(
            AlignUp(Data_, guardSize),
            guardSize
        ) == NCoro::CANARY;
    }

    bool TStack::UpperCanaryOk() const noexcept {
        constexpr ui32 guardSize = NCoro::CANARY.size();
        return Guard_ != EGuard::Canary || TStringBuf(
            AlignUp(Data_, guardSize) + Size_ - guardSize,
            guardSize
        ) == NCoro::CANARY;
    }


    TTrampoline::TTrampoline(ui32 stackSize, TStack::EGuard guard, TContFunc f, TCont* cont, void* arg) noexcept
        : Stack_(stackSize, guard)
        , Clo_{this, Stack_.Get()}
        , Ctx_(Clo_)
        , Func_(f)
        , Cont_(cont)
        , Arg_(arg)
    {}

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

    TArrayRef<char> TTrampoline::Stack() noexcept {
        return Stack_.Get();
    }


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
}
