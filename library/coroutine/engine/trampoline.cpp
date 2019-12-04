#include "impl.h"
#include "trampoline.h"

#include <util/system/info.h>
#include <util/system/protect.h>
#include <util/system/valgrind.h>
#include <util/system/yassert.h>

#include <cstdlib>
#include <util/stream/format.h>

namespace NCoro {
    namespace NPrivate {
        ui32 RawStackSize(ui32 sz, ui32 guardSize) {
            // reservation for unaligned allocation and 2 guards
            return Max<ui32>(sz, 1) + 4 * guardSize;
        }

        TArrayRef<char> AlignedRange(char* data, ui32 sz, ui32 guardSize) {
            return {
                AlignUp(data, guardSize),
                AlignDown(data + sz, guardSize)
            };
        }
    }

    namespace {
        constexpr TStringBuf CANARY = AsStringBuf(
            "[IfYouReadThisTheStackIsStillOK]"
        );
        static_assert(CANARY.size() == 32);

        ui32 GuardSize(TStack::EGuard guard) {
            static const ui32 pageSize = NSystemInfo::GetPageSize();
            switch (guard) {
            case TStack::EGuard::Canary:
                return CANARY.size();
            case TStack::EGuard::Page:
                return pageSize;
            }
        }

        void ProtectWithCanary(TArrayRef<char> alignedRange) {
            constexpr ui32 guardSize = CANARY.size();
            memcpy(
                alignedRange.data(),
                NCoro::CANARY.data(),
                guardSize
            );
            memcpy(
                alignedRange.end() - guardSize,
                NCoro::CANARY.data(),
                guardSize
            );
        }

        void ProtectWithPages(TArrayRef<char> alignedRange, EProtectMemory mode) {
            static const ui32 guardSize = NSystemInfo::GetPageSize();
            ProtectMemory(
                alignedRange.data(),
                guardSize,
                mode
            );
            ProtectMemory(
                alignedRange.end() - guardSize,
                guardSize,
                mode
            );
        }
    }

    TStack::TStack(ui32 sz, TStack::EGuard guard) noexcept
        : Guard_(guard)
        , RawSize_(NPrivate::RawStackSize(sz, GuardSize(Guard_)))
        , RawPtr_((char*) malloc(RawSize_))
    {
        const auto guardSize = GuardSize(Guard_);
        const auto alignedRange = NPrivate::AlignedRange(RawPtr_, RawSize_, guardSize);
        switch (Guard_) {
        case EGuard::Canary:
            ProtectWithCanary(alignedRange);
            break;
        case EGuard::Page:
            ProtectWithPages(alignedRange, PM_NONE);
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
            const auto alignedRange = NPrivate::AlignedRange(RawPtr_, RawSize_, GuardSize(Guard_));
            ProtectWithPages(alignedRange, PM_WRITE | PM_READ);
        }
        VALGRIND_STACK_DEREGISTER(StackId_);
        free(RawPtr_);
    }

    TArrayRef<char> TStack::Get() noexcept {
        const auto guardSize = GuardSize(Guard_);
        const auto alignedRange = NPrivate::AlignedRange(RawPtr_, RawSize_, guardSize);
        return {
            alignedRange.data() + guardSize,
            alignedRange.end() - guardSize
        };
    }

    bool TStack::LowerCanaryOk() const noexcept {
        constexpr auto guardSize = NCoro::CANARY.size();
        const auto alignedRange = NPrivate::AlignedRange(RawPtr_, RawSize_, guardSize);
        return Guard_ != EGuard::Canary || TStringBuf(
            alignedRange.data(),
            guardSize
        ) == NCoro::CANARY;
    }

    bool TStack::UpperCanaryOk() const noexcept {
        constexpr auto guardSize = NCoro::CANARY.size();
        const auto alignedRange = NPrivate::AlignedRange(RawPtr_, RawSize_, guardSize);
        return Guard_ != EGuard::Canary || TStringBuf(
            alignedRange.end() - guardSize,
            guardSize
        ) == NCoro::CANARY;
    }


    TTrampoline::TTrampoline(ui32 stackSize, TStack::EGuard guard, TContFunc f, TCont* cont, void* arg) noexcept
        : Stack_(stackSize, guard)
        , Clo_{this, Stack_.Get(), cont->Name()}
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

    const char* TTrampoline::ContName() const noexcept {
        return Cont_->Name();
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
