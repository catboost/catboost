#include "context.h"
#include "backtrace.h"

#if defined(FRAME_CNT) || defined(JUMP_FUNCTION) || defined(JUMP_ARGUMENT) || defined(JUMP_LINK) || \
    defined(MJB_X19) || defined(MJB_X20) || defined(MJB_LR) || defined(MJB_R12) ||                  \
    defined(MJB_R13) || defined(MJB_SI) || defined(MJB_DI) || defined(MJB_RBP) ||                   \
    defined(MJB_RSP) || defined(MJB_BP) || defined(MJB_SP) || defined(MJB_PC)
    #error "context implementation macros leaked from context.h"
#endif

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/deque.h>
#include <util/generic/yexception.h>

#include <algorithm>

struct TContextBacktraceState {
    TExceptionSafeContext* Main;
    TExceptionSafeContext* Next;
    void** Frames;
    size_t FrameCapacity;
    size_t FrameCount;
};

static Y_NO_INLINE void CaptureBacktraceFromContext(TContextBacktraceState* state) {
    state->FrameCount = BackTrace(state->Frames, state->FrameCapacity);
    state->Next->SwitchTo(state->Main);
}

Y_UNIT_TEST_SUITE(TestContext) {
    template <class F>
    static TContClosure Wrap(F& f) {
        struct TW: public ITrampoLine {
            inline TW(F* ff) noexcept
                : F_(ff)
            {
            }

            void DoRun() override {
                (*F_)();
            }

            F* F_;
            char Buf[1000000];
        };

        static TDeque<TW> w;

        auto& tw = w.emplace_back(&f);

        return {&tw, TArrayRef(tw.Buf, sizeof(tw.Buf))};
    }

    Y_UNIT_TEST(TestExceptionSafety) {
        TExceptionSafeContext main;
        TExceptionSafeContext* volatile nextPtr = nullptr;

        bool hasUncaught = true;

        auto func = [&]() {
            hasUncaught = UncaughtException();
            nextPtr->SwitchTo(&main);
        };

        auto closure = Wrap(func);

        TExceptionSafeContext next(closure);

        nextPtr = &next;

        struct THelper {
            inline ~THelper() {
                M->SwitchTo(N);
            }

            TExceptionSafeContext* M;
            TExceptionSafeContext* N;
        };

        bool throwed = false;

        try {
            THelper helper{&main, &next};

            throw 1;
        } catch (...) {
            throwed = true;
        }

        UNIT_ASSERT(throwed);
        UNIT_ASSERT(!hasUncaught);
    }

    Y_UNIT_TEST(TestBacktrace) {
        TExceptionSafeContext main;

        constexpr size_t FrameCapacity = 64;
        void* frames[FrameCapacity] = {};
        TContextBacktraceState state{&main, nullptr, frames, FrameCapacity, 0};

        auto func = [&]() {
            CaptureBacktraceFromContext(&state);
        };

        auto closure = Wrap(func);
        std::fill(closure.Stack.begin(), closure.Stack.end(), '\xA5');

        TExceptionSafeContext next(closure);
        state.Next = &next;

        main.SwitchTo(&next);

#if defined(_win_)
        UNIT_ASSERT_GE(state.FrameCount, 2u);
#else
        auto testBodyAddress = reinterpret_cast<size_t>(reinterpret_cast<void*>(&CaptureBacktraceFromContext));
        bool foundTestBody = false;
        for (size_t index = 0; index < state.FrameCount; ++index) {
            auto frameAddress = reinterpret_cast<size_t>(frames[index]);
            foundTestBody |= frameAddress >= testBodyAddress && frameAddress - testBodyAddress < 4096;
        }
        UNIT_ASSERT_C(foundTestBody, "Backtrace did not unwind to the coroutine body");
#endif
        UNIT_ASSERT_C(state.FrameCount < FrameCapacity, "Backtrace did not stop before reaching its capacity");
    }
} // Y_UNIT_TEST_SUITE(TestContext)
