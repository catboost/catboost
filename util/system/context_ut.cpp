#include "context_config.h"

// Snapshot USE_JUMP_CONT before context.h clears private backend macros.
#if defined(_unix_) && defined(USE_JUMP_CONT)
    #define HAS_ASSEMBLY_CONTEXT_TRAMPOLINE 1
#endif

#include "context.h"
#include "sanitizers.h"

#if defined(HAS_ASSEMBLY_CONTEXT_TRAMPOLINE)
    #include "backtrace.h"

    #include <csignal>
    #include <sys/wait.h>
    #include <unistd.h>

extern "C" void ContextTrampoLine();
#endif

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/deque.h>
#include <util/generic/yexception.h>

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

#if defined(HAS_ASSEMBLY_CONTEXT_TRAMPOLINE)
    static void AssertChildAborted(pid_t pid) {
        UNIT_ASSERT_C(pid >= 0, "fork failed");

        int status = 0;
        UNIT_ASSERT_VALUES_EQUAL(waitpid(pid, &status, 0), pid);
        UNIT_ASSERT_C(WIFSIGNALED(status), status);
        UNIT_ASSERT_VALUES_EQUAL(WTERMSIG(status), SIGABRT);
    }
#endif

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

#if defined(HAS_ASSEMBLY_CONTEXT_TRAMPOLINE) && !defined(_musl_)
    Y_UNIT_TEST(TestUnwindStopsAtTrampoline) {
        static constexpr size_t MaxFrames = 64;

        TExceptionSafeContext main;
        TExceptionSafeContext* nextPtr = nullptr;

        size_t frameCount = 0;
        size_t meaningfulFrameCount = 0;
        size_t trampolineFrameCount = 0;
        size_t trampolineFrameIndex = 0;
    #if defined(_x86_64_)
        void** inheritedReturnAddressSlot = nullptr;
    #endif

        auto func = [&]() {
            const void* const trampolineAddress = reinterpret_cast<void*>(&ContextTrampoLine);

    #if defined(_x86_64_)
            UNIT_ASSERT(inheritedReturnAddressSlot != nullptr);
            void* const inheritedReturnAddress = *inheritedReturnAddressSlot;
            UNIT_ASSERT(inheritedReturnAddress != nullptr);

            char inheritedSymbolBuffer[1024];
            const auto inheritedSymbol = ResolveSymbol(inheritedReturnAddress, inheritedSymbolBuffer, sizeof(inheritedSymbolBuffer));
            UNIT_ASSERT_VALUES_EQUAL(inheritedSymbol.NearestSymbol, trampolineAddress);
    #endif

            void* frames[MaxFrames];
            frameCount = BackTrace(frames, Y_ARRAY_SIZE(frames));
            meaningfulFrameCount = frameCount;
            // Some _Unwind_Backtrace implementations report a terminal zero-IP context.
            if (frameCount != 0 && frames[frameCount - 1] == nullptr) {
                --meaningfulFrameCount;
            }

            for (size_t index = 0; index < meaningfulFrameCount; ++index) {
                char symbolBuffer[1024];
                const auto symbol = ResolveSymbol(frames[index], symbolBuffer, sizeof(symbolBuffer));
                if (symbol.NearestSymbol == trampolineAddress) {
                    ++trampolineFrameCount;
                    trampolineFrameIndex = index;
                }
            }

            nextPtr->SwitchTo(&main);
        };

        auto closure = Wrap(func);
        TExceptionSafeContext next(closure);
        nextPtr = &next;

    #if defined(_x86_64_)
        // The assembly CALL initializes this slot without updating MSan shadow state.
        // Its nonzero value ensures the CFI root, not a zero return address, stops unwinding.
        auto* stackTop = AlignDown(closure.Stack.data() + closure.Stack.size(), STACK_ALIGN);
        inheritedReturnAddressSlot = reinterpret_cast<void**>(stackTop) - 3;
        NSan::Unpoison(inheritedReturnAddressSlot, sizeof(*inheritedReturnAddressSlot));
    #endif

        main.SwitchTo(&next);

        UNIT_ASSERT_C(frameCount + 1 < MaxFrames, frameCount);
        UNIT_ASSERT_VALUES_EQUAL(trampolineFrameCount, 1);
        UNIT_ASSERT_VALUES_EQUAL(trampolineFrameIndex + 1, meaningfulFrameCount);
    }
#endif

#if defined(HAS_ASSEMBLY_CONTEXT_TRAMPOLINE)
    Y_UNIT_TEST(TestReturningTrampolineAborts) {
        struct TReturningTrampoline final: ITrampoLine {
            void DoRunNaked() override {
            }

            char Stack[1000000];
        };

        static TReturningTrampoline trampoline;
        const pid_t pid = fork();

        if (pid == 0) {
            TExceptionSafeContext main;
            TExceptionSafeContext next({&trampoline, TArrayRef(trampoline.Stack, sizeof(trampoline.Stack))});

            main.SwitchTo(&next);
            _exit(0);
        }

        AssertChildAborted(pid);
    }
#endif

#if defined(HAS_ASSEMBLY_CONTEXT_TRAMPOLINE) && !defined(NDEBUG)
    Y_UNIT_TEST(TestCorruptedTrampolineArguments) {
        const auto run = [](ptrdiff_t corruptedCopy) {
            const pid_t pid = fork();

            if (pid == 0) {
                TExceptionSafeContext main;
                TExceptionSafeContext* nextPtr = nullptr;

                auto func = [&]() {
                    nextPtr->SwitchTo(&main);
                };

                auto closure = Wrap(func);
                TExceptionSafeContext next(closure);
                nextPtr = &next;

                auto* stackTop = AlignDown(closure.Stack.data() + closure.Stack.size(), STACK_ALIGN);
                reinterpret_cast<ITrampoLine**>(stackTop)[corruptedCopy] = nullptr;

                main.SwitchTo(&next);
                _exit(0);
            }

            AssertChildAborted(pid);
        };

        run(-1);
        run(-2);
    }
#endif
} // Y_UNIT_TEST_SUITE(TestContext)

#if defined(HAS_ASSEMBLY_CONTEXT_TRAMPOLINE)
    #undef HAS_ASSEMBLY_CONTEXT_TRAMPOLINE
#endif
