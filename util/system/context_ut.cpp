#include "context.h"

#include <library/cpp/unittest/registar.h>

#include <util/generic/deque.h>

Y_UNIT_TEST_SUITE(TestContext) {
    template <class F>
    static TContClosure Wrap(F & f) {
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

        bool uncaught = true;

        auto func = [&main, &nextPtr, &uncaught]() {
            uncaught = std::uncaught_exception();
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
        UNIT_ASSERT(!uncaught);
    }
}
