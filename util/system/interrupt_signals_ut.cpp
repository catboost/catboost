#include "interrupt_signals.h"

#include "atomic.h"

#include <util/datetime/base.h>

#include <library/cpp/unittest/registar.h>

#ifdef _win_
    #include <windows.h>
#endif


Y_UNIT_TEST_SUITE(TTestInterruptSignals){
    static TAtomic HandledSigNum = 0;

    static void Handler(int signum) {
        AtomicSet(HandledSigNum, signum);
    }

    Y_UNIT_TEST(Test1){
        SetInterruptSignalsHandler(Handler);
#ifdef _win_
        // TODO: unfortunately GenerateConsoleCtrlEvent fails under Wine
        /*
         for (auto [winEvent, posixSigNum] : {
                std::make_pair(CTRL_C_EVENT, SIGINT),
                std::make_pair(CTRL_BREAK_EVENT, SIGTERM)
            })
        {
            if (!GenerateConsoleCtrlEvent(winEvent, 0)) {
                UNIT_FAIL("GenerateConsoleCtrlEvent failed: " << LastSystemErrorText());
            }
            Sleep(TDuration::MilliSeconds(100));
            UNIT_ASSERT_VALUES_EQUAL(HandledSigNum, posixSigNum);
        }
        */
        for (int signum : {SIGINT, SIGTERM}) {
#else
        for (int signum : {SIGINT, SIGTERM, SIGHUP}) {
#endif
            std::raise(signum);
            Sleep(TDuration::MilliSeconds(100)); // give it time to handle an async signal
            UNIT_ASSERT_VALUES_EQUAL(HandledSigNum, signum);
        }
    }
}
