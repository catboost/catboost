#include <library/cpp/unittest/registar.h>
#include <library/cpp/threading/mux_event/mux_event.h>
#include <util/system/thread.h>

struct TMuxEventText: public TTestBase {
    UNIT_TEST_SUITE(TMuxEventText);
    UNIT_TEST(CheckOneWait);
    UNIT_TEST(CheckTwoWait);
    UNIT_TEST(CheckSignalAfterWait);
    UNIT_TEST_SUITE_END();

    void CheckOneWait() {
        TMuxEvent e0;
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, TDuration::MicroSeconds(0)), -1);
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, TDuration::MilliSeconds(50)), -1);

        e0.Signal();
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, TDuration::MicroSeconds(0)), 0);
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0), 0);

        e0.Reset();
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, TDuration::MicroSeconds(0)), -1);
    }

    void CheckTwoWait() {
        TMuxEvent e0, e1;
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, e1, TDuration::MicroSeconds(0)), -1);
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, e1, TDuration::MilliSeconds(50)), -1);

        e0.Signal();
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, e1, TDuration::MicroSeconds(0)), 0);

        e1.Signal();
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, e1, TDuration::MicroSeconds(0)), 0);

        e0.Reset();
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, e1), 1);

        e1.Reset();
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, e1, TDuration::MicroSeconds(0)), -1);
    }

    struct TTestFuncParams {
        TSystemEvent SyncEvent;
        TMuxEvent TestEvent;
    };

    static void* TestFunc(void* params) {
        static_cast<TTestFuncParams*>(params)->SyncEvent.Signal();
        Sleep(TDuration::MilliSeconds(100)); // ensure that WaitForAnyEvent is called
        static_cast<TTestFuncParams*>(params)->TestEvent.Signal();
        return nullptr;
    }

    void CheckSignalAfterWait() {
        TTestFuncParams params;
        TThread test(&TMuxEventText::TestFunc, &params);
        test.Start();

        params.SyncEvent.Wait();

        TMuxEvent e0, e2;
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, params.TestEvent, e2, TDuration::Seconds(1)), 1);
        UNIT_ASSERT_VALUES_EQUAL(WaitForAnyEvent(e0, params.TestEvent, e2), 1);
    }
};
UNIT_TEST_SUITE_REGISTRATION(TMuxEventText);

// TMuxEvent implements Event semantics too
#include <util/system/event.h> // for pragma once in util/system/event_ut.cpp
#define TSystemEvent TMuxEvent
#include <util/system/event_ut.cpp>
