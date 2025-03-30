#include "impl.h"
#include "condvar.h"
#include "network.h"

#include <library/cpp/deprecated/atomic/atomic.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/string/cast.h>
#include <util/system/pipe.h>
#include <util/system/env.h>
#include <util/system/info.h>
#include <util/system/thread.h>
#include <util/generic/xrange.h>
#include <util/generic/serialized_enum.h>

// TODO (velavokr): BALANCER-1345 add more tests on pollers

class TCoroTest: public TTestBase {
    UNIT_TEST_SUITE(TCoroTest);
    UNIT_TEST(TestSimpleX1);
    UNIT_TEST(TestSimpleX1MultiThread);
    UNIT_TEST(TestSimpleX2);
    UNIT_TEST(TestSimpleX3);
    UNIT_TEST(TestMemFun);
    UNIT_TEST(TestMutex);
    UNIT_TEST(TestCondVar);
    UNIT_TEST(TestJoinDefault);
    UNIT_TEST(TestJoinEpoll);
    UNIT_TEST(TestJoinKqueue);
    UNIT_TEST(TestJoinPoll);
    UNIT_TEST(TestJoinSelect);
    UNIT_TEST(TestException);
    UNIT_TEST(TestJoinCancelExitRaceBug);
    UNIT_TEST(TestWaitWakeLivelockBug);
// TODO (velavokr): BALANCER-1338 our epoll wrapper cannot handle pipe eofs
//    UNIT_TEST(TestFastPathWakeDefault)
//    UNIT_TEST(TestFastPathWakeEpoll)
    UNIT_TEST(TestFastPathWakeKqueue)
    UNIT_TEST(TestFastPathWakePoll)
    UNIT_TEST(TestFastPathWakeSelect)
    UNIT_TEST(TestLegacyCancelYieldRaceBug)
    UNIT_TEST(TestJoinRescheduleBug);
    UNIT_TEST(TestEventQueue)
    UNIT_TEST(TestNestedExecutor)
    UNIT_TEST(TestComputeCoroutineYield)
    UNIT_TEST(TestPollEngines);
    UNIT_TEST(TestUserEvent);
    UNIT_TEST(TestPause);
    UNIT_TEST(TestOverrideTime);
    UNIT_TEST(TestCancelWithException);
    UNIT_TEST_SUITE_END();

public:
    void TestException();
    void TestSimpleX1();
    void TestSimpleX1MultiThread();
    void TestSimpleX2();
    void TestSimpleX3();
    void TestMemFun();
    void TestMutex();
    void TestCondVar();
    void TestJoinDefault();
    void TestJoinEpoll();
    void TestJoinKqueue();
    void TestJoinPoll();
    void TestJoinSelect();
    void TestJoinCancelExitRaceBug();
    void TestWaitWakeLivelockBug();
    void TestFastPathWakeDefault();
    void TestFastPathWakeEpoll();
    void TestFastPathWakeKqueue();
    void TestFastPathWakePoll();
    void TestFastPathWakeSelect();
    void TestLegacyCancelYieldRaceBug();
    void TestJoinRescheduleBug();
    void TestEventQueue();
    void TestNestedExecutor();
    void TestComputeCoroutineYield();
    void TestPollEngines();
    void TestUserEvent();
    void TestPause();
    void TestOverrideTime();
    void TestCancelWithException();
};

void TCoroTest::TestException() {
    TContExecutor e(1000000);

    bool f2run = false;

    auto f1 = [&f2run](TCont* c) {
        struct TCtx {
            ~TCtx() {
                Y_ABORT_UNLESS(!*F2);

                C->Yield();
            }

            TCont* C;
            bool* F2;
        };

        try {
            TCtx ctx = {c, &f2run};

            throw 1;
        } catch (...) {
        }
    };

    bool unc = true;

    auto f2 = [&unc, &f2run](TCont*) {
        f2run = true;
        unc = std::uncaught_exceptions();

        // check segfault
        try {
            throw 2;
        } catch (int) {
        }
    };

    e.Create(f1, "f1");
    e.Create(f2, "f2");
    e.Execute();

    UNIT_ASSERT(!unc);
}

static int i0;

static void CoRun(TCont* c, void* /*run*/) {
    while (i0 < 100000) {
        ++i0;
        UNIT_ASSERT(RunningCont() == c);
        c->Yield();
        UNIT_ASSERT(RunningCont() == c);
    }
}

static void CoMain(TCont* c, void* /*arg*/) {
    for (volatile size_t i2 = 0; i2 < 10; ++i2) {
        UNIT_ASSERT(RunningCont() == c);
        c->Executor()->Create(CoRun, nullptr, "run");
        UNIT_ASSERT(RunningCont() == c);
    }
}

void TCoroTest::TestSimpleX1() {
    i0 = 0;
    TContExecutor e(32000);

    UNIT_ASSERT(RunningCont() == nullptr);

    e.Execute(CoMain);
    UNIT_ASSERT_VALUES_EQUAL(i0, 100000);

    UNIT_ASSERT(RunningCont() == nullptr);
}

void TCoroTest::TestSimpleX1MultiThread() {
    TVector<THolder<TThread>> threads;
    const size_t nThreads = 0;
    TAtomic c = 0;
    for (size_t i = 0; i < nThreads; ++i) {
        threads.push_back(MakeHolder<TThread>([&]() {
            TestSimpleX1();
            AtomicIncrement(c);
        }));
    }

    for (auto& t : threads) {
        t->Start();
    }

    for (auto& t: threads) {
        t->Join();
    }

    UNIT_ASSERT_EQUAL(c, nThreads);
}

struct TTestObject {
    int i = 0;
    int j = 0;

public:
    void RunTask1(TCont*) {
        i = 1;
    }
    void RunTask2(TCont*) {
        j = 2;
    }
};

void TCoroTest::TestMemFun() {
    i0 = 0;
    TContExecutor e(32000);
    TTestObject obj;
    e.Create<TTestObject, &TTestObject::RunTask1>(&obj, "test1");
    e.Execute<TTestObject, &TTestObject::RunTask2>(&obj);
    UNIT_ASSERT_EQUAL(obj.i, 1);
    UNIT_ASSERT_EQUAL(obj.j, 2);
}

void TCoroTest::TestSimpleX2() {
    {
        i0 = 0;

        {
            TContExecutor e(32000);
            e.Execute(CoMain);
        }

        UNIT_ASSERT_EQUAL(i0, 100000);
    }

    {
        i0 = 0;

        {
            TContExecutor e(32000);
            e.Execute(CoMain);
        }

        UNIT_ASSERT_EQUAL(i0, 100000);
    }
}

struct TRunner {
    inline TRunner()
        : Runs(0)
    {
    }

    inline void operator()(TCont* c) {
        ++Runs;
        c->Yield();
    }

    size_t Runs;
};

void TCoroTest::TestSimpleX3() {
    TContExecutor e(32000);
    TRunner runner;

    for (volatile size_t i3 = 0; i3 < 1000; ++i3) {
        e.Create(runner, "runner");
    }

    e.Execute();

    UNIT_ASSERT_EQUAL(runner.Runs, 1000);
}

static TString res;
static TContMutex mutex;

static void CoMutex(TCont* c, void* /*run*/) {
    {
        mutex.LockI(c);
        c->Yield();
        res += c->Name();
        mutex.UnLock();
    }

    c->Yield();

    {
        mutex.LockI(c);
        c->Yield();
        res += c->Name();
        mutex.UnLock();
    }
}

static void CoMutexTest(TCont* c, void* /*run*/) {
    c->Executor()->Create(CoMutex, nullptr, "1");
    c->Executor()->Create(CoMutex, nullptr, "2");
}

void TCoroTest::TestMutex() {
    TContExecutor e(32000);
    e.Execute(CoMutexTest);
    UNIT_ASSERT_EQUAL(res, "1212");
    res.clear();
}

static TContMutex m1;
static TContCondVar c1;

static void CoCondVar(TCont* c, void* /*run*/) {
    for (size_t i4 = 0; i4 < 3; ++i4) {
        UNIT_ASSERT_EQUAL(m1.LockI(c), 0);
        UNIT_ASSERT_EQUAL(c1.WaitI(c, &m1), 0);
        res += c->Name();
        m1.UnLock();
    }
}

static void CoCondVarTest(TCont* c, void* /*run*/) {
    c->Executor()->Create(CoCondVar, nullptr, "1");
    c->Yield();
    c->Executor()->Create(CoCondVar, nullptr, "2");
    c->Yield();
    c->Executor()->Create(CoCondVar, nullptr, "3");
    c->Yield();
    c->Executor()->Create(CoCondVar, nullptr, "4");
    c->Yield();
    c->Executor()->Create(CoCondVar, nullptr, "5");
    c->Yield();
    c->Executor()->Create(CoCondVar, nullptr, "6");
    c->Yield();

    for (size_t i5 = 0; i5 < 3; ++i5) {
        res += ToString((size_t)i5) + "^";
        c1.BroadCast();
        c->Yield();
    }
}

void TCoroTest::TestCondVar() {
    TContExecutor e(32000);
    e.Execute(CoCondVarTest);
    UNIT_ASSERT_EQUAL(res, "0^1234561^1234562^123456");
    res.clear();
}

namespace NCoroTestJoin {
    struct TSleepCont {
        const TInstant Deadline;
        int Result;

        inline void operator()(TCont* c) {
            Result = c->SleepD(Deadline);
        }
    };

    struct TReadCont {
        const TInstant Deadline;
        const SOCKET Sock;
        int Result;

        inline void operator()(TCont* c) {
            char buf = 0;
            Result = NCoro::ReadD(c, Sock, &buf, sizeof(buf), Deadline).Status();
        }
    };

    struct TJoinCont {
        const TInstant Deadline;
        TCont* const Cont;
        bool Result;

        inline void operator()(TCont* c) {
            Result = c->Join(Cont, Deadline);
        }
    };

    void DoTestJoin(EContPoller pollerType) {
        auto poller = IPollerFace::Construct(pollerType);

        if (!poller) {
            return;
        }

        TContExecutor e(32000, std::move(poller));

        TPipe in, out;
        TPipe::Pipe(in, out);
        SetNonBlock(in.GetHandle());

        {
            TSleepCont sc = {TInstant::Max(), 0};
            TJoinCont jc = {TDuration::MilliSeconds(100).ToDeadLine(), e.Create(sc, "sc"), true};

            e.Execute(jc);

            UNIT_ASSERT_EQUAL(sc.Result, ECANCELED);
            UNIT_ASSERT_EQUAL(jc.Result, false);
        }

        {
            TSleepCont sc = {TDuration::MilliSeconds(100).ToDeadLine(), 0};
            TJoinCont jc = {TDuration::MilliSeconds(200).ToDeadLine(), e.Create(sc, "sc"), false};

            e.Execute(jc);

            UNIT_ASSERT_EQUAL(sc.Result, ETIMEDOUT);
            UNIT_ASSERT_EQUAL(jc.Result, true);
        }

        {
            TSleepCont sc = {TDuration::MilliSeconds(200).ToDeadLine(), 0};
            TJoinCont jc = {TDuration::MilliSeconds(100).ToDeadLine(), e.Create(sc, "sc"), true};

            e.Execute(jc);

            UNIT_ASSERT_EQUAL(sc.Result, ECANCELED);
            UNIT_ASSERT_EQUAL(jc.Result, false);
        }

        {
            TReadCont rc = {TInstant::Max(), in.GetHandle(), 0};
            TJoinCont jc = {TDuration::MilliSeconds(100).ToDeadLine(), e.Create(rc, "rc"), true};

            e.Execute(jc);

            UNIT_ASSERT_EQUAL(rc.Result, ECANCELED);
            UNIT_ASSERT_EQUAL(jc.Result, false);
        }

        {
            TReadCont rc = {TDuration::MilliSeconds(100).ToDeadLine(), in.GetHandle(), 0};
            TJoinCont jc = {TDuration::MilliSeconds(200).ToDeadLine(), e.Create(rc, "rc"), false};

            e.Execute(jc);

            UNIT_ASSERT_EQUAL(rc.Result, ETIMEDOUT);
            UNIT_ASSERT_EQUAL(jc.Result, true);
        }

        {
            TReadCont rc = {TDuration::MilliSeconds(200).ToDeadLine(), in.GetHandle(), 0};
            TJoinCont jc = {TDuration::MilliSeconds(100).ToDeadLine(), e.Create(rc, "rc"), true};

            e.Execute(jc);

            UNIT_ASSERT_EQUAL(rc.Result, ECANCELED);
            UNIT_ASSERT_EQUAL(jc.Result, false);
        }
    }
}

void TCoroTest::TestJoinDefault() {
    NCoroTestJoin::DoTestJoin(EContPoller::Default);
}

void TCoroTest::TestJoinEpoll() {
    NCoroTestJoin::DoTestJoin(EContPoller::Epoll);
}

void TCoroTest::TestJoinKqueue() {
    NCoroTestJoin::DoTestJoin(EContPoller::Kqueue);
}

void TCoroTest::TestJoinPoll() {
    NCoroTestJoin::DoTestJoin(EContPoller::Poll);
}

void TCoroTest::TestJoinSelect() {
    NCoroTestJoin::DoTestJoin(EContPoller::Select);
}

namespace NCoroJoinCancelExitRaceBug {
    struct TState {
        TCont* Sub = nullptr;
    };

    static void DoAux(TCont*, void* argPtr) noexcept {
        TState& state = *(TState*)(argPtr);

        // 06.{Ready:[Sub2]} > {Ready:[Sub2,Sub]}
        state.Sub->Cancel();
    }

    static void DoSub2(TCont*, void*) noexcept {
        // 07.{Ready:[Sub]} > Exit > {Ready:[Sub],ToDelete:[Sub2]}
        // 08.{Ready:[Sub],ToDelete:[Sub2]} > Release(Sub2) > {Ready:[Sub],Deleted:[Sub2]}
    }

    static void DoSub(TCont* cont, void* argPtr) noexcept {
        TState& state = *(TState*)(argPtr);
        state.Sub = cont;

        // 04.{Ready:[Aux]} > {Ready:[Aux,Sub2]}
        auto* sub2 = cont->Executor()->Create(DoSub2, argPtr, "Sub2");

        // 05.{Ready:[Aux,Sub2]} > SwitchTo(Aux)
        // 09.{Ready:[],Deleted:[Sub2]} > Cancel(Sub2) > {Ready:[Sub2],Deleted:[Sub2]}
        // 10.{Ready:[Sub2],Deleted:[Sub2]} > SwitchTo(Sub2) > FAIL: can not return from exit
        cont->Join(sub2);

        state.Sub = nullptr;
    }

    static void DoMain(TCont* cont) noexcept {
        TState state;

        // 01.{Ready:[]} > {Ready:[Sub]}
        auto* sub = cont->Executor()->Create(DoSub, &state, "Sub");

        // 02.{Ready:[Sub]} > {Ready:[Sub,Aux]}
        cont->Executor()->Create(DoAux, &state, "Aux");

        // 03.{Ready:[Sub,Aux]} > SwitchTo(Sub)
        cont->Join(sub);
    }
}

void TCoroTest::TestJoinCancelExitRaceBug() {
    TContExecutor exec(20000);
    exec.SetFailOnError(true);
    exec.Execute(NCoroJoinCancelExitRaceBug::DoMain);
}

namespace NCoroWaitWakeLivelockBug {
    struct TState;

    struct TSubState {
        TSubState(TState& parent, ui32 self)
            : Parent(parent)
            , Name(TStringBuilder() << "Sub" << self)
            , Self(self)
        {
            UNIT_ASSERT(self < 2);
        }

        TSubState& OtherState();

        TState& Parent;
        TTimerEvent* Event = nullptr;
        TCont* Cont = nullptr;
        TString Name;
        ui32 Self = -1;
    };

    struct TState {
        TState()
            : Subs{{*this, 0}, {*this, 1}}
        {}

        TSubState Subs[2];
        bool Stop = false;
    };

    TSubState& TSubState::OtherState() {
        return Parent.Subs[1 - Self];
    }

    static void DoStop(TCont* cont, void* argPtr) {
        TState& state = *(TState*)(argPtr);

        TTimerEvent event(cont, TInstant::Now());
        ExecuteEvent(&event);
        state.Stop = true;
        for (auto& sub: state.Subs) {
            if (sub.Event) {
                sub.Event->Wake(EWAKEDUP);
            }
        }
    }

    static void DoSub(TCont* cont, void* argPtr) {
        TSubState& state = *(TSubState*)(argPtr);

        while (!state.Parent.Stop) {
            TTimerEvent event(cont, TInstant::Max());
            if (state.OtherState().Event) {
                state.OtherState().Event->Wake(EWAKEDUP);
            }
            state.Event = &event;
            ExecuteEvent(&event);
            state.Event = nullptr;
        }

        state.Cont = nullptr;
    }

    static void DoMain(TCont* cont) noexcept {
        TState state;

        for (auto& subState : state.Subs) {
            subState.Cont = cont->Executor()->Create(DoSub, &subState, subState.Name.data());
        }

        cont->Join(
            cont->Executor()->Create(DoStop, &state, "Stop")
        );

        for (auto& subState : state.Subs) {
            if (subState.Cont) {
                cont->Join(subState.Cont);
            }
        }
    }
}

void TCoroTest::TestWaitWakeLivelockBug() {
    TContExecutor exec(20000);
    exec.SetFailOnError(true);
    exec.Execute(NCoroWaitWakeLivelockBug::DoMain);
}

namespace NCoroTestFastPathWake {
    struct TState;

    struct TSubState {
        TSubState(TState& parent, ui32 self)
            : Parent(parent)
            , Name(TStringBuilder() << "Sub" << self)
        {}

        TState& Parent;
        TInstant Finish;
        TTimerEvent* Event = nullptr;
        TCont* Cont = nullptr;
        TString Name;
    };

    struct TState {
        TState()
            : Subs{{*this, 0}, {*this, 1}}
        {
            TPipe::Pipe(In, Out);
            SetNonBlock(In.GetHandle());
        }

        TSubState Subs[2];
        TPipe In, Out;
        bool IoSleepRunning = false;
    };

    static void DoIoSleep(TCont* cont, void* argPtr) noexcept {
        try {
            TState& state = *(TState*) (argPtr);
            state.IoSleepRunning = true;

            TTempBuf tmp;
            // Wait for the event from io
            auto res = NCoro::ReadD(cont, state.In.GetHandle(), tmp.Data(), 1, TDuration::Seconds(10).ToDeadLine());
            UNIT_ASSERT_VALUES_EQUAL(res.Checked(), 0);
            state.IoSleepRunning = false;
        } catch (const NUnitTest::TAssertException& ex) {
            Cerr << ex.AsStrBuf() << Endl;
            ex.BackTrace()->PrintTo(Cerr);
            throw;
        } catch (...) {
            Cerr << CurrentExceptionMessage() << Endl;
            throw;
        }
    }

    static void DoSub(TCont* cont, void* argPtr) noexcept {
        TSubState& state = *(TSubState*)(argPtr);

        TTimerEvent event(cont, TInstant::Max());
        state.Event = &event;
        ExecuteEvent(&event);
        state.Event = nullptr;
        state.Cont = nullptr;
        state.Finish = TInstant::Now();
    }

    static void DoMain(TCont* cont) noexcept {
        try {
            TState state;
            TInstant start = TInstant::Now();

            // This guy sleeps on io
            auto sleeper = cont->Executor()->Create(DoIoSleep, &state, "io_sleeper");

            // These guys are to be woken up right away
            for (auto& subState : state.Subs) {
                subState.Cont = cont->Executor()->Create(DoSub, &subState, subState.Name.data());
            }

            // Give way
            cont->Yield();

            // Check everyone has started, wake those to be woken
            UNIT_ASSERT(state.IoSleepRunning);

            for (auto& subState : state.Subs) {
                UNIT_ASSERT(subState.Event);
                subState.Event->Wake(EWAKEDUP);
            }

            // Give way again
            cont->Yield();

            // Check the woken guys have finished and quite soon
            for (auto& subState : state.Subs) {
                UNIT_ASSERT(subState.Finish - start < TDuration::MilliSeconds(100));
                UNIT_ASSERT(!subState.Cont);
            }

            // Wake the io guy and finish
            state.Out.Close();

            if (state.IoSleepRunning) {
                cont->Join(sleeper);
            }

            // Check everything has ended sooner than the timeout
            UNIT_ASSERT(TInstant::Now() - start < TDuration::Seconds(1));
        } catch (const NUnitTest::TAssertException& ex) {
            Cerr << ex.AsStrBuf() << Endl;
            ex.BackTrace()->PrintTo(Cerr);
            throw;
        } catch (...) {
            Cerr << CurrentExceptionMessage() << Endl;
            throw;
        }
    }

    static void DoTestFastPathWake(EContPoller pollerType) {
        if (auto poller = IPollerFace::Construct(pollerType)) {
            TContExecutor exec(20000, std::move(poller));
            exec.SetFailOnError(true);
            exec.Execute(NCoroTestFastPathWake::DoMain);
        }
    }
}

void TCoroTest::TestFastPathWakeDefault() {
    NCoroTestFastPathWake::DoTestFastPathWake(EContPoller::Default);
}

void TCoroTest::TestFastPathWakeEpoll() {
    NCoroTestFastPathWake::DoTestFastPathWake(EContPoller::Epoll);
}

void TCoroTest::TestFastPathWakeKqueue() {
    NCoroTestFastPathWake::DoTestFastPathWake(EContPoller::Kqueue);
}

void TCoroTest::TestFastPathWakePoll() {
    NCoroTestFastPathWake::DoTestFastPathWake(EContPoller::Poll);
}

void TCoroTest::TestFastPathWakeSelect() {
    NCoroTestFastPathWake::DoTestFastPathWake(EContPoller::Select);
}

namespace NCoroTestLegacyCancelYieldRaceBug {
    enum class EState {
        Idle, Running, Finished,
    };

    struct TState {
        EState SubState = EState::Idle;
    };

    static void DoSub(TCont* cont, void* argPtr) {
        TState& state = *(TState*)argPtr;
        state.SubState = EState::Running;
        cont->Yield();
        cont->Yield();
        state.SubState = EState::Finished;
    }

    static void DoMain(TCont* cont, void* argPtr) {
        TState& state = *(TState*)argPtr;
        TCont* sub =  cont->Executor()->Create(DoSub, argPtr, "Sub");
        sub->Cancel();
        cont->Yield();
        UNIT_ASSERT_EQUAL(state.SubState, EState::Finished);
    }
}

void TCoroTest::TestLegacyCancelYieldRaceBug() {
    NCoroTestLegacyCancelYieldRaceBug::TState state;
    TContExecutor exec(20000);
    exec.SetFailOnError(true);
    exec.Execute(NCoroTestLegacyCancelYieldRaceBug::DoMain, &state);
}

namespace NCoroTestJoinRescheduleBug {
    enum class EState {
        Idle, Running, Finished,
    };

    struct TState {
        TCont* volatile SubA = nullptr;
        volatile EState SubAState = EState::Idle;
        volatile EState SubBState = EState::Idle;
        volatile EState SubCState = EState::Idle;
    };

    static void DoSubC(TCont* cont, void* argPtr) {
        TState& state = *(TState*)argPtr;
        state.SubCState = EState::Running;
        while (state.SubBState != EState::Running) {
            cont->Yield();
        }
        while (cont->SleepD(TInstant::Max()) != ECANCELED) {
        }
        state.SubCState = EState::Finished;
    }

    static void DoSubB(TCont* cont, void* argPtr) {
        TState& state = *(TState*)argPtr;
        state.SubBState = EState::Running;
        while (state.SubAState != EState::Running && state.SubCState != EState::Running) {
            cont->Yield();
        }
        for (auto i : xrange(100)) {
            Y_UNUSED(i);
            if (!state.SubA) {
                break;
            }
            state.SubA->ReSchedule();
            cont->Yield();
        }
        state.SubBState = EState::Finished;
    }

    static void DoSubA(TCont* cont, void* argPtr) {
        TState& state = *(TState*)argPtr;
        state.SubAState = EState::Running;
        TCont* subC = cont->Executor()->Create(DoSubC, argPtr, "SubC");
        while (state.SubBState != EState::Running && state.SubCState != EState::Running) {
            cont->Yield();
        }
        cont->Join(subC);
        UNIT_ASSERT_EQUAL(state.SubCState, EState::Finished);
        state.SubA = nullptr;
        state.SubAState = EState::Finished;
    }

    static void DoMain(TCont* cont, void* argPtr) {
        TState& state = *(TState*)argPtr;
        TCont* subA = cont->Executor()->Create(DoSubA, argPtr, "SubA");
        state.SubA = subA;
        cont->Join(cont->Executor()->Create(DoSubB, argPtr, "SubB"));

        if (state.SubA) {
            subA->Cancel();
            cont->Join(subA);
        }
    }
}

void TCoroTest::TestJoinRescheduleBug() {
    using namespace NCoroTestJoinRescheduleBug;
    TState state;
    {
        TContExecutor exec(20000);
        exec.Execute(DoMain, &state);
    }
    UNIT_ASSERT_EQUAL(state.SubAState, EState::Finished);
    UNIT_ASSERT_EQUAL(state.SubBState, EState::Finished);
    UNIT_ASSERT_EQUAL(state.SubCState, EState::Finished);
}

void TCoroTest::TestEventQueue() {
    NCoro::TEventWaitQueue queue;
    UNIT_ASSERT(queue.Empty());
    UNIT_ASSERT_VALUES_EQUAL(queue.WakeTimedout(TInstant()), TInstant());
    TContExecutor exec(32000);
    exec.Execute([](TCont* cont, void* arg) {
        NCoro::TEventWaitQueue* q = (NCoro::TEventWaitQueue*)arg;
        TTimerEvent ev(cont, TInstant::Max());
        TTimerEvent ev2(cont, TInstant::Seconds(12345));
        q->Register(&ev);
        UNIT_ASSERT_VALUES_EQUAL(q->WakeTimedout(TInstant::Seconds(12344)), TInstant::Max());
        UNIT_ASSERT_VALUES_EQUAL(q->WakeTimedout(TInstant::Seconds(12344)), TInstant::Max());
        q->Register(&ev2);
        UNIT_ASSERT_VALUES_EQUAL(q->WakeTimedout(TInstant::Seconds(12344)), TInstant::Seconds(12345));
        UNIT_ASSERT_VALUES_EQUAL(q->WakeTimedout(TInstant::Seconds(12344)), TInstant::Seconds(12345));
        UNIT_ASSERT_VALUES_EQUAL(q->WakeTimedout(TInstant::Seconds(12345)), TInstant::Seconds(12345));
        UNIT_ASSERT_VALUES_EQUAL(q->WakeTimedout(TInstant::Seconds(12345)), TInstant::Max());
    }, &queue);
}

void TCoroTest::TestNestedExecutor() {
#ifndef _win_
    //nested executors actually don't work correctly, but anyway shouldn't break RunningCont() ptr
    TContExecutor exec(32000);
    UNIT_ASSERT(!RunningCont());

    exec.Execute([](TCont* cont, void*) {
        UNIT_ASSERT_VALUES_EQUAL(RunningCont(), cont);

        TContExecutor exec2(32000);
        exec2.Execute([](TCont* cont2, void*) {
            UNIT_ASSERT_VALUES_EQUAL(RunningCont(), cont2);
            TContExecutor exec3(32000);
            exec3.Execute([](TCont* cont3, void*) {
                UNIT_ASSERT_VALUES_EQUAL(RunningCont(), cont3);
            });

            UNIT_ASSERT_VALUES_EQUAL(RunningCont(), cont2);
        });

        UNIT_ASSERT_VALUES_EQUAL(RunningCont(), cont);
    });

    UNIT_ASSERT(!RunningCont());
#endif
}

void TCoroTest::TestComputeCoroutineYield() {
//if we have busy (e.g., on cpu) coroutine, when it yields, io must flow
    TContExecutor exec(32000);
    exec.SetFailOnError(true);

    TPipe in, out;
    TPipe::Pipe(in, out);
    SetNonBlock(in.GetHandle());
    size_t lastRead = 42;

    auto compute = [&](TCont* cont) {
        for (size_t i = 0; i < 10; ++i) {
            write(out.GetHandle(), &i, sizeof i);
            Sleep(TDuration::MilliSeconds(10));
            cont->Yield();
            UNIT_ASSERT(lastRead == i);
        }
    };

    auto io = [&](TCont* cont) {
        for (size_t i = 0; i < 10; ++i) {
            NCoro::ReadI(cont, in.GetHandle(), &lastRead, sizeof lastRead);
        }
    };

    exec.Create(compute, "compute");
    exec.Create(io, "io");

    exec.Execute();
}

void TCoroTest::TestPollEngines() {
    bool defaultChecked = false;
    for (auto engine : GetEnumAllValues<EContPoller>()) {
        auto poller = IPollerFace::Construct(engine);
        if (!poller) {
            continue;
        }

        TContExecutor exec(32000, IPollerFace::Construct(engine));

        if (engine == EContPoller::Default) {
            defaultChecked = true;
#if defined(HAVE_EPOLL_POLLER)
            UNIT_ASSERT_VALUES_EQUAL(exec.Poller()->PollEngine(), EContPoller::Epoll);
#elif defined(HAVE_KQUEUE_POLLER)
            UNIT_ASSERT_VALUES_EQUAL(exec.Poller()->PollEngine(), EContPoller::Kqueue);
#else
            UNIT_ASSERT_VALUES_EQUAL(exec.Poller()->PollEngine(), EContPoller::Select);
#endif
        } else {
            UNIT_ASSERT_VALUES_EQUAL(exec.Poller()->PollEngine(), engine);
        }
    }

    UNIT_ASSERT(defaultChecked);
}

void TCoroTest::TestPause() {
    TContExecutor executor{1024*1024, IPollerFace::Default(), nullptr, nullptr, NCoro::NStack::EGuard::Canary, Nothing()};

    int i = 0;
    executor.CreateOwned([&](TCont*) {
        i++;
        executor.Pause();
        i++;
    }, "coro");

    UNIT_ASSERT_EQUAL(i, 0);
    executor.Execute();
    UNIT_ASSERT_EQUAL(i, 1);
    executor.Execute();
    UNIT_ASSERT_EQUAL(i, 2);
}

void TCoroTest::TestUserEvent() {
    TContExecutor exec(32000);

    struct TUserEvent : public IUserEvent {
        bool Called = false;
        void Execute() override {
            Called = true;
        }
    } event;

    auto f = [&](TCont* cont) {
        UNIT_ASSERT(!event.Called);
        exec.ScheduleUserEvent(&event);
        UNIT_ASSERT(!event.Called);
        cont->Yield();
        UNIT_ASSERT(event.Called);
    };

    exec.Execute(f);

    UNIT_ASSERT(event.Called);
}

void TCoroTest::TestOverrideTime() {
    class TTime: public NCoro::ITime {
      public:
        TInstant Now() override {
            return Current;
        }

        TInstant Current = TInstant::Zero();
    };

    TTime time;
    TContExecutor executor{1024*1024, IPollerFace::Default(), nullptr, nullptr, NCoro::NStack::EGuard::Canary, Nothing(), &time};

    executor.CreateOwned([&](TCont* cont) {
        UNIT_ASSERT_EQUAL(cont->Executor()->Now(), TInstant::Zero());
        time.Current = TInstant::Seconds(1);
        cont->SleepD(TInstant::Seconds(1));
        UNIT_ASSERT_EQUAL(cont->Executor()->Now(), TInstant::Seconds(1));
    }, "coro");

    executor.Execute();
}

void TCoroTest::TestCancelWithException() {
    TContExecutor exec(32000);

    TString excText = "test exception";
    THolder<std::exception> excep = MakeHolder<yexception>(yexception() << excText);
    std::exception* excPtr = excep.Get();

    exec.CreateOwned([&](TCont* cont){
        TCont *cont1 = cont->Executor()->CreateOwned([&](TCont* c) {
            int result = c->SleepD(TDuration::MilliSeconds(200).ToDeadLine());
            UNIT_ASSERT_EQUAL(result, ECANCELED);
            UNIT_ASSERT_EQUAL(c->Cancelled(), true);
            THolder<std::exception> exc = c->TakeException();
            UNIT_ASSERT_EQUAL(exc.Get(), excPtr);
            UNIT_ASSERT_EQUAL(exc->what(), excText);
            UNIT_ASSERT(dynamic_cast<yexception*>(exc.Get()) != nullptr);
        }, "cancelExc");
        cont1->Cancel(std::move(excep));

        TCont* cont2 = cont->Executor()->CreateOwned([&](TCont* c) {
            int result = c->SleepD(TDuration::MilliSeconds(200).ToDeadLine());
            UNIT_ASSERT_EQUAL(result, ECANCELED);
            UNIT_ASSERT_EQUAL(c->Cancelled(), true);
            THolder<std::exception> exc = c->TakeException();
            UNIT_ASSERT_EQUAL(exc.Get(), nullptr);
        }, "cancelTwice");
        cont2->Cancel();
        THolder<std::exception> e = MakeHolder<yexception>(yexception() << "another exception");
        cont2->Cancel(std::move(e));
    }, "coro");

    exec.Execute();
}
UNIT_TEST_SUITE_REGISTRATION(TCoroTest);
