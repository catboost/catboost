#include "wait_any.h"
#include "wait_ut_common.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/generic/strbuf.h>

#include <exception>

using namespace NThreading;

Y_UNIT_TEST_SUITE(TWaitAnyTest) {

    Y_UNIT_TEST(TestTwoUnsignaled) {
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto w = NWait::WaitAny(p1.GetFuture(), p2.GetFuture());
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        p1.SetValue(10);
        UNIT_ASSERT(w.HasValue());
        p2.SetValue(1);
    }

    Y_UNIT_TEST(TestTwoUnsignaledWithException) {
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto w = NWait::WaitAny(p1.GetFuture(), p2.GetFuture());
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        constexpr TStringBuf message = "Test exception";
        p2.SetException(std::make_exception_ptr(yexception() << message));
        UNIT_ASSERT_EXCEPTION_SATISFIES(w.TryRethrow(), yexception, [message](auto const& e) {
            return message == e.what();
        });

        p1.SetValue(-11);
    }

    Y_UNIT_TEST(TestOneUnsignaledOneSignaled) {
        auto p = NewPromise();
        auto f = MakeFuture();
        auto w = NWait::WaitAny(p.GetFuture(), f);
        UNIT_ASSERT(w.HasValue());

        p.SetValue();
    }

    Y_UNIT_TEST(TestOneUnsignaledOneSignaledWithException) {
        auto p = NewPromise();
        constexpr TStringBuf message = "Test exception 2";
        auto f = MakeErrorFuture<void>(std::make_exception_ptr(yexception() << message));
        auto w = NWait::WaitAny(f, p.GetFuture());
        UNIT_ASSERT_EXCEPTION_SATISFIES(w.TryRethrow(), yexception, [message](auto const& e) {
            return message == e.what();
        });

        p.SetValue();
    }

    Y_UNIT_TEST(TestEmptyInitializer) {
        auto w = NWait::WaitAny(std::initializer_list<TFuture<void> const>({}));
        UNIT_ASSERT(w.HasValue());
    }

    Y_UNIT_TEST(TestEmptyVector) {
        auto w = NWait::WaitAny(TVector<TFuture<int>>());
        UNIT_ASSERT(w.HasValue());
    }

    Y_UNIT_TEST(TestOneUnsignaledWithInitializer) {
        auto p = NewPromise<int>();
        auto w = NWait::WaitAny({ p.GetFuture() });
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        p.SetValue(1);
        UNIT_ASSERT(w.HasValue());
    }

    Y_UNIT_TEST(TestOneUnsignaledWithVector) {
        auto p = NewPromise();
        auto w = NWait::WaitAny(TVector<TFuture<void>>{ p.GetFuture() });
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        constexpr TStringBuf message = "Test exception 3";
        p.SetException(std::make_exception_ptr(yexception() << message));
        UNIT_ASSERT_EXCEPTION_SATISFIES(w.TryRethrow(), yexception, [message](auto const& e) {
            return message == e.what();
        });
    }

    Y_UNIT_TEST(TestManyUnsignaledWithInitializer) {
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto p3 = NewPromise<int>();
        auto w = NWait::WaitAny({ p1.GetFuture(), p2.GetFuture(), p3.GetFuture() });
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        p1.SetValue(42);
        UNIT_ASSERT(w.HasValue());

        p2.SetValue(-3);
        p3.SetValue(12);
    }

    Y_UNIT_TEST(TestManyMixedWithInitializer) {
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto f = MakeFuture(42);
        auto w = NWait::WaitAny({ p1.GetFuture(), f, p2.GetFuture() });
        UNIT_ASSERT(w.HasValue());

        p1.SetValue(10);
        p2.SetValue(-3);
    }


    Y_UNIT_TEST(TestManyUnsignaledWithVector) {
        auto p1 = NewPromise();
        auto p2 = NewPromise();
        auto p3 = NewPromise();
        auto w = NWait::WaitAny(TVector<TFuture<void>>{ p1.GetFuture(), p2.GetFuture(), p3.GetFuture() });
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        constexpr TStringBuf message = "Test exception 4";
        p2.SetException(std::make_exception_ptr(yexception() << message));
        UNIT_ASSERT_EXCEPTION_SATISFIES(w.TryRethrow(), yexception, [message](auto const& e) {
            return message == e.what();
        });

        p1.SetValue();
        p3.SetValue();
    }


    Y_UNIT_TEST(TestManyMixedWithVector) {
        auto p1 = NewPromise();
        auto p2 = NewPromise();
        auto f = MakeFuture();
        auto w = NWait::WaitAny(TVector<TFuture<void>>{ p1.GetFuture(), p2.GetFuture(), f });
        UNIT_ASSERT(w.HasValue());

        p1.SetValue();
        p2.SetValue();
    }

    Y_UNIT_TEST(TestManyStress) {
        NTest::TestManyStress<void>([](auto&& futures) { return NWait::WaitAny(futures); }
                                    , [](size_t) {
                                        return [](auto&& p) { p.SetValue(); };
                                    }
                                    , [](auto&& waiter) { UNIT_ASSERT(waiter.HasValue()); });

        NTest::TestManyStress<int>([](auto&& futures) { return NWait::WaitAny(futures); }
                                    , [](size_t) {
                                        return [](auto&& p) { p.SetValue(22); };
                                    }
                                    , [](auto&& waiter) { UNIT_ASSERT(waiter.HasValue()); });
        auto e = std::make_exception_ptr(yexception() << "Test exception 5");
        NTest::TestManyStress<void>([](auto&& futures) { return NWait::WaitAny(futures); }
                                    , [e](size_t) {
                                        return [e](auto&& p) { p.SetException(e); };
                                    }
                                    , [](auto&& waiter) { UNIT_ASSERT(waiter.HasException()); });
    }

}
