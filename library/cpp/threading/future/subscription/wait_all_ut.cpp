#include "wait_all.h"
#include "wait_ut_common.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/generic/strbuf.h>

#include <atomic>
#include <exception>

using namespace NThreading;

Y_UNIT_TEST_SUITE(TWaitAllTest) {

    Y_UNIT_TEST(TestTwoUnsignaled) {
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto w = NWait::WaitAll(p1.GetFuture(), p2.GetFuture());
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        p1.SetValue(10);
        UNIT_ASSERT(!w.HasValue() && !w.HasException());
        p2.SetValue(1);
        UNIT_ASSERT(w.HasValue());
    }

    Y_UNIT_TEST(TestTwoUnsignaledWithException) {
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto w = NWait::WaitAll(p1.GetFuture(), p2.GetFuture());
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        constexpr TStringBuf message = "Test exception";
        p1.SetException(std::make_exception_ptr(yexception() << message));
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        p2.SetValue(-11);
        UNIT_ASSERT_EXCEPTION_SATISFIES(w.TryRethrow(), yexception, [message](auto const& e) {
            return message == e.what();
        });
    }

    Y_UNIT_TEST(TestOneUnsignaledOneSignaled) {
        auto p = NewPromise();
        auto f = MakeFuture();
        auto w = NWait::WaitAll(p.GetFuture(), f);
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        p.SetValue();
        UNIT_ASSERT(w.HasValue());
    }

    Y_UNIT_TEST(TestOneUnsignaledOneSignaledWithException) {
        auto p = NewPromise();
        auto f = MakeFuture();
        auto w = NWait::WaitAll(f, p.GetFuture());
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        constexpr TStringBuf message = "Test exception 2";
        p.SetException(std::make_exception_ptr(yexception() << message));
        UNIT_ASSERT_EXCEPTION_SATISFIES(w.TryRethrow(), yexception, [message](auto const& e) {
            return message == e.what();
        });
    }

    Y_UNIT_TEST(TestEmptyInitializer) {
        auto w = NWait::WaitAll(std::initializer_list<TFuture<void> const>({}));
        UNIT_ASSERT(w.HasValue());
    }

    Y_UNIT_TEST(TestEmptyVector) {
        auto w = NWait::WaitAll(TVector<TFuture<int>>());
        UNIT_ASSERT(w.HasValue());
    }

    Y_UNIT_TEST(TestOneUnsignaledWithInitializer) {
        auto p = NewPromise<int>();
        auto w = NWait::WaitAll({ p.GetFuture() });
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        p.SetValue(1);
        UNIT_ASSERT(w.HasValue());
    }

    Y_UNIT_TEST(TestOneUnsignaledWithVector) {
        auto p = NewPromise();
        auto w = NWait::WaitAll(TVector<TFuture<void>>{ p.GetFuture() });
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        constexpr TStringBuf message = "Test exception 3";
        p.SetException(std::make_exception_ptr(yexception() << message));
        UNIT_ASSERT_EXCEPTION_SATISFIES(w.TryRethrow(), yexception, [message](auto const& e) {
            return message == e.what();
        });
    }

    Y_UNIT_TEST(TestManyWithInitializer) {
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto f = MakeFuture(42);
        auto w = NWait::WaitAll({ p1.GetFuture(), f, p2.GetFuture() });
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        p1.SetValue(10);
        UNIT_ASSERT(!w.HasValue() && !w.HasException());
        p2.SetValue(-3);
        UNIT_ASSERT(w.HasValue());
    }

    Y_UNIT_TEST(TestManyWithVector) {
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto f = MakeFuture(42);
        auto w = NWait::WaitAll(TVector<TFuture<int>>{ p1.GetFuture(), f, p2.GetFuture() });
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        constexpr TStringBuf message = "Test exception 4";
        p1.SetException(std::make_exception_ptr(yexception() << message));
        UNIT_ASSERT(!w.HasValue() && !w.HasException());

        p2.SetValue(34);
        UNIT_ASSERT_EXCEPTION_SATISFIES(w.TryRethrow(), yexception, [message](auto const& e) {
            return message == e.what();
        });
    }

    Y_UNIT_TEST(TestManyStress) {
        NTest::TestManyStress<int>([](auto&& futures) { return NWait::WaitAll(futures); }
                                    , [](size_t) {
                                        return [](auto&& p) { p.SetValue(42); };
                                    }
                                    , [](auto&& waiter) { UNIT_ASSERT(waiter.HasValue()); });

        NTest::TestManyStress<void>([](auto&& futures) { return NWait::WaitAll(futures); }
                                    , [](size_t) {
                                        return [](auto&& p) { p.SetValue(); };
                                    }
                                    , [](auto&& waiter) { UNIT_ASSERT(waiter.HasValue()); });
        auto e = std::make_exception_ptr(yexception() << "Test exception 5");
        NTest::TestManyStress<void>([](auto&& futures) { return NWait::WaitAll(futures); }
                                    , [e](size_t) {
                                        return [e](auto&& p) { p.SetException(e); };
                                    }
                                    , [](auto&& waiter) { UNIT_ASSERT(waiter.HasException()); });
        e = std::make_exception_ptr(yexception() << "Test exception 6");
        std::atomic<size_t> index = 0;
        NTest::TestManyStress<int>([](auto&& futures) { return NWait::WaitAll(futures); }
                                    , [e, &index](size_t size) {
                                        auto exceptionIndex = size / 2;
                                        index = 0;
                                        return [e, exceptionIndex, &index](auto&& p) {
                                            if (index++ == exceptionIndex) {
                                                p.SetException(e);
                                            } else {
                                                p.SetValue(index);
                                            }
                                        };
                                    }
                                    , [](auto&& waiter) { UNIT_ASSERT(waiter.HasException()); });
    }

}
