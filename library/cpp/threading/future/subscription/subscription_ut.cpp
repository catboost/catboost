#include "subscription.h"

#include <library/cpp/testing/unittest/registar.h>

using namespace NThreading;

Y_UNIT_TEST_SUITE(TSubscriptionManagerTest) {

    Y_UNIT_TEST(TestSubscribeUnsignaled) {
        auto m = TSubscriptionManager::NewInstance();
        auto p = NewPromise();

        size_t callCount = 0;
        auto id = m->Subscribe(p.GetFuture(), [&callCount](auto&&) { ++callCount; } );
        UNIT_ASSERT(id.has_value());
        UNIT_ASSERT_EQUAL(callCount, 0);

        p.SetValue();
        UNIT_ASSERT_EQUAL(callCount, 1);
    }

    Y_UNIT_TEST(TestSubscribeSignaled) {
        auto m = TSubscriptionManager::NewInstance();
        auto f = MakeFuture();

        size_t callCount = 0;
        auto id = m->Subscribe(f, [&callCount](auto&&) { ++callCount; } );
        UNIT_ASSERT(!id.has_value());
        UNIT_ASSERT_EQUAL(callCount, 1);
    }

    Y_UNIT_TEST(TestSubscribeUnsignaledAndSignaled) {
        auto m = TSubscriptionManager::NewInstance();
        auto p = NewPromise();

        size_t callCount1 = 0;
        auto id1 = m->Subscribe(p.GetFuture(), [&callCount1](auto&&) { ++callCount1; } );
        UNIT_ASSERT(id1.has_value());
        UNIT_ASSERT_EQUAL(callCount1, 0);

        p.SetValue();
        UNIT_ASSERT_EQUAL(callCount1, 1);

        size_t callCount2 = 0;
        auto id2 = m->Subscribe(p.GetFuture(), [&callCount2](auto&&) { ++callCount2; } );
        UNIT_ASSERT(!id2.has_value());
        UNIT_ASSERT_EQUAL(callCount2, 1);
        UNIT_ASSERT_EQUAL(callCount1, 1);
    }

    Y_UNIT_TEST(TestSubscribeUnsubscribeUnsignaled) {
        auto m = TSubscriptionManager::NewInstance();
        auto p = NewPromise();

        size_t callCount = 0;
        auto id = m->Subscribe(p.GetFuture(), [&callCount](auto&&) { ++callCount; } );
        UNIT_ASSERT(id.has_value());
        UNIT_ASSERT_EQUAL(callCount, 0);

        m->Unsubscribe(id.value());

        p.SetValue();
        UNIT_ASSERT_EQUAL(callCount, 0);
    }

    Y_UNIT_TEST(TestSubscribeUnsignaledUnsubscribeSignaled) {
        auto m = TSubscriptionManager::NewInstance();
        auto p = NewPromise();

        size_t callCount = 0;
        auto id = m->Subscribe(p.GetFuture(), [&callCount](auto&&) { ++callCount; } );
        UNIT_ASSERT(id.has_value());
        UNIT_ASSERT_EQUAL(callCount, 0);

        p.SetValue();
        UNIT_ASSERT_EQUAL(callCount, 1);

        m->Unsubscribe(id.value());
        UNIT_ASSERT_EQUAL(callCount, 1);
    }

    Y_UNIT_TEST(TestUnsubscribeTwice) {
        auto m = TSubscriptionManager::NewInstance();
        auto p = NewPromise();

        size_t callCount = 0;
        auto id = m->Subscribe(p.GetFuture(), [&callCount](auto&&) { ++callCount; } );
        UNIT_ASSERT(id.has_value());
        UNIT_ASSERT_EQUAL(callCount, 0);

        m->Unsubscribe(id.value());
        UNIT_ASSERT_EQUAL(callCount, 0);
        m->Unsubscribe(id.value());
        UNIT_ASSERT_EQUAL(callCount, 0);

        p.SetValue();
        UNIT_ASSERT_EQUAL(callCount, 0);
    }

    Y_UNIT_TEST(TestSubscribeOneUnsignaledManyTimes) {
        auto m = TSubscriptionManager::NewInstance();
        auto p = NewPromise();

        size_t callCount1 = 0;
        auto id1 = m->Subscribe(p.GetFuture(), [&callCount1](auto&&) { ++callCount1; } );
        size_t callCount2 = 0;
        auto id2 = m->Subscribe(p.GetFuture(), [&callCount2](auto&&) { ++callCount2; } );
        size_t callCount3 = 0;
        auto id3 = m->Subscribe(p.GetFuture(), [&callCount3](auto&&) { ++callCount3; } );

        UNIT_ASSERT(id1.has_value());
        UNIT_ASSERT(id2.has_value());
        UNIT_ASSERT(id3.has_value());
        UNIT_ASSERT_UNEQUAL(id1.value(), id2.value());
        UNIT_ASSERT_UNEQUAL(id2.value(), id3.value());
        UNIT_ASSERT_UNEQUAL(id3.value(), id1.value());
        UNIT_ASSERT_EQUAL(callCount1, 0);
        UNIT_ASSERT_EQUAL(callCount2, 0);
        UNIT_ASSERT_EQUAL(callCount3, 0);

        p.SetValue();
        UNIT_ASSERT_EQUAL(callCount1, 1);
        UNIT_ASSERT_EQUAL(callCount2, 1);
        UNIT_ASSERT_EQUAL(callCount3, 1);
    }

    Y_UNIT_TEST(TestSubscribeOneSignaledManyTimes) {
        auto m = TSubscriptionManager::NewInstance();
        auto f = MakeFuture();

        size_t callCount1 = 0;
        auto id1 = m->Subscribe(f, [&callCount1](auto&&) { ++callCount1; } );
        size_t callCount2 = 0;
        auto id2 = m->Subscribe(f, [&callCount2](auto&&) { ++callCount2; } );
        size_t callCount3 = 0;
        auto id3 = m->Subscribe(f, [&callCount3](auto&&) { ++callCount3; } );

        UNIT_ASSERT(!id1.has_value());
        UNIT_ASSERT(!id2.has_value());
        UNIT_ASSERT(!id3.has_value());
        UNIT_ASSERT_EQUAL(callCount1, 1);
        UNIT_ASSERT_EQUAL(callCount2, 1);
        UNIT_ASSERT_EQUAL(callCount3, 1);
    }

    Y_UNIT_TEST(TestSubscribeUnsubscribeOneUnsignaledManyTimes) {
        auto m = TSubscriptionManager::NewInstance();
        auto p = NewPromise();

        size_t callCount1 = 0;
        auto id1 = m->Subscribe(p.GetFuture(), [&callCount1](auto&&) { ++callCount1; } );
        size_t callCount2 = 0;
        auto id2 = m->Subscribe(p.GetFuture(), [&callCount2](auto&&) { ++callCount2; } );
        size_t callCount3 = 0;
        auto id3 = m->Subscribe(p.GetFuture(), [&callCount3](auto&&) { ++callCount3; } );
        size_t callCount4 = 0;
        auto id4 = m->Subscribe(p.GetFuture(), [&callCount4](auto&&) { ++callCount4; } );

        UNIT_ASSERT(id1.has_value());
        UNIT_ASSERT(id2.has_value());
        UNIT_ASSERT(id3.has_value());
        UNIT_ASSERT(id4.has_value());
        UNIT_ASSERT_EQUAL(callCount1, 0);
        UNIT_ASSERT_EQUAL(callCount2, 0);
        UNIT_ASSERT_EQUAL(callCount3, 0);
        UNIT_ASSERT_EQUAL(callCount4, 0);

        m->Unsubscribe(id3.value());
        m->Unsubscribe(id1.value());
        UNIT_ASSERT_EQUAL(callCount1, 0);
        UNIT_ASSERT_EQUAL(callCount2, 0);
        UNIT_ASSERT_EQUAL(callCount3, 0);
        UNIT_ASSERT_EQUAL(callCount4, 0);

        p.SetValue();
        UNIT_ASSERT_EQUAL(callCount1, 0);
        UNIT_ASSERT_EQUAL(callCount2, 1);
        UNIT_ASSERT_EQUAL(callCount3, 0);
        UNIT_ASSERT_EQUAL(callCount4, 1);
    }

    Y_UNIT_TEST(TestSubscribeManyUnsignaled) {
        auto m = TSubscriptionManager::NewInstance();
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();

        size_t callCount1 = 0;
        auto id1 = m->Subscribe(p1.GetFuture(), [&callCount1](auto&&) { ++callCount1; } );
        size_t callCount2 = 0;
        auto id2 = m->Subscribe(p2.GetFuture(), [&callCount2](auto&&) { ++callCount2; } );
        size_t callCount3 = 0;
        auto id3 = m->Subscribe(p1.GetFuture(), [&callCount3](auto&&) { ++callCount3; } );

        UNIT_ASSERT(id1.has_value());
        UNIT_ASSERT(id2.has_value());
        UNIT_ASSERT(id3.has_value());
        UNIT_ASSERT_UNEQUAL(id1.value(), id2.value());
        UNIT_ASSERT_UNEQUAL(id2.value(), id3.value());
        UNIT_ASSERT_UNEQUAL(id3.value(), id1.value());
        UNIT_ASSERT_EQUAL(callCount1, 0);
        UNIT_ASSERT_EQUAL(callCount2, 0);
        UNIT_ASSERT_EQUAL(callCount3, 0);

        p1.SetValue(33);
        UNIT_ASSERT_EQUAL(callCount1, 1);
        UNIT_ASSERT_EQUAL(callCount2, 0);
        UNIT_ASSERT_EQUAL(callCount3, 1);

        p2.SetValue(111);
        UNIT_ASSERT_EQUAL(callCount1, 1);
        UNIT_ASSERT_EQUAL(callCount2, 1);
        UNIT_ASSERT_EQUAL(callCount3, 1);
    }

    Y_UNIT_TEST(TestSubscribeManySignaled) {
        auto m = TSubscriptionManager::NewInstance();
        auto f1 = MakeFuture(0);
        auto f2 = MakeFuture(1);

        size_t callCount1 = 0;
        auto id1 = m->Subscribe(f1, [&callCount1](auto&&) { ++callCount1; } );
        size_t callCount2 = 0;
        auto id2 = m->Subscribe(f2, [&callCount2](auto&&) { ++callCount2; } );
        size_t callCount3 = 0;
        auto id3 = m->Subscribe(f2, [&callCount3](auto&&) { ++callCount3; } );

        UNIT_ASSERT(!id1.has_value());
        UNIT_ASSERT(!id2.has_value());
        UNIT_ASSERT(!id3.has_value());
        UNIT_ASSERT_EQUAL(callCount1, 1);
        UNIT_ASSERT_EQUAL(callCount2, 1);
        UNIT_ASSERT_EQUAL(callCount3, 1);
    }

    Y_UNIT_TEST(TestSubscribeManyMixed) {
        auto m = TSubscriptionManager::NewInstance();
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto f = MakeFuture(42);

        size_t callCount1 = 0;
        auto id1 = m->Subscribe(p1.GetFuture(), [&callCount1](auto&&) { ++callCount1; } );
        size_t callCount2 = 0;
        auto id2 = m->Subscribe(p2.GetFuture(), [&callCount2](auto&&) { ++callCount2; } );
        size_t callCount3 = 0;
        auto id3 = m->Subscribe(f, [&callCount3](auto&&) { ++callCount3; } );

        UNIT_ASSERT(id1.has_value());
        UNIT_ASSERT(id2.has_value());
        UNIT_ASSERT(!id3.has_value());
        UNIT_ASSERT_UNEQUAL(id1.value(), id2.value());
        UNIT_ASSERT_EQUAL(callCount1, 0);
        UNIT_ASSERT_EQUAL(callCount2, 0);
        UNIT_ASSERT_EQUAL(callCount3, 1);

        p1.SetValue(45);
        UNIT_ASSERT_EQUAL(callCount1, 1);
        UNIT_ASSERT_EQUAL(callCount2, 0);
        UNIT_ASSERT_EQUAL(callCount3, 1);

        p2.SetValue(-7);
        UNIT_ASSERT_EQUAL(callCount1, 1);
        UNIT_ASSERT_EQUAL(callCount2, 1);
        UNIT_ASSERT_EQUAL(callCount3, 1);
    }

    Y_UNIT_TEST(TestSubscribeUnsubscribeMany) {
        auto m = TSubscriptionManager::NewInstance();
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto p3 = NewPromise<int>();

        size_t callCount1 = 0;
        auto id1 = m->Subscribe(p1.GetFuture(), [&callCount1](auto&&) { ++callCount1; } );
        size_t callCount2 = 0;
        auto id2 = m->Subscribe(p2.GetFuture(), [&callCount2](auto&&) { ++callCount2; } );
        size_t callCount3 = 0;
        auto id3 = m->Subscribe(p3.GetFuture(), [&callCount3](auto&&) { ++callCount3; } );
        size_t callCount4 = 0;
        auto id4 = m->Subscribe(p2.GetFuture(), [&callCount4](auto&&) { ++callCount4; } );
        size_t callCount5 = 0;
        auto id5 = m->Subscribe(p1.GetFuture(), [&callCount5](auto&&) { ++callCount5; } );

        UNIT_ASSERT(id1.has_value());
        UNIT_ASSERT(id2.has_value());
        UNIT_ASSERT(id3.has_value());
        UNIT_ASSERT(id4.has_value());
        UNIT_ASSERT(id5.has_value());
        UNIT_ASSERT_EQUAL(callCount1, 0);
        UNIT_ASSERT_EQUAL(callCount2, 0);
        UNIT_ASSERT_EQUAL(callCount3, 0);
        UNIT_ASSERT_EQUAL(callCount4, 0);
        UNIT_ASSERT_EQUAL(callCount5, 0);

        m->Unsubscribe(id1.value());
        p1.SetValue(-1);
        UNIT_ASSERT_EQUAL(callCount1, 0);
        UNIT_ASSERT_EQUAL(callCount2, 0);
        UNIT_ASSERT_EQUAL(callCount3, 0);
        UNIT_ASSERT_EQUAL(callCount4, 0);
        UNIT_ASSERT_EQUAL(callCount5, 1);

        m->Unsubscribe(id4.value());
        p2.SetValue(23);
        UNIT_ASSERT_EQUAL(callCount1, 0);
        UNIT_ASSERT_EQUAL(callCount2, 1);
        UNIT_ASSERT_EQUAL(callCount3, 0);
        UNIT_ASSERT_EQUAL(callCount4, 0);
        UNIT_ASSERT_EQUAL(callCount5, 1);

        p3.SetValue(100500);
        UNIT_ASSERT_EQUAL(callCount1, 0);
        UNIT_ASSERT_EQUAL(callCount2, 1);
        UNIT_ASSERT_EQUAL(callCount3, 1);
        UNIT_ASSERT_EQUAL(callCount4, 0);
        UNIT_ASSERT_EQUAL(callCount5, 1);
    }

    Y_UNIT_TEST(TestBulkSubscribeManyUnsignaled) {
        auto m = TSubscriptionManager::NewInstance();
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();

        size_t callCount = 0;
        auto ids = m->Subscribe({ p1.GetFuture(), p2.GetFuture(), p1.GetFuture() }, [&callCount](auto&&) { ++callCount; });

        UNIT_ASSERT_EQUAL(ids.size(), 3);
        UNIT_ASSERT_UNEQUAL(ids[0], ids[1]);
        UNIT_ASSERT_UNEQUAL(ids[1], ids[2]);
        UNIT_ASSERT_UNEQUAL(ids[2], ids[0]);
        UNIT_ASSERT_EQUAL(callCount, 0);

        p1.SetValue(33);
        UNIT_ASSERT_EQUAL(callCount, 2);

        p2.SetValue(111);
        UNIT_ASSERT_EQUAL(callCount, 3);
    }

    Y_UNIT_TEST(TestBulkSubscribeManySignaledNoRevert) {
        auto m = TSubscriptionManager::NewInstance();
        auto f1 = MakeFuture(0);
        auto f2 = MakeFuture(1);

        size_t callCount = 0;
        auto ids = m->Subscribe({ f1, f2, f1 }, [&callCount](auto&&) { ++callCount; });

        UNIT_ASSERT_EQUAL(ids.size(), 3);
        UNIT_ASSERT_UNEQUAL(ids[0], ids[1]);
        UNIT_ASSERT_UNEQUAL(ids[1], ids[2]);
        UNIT_ASSERT_UNEQUAL(ids[2], ids[0]);
        UNIT_ASSERT_EQUAL(callCount, 3);
    }

    Y_UNIT_TEST(TestBulkSubscribeManySignaledRevert) {
        auto m = TSubscriptionManager::NewInstance();
        auto f1 = MakeFuture(0);
        auto f2 = MakeFuture(1);

        size_t callCount = 0;
        auto ids = m->Subscribe({ f1, f2, f1 }, [&callCount](auto&&) { ++callCount; }, true);

        UNIT_ASSERT(ids.empty());
        UNIT_ASSERT_EQUAL(callCount, 1);
    }

    Y_UNIT_TEST(TestBulkSubscribeManyMixedNoRevert) {
        auto m = TSubscriptionManager::NewInstance();
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto f = MakeFuture(42);

        size_t callCount = 0;
        auto ids = m->Subscribe({ p1.GetFuture(), p2.GetFuture(), f }, [&callCount](auto&&) { ++callCount; } );

        UNIT_ASSERT_EQUAL(ids.size(), 3);
        UNIT_ASSERT_UNEQUAL(ids[0], ids[1]);
        UNIT_ASSERT_UNEQUAL(ids[1], ids[2]);
        UNIT_ASSERT_UNEQUAL(ids[2], ids[0]);
        UNIT_ASSERT_EQUAL(callCount, 1);

        p1.SetValue(45);
        UNIT_ASSERT_EQUAL(callCount, 2);

        p2.SetValue(-7);
        UNIT_ASSERT_EQUAL(callCount, 3);
    }

    Y_UNIT_TEST(TestBulkSubscribeManyMixedRevert) {
        auto m = TSubscriptionManager::NewInstance();
        auto p1 = NewPromise();
        auto p2 = NewPromise();
        auto f = MakeFuture();

        size_t callCount = 0;
        auto ids = m->Subscribe({ p1.GetFuture(), f, p2.GetFuture() }, [&callCount](auto&&) { ++callCount; }, true);

        UNIT_ASSERT(ids.empty());
        UNIT_ASSERT_EQUAL(callCount, 1);

        p1.SetValue();
        p2.SetValue();
        UNIT_ASSERT_EQUAL(callCount, 1);
    }

    Y_UNIT_TEST(TestBulkSubscribeUnsubscribeMany) {
        auto m = TSubscriptionManager::NewInstance();
        auto p1 = NewPromise<int>();
        auto p2 = NewPromise<int>();
        auto p3 = NewPromise<int>();

        size_t callCount = 0;
        auto ids = m->Subscribe(
                        TVector<TFuture<int>>{ p1.GetFuture(), p2.GetFuture(), p3.GetFuture(), p2.GetFuture(), p1.GetFuture() }
                        , [&callCount](auto&&) { ++callCount; } );

        UNIT_ASSERT_EQUAL(ids.size(), 5);
        UNIT_ASSERT_EQUAL(callCount, 0);

        m->Unsubscribe(TVector<TSubscriptionId>{ ids[0], ids[3] });
        UNIT_ASSERT_EQUAL(callCount, 0);

        p1.SetValue(-1);
        UNIT_ASSERT_EQUAL(callCount, 1);

        p2.SetValue(23);
        UNIT_ASSERT_EQUAL(callCount, 2);

        p3.SetValue(100500);
        UNIT_ASSERT_EQUAL(callCount, 3);
    }
}
