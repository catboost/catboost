
#include <util/system/atomic.h>
#include <util/system/event.h>
#include <library/threading/future/legacy_future.h>

#include <library/unittest/registar.h>

#include "lfstack.h"

SIMPLE_UNIT_TEST_SUITE(TLockFreeStackTests) {
    class TCountDownLatch {
    private:
        TAtomic Current;
        Event EventObject;

    public:
        TCountDownLatch(unsigned initial)
            : Current(initial)
        {
        }

        void CountDown() {
            if (AtomicDecrement(Current) == 0) {
                EventObject.Signal();
            }
        }

        void Await() {
            EventObject.Wait();
        }

        bool Await(TDuration timeout) {
            return EventObject.WaitT(timeout);
        }
    };

    template <bool SingleConsumer>
    struct TDequeueAllTester {
        size_t EnqueueThreads;
        size_t DequeueThreads;

        size_t EnqueuesPerThread;
        TAtomic LeftToDequeue;

        TCountDownLatch StartLatch;
        TLockFreeStack<int> Stack;

        TDequeueAllTester()
            : EnqueueThreads(4)
            , DequeueThreads(SingleConsumer ? 1 : 3)
            , EnqueuesPerThread(100000)
            , LeftToDequeue(EnqueueThreads * EnqueuesPerThread)
            , StartLatch(EnqueueThreads + DequeueThreads)
        {
        }

        void Enqueuer() {
            StartLatch.CountDown();
            StartLatch.Await();

            for (size_t i = 0; i < EnqueuesPerThread; ++i) {
                Stack.Enqueue(i);
            }
        }

        void DequeuerAll() {
            StartLatch.CountDown();
            StartLatch.Await();

            yvector<int> temp;
            while (AtomicGet(LeftToDequeue) > 0) {
                size_t dequeued = 0;
                for (size_t i = 0; i < 100; ++i) {
                    temp.clear();
                    if (SingleConsumer) {
                        Stack.DequeueAllSingleConsumer(&temp);
                    } else {
                        Stack.DequeueAll(&temp);
                    }
                    dequeued += temp.size();
                }
                AtomicAdd(LeftToDequeue, -dequeued);
            }
        }

        void Run() {
            yvector<TSimpleSharedPtr<NThreading::TLegacyFuture<>>> futures;

            for (size_t i = 0; i < EnqueueThreads; ++i) {
                futures.push_back(new NThreading::TLegacyFuture<>(std::bind(&TDequeueAllTester<SingleConsumer>::Enqueuer, this)));
            }

            for (size_t i = 0; i < DequeueThreads; ++i) {
                futures.push_back(new NThreading::TLegacyFuture<>(std::bind(&TDequeueAllTester<SingleConsumer>::DequeuerAll, this)));
            }

            // effectively join
            futures.clear();

            UNIT_ASSERT_VALUES_EQUAL(0, int(AtomicGet(LeftToDequeue)));

            yvector<int> left;
            Stack.DequeueAll(&left);
            UNIT_ASSERT(left.empty());
        }
    };

    SIMPLE_UNIT_TEST(TestDequeueAll) {
        TDequeueAllTester<false>().Run();
    }

    SIMPLE_UNIT_TEST(TestDequeueAllSingleConsumer) {
        TDequeueAllTester<true>().Run();
    }

    SIMPLE_UNIT_TEST(TestDequeueAllEmptyStack) {
        TLockFreeStack<int> stack;

        yvector<int> r;
        stack.DequeueAll(&r);

        UNIT_ASSERT(r.empty());
    }

    SIMPLE_UNIT_TEST(TestDequeueAllReturnsInReverseOrder) {
        TLockFreeStack<int> stack;

        stack.Enqueue(17);
        stack.Enqueue(19);
        stack.Enqueue(23);

        yvector<int> r;

        stack.DequeueAll(&r);

        UNIT_ASSERT_VALUES_EQUAL(size_t(3), r.size());
        UNIT_ASSERT_VALUES_EQUAL(23, r.at(0));
        UNIT_ASSERT_VALUES_EQUAL(19, r.at(1));
        UNIT_ASSERT_VALUES_EQUAL(17, r.at(2));
    }

    SIMPLE_UNIT_TEST(TestEnqueueAll) {
        TLockFreeStack<int> stack;

        yvector<int> v;
        yvector<int> expected;

        stack.EnqueueAll(v); // add empty

        v.push_back(2);
        v.push_back(3);
        v.push_back(5);
        expected.insert(expected.end(), v.begin(), v.end());
        stack.EnqueueAll(v);

        v.clear();

        stack.EnqueueAll(v); // add empty

        v.push_back(7);
        v.push_back(11);
        v.push_back(13);
        v.push_back(17);
        expected.insert(expected.end(), v.begin(), v.end());
        stack.EnqueueAll(v);

        yvector<int> actual;
        stack.DequeueAll(&actual);

        UNIT_ASSERT_VALUES_EQUAL(expected.size(), actual.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(expected.at(expected.size() - i - 1), actual.at(i));
        }
    }

    SIMPLE_UNIT_TEST(CleanInDestructor) {
        TSimpleSharedPtr<bool> p(new bool);
        UNIT_ASSERT_VALUES_EQUAL(1u, p.RefCount());

        {
            TLockFreeStack<TSimpleSharedPtr<bool>> stack;

            stack.Enqueue(p);
            stack.Enqueue(p);

            UNIT_ASSERT_VALUES_EQUAL(3u, p.RefCount());
        }

        UNIT_ASSERT_VALUES_EQUAL(1, p.RefCount());
    }
}
