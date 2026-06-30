
#include "lfstack.h"

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/threading/future/legacy_future.h>

#include <util/generic/deque.h>
#include <util/system/event.h>

#include <atomic>

Y_UNIT_TEST_SUITE(TLockFreeStackTests) {
    class TCountDownLatch {
    private:
        std::atomic<size_t> Current_;
        TSystemEvent EventObject_;

    public:
        TCountDownLatch(unsigned initial)
            : Current_(initial)
        {
        }

        void CountDown() {
            if (--Current_ == 0) {
                EventObject_.Signal();
            }
        }

        void Await() {
            EventObject_.Wait();
        }

        bool Await(TDuration timeout) {
            return EventObject_.WaitT(timeout);
        }
    };

    template <bool SingleConsumer>
    struct TDequeueAllTester {
        size_t EnqueueThreads;
        size_t DequeueThreads;

        size_t EnqueuesPerThread;
        std::atomic<size_t> LeftToDequeue;

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

            TVector<int> temp;
            while (LeftToDequeue.load() > 0) {
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
                LeftToDequeue -= dequeued;
            }
        }

        void Run() {
            TVector<TSimpleSharedPtr<NThreading::TLegacyFuture<>>> futures;

            for (size_t i = 0; i < EnqueueThreads; ++i) {
                futures.push_back(new NThreading::TLegacyFuture<>(std::bind(&TDequeueAllTester<SingleConsumer>::Enqueuer, this)));
            }

            for (size_t i = 0; i < DequeueThreads; ++i) {
                futures.push_back(new NThreading::TLegacyFuture<>(std::bind(&TDequeueAllTester<SingleConsumer>::DequeuerAll, this)));
            }

            // effectively join
            futures.clear();

            UNIT_ASSERT_VALUES_EQUAL(0, int(LeftToDequeue.load()));

            TVector<int> left;
            Stack.DequeueAll(&left);
            UNIT_ASSERT(left.empty());
        }
    };

    Y_UNIT_TEST(TestDequeueAll) {
        TDequeueAllTester<false>().Run();
    }

    Y_UNIT_TEST(TestDequeueAllSingleConsumer) {
        TDequeueAllTester<true>().Run();
    }

    Y_UNIT_TEST(TestDequeueAllEmptyStack) {
        TLockFreeStack<int> stack;

        TVector<int> r;
        stack.DequeueAll(&r);

        UNIT_ASSERT(r.empty());
    }

    Y_UNIT_TEST(TestDequeueAllReturnsInReverseOrder) {
        TLockFreeStack<int> stack;

        stack.Enqueue(17);
        stack.Enqueue(19);
        stack.Enqueue(23);

        TVector<int> r;

        stack.DequeueAll(&r);

        UNIT_ASSERT_VALUES_EQUAL(size_t(3), r.size());
        UNIT_ASSERT_VALUES_EQUAL(23, r.at(0));
        UNIT_ASSERT_VALUES_EQUAL(19, r.at(1));
        UNIT_ASSERT_VALUES_EQUAL(17, r.at(2));
    }

    Y_UNIT_TEST(TestEnqueueAll) {
        TLockFreeStack<int> stack;

        TVector<int> v;
        TVector<int> expected;

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

        TVector<int> actual;
        stack.DequeueAll(&actual);

        UNIT_ASSERT_VALUES_EQUAL(expected.size(), actual.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(expected.at(expected.size() - i - 1), actual.at(i));
        }
    }

    Y_UNIT_TEST(CleanInDestructor) {
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

    Y_UNIT_TEST(NoCopyTest) {
        static unsigned copied = 0;
        struct TCopyCount {
            TCopyCount(int) {
            }
            TCopyCount(const TCopyCount&) {
                ++copied;
            }

            TCopyCount(TCopyCount&&) {
            }

            TCopyCount& operator=(const TCopyCount&) {
                ++copied;
                return *this;
            }

            TCopyCount& operator=(TCopyCount&&) {
                return *this;
            }
        };

        TLockFreeStack<TCopyCount> stack;
        stack.Enqueue(TCopyCount(1));
        TCopyCount val(0);
        stack.Dequeue(&val);
        UNIT_ASSERT_VALUES_EQUAL(0, copied);
    }

    Y_UNIT_TEST(MoveOnlyTest) {
        TLockFreeStack<THolder<bool>> stack;
        stack.Enqueue(MakeHolder<bool>(true));
        THolder<bool> val;
        stack.Dequeue(&val);
        UNIT_ASSERT(val);
        UNIT_ASSERT_VALUES_EQUAL(true, *val);
    }

    template <class TTest>
    struct TMultiThreadTester {
        using ThisType = TMultiThreadTester<TTest>;

        size_t Threads;
        size_t OperationsPerThread;

        TCountDownLatch StartLatch;
        TLockFreeStack<typename TTest::ValueType> Stack;

        TMultiThreadTester()
            : Threads(10)
            , OperationsPerThread(100000)
            , StartLatch(Threads)
        {
        }

        void Worker() {
            StartLatch.CountDown();
            StartLatch.Await();

            TVector<typename TTest::ValueType> unused;
            for (size_t i = 0; i < OperationsPerThread; ++i) {
                switch (GetCycleCount() % 4) {
                    case 0: {
                        TTest::Enqueue(Stack, i);
                        break;
                    }
                    case 1: {
                        TTest::Dequeue(Stack);
                        break;
                    }
                    case 2: {
                        TTest::EnqueueAll(Stack);
                        break;
                    }
                    case 3: {
                        TTest::DequeueAll(Stack);
                        break;
                    }
                }
            }
        }

        void Run() {
            TDeque<NThreading::TLegacyFuture<>> futures;

            for (size_t i = 0; i < Threads; ++i) {
                futures.emplace_back(std::bind(&ThisType::Worker, this));
            }
            futures.clear();
            TTest::DequeueAll(Stack);
        }
    };

    struct TFreeListTest {
        using ValueType = int;

        static void Enqueue(TLockFreeStack<int>& stack, size_t i) {
            stack.Enqueue(static_cast<int>(i));
        }

        static void Dequeue(TLockFreeStack<int>& stack) {
            int value;
            stack.Dequeue(&value);
        }

        static void EnqueueAll(TLockFreeStack<int>& stack) {
            TVector<int> values(5);
            stack.EnqueueAll(values);
        }

        static void DequeueAll(TLockFreeStack<int>& stack) {
            TVector<int> value;
            stack.DequeueAll(&value);
        }
    };

    // Test for catching thread sanitizer problems
    Y_UNIT_TEST(TestFreeList) {
        TMultiThreadTester<TFreeListTest>().Run();
    }

    struct TMoveTest {
        using ValueType = THolder<int>;

        static void Enqueue(TLockFreeStack<ValueType>& stack, size_t i) {
            stack.Enqueue(MakeHolder<int>(static_cast<int>(i)));
        }

        static void Dequeue(TLockFreeStack<ValueType>& stack) {
            ValueType value;
            if (stack.Dequeue(&value)) {
                UNIT_ASSERT(value);
            }
        }

        static void EnqueueAll(TLockFreeStack<ValueType>& stack) {
            // there is no enqueAll with moving signature in LockFreeStack
            Enqueue(stack, 0);
        }

        static void DequeueAll(TLockFreeStack<ValueType>& stack) {
            TVector<ValueType> values;
            stack.DequeueAll(&values);
            for (auto& v : values) {
                UNIT_ASSERT(v);
            }
        }
    };

    // Test for catching thread sanitizer problems
    Y_UNIT_TEST(TesMultiThreadMove) {
        TMultiThreadTester<TMoveTest>().Run();
    }
} // Y_UNIT_TEST_SUITE(TLockFreeStackTests)
