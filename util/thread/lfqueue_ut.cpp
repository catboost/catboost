#include <library/cpp/threading/future/future.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/ptr.h>
#include <util/thread/pool.h>

#include "lfqueue.h"

class TMoveTest {
public:
    TMoveTest(int marker = 0, int value = 0)
        : Marker_(marker)
        , Value_(value)
    {
    }

    TMoveTest(const TMoveTest& other) {
        *this = other;
    }

    TMoveTest(TMoveTest&& other) {
        *this = std::move(other);
    }

    TMoveTest& operator=(const TMoveTest& other) {
        Value_ = other.Value_;
        Marker_ = other.Marker_ + 1024;
        return *this;
    }

    TMoveTest& operator=(TMoveTest&& other) {
        Value_ = other.Value_;
        Marker_ = other.Marker_;
        other.Marker_ = 0;
        return *this;
    }

    int Marker() const {
        return Marker_;
    }

    int Value() const {
        return Value_;
    }

private:
    int Marker_ = 0;
    int Value_ = 0;
};

class TOperationsChecker {
public:
    TOperationsChecker() {
        ++DefaultCtor_;
    }

    TOperationsChecker(TOperationsChecker&&) {
        ++MoveCtor_;
    }

    TOperationsChecker(const TOperationsChecker&) {
        ++CopyCtor_;
    }

    TOperationsChecker& operator=(TOperationsChecker&&) {
        ++MoveAssign_;
        return *this;
    }

    TOperationsChecker& operator=(const TOperationsChecker&) {
        ++CopyAssign_;
        return *this;
    }

    static void Check(int defaultCtor, int moveCtor, int copyCtor, int moveAssign, int copyAssign) {
        UNIT_ASSERT_VALUES_EQUAL(defaultCtor, DefaultCtor_);
        UNIT_ASSERT_VALUES_EQUAL(moveCtor, MoveCtor_);
        UNIT_ASSERT_VALUES_EQUAL(copyCtor, CopyCtor_);
        UNIT_ASSERT_VALUES_EQUAL(moveAssign, MoveAssign_);
        UNIT_ASSERT_VALUES_EQUAL(copyAssign, CopyAssign_);
        Clear();
    }

private:
    static void Clear() {
        DefaultCtor_ = MoveCtor_ = CopyCtor_ = MoveAssign_ = CopyAssign_ = 0;
    }

    static int DefaultCtor_;
    static int MoveCtor_;
    static int CopyCtor_;
    static int MoveAssign_;
    static int CopyAssign_;
};

int TOperationsChecker::DefaultCtor_ = 0;
int TOperationsChecker::MoveCtor_ = 0;
int TOperationsChecker::CopyCtor_ = 0;
int TOperationsChecker::MoveAssign_ = 0;
int TOperationsChecker::CopyAssign_ = 0;

Y_UNIT_TEST_SUITE(TLockFreeQueueTests) {
    Y_UNIT_TEST(TestMoveEnqueue) {
        TMoveTest value(0xFF, 0xAA);
        TMoveTest tmp;

        TLockFreeQueue<TMoveTest> queue;

        queue.Enqueue(value);
        UNIT_ASSERT_VALUES_EQUAL(value.Marker(), 0xFF);
        UNIT_ASSERT(queue.Dequeue(&tmp));
        UNIT_ASSERT_VALUES_UNEQUAL(tmp.Marker(), 0xFF);
        UNIT_ASSERT_VALUES_EQUAL(tmp.Value(), 0xAA);

        queue.Enqueue(std::move(value));
        UNIT_ASSERT_VALUES_EQUAL(value.Marker(), 0);
        UNIT_ASSERT(queue.Dequeue(&tmp));
        UNIT_ASSERT_VALUES_EQUAL(tmp.Value(), 0xAA);
    }

    Y_UNIT_TEST(TestSimpleEnqueueDequeue) {
        TLockFreeQueue<int> queue;

        int i = -1;

        UNIT_ASSERT(!queue.Dequeue(&i));
        UNIT_ASSERT_VALUES_EQUAL(i, -1);

        queue.Enqueue(10);
        queue.Enqueue(11);
        queue.Enqueue(12);

        UNIT_ASSERT(queue.Dequeue(&i));
        UNIT_ASSERT_VALUES_EQUAL(10, i);
        UNIT_ASSERT(queue.Dequeue(&i));
        UNIT_ASSERT_VALUES_EQUAL(11, i);

        queue.Enqueue(13);

        UNIT_ASSERT(queue.Dequeue(&i));
        UNIT_ASSERT_VALUES_EQUAL(12, i);
        UNIT_ASSERT(queue.Dequeue(&i));
        UNIT_ASSERT_VALUES_EQUAL(13, i);

        UNIT_ASSERT(!queue.Dequeue(&i));

        const int tmp = 100;
        queue.Enqueue(tmp);
        UNIT_ASSERT(queue.Dequeue(&i));
        UNIT_ASSERT_VALUES_EQUAL(i, tmp);
    }

    Y_UNIT_TEST(TestSimpleEnqueueAllDequeue) {
        TLockFreeQueue<int> queue;

        int i = -1;

        UNIT_ASSERT(!queue.Dequeue(&i));
        UNIT_ASSERT_VALUES_EQUAL(i, -1);

        TVector<int> v;
        v.push_back(20);
        v.push_back(21);

        queue.EnqueueAll(v);

        v.clear();
        v.push_back(22);
        v.push_back(23);
        v.push_back(24);

        queue.EnqueueAll(v);

        v.clear();
        queue.EnqueueAll(v);

        v.clear();
        v.push_back(25);

        queue.EnqueueAll(v);

        for (int j = 20; j <= 25; ++j) {
            UNIT_ASSERT(queue.Dequeue(&i));
            UNIT_ASSERT_VALUES_EQUAL(j, i);
        }

        UNIT_ASSERT(!queue.Dequeue(&i));
    }

    void DequeueAllRunner(TLockFreeQueue<int>& queue, bool singleConsumer) {
        size_t threadsNum = 4;
        size_t enqueuesPerThread = 10'000;
        TThreadPool p;
        p.Start(threadsNum, 0);

        TVector<NThreading::TFuture<void>> futures;

        for (size_t i = 0; i < threadsNum; ++i) {
            NThreading::TPromise<void> promise = NThreading::NewPromise();
            futures.emplace_back(promise.GetFuture());

            p.SafeAddFunc([enqueuesPerThread, &queue, promise]() mutable {
                for (size_t i = 0; i != enqueuesPerThread; ++i) {
                    queue.Enqueue(i);
                }

                promise.SetValue();
            });
        }

        std::atomic<size_t> elementsLeft = threadsNum * enqueuesPerThread;

        ui64 numOfConsumers = singleConsumer ? 1 : threadsNum;

        TVector<TVector<int>> dataBuckets(numOfConsumers);

        for (size_t i = 0; i < numOfConsumers; ++i) {
            NThreading::TPromise<void> promise = NThreading::NewPromise();
            futures.emplace_back(promise.GetFuture());

            p.SafeAddFunc([&queue, &elementsLeft, promise, consumerData{&dataBuckets[i]}]() mutable {
                TVector<int> vec;
                while (static_cast<i64>(elementsLeft.load()) > 0) {
                    for (size_t i = 0; i != 100; ++i) {
                        vec.clear();
                        queue.DequeueAll(&vec);

                        elementsLeft -= vec.size();
                        consumerData->insert(consumerData->end(), vec.begin(), vec.end());
                    }
                }

                promise.SetValue();
            });
        }

        NThreading::WaitExceptionOrAll(futures).GetValueSync();
        p.Stop();

        TVector<int> left;
        queue.DequeueAll(&left);

        UNIT_ASSERT(left.empty());

        TVector<int> data;
        for (auto& dataBucket : dataBuckets) {
            data.insert(data.end(), dataBucket.begin(), dataBucket.end());
        }

        UNIT_ASSERT_EQUAL(data.size(), threadsNum * enqueuesPerThread);

        size_t threadIdx = 0;
        size_t cntValue = 0;

        Sort(data.begin(), data.end());
        for (size_t i = 0; i != data.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(cntValue, data[i]);
            ++threadIdx;

            if (threadIdx == threadsNum) {
                ++cntValue;
                threadIdx = 0;
            }
        }
    }

    Y_UNIT_TEST(TestDequeueAllSingleConsumer) {
        TLockFreeQueue<int> queue;
        DequeueAllRunner(queue, true);
    }

    Y_UNIT_TEST(TestDequeueAllMultipleConsumers) {
        TLockFreeQueue<int> queue;
        DequeueAllRunner(queue, false);
    }

    Y_UNIT_TEST(TestDequeueAllEmptyQueue) {
        TLockFreeQueue<int> queue;
        TVector<int> vec;

        queue.DequeueAll(&vec);

        UNIT_ASSERT(vec.empty());
    }

    Y_UNIT_TEST(TestDequeueAllQueueOrder) {
        TLockFreeQueue<int> queue;
        queue.Enqueue(1);
        queue.Enqueue(2);
        queue.Enqueue(3);

        TVector<int> v;
        queue.DequeueAll(&v);

        UNIT_ASSERT_VALUES_EQUAL(v.size(), 3);
        UNIT_ASSERT_VALUES_EQUAL(v[0], 1);
        UNIT_ASSERT_VALUES_EQUAL(v[1], 2);
        UNIT_ASSERT_VALUES_EQUAL(v[2], 3);
    }

    Y_UNIT_TEST(CleanInDestructor) {
        TSimpleSharedPtr<bool> p(new bool);
        UNIT_ASSERT_VALUES_EQUAL(1u, p.RefCount());

        {
            TLockFreeQueue<TSimpleSharedPtr<bool>> stack;

            stack.Enqueue(p);
            stack.Enqueue(p);

            UNIT_ASSERT_VALUES_EQUAL(3u, p.RefCount());
        }

        UNIT_ASSERT_VALUES_EQUAL(1, p.RefCount());
    }

    Y_UNIT_TEST(CheckOperationsCount) {
        TOperationsChecker o;
        o.Check(1, 0, 0, 0, 0);
        TLockFreeQueue<TOperationsChecker> queue;
        o.Check(0, 0, 0, 0, 0);
        queue.Enqueue(std::move(o));
        o.Check(0, 1, 0, 0, 0);
        queue.Enqueue(o);
        o.Check(0, 0, 1, 0, 0);
        queue.Dequeue(&o);
        o.Check(0, 0, 2, 1, 0);
    }
} // Y_UNIT_TEST_SUITE(TLockFreeQueueTests)
