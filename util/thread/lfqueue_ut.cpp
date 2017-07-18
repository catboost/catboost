#include <library/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/generic/ptr.h>

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

SIMPLE_UNIT_TEST_SUITE(TLockFreeQueueTests) {
    SIMPLE_UNIT_TEST(TestMoveEnqueue) {
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

    SIMPLE_UNIT_TEST(TestSimpleEnqueueDequeue) {
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

    SIMPLE_UNIT_TEST(TestSimpleEnqueueAllDequeue) {
        TLockFreeQueue<int> queue;

        int i = -1;

        UNIT_ASSERT(!queue.Dequeue(&i));
        UNIT_ASSERT_VALUES_EQUAL(i, -1);

        yvector<int> v;
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

    SIMPLE_UNIT_TEST(CleanInDestructor) {
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

    SIMPLE_UNIT_TEST(CheckOperationsCount) {
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
}
