#include "queue.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/set.h>
#include <util/generic/string.h>
#include <util/string/cast.h>

namespace NThreading {
    ////////////////////////////////////////////////////////////////////////////////

    Y_UNIT_TEST_SUITE(TOneOneQueueTest){
        Y_UNIT_TEST(ShouldBeEmptyAtStart){
            TOneOneQueue<int> queue;

    int result = 0;
    UNIT_ASSERT(queue.IsEmpty());
    UNIT_ASSERT(!queue.Dequeue(result));
}

Y_UNIT_TEST(ShouldReturnEntries) {
    TOneOneQueue<int> queue;
    queue.Enqueue(1);
    queue.Enqueue(2);
    queue.Enqueue(3);

    int result = 0;
    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT_EQUAL(result, 1);

    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT_EQUAL(result, 2);

    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT_EQUAL(result, 3);

    UNIT_ASSERT(queue.IsEmpty());
    UNIT_ASSERT(!queue.Dequeue(result));
}

Y_UNIT_TEST(ShouldStoreMultipleChunks) {
    TOneOneQueue<int, 100> queue;
    for (int i = 0; i < 1000; ++i) {
        queue.Enqueue(i);
    }

    for (int i = 0; i < 1000; ++i) {
        int result = 0;
        UNIT_ASSERT(!queue.IsEmpty());
        UNIT_ASSERT(queue.Dequeue(result));
        UNIT_ASSERT_EQUAL(result, i);
    }
}

struct alignas(64) TOverAligned {
    size_t Value = 0;

    TOverAligned() = default;

    explicit TOverAligned(size_t value)
        : Value(value)
    {
        UNIT_ASSERT(reinterpret_cast<uintptr_t>(this) % alignof(TOverAligned) == 0);
    }
};

// ChunkSize = 128, alignof = 64: EntriesOffset = 64, sizeof = 64, so
// MaxCount = 1 — every Enqueue exercises the chunk hand-off.
Y_UNIT_TEST(ShouldKeepOverAlignedEntriesAligned) {
    TOneOneQueue<TOverAligned, 128> queue;

    for (size_t i = 0; i < 10; ++i) {
        queue.Enqueue(TOverAligned{i});
    }

    for (size_t i = 0; i < 10; ++i) {
        TOverAligned result;
        UNIT_ASSERT(queue.Dequeue(result));
        UNIT_ASSERT_EQUAL(result.Value, i);
    }

    UNIT_ASSERT(queue.IsEmpty());
}

Y_UNIT_TEST(ShouldDestroyNonTrivialEntriesOnDestruction) {
    // Small chunks force the queue to span several chunks, and half of the
    // entries are still alive when the queue is destroyed: ~TOneOneQueue()
    // must destroy the leftovers in every chunk exactly once.
    TOneOneQueue<TString, 128> queue;

    for (int i = 0; i < 100; ++i) {
        queue.Enqueue(ToString(i));
    }

    for (int i = 0; i < 50; ++i) {
        TString result;
        UNIT_ASSERT(queue.Dequeue(result));
        UNIT_ASSERT_EQUAL(result, ToString(i));
    }

    UNIT_ASSERT(!queue.IsEmpty());
}
}

////////////////////////////////////////////////////////////////////////////////

Y_UNIT_TEST_SUITE(TManyOneQueueTest){
    Y_UNIT_TEST(ShouldBeEmptyAtStart){
        TManyOneQueue<int> queue;

int result;
UNIT_ASSERT(queue.IsEmpty());
UNIT_ASSERT(!queue.Dequeue(result));
}

Y_UNIT_TEST(ShouldReturnEntries) {
    TManyOneQueue<int> queue;
    queue.Enqueue(1);
    queue.Enqueue(2);
    queue.Enqueue(3);

    int result = 0;
    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT_EQUAL(result, 1);

    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT_EQUAL(result, 2);

    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT_EQUAL(result, 3);

    UNIT_ASSERT(queue.IsEmpty());
    UNIT_ASSERT(!queue.Dequeue(result));
}

Y_UNIT_TEST(ShouldHandleNonTrivialEntries) {
    TManyOneQueue<TString> queue;

    for (int i = 0; i < 100; ++i) {
        queue.Enqueue(ToString(i));
    }

    for (int i = 0; i < 100; ++i) {
        TString result;
        UNIT_ASSERT(queue.Dequeue(result));
        UNIT_ASSERT_EQUAL(result, ToString(i));
    }

    UNIT_ASSERT(queue.IsEmpty());
}
}

////////////////////////////////////////////////////////////////////////////////

Y_UNIT_TEST_SUITE(TManyManyQueueTest){
    Y_UNIT_TEST(ShouldBeEmptyAtStart){
        TManyManyQueue<int> queue;

int result = 0;
UNIT_ASSERT(queue.IsEmpty());
UNIT_ASSERT(!queue.Dequeue(result));
}

Y_UNIT_TEST(ShouldReturnEntries) {
    TManyManyQueue<int> queue;
    queue.Enqueue(1);
    queue.Enqueue(2);
    queue.Enqueue(3);

    int result = 0;
    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT_EQUAL(result, 1);

    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT_EQUAL(result, 2);

    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT_EQUAL(result, 3);

    UNIT_ASSERT(queue.IsEmpty());
    UNIT_ASSERT(!queue.Dequeue(result));
}
}

////////////////////////////////////////////////////////////////////////////////

Y_UNIT_TEST_SUITE(TRelaxedManyOneQueueTest){
    Y_UNIT_TEST(ShouldBeEmptyAtStart){
        TRelaxedManyOneQueue<int> queue;

int result;
UNIT_ASSERT(queue.IsEmpty());
UNIT_ASSERT(!queue.Dequeue(result));
}

Y_UNIT_TEST(ShouldReturnEntries) {
    TSet<int> items = {1, 2, 3};

    TRelaxedManyOneQueue<int> queue;
    for (int item : items) {
        queue.Enqueue(item);
    }

    int result = 0;
    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT(items.erase(result));

    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT(items.erase(result));

    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT(items.erase(result));

    UNIT_ASSERT(queue.IsEmpty());
    UNIT_ASSERT(!queue.Dequeue(result));
}
}

////////////////////////////////////////////////////////////////////////////////

Y_UNIT_TEST_SUITE(TRelaxedManyManyQueueTest){
    Y_UNIT_TEST(ShouldBeEmptyAtStart){
        TRelaxedManyManyQueue<int> queue;

int result = 0;
UNIT_ASSERT(queue.IsEmpty());
UNIT_ASSERT(!queue.Dequeue(result));
}

Y_UNIT_TEST(ShouldReturnEntries) {
    TSet<int> items = {1, 2, 3};

    TRelaxedManyManyQueue<int> queue;
    for (int item : items) {
        queue.Enqueue(item);
    }

    int result = 0;
    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT(items.erase(result));

    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT(items.erase(result));

    UNIT_ASSERT(!queue.IsEmpty());
    UNIT_ASSERT(queue.Dequeue(result));
    UNIT_ASSERT(items.erase(result));

    UNIT_ASSERT(queue.IsEmpty());
    UNIT_ASSERT(!queue.Dequeue(result));
}
}
}
