#include "queue.h"
#include "deque.h"
#include "list.h"
#include "ptr.h"

#include <library/cpp/testing/unittest/registar.h>

#include <utility>

Y_UNIT_TEST_SUITE(TYQueueTest) {
    Y_UNIT_TEST(ConstructorsAndAssignments) {
        {
            using container = TQueue<int>;

            container c1;
            UNIT_ASSERT(!c1);
            c1.push(100);
            c1.push(200);
            UNIT_ASSERT(c1);

            container c2(c1);

            UNIT_ASSERT_VALUES_EQUAL(2, c1.size());
            UNIT_ASSERT_VALUES_EQUAL(2, c2.size());

            container c3(std::move(c1));

            UNIT_ASSERT_VALUES_EQUAL(0, c1.size());
            UNIT_ASSERT_VALUES_EQUAL(2, c3.size());

            c2.push(300);
            c3 = c2;

            UNIT_ASSERT_VALUES_EQUAL(3, c2.size());
            UNIT_ASSERT_VALUES_EQUAL(3, c3.size());

            c2.push(400);
            c3 = std::move(c2);

            UNIT_ASSERT_VALUES_EQUAL(0, c2.size());
            UNIT_ASSERT_VALUES_EQUAL(4, c3.size());
        }

        {
            using container = TPriorityQueue<int>;

            container c1;
            UNIT_ASSERT(!c1);
            c1.push(100);
            c1.push(200);
            UNIT_ASSERT(c1);

            container c2(c1);

            UNIT_ASSERT_VALUES_EQUAL(2, c1.size());
            UNIT_ASSERT_VALUES_EQUAL(2, c2.size());

            container c3(std::move(c1));

            UNIT_ASSERT_VALUES_EQUAL(0, c1.size());
            UNIT_ASSERT_VALUES_EQUAL(2, c3.size());

            c2.push(300);
            c3 = c2;

            UNIT_ASSERT_VALUES_EQUAL(3, c2.size());
            UNIT_ASSERT_VALUES_EQUAL(3, c3.size());

            c2.push(400);
            c3 = std::move(c2);

            UNIT_ASSERT_VALUES_EQUAL(0, c2.size());
            UNIT_ASSERT_VALUES_EQUAL(4, c3.size());
        }
    }

    Y_UNIT_TEST(pqueue1) {
        TPriorityQueue<int, TDeque<int>, TLess<int>> q;

        q.push(42);
        q.push(101);
        q.push(69);
        UNIT_ASSERT(q.top() == 101);

        q.pop();
        UNIT_ASSERT(q.top() == 69);

        q.pop();
        UNIT_ASSERT(q.top() == 42);

        q.pop();
        UNIT_ASSERT(q.empty());
    }

    Y_UNIT_TEST(pqueue2) {
        using TPQueue = TPriorityQueue<int, TDeque<int>, TLess<int>>;
        TPQueue q;

        {
            TPQueue qq;

            qq.push(42);
            qq.push(101);
            qq.push(69);

            qq.swap(q);
        }

        UNIT_ASSERT(q.top() == 101);

        q.pop();
        UNIT_ASSERT(q.top() == 69);

        q.pop();
        UNIT_ASSERT(q.top() == 42);

        q.pop();
        UNIT_ASSERT(q.empty());
    }

    Y_UNIT_TEST(pqueue3) {
        TPriorityQueue<int, TDeque<int>, TLess<int>> q;

        q.push(42);
        q.push(101);
        q.push(69);
        q.clear();

        UNIT_ASSERT(q.empty());
    }

    Y_UNIT_TEST(pqueue4) {
        TDeque<int> c;
        c.push_back(42);
        c.push_back(101);
        c.push_back(69);

        TPriorityQueue<int, TDeque<int>, TLess<int>> q(TLess<int>(), std::move(c));

        UNIT_ASSERT(c.empty());

        UNIT_ASSERT_EQUAL(q.size(), 3);

        UNIT_ASSERT(q.top() == 101);

        q.pop();
        UNIT_ASSERT(q.top() == 69);

        q.pop();
        UNIT_ASSERT(q.top() == 42);

        q.pop();
        UNIT_ASSERT(q.empty());
    }

    struct THolderWithPriority {
        THolderWithPriority(const TString& value, int priority)
            : Value(MakeHolder<TString>(value))
            , Priority(priority)
        {
        }

        THolder<TString> Value; // THolder to test move-ctor
        int Priority;

        std::weak_ordering operator<=>(const THolderWithPriority& other) const noexcept {
            return Priority <=> other.Priority;
        }
    };

    Y_UNIT_TEST(pqueue5) {
        // Test move-and-pop
        TPriorityQueue<THolderWithPriority> q;

        UNIT_ASSERT(q.empty());
        q.emplace("min", 1);
        q.emplace("max", 3);
        q.emplace("middle", 2);

        auto value = q.PopValue().Value;
        UNIT_ASSERT(*value == "max");

        value = q.PopValue().Value;
        UNIT_ASSERT(*value == "middle");

        value = q.PopValue().Value;
        UNIT_ASSERT(*value == "min");
        UNIT_ASSERT(q.empty());
    }

    Y_UNIT_TEST(queue1) {
        TQueue<int, TList<int>> q;

        q.push(42);
        q.push(101);
        q.push(69);
        UNIT_ASSERT(q.front() == 42);

        q.pop();
        UNIT_ASSERT(q.front() == 101);

        q.pop();
        UNIT_ASSERT(q.front() == 69);

        q.pop();
        UNIT_ASSERT(q.empty());
    }

    Y_UNIT_TEST(queue2) {
        using TQueueType = TQueue<int>;
        TQueueType q;

        {
            TQueueType qq;

            qq.push(42);
            qq.push(101);
            qq.push(69);

            qq.swap(q);
        }

        UNIT_ASSERT(q.front() == 42);

        q.pop();
        UNIT_ASSERT(q.front() == 101);

        q.pop();
        UNIT_ASSERT(q.front() == 69);

        q.pop();
        UNIT_ASSERT(q.empty());
    }

    Y_UNIT_TEST(queue3) {
        using TQueueType = TQueue<int>;
        TQueueType q;

        q.push(42);
        q.push(101);
        q.push(69);
        q.clear();

        UNIT_ASSERT(q.empty());
    }
} // Y_UNIT_TEST_SUITE(TYQueueTest)
