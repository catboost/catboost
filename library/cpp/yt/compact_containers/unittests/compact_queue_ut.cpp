#include <library/cpp/yt/compact_containers/compact_queue.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <queue>
#include <random>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TCompactQueueTest, Simple)
{
    TCompactQueue<int, 4> queue;
    queue.Push(1);
    queue.Push(2);
    queue.Push(3);

    for (int i = 1; i <= 10; i++) {
        EXPECT_EQ(queue.Size(), 3u);
        EXPECT_EQ(queue.Front(), i);
        EXPECT_EQ(queue.Pop(), i);
        queue.Push(i + 3);
    }

    for (int i = 11; i <= 13; i++) {
        EXPECT_EQ(queue.Front(), i);
        queue.Pop();
    }

    EXPECT_TRUE(queue.Empty());
}

TEST(TCompactQueueTest, Reallocation1)
{
    TCompactQueue<int, 2> queue;
    queue.Push(1);
    queue.Push(2);
    queue.Push(3);

    for (int i = 1; i <= 10; i++) {
        EXPECT_EQ(queue.Size(), 3u);
        EXPECT_EQ(queue.Front(), i);
        EXPECT_EQ(queue.Pop(), i);
        queue.Push(i + 3);
    }

    for (int i = 11; i <= 13; i++) {
        EXPECT_EQ(queue.Front(), i);
        queue.Pop();
    }

    EXPECT_TRUE(queue.Empty());
}

TEST(TCompactQueueTest, Reallocation2)
{
    TCompactQueue<int, 3> queue;
    queue.Push(1);
    queue.Push(2);
    queue.Push(3);
    EXPECT_EQ(queue.Pop(), 1);
    queue.Push(4);
    queue.Push(5);

    EXPECT_EQ(queue.Size(), 4u);

    for (int i = 2; i <= 10; i++) {
        EXPECT_EQ(queue.Size(), 4u);
        EXPECT_EQ(queue.Front(), i);
        EXPECT_EQ(queue.Pop(), i);
        queue.Push(i + 4);
    }

    for (int i = 11; i <= 14; i++) {
        EXPECT_EQ(queue.Front(), i);
        queue.Pop();
    }

    EXPECT_TRUE(queue.Empty());
}

TEST(TCompactQueueTest, Stress)
{
    std::mt19937_64 rng(42);

    for (int iteration = 0; iteration < 1000; ++iteration) {
        TCompactQueue<int, 4> queue1;
        std::queue<int> queue2;

        for (int step = 0; step < 10'000; ++step) {
            EXPECT_EQ(queue1.Size(), queue2.size());
            EXPECT_EQ(queue1.Empty(), queue2.empty());
            if (!queue1.Empty()) {
                EXPECT_EQ(queue1.Front(), queue2.front());
            }

            if (queue2.empty() || rng() % 2 == 0) {
                int value = rng() % 1'000'000;
                queue1.Push(value);
                queue2.push(value);
            } else {
                queue1.Pop();
                queue2.pop();
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
