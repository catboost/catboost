#include <library/cpp/yt/small_containers/compact_heap.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <random>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(CompactHeapTest, Simple)
{
    TCompactHeap<int, 2> heap;
    heap.push(3);
    heap.push(1);
    heap.push(2);

    EXPECT_EQ(3u, heap.size());
    EXPECT_EQ(1, heap.extract_min());

    EXPECT_EQ(2, heap.get_min());
    EXPECT_EQ(2, heap.extract_min());

    EXPECT_EQ(3, heap.extract_min());

    EXPECT_TRUE(heap.empty());
}

TEST(CompactHeapTest, SimpleComparator)
{
    TCompactHeap<int, 2, std::greater<int>> heap;
    heap.push(3);
    heap.push(1);
    heap.push(2);

    EXPECT_EQ(3u, heap.size());
    EXPECT_EQ(3, heap.extract_min());
    EXPECT_EQ(2, heap.get_min());
    EXPECT_EQ(2, heap.extract_min());
    EXPECT_EQ(1, heap.extract_min());
    EXPECT_TRUE(heap.empty());
}

TEST(CompactHeapTest, Capacity)
{
    TCompactHeap<int, 2> heap;
    EXPECT_EQ(2u, heap.capacity());
    EXPECT_EQ(0u, heap.size());

    for (int i = 0; i < 100; ++i) {
        heap.push(i);
    }
    EXPECT_LE(100u, heap.capacity());
    EXPECT_EQ(100u, heap.size());

    for (int i = 0; i < 99; ++i) {
        heap.pop();
    }
    EXPECT_LE(100u, heap.capacity());
    EXPECT_EQ(1u, heap.size());

    heap.shrink_to_small();

    EXPECT_EQ(2u, heap.capacity());
    EXPECT_EQ(1u, heap.size());
}

TEST(CompactHeapTest, Stress)
{
    TCompactHeap<int, 3, std::greater<int>> heap;
    std::vector<int> values;

    std::mt19937 rng(42);
    for (int iteration = 0; iteration < 1000; ++iteration) {
        EXPECT_EQ(values.size(), heap.size());
        EXPECT_EQ(values.empty(), heap.empty());

        {
            std::vector<int> content(heap.begin(), heap.end());
            std::sort(content.rbegin(), content.rend());
            EXPECT_EQ(values, content);
        }

        if (!values.empty()) {
            EXPECT_EQ(values[0], heap.get_min());
        }

        if (values.empty() || rng() % 2 == 0) {
            int x = rng() % 100;
            heap.push(x);
            values.push_back(x);
            std::sort(values.rbegin(), values.rend());
        } else {
            if (rng() % 2 == 0) {
                EXPECT_EQ(values[0], heap.extract_min());
            } else {
                heap.pop();
            }
            values.erase(values.begin());
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
