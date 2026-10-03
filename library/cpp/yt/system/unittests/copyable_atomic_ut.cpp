#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/copyable_atomic.h>

#include <vector>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TCopyableAtomicTest, LoadStore)
{
    TCopyableAtomic<int> atomic;
    EXPECT_EQ(0, atomic.load());

    atomic.store(42);
    EXPECT_EQ(42, atomic.load());

    atomic = 7;
    EXPECT_EQ(7, atomic.load());
}

TEST(TCopyableAtomicTest, ValueConstructor)
{
    TCopyableAtomic<int> atomic(42);
    EXPECT_EQ(42, atomic.load());
}

TEST(TCopyableAtomicTest, AtomicApi)
{
    TCopyableAtomic<int> atomic(42);
    EXPECT_EQ(42, atomic.exchange(1));
    EXPECT_EQ(1, atomic.fetch_add(1));

    int expected = 2;
    EXPECT_TRUE(atomic.compare_exchange_strong(expected, 100));
    EXPECT_EQ(100, atomic.load(std::memory_order::relaxed));
}

TEST(TCopyableAtomicTest, Copy)
{
    TCopyableAtomic<int> source(42);

    TCopyableAtomic<int> constructed(source);
    EXPECT_EQ(42, constructed.load());
    EXPECT_EQ(42, source.load());

    TCopyableAtomic<int> assigned;
    assigned = source;
    EXPECT_EQ(42, assigned.load());
    EXPECT_EQ(42, source.load());

    source.store(7);
    EXPECT_EQ(42, constructed.load());
    EXPECT_EQ(42, assigned.load());
}

TEST(TCopyableAtomicTest, Move)
{
    TCopyableAtomic<int> source(42);

    TCopyableAtomic<int> constructed(std::move(source));
    EXPECT_EQ(42, constructed.load());

    TCopyableAtomic<int> assigned;
    assigned = std::move(constructed);
    EXPECT_EQ(42, assigned.load());
}

TEST(TCopyableAtomicTest, VectorReallocation)
{
    std::vector<TCopyableAtomic<int>> atomics;
    for (int index = 0; index < 100; ++index) {
        atomics.emplace_back(index);
    }

    for (int index = 0; index < 100; ++index) {
        EXPECT_EQ(index, atomics[index].load());
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
