#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/chunked_memory_pool.h>
#include <library/cpp/yt/memory/chunked_memory_pool_allocator.h>

#include <random>
#include <set>
#include <vector>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TChunkedMemoryPoolAllocatorTest, SimpleSet)
{
    TChunkedMemoryPool pool;
    TChunkedMemoryPoolAllocator<int> allocator(&pool);

    using TSet = std::set<int, std::less<int>, TChunkedMemoryPoolAllocator<int>>;
    TSet set(allocator);

    constexpr int seed = 0;
    std::mt19937 generator(seed);
    auto valueDistribution = std::uniform_int_distribution<int>(
        -10000,
        10000);

    constexpr int iterationCount = 10000;
    constexpr int nodeSize = sizeof(TSet::node_type);

    ASSERT_EQ(pool.GetSize(), 0ull);
    size_t previousPoolSize = 0;
    for (int iteration = 0; iteration < iterationCount; ++iteration) {
        bool inserted = set.insert(valueDistribution(generator)).second;

        ASSERT_GE(pool.GetSize(), set.size() * nodeSize);
        if (inserted) {
            ASSERT_GT(pool.GetSize(), previousPoolSize);
            previousPoolSize = pool.GetSize();
        } else {
            ASSERT_EQ(pool.GetSize(), previousPoolSize);
        }
    }
}

TEST(TChunkedMemoryPoolAllocatorTest, SimpleVector)
{
    TChunkedMemoryPool pool;
    TChunkedMemoryPoolAllocator<int> allocator(&pool);

    using TVector = std::vector<int, TChunkedMemoryPoolAllocator<int>>;
    TVector vector(allocator);

    constexpr int seed = 0;
    std::mt19937 generator(seed);
    auto valueDistribution = std::uniform_int_distribution<int>(
        -10000,
        10000);

    constexpr int iterationCount = 10000;

    ASSERT_EQ(pool.GetSize(), 0ull);
    for (int iteration = 0; iteration < iterationCount; ++iteration) {
        vector.push_back(valueDistribution(generator));

        ASSERT_GE(pool.GetSize(), vector.size());
    }
}

TEST(TChunkedMemoryPoolAllocatorTest, DestructorCalled)
{
    TChunkedMemoryPool pool;
    TChunkedMemoryPoolAllocator<int> allocator(&pool);

    class TSetInDestructor
    {
    public:
        explicit TSetInDestructor(bool* flag)
            : Flag_(flag)
        { }

        ~TSetInDestructor()
        {
            *Flag_ = true;
        }

        bool operator<(const TSetInDestructor& other) const
        {
            return Flag_ < other.Flag_;
        }

    private:
        bool* const Flag_;
    };

    using TSet = std::set<TSetInDestructor, std::less<TSetInDestructor>, TChunkedMemoryPoolAllocator<TSetInDestructor>>;

    bool destructorCalled = false;
    auto setInDestructor = TSetInDestructor{&destructorCalled};
    {
        auto set = std::make_unique<TSet>(allocator);
        set->insert(setInDestructor);
    }

    ASSERT_TRUE(destructorCalled);
}

TEST(TChunkedMemoryPoolAllocatorTest, Copy)
{
    TChunkedMemoryPool pool;
    TChunkedMemoryPoolAllocator<int> allocator(&pool);

    using TSet = std::set<int, std::less<int>, TChunkedMemoryPoolAllocator<int>>;
    TSet set(allocator);

    constexpr int iterationCount = 1000;
    for (int iteration = 0; iteration < iterationCount; ++iteration) {
        set.insert(iteration);
    }

    auto poolSize = pool.GetSize();

    auto copy = set;

    ASSERT_EQ(std::ssize(set), iterationCount);
    ASSERT_EQ(std::ssize(copy), iterationCount);
    ASSERT_EQ(pool.GetSize(), poolSize * 2);
}

TEST(TChunkedMemoryPoolAllocatorTest, Move)
{
    TChunkedMemoryPool pool;
    TChunkedMemoryPoolAllocator<void> allocator(&pool);

    using TSet = std::set<int, std::less<int>, TChunkedMemoryPoolAllocator<int>>;
    TSet set(allocator);

    constexpr int iterationCount = 1000;
    for (int iteration = 0; iteration < iterationCount; ++iteration) {
        set.insert(iteration);
    }

    auto size = pool.GetSize();

    auto movedTo = std::move(set);

    ASSERT_EQ(std::ssize(movedTo), iterationCount);
    ASSERT_EQ(pool.GetSize(), size);
}

TEST(TChunkedMemoryPoolAllocatorTest, NoLeak)
{
    TChunkedMemoryPool pool;
    TChunkedMemoryPoolAllocator<void> allocator(&pool);

    using TSet = std::set<int, std::less<int>, TChunkedMemoryPoolAllocator<int>>;
    auto* set = pool.AllocateUninitialized<TSet>(1);
    new(set) TSet(allocator);

    constexpr int iterationCount = 1000;
    for (int iteration = 0; iteration < iterationCount; ++iteration) {
        set->insert(iteration);
    }

    ASSERT_GE(pool.GetSize(), sizeof(int) * iterationCount);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
