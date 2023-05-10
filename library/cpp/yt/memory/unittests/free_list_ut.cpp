#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/free_list.h>

#include <util/random/random.h>

#include <thread>
#include <stack>
#include <latch>

namespace NYT {
namespace {

using namespace std::chrono_literals;

////////////////////////////////////////////////////////////////////////////////

TEST(TFreeListTest, CompareAndSet)
{
    TAtomicUint128 v = 0;
    ui64 p1 = 0;
    ui64 p2 = 0;
    EXPECT_TRUE(CompareAndSet(&v, p1, p2, ui64{13}, ui64{9}));
    EXPECT_FALSE(CompareAndSet(&v, p1, p2, ui64{100}, ui64{500}));
    EXPECT_EQ(13u, p1);
    EXPECT_EQ(9u, p2);
    EXPECT_TRUE(CompareAndSet(&v, p1, p2, ui64{100}, ui64{500}));
    EXPECT_EQ(TAtomicUint128{500} << 64 | 100, v);
}

////////////////////////////////////////////////////////////////////////////////

struct TTestConfig
{
    ui64 Threads;
    ui64 MaxBatchSize;
    std::chrono::seconds TimeLimit;
};

class TFreeListStressTest
    : public testing::TestWithParam<TTestConfig>
{ };

struct TTestItem
    : public NYT::TFreeListItemBase<TTestItem>
{
    TTestItem() = default;

    ui64 Value = 0;
    ui64 IndexInSet = 0;
    // Avoid false sharing.
    char Padding[CacheLineSize - 2 * sizeof(ui64)];
};

class TTestItemSet
{
public:
    static void Reset()
    {
        Items_.clear();
    }

    static TTestItemSet Allocate(size_t setSize)
    {
        TTestItemSet set;
        for (size_t i = 0; i < setSize; ++i) {
            Items_.push_back(std::make_unique<TTestItem>());
            Items_.back()->IndexInSet = Items_.size() - 1;
            set.Acquire(Items_.back().get());
        }
        return set;
    }

    void Acquire(TTestItem* item)
    {
        AcquiredItemIndices_.push(item->IndexInSet);
    }

    TTestItem* Release()
    {
        YT_VERIFY(!AcquiredItemIndices_.empty());
        size_t index = AcquiredItemIndices_.top();
        AcquiredItemIndices_.pop();
        return Items_[index].get();
    }

private:
    inline static std::vector<std::unique_ptr<TTestItem>> Items_;

    std::stack<size_t> AcquiredItemIndices_;
};


TEST_P(TFreeListStressTest, Stress)
{
    TTestItemSet::Reset();
    SetRandomSeed(0x424242);
    auto params = GetParam();

    TFreeList<TTestItem> list;

    std::latch start(params.Threads);

    std::atomic<bool> running{true};
    std::atomic<ui64> put{0};
    std::atomic<ui64> extracted{0};

    std::vector<std::thread> workers;
    for (ui64 i = 0; i < params.Threads; ++i) {
        auto itemSet = TTestItemSet::Allocate(params.MaxBatchSize);
        workers.emplace_back([&, params, itemSet = std::move(itemSet)]() mutable {
            start.arrive_and_wait();

            while (running.load(std::memory_order::relaxed)) {
                // Push batch of items.
                ui64 batchSize = 1 + RandomNumber<ui64>(params.MaxBatchSize);
                for (ui64 i = 0; i < batchSize; ++i) {
                    auto* item = itemSet.Release();
                    item->Value = 1 + RandomNumber<ui64>(1e9);
                    put.fetch_add(item->Value, std::memory_order::relaxed);
                    list.Put(item);
                }

                // Pop batch of items.
                for (ui64 i = 0; i < batchSize; ++i) {
                    auto* item = list.Extract();
                    ASSERT_NE(item, nullptr);
                    extracted.fetch_add(item->Value, std::memory_order::relaxed);
                    itemSet.Acquire(item);
                }
            }
        });
    }

    Sleep(params.TimeLimit);
    running.store(false);

    for (auto& worker : workers) {
        worker.join();
    }

    Cerr << "Put: " << put.load() << Endl;
    Cerr << "Extracted: " << extracted.load() << Endl;
    EXPECT_EQ(put.load(), extracted.load());
}

INSTANTIATE_TEST_SUITE_P(
    TFreeListTest,
    TFreeListStressTest,
    testing::Values(
        TTestConfig{4, 1, 15s},
        TTestConfig{4, 3, 15s},
        TTestConfig{4, 5, 15s}));

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
