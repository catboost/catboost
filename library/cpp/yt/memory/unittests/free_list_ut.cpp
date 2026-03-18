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

TEST(TFreeListDetailsTest, CasFreeListPackedPair)
{
    using P = NDetail::TFreeListPackedPair;
    using C = NDetail::TFreeListPackedPairComponent;
    P v = 0;
    C p1 = 0;
    C p2 = 0;
    EXPECT_TRUE(NDetail::CasFreeListPackedPair(&v, p1, p2, C(13), C(9)));
    EXPECT_FALSE(NDetail::CasFreeListPackedPair(&v, p1, p2, C(100), C(500)));
    EXPECT_EQ(C(13), p1);
    EXPECT_EQ(C(9), p2);
    EXPECT_TRUE(NDetail::CasFreeListPackedPair(&v, p1, p2, C(100), C(500)));
    EXPECT_EQ((P(500) << (sizeof(C) * 8)) | 100, v);
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

TEST_P(TFreeListStressTest, Do)
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
    TFreeListStressTest,
    TFreeListStressTest,
    testing::Values(
        TTestConfig{4, 1, 15s},
        TTestConfig{4, 3, 15s},
        TTestConfig{4, 5, 15s}));

////////////////////////////////////////////////////////////////////////////////

struct TFreeListItem
    : public TFreeListItemBase<TFreeListItem>
{
    explicit TFreeListItem(i64 value)
        : Value(value)
    { }
    i64 Value;
};

static std::chrono::steady_clock::time_point Now()
{
    return std::chrono::steady_clock::now();
}

void ProducerThread(TFreeList<TFreeListItem>* freeList, i64 startValue, std::chrono::steady_clock::time_point until)
{
    i64 v = startValue;
    while (Now() < until) {
        freeList->Put(new TFreeListItem(v));
        v += 2;
    }
}

void ConsumerThread(TFreeList<TFreeListItem>* freeList, std::chrono::steady_clock::time_point until)
{
    i64 latestEven = -2;
    i64 latestOdd = -1;
    std::vector<i64> buffer;
    while (Now() < until) {
        buffer.clear();
        auto* extractedList = freeList->ExtractAll();
        while (extractedList) {
            buffer.push_back(extractedList->Value);
            auto* next = extractedList->Next.load(std::memory_order::acquire);
            delete extractedList;
            extractedList = next;
        }

        for (auto it = buffer.rbegin(); it != buffer.rend(); ++it) {
            auto value = *it;
            if (value % 2 == 0) {
                ASSERT_GT(value, latestEven);
                latestEven = value;
            } else {
                ASSERT_GT(value, latestOdd);
                latestOdd = value;
            }
        }
    }
}

void Cleanup(TFreeList<TFreeListItem>* freeList)
{
    auto* node = freeList->ExtractAll();
    while (node) {
        auto* next = node->Next.load(std::memory_order::acquire);
        delete node;
        node = next;
    }
}

TEST(TFreeListTest, ProducerConsumer)
{
    auto now = Now();
    auto until = now + std::chrono::seconds(15);

    auto freeList = TFreeList<TFreeListItem>();

    std::vector<std::thread> threads;

    threads.emplace_back(ProducerThread, &freeList, 0, until);
    threads.emplace_back(ProducerThread, &freeList, 1, until);

    threads.emplace_back(ConsumerThread, &freeList, until);
    threads.emplace_back(ConsumerThread, &freeList, until);

    for (auto& thread : threads) {
        thread.join();
    }

    Cleanup(&freeList);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
